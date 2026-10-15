// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/plain_generate.cpp - `strata generate` for the models of plain_model.hpp (Ornith-1.5, llm-jp-4.1).
//
//     strata --native MODEL.gguf --tokens 1,2,3 [options: plain_usage, --help]
//
// The prompt runs in chunks of up to N tokens through the batched products (--prompt-chunk, default 4096; 0: in
// windows of up to T tokens), then one token a window (--spec: the pending token and the MTP layer's drafts, the
// accepted ones emitted together).  stdout gets one line,
// `tokens: a,b,c`, the generated ids; --dump-logits writes the logits each generated token was chosen from, n_vocab
// floats a row.  The report goes to stderr.
#include "strata/program/plain_generate.hpp"
#include "strata/program/option_scope.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;

bool parse_ids(const std::string& text, std::vector<int32_t>& out) {
    out.clear();
    const char* p = text.c_str();
    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        if (*p == '\0') break;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > 0x7fffffff) return false;
        out.push_back((int32_t) v);
        p = end;
    }
    return !out.empty();
}

// The model's architecture, from its first shard's metadata only; empty when it cannot be read.
std::string architecture(const std::string& path) {
    try {
        const GgufModel m = GgufModel::open(path);
        const MetaValue* a = m.meta().get("general.architecture");
        return a != nullptr && a->type == MetaType::STRING ? a->s : std::string();
    } catch (const std::exception&) {
        return std::string();
    }
}

// The tokens that end a generation: the GGUF's end-of-sequence, end-of-turn and end-of-message tokens it names
std::vector<int32_t> eos_tokens(const std::string& path) {
    std::vector<int32_t> ids;
    try {
        const GgufModel m = GgufModel::open(path);
        for (const char* key :
             {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
            const MetaValue* v = m.meta().get(key);
            if (v != nullptr && v->is_num() &&
                std::find(ids.begin(), ids.end(), (int32_t) v->u) == ids.end())
                ids.push_back((int32_t) v->u);
        }
    } catch (const std::exception&) {
        return {};   // an unreadable file has no stop ids to add; loading it reports why
    }
    return ids;
}

}  // namespace

// `--serve` for these models: generate.cpp's --serve protocol (server.py's StrataEngine) for text requests.  READY
// <context> stop; then `GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N] <ids>` -> RESUME <reused>,
// one `T <id>` a token, DONE (generate.cpp's fields; the RAM, file and offload ones zero); STOP ends a request,
// QUIT the engine.  The commands of Qwen3.8-Flash-Next's engine this one has no part for (VRAM, SAVE, RESTORE,
// GENI, BGEN) answer ERR.  The conversation's tokens stay read between requests: a request that starts with them
// reads only what follows (a model with a recurrent state only when it extends them all; else it starts over).
static int plain_serve(strata::core::PlainModel& model, const strata::core::PlainOptions& opt, const std::string& arch,
                       const std::vector<int32_t>& eos, const std::string& profile_save, double save_min) {
    std::atomic<bool> stop_req{false};
    std::mutex in_mu;
    std::condition_variable in_cv;
    std::deque<std::string> in_lines;
    bool in_eof = false;
    std::thread([&] {   // stdin: STOP takes effect mid-request, every other line queues
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_req.store(true); continue; }
            std::lock_guard<std::mutex> lk(in_mu);
            in_lines.push_back(l);
            in_cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(in_mu);
        in_eof = true;
        in_cv.notify_one();
    }).detach();
    auto next_line = [&](std::string& out) {
        std::unique_lock<std::mutex> lk(in_mu);
        in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
        if (in_lines.empty()) return false;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };
    const int64_t ctx = model.max_cells(), V = model.vocab();
    std::vector<int32_t> live;   // the tokens the state holds, positions 0 .. live.size() - 1
    std::vector<int32_t> out((size_t) opt.max_t);
    std::string err;
    std::printf("INFO context=%lld expert_slots=%lld\n", (long long) ctx, (long long) model.slots());
    std::printf("READY %lld stop\n", (long long) ctx);
    std::fflush(stdout);
    std::string line;
    // what the tier learned, every save_min minutes before the next request (at QUIT: after the loop), as
    // Qwen3.8-Flash-Next's serve does (#477)
    auto saved_at = std::chrono::steady_clock::now();
    while (next_line(line)) {
        if (line == "QUIT") break;
        if (!profile_save.empty() && save_min > 0 &&
            std::chrono::steady_clock::now() - saved_at >= std::chrono::duration<double>(save_min * 60.0)) {
            if (!model.save_profile(profile_save, err))
                std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            saved_at = std::chrono::steady_clock::now();
        }
        if (line.empty()) continue;
        stop_req.store(false);   // a STOP between requests ends nothing
        if (line.rfind("GEN ", 0) != 0) {
            const std::string head = line.substr(0, line.find(' '));
            std::printf("ERR %s is not available for %s (this engine takes GEN, STOP and QUIT)\n", head.c_str(),
                        arch.c_str());
            std::fflush(stdout);
            continue;
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        strata::kernels::SamplerParams sp;   // greedy unless the request sets a temperature
        sp.greedy = true;
        sp.temperature = 0.0f;
        sp.top_p = 1.0f;
        for (;;) {   // the sampling keys between max_new and the ids; the ids start at the first token without '='
            while (endp != nullptr && *endp == ' ') ++endp;
            const char* start = endp;
            while (endp != nullptr && *endp != '\0' && *endp != ' ') ++endp;
            if (endp == nullptr || endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
            if (key == "temperature") sp.temperature = std::strtof(val.c_str(), nullptr);
            else if (key == "top_p") sp.top_p = std::strtof(val.c_str(), nullptr);
            else if (key == "top_k") sp.top_k = (int) std::strtol(val.c_str(), nullptr, 10);
            else if (key == "min_p") sp.min_p = std::strtof(val.c_str(), nullptr);
            else if (key == "seed") sp.seed = std::strtoull(val.c_str(), nullptr, 10);
        }
        sp.greedy = sp.temperature <= 0.0f;
        std::vector<int32_t> ids;
        if (max_new < 1 || endp == nullptr || !parse_ids(endp, ids) || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        const int64_t n = (int64_t) ids.size();
        if (n + max_new + 8 > ctx) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                        max_new, (long long) ctx);
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t < 0 || t >= V; })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        // what of the conversation to keep: the common prefix, at least the last prompt token read again (its
        // logits pick the first token); with a recurrent state, all of it or nothing
        size_t common = 0;
        while (common < live.size() && common < (size_t) n && live[common] == ids[common]) ++common;
        int64_t resume = std::min<int64_t>((int64_t) common, n - 1);
        if (model.recurrent() && resume != (int64_t) live.size()) resume = 0;
        if (resume == 0 && !live.empty() && !model.reset(err)) {
            std::printf("FATAL %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        live.resize((size_t) resume);
        std::printf("RESUME %lld\n", (long long) resume);
        std::fflush(stdout);
        model.set_sampler(sp);
        const Clock::time_point t0 = Clock::now();
        const int64_t hits0 = model.hits(), look0 = model.hits() + model.misses();
        const int64_t acc0 = model.spec_accepted(), off0 = model.spec_drafted();
        bool ok = true;
        int32_t next = -1;
        if (opt.prompt_chunk > 0) {
            ok = model.prompt(ids.data() + resume, n - resume, resume, out.data(), nullptr, err);
            next = out[0];
        } else {
            for (int64_t p = resume; ok && p < n;) {
                const int w = (int) std::min<int64_t>(opt.max_t, n - p);
                ok = model.forward(ids.data() + p, w, p, out.data(), nullptr, err);
                p += w;
                next = out[(size_t) w - 1];
            }
        }
        if (!ok) {   // the state is the device's, half written: the next request starts over
            live.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        live.assign(ids.begin(), ids.end());
        const Clock::time_point t1 = Clock::now();
        const char* finish = "length";
        long long produced = 0;
        // with drafts (greedy requests): a window emits the accepted drafts and the token after them, sent one by one;
        // the state holds every one of them but the last, as `live` does
        const bool spec = model.drafts() > 0 && sp.greedy;
        if (spec && !model.spec_start(next, (int64_t) live.size(), err)) ok = false;
        std::vector<int32_t> pend;
        size_t at = 0;
        while (ok) {
            std::printf("T %d\n", next);
            std::fflush(stdout);
            ++produced;
            if (std::find(eos.begin(), eos.end(), next) != eos.end()) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (at < pend.size()) {   // the window read this one already
                live.push_back(next);
                next = pend[at++];
                continue;
            }
            if (stop_req.load()) { finish = "cancel"; break; }
            if (spec) {
                int n_emit = 0;
                if (!model.spec_step(eos, max_new - produced, out.data(), n_emit, nullptr, err)) { ok = false; break; }
                live.push_back(next);
                pend.assign(out.begin(), out.begin() + n_emit);
                at = 0;
                next = pend[at++];
                continue;
            }
            if (!model.forward(&next, 1, (int64_t) live.size(), out.data(), nullptr, err)) { ok = false; break; }
            live.push_back(next);
            next = out[0];
        }
        if (!ok) {
            live.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        const Clock::time_point t2 = Clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld 0 0 0.0 %lld 0\n", produced, (long long) n,
                    prompt_ms, decode_ms, finish, (long long) (model.spec_accepted() - acc0),
                    (long long) (model.spec_drafted() - off0), (long long) resume, (long long) (model.hits() - hits0),
                    (long long) (model.hits() + model.misses() - look0), (long long) (n - resume));
        std::fflush(stdout);
    }
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    return 0;
}

static void plain_usage(const std::string& arch) {
    std::fprintf(stderr,
                 "strata --native GGUF --tokens \"1,2,3\" [options]   (%s: the models other than Qwen3.8-Flash-Next)\n"
                 "\n"
                 "  --native PATH        the model's GGUF (its first shard)\n"
                 "  --tokens LIST        the prompt as comma-separated token ids\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      the K/V cache's cells (default 4096; raised to the prompt and --max-new)\n"
                 "  --greedy             the most likely token (the default)\n"
                 "  --seed N             sample, from this seed (with --temperature, --top-k, --top-p)\n"
                 "  --temperature F      sampling temperature (default 1.0)\n"
                 "  --top-k N            sample from the N most likely (default 20)\n"
                 "  --top-p F            sample from the smallest set of this probability (default 0.95)\n"
                 "  --stop-eos           stop at an end-of-generation token\n"
                 "  --eos-ids LIST       those tokens (default the GGUF's end-of-sequence and end-of-turn tokens;\n"
                 "                       implies --stop-eos)\n"
                 "  --pool-workers N     the CPU's expert threads (default every physical core but one)\n"
                 "  --expert-profile PATH        fill the VRAM tier from this profile at load\n"
                 "  --expert-profile-save PATH   write what this run learned as a profile\n"
                 "  --expert-profile-save-every MIN   with --serve: also every MIN minutes between requests (default\n"
                 "                       10; 0: at QUIT only)\n"
                 "  --adapt-every N      windows between the tier's adaptive rounds (default 4; 0 off)\n"
                 "  --adapt-swaps N      the most experts one round moves into the VRAM tier (default 96)\n"
                 "  --adapt-mib-per-token N   and at most N MiB of them for each token read since the last round\n"
                 "                       (default 48; 0: no such bound)\n"
                 "  --mmap-experts       read the experts from the GGUF through mmap instead of pinned RAM\n"
                 "  --no-pcie-split      the CPU computes every expert the VRAM tier misses\n"
                 "  --prompt-chunk N     the prompt's most tokens a chunk through the batched products (default\n"
                 "                       4096; 0: the prompt in windows)\n"
                 "  --max-t N            the decode window's most tokens (default 8)\n"
                 "  --spec N             speculative decoding with the model's MTP layer: windows of up to N tokens,\n"
                 "                       the last one and N - 1 drafts (default 3; 0: off; greedy only)\n"
                 "  --rope-scaling none|linear|yarn, --rope-scale F, --yarn-orig-ctx N   context extension\n"
                 "  --dump-logits PATH   each generated token's logits, FP32 rows\n"
                 "  --serve              stay resident and take requests on stdin (server.py's protocol, text requests)\n"
                 "  --gpu PCI            the GPU (its PCI address)\n",
                 arch.c_str());
}

int plain_main(int argc, char** argv) {
    std::string gguf;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--native") == 0) gguf = argv[i + 1];
    if (gguf.empty()) return -1;
    const std::string arch = architecture(gguf);
    if (arch.empty() || arch == "qwen4exp") return -1;

    strata::core::PlainOptions opt;
    opt.gguf = gguf;
    std::vector<int32_t> prompt;
    int64_t n_new = 16;
    strata::kernels::SamplerParams sp;   // greedy unless --seed
    sp.greedy = true;
    bool stop_eos = false, serve = false;
    std::vector<int32_t> eos;
    std::string dump_path, profile_save;
    double save_min = 10.0;   // --expert-profile-save-every (serve)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--native") (void) next("--native");
        else if (a == "--tokens") {
            if (!parse_ids(next("--tokens"), prompt)) { std::fprintf(stderr, "--tokens: a list of token ids\n"); return 2; }
        } else if (a == "--tokens-file") {
            std::ifstream in(next("--tokens-file"));
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!in.good() && !in.eof()) { std::fprintf(stderr, "cannot read the token file\n"); return 2; }
            if (!parse_ids(text, prompt)) { std::fprintf(stderr, "--tokens-file: a list of token ids\n"); return 2; }
        }
        else if (a == "--help" || a == "-h") { plain_usage(arch); return 0; }
        else if (a == "--max-new") n_new = std::strtoll(next("--max-new"), nullptr, 10);
        else if (a == "--serve") serve = true;
        else if (a == "--greedy") sp.greedy = true;
        else if (a == "--seed") { sp.seed = std::strtoull(next("--seed"), nullptr, 10); sp.greedy = false; }
        else if (a == "--top-k") sp.top_k = (int) std::strtol(next("--top-k"), nullptr, 10);
        else if (a == "--top-p") sp.top_p = std::strtof(next("--top-p"), nullptr);
        else if (a == "--temperature") sp.temperature = std::strtof(next("--temperature"), nullptr);
        else if (a == "--stop-eos") stop_eos = true;
        else if (a == "--eos-ids") {
            if (!parse_ids(next("--eos-ids"), eos)) { std::fprintf(stderr, "--eos-ids: a list of token ids\n"); return 2; }
            stop_eos = true;
        }
        else if (a == "--max-context") opt.max_cells = std::strtoll(next("--max-context"), nullptr, 10);
        else if (a == "--max-t") opt.max_t = (int) std::strtol(next("--max-t"), nullptr, 10);
        else if (a == "--spec") opt.spec = (int) std::strtol(next("--spec"), nullptr, 10);
        else if (a == "--pool-workers") opt.pool_workers = (int) std::strtol(next("--pool-workers"), nullptr, 10);
        else if (a == "--dump-logits") dump_path = next("--dump-logits");
        else if (a == "--adapt-every") opt.adapt_every = (int) std::strtol(next("--adapt-every"), nullptr, 10);
        else if (a == "--adapt-swaps") opt.adapt_swaps = (int) std::strtol(next("--adapt-swaps"), nullptr, 10);
        else if (a == "--adapt-mib-per-token")
            opt.adapt_mib_per_token = (int) std::strtol(next("--adapt-mib-per-token"), nullptr, 10);
        else if (a == "--gpu") (void) next("--gpu");
        else if (a == "--mmap-experts") opt.mmap_experts = true;
        else if (a == "--no-pcie-split") opt.pcie_split = false;
        else if (a == "--expert-profile") opt.profile = next("--expert-profile");
        else if (a == "--prompt-chunk") opt.prompt_chunk = std::strtoll(next("--prompt-chunk"), nullptr, 10);
        else if (a == "--rope-scaling") opt.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") opt.rope_scale = std::strtod(next("--rope-scale"), nullptr);
        else if (a == "--yarn-orig-ctx") opt.yarn_orig_ctx = std::strtod(next("--yarn-orig-ctx"), nullptr);
        else if (a == "--expert-profile-save") profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every") save_min = std::strtod(next("--expert-profile-save-every"), nullptr);
        else {
            switch (option_scope(a)) {
                case OptionScope::qwen38_only:
                    std::fprintf(stderr, "strata (%s): %s is Qwen3.8-Flash-Next's (qwen4exp) only: this model has no "
                                         "such part\n", arch.c_str(), a.c_str());
                    break;
                case OptionScope::not_yet:
                    std::fprintf(stderr, "strata (%s): %s is not available for this model yet (--help lists what is)\n",
                                 arch.c_str(), a.c_str());
                    break;
                case OptionScope::unknown:
                    std::fprintf(stderr, "strata (%s): unknown option %s\n", arch.c_str(), a.c_str());
                    break;
            }
            return 2;
        }
    }
    if (!serve && (prompt.empty() || n_new < 1)) {
        std::fprintf(stderr, "strata (%s): needs --tokens and --max-new >= 1\n", arch.c_str());
        return 2;
    }
    // the last window's drafts and the MTP layer's cells reach a window past the last token
    if ((int64_t) prompt.size() + n_new + opt.max_t > opt.max_cells)
        opt.max_cells = (int64_t) prompt.size() + n_new + opt.max_t;

    if (sp.greedy) sp.temperature = 0.0f;
    if (stop_eos && eos.empty()) eos = eos_tokens(gguf);
    auto at_eos = [&](int32_t t) { return stop_eos && std::find(eos.begin(), eos.end(), t) != eos.end(); };

    strata::core::PlainModel model;
    model.set_sampler(sp);
    std::string err;
    const Clock::time_point t0 = Clock::now();
    if (opt.prompt_chunk > 0) strata::prefill::Gemm::prepare();   // the products' library loads beside the model
    if (!model.load(opt, err) || !model.prompt_warm(err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    if (opt.prompt_chunk > 0) std::fprintf(stderr, "strata (%s): prompt matrix products on %s\n", arch.c_str(),
                                           strata::prefill::Gemm::path());
    if (serve) return plain_serve(model, opt, arch, eos.empty() ? eos_tokens(gguf) : eos, profile_save, save_min);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> dump_file(nullptr, &std::fclose);
    if (!dump_path.empty()) {
        dump_file.reset(std::fopen(dump_path.c_str(), "wb"));
        if (!dump_file) {
            std::fprintf(stderr, "cannot write %s\n", dump_path.c_str());
            return 1;
        }
    }
    std::FILE* const dump = dump_file.get();
    const int T = opt.max_t;
    const int64_t V = model.vocab();
    std::vector<int32_t> out((size_t) T);
    std::vector<float> logits((size_t) T * (size_t) V);
    std::vector<int32_t> produced;
    auto emit = [&](int row) -> bool {
        produced.push_back(out[(size_t) row]);
        if (dump != nullptr &&
            std::fwrite(logits.data() + (size_t) row * (size_t) V, sizeof(float), (size_t) V, dump) != (size_t) V) {
            std::fprintf(stderr, "cannot write the logits\n");
            return false;
        }
        return true;
    };
    float* lg = dump != nullptr ? logits.data() : nullptr;
    const Clock::time_point t1 = Clock::now();
    int64_t pos = 0;
    // the prompt through the batched products (all but nothing left for the windows), else in windows
    if (opt.prompt_chunk > 0) {
        if (!model.prompt(prompt.data(), (int64_t) prompt.size(), 0, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        pos = (int64_t) prompt.size();
        if (!emit(0)) return 1;
    }
    while (pos < (int64_t) prompt.size()) {
        const int n = (int) std::min<int64_t>(T, (int64_t) prompt.size() - pos);
        if (!model.forward(prompt.data() + pos, n, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        pos += n;
        if (pos == (int64_t) prompt.size() && !emit(n - 1)) return 1;
    }
    const Clock::time_point t2 = Clock::now();
    // with drafts: a window checks them and emits the accepted ones and the token after them
    const bool spec = model.drafts() > 0 && sp.greedy;
    if (spec && (int64_t) produced.size() < n_new && !at_eos(produced.back()) &&
        !model.spec_start(produced.back(), pos, err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    const std::vector<int32_t> stop = stop_eos ? eos : std::vector<int32_t>();
    while ((int64_t) produced.size() < n_new && !at_eos(produced.back())) {
        if (spec) {
            int n_emit = 0;
            if (!model.spec_step(stop, n_new - (int64_t) produced.size(), out.data(), n_emit, lg, err)) {
                std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
                return 1;
            }
            for (int j = 0; j < n_emit; ++j)
                if (!emit(j)) return 1;
            pos += n_emit;
            continue;
        }
        const int32_t x = produced.back();
        if (!model.forward(&x, 1, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        ++pos;
        if (!emit(0)) return 1;
    }
    const Clock::time_point t3 = Clock::now();
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    std::printf("tokens: ");
    for (size_t i = 0; i < produced.size(); ++i) std::printf(i ? ",%d" : "%d", produced[i]);
    std::printf("\n");
    const double pp = std::chrono::duration<double>(t2 - t1).count(), tg = std::chrono::duration<double>(t3 - t2).count();
    const int64_t routed = model.hits() + model.misses();
    std::fprintf(stderr, "strata (%s): load %.1f s, prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s); "
                         "VRAM tier %lld of %lld slots filled, %.1f%% of routed experts hit, %lld swaps\n",
                 arch.c_str(), load_s, prompt.size(), pp, (double) prompt.size() / pp, produced.size() - 1, tg,
                 (double) (produced.size() - 1) / (tg > 0 ? tg : 1), (long long) model.resident(),
                 (long long) model.slots(), routed > 0 ? 100.0 * (double) model.hits() / (double) routed : 0.0,
                 (long long) model.swaps());
    if (model.spec_windows() > 0)
        std::fprintf(stderr, "strata (%s): %lld windows checked %lld drafts, accepted %lld (%.1f%%), %.2f tokens a window; "
                             "a window %.2f ms, its drafts %.2f ms\n",
                     arch.c_str(), (long long) model.spec_windows(), (long long) model.spec_drafted(),
                     (long long) model.spec_accepted(),
                     model.spec_drafted() > 0 ? 100.0 * (double) model.spec_accepted() / (double) model.spec_drafted() : 0.0,
                     (double) (model.spec_accepted() + model.spec_windows()) / (double) model.spec_windows(),
                     model.spec_ms_check() / (double) model.spec_windows(),
                     model.spec_ms_draft() / (double) model.spec_windows());
    return 0;
}

}  // namespace strata::program
