// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/plain_prompt.cpp - PlainModel::prompt: a plain model's prompt in chunks of batched products.
//
// The block of plain_model.cpp, T tokens at a time: every projection a matrix product of the chunk's rows by the weight
// dequantized to FP16 (Gemm::native), the GDN conv and recurrence walking the chunk inside their kernels (the prompt
// path's, kernels.hpp), and the routed experts grouped by expert - each expert's weights read once for all the tokens
// routed to it, from its VRAM slot or copied from RAM, dequantized to FP16 and multiplied in one grouped product per
// group of experts.  The norms with the rotation, the gate, the K/V append, the attention over every cell, the router
// and the combine are the decode's kernels, called over the chunk's rows.
//
// The state is left as the windows leave it: the GDN recurrent state and conv history (the same layouts), the K/V
// cells of the chunk's positions.  The decode then continues at the next position.
#include "strata/core/plain_model.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <tuple>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

/// The products' state: Gemm over the dequantization scratch carved with the other buffers.
struct PlainModel::PromptGemm {
    prefill::Gemm gemm;
    prefill::mmq::Context mmq;
    bool identity = false;   // pb_.identity written
    // the int8 products' copies from RAM on a stream of their own, into the staging's two halves in turn: ready[h]
    // when half h holds its group, used[h] when that group's products are done with it
    strata::gpu::Stream xs = nullptr;
    strata::gpu::Event *ready[2] = {}, *used[2] = {};
    bool used_live[2] = {};
    ~PromptGemm() {
        if (xs != nullptr) strata::gpu::stream_sync(xs);
        for (int h = 0; h < 2; ++h) {
            if (ready[h] != nullptr) strata::gpu::event_destroy(ready[h]);
            if (used[h] != nullptr) strata::gpu::event_destroy(used[h]);
        }
        if (xs != nullptr) strata::gpu::stream_destroy(xs);
    }
    bool overlap() const { return xs != nullptr && ready[0] && ready[1] && used[0] && used[1]; }
};

void PlainModel::prompt_gemm_free(PromptGemm* g) { delete g; }

PlainModel::PromptGemm& PlainModel::prompt_gemm() {
    if (!prompt_gemm_) {
        prompt_gemm_ = std::unique_ptr<PromptGemm, void (*)(PromptGemm*)>(new PromptGemm, &PlainModel::prompt_gemm_free);
        prompt_gemm_->gemm.init_external(cs_, pb_.gemm_scratch, pb_.gemm_elems);
        PromptGemm& g = *prompt_gemm_;
        g.xs = strata::gpu::stream_create();
        for (int h = 0; h < 2; ++h) {
            if (!strata::gpu::event_create(&g.ready[h])) g.ready[h] = nullptr;
            if (!strata::gpu::event_create(&g.used[h])) g.used[h] = nullptr;
        }
        // without them the copies stay on the compute stream, in order with the products
    }
    return *prompt_gemm_;
}

bool PlainModel::prompt_borrow(int64_t T, std::string& err) {
    if (!settle_swaps(true, err)) return false;   // the tier's copies may still be writing the slots lent below
    const int64_t L = n_moe_, E = spec_.n_expert, n = n_slots_;
    const uint64_t* off = cache_.slot_offsets();
    if (borrow_first_ >= 0 || off == nullptr || L <= 0 || n < 2) { err = "plain prompt: no tier to borrow from"; return false; }
    // the chunk's device bytes from the tier's last slots: halved until they take at most half the tier, within its
    // last piece (the slots lie back to back only inside one)
    uint64_t db = 0, hb = 0;
    const int64_t want = T;
    int64_t first = n;
    const int64_t lowest = std::max<int64_t>(n / 2, cache_.last_piece_first());
    for (;; T /= 2) {
        if (T < 16) break;
        prompt_carve(T, pb_, nullptr, nullptr, db, hb);
        first = n;
        while (first > lowest && off[n] - off[first] < db) --first;
        if (off[n] - off[first] >= db) break;
    }
    // the buffers of their own when their chunk is no shorter than the borrowed one: no slot lent, prompt_return
    // gives back none
    if (prompt_dev_ != nullptr && (T < 16 || std::min(want, prompt_dev_T_) >= T)) {
        const int64_t pieces = (want + prompt_dev_T_ - 1) / prompt_dev_T_;   // even chunks: no short last one
        T = (want + pieces - 1) / pieces;
        borrowed_res_.clear();
        borrow_first_ = n;
        prompt_carve(T, pb_, (uint8_t*) prompt_dev_, (uint8_t*) prompt_host_, db, hb);
        PromptGemm& g = prompt_gemm();
        g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
        g.identity = false;
        return true;
    }
    if (T < 16) { err = "plain prompt: the VRAM tier is too small for the prompt's buffers"; return false; }
    borrowed_res_.clear();
    for (int64_t l = 0; l < L; ++l)
        for (int64_t e = 0; e < E; ++e) {
            int32_t& r = res_[(size_t) (l * E + e)];
            if (r >= first) {
                borrowed_res_.emplace_back((int32_t) l, (int32_t) e);
                r = -1;
            }
        }
    for (auto& f : free_) f.erase(std::remove_if(f.begin(), f.end(), [&](int32_t x) { return x >= first; }), f.end());
    borrow_first_ = first;
    prompt_carve(T, pb_, cache_.device_slot((int32_t) first), (uint8_t*) prompt_host_, db, hb);
    PromptGemm& g = prompt_gemm();
    g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
    g.identity = false;   // the identity map is rewritten where the buffers now lie
    return true;
}

bool PlainModel::prompt_return(std::string& err) {
    const int64_t L = n_moe_, n = n_slots_;
    if (borrow_first_ < 0 || L <= 0) return true;   // a borrow needs slots, and slots need MoE layers
    const auto& lay = k::cpu::expert_layout();
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain prompt: ") + strata::gpu::last_error(); return false; }
    for (int64_t s = borrow_first_; s < n; ++s) free_[(size_t) (s % L)].push_back((int32_t) s);
    borrow_first_ = -1;
    // the experts the prompt displaced, back into the slots: straight from pinned RAM, else through fill_slot
    for (const auto& [l, e] : borrowed_res_) {
        auto& f = free_[(size_t) l];
        if (f.empty()) break;
        const uint8_t* hb = src_->transient(l, e) ? nullptr : src_->blob(l, e);
        if (hb != nullptr && strata::gpu::is_host_usm(hb)) {
            const int32_t slot = f.back();
            if (!strata::gpu::copy_async(cache_.device_slot(slot), hb, lay.blob_bytes(l), cs_)) {
                err = std::string("plain prompt: ") + strata::gpu::last_error();
                return false;
            }
            f.pop_back();
            res_[(size_t) (l * spec_.n_expert + e)] = slot;
        } else if (!fill_slot(l, e, err)) {
            return false;
        }
    }
    borrowed_res_.clear();
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain prompt: ") + strata::gpu::last_error(); return false; }
    return true;
}

bool PlainModel::prompt_cpu_experts(int64_t ml, const std::vector<int32_t>& experts, const std::vector<int32_t>& count,
                                    const std::vector<int32_t>& start, std::string& err) {
    // On the host while the GPU computes its share: each routed row's token quantized once to the layer's native
    // activation, one multi-token job an expert, the outputs back to the GPU into their routing entries' rows of parts.
    namespace kc = k::cpu;
    PromptBufs& P = pb_;
    const int64_t N = spec_.n_embd;
    const kc::NativeFmt& f = kc::expert_layout().fmt[(size_t) ml];
    std::vector<char> quantized((size_t) P.T, 0);
    std::vector<kc::ExpertJobMulti> jobs;
    jobs.reserve(experts.size());
    int64_t q = 0;
    for (int32_t e : experts) {
        kc::ExpertJobMulti j;
        j.blob = src_->blob(ml, e);
        if (j.blob == nullptr) { err = "plain prompt: an expert could not be read for the CPU"; return false; }
        for (int32_t r = start[(size_t) e]; r < start[(size_t) e] + count[(size_t) e]; ++r) {
            const int32_t tok = P.h_src[r];
            uint8_t* act = P.h_nact + (size_t) tok * kc::kNativeActBytes;
            if (!quantized[(size_t) tok]) {
                kc::native_quant_act(f, P.h_hn + (size_t) tok * N, act);
                quantized[(size_t) tok] = 1;
            }
            j.nact[j.nt] = act;
            j.out[j.nt] = P.h_cpu + q * N;
            P.h_cpu_row[q] = P.h_entry[r];
            ++j.nt;
            ++q;
        }
        jobs.push_back(j);
    }
    for (size_t i = 0; i < jobs.size(); i += kc::ExpertPool::kMaxSplitMulti)
        pool_->run_split_multi_native(f, jobs.data() + i,
                                      (int) std::min<size_t>(kc::ExpertPool::kMaxSplitMulti, jobs.size() - i));
    if (!strata::gpu::copy_async(P.cpu, P.h_cpu, (size_t) (q * N) * 4, cs_) ||
        !strata::gpu::copy_async(P.cpu_row, P.h_cpu_row, (size_t) q * 4, cs_)) {
        err = std::string("plain prompt: the CPU's rows: ") + strata::gpu::last_error();
        return false;
    }
    prefill::scatter_rows(P.cpu, P.cpu_row, P.parts, q, N, cs_);
    return true;
}

bool PlainModel::prompt_warm(std::string& err) {
    if (opt_.prompt_chunk <= 0 || prompt_host_ == nullptr) return true;
    if (!prompt_borrow(std::min<int64_t>(opt_.prompt_chunk, 512), err)) return false;
    const bool ok = prompt_warm_products(err);
    std::string err2;
    if (!prompt_return(err2) && ok) { err = err2; return false; }
    return ok;
}

bool PlainModel::prompt_warm_products(std::string& err) {
    // A library's first product of a kind loads its kernels for it, inside whichever prompt comes first (rocBLAS on
    // the RX 9060 XT: 0.5 s for the first product, 0.9 s more for the first of 32 rows): each dense shape the prompt
    // takes, once, on a chunk of zeros into the buffers the prompt overwrites.
    PromptBufs& P = pb_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    const int64_t T = P.T, HV = spec_.ssm_dt_rank;
    std::set<std::tuple<int, int, int, int64_t>> seen;
    auto warm = [&](const W& w, float* y, int64_t ldy = 0) {
        if (w.data != nullptr && seen.insert({w.type, w.n_out, w.n_in, ldy}).second)
            gm.native(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    try {
        if (!strata::gpu::memset_async(P.x16, 0, (size_t) (T * spec_.n_embd) * 2, cs_)) {
            err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
            return false;
        }
        std::vector<const Block*> all;
        for (const Block& b : blocks_) all.push_back(&b);
        if (has_mtp_) {
            all.push_back(&mtp_);
            warm(mtp_eh_, P.x);
        }
        for (const Block* bp : all) {
            const Block& b = *bp;
            if (b.recurrent) {
                warm(b.qkv, P.qkv);
                warm(b.z, P.z);
                warm(b.alpha, P.ab, 2 * HV);
                warm(b.beta, P.ab + HV, 2 * HV);
                warm(b.ssm_out, P.mix);
            } else {
                warm(b.q, P.qfull);
                warm(b.k, P.kraw);
                warm(b.v, P.vcur);
                warm(b.o, P.mix);
            }
            if (has_shared_) {
                warm(b.sh_gate, P.sh_gate);
                warm(b.sh_up, P.sh_up);
                warm(b.sh_down, P.shared);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: warming: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool PlainModel::prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err) {
    if (opt_.prompt_chunk <= 0 || n <= 0 || prompt_host_ == nullptr) { err.clear(); return false; }
    if (pos < 0 || pos + n > opt_.max_cells) { err = "plain prompt: past the context"; return false; }
    // one chunk the prompt's length, up to prompt_chunk (prompt_borrow may take fewer)
    if (!prompt_borrow(std::min<int64_t>(n, opt_.prompt_chunk), err)) return false;
    const bool ok = prompt_chunks(tokens, n, pos, out, logits, err);
    std::string err2;
    if (!prompt_return(err2) && ok) { err = err2; return false; }
    return ok;
}

bool PlainModel::prompt_chunks(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                               std::string& err) {
    PromptBufs& P = pb_;
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, NKV = nkv_, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t E = s.n_expert, K = s.n_expert_used;
    const int64_t FS = s.n_ff_shexp, FF = s.n_ff_exp, qw = (has_gate_ ? 2 : 1) * NH * HD;
    const float eps = (float) s.rms_eps;
    const k::QsaShapes qs = [&] {
        k::QsaShapes q = k::qsa_real_shapes();
        q.n_head = NH; q.n_head_kv = NKV; q.head_dim = HD; q.n_rot = s.rope_dims;
        return q;
    }();
    const k::RopeScaling& rs = k::rope_scaling();
    strata::gpu::Stream cs = cs_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    auto f16 = [&](const float* x, int64_t rows, int64_t cols) { prefill::to_f16(x, P.x16, rows * cols, cs); };
    // a dense product from float rows x (ldx a row) in int8 (iq_mmq, as one expert of T rows) where the weight's type
    // and shape are the int8 products'; false: the caller takes Gemm::native from the FP16 copy
    static const bool dense_env = [] {   // STRATA_PLAIN_DENSE_MMQ=0: every dense product through Gemm::native
        const char* v = std::getenv("STRATA_PLAIN_DENSE_MMQ");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool dense_on = dense_env && prefill::mmq::chosen();   // as for the experts: PlainModel::prompt_int8
    auto dense = [&](const W& w, const float* x, int64_t ldx, float* y, int64_t T, int64_t ldy = 0) {
        if (!dense_on || w.data == nullptr || !prefill::mmq::supported(w.type) || w.n_out % 128 != 0 || w.n_in % 32 != 0)
            return false;
        prefill::mmq::quantize(x, nullptr, P.xq, w.type, w.n_in, ldx, T, cs);
        prefill::mmq::Product p;
        p.w = w.data; p.type = w.type; p.w_rows = w.n_out; p.w_cols = w.n_in; p.n = 1; p.xq = P.xq;
        p.bounds = P.dense_bounds; p.ids = P.identity; p.total_rows = T; p.max_rows = T;
        p.dst = y; p.ld_dst = ldy > 0 ? ldy : w.n_out;
        prompt_gemm().mmq.run(p, cs);
        return true;
    };
    auto proj = [&](const W& w, float* y, int64_t T, int64_t ldy = 0) {
        gm.native(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    // the routed experts in their quantized blocks (int8 products, iq_mmq: on the matrix engines, else by dp4a where the
    // FP16 products have none either) where every layer's types are theirs (prompt_int8, decided with the buffers),
    // else dequantized to FP16
    const bool use_mmq = P.int8;
    // the CPU's share of the missed experts needs the split's times (measured at load from pinned RAM, on the decode
    // window's kernels: the int8 and the FP16 path take them alike); STRATA_PLAIN_CPU_SHARE=0: every expert on the GPU
    static const bool cpu_env = [] {
        const char* v = std::getenv("STRATA_PLAIN_CPU_SHARE");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool cpu_share = cpu_env && split_.c > 0.0 && split_.p > 0.0 && P.cpu_cap > 0;
    static const bool prompt_attn_on = [] {   // STRATA_PLAIN_PROMPT_ATTN=0: the decode's attention kernel
        const char* v = std::getenv("STRATA_PLAIN_PROMPT_ATTN");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    PromptGemm& pg = prompt_gemm();
    if (use_mmq && !pg.identity) {
        prefill::mmq::iota(P.identity, P.T * K, cs);
        pg.identity = true;
    }
    // STRATA_PROMPT_PROFILE=1: the time of each stage, the stream synchronized after each (slower; for finding where it
    // goes): 0 the GDN layers, 4 the attention layers, 1 the router and the host's grouping, 2 the experts, 3 the shared
    // expert and the combine
    static const bool profile = std::getenv("STRATA_PROMPT_PROFILE") != nullptr;
    double stage_ms[10] = {};   // 5-9: the GDN layers' parts (stage 0 is their sum)
    int64_t cpu_n = 0, cpu_rows = 0, miss_n = 0;   // for the profile: the CPU's experts and rows, the missed experts
    auto mark = [&](int stage, std::chrono::steady_clock::time_point& t) {
        if (!profile) return;
        strata::gpu::stream_sync(cs);
        const auto now = std::chrono::steady_clock::now();
        stage_ms[stage] += std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
    };
    auto t_mark = std::chrono::steady_clock::now();
    try {
        // the chunk's routes, every MoE layer's, for the adaptive tier's counts (below, after the chunk)
        const int64_t L = n_moe_;
        std::vector<int32_t> routes(disp_.usage.empty() ? 0 : (size_t) (L * P.T * K));
        for (int64_t c0 = 0; c0 < n; c0 += P.T) {
            const int64_t T = std::min<int64_t>(P.T, n - c0), p0 = pos + c0;
            // the chunk's inputs: embedding rows, per-token step counts and rotary positions
            for (int64_t t = 0; t < T; ++t) {
                const int32_t tok = tokens[c0 + t];
                if (tok < 0 || tok >= n_vocab_ ||
                    !k::cpu::native_row_to_float(emb_type_, emb_ + (size_t) tok * emb_row_, P.emb + t * N, N)) {
                    err = "plain prompt: a token is out of the vocabulary";
                    return false;
                }
                k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + t, qs);
                for (int64_t hd = 0; hd < NH; ++hd) P.h_pos[t * NH + hd] = (int32_t) (p0 + t);
                for (int64_t hd = 0; hd < NKV; ++hd) P.h_pos[T * NH + t * NKV + hd] = (int32_t) (p0 + t);
            }
            if (!strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs) ||
                !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs) ||
                !strata::gpu::copy_async(P.pos_q, P.h_pos, (size_t) (T * NH) * 4, cs) ||
                !strata::gpu::copy_async(P.pos_k, P.h_pos + T * NH, (size_t) (T * NKV) * 4, cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            P.h_dense_bounds[0] = 0;
            P.h_dense_bounds[1] = (int32_t) T;
            if (!strata::gpu::copy_async(P.dense_bounds, P.h_dense_bounds, 2 * sizeof(int32_t), cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            if (dense_on && !pg.identity) {   // the dense products' row map (the experts' sets it too)
                prefill::mmq::iota(P.identity, P.T * K, cs);
                pg.identity = true;
            }
            // from the normed rows: int8 where it can, else FP16 (x16, written once a layer if any product needs it)
            auto proj_hn = [&](const W& w, float* y, int64_t ldy = 0) {
                if (!dense(w, P.hn, N, y, T, ldy)) proj(w, y, T, ldy);
            };
            // the last token's logits, by the decode's head
            auto last_logits = [&] {
                if (c0 + T < n) return;
                k::native_qsa_rms_norm_weighted(P.x + (T - 1) * N, out_norm_, P.hn, (int) N, 1, eps, cs);
                if (head_.type == 30) {   // a BF16 head, as the decode reads it
                    k::bf16_gemv_fp32_mmvf_multi(P.hn, head_.n_in, static_cast<const uint16_t*>(head_.data), logits_,
                                                 head_.n_out, head_.n_in, head_.n_out, 1, cs);
                } else {
                    k::native_quantize_q8_1(P.hn, xq_, head_.n_in, 1, cs);
                    k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, 1, cs);
                }
                k::SamplerParams sp = sampler_;
                sp.counter = (uint64_t) (p0 + T - 1);   // the last token's position, as forward counts
                k::sample_tokens(logits_, 1, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
            };
            // the MTP layer after the trunk, its K/V a cell ahead: row t pairs the trunk's normed row t with the token
            // after it (the prompt's last row with a stand-in, rewritten by spec_start once the token is picked)
            const bool mtp_chunk = has_mtp_ && p0 + T < opt_.max_cells;
            for (int64_t l = 0; l < s.n_layer + (mtp_chunk ? 1 : 0); ++l) {
                const bool is_mtp = l == s.n_layer;
                if (is_mtp) {
                    last_logits();
                    k::native_qsa_rms_norm_weighted(P.x, out_norm_, P.hn, (int) N, (int) T, eps, cs);
                    if (c0 + T >= n &&
                        !strata::gpu::copy_async(mtp_h_, P.hn + (T - 1) * N, (size_t) N * 4, cs)) {
                        err = std::string("plain prompt: the MTP layer's input: ") + strata::gpu::last_error();
                        return false;
                    }
                    k::native_qsa_rms_norm_weighted(P.hn, mtp_hnorm_, P.mix, (int) N, (int) T, eps, cs);
                    for (int64_t t = 0; t < T; ++t) {   // the stream has synchronized since the chunk's uploads
                        const int32_t tok = c0 + t + 1 < n ? tokens[c0 + t + 1] : tokens[n - 1];
                        if (!k::cpu::native_row_to_float(emb_type_, emb_ + (size_t) tok * emb_row_, P.emb + t * N, N)) {
                            err = "plain prompt: a token is out of the vocabulary";
                            return false;
                        }
                        k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + 1 + t, qs);
                        for (int64_t hd = 0; hd < NH; ++hd) P.h_pos[t * NH + hd] = (int32_t) (p0 + 1 + t);
                        for (int64_t hd = 0; hd < NKV; ++hd) P.h_pos[T * NH + t * NKV + hd] = (int32_t) (p0 + 1 + t);
                    }
                    if (!strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs) ||
                        !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs) ||
                        !strata::gpu::copy_async(P.pos_q, P.h_pos, (size_t) (T * NH) * 4, cs) ||
                        !strata::gpu::copy_async(P.pos_k, P.h_pos + T * NH, (size_t) (T * NKV) * 4, cs)) {
                        err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                        return false;
                    }
                    k::native_qsa_rms_norm_weighted(P.x, mtp_enorm_, P.hn, (int) N, (int) T, eps, cs);
                    k::concat_rows(P.qfull, P.hn, P.mix, N, T, cs);
                    if (!dense(mtp_eh_, P.qfull, 2 * N, P.x, T)) {
                        f16(P.qfull, T, 2 * N);
                        proj(mtp_eh_, P.x, T);
                    }
                }
                const Block& b = is_mtp ? mtp_ : blocks_[(size_t) l];
                const int64_t pa = is_mtp ? p0 + 1 : p0;   // the rows' first position
                k::native_qsa_rms_norm_weighted(P.x, b.attn_norm, P.hn, (int) N, (int) T, eps, cs);
                f16(P.hn, T, N);
                if (b.recurrent) {
                    auto part = [&](int stage) {   // the profile's GDN parts, summed into stage 0 as well
                        if (!profile) return;
                        const auto t0 = t_mark;
                        mark(stage, t_mark);
                        if (stage != 0) stage_ms[0] += std::chrono::duration<double, std::milli>(t_mark - t0).count();
                    };
                    part(0);   // the norm and its FP16 copy
                    proj_hn(b.qkv, P.qkv);
                    proj_hn(b.z, P.z);
                    part(5);
                    proj(b.alpha, P.ab, T, 2 * HV);
                    proj(b.beta, P.ab + HV, T, 2 * HV);
                    part(6);
                    prefill::gdn_gates(P.ab, b.dt, b.a, P.gate, P.beta, T, cs, HK, HV);
                    prefill::gdn_conv(gdn_conv_[(size_t) b.gdn], P.qkv, b.conv, P.h, T, eps, cs, HK, HV);
                    part(7);
                    prefill::gdn_recurrence(gdn_state_[(size_t) b.gdn], P.h, P.gate, P.beta, P.z, b.ssm_norm, eps, P.y,
                                            P.y16, T, cs, HK, HV, /*silu=*/true);
                    part(8);
                    // (y16, the normed and gated output, exists only in FP16: no float rows to round to int8)
                    gm.native(P.y16, b.ssm_out.type, b.ssm_out.data, P.mix, T, b.ssm_out.n_out, b.ssm_out.n_in);
                    part(9);
                } else {
                    const int64_t a = b.attn;
                    proj_hn(b.q, P.qfull);
                    proj_hn(b.k, P.kraw);
                    proj_hn(b.v, P.vcur);
                    // the norm and RoPE take at most 65535 rows a launch: the chunk's heads in token slices
                    auto norm_rope = [&](const float* in, int64_t in_row, const float* gamma, float* out, int64_t heads,
                                         const int32_t* pos) {
                        const int64_t per = 65535 / heads;
                        for (int64_t t0 = 0; t0 < T; t0 += per)
                            k::native_qsa_rms_norm_rope(in + t0 * heads * in_row, (int) in_row, gamma, out + t0 * heads * HD,
                                                        (int) (std::min(per, T - t0) * heads), (int) HD,
                                                        (int) s.rope_dims, eps, rs, pos + t0 * heads, cs);
                    };
                    norm_rope(P.qfull, qw / NH, b.q_norm, P.qcur, NH, P.pos_q);
                    norm_rope(P.kraw, HD, b.k_norm, P.kcur, NKV, P.pos_k);
                    k::kv_append_steps(k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_, P.steps, k::kStepCount,
                                       P.kcur, P.vcur, (int) (NKV * HD), (int) T, qs, cs);
                    k::QsaAttnPools pools;
                    pools.k_pool = k_pool_[(size_t) a];
                    pools.v_pool = v_pool_[(size_t) a];
                    pools.page_table = page_table_;
                    // the prompt attention (prompt_attn: K and V read once a block of queries), else the decode's
                    // kernel: every query reads the cells up to its own; the grid as wide as the chunk's last context (not
                    // the cache's capacity: no graph holds it), as many queries a launch as the scratch takes
                    const int64_t cap = std::min<int64_t>(opt_.max_cells, pa + T);
                    const int64_t per_q = (int64_t) k::qsa_decode_attn_scratch_floats(cap, qs);
                    const int64_t nq_max = std::max<int64_t>(1, std::min<int64_t>((int64_t) opt_.max_t * attn_scratch_floats_ / per_q, 4096));
                    const bool batched = prompt_attn_on &&
                                         k::prompt_attn(P.qcur, k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_, pa,
                                                        T, qs, P.attn, cs);
                    for (int64_t t = 0; !batched && t < T; t += nq_max) {
                        const int nq = (int) std::min<int64_t>(nq_max, T - t);
                        k::qsa_dense_attn_batch(P.qcur + t * NH * HD, pools, P.steps + t * k::kStepCount, cap, qs,
                                                attn_scratch_, P.attn + t * NH * HD, nq, cs);
                    }
                    const float* ao = P.attn;
                    if (has_gate_) {
                        k::native_qsa_gate_apply(P.attn, P.qfull, P.attn32, (int) (T * NH), (int) HD, cs);
                        ao = P.attn32;
                    }
                    if (!dense(b.o, ao, NH * HD, P.mix, T)) {
                        f16(ao, T, NH * HD);
                        proj(b.o, P.mix, T);
                    }
                }
                k::add_inplace(P.x, P.mix, T * N, cs);
                mark(b.recurrent ? 0 : 4, t_mark);

                // the MoE
                k::native_qsa_rms_norm_weighted(P.x, b.post_norm, P.hn, (int) N, (int) T, eps, cs);
                k::f32_gemm_rows(P.hn, N, b.router, P.rlog, E, N, E, T, cs);
                k::router_top10(P.rlog, (int) T, (int) E, (int) K, P.ids, P.w, cs);
                f16(P.hn, T, N);
                if (has_shared_) {
                    proj_hn(b.sh_gate, P.sh_gate);
                    proj_hn(b.sh_up, P.sh_up);
                    k::f32_gemm_rows(P.hn, N, b.sh_gate_inp, P.sh_g, 1, N, 1, T, cs);
                }
                // the routed experts grouped by expert, on the host from this layer's ids (and the normed rows, for the
                // CPU's share)
                if ((cpu_share && !strata::gpu::copy_async(P.h_hn, P.hn, (size_t) (T * N) * 4, cs)) ||
                    !strata::gpu::copy_async(P.h_ids, P.ids, (size_t) (T * K) * 4, cs) || !strata::gpu::stream_sync(cs)) {
                    err = std::string("plain prompt: the routes: ") + strata::gpu::last_error();
                    return false;
                }
                const int64_t ml = is_mtp ? n_moe_ - 1 : s.moe_index(l);
                std::vector<int32_t> count((size_t) E, 0);
                for (int64_t i = 0; i < T * K; ++i) {
                    const int32_t e = P.h_ids[i];
                    if (e < 0 || e >= E) { err = "plain prompt: a routed expert is out of range"; return false; }
                    ++count[(size_t) e];
                    heat_[(size_t) (ml * E + e)] += 1.0;
                }
                if (!routes.empty()) std::copy(P.h_ids, P.h_ids + T * K, routes.begin() + ml * P.T * K);
                const auto& f = lay.fmt[(size_t) ml];
                // the CPU's share of the missed experts (after Project Maya's CPU lane, as the windows split them): of
                // the f experts not in VRAM, pcie_split_m's count over PCIe from the times measured at load, the rest
                // on the CPU pool beside the GPU, those with the fewest rows first (a window's tokens at most: one
                // multi-token job, its weights read once)
                std::vector<char> on_cpu((size_t) E, 0);
                std::vector<int32_t> cpu_experts;
                if (cpu_share && f.gu_type != 42) {
                    std::vector<std::pair<int32_t, int32_t>> missed;
                    int hits = 0;
                    for (int64_t e = 0; e < E; ++e)
                        if (count[(size_t) e] > 0) {
                            if (res_[(size_t) (ml * E + e)] >= 0) ++hits;
                            else missed.emplace_back(count[(size_t) e], (int32_t) e);
                        }
                    int want = (int) missed.size() - pcie_split_m(split_, true, (int) missed.size(), hits);
                    std::sort(missed.begin(), missed.end());
                    int64_t rows = 0;
                    for (const auto& [c, e] : missed) {
                        if (want <= 0 || c > k::cpu::MAXT || rows + c > P.cpu_cap) break;
                        on_cpu[(size_t) e] = 1;
                        cpu_experts.push_back(e);
                        rows += c;
                        --want;
                    }
                }
                // rows in expert order, the GPU's experts first (their bounds one run), the CPU's after them
                std::vector<int32_t> experts, start((size_t) E, 0);
                int64_t gpu_rows = 0;
                {
                    int32_t at = 0;
                    for (int pass = 0; pass < 2; ++pass)
                        for (int64_t e = 0; e < E; ++e)
                            if (count[(size_t) e] > 0 && on_cpu[(size_t) e] == pass) {
                                start[(size_t) e] = at;
                                at += count[(size_t) e];
                                if (pass == 0) experts.push_back((int32_t) e);
                            }
                    for (int32_t e : experts) gpu_rows += count[(size_t) e];
                }
                {
                    std::vector<int32_t> fill(start);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = P.h_ids[i], row = fill[(size_t) e]++;
                        P.h_src[row] = (int32_t) (i / K);   // the token whose row it reads
                        P.h_slot[i] = row;                  // where routing entry i finds its expert's output
                        P.h_entry[row] = (int32_t) i;       // the int8 down product's: the row it writes
                    }
                }
                // the group bounds: the routed rows of the experts in `experts` order, absolute
                int64_t nb = 0;
                for (int32_t e : experts) P.h_bounds[nb++] = start[(size_t) e];
                P.h_bounds[nb] = (int32_t) gpu_rows;
                if (!strata::gpu::copy_async(P.src, P.h_src, (size_t) (T * K) * 4, cs) ||
                    !(use_mmq ? strata::gpu::copy_async(P.entry, P.h_entry, (size_t) (T * K) * 4, cs)
                              : strata::gpu::copy_async(P.slot, P.h_slot, (size_t) (T * K) * 4, cs)) ||
                    !strata::gpu::copy_async(P.bounds, P.h_bounds, (size_t) (nb + 1) * 4, cs)) {
                    err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
                    return false;
                }
                // a layer whose gate and up differ in type: up's rows too, into xq2 where the two repeat the
                // blocks differently (Q2_K against another type)
                const int up_type = k::cpu::native_mixed_gu(f) ? f.up_type : f.gu_type;
                const bool up_xq2 = k::iq_mmq_type_dup(up_type) != k::iq_mmq_type_dup(f.gu_type);
                if (use_mmq) prefill::mmq::quantize(P.hn, P.src, P.xq, f.gu_type, N, N, T * K, cs);
                if (use_mmq && up_xq2) prefill::mmq::quantize(P.hn, P.src, P.xq2, up_type, N, N, T * K, cs);
                else prefill::gather_rows16(P.x16, P.src, P.xs16, T * K, N, cs);
                mark(1, t_mark);
                const bool transient = src_->transient(ml, experts.empty() ? 0 : experts[0]);
                int64_t max_rows = 0;
                for (int32_t e : experts) max_rows = std::max<int64_t>(max_rows, count[(size_t) e]);
                // int8: as many experts a launch as the staging lets (those in VRAM read where they lie, kGroup from
                // RAM at most), gate/up, the SwiGLU, its rows to int8, down
                for (size_t g0 = 0, gi = 0; use_mmq && g0 < experts.size(); ++gi) {
                    if (transient && !strata::gpu::stream_sync(cs)) {   // the staging's last readers are done
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    size_t g1 = g0;
                    int staged = 0;
                    int64_t mr = 0;
                    // the half of the staging this group takes; its copies wait for the products two groups back
                    const bool ovl = !transient && pg.overlap();
                    const int h = (int) (gi % 2);
                    strata::gpu::Stream ccs = ovl ? pg.xs : cs;
                    if (ovl && pg.used_live[h] && !strata::gpu::stream_wait_event(ccs, pg.used[h])) {
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    for (; g1 < experts.size(); ++g1) {
                        const int32_t e = experts[g1];
                        const int32_t sl = res_[(size_t) (ml * E + e)];
                        const uint8_t* blob = nullptr;
                        if (sl >= 0) {
                            blob = cache_.device_slot(sl);
                        } else {
                            if (staged == PromptBufs::kGroup) break;
                            uint8_t* dst = P.stage + (size_t) ((ovl ? h * PromptBufs::kGroup : 0) + staged++) * lay.max_blob;
                            const uint8_t* hb = src_->blob(ml, e);
                            const bool copied = hb != nullptr &&
                                (transient ? strata::gpu::copy(dst, hb, f.bytes) : strata::gpu::copy_async(dst, hb, f.bytes, ccs));
                            if (!copied) { err = "plain prompt: an expert could not be read"; return false; }
                            blob = dst;
                        }
                        P.h_wtab[g1] = blob;
                        mr = std::max<int64_t>(mr, count[(size_t) e]);
                    }
                    if (ovl && (!strata::gpu::event_record(pg.ready[h], ccs) || !strata::gpu::stream_wait_event(cs, pg.ready[h]))) {
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    const int G = (int) (g1 - g0);
                    const int32_t r0 = P.h_bounds[g0], nr = P.h_bounds[g1] - r0;
                    int32_t* hrb = P.h_rbounds + g0 + gi;
                    for (int q = 0; q <= G; ++q) hrb[q] = P.h_bounds[g0 + (size_t) q] - r0;
                    if (!strata::gpu::copy_async(static_cast<void*>(P.wtab + g0), static_cast<const void*>(P.h_wtab + g0),
                                                 (size_t) G * sizeof(void*), cs) ||
                        !strata::gpu::copy_async(P.rbounds + g0 + gi, hrb, (size_t) (G + 1) * 4, cs)) {
                        err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
                        return false;
                    }
                    prefill::mmq::Product gu;
                    gu.type = f.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.n = G; gu.xq = P.xq;
                    gu.bounds = P.bounds + g0; gu.ids = P.identity; gu.total_rows = T * K; gu.max_rows = mr; gu.rows = nr;
                    gu.dst = P.gu; gu.ld_dst = 2 * FF; gu.wtab = P.wtab + g0; gu.woff = 0;
                    if (k::cpu::native_mixed_gu(f)) {   // the gate rows, then the up rows in their own type
                        gu.w_rows = FF;
                        pg.mmq.run(gu, cs);
                        gu.type = up_type; gu.xq = up_xq2 ? P.xq2 : P.xq; gu.dst = P.gu + FF; gu.woff = f.up_off;
                    }
                    pg.mmq.run(gu, cs);
                    prefill::mmq::swiglu(P.gu + (int64_t) r0 * 2 * FF, P.hh + (int64_t) r0 * FF, nr, FF,
                                         /*interleaved=*/false, cs);
                    prefill::mmq::quantize(P.hh + (int64_t) r0 * FF, nullptr, P.hq, f.d_type, FF, FF, nr, cs);
                    prefill::mmq::Product dn;
                    dn.type = f.d_type; dn.w_rows = N; dn.w_cols = FF; dn.n = G; dn.xq = P.hq;
                    dn.bounds = P.rbounds + g0 + gi; dn.ids = P.entry + r0; dn.total_rows = nr; dn.max_rows = mr;
                    dn.dst = P.parts; dn.ld_dst = N; dn.wtab = P.wtab + g0; dn.woff = f.down_off;   // routing order
                    pg.mmq.run(dn, cs);
                    if (ovl) {
                        if (!strata::gpu::event_record(pg.used[h], cs)) {
                            err = std::string("plain prompt: ") + strata::gpu::last_error();
                            return false;
                        }
                        pg.used_live[h] = true;
                    }
                    g0 = g1;
                }
                // the CPU's rows go into parts after the GPU's products are queued (it computes beside them): here on
                // the int8 path, which writes parts itself; after the FP16 path's gather, which writes every routing
                // entry's row of parts (the CPU's from rows of dm no product wrote)
                if (use_mmq && !cpu_experts.empty() && !prompt_cpu_experts(ml, cpu_experts, count, start, err))
                    return false;
                cpu_n += (int64_t) cpu_experts.size();
                for (int32_t e : cpu_experts) cpu_rows += count[(size_t) e];
                for (int64_t e = 0; e < E; ++e) miss_n += count[(size_t) e] > 0 && res_[(size_t) (ml * E + e)] < 0;
                for (size_t g0 = 0; !use_mmq && g0 < experts.size(); g0 += PromptBufs::kGroup) {
                    const int G = (int) std::min<size_t>(PromptBufs::kGroup, experts.size() - g0);
                    if (transient && !strata::gpu::stream_sync(cs)) {   // the staging's last readers are done
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    for (int q = 0; q < G; ++q) {   // each expert's weights, then FP16
                        const int32_t e = experts[g0 + (size_t) q];
                        const int32_t sl = res_[(size_t) (ml * E + e)];
                        const uint8_t* blob = nullptr;
                        if (sl >= 0) {
                            blob = cache_.device_slot(sl);
                        } else {
                            uint8_t* dst = P.stage + (size_t) q * lay.max_blob;
                            const uint8_t* hb = src_->blob(ml, e);
                            const bool copied = hb != nullptr &&
                                (transient ? strata::gpu::copy(dst, hb, f.bytes) : strata::gpu::copy_async(dst, hb, f.bytes, cs));
                            if (!copied) { err = "plain prompt: an expert could not be read"; return false; }
                            blob = dst;
                        }
                        uint16_t* dq = P.dq_gu + (size_t) q * P.gu_elems;
                        if (k::cpu::native_mixed_gu(f)) {   // gate and up of different types: each by its own
                            k::dequant_f16(f.gu_type, blob, 0, FF, N, dq, cs);
                            k::dequant_f16(f.up_type, blob + f.up_off, 0, FF, N, dq + FF * N, cs);
                        } else {
                            k::dequant_f16(f.gu_type, blob, 0, 2 * FF, N, dq, cs);
                        }
                        k::dequant_f16(f.d_type, blob + f.down_off, 0, N, FF, P.dq_d + (size_t) q * P.d_elems, cs);
                    }
                    const int32_t* hbnd = P.h_bounds + g0;
                    const int32_t* dbnd = P.bounds + g0;
                    const int64_t r0 = hbnd[0], r1 = hbnd[G];
                    if (!prefill::Gemm::f16_groups(P.xs16, P.dq_gu, P.gu_elems, P.gu, hbnd, G, 2 * FF, N, cs))
                        k::xmx_gemm_grouped(P.xs16, P.dq_gu, P.gu_elems, P.gu, dbnd, G, max_rows, 2 * FF, N, cs);
                    // gate rows then up rows (the GGUF order): h = silu(gate) * up, then FP16 for the down product
                    prefill::mmq::swiglu(P.gu + r0 * 2 * FF, P.hh + r0 * FF, r1 - r0, FF, /*interleaved=*/false, cs);
                    prefill::to_f16(P.hh + r0 * FF, P.h16 + r0 * FF, (r1 - r0) * FF, cs);
                    if (!prefill::Gemm::f16_groups(P.h16, P.dq_d, P.d_elems, P.dm, hbnd, G, N, FF, cs))
                        k::xmx_gemm_grouped(P.h16, P.dq_d, P.d_elems, P.dm, dbnd, G, max_rows, N, FF, cs);
                }
                mark(2, t_mark);
                // routing entry i's output into row i, then the weighted sum (and the gated shared expert)
                if (!use_mmq)
                    k::gather_rows(reinterpret_cast<const uint8_t*>(P.dm), N * 4, P.slot, T * K,
                                   reinterpret_cast<uint8_t*>(P.parts), cs);
                if (!use_mmq && !cpu_experts.empty() && !prompt_cpu_experts(ml, cpu_experts, count, start, err))
                    return false;
                if (has_shared_) {   // shared = down(silu(gate) * up): the SwiGLU into FP16 (x16 is free again)
                    prefill::swiglu_pair(P.sh_gate, P.sh_up, P.x16, T, cs, FS);
                    proj(b.sh_down, P.shared, T);
                }
                k::native_moe_combine_multi(P.parts, P.w, has_shared_ ? P.shared : nullptr, P.moe_out, N, K, (int) T,
                                            cs, has_shared_ ? P.sh_g : nullptr);
                k::add_inplace(P.x, P.moe_out, T * N, cs);
                mark(3, t_mark);
            }
            if (!mtp_chunk) last_logits();
            if (!routes.empty()) {
                // each window's counts as the windows would leave them at the prompt's end: times the decay of every
                // round that would come after it; the rounds themselves are left to the decode (their swaps would
                // cost the prompt more than they save)
                const int64_t W = opt_.max_t, last = (n - 1) / W;
                for (int64_t t0 = 0; t0 < T; t0 += W) {
                    const int64_t win = (c0 + t0) / W;
                    // the adaptive rounds after this window: whole rounds, so integer divisions
                    const int64_t rounds = (windows_ + last + 1) / opt_.adapt_every - (windows_ + win) / opt_.adapt_every;
                    const float wgt = std::pow(opt_.adapt_decay, (float) rounds);
                    for (int64_t ml = 0; ml < L; ++ml)
                        for (int64_t i = t0 * K; i < std::min(T, t0 + W) * K; ++i)
                            disp_.usage[(size_t) (ml * E + routes[(size_t) (ml * P.T * K + i)])] += wgt;
                }
            }
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs)) {
        err = std::string("plain prompt: ") + strata::gpu::last_error();
        return false;
    }
    windows_ += (n + opt_.max_t - 1) / opt_.max_t;
    mtp_h_set_ = has_mtp_ && pos + n < opt_.max_cells;   // the trunk's last row is in mtp_h_ for spec_start
    if (profile)
        std::fprintf(stderr, "strata plain: prompt of %lld tokens: GDN %.1f ms, attention %.1f, router and grouping %.1f, "
                             "experts %.1f, shared and combine %.1f\n", (long long) n, stage_ms[0], stage_ms[4], stage_ms[1],
                     stage_ms[2], stage_ms[3]);
    if (profile && stage_ms[0] > 0.0)
        std::fprintf(stderr, "strata plain: the GDN layers: in-projections %.1f ms, alpha and beta %.1f, gates and conv %.1f, "
                             "recurrence %.1f, out-projection %.1f\n", stage_ms[5], stage_ms[6], stage_ms[7], stage_ms[8],
                     stage_ms[9]);
    if (profile)
        std::fprintf(stderr, "strata plain: prompt experts not in VRAM %lld, of them on the CPU %lld (%lld rows)\n",
                     (long long) miss_n, (long long) cpu_n, (long long) cpu_rows);
    *out = ((volatile int32_t*) h_out_)[0];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) n_vocab_ * 4)) {
        err = "plain prompt: reading the logits failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
