// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/artifact/model_spec_test.cpp - read_model_spec on synthetic GGUFs (no model, no GPU).
//
//   1. qwen35moe (Ornith-1.5's shape): the trunk excludes the MTP layer, every 4th layer is full attention, the
//      shared expert counts as one when only its width is written, the gating defaults to softmax;
//   2. glm5next (Project Maya's spelling of glm5-next, GLM-5.3-Flash's shape): a layer without K/V heads is
//      recurrent, the three dense layers do not route and the MoE index starts after them, the per-layer clamp;
//   3. qwen3moe (llm-jp-4.1's shape): nothing recurrent, no shared expert;
//   4. refusals, each naming what is wrong: an unknown architecture, a missing required key, an mHC of 3 streams.
//
// The fixtures carry only the metadata the reader looks at, and one F32[8] tensor so the file has a data section.
#include "strata/artifact/model_spec.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// One metadata entry, already encoded after its key: the value type, then the value.
struct Kv {
    std::string key;
    std::vector<uint8_t> val;
};
Kv u32(const std::string& k, uint32_t v) {
    Kv e{k, {}};
    put<uint32_t>(e.val, 4);   // U32
    put<uint32_t>(e.val, v);
    return e;
}
Kv f32(const std::string& k, float v) {
    Kv e{k, {}};
    put<uint32_t>(e.val, 6);   // F32
    put<float>(e.val, v);
    return e;
}
Kv boolean(const std::string& k, bool v) {
    Kv e{k, {}};
    put<uint32_t>(e.val, 7);   // BOOL
    put<uint8_t>(e.val, v ? 1 : 0);
    return e;
}
Kv str(const std::string& k, const std::string& v) {
    Kv e{k, {}};
    put<uint32_t>(e.val, 8);   // STRING
    put_str(e.val, v);
    return e;
}
Kv u32s(const std::string& k, const std::vector<uint32_t>& v) {
    Kv e{k, {}};
    put<uint32_t>(e.val, 9);   // ARRAY
    put<uint32_t>(e.val, 4);   // of U32
    put<uint64_t>(e.val, v.size());
    for (uint32_t x : v) put<uint32_t>(e.val, x);
    return e;
}
Kv f32s(const std::string& k, const std::vector<float>& v) {
    Kv e{k, {}};
    put<uint32_t>(e.val, 9);   // ARRAY
    put<uint32_t>(e.val, 6);   // of F32
    put<uint64_t>(e.val, v.size());
    for (float x : v) put<float>(e.val, x);
    return e;
}

// A GGUF v3 file with the metadata `kvs` and one F32[8] tensor, 32-byte aligned.
std::filesystem::path write_gguf(const std::vector<Kv>& kvs) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);             // n_tensors
    put<uint64_t>(b, kvs.size());    // n_kv
    for (const Kv& e : kvs) {
        put_str(b, e.key);
        b.insert(b.end(), e.val.begin(), e.val.end());
    }
    put_str(b, "t");
    put<uint32_t>(b, 1);             // n_dims
    put<uint64_t>(b, 8);
    put<uint32_t>(b, 0);             // F32
    put<uint64_t>(b, 0);             // offset from data_start
    b.resize((b.size() + 31) / 32 * 32 + 32, 0);
    const auto path = std::filesystem::temp_directory_path() / "strata_model_spec_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

bool read(const std::vector<Kv>& kvs, strata::ModelSpec& s, std::string& err) {
    const auto path = write_gguf(kvs);
    bool ok;
    {
        strata::GgufFile g(path.string());
        ok = strata::read_model_spec(g, s, err);
    }
    std::filesystem::remove(path);
    return ok;
}

std::vector<Kv> qwen35moe() {
    const std::string a = "qwen35moe.";
    return {str("general.architecture", "qwen35moe"), u32(a + "block_count", 41), u32(a + "nextn_predict_layers", 1),
            u32(a + "embedding_length", 2048), u32(a + "context_length", 262144),
            u32(a + "attention.head_count", 16), u32(a + "attention.head_count_kv", 2),
            u32(a + "attention.key_length", 256), u32(a + "attention.value_length", 256),
            f32(a + "attention.layer_norm_rms_epsilon", 1e-6f), u32(a + "rope.dimension_count", 64),
            u32s(a + "rope.dimension_sections", {11, 11, 10, 0}), f32(a + "rope.freq_base", 1e7f),
            u32(a + "ssm.conv_kernel", 4), u32(a + "ssm.inner_size", 4096), u32(a + "ssm.state_size", 128),
            u32(a + "ssm.time_step_rank", 32), u32(a + "ssm.group_count", 16),
            u32(a + "full_attention_interval", 4), u32(a + "expert_count", 256), u32(a + "expert_used_count", 8),
            u32(a + "expert_feed_forward_length", 512), u32(a + "expert_shared_feed_forward_length", 512)};
}

std::vector<Kv> glm5next(uint32_t hc) {
    const std::string a = "glm5next.";
    std::vector<uint32_t> kv(46, 0);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = (i % 4 == 3 || i == 45) ? 1 : 0;
    return {str("general.architecture", "glm5next"), u32(a + "block_count", 46), u32(a + "nextn_predict_layers", 1),
            u32(a + "embedding_length", 4096), u32(a + "attention.head_count", 64), u32s(a + "attention.head_count_kv", kv),
            f32(a + "attention.layer_norm_rms_epsilon", 1e-5f), u32(a + "expert_count", 288),
            u32(a + "expert_used_count", 8), u32(a + "expert_feed_forward_length", 2048),
            u32(a + "expert_shared_feed_forward_length", 2048), u32(a + "expert_shared_count", 1),
            u32(a + "leading_dense_block_count", 3), u32(a + "feed_forward_length", 12288),
            u32(a + "expert_gating_func", 2), boolean(a + "expert_weights_norm", true),
            f32(a + "expert_weights_scale", 2.5f), f32s(a + "swiglu_clamp_exp", std::vector<float>(46, 10.0f)),
            u32(a + "hyper_connection.count", hc), u32(a + "hyper_connection.sinkhorn_iterations", 20),
            u32(a + "kda.head_dim", 128), f32(a + "kda.gate_lower_bound", -5.0f),
            u32(a + "attention.q_lora_rank", 1536), u32(a + "attention.kv_lora_rank", 512),
            u32(a + "attention.indexer.head_count", 32), u32(a + "attention.indexer.key_length", 128),
            u32(a + "attention.indexer.top_k", 2048), u32(a + "attention.indexer.kpool", 4),
            boolean(a + "attention.indexer.kpool_select_tail", true)};
}

std::vector<Kv> qwen3moe() {
    const std::string a = "qwen3moe.";
    return {str("general.architecture", "qwen3moe"), u32(a + "block_count", 32), u32(a + "embedding_length", 2560),
            u32(a + "attention.head_count", 40), u32(a + "attention.head_count_kv", 4),
            u32(a + "attention.key_length", 128), f32(a + "attention.layer_norm_rms_epsilon", 1e-6f),
            f32(a + "rope.freq_base", 5e5f), u32(a + "expert_count", 128), u32(a + "expert_used_count", 8),
            u32(a + "expert_feed_forward_length", 960)};
}
}  // namespace

static int run() {
    std::string err;

    std::printf("qwen35moe (Ornith-1.5)\n");
    {
        strata::ModelSpec s;
        const bool ok = read(qwen35moe(), s, err);
        check(ok, ("reads: " + err).c_str());
        check(s.arch == strata::ModelArch::Qwen35Moe && s.n_layer == 40 && s.n_nextn == 1, "40 trunk layers and 1 MTP");
        check(s.recurrent[0] && s.recurrent[2] && !s.recurrent[3] && !s.recurrent[39] && !s.recurrent[40],
              "every 4th layer full attention, the MTP layer not recurrent");
        check(s.n_head_kv[3] == 2 && s.rope_sections.size() == 4 && s.rope_sections[2] == 10, "K/V heads, MRoPE sections");
        check(s.n_expert_shared == 1 && s.gating == strata::ExpertGating::Softmax, "one shared expert, softmax gating");
        check(s.n_moe_layers() == 40 && s.moe_index(5) == 5 && s.moe_index(40) == -1, "every trunk layer routes");
    }

    std::printf("glm5next (GLM-5.3-Flash, Project Maya's spelling)\n");
    {
        strata::ModelSpec s;
        const bool ok = read(glm5next(4), s, err);
        check(ok, ("reads: " + err).c_str());
        check(s.arch == strata::ModelArch::Glm5Next && s.n_layer == 45 && s.n_nextn == 1, "45 trunk layers and 1 MTP");
        check(s.recurrent[0] && !s.recurrent[3] && s.recurrent[44] && !s.recurrent[45], "no K/V heads = KDA");
        check(!s.layer_routes(2) && s.moe_index(2) == -1 && s.moe_index(3) == 0 && s.moe_index(44) == 41,
              "the 3 dense layers do not route; the MoE index starts after them");
        check(s.n_moe_layers() == 42 && s.n_ff == 12288, "42 MoE layers, a 12288-wide dense FFN");
        check(s.gating == strata::ExpertGating::Sigmoid && s.expert_weights_norm && s.expert_weights_scale == 2.5,
              "sigmoid gating, normalised, x2.5");
        check(s.swiglu_clamp_exp.size() == 46 && s.swiglu_clamp_exp[10] == 10.0 && s.swiglu_clamp_shexp.empty(),
              "the per-layer clamp read, the absent one empty");
        check(s.hc == 4 && s.idx_kpool == 4 && s.idx_top_k == 2048 && s.idx_kpool_select_tail && s.kda_gate_lower_bound == -5.0,
              "mHC, indexer and KDA keys");
    }

    std::printf("qwen3moe (llm-jp-4.1)\n");
    {
        strata::ModelSpec s;
        const bool ok = read(qwen3moe(), s, err);
        check(ok, ("reads: " + err).c_str());
        bool any = false;
        for (bool r : s.recurrent) any = any || r;
        check(s.n_layer == 32 && !any && s.n_expert_shared == 0 && s.n_head_kv[0] == 4, "full attention only, no shared expert");
    }

    std::printf("refusals\n");
    {
        strata::ModelSpec s;
        auto kvs = qwen3moe();
        kvs[0] = str("general.architecture", "llama");
        check(!read(kvs, s, err) && err.find("'llama'") != std::string::npos, "an unknown architecture, named");
        kvs = qwen3moe();
        kvs.erase(kvs.begin() + 8);   // expert_count
        check(!read(kvs, s, err) && err.find("qwen3moe.expert_count") != std::string::npos, "a missing key, named");
        check(!read(glm5next(3), s, err) && err.find("hyper_connection.count") != std::string::npos, "an mHC of 3 streams");
    }

    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("model_spec_test: %s\n", e.what());
        return 1;
    }
}
