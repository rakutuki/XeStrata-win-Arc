// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool kq256_supported(int type) noexcept { return type == 10 || type == 11 || type == 12 || type == 13 || type == 14; }

size_t native_row_bytes(int type, int64_t n) {
    if (type < 0 || type >= GGML_TYPE_COUNT || n <= 0 || n % ggml_blck_size((ggml_type) type)) return 0;
    return ggml_row_size((ggml_type) type, n);
}

bool native_row_to_float(int type, const void* src, float* dst, int64_t n) {
    if (native_row_bytes(type, n) == 0) return false;
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) return false;
    t->to_float(src, dst, n);
    return true;
}

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err, int up_type) {
    init_once();
    if (up_type < 0) up_type = gu_type;
    const ggml_type_traits_cpu* tu = traits(up_type);
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (tu == nullptr || tu->vec_dot == nullptr || tu->vec_dot_type != tg->vec_dot_type) {
        err = "native experts: gate and up of types that read different activations";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_embd % ggml_blck_size((ggml_type) up_type) ||
        n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.up_type = up_type;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.up_row = ggml_row_size((ggml_type) up_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = f.up_off + f.up_row * (size_t) n_ff;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes || n_ff > kNativeMaxFF) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
}  // namespace

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

int native_gu_mt_min(int gu_type) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min = env ? (int) std::strtol(env, nullptr, 10) : 2;
    // A CPU with AVX-2 and without AVX-512 (whose kernel is iq256_gu_rows): there the multi-token kernel beats ggml's
    // dot for ONE token too, and in decode most CPU experts serve one token of the window.  An expert's gate/up
    // rows, one token, kernel against ggml's dot (i7-14700 P-core, iq_avx2_parity --bench --dispatch): IQ2_S 0.145 /
    // 0.185 ms, IQ3_S 0.162 / 0.298, IQ4_XS 0.099 / 0.115, IQ3_XXS 0.182 / 0.188, IQ2_XS 0.185 / 0.188; its E-core:
    // IQ2_S 0.285 / 0.390; without AVX-VNNI (STRATA_NO_AVXVNNI=1) IQ2_S 0.153 / 0.211.  IQ2_XXS was slower (0.154 /
    // 0.147) and keeps ggml's dot for one token.  The formats that take it no longer depend on the drafting.  A
    // machine-wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in
    // expert_layout.cpp and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  STRATA_IQ_MT_MIN set
    // keeps its rule.
    static const bool avx2_one = env == nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok();
    // The K-quants and Q8_0 (kq256_* / q80_256_*) share only the unpacking between tokens: one token keeps ggml-cpu's
    // dot until a measurement says otherwise.
    return avx2_one && iq256_supported(gu_type) && gu_type != 16 ? 1 : mt_min;   // 16: IQ2_XXS
}

namespace {
// The K-quants (Q4_K, Q5_K, Q6_K) and Q8_0 of the other models' experts on the AVX-2 multi-token kernels (iq_avx2.cpp,
// the AVX-VNNI copy where the CPU has it): every CPU with AVX2, AVX-512 ones too (ggml-cpu's dots for these are
// single-token there as well).  STRATA_NO_KQ256=1: ggml-cpu's dot for every token count.  Q8_0 only with AVX-VNNI: on
// a Ryzen 7 5700X (AVX2 alone, one thread, 2 to 4 tokens) its gate/up took 0.185 / 0.289 / 0.340 ms an expert of
// Ornith's shape against ggml-cpu's 0.170 / 0.235 / 0.309, and llm-jp's down 0.275 / 0.334 / 0.383 against 0.171 /
// 0.240 / 0.317; the K-quants were 1.3 to 1.6 times ggml-cpu's there.
bool kq_on(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_KQ256") == nullptr;
    static const bool q80 = on && cpu_avxvnni_ok();
    return type == (int) GGML_TYPE_Q8_0 ? q80 : on && kq256_supported(type);
}
// Q2_K and Q3_K (GSQ-RCO's experts) from four tokens on: ggml-cpu's own AVX2 dots for them are faster below that
// (one thread, Ornith's shape, ms an expert, the kernel against ggml-cpu - i7-14700: Q2_K 0.103 / 0.069 at one token,
// 0.135 / 0.121 at two, 0.210 / 0.230 at four; Ryzen 7 5700X: Q2_K 0.154 / 0.147 at two, 0.266 / 0.295 at four, Q3_K
// 0.229 / 0.195 at two, 0.345 / 0.390 at four).
int kq_mt_min(int type, int mt_min) {
    return type == (int) GGML_TYPE_Q2_K || type == (int) GGML_TYPE_Q3_K ? (mt_min > 4 ? mt_min : 4) : mt_min;
}
}  // namespace

namespace {
// out[t][r] = w_r . act[t] for rows [r0, r1) of one part of type `type` (the multi-token kernels where the type has one
// and the group is large enough, else ggml-cpu's dot)
void part_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
               int r0, int r1, int mt_min) {
    if (nt >= kq_mt_min(type, mt_min) && kq_on(type)) {
        if (type == (int) GGML_TYPE_Q8_0) q80_256_rows(w, row_bytes, n, act, nt, out, r0, r1);
        else kq256_rows(type, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(type)->vec_dot;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) dot(n, &out[t][r], 0, w + (size_t) r * row_bytes, 0, act[t], 0, 1);
}
}  // namespace

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    if (native_mixed_gu(f)) {   // gate and up of different types: each part on its own, then the SwiGLU
        thread_local float ub[MAXT][kNativeMaxFF];
        float* up[MAXT] = {};
        for (int t = 0; t < nt; ++t) up[t] = ub[t];
        const int mt = native_gu_mt_min(f.gu_type);
        part_rows(f.gu_type, blob, f.gu_row, (int) f.n_embd, act, nt, ff, r0, r1, mt);
        part_rows(f.up_type, blob + f.up_off, f.up_row, (int) f.n_embd, act, nt, up, r0, r1, mt);
        for (int t = 0; t < nt; ++t)
            for (int r = r0; r < r1; ++r) ff[t][r] = (ff[t][r] / (1.f + std::exp(-ff[t][r]))) * up[t][r];
        return;
    }
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    static const bool avx2 = std::getenv("STRATA_NO_IQ256") == nullptr;
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    const int mt_min = native_gu_mt_min(f.gu_type);   // #152
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    if (nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    if (nt >= kq_mt_min(f.gu_type, mt_min) && kq_on(f.gu_type)) {
        if (f.gu_type == (int) GGML_TYPE_Q8_0) q80_256_gu_rows(blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        else kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? (int) std::strtol(e, nullptr, 10) : 2; }();
    if (nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (nt >= kq_mt_min(f.d_type, mt_min) && kq_on(f.d_type)) {   // the K-quant and Q8_0 down rows: the same rule
        if (f.d_type == (int) GGML_TYPE_Q8_0) q80_256_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        else kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
