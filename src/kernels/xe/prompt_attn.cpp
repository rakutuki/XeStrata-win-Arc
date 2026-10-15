// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {

constexpr int NT = 256;   // work-items a work-group

// the device's own conversion (f16_bits.hpp's exact one branches and loops: in the inner loops it took most of the
// time); the pools hold no subnormals worth the difference
[[maybe_unused]] inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// BQ queries of one KV head: R = BQ x G rows (query, head), Q in local memory as FP32 (scaled), the step's BK cells of
// K and V as FP16, the step's scores (then probabilities) per row.  Local memory is read 4 values at a time (Q and
// the probabilities as float4, K as 4 halves), the rows padded to keep those reads aligned and K's rows off one bank.
// Scores: work-item t takes cell t % BK of rows t / BK, + NT / BK, ...: neighbouring work-items read neighbouring K
// rows and the same Q row.  Output: work-item t owns column d = t % HD of rows t / HD, + NT / HD, ... in registers.
template <int HD, int G, int BQ, int BK>
struct Kernel {
    static constexpr int R = BQ * G, RS = NT / HD, OPT = R / RS, QS = HD + 4, KS = HD + 4;
    static constexpr int SG = NT / BK, SR = (R + SG - 1) / SG;   // the scores: row groups, rows a work-item
    static_assert(NT % HD == 0 && R % RS == 0 && HD % 8 == 0 && BK % 4 == 0, "the work splits evenly");
    using f4 = sycl::vec<float, 4>;
    using h4 = sycl::vec<uint16_t, 4>;
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    const uint16_t* kp;
    const uint16_t* vp;
    const int32_t* table;
    float* attn;
    int64_t pos0, T;
    int n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<float, 1> qs, sc, mrow, lrow, arow;
    sycl::local_accessor<uint16_t, 1> ks, vs;

    void operator()(sycl::nd_item<2> it) const {
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        const int d = t % HD, r0 = t / HD;
        float* Q = qs.get_multi_ptr<sycl::access::decorated::no>().get();
        float* S = sc.get_multi_ptr<sycl::access::decorated::no>().get();
        float* M = mrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* L = lrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* A = arow.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* V = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = t; i < R * HD; i += NT) {
            const int r = i / HD, dd = i - r * HD;
            const int64_t tq = qbase + r / G;
            Q[r * QS + dd] = tq < T ? q[(tq * n_head + (int64_t) h * G + r % G) * HD + dd] * scale : 0.0f;
        }
        if (t < R) {
            M[t] = NEG;
            L[t] = 0.0f;
        }
        float o[OPT];
        for (int k = 0; k < OPT; ++k) o[k] = 0.0f;
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        const int j = t % BK, rg = t / BK;
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K, V and probabilities are read
            for (int i = t; i < BK * HD / 8; i += NT) {   // 8 halves a work-item
                const int jj = i / (HD / 8), dd = (i - jj * (HD / 8)) * 8;
                const int64_t c = c0 + jj;
                h8 kv(0), vv(0);
                if (c < n_cells) {
                    const int64_t page = table[c / page_size];
                    const int64_t row = ((page * n_kv + h) * page_size + c % page_size) * HD + dd;
                    kv = *reinterpret_cast<const h8*>(kp + row);
                    vv = *reinterpret_cast<const h8*>(vp + row);
                }
                *reinterpret_cast<h4*>(K + (ptrdiff_t) jj * KS + dd) = kv.template swizzle<0, 1, 2, 3>();
                *reinterpret_cast<h4*>(K + (ptrdiff_t) jj * KS + dd + 4) = kv.template swizzle<4, 5, 6, 7>();
                *reinterpret_cast<h8*>(V + (ptrdiff_t) jj * HD + dd) = vv;
            }
            sycl::group_barrier(it.get_group());
            {   // scores, masked past each row's own position
                const int64_t c = c0 + j;
                float acc[SR];
                for (int k = 0; k < SR; ++k) acc[k] = 0.0f;
                for (int dd = 0; dd < HD; dd += 4) {
                    const h4 kh = *reinterpret_cast<const h4*>(K + (ptrdiff_t) j * KS + dd);
                    const float k0 = h2f(kh[0]), k1 = h2f(kh[1]), k2 = h2f(kh[2]), k3 = h2f(kh[3]);
                    for (int k = 0; k < SR; ++k) {
                        const int r = rg + k * SG;
                        if (SR * SG > R && r >= R) break;
                        const f4 qv = *reinterpret_cast<const f4*>(Q + (ptrdiff_t) r * QS + dd);
                        acc[k] = sycl::fma(qv[0], k0, sycl::fma(qv[1], k1, sycl::fma(qv[2], k2, sycl::fma(qv[3], k3, acc[k]))));
                    }
                }
                for (int k = 0; k < SR; ++k) {
                    const int r = rg + k * SG;
                    if (r >= R) break;
                    S[r * BK + j] = c <= pos0 + qbase + r / G && c < n_cells ? acc[k] : NEG;
                }
            }
            sycl::group_barrier(it.get_group());
            if (t < R) {   // the row's online softmax: its new maximum, the old sum's factor, the step's probabilities
                float mx = M[t];
                for (int jj = 0; jj < BK; ++jj) mx = sycl::fmax(mx, S[t * BK + jj]);
                const float a = M[t] == NEG ? 0.0f : sycl::exp(M[t] - mx);
                float sum = 0.0f;
                for (int jj = 0; jj < BK; ++jj) {
                    const float p = S[t * BK + jj] == NEG ? 0.0f : sycl::exp(S[t * BK + jj] - mx);
                    S[t * BK + jj] = p;
                    sum += p;
                }
                L[t] = L[t] * a + sum;
                M[t] = mx;
                A[t] = a;
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < OPT; ++k) o[k] *= A[r0 + k * RS];
            for (int jj = 0; jj < BK; jj += 4) {
                const float v0 = h2f(V[jj * HD + d]), v1 = h2f(V[(jj + 1) * HD + d]), v2 = h2f(V[(jj + 2) * HD + d]),
                            v3 = h2f(V[(jj + 3) * HD + d]);
                for (int k = 0; k < OPT; ++k) {
                    const f4 p = *reinterpret_cast<const f4*>(S + (ptrdiff_t) (r0 + k * RS) * BK + jj);
                    o[k] = sycl::fma(p[0], v0, sycl::fma(p[1], v1, sycl::fma(p[2], v2, sycl::fma(p[3], v3, o[k]))));
                }
            }
        }
        for (int k = 0; k < OPT; ++k) {
            const int r = r0 + k * RS;
            const int64_t tq = qbase + r / G;
            if (tq < T) attn[(tq * n_head + (int64_t) h * G + r % G) * HD + d] = o[k] / L[r];
        }
    }
};

// RDNA3 and RDNA4 (GFX11, GFX12): Q.K^T and P.V through the FP16 WMMA instructions (16 x 16 x 16, FP32
// accumulators; the layouts as iq_mmq.cpp's int8 ones: lane l holds row l % 16 of A and column l % 16 of B and of the
// accumulator; GFX12 8 of the 16 K values a lane (l / 16 picks which) and accumulator rows 8 (l / 16) + e, GFX11 all
// 16 and rows 2 e + l / 16).  As flash attention 2: each wave owns 16 rows (query, head) of the BQ queries' R = BQ x G,
// with their Q in registers (head_dim 128: FP16 high and low parts, two products, for FP32's precision), and keeps
// the scores, the online softmax (the rows' maxima and sums across the 16 lanes by shuffles) and the output in its
// accumulators; the waves share only the step's 32 cells of K and V (transposed) in local memory.  The probabilities
// go through a wave's own local rows to become the A operand of P.V.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && (defined(__GFX11__) || defined(__GFX12__))
#define STRATA_PA_WMMA 1
#else
#define STRATA_PA_WMMA 0
#endif

template <int HD, int G, int BQ>
struct WmmaKernel {
    static constexpr int BK = 32, R = BQ * G, W = R / 16, DT = HD / 16, CT = BK / 16;
    static constexpr int KS = HD + 8, VS = BK + 8, PS = BK + 8;   // local row strides (halves), 16-byte multiples
    static constexpr bool LO = HD <= 128;                         // Q's low part (registers permitting)
    static_assert(R % 16 == 0 && HD % 16 == 0, "whole tiles");
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    const uint16_t* kp;
    const uint16_t* vp;
    const int32_t* table;
    float* attn;
    int64_t pos0, T;
    int n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<uint16_t, 1> ks, vs, ps;

    void operator()(sycl::nd_item<2> it) const {
#if STRATA_PA_WMMA
        using fv8 = float __attribute__((ext_vector_type(8)));
#if defined(__GFX12__)
        using frag = _Float16 __attribute__((ext_vector_type(8)));
        constexpr int KOFF = 8, KN = 8;   // a lane's K values: 8 of the 16, the half l / 16 picks
        auto row_of = [](int hi, int e) { return 8 * hi + e; };
#else
        using frag = _Float16 __attribute__((ext_vector_type(16)));
        constexpr int KOFF = 0, KN = 16;
        auto row_of = [](int hi, int e) { return 2 * e + hi; };
#endif
        auto mma = [](frag a, frag b, fv8 c) {
#if defined(__GFX12__)
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
        };
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const auto sg = it.get_sub_group();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int wave = t / 32, lane = t % 32, lo = lane & 15, hi = lane >> 4;
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* Vt = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* P = ps.get_multi_ptr<sycl::access::decorated::no>().get() + wave * 16 * PS;
        auto frag_at = [](const uint16_t* p) { return *reinterpret_cast<const frag*>(p); };
        // the wave's Q rows as A fragments: row lo of its tile, K values k0 + hi KOFF .. + KN
        frag qh[DT], ql[LO ? DT : 1];
        {
            const int r = wave * 16 + lo;
            const int64_t tq = qbase + r / G;
            const float* src = q + (tq * n_head + (int64_t) h * G + r % G) * HD;
            for (int k = 0; k < DT; ++k)
                for (int e = 0; e < KN; ++e) {
                    const float v = tq < T ? src[k * 16 + hi * KOFF + e] * scale : 0.0f;
                    const _Float16 vh = (_Float16) v;
                    qh[k][e] = vh;
                    if constexpr (LO) ql[k][e] = (_Float16) (v - (float) vh);
                }
        }
        float m[8], l[8];
        fv8 o[DT];
        for (int e = 0; e < 8; ++e) {
            m[e] = NEG;
            l[e] = 0.0f;
        }
        for (int k = 0; k < DT; ++k) o[k] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        const int nt = W * 32;
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K and V are read
            for (int i = t; i < BK * HD / 8; i += nt) {   // K as [cell][dim], V transposed as [dim][cell]
                const int j = i / (HD / 8), dd = (i - j * (HD / 8)) * 8;
                const int64_t c = c0 + j;
                h8 kv(0), vv(0);
                if (c < n_cells) {
                    const int64_t row = ((int64_t) table[c / page_size] * n_kv + h) * page_size + c % page_size;
                    kv = *reinterpret_cast<const h8*>(kp + row * HD + dd);
                    vv = *reinterpret_cast<const h8*>(vp + row * HD + dd);
                }
                *reinterpret_cast<h8*>(K + j * KS + dd) = kv;
                for (int e = 0; e < 8; ++e) Vt[(dd + e) * VS + j] = vv[e];
            }
            sycl::group_barrier(it.get_group());
            fv8 sc[CT];
            for (int ct = 0; ct < CT; ++ct) {   // S = Q . K^T for the wave's rows
                fv8 acc = fv8{0, 0, 0, 0, 0, 0, 0, 0};
                for (int k = 0; k < DT; ++k) {
                    const frag b = frag_at(K + (ct * 16 + lo) * KS + k * 16 + hi * KOFF);
                    acc = mma(qh[k], b, acc);
                    if constexpr (LO) acc = mma(ql[k], b, acc);
                }
                sc[ct] = acc;
            }
            float alpha[8];
            for (int e = 0; e < 8; ++e) {   // row 16 wave + row_of(hi, e): its 16 lanes hold its BK columns
                const int64_t qpos = pos0 + qbase + (wave * 16 + row_of(hi, e)) / G;
                float mx = m[e];
                for (int ct = 0; ct < CT; ++ct) {
                    const int64_t c = c0 + ct * 16 + lo;
                    if (c > qpos || c >= n_cells) sc[ct][e] = NEG;
                    mx = sycl::fmax(mx, sc[ct][e]);
                }
                for (int x = 1; x < 16; x <<= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, x));
                alpha[e] = m[e] == NEG ? 0.0f : sycl::exp(m[e] - mx);
                float sum = 0.0f;
                for (int ct = 0; ct < CT; ++ct) {
                    const _Float16 p = (_Float16) (sc[ct][e] == NEG ? 0.0f : sycl::exp(sc[ct][e] - mx));
                    P[row_of(hi, e) * PS + ct * 16 + lo] = sycl::bit_cast<uint16_t>(p);
                    sum += (float) p;
                }
                for (int x = 1; x < 16; x <<= 1) sum += sycl::permute_group_by_xor(sg, sum, x);
                l[e] = l[e] * alpha[e] + sum;
                m[e] = mx;
            }
            sycl::group_barrier(sg);   // the wave's probabilities are written
            for (int k = 0; k < DT; ++k) {   // O += P . V
                for (int e = 0; e < 8; ++e) o[k][e] *= alpha[e];
                for (int k0 = 0; k0 < BK; k0 += 16)
                    o[k] = mma(frag_at(P + lo * PS + k0 + hi * KOFF), frag_at(Vt + (k * 16 + lo) * VS + k0 + hi * KOFF), o[k]);
            }
        }
        for (int k = 0; k < DT; ++k)
            for (int e = 0; e < 8; ++e) {
                const int r = wave * 16 + row_of(hi, e);
                const int64_t tq = qbase + r / G;
                if (tq < T) attn[(tq * n_head + (int64_t) h * G + r % G) * HD + k * 16 + lo] = o[k][e] / l[e];
            }
#else
        (void) it;
#endif
    }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::sub_group_size<STRATA_SUB_GROUP(32)>};
    }
};

template <int HD, int G, int BQ>
void launch_wmma(sycl::queue& q, const float* qv, const uint16_t* kp, const uint16_t* vp, const int32_t* table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn) {
    using KT = WmmaKernel<HD, G, BQ>;
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ), nt = (size_t) KT::W * 32;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>((size_t) KT::BK * KT::KS), h),
            vs(sycl::range<1>((size_t) HD * KT::VS), h), ps(sycl::range<1>((size_t) KT::W * 16 * KT::PS), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * nt), sycl::range<2>(1, nt)),
                       KT{qv, kp, vp, table, attn, pos0, T, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), ks, vs, ps});
    });
}

template <int HD, int G, int BQ, int BK>
void launch(sycl::queue& q, const float* qv, const uint16_t* kp, const uint16_t* vp, const int32_t* table, int64_t pos0,
            int64_t T, const QsaShapes& s, float* attn) {
    using KT = Kernel<HD, G, BQ, BK>;
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) KT::R * KT::QS), h), sc(sycl::range<1>((size_t) KT::R * BK), h),
            mrow(sycl::range<1>(KT::R), h), lrow(sycl::range<1>(KT::R), h), arow(sycl::range<1>(KT::R), h);
        sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>((size_t) BK * KT::KS), h), vs(sycl::range<1>((size_t) BK * HD), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * NT), sycl::range<2>(1, NT)),
                       KT{qv, kp, vp, table, attn, pos0, T, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), qs, sc, mrow, lrow, arow, ks, vs});
    });
}

}  // namespace

bool prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, void* stream) {
    if (T <= 0) return true;
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0) return false;
    const int64_t g = s.n_head / s.n_head_kv;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    static const bool no_mma = [] {   // STRATA_PROMPT_ATTN_MMA=0: the FMA kernel on every GPU
        const char* v = std::getenv("STRATA_PROMPT_ATTN_MMA");
        return v != nullptr && std::strtol(v, nullptr, 10) == 0;
    }();
    bool mma = false;
    if (!no_mma && xe::amd_wmma(queue.get_device())) {
        // a wave per 16 rows: 5 for llm-jp's 8 queries of 10 heads, 4 for 8 queries of 8 (local memory 25 to 43 KiB)
        mma = true;
        if (s.head_dim == 128 && g == 10) launch_wmma<128, 10, 8>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 8) launch_wmma<256, 8, 8>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 12) launch_wmma<256, 12, 4>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else mma = false;
    }
    // local memory: BQ x G rows of FP32 Q and BK cells of K and V, 37 to 42 KiB: within the 48 KiB every GPU gives
    if (!mma) {
        if (s.head_dim == 128 && g == 10) launch<128, 10, 4, 32>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 8) launch<256, 8, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 12) launch<256, 12, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else return false;
    }
    if (stream == nullptr) queue.wait_and_throw();
    return true;
}

}  // namespace strata::kernels
