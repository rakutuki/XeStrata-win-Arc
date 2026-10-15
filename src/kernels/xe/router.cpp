// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/router.cpp - the Xe ports of Strata's src/kernels/cuda/router_top10.cu (the reference-faithful router) and
// Strata's src/kernels/cuda/native_router.cu (the pinned ggml topk-moe contract for 512 experts, top 10).
//
// native_router follows ggml-cuda's topk-moe at llama.cpp 3cf03257; MIT License, Copyright (c) 2023-2026 The ggml
// authors, see third_party/main/ggml/LICENSE.  CUDA compiled it with --use_fast_math; here exp is the precise function.
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_router
#include "strata/kernels/device_arch_tag.hpp"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;
constexpr int WARP = 32;

// The argmax order of both routers: the larger probability wins, a tie keeps the lower expert index.  It is a total
// order, so the xor butterflies below reach the same winner as CUDA's shuffle-down trees.
inline void better(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

std::atomic<bool> native_enabled{false};

bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

sycl::queue& queue_for(void* stream, bool required) {
    if (required && !stream) throw std::invalid_argument("the native router requires an explicit stream");
    return core::Runtime::get().stream(stream);
}

}  // namespace

namespace {
// D: double where the device has FP64 (the reference's exponentials and sums), float elsewhere (device_caps.hpp)
template <typename D>
sycl::event router_top10_launch(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k, int* ids,
                                float* weights, int threads, int nw) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned char, 1> s_taken(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<D, 1> s_ex(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_p(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_red(sycl::range<1>(RT_MAX_THREADS / WARP), h);
        sycl::local_accessor<int, 1> s_rid(sycl::range<1>(RT_MAX_THREADS / WARP), h);
        sycl::local_accessor<D, 1> s_sum(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * threads, threads),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto g = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0), nt = threads;
            const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const float* l = logits + (size_t) t * n_expert;

            // ---- softmax over ALL experts, for stability: the max (an exact, order-independent tree)
            float mx = -INFINITY;
            for (int x = tid; x < n_expert; x += nt) mx = sycl::fmax(mx, l[x]);
            mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
            if (lane == 0) s_red[warp] = mx;
            sycl::group_barrier(g);
            if (warp == 0) {
                float v = lane < nw ? s_red[lane] : -INFINITY;
                v = sycl::reduce_over_group(sg, v, sycl::maximum<float>());
                if (lane == 0) s_red[0] = v;
            }
            sycl::group_barrier(g);
            mx = s_red[0];

            // ---- the exponentials, once each, in parallel
            for (int x = tid; x < n_expert; x += nt) s_ex[x] = sycl::exp((D) l[x] - (D) mx);
            sycl::group_barrier(g);

            // ---- the sum, ascending, on one work-item
            if (tid == 0) {
                D sum = 0;
                for (int x = 0; x < n_expert; ++x) sum += s_ex[x];
                s_sum[0] = sum;
            }
            sycl::group_barrier(g);
            const float inv = (float) ((D) 1 / s_sum[0]);

            // ---- p[] once, the same expression the reference-faithful version evaluated
            for (int x = tid; x < n_expert; x += nt) { s_p[x] = (float) (s_ex[x] * inv); s_taken[x] = 0; }
            sycl::group_barrier(g);

            // ---- k passes of a stable argmax: ascending scan with strict `>` keeps the lowest index on a tie
            for (int i = 0; i < k; ++i) {
                float bv = -INFINITY;
                int bi = n_expert;               // a sentinel that loses to every real index
                for (int x = tid; x < n_expert; x += nt) {
                    if (s_taken[x]) continue;
                    const float pe = s_p[x];
                    if (pe > bv) { bv = pe; bi = x; }
                }
                for (int off = 16; off > 0; off >>= 1)
                    better(bv, bi, sycl::permute_group_by_xor(sg, bv, off), sycl::permute_group_by_xor(sg, bi, off));
                if (lane == 0) { s_red[warp] = bv; s_rid[warp] = bi; }
                sycl::group_barrier(g);
                if (warp == 0) {
                    float v = lane < nw ? s_red[lane] : -INFINITY;
                    int ix = lane < nw ? s_rid[lane] : n_expert;
                    for (int off = 16; off > 0; off >>= 1)
                        better(v, ix, sycl::permute_group_by_xor(sg, v, off), sycl::permute_group_by_xor(sg, ix, off));
                    if (lane == 0 && ix < n_expert) {
                        ids[(size_t) t * k + i] = ix;
                        weights[(size_t) t * k + i] = v;
                        s_taken[ix] = 1;
                    }
                }
                sycl::group_barrier(g);
            }

            // ---- renormalise, with ggml's lower clamp.  Order preserved.
            if (tid == 0) {
                D s = 0;
                for (int i = 0; i < k; ++i) s += (D) weights[(size_t) t * k + i];
                const D sc = sycl::fmax(s, (D) 6.103515625e-05);       // 2**-14
                for (int i = 0; i < k; ++i)
                    weights[(size_t) t * k + i] = (float) ((D) weights[(size_t) t * k + i] / sc);
            }
        });
    });
}
}  // namespace

// One work-group per token.  See the CUDA file for why each step is shaped as it is: the max is an exact tree, the
// exponentials are computed once in parallel in double, the double sum stays serial and ascending, and the top k
// are k passes of a stable argmax.
void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) throw core::DeviceError("router_top10: k " + std::to_string(k) + " exceeds the kernel's 64");
    if (n_expert > RT_MAX_THREADS * 64)
        throw core::DeviceError("router_top10: n_expert " + std::to_string(n_expert) + " is past the kernel's " +
                                std::to_string(RT_MAX_THREADS * 64));
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + WARP - 1) & ~(WARP - 1);                  // whole sub-groups, for the reductions
    const int nw = threads / WARP;
    auto& q = queue_for(stream, false);
    const auto e = xe::has_fp64(q)
                       ? router_top10_launch<double>(q, logits, n_tokens, n_expert, k, ids, weights, threads, nw)
                       : router_top10_launch<float>(q, logits, n_tokens, n_expert, k, ids, weights, threads, nw);
    if (!stream) core::Runtime::get().wait(e, "router_top10");
}

void native_router_set_enabled(bool value) { native_enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return native_enabled.load(std::memory_order_relaxed); }

namespace {
// One sub-group per token: the pinned kernel's 32 x 8 block has only its first warp active.
void launch_native_router(sycl::queue& q, const float* logits_all, int32_t* ids_all, float* weights_all, int n_tok) {
    q.parallel_for(sycl::nd_range<1>((size_t) n_tok * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t tok = it.get_group(0);
        const float* logits = logits_all + tok * 512;
        int32_t* ids = ids_all + tok * 10;
        float* weights = weights_all + tok * 10;
        const int lane = (int) sg.get_local_linear_id();
        float values[16];
        for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
        float maximum = -INFINITY;
        for (int i = 0; i < 16; ++i) maximum = sycl::max(maximum, values[i]);
        for (int mask = 16; mask; mask >>= 1) maximum = sycl::fmax(maximum, sycl::permute_group_by_xor(sg, maximum, mask));
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            values[i] = sycl::exp(values[i] - maximum);
            sum += values[i];
        }
        for (int mask = 16; mask; mask >>= 1) sum += sycl::permute_group_by_xor(sg, sum, mask);
        const float reciprocal = 1.0f / sum;
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
            for (int mask = 16; mask; mask >>= 1)
                better(best, expert, sycl::permute_group_by_xor(sg, best, mask), sycl::permute_group_by_xor(sg, expert, mask));
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[rank] = expert;
                // Deliberately accumulate by WINNING EXPERT lane, not output rank.
                // Multiple selected experts in one lane add in selection order.
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        for (int mask = 16; mask; mask >>= 1) selected_sum += sycl::permute_group_by_xor(sg, selected_sum, mask);
        selected_sum = sycl::max(selected_sum, 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    });
}
}  // namespace

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    launch_native_router(queue_for(stream, true), logits, ids, weights, 1);
}

// STRATA_ROUTE_PRIOR=lambda (upstream 2de9ae06, opt-in, changes the output): cache-conditioned routing, after "Mixture
// of Cache-Conditional Experts" (arXiv 2412.00099) without training.  When a token's 10 experts are picked, an expert
// already in VRAM gets a `lambda` bonus on its logit: the SELECTION uses the biased logits, the WEIGHTS stay the
// router's (softmax over the 512 unbiased logits) renormalised over the 10 picked.  One sub-group per token, 16 values
// a lane, as the native router.
void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream) {
    if (n_tok <= 0) return;
    if (!stream || res_layer == nullptr) throw std::invalid_argument("route-prior requires a stream and the residency row");
    queue_for(stream, true).parallel_for(sycl::nd_range<1>((size_t) n_tok * WARP, WARP), [=](sycl::nd_item<1> it)
                                         [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t tok = it.get_group(0);
        const float* lg = logits + tok * 512;
        int32_t* id = ids + tok * 10;
        float* w = weights + tok * 10;
        const int lane = (int) sg.get_local_linear_id();
        float p[16], s[16];
        float mx = -INFINITY;
        for (int i = 0; i < 16; ++i) { p[i] = lg[lane + 32 * i]; mx = sycl::fmax(mx, p[i]); }
        for (int m = 16; m; m >>= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, m));
        for (int i = 0; i < 16; ++i) {
            const int e = lane + 32 * i;
            s[i] = p[i] + (res_layer[e] >= 0 ? lambda : 0.0f);   // the selection score
            p[i] = sycl::exp(p[i] - mx);                         // the router's probability (unnormalised)
        }
        float sel_p = 0.0f, sel_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = -FLT_MAX, bp = 0.0f;
            int be = 1 << 30;
            for (int i = 0; i < 16; ++i)
                if (s[i] > best) { best = s[i]; be = lane + 32 * i; bp = p[i]; }
            for (int m = 16; m; m >>= 1) {
                const float ob = sycl::permute_group_by_xor(sg, best, m);
                const int oe = sycl::permute_group_by_xor(sg, be, m);
                const float op = sycl::permute_group_by_xor(sg, bp, m);
                if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
            }
            if ((be & 31) == lane) s[be >> 5] = -FLT_MAX;
            if (lane == rank) { id[rank] = be; sel_p = bp; }
            sel_sum += bp;                                       // the same on every lane
        }
        if (lane < 10) w[lane] = sel_p / sycl::fmax(sel_sum, 1e-30f);
    });
}

// STRATA_ROUTE_RESIDENT (upstream 5df35dcb, EXPERIMENTAL, opt-in, changes the output): residency-biased routing for
// 512 x 10.  After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not
// already picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           float margin, int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || res_layer == nullptr)
        throw std::invalid_argument("route-resident requires a stream, tokens and the residency row");
    const int lo = rank_lo, hi = rank_hi;
    // upstream 33fd01b4: a sub-group a token, lane j scanning experts j, j + 32, ...; the serial scan per token was the
    // window's own cost (upstream: 2.75 ms of a 22 ms window).  The same choice bit for bit: a strictly greater logit
    // wins, the smaller expert index breaks a tie, the ranks in order (a swap changes the set already picked).
    // STRATA_ROUTE_RESIDENT_SERIAL=1: the serial form, to compare.
    static const bool serial_env = [] {
        const char* v = std::getenv("STRATA_ROUTE_RESIDENT_SERIAL");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    if (!serial_env) {
        queue_for(stream, true).parallel_for(
            sycl::nd_range<1>((size_t) n_tok * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id();
                const size_t t = it.get_group_linear_id();
                const float* l = logits + t * 512;
                int32_t* idp = ids + t * 10;
                int32_t id[10];
                for (int r = 0; r < 10; ++r) id[r] = idp[r];
                int swaps = 0, tail = 0, before = 0;
                for (int r = 0; r < 10; ++r) before += res_layer[id[r]] < 0;
                for (int r = lo; r <= hi && r < 10; ++r) {
                    const int e = id[r];
                    if (res_layer[e] >= 0) continue;
                    ++tail;
                    int best = -1;
                    float bl = -INFINITY;
                    for (int f = lane; f < 512; f += WARP) {   // ascending: the lane keeps its first maximum
                        if (res_layer[f] < 0 || l[f] <= bl) continue;
                        bool used = false;
                        for (int q = 0; q < 10; ++q) used |= id[q] == f;
                        if (!used) { best = f; bl = l[f]; }
                    }
                    for (int o = WARP / 2; o > 0; o >>= 1) {   // the greatest logit, the smallest index on a tie
                        const float ob = sycl::permute_group_by_xor(sg, bl, o);
                        const int oi = sycl::permute_group_by_xor(sg, best, o);
                        if (oi >= 0 && (best < 0 || ob > bl || (ob == bl && oi < best))) { bl = ob; best = oi; }
                    }
                    if (best >= 0 && l[e] - bl <= margin) { id[r] = best; ++swaps; }
                }
                if (lane != 0) return;
                int after = 0;
                for (int r = 0; r < 10; ++r) after += res_layer[id[r]] < 0;
                if (swaps) {
                    for (int r = 0; r < 10; ++r) idp[r] = id[r];
                    float m = -INFINITY;
                    for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[id[r]]);
                    float ex[10], sum = 0.0f;
                    for (int r = 0; r < 10; ++r) { ex[r] = sycl::exp(l[id[r]] - m); sum += ex[r]; }
                    for (int r = 0; r < 10; ++r) weights[t * 10 + r] = ex[r] / sum;
                }
                if (stats) {
                    using A = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                                               sycl::memory_scope::device, sycl::access::address_space::global_space>;
                    A(stats[0]).fetch_add((unsigned long long) tail);
                    A(stats[1]).fetch_add((unsigned long long) swaps);
                    A(stats[2]).fetch_add((unsigned long long) before);
                    A(stats[3]).fetch_add((unsigned long long) after);
                }
            });
        return;
    }
    queue_for(stream, true).parallel_for(sycl::range<1>((size_t) n_tok), [=](sycl::id<1> i) {
        const size_t t = i[0];
        const float* l = logits + t * 512;
        int32_t* id = ids + t * 10;
        int swaps = 0, tail = 0, before = 0;
        for (int r = 0; r < 10; ++r) before += res_layer[id[r]] < 0;
        for (int r = lo; r <= hi && r < 10; ++r) {
            const int e = id[r];
            if (res_layer[e] >= 0) continue;
            ++tail;
            int best = -1;
            float bl = -INFINITY;
            for (int f = 0; f < 512; ++f) {
                if (res_layer[f] < 0 || l[f] <= bl) continue;
                bool used = false;
                for (int q = 0; q < 10; ++q) used |= id[q] == f;
                if (!used) { best = f; bl = l[f]; }
            }
            if (best >= 0 && l[e] - bl <= margin) { id[r] = best; ++swaps; }
        }
        int after = 0;
        for (int r = 0; r < 10; ++r) after += res_layer[id[r]] < 0;
        if (swaps) {
            float m = -INFINITY;
            for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[id[r]]);
            float ex[10], sum = 0.0f;
            for (int r = 0; r < 10; ++r) { ex[r] = sycl::exp(l[id[r]] - m); sum += ex[r]; }
            for (int r = 0; r < 10; ++r) weights[t * 10 + r] = ex[r] / sum;
        }
        if (stats) {
            using A = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>;
            A(stats[0]).fetch_add((unsigned long long) tail);
            A(stats[1]).fetch_add((unsigned long long) swaps);
            A(stats[2]).fetch_add((unsigned long long) before);
            A(stats[3]).fetch_add((unsigned long long) after);
        }
    });
}

void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    launch_native_router(queue_for(stream, true), logits, ids, weights, n_tok);
}

}  // namespace strata::kernels
