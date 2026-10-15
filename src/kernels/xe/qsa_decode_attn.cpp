// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/qsa_decode_attn.cpp - the Xe port of Strata's src/kernels/cuda/qsa_decode_attn.cu (see the header): split-K
// decode attention over the selected cells for fp16, INT8 and Q4_0 KV, then a merge of the chunk partials.
//
// The head size HD and the query heads per KV head G are template parameters: (256, 12) is Qwen3.8's QSA, (256, 8)
// Ornith-1.5's and (128, 10) llm-jp-4.1's full attention, which read EVERY cell (`qsa_dense_attn_*`: no selection,
// cell c of the cache for slot c).  A work-group has HD work-items, one per dimension in the value loop; a lane holds
// HD/32 dimensions in the score loop.  For (256, 12) every expression is the one this file had with HD and G fixed.
//
// A warp is a sub-group of 32.  CUDA used the __expf intrinsic here; this uses the precise exp.  Explicit fmaf calls
// stay fused (sycl::fma), and the library's -ffp-contract=off leaves every other multiply-add as written.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_qsa_decode_attn
#include "strata/kernels/device_arch_tag.hpp"

#include <cfloat>
#include <cmath>
#include <string>

namespace strata::kernels {
namespace {

constexpr int CHUNK = 64;        // cells per work-group
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}
inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// The G <= 16 heads' lane sums (`part[G..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5 (upstream fd95405).  Lane l ends holding head head_of_lane(l); lanes l and l^1
// agree.
inline float reduce12(const sycl::sub_group& sg, const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + sycl::permute_group_by_xor(sg, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + sycl::permute_group_by_xor(sg, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + sycl::permute_group_by_xor(sg, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + sycl::permute_group_by_xor(sg, hi2 ? r2[0] : r2[1], 2);
    return r1 + sycl::permute_group_by_xor(sg, r1, 1);
}
inline int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own work-item), per KV format.
template <int HD, int KV_MODE>
inline float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return h2f(p.v_pool[row * HD + d]);
    } else if constexpr (KV_MODE == 1) {
        const float sc = h2f(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]);
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = h2f(blk->d);
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

// N = HD / 32 consecutive values of one cell's key or value row, dimensions [d0, d0+N).  KV_MODE 0: FP16, 1: INT8,
// 2: rotated Q4_0, 3: the hybrid K8V4 (K as INT8, V as rotated Q4_0; upstream 2aa8f72).  N divides 16, so the run
// stays inside one Q4_0 half (the low nibbles hold elements 0-15 of a block, the high ones 16-31).
template <int HD, int KV_MODE>
inline void loadn(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int N = HD / WARP;
    if constexpr (KV_MODE == 3) {
        if (value) loadn<HD, 2>(p, true, row, d0, out);
        else loadn<HD, 1>(p, false, row, d0, out);
    } else if constexpr (KV_MODE == 0) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        for (int j = 0; j < N; ++j) out[j] = h2f(base[j]);
    } else if constexpr (KV_MODE == 1) {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const float sc = h2f((value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
        for (int j = 0; j < N; ++j) out[j] = (float) codes[j] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d0 / QK4_0;
        const int rem = d0 % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
        const float d = h2f(blk->d);
        const uint8_t* bytes = blk->qs + (rem & 15);
        if (rem < 16) {
            for (int k = 0; k < N; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
            for (int k = 0; k < N; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}

// One lane's share of q.k: the N products summed left to right, which for N = 8 is the expression this file wrote out.
template <int N>
inline float dotn(const float* k, const float* q) {
    float s = k[0] * q[0];
#pragma unroll
    for (int e = 1; e < N; ++e) s = s + k[e] * q[e];
    return s;
}

template <int HD, int G, int KV_MODE>
sycl::event launch_chunk(sycl::queue& queue, const float* q_all, const QsaAttnPools p, const int32_t* ids_all,
                         const int32_t* step_all, int n_kv_heads, int page_size, float scale, float* part_acc_all,
                         float* part_m_all, float* part_l_all, int n_chunks, size_t n_q, int cap,
                         long long scratch_stride) {
    static_assert(G >= 1 && G <= 16, "reduce12 sums at most 16 heads");
    static_assert(HD % 128 == 0 && HD <= 256, "a lane holds HD/32 dimensions inside one Q4_0 half");
    constexpr int THREADS = HD;           // one work-item per dimension in the value loop
    constexpr int WARPS = THREADS / WARP;
    constexpr int N = HD / WARP;          // dimensions a lane holds in the score loop
    constexpr int GP = (G + 3) / 4 * 4;   // a cell's weights padded to whole float4s
    // ids_all null: every cell, slot c of the chunk is cell c0 + c, and the count is the cache's (step[kStepNKv])
    const bool dense = ids_all == nullptr;
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(G * HD), hd);          // this KV head's query heads
        sycl::local_accessor<float, 1> sp(sycl::range<1>((size_t) G * CHUNK), hd);       // scores
        sycl::local_accessor<sycl::float4, 1> spt(sycl::range<1>(CHUNK * GP / 4), hd);  // probabilities, cell-major
        sycl::local_accessor<long long, 1> srow(sycl::range<1>(CHUNK), hd);     // pool row of each cell
        hd.parallel_for(sycl::nd_range<3>({n_q, (size_t) n_kv_heads, (size_t) n_chunks * THREADS}, {1, 1, THREADS}),
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const size_t z = it.get_group(0);
            // batched form: query z, with its own q row, selection, step and scratch
            const float* q = q_all + z * (size_t) (n_kv_heads * G) * HD;
            const int32_t* ids = dense ? nullptr : ids_all + z * (size_t) cap;
            const int32_t* step = step_all + z * kStepCount;
            float* part_acc = part_acc_all + z * (size_t) scratch_stride;
            float* part_m = part_m_all + z * (size_t) scratch_stride;
            float* part_l = part_l_all + z * (size_t) scratch_stride;
            const int n_ids = dense ? step[kStepNKv] : step[kStepWidth];
            const int chunk = (int) it.get_group(2), kvh = (int) it.get_group(1);
            const int t = (int) it.get_local_id(2);
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int c0 = chunk * CHUNK;
            const int n_here = sycl::min(CHUNK, n_ids - c0);
            const int slot = kvh * n_chunks + chunk;
            if (n_here <= 0) {                                    // uniform across the group
                if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
                return;
            }
            for (int i = t; i < G * HD; i += THREADS) sq[i] = q[(size_t) (kvh * G) * HD + i];
            if (t < CHUNK) {
                long long r = -1;
                if (t < n_here) {
                    const int cell = dense ? c0 + t : ids[c0 + t];
                    const long long page = (long long) p.page_table[cell / page_size];
                    // a block the KV streaming could not make resident keeps page -1; its cells are masked (score
                    // -FLT_MAX, weight 0) instead of being read from before the pool (upstream f3925cf)
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                }
                srow[t] = r;
            }
            sycl::group_barrier(grp);
            // scores: each sub-group takes cells warp, warp+WARPS, ...; each lane holds N of the HD dimensions.  NC
            // cells per step (warp + WARPS * (i + j)): a lane's q slice comes out of local memory once for all of them
            // (upstream fd95405); every dot product is the same expression as for a single cell, so the scores are
            // unchanged.
            constexpr int NC = 2;
            const int hd_lane = head_of_lane(lane);
            for (int i = 0; i < CHUNK / WARPS; i += NC) {
                int cc[NC];
                bool all_ok = true;
                for (int j = 0; j < NC; ++j) {
                    cc[j] = warp + WARPS * (i + j);
                    all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
                }
                if (all_ok) {
                    float kk[NC][N];
                    for (int j = 0; j < NC; ++j) loadn<HD, KV_MODE>(p, false, srow[cc[j]], lane * N, kk[j]);
                    float pp[NC][16];
#pragma unroll
                    for (int h = 0; h < G; ++h) {
                        const float* qa = &sq[h * HD + lane * N];
#pragma unroll
                        for (int j = 0; j < NC; ++j) pp[j][h] = dotn<N>(kk[j], qa);
                    }
#pragma unroll
                    for (int j = 0; j < NC; ++j) {
                        for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                        const float sj = reduce12(sg, pp[j], lane);
                        if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + cc[j]] = sj * scale;
                    }
                } else {
                    for (int j = 0; j < NC; ++j) {   // a masked or out-of-range cell in the step: one cell at a time
                        const int c = cc[j];
                        if (c >= n_here || srow[c] < 0) {
                            if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                            continue;
                        }
                        float kn[N];
                        loadn<HD, KV_MODE>(p, false, srow[c], lane * N, kn);
                        float part[16];
                        for (int h = 0; h < G; ++h) part[h] = dotn<N>(kn, &sq[h * HD + lane * N]);
                        for (int h = G; h < 16; ++h) part[h] = 0.0f;
                        const float sc = reduce12(sg, part, lane);
                        if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + c] = sc * scale;
                    }
                }
            }
            sycl::group_barrier(grp);
            // per-head chunk max and exp-sum: sub-group w handles heads w, w+WARPS, ...
            for (int h = warp; h < G; h += WARPS) {
                const float a = sp[h * CHUNK + lane], b = sp[h * CHUNK + lane + 32];
                const float m = warp_max(sg, sycl::fmax(a, b));
                const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::exp(a - m) : 0.0f;
                const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::exp(b - m) : 0.0f;
                float* pt = reinterpret_cast<float*>(&spt[0]);
                pt[lane * GP + h] = ea;
                pt[(lane + 32) * GP + h] = eb;
                const float l = warp_sum(sg, ea + eb);
                if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
            }
            sycl::group_barrier(grp);
            // values: work-item t owns dimension t for all G heads.
            float acc[GP];
            for (int h = 0; h < GP; ++h) acc[h] = 0.0f;
            // one cell's weights (GP/4 float4 loads from local memory) into the accumulators; the cells are folded in
            // ascending order, so the sums are the plain loop's, bit for bit (upstream fd95405).  The padding slots
            // past G accumulate whatever local memory holds and are never stored.
            const auto fold = [&](int c, float v) {
#pragma unroll
                for (int g4 = 0; g4 < GP / 4; ++g4) {
                    const sycl::float4 w = spt[(size_t) c * (GP / 4) + g4];
                    acc[4 * g4] = sycl::fma(w.x(), v, acc[4 * g4]);
                    acc[4 * g4 + 1] = sycl::fma(w.y(), v, acc[4 * g4 + 1]);
                    acc[4 * g4 + 2] = sycl::fma(w.z(), v, acc[4 * g4 + 2]);
                    acc[4 * g4 + 3] = sycl::fma(w.w(), v, acc[4 * g4 + 3]);
                }
            };
            int c = 0;
            // four cells at a time with their V loads issued together; a group holding a masked cell (page -1,
            // rare) leaves this loop and the one below finishes the chunk
            for (; c + 4 <= n_here; c += 4) {
                const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
                if ((r0 | r1 | r2 | r3) < 0) break;
                const float v0 = load_v1<HD, KV_MODE>(p, r0, t), v1 = load_v1<HD, KV_MODE>(p, r1, t);
                const float v2 = load_v1<HD, KV_MODE>(p, r2, t), v3 = load_v1<HD, KV_MODE>(p, r3, t);
                fold(c, v0);
                fold(c + 1, v1);
                fold(c + 2, v2);
                fold(c + 3, v3);
            }
            for (; c < n_here; ++c) {
                if (srow[c] < 0) continue;   // masked above, weight 0
                fold(c, load_v1<HD, KV_MODE>(p, srow[c], t));
            }
            for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        });
    });
}

template <int HD, int G>
sycl::event launch_merge(sycl::queue& queue, const float* part_acc_all, const float* part_m_all, const float* part_l_all,
                         int n_chunks, float* attn_all, size_t n_head, size_t n_q, long long scratch_stride) {
    return queue.parallel_for(sycl::nd_range<2>({n_q, n_head * HD}, {1, HD}), [=](sycl::nd_item<2> it) {
        const size_t y = it.get_group(0);
        const float* part_acc = part_acc_all + y * (size_t) scratch_stride;
        const float* part_m = part_m_all + y * (size_t) scratch_stride;
        const float* part_l = part_l_all + y * (size_t) scratch_stride;
        float* attn = attn_all + y * n_head * HD;
        const int h = (int) it.get_group(1);                 // global query head
        const int kvh = h / G, hl = h % G;
        const int d = (int) it.get_local_id(1);
        float M = -FLT_MAX;
        for (int c = 0; c < n_chunks; ++c) M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int slot = kvh * n_chunks + c;
            const float m = part_m[slot * G + hl];
            if (m == -FLT_MAX) continue;
            const float w = sycl::exp(m - M);
            L = sycl::fma(part_l[slot * G + hl], w, L);
            acc = sycl::fma(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
        }
        attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
    });
}

int kv_mode_of(const QsaAttnPools& pools) {
    return pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0));
}

template <int HD, int G>
void run_shape(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream) {
    const int kv_mode = kv_mode_of(pools);
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / std::sqrt((float) HD);
    auto& queue = queue_for(stream);
    const int nkv = (int) s.n_head_kv, ps = (int) s.page_size;
    if (kv_mode == 3) launch_chunk<HD, G, 3>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else if (kv_mode == 2) launch_chunk<HD, G, 2>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else if (kv_mode == 1) launch_chunk<HD, G, 1>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else launch_chunk<HD, G, 0>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    const auto e = launch_merge<HD, G>(queue, part_acc, part_m, part_l, n_chunks, attn, (size_t) s.n_head, n_q, stride);
    if (!stream) core::Runtime::get().wait(e, "qsa_decode_attn");
}

// The shapes this file is built for: (head_dim, query heads per KV head).  A model with another shape needs its line
// here; the kernels' local arrays are sized by G and HD.
bool shape_ok(const QsaShapes& s) {
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv) return false;
    const int64_t g = s.n_head / s.n_head_kv;
    return (s.head_dim == 256 && (g == 12 || g == 8)) || (s.head_dim == 128 && g == 10);
}

void run(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
         const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream) {
    const int64_t g = s.n_head / s.n_head_kv;
    if (s.head_dim == 256 && g == 12) run_shape<256, 12>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream);
    else if (s.head_dim == 256 && g == 8) run_shape<256, 8>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream);
    else run_shape<128, 10>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream);
}

bool pools_ok(const QsaAttnPools& pools) {
    const int kv_mode = kv_mode_of(pools);
    return kv_mode == 3 ? pools.k_scale != nullptr
         : kv_mode == 2 ? pools.v_q4 != nullptr
         : kv_mode == 1 ? (pools.v_q && pools.k_scale && pools.v_scale) : (pools.k_pool && pools.v_pool);
}

std::string shape_name(const QsaShapes& s) {
    return std::to_string(s.n_head) + "/" + std::to_string(s.n_head_kv) + " heads of " + std::to_string(s.head_dim);
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !steps || !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_decode_attn_batch: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    run(q, pools, ids, steps, cap, s, scratch, attn, (size_t) n_q, (long long) qsa_decode_attn_scratch_floats(cap, s), stream);
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (uint64_t) (s.head_dim + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !step || !pools.page_table)
        throw core::DeviceError("qsa_decode_attn: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_decode_attn: incomplete KV pools");
    run(q, pools, ids, step, cap, s, scratch, attn, 1, 0, stream);
}

void qsa_dense_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* step, int64_t max_cells,
                         const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (!shape_ok(s) || max_cells <= 0 || !scratch || !step || !pools.page_table)
        throw core::DeviceError("qsa_dense_attn: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_dense_attn: incomplete KV pools");
    run(q, pools, nullptr, step, max_cells, s, scratch, attn, 1, 0, stream);
}

void qsa_dense_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* steps, int64_t max_cells,
                          const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || max_cells <= 0 || !scratch || !steps || !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_dense_attn_batch: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_dense_attn_batch: incomplete KV pools");
    run(q, pools, nullptr, steps, max_cells, s, scratch, attn, (size_t) n_q,
        (long long) qsa_decode_attn_scratch_floats(max_cells, s), stream);
}

}  // namespace strata::kernels
