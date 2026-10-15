// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/qsa_dense_attn_parity.cpp - the decode attention over every cell (`qsa_dense_attn_step`) for the shapes it
// is built for, against an FP64 host reference (GPU, synthetic, no model):
//
//   Qwen3.8's 24/2 heads of 256, Ornith-1.5's 16/2 of 256 and llm-jp-4.1's 40/4 of 128, on FP16 pools behind a
//   shuffled page table, with a context that ends inside a chunk;
//   1. the output agrees with FP64 to 1e-4 of the output's scale;
//   2. it is bit for bit `qsa_decode_attn_step` given the selection 0, 1, ..., n-1 (the same kernels, cell c in slot c).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace par = strata::parity;

template <typename T> T* up(par::Device& dev, const std::vector<T>& h) {
    T* d = dev.alloc<T>(h.size() + 64 / sizeof(T));
    par::copy_in(d, h.data(), h.size());
    return d;
}

int check(const char* name, int64_t nh, int64_t nkv, int64_t hd) {
    par::Device dev;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = nh; s.n_head_kv = nkv; s.head_dim = hd;
    const int64_t PS = s.page_size, max_cells = 512, n = 300;   // 300: four whole chunks of 64 and a partial one
    const int64_t pages = max_cells / PS, rows = pages * nkv * PS, G = nh / nkv;
    std::mt19937 rng(77 + (unsigned) (nh + hd));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> kh((size_t) (rows * hd)), vh((size_t) (rows * hd));
    for (auto& x : kh) x = k::f16_from_f32(nd(rng) * 1.5f);
    for (auto& x : vh) x = k::f16_from_f32(nd(rng));
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<float> q((size_t) (nh * hd));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> step((size_t) k::kStepCount, 0), ids((size_t) max_cells, 0);
    step[k::kStepNKv] = (int32_t) n;
    step[k::kStepWidth] = (int32_t) n;
    for (int64_t c = 0; c < n; ++c) ids[(size_t) c] = (int32_t) c;

    // FP64 reference: query head h reads KV head h / G (QsaShapes' map)
    std::vector<double> ref((size_t) (nh * hd), 0.0);
    double scale_out = 0.0;
    for (int64_t h = 0; h < nh; ++h) {
        const int64_t kvh = h / G;
        std::vector<double> sc((size_t) n);
        double m = -1e300;
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
            double d = 0.0;
            for (int64_t i = 0; i < hd; ++i) d += (double) q[(size_t) (h * hd + i)] * k::f32_from_f16(kh[(size_t) (row * hd + i)]);
            sc[(size_t) c] = d / std::sqrt((double) hd);
            m = std::max(m, sc[(size_t) c]);
        }
        double l = 0.0;
        for (auto& v : sc) { v = std::exp(v - m); l += v; }
        for (int64_t c = 0; c < n; ++c) {
            const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
            for (int64_t i = 0; i < hd; ++i)
                ref[(size_t) (h * hd + i)] += sc[(size_t) c] / l * k::f32_from_f16(vh[(size_t) (row * hd + i)]);
        }
    }
    for (double v : ref) scale_out = std::max(scale_out, std::fabs(v));

    k::QsaAttnPools pl;
    pl.k_pool = up(dev, kh);
    pl.v_pool = up(dev, vh);
    pl.page_table = up(dev, table);
    const float* d_q = up(dev, q);
    const int32_t* d_step = up(dev, step);
    const int32_t* d_ids = up(dev, ids);
    float* scratch = dev.alloc<float>((size_t) k::qsa_decode_attn_scratch_floats(max_cells, s));
    float* d_dense = dev.alloc<float>((size_t) (nh * hd));
    float* d_sel = dev.alloc<float>((size_t) (nh * hd));
    k::qsa_dense_attn_step(d_q, pl, d_step, max_cells, s, scratch, d_dense, nullptr);
    k::qsa_decode_attn_step(d_q, pl, d_ids, d_step, max_cells, s, scratch, d_sel, nullptr);
    std::vector<float> dense((size_t) (nh * hd)), sel((size_t) (nh * hd));
    par::copy_out(dense.data(), d_dense, dense.size());
    par::copy_out(sel.data(), d_sel, sel.size());

    double err = 0.0;
    for (size_t i = 0; i < dense.size(); ++i) err = std::max(err, std::fabs((double) dense[i] - ref[i]));
    const bool same = std::memcmp(dense.data(), sel.data(), dense.size() * sizeof(float)) == 0;
    const double rel = err / scale_out;
    std::printf("  %-34s vs FP64 %.3e of the scale %s; the identity selection's bits %s\n", name, rel,
                rel <= 1e-4 ? "ok" : "FAIL", same ? "ok" : "FAIL");
    return (rel <= 1e-4 ? 0 : 1) + (same ? 0 : 1);
}
}  // namespace

int main() {
    int bad = 0;
    bad += check("Qwen3.8 QSA (24/2 x 256)", 24, 2, 256);
    bad += check("Ornith-1.5 (16/2 x 256)", 16, 2, 256);
    bad += check("llm-jp-4.1 (40/4 x 128)", 40, 4, 128);
    std::printf("qsa_dense_attn: %d failures\n", bad);
    return bad ? 1 : 0;
}
