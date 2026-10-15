// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/prompt_attn.hpp - the plain models' prompt attention: causal, over every cell of the paged
// FP16 K/V pools, many queries a launch.
//
// The decode kernel the prompt used before (qsa_dense_attn_batch) reads every cell's K and V again for each query:
// at 16K tokens llm-jp-4.1's attention took 62 s of a 73 s prompt.  Here a work-group takes BQ queries of one KV head
// (BQ x the heads that share it: GQA's rows), walks the cells up to its last query 16 at a time through local memory,
// and keeps an online softmax per row, so a cell's K and V are read once per work-group.  RDNA3 and RDNA4 take Q.K^T
// and P.V on their FP16 matrix instructions (WMMA; Q and the probabilities rounded to FP16); every other GPU a kernel
// in FP32 FMAs (K and V enter from FP16).  STRATA_PROMPT_ATTN_MMA=0 takes the FMA kernel everywhere.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// attn[t, h, :] = softmax(q[t, h, :] . K[c, h / G, :] / sqrt(head_dim), c <= pos0 + t) . V for the T queries at
/// positions pos0 .. pos0 + T - 1, whose cells are already in the pools: q and attn [T, n_head, head_dim] FP32, the
/// pools `[page][kv_head][page_size][head_dim]` FP16 through `page_table`.  False, nothing launched, for a shape it is
/// not built for (head_dim 128 with 10 query heads a KV head, 256 with 8 or 12): the caller keeps the decode kernel.
bool prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, void* stream);

}  // namespace strata::kernels
