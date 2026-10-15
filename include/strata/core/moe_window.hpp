// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/moe_window.hpp - the routed experts of a window of 1..8 tokens, for any MoE model.
//
// The verify window's expert protocol (verify.cpp, Verifier::record_window's pre/post and Verifier::run) taken apart
// from Qwen3.8's layers: it depends only on the activation width, the expert count, k, and the process's expert
// layout (`expert_layout()`, one entry per MoE layer).  In the graph a caller records, per MoE layer and token group:
//
//     publish   the normed input, the ids and the weights to mapped memory, the ring rung; the input's q8_1 image
//     experts   wait for the host's plan, the VRAM groups, the PCIe groups, wait for the CPU's rows, add them up:
//               parts() rows [tb * k, (tb + n) * k) hold every routed expert's output, unweighted, in routing order
//
// and on the host, after launching the graph, `serve` answers each ring with the CPU pool and the plan (the same
// GpuPlanSink the verify window fills: VRAM hits, the PCIe share, the CPU's rows).  The combine (weights, shared
// expert) is the caller's, as its form is the model's.
//
// Ring r (1, 2, ...) is the r-th publish of the window, in the order the graph records them; `serve` must be told
// which MoE layer and token group each one belongs to, in that order.
#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The CPU pool for one ring: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), the GPU's rows zeroed;
/// `layer` is the MoE layer (the expert layout's index).  Fills the plan through the window's sink (`sink()`).
using MoePoolFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                           int64_t layer);

struct MoeWindowShape {
    int64_t n_embd = 0;
    int64_t n_expert = 0;
    int64_t k = 0;
    int max_t = 1;          ///< tokens a window holds, 1..kVerifyMaxT
    int groups = 1;         ///< token groups a window is cut into (1 or 2), each with its own plan and staging
};

/// The VRAM expert tier the window reads: its slot arena and the residency table of the MoE layers.
struct MoeWindowTier {
    const uint8_t* cache_base = nullptr;   ///< slot 0 of the VRAM expert arena
    int64_t blob = 0;                      ///< bytes per slot (the largest layer's blob)
};

class MoeWindow {
public:
    MoeWindow() = default;
    ~MoeWindow();
    MoeWindow(const MoeWindow&) = delete;
    MoeWindow& operator=(const MoeWindow&) = delete;

    bool init(const MoeWindowShape& shape, const MoeWindowTier& tier, std::string& err);

    /// The running kernels of this device do not see the host's writes (doorbell_visible): the caller records the
    /// window as segments that end before each `experts`, and `serve` launches them (`set_segments`).
    bool segmented() const { return segmented_; }
    void set_segments(const std::vector<strata::gpu::Graph*>* segments) { segs_ = segments; }

    // ---- recorded in the caller's graph, on its stream
    /// Ring `ring` (1-based, the window's publish count): tokens [tb, tb + n) of the token group `grp`.  `x` is the
    /// layer's normed input (n rows of n_embd), `ids` / `w` the router's (n rows of k).
    void publish(uint32_t ring, int grp, int tb, int n, const float* x, const int32_t* ids, const float* w,
                 strata::gpu::Stream cs);
    /// The experts of MoE layer `layer` for the same ring: parts() rows [tb * k, (tb + n) * k).
    bool experts(int64_t layer, uint32_t ring, int grp, int tb, int n, strata::gpu::Stream cs, std::string& err);
    /// Device (max_t * k, n_embd): the experts' outputs, row t * k + j for token t's j-th routed expert.
    float* parts() const { return parts_; }

    // ---- the host, around one launch of the caller's graph
    /// Before the launch: the rings and flags back to 0.
    void reset();
    /// After the launch: answers `rings` rings in order, ring r being MoE layer layers[r - 1] for token group
    /// grps[r - 1] (tokens [tb, tb + n) of the group's split).  `cs` is the stream the graph runs on.
    bool serve(int64_t rings, const int64_t* layers, const int* grps, const int* tbs, const int* ns, MoePoolFn pool,
               void* user, strata::gpu::Stream cs, std::string& err);
    /// The plan the pool fills (ExpertDispatch::plan).
    GpuPlanSink* sink() { return &sink_; }

private:
    static void publish_plan(void* ctx);
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    void set_plan_slot(int grp);

    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;

    MoeWindowShape sh_;
    MoeWindowTier tier_;
    bool segmented_ = false;
    const std::vector<strata::gpu::Graph*>* segs_ = nullptr;
    strata::gpu::Stream copy_ = nullptr;
    uint32_t cur_ring_ = 0;

    // mapped staging: the host's view (h_) and the device's (m_) of the same bytes
    float *h_x_ = nullptr, *m_x_ = nullptr;
    int32_t *h_ids_ = nullptr, *m_ids_ = nullptr;
    float *h_w_ = nullptr, *m_w_ = nullptr;
    float *h_ymiss_ = nullptr, *m_ymiss_ = nullptr;
    uint32_t *h_seq_ = nullptr, *m_seq_ = nullptr;
    uint32_t *h_flag_ = nullptr, *m_flag_ = nullptr;     ///< the CPU's rows are in place
    uint32_t *h_flagA_ = nullptr, *m_flagA_ = nullptr;   ///< the plan is published
    uint32_t *h_flagB_ = nullptr, *m_flagB_ = nullptr;   ///< the PCIe share has landed in staging
    int32_t *h_plan_ = nullptr, *m_plan_ = nullptr;
    int64_t plan_i32_ = 0;
    GpuPlanSink sink_;
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[64] = {};

    // the device arena
    void* arena_ = nullptr;
    int32_t* plan_ = nullptr;
    float* parts_ = nullptr;
    float* hit_out_ = nullptr;
    uint8_t* staging_ = nullptr;
    uint8_t* nat_xq_ = nullptr;
    float* nat_xf_ = nullptr;      ///< the tokens' FP32 rows, for BF16 experts (bf16_)
    bool bf16_ = false;            ///< a MoE layer's experts are BF16 (native_expert_grouped_bf16)
    uint8_t* hit_xq_ = nullptr;
    float* hit_xs_ = nullptr;
    uint8_t* hit_scratch_ = nullptr;
};

}  // namespace strata::core
