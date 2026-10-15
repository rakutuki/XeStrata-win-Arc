// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/draft_kv_plan.hpp - which code computes the draft layer's prompt K/V for a chunk.
//
// One GPU: the prompt path's batched pass (E-9, `Prefill::draft_kv`) tries first and the drafter's own
// per-8-row pass (`MtpDrafter::prefill`) takes what it declines.  A layer split: the last stage's prompt path
// tries E-9 as well (opt-in upstream as `STRATA_SPLIT_MTP_BATCH=1`; the default here, `STRATA_SPLIT_MTP_BATCH=0`
// keeps the drafter's own pass).  B70 + RTX 4070, IQ2_XS, K=45, a 32K prompt: 2,350 -> 2,424 tok/s; upstream
// measured 21.9% slower with every expert in VRAM (2 x R9700).  The drafter's K/V then come from Q8_1 x Q8_0 MMQ
// instead of its own mmvq, so its drafts can differ a little; the target's tokens are decided by the verify window
// and do not.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include <cstdlib>

namespace strata::program::draft_kv {

/// `STRATA_SPLIT_MTP_BATCH`'s value: `0` turns it off; unset, empty, any other whole number and text that is not a
/// number leave it on.
inline bool split_env_on(const char* value) {
    if (value == nullptr) return true;
    char* end = nullptr;
    const long v = std::strtol(value, &end, 10);
    return end == value || v != 0;
}

struct Plan {
    bool try_batched = false;      ///< call Prefill::draft_kv for this chunk
    const char* why = nullptr;     ///< when `try_batched` is false only because of a missing prerequisite: what is missing
};

/// `use_mtp`: the engine runs a draft layer. `multi_gpu`: a layer split. `split_env`: split_env_on(...) of the
/// variable. `has_native_embed`: the GGUF-form token embedding table exists (a --native pack): the split's
/// last stage reads it, a single GPU's path can also gather from its own weights.
inline Plan plan(bool use_mtp, bool multi_gpu, bool split_env, bool has_native_embed) {
    Plan p;
    if (!use_mtp) return p;
    if (!multi_gpu) {
        p.try_batched = true;
        return p;
    }
    if (!split_env) return p;   // STRATA_SPLIT_MTP_BATCH=0: the drafter's own pass, nothing to say
    if (!has_native_embed) {
        p.why = "the split's last stage has no GGUF-form token embedding table (not --native)";
        return p;
    }
    p.try_batched = true;
    return p;
}

}  // namespace strata::program::draft_kv
