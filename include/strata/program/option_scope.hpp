// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/option_scope.hpp - which of the engine's options a model other than Qwen3.8-Flash-Next takes.
//
// `strata` runs Qwen3.8-Flash-Next (qwen4exp) through generate.cpp and the models of plain_model.hpp (qwen35moe,
// qwen3moe) through plain_generate.cpp, which takes a subset of the options.  An option it does not take is refused
// with the reason: one of Qwen3.8-Flash-Next's own structure (its pack, PLE, QSA, GR streams, vision tower, and the
// switches of their kernels), or one the other models' path does not have yet.
#pragma once

#include <string>

namespace strata::program {

enum class OptionScope {
    unknown,       ///< not an option of the engine
    qwen38_only,   ///< Qwen3.8-Flash-Next's own structure: no other model has it
    not_yet,       ///< every model could take it; the other models' path does not yet
};

/// The scope of `name` (with its leading "--") for a model the plain path runs.  Options the plain path takes are
/// not asked about.
OptionScope option_scope(const std::string& name);

}  // namespace strata::program
