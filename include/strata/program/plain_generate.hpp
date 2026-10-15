// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/plain_generate.hpp - `strata generate` for the models of plain_model.hpp.
#pragma once

namespace strata::program {

/// When `--native PATH` names a model that PlainModel runs (not qwen4exp), runs it and returns the exit status;
/// otherwise returns -1 and Qwen3.8's engine takes the command line.
int plain_main(int argc, char** argv);

}  // namespace strata::program
