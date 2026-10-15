// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses (2560 values as Q8_K: 10 x 292; a BF16 layer's
/// activation is BF16, llm-jp-4.1's 2560 values 5120 bytes).
inline constexpr size_t kNativeActBytes = 8192;
/// Bytes of the largest quantized down activation (640 values as Q8_0: 20 x 34, or Q8_K 3 x 292; llm-jp-4.1's 960
/// as Q8_0: 30 x 34, as BF16 1920).
inline constexpr size_t kNativeHBytes = 2048;
/// The widest expert intermediate a native layer may have (llm-jp-4.1: 960); the pool's row buffers are this wide.
inline constexpr int64_t kNativeMaxFF = 1024;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    /// The up rows' type: gu_type, or another one with the same activation format (Ornith-1.5's GSQ-RCO files mix
    /// Q2_K, Q3_K and Q4_K between gate and up in some layers).  gu_row and up_off are the gate rows'.
    int up_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_row = 0;                  ///< bytes per up row (gu_row when up_type is gu_type)
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
};

/// Whether the layer's gate and up rows are of different types (a NativeFmt filled by native_fmt; one built another
/// way leaves up_type at -1: the gate's).
inline bool native_mixed_gu(const NativeFmt& f) { return f.up_type >= 0 && f.up_type != f.gu_type; }

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.  `up_type` (-1: the gate's)
/// is the up rows' when they differ, which must quantize the activation the same way.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                int up_type = -1);

/// One row of `n` values of a GGUF tensor of ggml type `type` to floats, by ggml's own dequantizer (a model's
/// embedding row read on the host).  False when ggml has none for the type or `n` is not whole blocks.
bool native_row_to_float(int type, const void* src, float* dst, int64_t n);
/// Bytes of one row of `n` values of ggml type `type` (0 when `n` is not whole blocks).
size_t native_row_bytes(int type, int64_t n);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
