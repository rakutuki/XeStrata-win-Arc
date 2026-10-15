// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_model.hpp - an MoE model with one residual stream, run from its GGUF.
//
// qwen35moe (Ornith-1.5: Gated DeltaNet layers and full-attention layers, a shared expert) and qwen3moe (llm-jp-4.1:
// full attention in every block, no shared expert) - the models whose block is
//
//     x += mixer(rms_norm(x))            the mixer a GDN layer or a gated, partly rotated full attention
//     x += moe(rms_norm(x))              routed experts (softmax top-k, renormalized) + the gated shared expert
//
// and whose head is rms_norm, then the output projection.  Qwen3.8 (four residual streams, the QSA indexer, PLE) keeps
// its own engine; GLM-5.3 will have its own.
//
// The dense weights are uploaded once in their GGUF form and read by the native kernels.  The routed experts go
// through the engine's expert tiers, as Qwen3.8's do: a VRAM cache of slots (expert_cache) refilled between windows by
// the adaptive rule, the GGUF read in place (FileExpertSource: the page cache, the drive), the CPU pool for what the
// VRAM does not hold, all in one window protocol (MoeWindow).  A window is 1..max_t tokens at consecutive positions,
// recorded once per size as a graph.
//
// A qwen35moe file with its multi-token-prediction layer (`blk.<n_layer>.nextn.*`, Ornith's official K-quants) drafts
// with it (`spec`): that layer is a full-attention block of the trunk's form with its own K/V, after a projection of
// [rms_norm(embedding of token p) | rms_norm(the trunk's final normed row at p - 1)] (llama.cpp's graph_mtp), and its
// routed experts are one more MoE layer of the tiers.  A window checks the last token and the drafts; the recurrent
// states are copied after each row but the last, and put back to the last accepted row's copy when a draft is wrong.
#pragma once

#include "strata/artifact/model_spec.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/moe_window.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/sampler.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufModel;
}

namespace strata::core {

struct PlainOptions {
    std::string gguf;               ///< any shard of the model
    int64_t max_cells = 4096;       ///< the context the K/V cache holds
    int max_t = 8;                  ///< the largest window
    int pool_workers = 0;           ///< CPU pool workers (0: every physical core but the first)
    int64_t reserve_mib = -1;       ///< VRAM left free after the expert slots (-1: from the device's memory)
    int adapt_every = 4;            ///< windows between adaptive rounds (0: off)
    int adapt_swaps = 96;           ///< the most experts one round moves into the VRAM tier
    /// ... and at most this many MiB of experts for each token the windows read since the last round: the copies share
    /// the RAM's bandwidth with the missed experts.  48: on the RX 9060 XT with a Ryzen 7 5700X (DDR4, 2026-10-12),
    /// llm-jp-4.1 (4.6 MB experts, a token a window) went 31.3 -> 34.3 tok/s at 8 GB (96 a round had been 100 MB a token)
    /// and 58.6 -> 60.0 at 16 GB (32: 35.3 and 58.1); Ornith Q6_K (2.5 MB, MTP windows) stays at its 96 a round, 21-24 MB
    /// a token.  On the B70 with an i7-14700 (DDR5) the bound changed nothing (llm-jp 8 GB 41.5 / 41.6).  0: none.
    int adapt_mib_per_token = 48;
    float adapt_decay = 0.7f;       ///< the routing counts are multiplied by this after each round
    /// The experts read from the GGUF in place (the page cache, the drive) instead of loaded into pinned RAM: the
    /// low-RAM mode, taken by itself when the RAM available is less than the experts.  It has no PCIe share.
    bool mmap_experts = false;
    bool pcie_split = true;         ///< split the missed experts between the CPU and PCIe by measured times
    /// A ranking of (MoE layer, expert) pairs (expert_cache.hpp's profile format) the VRAM tier is filled from at load,
    /// each layer up to its slots; empty: the tier starts empty and the adaptive rule fills it.
    std::string profile;
    /// The rope scaling, as llama.cpp's options: "yarn" / "linear" / "none" (empty: the file's), the factor (0: the
    /// file's), the trained context (0: the file's, else its context length).  llm-jp-4.1 at 256K: yarn, 4, 65536.
    std::string rope_scaling;
    double rope_scale = 0;
    double yarn_orig_ctx = 0;
    /// The most tokens a prompt chunk takes through the batched products (`prompt`; a shorter prompt, one chunk of its
    /// length); 0: the prompt runs in windows.
    int64_t prompt_chunk = 4096;
    /// Speculative decoding with the model's MTP layer: windows of up to this many tokens, the last accepted one and
    /// spec - 1 drafts (0 or 1: off; a model without the layer: off).  Greedy requests only.  3 by default: two
    /// drafts were the fastest of one to three on the RX 9060 XT (16 and 8 GB), the B70 and the RTX 4070 (Ornith Q6_K
    /// and Q4_K_M, 2026-10-12).
    int spec = 3;
};

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding): one object, its members grouped by what they serve
class PlainModel {
public:
    PlainModel();
    ~PlainModel();
    PlainModel(const PlainModel&) = delete;
    PlainModel& operator=(const PlainModel&) = delete;

    /// Reads the model's shape, uploads its dense weights, opens its experts and sizes the VRAM tier from what the
    /// device has left.  False, with the reason, for a model this engine does not run.
    bool load(const PlainOptions& opt, std::string& err);
    const ModelSpec& spec() const { return spec_; }
    int64_t vocab() const { return n_vocab_; }
    /// The model carries a recurrent state (GDN layers): a sequence can be extended, not cut back to a prefix.
    bool recurrent() const { return !gdn_state_.empty(); }
    int64_t max_cells() const { return opt_.max_cells; }

    /// Zeroes the recurrent states and the K/V cache: the next window starts a sequence at position 0.
    bool reset(std::string& err);
    /// Tokens [0, T) at positions pos .. pos + T - 1 (T <= max_t): out[t] is the greedy token after token t, and with
    /// `logits` the T rows of n_vocab logits.  Positions must follow the previous window's.
    bool forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err);
    /// The prompt: tokens [0, n) at positions pos .. pos + n - 1 in chunks of `prompt_chunk` tokens, every projection a
    /// matrix product and the routed experts grouped by expert, all on the GPU (src/prefill/plain_prompt.cpp, in the
    /// prompt path's library).  `out`: the greedy token after the last; `logits`: its row (or null).  The state is left
    /// where the windows would have left it.
    bool prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    /// How forward and prompt pick their tokens (greedy by default): each row's draw counted from its position, so a
    /// seed repeats a run.
    void set_sampler(const strata::kernels::SamplerParams& sp) { sampler_ = sp; }
    /// Runs each dense product the prompt takes once, so the products' library has loaded its kernels before the
    /// first prompt (plain_prompt.cpp); nothing when the prompt runs in windows.
    bool prompt_warm(std::string& err);

    /// Drafts a window can check (0: no speculative decoding): spec - 1 when the model has its MTP layer.
    int drafts() const { return has_mtp_ ? n_snap_ : 0; }
    /// After prompt() (or the windows of a prompt): `first`, the token picked after it, at position `pos`, the next
    /// window's first token; the MTP layer reads it and drafts the tokens after it.
    bool spec_start(int32_t first, int64_t pos, std::string& err);
    /// One window over the pending token and its drafts: `emitted` gets the accepted drafts and the token after them
    /// (`n_emit` of them, at most `limit`), cut after the first token in `stop`; with `logits`, their rows.  The state
    /// then holds every emitted token but the last, which is pending, and the next drafts are made.
    bool spec_step(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit, float* logits,
                   std::string& err);
    /// The windows and the drafts they checked and accepted since the load, for the report.
    int64_t spec_windows() const { return spec_windows_; }
    int64_t spec_drafted() const { return spec_drafted_; }
    int64_t spec_accepted() const { return spec_accepted_; }
    /// The time spent in the windows that checked drafts, and in making the drafts (ms, since the load).
    double spec_ms_check() const { return spec_ms_check_; }
    double spec_ms_draft() const { return spec_ms_draft_; }

    /// The VRAM tier, for the report: slots, slots holding an expert, and the routed entries it served so far.
    int64_t slots() const { return n_slots_; }
    int64_t resident() const;
    int64_t hits() const { return disp_.cache_hits; }
    int64_t misses() const { return disp_.cache_refused; }
    int64_t swaps() const { return swaps_total_; }
    /// Writes what this run learned as a profile (rank_learned_profile: the experts in VRAM now, then by the routing
    /// counted since the load, then by the profile it started from), for the next load's `profile`.
    bool save_profile(const std::string& path, std::string& err) const;

private:
    struct W {   // one 2-D projection in VRAM, in its GGUF blocks (or the engine's row order for Q6_K)
        const void* data = nullptr;
        int type = -1;
        int n_in = 0, n_out = 0;
    };
    struct Block {
        bool recurrent = false;
        int64_t gdn = -1, attn = -1;    ///< index among the GDN / attention blocks
        const float *attn_norm = nullptr, *post_norm = nullptr;
        W qkv, z, alpha, beta, ssm_out;                       // GDN
        const float *conv = nullptr, *dt = nullptr, *a = nullptr, *ssm_norm = nullptr;
        W q, k, v, o;                                         // attention
        const float *q_norm = nullptr, *k_norm = nullptr;
        const float* router = nullptr;                        // F32 [n_embd, n_expert]
        const float* sh_gate_inp = nullptr;                   // F32 [n_embd]
        W sh_gate, sh_up, sh_down;
    };

    bool upload(const GgufModel& m, const std::string& name, W& w, std::string& err);
    /// Block `l`'s weights (blk.<l>.*): its mixer, norms, router and shared expert.
    bool load_block(const GgufModel& m, int64_t l, Block& b, std::string& err);
    bool upload_f32(const GgufModel& m, const std::string& name, uint64_t elements, const float*& out, std::string& err);
    bool open_experts(std::string& err);
    bool record(int T, bool mtp, std::string& err);
    /// The window's inputs (embeddings, steps, positions) for `tokens` at pos .. pos + T - 1 into the mapped staging.
    bool stage(const int32_t* tokens, int T, int64_t pos, std::string& err);
    bool record_window(int T, strata::gpu::Stream cs, std::string& err);
    /// The head over T normed rows `x` into logits_.
    void head_logits(const float* x, int T, strata::gpu::Stream cs);
    /// One block over the window's T rows of x_ (MoE layer `ml` of the tiers, ring `ring` + 1): a segmented window's
    /// graphs go into `segs`.
    bool record_block(const Block& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                      strata::gpu::Stream cs, std::string& err);
    /// The MTP layer over T rows: the embeddings staged in m_emb_, the trunk's rows in mtp_h_; the last row's normed
    /// output into mtp_h_'s first row (the next draft step's input) and its logits.
    bool record_mtp(int T, strata::gpu::Stream cs, std::string& err);
    /// Launches a window's graph (or its segments) and answers its `rings` rings, MoE layers layers[0 ..].
    bool launch_serve(strata::gpu::Graph* g, std::vector<strata::gpu::Graph*>& segs, int64_t rings,
                      const int64_t* layers, int T, std::string& err);
    /// The MTP layer over T rows at positions pos .. pos + T - 1 (tokens[t] the token at pos + t, mtp_h_'s row t the
    /// trunk's row before it): the greedy draft after the last.
    bool mtp_run(int T, const int32_t* tokens, int64_t pos, int32_t* draft, std::string& err);
    /// The chain after mtp_run's first draft: drafts_ up to `want` from position `pos` (the first draft's).
    bool mtp_chain(int want, int64_t pos, std::string& err);
    bool adapt(std::string& err);
    /// The adaptive tier's copies in flight: their experts become resident once the copies are done (`wait`: wait
    /// for them; else only if they are).
    bool settle_swaps(bool wait, std::string& err);
    bool fill_slot(int64_t layer, int32_t expert, std::string& err);
    /// The prompt path's buffers for chunks of `T` tokens, carved by prompt_carve: the pinned host block allocated at
    /// load for the longest chunk, the device block borrowed from the tier for each prompt (plain_prompt.cpp uses them).
    struct PromptBufs {
        static constexpr int kGroup = 8;   ///< experts dequantized and multiplied in one grouped product
        int64_t T = 0, gu_elems = 0, d_elems = 0, gemm_elems = 0;
        bool int8 = false;                 ///< every MoE layer's experts through the int8 products (no FP16 buffers)
        float *x = nullptr, *hn = nullptr, *mix = nullptr;
        uint16_t* x16 = nullptr;           ///< T rows of the widest projection input, FP16
        float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *h = nullptr, *y = nullptr;
        uint16_t* y16 = nullptr;
        float *qfull = nullptr, *qcur = nullptr, *kraw = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr,
              *attn32 = nullptr;
        int32_t *steps = nullptr, *pos_q = nullptr, *pos_k = nullptr;
        float *rlog = nullptr, *w = nullptr, *shared = nullptr, *sh_g = nullptr, *sh_gate = nullptr, *sh_up = nullptr;
        int32_t* ids = nullptr;
        uint16_t* xs16 = nullptr;          ///< the routed rows grouped by expert, FP16 [T * k, n_embd]
        float *gu = nullptr, *hh = nullptr, *dm = nullptr, *parts = nullptr, *moe_out = nullptr;
        uint16_t* h16 = nullptr;
        int32_t *src = nullptr, *slot = nullptr, *bounds = nullptr;
        int32_t* entry = nullptr;          ///< grouped row -> routing entry: where the int8 down product writes
        uint16_t *dq_gu = nullptr, *dq_d = nullptr;   ///< kGroup experts' FP16 weights
        uint8_t* stage = nullptr;          ///< 2 kGroup expert blobs copied from RAM (two halves)
        uint8_t *xq = nullptr, *hq = nullptr;   ///< the int8 products' activations: the routed rows, the SwiGLU's
        uint8_t* xq2 = nullptr;            ///< the routed rows for the up product of a layer whose gate and up
                                           ///< differ in the blocks' repetition (iq_mmq_type_dup); else null
        int32_t* identity = nullptr;       ///< row i is row i (the int8 products' output map)
        const uint8_t** wtab = nullptr;    ///< a layer's experts' blobs, for the int8 products
        int32_t* rbounds = nullptr;        ///< each group's bounds from its first row (the down product's)
        int32_t* dense_bounds = nullptr;   ///< {0, T}: a dense product as one int8 "expert"
        int64_t cpu_cap = 0;               ///< routed rows the CPU's share of a layer may take
        float* cpu = nullptr;              ///< the CPU's rows, sent back to be scattered into parts
        int32_t* cpu_row = nullptr;        ///< the routing entry each of them belongs to
        uint16_t* gemm_scratch = nullptr;  ///< Gemm::native's dequantized weight
        // pinned host
        float* emb = nullptr;              ///< T rows of n_embd
        int32_t *h_steps = nullptr, *h_pos = nullptr, *h_ids = nullptr, *h_src = nullptr, *h_slot = nullptr,
                *h_bounds = nullptr, *h_entry = nullptr;
        const uint8_t** h_wtab = nullptr;
        float *h_hn = nullptr, *h_cpu = nullptr;   ///< the layer's normed rows, the CPU's output rows
        int32_t* h_cpu_row = nullptr;
        int32_t* h_dense_bounds = nullptr;
        uint8_t* h_nact = nullptr;         ///< the CPU's activations: a token's in the layer's native format
        int32_t* h_rbounds = nullptr;
    };
    /// Lays the buffers for chunks of `T` tokens out from `dev` / `host` (null: only counts), returning the device and
    /// host bytes.
    /// Whether every MoE layer's experts take the int8 products on this GPU (iq_mmq; STRATA_PLAIN_MMQ=0: no).
    bool prompt_int8() const;
    void prompt_carve(int64_t T, PromptBufs& b, uint8_t* dev, uint8_t* host, uint64_t& dev_bytes,
                      uint64_t& host_bytes) const;
    PromptBufs pb_;
    void* prompt_host_ = nullptr;
    /// The prompt's device buffers of their own, for chunks of up to prompt_dev_T_ tokens, when every expert has a
    /// slot and the VRAM beside the tier holds them (else null: prompt_borrow lends the tier's last slots).
    void* prompt_dev_ = nullptr;
    int64_t prompt_dev_T_ = 0;
    /// The prompt's device buffers, carved from the tier's last slots for one prompt (prompt_borrow) and given back
    /// after it (prompt_return), with the experts that were in them put back: the tier keeps that memory between
    /// prompts.  The slots go to the layers in turn (slot s is layer s % L's), so the last ones are every layer's.
    bool prompt_borrow(int64_t T, std::string& err);
    bool prompt_chunks(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    bool prompt_warm_products(std::string& err);
    bool prompt_cpu_experts(int64_t ml, const std::vector<int32_t>& experts, const std::vector<int32_t>& count,
                            const std::vector<int32_t>& start, std::string& err);
    bool prompt_return(std::string& err);
    int64_t borrow_first_ = -1;
    std::vector<std::pair<int32_t, int32_t>> borrowed_res_;   // (layer, expert) the borrowed slots held
    struct PromptGemm;   // plain_prompt.cpp: the products' state over pb_.gemm_scratch
    static void prompt_gemm_free(PromptGemm* g);
    PromptGemm& prompt_gemm();
    std::unique_ptr<PromptGemm, void (*)(PromptGemm*)> prompt_gemm_{nullptr, nullptr};
    static void pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                        int64_t layer);

    PlainOptions opt_;
    ModelSpec spec_;
    std::unique_ptr<GgufModel> model_;
    int64_t n_vocab_ = 0;
    int64_t nkv_ = 0;                ///< the attention blocks' K/V heads (the same in every one)
    bool has_gate_ = false;          ///< the attention's output gate: [q | gate] per head (qwen35moe)
    bool has_shared_ = false;        ///< the gated shared expert (qwen35moe)
    int emb_type_ = -1;
    const uint8_t* emb_ = nullptr;   ///< token_embd in the GGUF mapping (rows dequantized on the host)
    size_t emb_row_ = 0;
    W head_;
    const float* out_norm_ = nullptr;
    std::vector<Block> blocks_;
    // the MTP layer (has_mtp_): its block, the projection of [norm(embedding) | norm(trunk row)], the two norms, the
    // norm before the head (the trunk's when the file has none)
    bool has_mtp_ = false;
    Block mtp_;
    W mtp_eh_;
    const float *mtp_enorm_ = nullptr, *mtp_hnorm_ = nullptr, *mtp_head_norm_ = nullptr;
    int64_t n_moe_ = 0;   ///< MoE layers of the tiers: the trunk's, then the MTP layer's
    std::vector<void*> allocs_;      ///< every device allocation, freed in the destructor

    // the expert tiers
    std::unique_ptr<ExpertSource> src_;
    PcieSplitTimes split_;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_;
    ExpertCache cache_;
    ExpertDispatch disp_;
    MoeWindow win_;
    std::vector<int32_t> res_;                    ///< [moe layer][expert] slot or -1
    std::vector<std::vector<int32_t>> free_;      ///< per moe layer, its free slots
    std::vector<double> heat_;                    ///< [moe layer][expert] routed entries since the load
    std::vector<std::pair<int32_t, int32_t>> prior_;   ///< the profile the tier was filled from
    int64_t n_slots_ = 0;
    int64_t windows_ = 0, swaps_total_ = 0;
    int64_t tokens_since_adapt_ = 0;   ///< tokens the windows read since the last adaptive round
    uint8_t* fill_tmp_ = nullptr;                 ///< pinned staging for a slot fill

    // per-sequence device state
    std::vector<float*> gdn_state_, gdn_conv_;   ///< per GDN block
    /// The recurrent states after a window's rows 0 .. n_snap_ - 1, per GDN block (gdn_snap_[block][row]), for
    /// spec_step to go back to the last accepted row.
    int n_snap_ = 0;
    std::vector<std::vector<float*>> gdn_snap_, conv_snap_;
    // the drafting: the pending token, its position, the drafts after it; the MTP layer's input rows (device, max_t of
    // n_embd); the counts for the report
    int32_t cur_ = -1;
    int64_t cur_pos_ = 0;
    std::vector<int32_t> drafts_;
    float* mtp_h_ = nullptr;
    bool mtp_h_set_ = false;   ///< prompt() left the trunk's last row in mtp_h_ (else forward's last row in hn_)
    int last_T_ = 0;           ///< the last forward window's size
    int64_t spec_windows_ = 0, spec_drafted_ = 0, spec_accepted_ = 0;
    double spec_ms_check_ = 0, spec_ms_draft_ = 0;
    std::vector<uint16_t*> k_pool_, v_pool_;      ///< per attention block
    int32_t* page_table_ = nullptr;
    int64_t n_pages_ = 0;

    // the window's buffers (device unless noted)
    strata::gpu::Stream cs_ = nullptr;
    // the adaptive tier's copies (from the pinned RAM copy, beside the windows) and the experts they bring in
    strata::gpu::Stream swap_cs_ = nullptr;
    strata::gpu::Event* swap_ev_ = nullptr;
    struct PendingSwap { int32_t layer, expert, slot; };
    std::vector<PendingSwap> pending_;
    std::vector<strata::gpu::Graph*> exec_;                      ///< per T: the whole window
    std::vector<std::vector<strata::gpu::Graph*>> segs_;         ///< per T: the segments (MoeWindow::segmented)
    std::vector<strata::gpu::Graph*> mexec_;                     ///< per T: the MTP layer
    std::vector<std::vector<strata::gpu::Graph*>> msegs_;
    float *h_emb_ = nullptr, *m_emb_ = nullptr;                  ///< mapped: T rows of n_embd
    int32_t *h_step_ = nullptr, *m_step_ = nullptr;              ///< mapped: T x kStepCount
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr;                ///< mapped: T x n_head, then T x n_head_kv
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;                ///< mapped: T picked tokens
    strata::kernels::SamplerParams sampler_ = [] {
        strata::kernels::SamplerParams p;
        p.greedy = true;
        p.temperature = 0.0f;
        return p;
    }();
    int32_t *step_ = nullptr, *pos_ = nullptr;
    float *x_ = nullptr, *hn_ = nullptr, *mix_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qkv_ = nullptr, *craw_ = nullptr, *hc_ = nullptr, *z_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr,
          *gate_ = nullptr, *o_ = nullptr, *y_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kraw_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr,
          *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float *rlog_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *sh_g_ = nullptr, *sh_gate_ = nullptr, *sh_up_ = nullptr,
          *moe_out_ = nullptr, *logits_ = nullptr;
    int32_t* ids_ = nullptr;
    int64_t attn_scratch_floats_ = 0;
};

}  // namespace strata::core
