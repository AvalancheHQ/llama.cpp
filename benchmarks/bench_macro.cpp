// CodSpeed macro benchmarks for the ggml compute core of llama.cpp.
//
// While bench_ggml.cpp measures isolated kernels under CPU simulation, this
// file contains larger, end-to-end style workloads intended for wall-time
// measurement on dedicated bare-metal runners (CodSpeed macro runners). Each
// benchmark evaluates a complete multi-operator compute graph on the CPU
// backend with multiple threads, mirroring what llama.cpp actually executes
// during inference:
//
//   - prompt_layer:  a full transformer layer (RMS-norm, QKV projections,
//                    RoPE, causal masked attention, output projection,
//                    residuals and a SwiGLU feed-forward block) processing a
//                    64-token prompt batch — the prompt-processing hot path.
//   - decode_layer:  the same layer generating a single token against a
//                    1024-entry KV cache — the token-generation hot path.
//   - decode_deep:   four stacked decode layers in one graph, exercising the
//                    graph scheduler across layer boundaries.
//   - lm_head:       the final logits projection onto a 32000-entry
//                    vocabulary.
//
// The layer geometry matches a TinyLlama-1.1B block (n_embd=2048, 32 heads,
// n_ff=5632) so the tensor shapes are representative of real models, and the
// dominant weight matrices are benchmarked in F32 as well as the most common
// quantized formats. No model files are required: weights are synthesized
// deterministically and quantized in-process.

#include "ggml.h"
#include "ggml-cpu.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

// TinyLlama-1.1B layer geometry.
static constexpr int64_t N_EMBD   = 2048;
static constexpr int64_t N_HEAD   = 32;
static constexpr int64_t HEAD_DIM = N_EMBD / N_HEAD;
static constexpr int64_t N_FF     = 5632;
static constexpr int64_t N_VOCAB  = 32000;

static constexpr int64_t PROMPT_TOKENS = 64;   // prompt-processing batch
static constexpr int64_t KV_LEN        = 1024; // KV-cache entries during decode
static constexpr int64_t LM_TOKENS     = 8;    // tokens hitting the LM head
static constexpr int     DEEP_LAYERS   = 4;    // stacked layers in decode_deep

static int n_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return (int) std::max(1u, std::min(hw, 8u));
}

// A self-contained graph context: owns the tensor arena and the graph.
struct graph_ctx {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;

    explicit graph_ctx(size_t mem_size) {
        ggml_init_params params = {
            /*.mem_size   =*/ mem_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ctx = ggml_init(params);
        gf  = ggml_new_graph(ctx);
    }

    ~graph_ctx() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    // Fill a tensor with deterministic pseudo-random fp32 data.
    void fill_f32(ggml_tensor * t) const {
        float * d = (float *) t->data;
        const int64_t n = ggml_nelements(t);
        for (int64_t i = 0; i < n; ++i) {
            d[i] = 0.1f + 0.5f * std::cos(static_cast<float>(i) * 0.017f);
        }
    }

    // Create a 2D weight tensor of the requested storage type, filled with
    // deterministic synthetic data (quantized in-process when needed).
    ggml_tensor * new_weight(ggml_type type, int64_t ne0, int64_t ne1) {
        std::vector<float> tmp(ne0 * ne1);
        for (size_t i = 0; i < tmp.size(); ++i) {
            tmp[i] = 0.1f + 0.5f * std::cos(static_cast<float>(i) * 0.013f);
        }
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, tmp.data(), w->data, /*start=*/0, ne1, ne0, /*imatrix=*/nullptr);
        return w;
    }
};

// Runs a finalized graph repeatedly with a persistent work buffer. Unlike
// ggml_graph_compute_with_ctx(), this does not allocate a new work-buffer
// object inside the ggml context on every call, so it is safe for the high
// iteration counts of wall-time measurement.
struct graph_runner {
    ggml_cplan           plan{};
    std::vector<uint8_t> work;

    graph_runner(ggml_cgraph * gf, int nth) {
        plan = ggml_graph_plan(gf, nth, /*threadpool=*/nullptr);
        if (plan.work_size > 0) {
            work.resize(plan.work_size);
            plan.work_data = work.data();
        }
    }

    void run(ggml_cgraph * gf) {
        ggml_graph_compute(gf, &plan);
    }
};

// The weight set of one transformer layer.
struct layer_weights {
    ggml_tensor * attn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * ffn_norm;
    ggml_tensor * w_gate;
    ggml_tensor * w_up;
    ggml_tensor * w_down;
};

static layer_weights make_layer_weights(graph_ctx & g, ggml_type wtype) {
    layer_weights w{};
    w.attn_norm = g.new_weight(GGML_TYPE_F32, N_EMBD, 1);
    w.wq        = g.new_weight(wtype, N_EMBD, N_EMBD);
    w.wk        = g.new_weight(wtype, N_EMBD, N_EMBD);
    w.wv        = g.new_weight(wtype, N_EMBD, N_EMBD);
    w.wo        = g.new_weight(wtype, N_EMBD, N_EMBD);
    w.ffn_norm  = g.new_weight(GGML_TYPE_F32, N_EMBD, 1);
    w.w_gate    = g.new_weight(wtype, N_EMBD, N_FF);
    w.w_up      = g.new_weight(wtype, N_EMBD, N_FF);
    w.w_down    = g.new_weight(wtype, N_FF, N_EMBD);
    return w;
}

// SwiGLU feed-forward block with pre-norm and residual, shared by the prompt
// and decode layer builders.
static ggml_tensor * build_ffn(graph_ctx & g, const layer_weights & w, ggml_tensor * x) {
    ggml_tensor * cur  = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, 1e-5f), w.ffn_norm);
    ggml_tensor * gate = ggml_silu(g.ctx, ggml_mul_mat(g.ctx, w.w_gate, cur));
    ggml_tensor * up   = ggml_mul_mat(g.ctx, w.w_up, cur);
    ggml_tensor * ffn  = ggml_mul_mat(g.ctx, w.w_down, ggml_mul(g.ctx, gate, up));
    return ggml_add(g.ctx, x, ffn);
}

static ggml_tensor * apply_rope(graph_ctx & g, ggml_tensor * cur, ggml_tensor * pos) {
    return ggml_rope_ext(
        g.ctx, cur, pos, /*freq_factors=*/nullptr,
        (int) HEAD_DIM, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/2048,
        /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
        /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
        /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
}

// Full transformer layer over a prompt batch: causal self-attention across the
// batch itself (no KV cache), followed by the SwiGLU block.
static ggml_tensor * build_prompt_layer(
        graph_ctx & g, const layer_weights & w,
        ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask, int64_t n_tokens) {
    ggml_tensor * cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, 1e-5f), w.attn_norm);

    ggml_tensor * qcur = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, w.wq, cur), HEAD_DIM, N_HEAD, n_tokens);
    ggml_tensor * kcur = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, w.wk, cur), HEAD_DIM, N_HEAD, n_tokens);
    ggml_tensor * vcur = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, w.wv, cur), HEAD_DIM, N_HEAD, n_tokens);

    qcur = apply_rope(g, qcur, pos);
    kcur = apply_rope(g, kcur, pos);

    ggml_tensor * q = ggml_permute(g.ctx, qcur, 0, 2, 1, 3); // [head_dim, n_tokens, n_head]
    ggml_tensor * k = ggml_permute(g.ctx, kcur, 0, 2, 1, 3); // [head_dim, n_tokens, n_head]
    ggml_tensor * v = ggml_cont(g.ctx, ggml_permute(g.ctx, vcur, 1, 2, 0, 3)); // [n_tokens, head_dim, n_head]

    const float scale = 1.0f / std::sqrt((float) HEAD_DIM);

    ggml_tensor * kq = ggml_mul_mat(g.ctx, k, q); // [n_tokens, n_tokens, n_head]
    kq = ggml_soft_max_ext(g.ctx, kq, mask, scale, /*max_bias=*/0.0f);

    ggml_tensor * kqv = ggml_mul_mat(g.ctx, v, kq); // [head_dim, n_tokens, n_head]
    ggml_tensor * merged = ggml_cont(g.ctx, ggml_permute(g.ctx, kqv, 0, 2, 1, 3));
    merged = ggml_reshape_2d(g.ctx, merged, N_EMBD, n_tokens);

    ggml_tensor * attn_out = ggml_mul_mat(g.ctx, w.wo, merged);
    x = ggml_add(g.ctx, x, attn_out);

    return build_ffn(g, w, x);
}

// Full transformer layer generating one token against a pre-populated KV
// cache: the current token's K/V are still computed (as they would be before
// being appended to the cache), but attention reads from the cache tensors.
static ggml_tensor * build_decode_layer(
        graph_ctx & g, const layer_weights & w,
        ggml_tensor * x, ggml_tensor * pos,
        ggml_tensor * k_cache, ggml_tensor * v_cache) {
    ggml_tensor * cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, 1e-5f), w.attn_norm);

    ggml_tensor * qcur = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, w.wq, cur), HEAD_DIM, N_HEAD, 1);
    ggml_tensor * kcur = ggml_reshape_3d(g.ctx, ggml_mul_mat(g.ctx, w.wk, cur), HEAD_DIM, N_HEAD, 1);
    ggml_tensor * vcur = ggml_mul_mat(g.ctx, w.wv, cur);

    qcur = apply_rope(g, qcur, pos);
    kcur = apply_rope(g, kcur, pos);

    // The freshly computed K/V would be appended to the cache; keep them alive
    // as graph outputs so their computation is part of the measurement.
    ggml_build_forward_expand(g.gf, kcur);
    ggml_build_forward_expand(g.gf, vcur);

    ggml_tensor * q = ggml_permute(g.ctx, qcur, 0, 2, 1, 3); // [head_dim, 1, n_head]

    const float scale = 1.0f / std::sqrt((float) HEAD_DIM);

    ggml_tensor * kq = ggml_mul_mat(g.ctx, k_cache, q); // [kv_len, 1, n_head]
    kq = ggml_soft_max_ext(g.ctx, kq, /*mask=*/nullptr, scale, /*max_bias=*/0.0f);

    ggml_tensor * kqv = ggml_mul_mat(g.ctx, v_cache, kq); // [head_dim, 1, n_head]
    ggml_tensor * merged = ggml_cont(g.ctx, ggml_permute(g.ctx, kqv, 0, 2, 1, 3));
    merged = ggml_reshape_2d(g.ctx, merged, N_EMBD, 1);

    ggml_tensor * attn_out = ggml_mul_mat(g.ctx, w.wo, merged);
    x = ggml_add(g.ctx, x, attn_out);

    return build_ffn(g, w, x);
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

// Prompt processing: one full transformer layer over a 64-token batch.
static void prompt_layer(benchmark::State & state, ggml_type wtype) {
    const size_t mem =
        (size_t) 768 * 1024 * 1024 + ggml_tensor_overhead() * 128 + ggml_graph_overhead();
    graph_ctx g(mem);

    const layer_weights w = make_layer_weights(g, wtype);

    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, PROMPT_TOKENS);
    g.fill_f32(x);

    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, PROMPT_TOKENS);
    for (int64_t i = 0; i < PROMPT_TOKENS; ++i) {
        ((int32_t *) pos->data)[i] = (int32_t) i;
    }

    // Causal attention mask.
    ggml_tensor * mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, PROMPT_TOKENS, PROMPT_TOKENS);
    for (int64_t iq = 0; iq < PROMPT_TOKENS; ++iq) {
        for (int64_t ik = 0; ik < PROMPT_TOKENS; ++ik) {
            ((float *) mask->data)[iq * PROMPT_TOKENS + ik] = ik <= iq ? 0.0f : -INFINITY;
        }
    }

    ggml_tensor * out = build_prompt_layer(g, w, x, pos, mask, PROMPT_TOKENS);
    ggml_build_forward_expand(g.gf, out);

    graph_runner runner(g.gf, n_threads());
    for (auto _ : state) {
        runner.run(g.gf);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations() * PROMPT_TOKENS);
}
BENCHMARK_CAPTURE(prompt_layer, f32,  GGML_TYPE_F32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(prompt_layer, q8_0, GGML_TYPE_Q8_0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(prompt_layer, q4_0, GGML_TYPE_Q4_0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(prompt_layer, q4_k, GGML_TYPE_Q4_K)->Unit(benchmark::kMillisecond);

// Token generation: one full transformer layer decoding a single token
// against a 1024-entry KV cache.
static void decode_layer(benchmark::State & state, ggml_type wtype) {
    const size_t mem =
        (size_t) 768 * 1024 * 1024 + ggml_tensor_overhead() * 128 + ggml_graph_overhead();
    graph_ctx g(mem);

    const layer_weights w = make_layer_weights(g, wtype);

    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, 1);
    g.fill_f32(x);

    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 1);
    ((int32_t *) pos->data)[0] = (int32_t) KV_LEN;

    ggml_tensor * k_cache = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, HEAD_DIM, KV_LEN, N_HEAD);
    ggml_tensor * v_cache = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, KV_LEN, HEAD_DIM, N_HEAD);
    g.fill_f32(k_cache);
    g.fill_f32(v_cache);

    ggml_tensor * out = build_decode_layer(g, w, x, pos, k_cache, v_cache);
    ggml_build_forward_expand(g.gf, out);

    graph_runner runner(g.gf, n_threads());
    for (auto _ : state) {
        runner.run(g.gf);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_CAPTURE(decode_layer, f32,  GGML_TYPE_F32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(decode_layer, q8_0, GGML_TYPE_Q8_0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(decode_layer, q4_0, GGML_TYPE_Q4_0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(decode_layer, q4_k, GGML_TYPE_Q4_K)->Unit(benchmark::kMillisecond);

// Multi-layer decode: four stacked transformer layers in a single graph. The
// weights and KV cache are shared between layers to keep the memory footprint
// small — the point is to measure scheduling and data flow across a deep
// graph, not to hold four layers of parameters.
static void decode_deep(benchmark::State & state, ggml_type wtype) {
    const size_t mem =
        (size_t) 768 * 1024 * 1024 + ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    graph_ctx g(mem);

    const layer_weights w = make_layer_weights(g, wtype);

    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, 1);
    g.fill_f32(x);

    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 1);
    ((int32_t *) pos->data)[0] = (int32_t) KV_LEN;

    ggml_tensor * k_cache = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, HEAD_DIM, KV_LEN, N_HEAD);
    ggml_tensor * v_cache = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, KV_LEN, HEAD_DIM, N_HEAD);
    g.fill_f32(k_cache);
    g.fill_f32(v_cache);

    ggml_tensor * cur = x;
    for (int il = 0; il < DEEP_LAYERS; ++il) {
        cur = build_decode_layer(g, w, cur, pos, k_cache, v_cache);
    }
    ggml_build_forward_expand(g.gf, cur);

    graph_runner runner(g.gf, n_threads());
    for (auto _ : state) {
        runner.run(g.gf);
        benchmark::DoNotOptimize(cur->data);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_CAPTURE(decode_deep, f32,  GGML_TYPE_F32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(decode_deep, q4_k, GGML_TYPE_Q4_K)->Unit(benchmark::kMillisecond);

// LM head: projecting the last hidden states onto the vocabulary to produce
// logits — the single largest matrix multiplication of every decode step.
static void lm_head(benchmark::State & state, ggml_type wtype) {
    const size_t mem =
        (size_t) 512 * 1024 * 1024 + ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    graph_ctx g(mem);

    ggml_tensor * w_out = g.new_weight(wtype, N_EMBD, N_VOCAB);

    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, LM_TOKENS);
    g.fill_f32(x);

    ggml_tensor * logits = ggml_mul_mat(g.ctx, w_out, x); // [n_vocab, LM_TOKENS]
    ggml_build_forward_expand(g.gf, logits);

    graph_runner runner(g.gf, n_threads());
    for (auto _ : state) {
        runner.run(g.gf);
        benchmark::DoNotOptimize(logits->data);
    }

    state.SetItemsProcessed(state.iterations() * LM_TOKENS);
}
BENCHMARK_CAPTURE(lm_head, f16,  GGML_TYPE_F16)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(lm_head, q8_0, GGML_TYPE_Q8_0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(lm_head, q4_k, GGML_TYPE_Q4_K)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
