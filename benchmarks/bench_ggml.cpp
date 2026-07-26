// CodSpeed performance benchmarks for the ggml compute core of llama.cpp.
//
// These benchmarks exercise CPU-bound, model-free hot paths that are central to
// llama.cpp inference and model conversion. They fall into two groups:
//
//   1. Low-level kernels invoked directly:
//        - quantization of fp32 weights into the various ggml block formats
//          (ggml_quantize_chunk), and
//        - fp32 <-> fp16 row conversions (ggml_fp32_to_fp16_row /
//          ggml_fp16_to_fp32_row).
//
//   2. Full ggml compute graphs evaluated on the CPU backend, mirroring the
//      operators that dominate transformer inference:
//        - matrix multiplication (F32 and quantized weights, i.e. the bulk of
//          every linear/projection layer),
//        - scaled masked soft-max (the attention score normalization),
//        - rotary position embedding (RoPE),
//        - RMS normalization, and
//        - a SwiGLU feed-forward block combining several of the above.
//
// None of these require model files, and each runs in well under a second,
// which makes them a good fit for deterministic CPU-simulation measurement with
// CodSpeed.

#include "ggml.h"
#include "ggml-cpu.h"

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Low-level kernel benchmarks
// ---------------------------------------------------------------------------

// Number of elements per row. This is a multiple of the largest ggml block
// size so every quantization type can be exercised without padding issues.
static constexpr int64_t N_PER_ROW = 4096;
static constexpr int64_t N_ROWS    = 8;

// Deterministic synthetic weight data resembling a normalized weight matrix.
static std::vector<float> make_input(int64_t n) {
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = 0.1f + 2.0f * std::cos(static_cast<float>(i));
    }
    return data;
}

// Benchmark quantization of a chunk of fp32 data into a given ggml type.
static void quantize_chunk(benchmark::State & state, ggml_type type) {
    const int64_t n = N_ROWS * N_PER_ROW;
    const std::vector<float> src = make_input(n);

    // Worst-case output buffer: as large as the input in bytes.
    std::vector<uint8_t> dst(n * sizeof(float));

    ggml_quantize_init(type);

    size_t written = 0;
    for (auto _ : state) {
        written = ggml_quantize_chunk(
            type, src.data(), dst.data(), /*start=*/0, N_ROWS, N_PER_ROW, /*imatrix=*/nullptr);
        benchmark::DoNotOptimize(written);
        benchmark::DoNotOptimize(dst.data());
    }

    ggml_quantize_free();

    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK_CAPTURE(quantize_chunk, q4_0, GGML_TYPE_Q4_0);
BENCHMARK_CAPTURE(quantize_chunk, q4_1, GGML_TYPE_Q4_1);
BENCHMARK_CAPTURE(quantize_chunk, q5_0, GGML_TYPE_Q5_0);
BENCHMARK_CAPTURE(quantize_chunk, q8_0, GGML_TYPE_Q8_0);
BENCHMARK_CAPTURE(quantize_chunk, q4_k, GGML_TYPE_Q4_K);
BENCHMARK_CAPTURE(quantize_chunk, q6_k, GGML_TYPE_Q6_K);

// Benchmark fp32 -> fp16 row conversion.
static void fp32_to_fp16_row(benchmark::State & state) {
    const int64_t n = N_ROWS * N_PER_ROW;
    const std::vector<float> src = make_input(n);
    std::vector<ggml_fp16_t> dst(n);

    for (auto _ : state) {
        ggml_fp32_to_fp16_row(src.data(), dst.data(), n);
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(fp32_to_fp16_row);

// Benchmark fp16 -> fp32 row conversion.
static void fp16_to_fp32_row(benchmark::State & state) {
    const int64_t n = N_ROWS * N_PER_ROW;
    const std::vector<float> src_f32 = make_input(n);

    std::vector<ggml_fp16_t> src(n);
    ggml_fp32_to_fp16_row(src_f32.data(), src.data(), n);

    std::vector<float> dst(n);

    for (auto _ : state) {
        ggml_fp16_to_fp32_row(src.data(), dst.data(), n);
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(fp16_to_fp32_row);

// ---------------------------------------------------------------------------
// Compute-graph benchmarks (CPU backend)
//
// Each of these builds a small ggml graph once, fills its leaf tensors with
// deterministic data, and then repeatedly evaluates the graph on the CPU. This
// is the same execution path used by llama.cpp inference, so the measurements
// reflect the real operator kernels rather than isolated helpers.
// ---------------------------------------------------------------------------

// A self-contained graph context: owns the tensor arena, a reusable work
// buffer, and the graph. Tensors created via `ctx` allocate their data inline.
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

    // Finalize the graph (marking `out` as the output) and run it once with the
    // given number of threads, using a stack-local plan/work buffer.
    void compute(ggml_tensor * out, int n_threads) {
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    }
};

// Shared problem sizes, roughly matching a small-model decode step.
static constexpr int64_t N_EMBD  = 2048;  // hidden size
static constexpr int64_t N_FF    = 5632;  // feed-forward inner size
static constexpr int64_t N_TOKEN = 32;    // sequence length / batch
static constexpr int64_t N_HEAD  = 16;    // attention heads
static constexpr int64_t N_KV    = 256;   // KV-cache length
static constexpr int     N_THREADS = 1;   // deterministic single-threaded runs

// Matrix multiplication: weight [N_EMBD x N_FF] times activations
// [N_EMBD x N_TOKEN] -> [N_FF x N_TOKEN]. This is the shape of a linear
// projection and is by far the dominant cost in transformer inference.
static void mul_mat(benchmark::State & state, ggml_type wtype) {
    const size_t mem =
        (size_t) 512 * 1024 * 1024 + ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    graph_ctx g(mem);

    // Build the fp32 weights, then optionally quantize to `wtype`.
    ggml_tensor * w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, N_FF);
    g.fill_f32(w);
    if (wtype != GGML_TYPE_F32) {
        ggml_tensor * wq = ggml_new_tensor_2d(g.ctx, wtype, N_EMBD, N_FF);
        ggml_quantize_init(wtype);
        ggml_quantize_chunk(wtype, (const float *) w->data, wq->data,
                            0, N_FF, N_EMBD, nullptr);
        ggml_quantize_free();
        w = wq;
    }

    ggml_tensor * x   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, N_TOKEN);
    g.fill_f32(x);

    ggml_tensor * out = ggml_mul_mat(g.ctx, w, x);
    ggml_build_forward_expand(g.gf, out);

    for (auto _ : state) {
        ggml_graph_compute_with_ctx(g.ctx, g.gf, N_THREADS);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations() * N_FF * N_TOKEN * N_EMBD);
}
BENCHMARK_CAPTURE(mul_mat, f32,  GGML_TYPE_F32);
BENCHMARK_CAPTURE(mul_mat, q4_0, GGML_TYPE_Q4_0);
BENCHMARK_CAPTURE(mul_mat, q4_k, GGML_TYPE_Q4_K);

// Scaled, masked soft-max over attention scores of shape
// [N_KV x N_TOKEN x N_HEAD] with a causal mask. This is the normalization step
// at the heart of scaled dot-product attention.
static void soft_max(benchmark::State & state) {
    const size_t mem =
        (size_t) 256 * 1024 * 1024 + ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    graph_ctx g(mem);

    ggml_tensor * scores = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, N_KV, N_TOKEN, N_HEAD);
    g.fill_f32(scores);

    ggml_tensor * mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_KV, N_TOKEN);
    g.fill_f32(mask);

    const float scale = 1.0f / std::sqrt((float) (N_EMBD / N_HEAD));
    ggml_tensor * out = ggml_soft_max_ext(g.ctx, scores, mask, scale, /*max_bias=*/0.0f);
    ggml_build_forward_expand(g.gf, out);

    for (auto _ : state) {
        ggml_graph_compute_with_ctx(g.ctx, g.gf, N_THREADS);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations() * N_KV * N_TOKEN * N_HEAD);
}
BENCHMARK(soft_max);

// Rotary position embedding (NeoX style) applied to a [head_dim x n_head x
// n_token] tensor. RoPE runs on the query and key projections of every layer.
static void rope(benchmark::State & state) {
    const int64_t head_dim = N_EMBD / N_HEAD;

    const size_t mem =
        (size_t) 128 * 1024 * 1024 + ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    graph_ctx g(mem);

    ggml_tensor * cur = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, head_dim, N_HEAD, N_TOKEN);
    g.fill_f32(cur);

    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, N_TOKEN);
    for (int64_t i = 0; i < N_TOKEN; ++i) {
        ((int32_t *) pos->data)[i] = (int32_t) i;
    }

    ggml_tensor * out = ggml_rope_ext(
        g.ctx, cur, pos, /*freq_factors=*/nullptr,
        (int) head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/2048,
        /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
        /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
        /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    ggml_build_forward_expand(g.gf, out);

    for (auto _ : state) {
        ggml_graph_compute_with_ctx(g.ctx, g.gf, N_THREADS);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations() * head_dim * N_HEAD * N_TOKEN);
}
BENCHMARK(rope);

// RMS normalization over [N_EMBD x N_TOKEN] followed by an element-wise scale,
// exactly as used before each attention and feed-forward block.
static void rms_norm(benchmark::State & state) {
    const size_t mem =
        (size_t) 64 * 1024 * 1024 + ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    graph_ctx g(mem);

    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, N_TOKEN);
    g.fill_f32(x);

    ggml_tensor * w = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, N_EMBD);
    g.fill_f32(w);

    ggml_tensor * out = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, 1e-5f), w);
    ggml_build_forward_expand(g.gf, out);

    for (auto _ : state) {
        ggml_graph_compute_with_ctx(g.ctx, g.gf, N_THREADS);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations() * N_EMBD * N_TOKEN);
}
BENCHMARK(rms_norm);

// A full SwiGLU feed-forward block: two parallel gate/up projections, a SiLU
// gate, an element-wise product and a down projection. This chains several
// operators the way a real transformer layer does, exercising the graph
// scheduler in addition to the individual kernels.
static void ffn_swiglu(benchmark::State & state) {
    const size_t mem =
        (size_t) 512 * 1024 * 1024 + ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    graph_ctx g(mem);

    ggml_tensor * w_gate = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, N_FF);
    ggml_tensor * w_up   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, N_FF);
    ggml_tensor * w_down = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_FF,   N_EMBD);
    ggml_tensor * x      = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, N_TOKEN);
    g.fill_f32(w_gate);
    g.fill_f32(w_up);
    g.fill_f32(w_down);
    g.fill_f32(x);

    ggml_tensor * gate = ggml_silu(g.ctx, ggml_mul_mat(g.ctx, w_gate, x));
    ggml_tensor * up   = ggml_mul_mat(g.ctx, w_up, x);
    ggml_tensor * out  = ggml_mul_mat(g.ctx, w_down, ggml_mul(g.ctx, gate, up));
    ggml_build_forward_expand(g.gf, out);

    for (auto _ : state) {
        ggml_graph_compute_with_ctx(g.ctx, g.gf, N_THREADS);
        benchmark::DoNotOptimize(out->data);
    }

    state.SetItemsProcessed(state.iterations() * N_EMBD * N_TOKEN);
}
BENCHMARK(ffn_swiglu);

BENCHMARK_MAIN();
