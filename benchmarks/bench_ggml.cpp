// CodSpeed performance benchmarks for the ggml compute core of llama.cpp.
//
// These benchmarks exercise CPU-bound, model-free hot paths that are central to
// llama.cpp inference and model conversion:
//   - quantization of fp32 weights into the various ggml block formats
//     (ggml_quantize_chunk), and
//   - fp32 <-> fp16 row conversions (ggml_fp32_to_fp16_row / ggml_fp16_to_fp32_row).
//
// They require no model files and run in well under a second each, which makes
// them a good fit for deterministic CPU-simulation measurement with CodSpeed.

#include "ggml.h"

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <vector>

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

BENCHMARK_MAIN();
