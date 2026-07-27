# CodSpeed performance benchmarks for llama.cpp
#
# This is a self-contained CMake project that builds only the `ggml` library
# (the performance-critical compute core of llama.cpp) together with Google
# Benchmark. It intentionally avoids building the full llama.cpp stack so the
# benchmarks build quickly and require no model files.
#
# Two benchmark executables are produced:
#   - ggml-benchmarks:       micro benchmarks (kernels and small graphs), run
#                            in CodSpeed simulation mode.
#   - ggml-macro-benchmarks: macro benchmarks (full transformer-layer graphs),
#                            run in CodSpeed walltime mode on macro runners.
#
# Build:
#   cmake -B build -S benchmarks -DCODSPEED_MODE=<simulation|walltime>
#   cmake --build build -j
#
# Run (via the CodSpeed CLI):
#   codspeed run --mode simulation -- ./build/ggml-benchmarks
#   codspeed run --mode walltime   -- ./build/ggml-macro-benchmarks

cmake_minimum_required(VERSION 3.14)

project("llama-cpp-benchmarks" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable release mode with debug symbols so CodSpeed reports include useful
# profiling data (source-level flamegraphs).
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Build everything (ggml, Google Benchmark and the CodSpeed core) as static
# libraries. This keeps the benchmark executable self-contained and avoids
# shared-library symbol resolution issues between the CodSpeed core and the
# google_benchmark compatibility layer.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# The CodSpeed simulation instrument runs the benchmark binary under Valgrind,
# which does not implement every native CPU instruction (notably AVX-512). By
# default ggml compiles the CPU backend with `-march=native`, which on modern
# CI/build hosts emits instructions Valgrind cannot execute, causing the graph
# compute benchmarks to abort with SIGILL. Disable the native build and instead
# target a conservative x86 baseline (AVX2 / FMA / F16C) that Valgrind supports
# while still exercising the vectorized kernels. On non-x86 hosts these AVX
# options are simply ignored and the portable code paths are used.
#
# This constraint only applies to simulation builds: walltime builds execute
# natively on real hardware (CodSpeed macro runners), so they keep ggml's
# default `-march=native` for realistic measurements.
if (CODSPEED_MODE STREQUAL "simulation")
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_AVX    ON  CACHE BOOL "" FORCE)
    set(GGML_AVX2   ON  CACHE BOOL "" FORCE)
    set(GGML_FMA    ON  CACHE BOOL "" FORCE)
    set(GGML_F16C   ON  CACHE BOOL "" FORCE)
endif()

# Build the ggml compute core shipped in this repository.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ggml ${CMAKE_BINARY_DIR}/ggml)

# Fetch CodSpeed's google_benchmark compatibility layer. It behaves exactly like
# upstream Google Benchmark but, when configured with -DCODSPEED_MODE, emits the
# instrumentation CodSpeed needs to measure each benchmark.
include(FetchContent)

set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp
    SOURCE_SUBDIR google_benchmark
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(google_benchmark)

add_executable(ggml-benchmarks bench_ggml.cpp)
add_executable(ggml-macro-benchmarks bench_macro.cpp)

# Link against the ggml meta-target (which pulls in the compute core and the CPU
# backend) plus ggml-cpu explicitly. The compute-graph benchmarks evaluate
# graphs through the CPU backend's ggml_graph_plan()/ggml_graph_compute(), so
# ggml-cpu must be linked directly to guarantee its symbols are resolved in a
# fully static build.
target_link_libraries(ggml-benchmarks       PRIVATE ggml ggml-cpu benchmark::benchmark)
target_link_libraries(ggml-macro-benchmarks PRIVATE ggml ggml-cpu benchmark::benchmark)
