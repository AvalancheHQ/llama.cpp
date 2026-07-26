# CodSpeed performance benchmarks for llama.cpp
#
# This is a self-contained CMake project that builds only the `ggml` library
# (the performance-critical compute core of llama.cpp) together with Google
# Benchmark. It intentionally avoids building the full llama.cpp stack so the
# benchmarks build quickly and require no model files.
#
# Build:
#   cmake -B build -S benchmarks -DCODSPEED_MODE=simulation
#   cmake --build build -j
#
# Run (via the CodSpeed CLI):
#   codspeed run --mode simulation -- ./build/ggml-benchmarks

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

target_link_libraries(ggml-benchmarks PRIVATE ggml benchmark::benchmark)
