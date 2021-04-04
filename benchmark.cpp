#include <benchmark/benchmark.h>

#include "matmul.hpp"
#include <iostream>



static void BM_MatmulNaive(benchmark::State& state) {
    const int n = state.range(0);
//    std::cout << "N:" << n << std::endl;
    auto A = Matrix<float>::randn(n, n);
    auto B = Matrix<float>::randn(n, n);
    auto C = Matrix<float>::zeros(n, n);
  
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_naive(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                    end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetComplexityN(state.range(0));
}

static void BM_MatmulOpenMP(benchmark::State& state) {
    const int n = state.range(0);
    auto A = Matrix<float>::randn(n, n);
    auto B = Matrix<float>::randn(n, n);
    auto C = Matrix<float>::zeros(n, n);
  
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                    end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetComplexityN(state.range(0));
}

static void BM_MatmulCUDANaive(benchmark::State& state) {
    const int n = state.range(0);
    auto A = Matrix<float>::randn(n, n).cuda();
    auto B = Matrix<float>::randn(n, n).cuda();
    auto C = Matrix<float>::zeros(n, n).cuda();
  
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_cuda_naive(A, B, C);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                    end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetComplexityN(state.range(0));
}

static void BM_MatmulCUDAShared(benchmark::State& state) {
    const int n = state.range(0);
    auto A = Matrix<float>::randn(n, n).cuda();
    auto B = Matrix<float>::randn(n, n).cuda();
    auto C = Matrix<float>::zeros(n, n).cuda();
  
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_cuda_shared(A, B, C);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                    end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetComplexityN(state.range(0));
}


BENCHMARK(BM_MatmulNaive)->Name("CPU-Naive")
            ->RangeMultiplier(2)->Range(8, 8<<8) //8, 16, 32, ..., 2k
            ->Complexity(benchmark::oN) 
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatmulOpenMP)->Name("CPU-OpenMP")
            ->RangeMultiplier(2)->Range(8, 8<<9) //8, 16, 32, ..., 2k, 4k
            ->Complexity(benchmark::oN) 
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatmulCUDANaive)->Name("GPU-Naive")
            ->RangeMultiplier(2)->Range(8, 8<<10) //8, 16, 32, ..., 2k, 4k, 8k
            ->Complexity(benchmark::oN) 
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatmulCUDAShared)->Name("GPU-Shared")
            ->RangeMultiplier(2)->Range(8, 8<<10) //8, 16, 32, ..., 2k, 4k, 8k
            ->Complexity(benchmark::oN) 
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
