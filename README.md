# Matrix Multiplication Optimization Methods
This is a toy benchmark experiment of several matrix multiplication algorithms. 

## Requirements
- googletest
- google/benchmark
- openmp
- CUDA

## Algorithms
- Naive
- Using OpenMP
- Using SIMD (CUDA)
- Strassen 


## Benchmark
run the benchmark via the executable `bench`, an example log : 
```
Running ./bench
Run on (48 X 3500 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x24)
  L1 Instruction 32 KiB (x24)
  L2 Unified 1024 KiB (x24)
  L3 Unified 16896 KiB (x2)
Load Average: 7.02, 6.38, 3.37
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
----------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations
----------------------------------------------------------------------
CPU-Naive/8/manual_time          0.000 ms        0.000 ms      1889971
CPU-Naive/16/manual_time         0.003 ms        0.003 ms       250962
CPU-Naive/32/manual_time         0.024 ms        0.024 ms        28661
CPU-Naive/64/manual_time         0.227 ms        0.227 ms         3089
CPU-Naive/128/manual_time         2.10 ms         2.10 ms          334
CPU-Naive/256/manual_time         18.2 ms         18.2 ms           38
CPU-Naive/512/manual_time          178 ms          178 ms            4
CPU-Naive/1024/manual_time        3092 ms         3091 ms            1
CPU-Naive/2048/manual_time       40942 ms        40934 ms            1
CPU-Naive/manual_time_BigO  15576951.06 N    15573943.02 N    
CPU-Naive/manual_time_RMS          123 %           123 %    
CPU-OpenMP/8/manual_time         0.011 ms        0.011 ms        63266
CPU-OpenMP/16/manual_time        0.026 ms        0.026 ms        31941
CPU-OpenMP/32/manual_time        0.013 ms        0.013 ms        52791
CPU-OpenMP/64/manual_time        0.021 ms        0.021 ms        32861
CPU-OpenMP/128/manual_time       0.085 ms        0.085 ms         8659
CPU-OpenMP/256/manual_time       0.636 ms        0.636 ms         1065
CPU-OpenMP/512/manual_time        7.08 ms         7.01 ms           84
CPU-OpenMP/1024/manual_time        218 ms          213 ms            3
CPU-OpenMP/2048/manual_time       2185 ms         2130 ms            1
CPU-OpenMP/4096/manual_time      41787 ms        39130 ms            1
CPU-OpenMP/manual_time_BigO 7861624.83 N    7369791.82 N    
CPU-OpenMP/manual_time_RMS         137 %           137 %    
GPU-Naive/8/manual_time          0.006 ms        0.006 ms       108625
GPU-Naive/16/manual_time         0.007 ms        0.007 ms       103374
GPU-Naive/32/manual_time         0.007 ms        0.007 ms        93742
GPU-Naive/64/manual_time         0.007 ms        0.007 ms        87780
GPU-Naive/128/manual_time        0.010 ms        0.010 ms        72506
GPU-Naive/256/manual_time        0.034 ms        0.034 ms        20779
GPU-Naive/512/manual_time        0.221 ms        0.221 ms         3175
GPU-Naive/1024/manual_time        1.67 ms         1.67 ms          422
GPU-Naive/2048/manual_time        14.4 ms         14.4 ms           49
GPU-Naive/4096/manual_time         115 ms          115 ms            6
GPU-Naive/8192/manual_time         950 ms          950 ms            1
GPU-Naive/manual_time_BigO    92642.14 N      92635.99 N    
GPU-Naive/manual_time_RMS          118 %           118 %    
GPU-Shared/8/manual_time         0.006 ms        0.006 ms       120449
GPU-Shared/16/manual_time        0.006 ms        0.006 ms       112922
GPU-Shared/32/manual_time        0.007 ms        0.007 ms        96647
GPU-Shared/64/manual_time        0.009 ms        0.009 ms        77716
GPU-Shared/128/manual_time       0.012 ms        0.012 ms        56195
GPU-Shared/256/manual_time       0.022 ms        0.022 ms        31654
GPU-Shared/512/manual_time       0.132 ms        0.132 ms         5334
GPU-Shared/1024/manual_time       1.00 ms         1.00 ms          698
GPU-Shared/2048/manual_time       8.03 ms         8.03 ms           87
GPU-Shared/4096/manual_time       70.3 ms         70.3 ms           11
GPU-Shared/8192/manual_time        547 ms          547 ms            1
GPU-Shared/manual_time_BigO   53486.84 N      53484.40 N    
GPU-Shared/manual_time_RMS         116 %           116 %   
```

![matmul_benchmark](figures/matmul_benchmark.png)

