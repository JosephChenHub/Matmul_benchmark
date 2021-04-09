# Matrix Multiplication Optimization Methods

This is a toy benchmark experiment of several matrix multiplication algorithms. 

## Requirements
- googletest
- google/benchmark
- openmp
- CUDA

## Algorithms
- Naive: the pure method for calculating `C = A * B`, the time complexity is `O(N^3)`
```
for(int i = 0; i < A.rows; ++i) {
    for(int j = 0; j < B.cols; ++j) {
        C[i][j] = 0;
        for(int k = 0; k < A.cols; ++k) {
            C[i][j] += A[i][k] * B[k][i];
        }
    }
}
```
- Naive-Trans:
in `Naive`, the access of `B[:][i]` is not cache-friendly, and we can transpose `B` (cost `O(N^2)`) before doing matmul, 
```
// Bt = B.transpose()
for(int i = 0; i < B.rows; ++i) {
    for(int j = 0; j < B.cols; ++j) {
        Bt[j][i] = B[i][j];
    }
}
// C = A * B
for(int i = 0; i < A.rows; ++i) {
    for(int j = 0; j < B.cols; ++j) {
        C[i][j] = 0;
        for(int k = 0; k < A.cols; ++k) {
            C[i][j] += A[i][k] * Bt[i][k]; // cache friendly 
        }
    }
}
```
- Using OpenMP : a simple method via multi-cores

- Using SIMD (CUDA) : naive method, shared memory based method

- Using SIMD (CPU): Intel SSE, AVX instructions, etc.

- Strassen (Divide and Conquer): 
1. naive
```
A = [a, b; c, d] B = [e, f; g, h]  C=A*B=[ae +bg, af+bh; ce+dg, cf+dh]
```
which needs 8 multiplications and 4 additions. the time complexity 
```

T(n) = 8T(n/2) + O(n^2)
T(n) = O(n^log2(8)) = O(n^3)
```
2. strassen 
```
A = [a, b; c, d] B = [e, f; g, h]  C=A*B
p1 = a(f-h), p2 = (a+b)h
p3 = (c+d)e, p4 = d(g-e)
p5 = (a+d)(e+h), p6 = (b-d)(g+h)
p7 = (a-c)(e+f)

C = [p5+p4-p2+p6, p1+p2; p3+p4, p1+p5-p3-p7]
```
```
T(n) = 7T(n/2) + O(n^2)
T(n) = O(n^log2(7)) ~ O(n^2.8074)
```


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
Load Average: 0.75, 4.20, 2.50
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
cpu-naive/8/manual_time               0.388 us        0.418 us      1816080
cpu-naive/16/manual_time               2.79 us         2.82 us       246384
cpu-naive/32/manual_time               25.7 us         25.7 us        26714
cpu-naive/64/manual_time                230 us          230 us         2918
cpu-naive/128/manual_time              2286 us         2286 us          303
cpu-naive/256/manual_time             20434 us        20433 us           35
cpu-naive/512/manual_time            184615 us       184608 us            4
cpu-naive/1024/manual_time          3246285 us      3246097 us            1
cpu-naive/2048/manual_time         43417450 us     43414172 us            1
cpu-naive/manual_time_BigO       16512312.92 N    16511077.40 N    
cpu-naive/manual_time_RMS               123 %           123 %    
cpu-trans/8/manual_time                14.2 us         14.3 us        48495
cpu-trans/16/manual_time               19.0 us         19.1 us        29552
cpu-trans/32/manual_time               56.8 us         57.0 us        12305
cpu-trans/64/manual_time                381 us          381 us         1804
cpu-trans/128/manual_time              3230 us         3208 us          215
cpu-trans/256/manual_time             22208 us        22055 us           33
cpu-trans/512/manual_time            168722 us       168712 us            4
cpu-trans/1024/manual_time          1368736 us      1368644 us            1
cpu-trans/2048/manual_time         11090561 us     11089758 us            1
cpu-trans/manual_time_BigO       4328666.32 N    4328347.12 N    
cpu-trans/manual_time_RMS               106 %           106 %    
cpu-trans-block/8/manual_time          15.7 us         15.8 us        44733
cpu-trans-block/16/manual_time         20.3 us         20.4 us        28145
cpu-trans-block/32/manual_time         53.1 us         53.2 us        13175
cpu-trans-block/64/manual_time          320 us          320 us         1845
cpu-trans-block/128/manual_time        3059 us         3023 us          283
cpu-trans-block/256/manual_time       15324 us        15324 us           46
cpu-trans-block/512/manual_time      108463 us       108461 us            6
cpu-trans-block/1024/manual_time     855927 us       855911 us            1
cpu-trans-block/2048/manual_time    6917796 us      6917427 us            1
cpu-trans-block/manual_time_BigO 2700813.26 N    2700674.24 N    
cpu-trans-block/manual_time_RMS         106 %           106 %    
cpu-openmp/8/manual_time               14.5 us         15.0 us        48115
cpu-openmp/16/manual_time              42.8 us         43.3 us        16331
cpu-openmp/32/manual_time               122 us          122 us         5732
cpu-openmp/64/manual_time              46.4 us         46.9 us        13779
cpu-openmp/128/manual_time              131 us          132 us         5341
cpu-openmp/256/manual_time              784 us          785 us          888
cpu-openmp/512/manual_time             6514 us         6514 us          106
cpu-openmp/1024/manual_time          240668 us       240661 us            3
cpu-openmp/2048/manual_time         1980150 us      1919741 us            1
cpu-openmp/manual_time_BigO       769859.85 N     747735.79 N    
cpu-openmp/manual_time_RMS              110 %           109 %    
cpu-omp-sse/8/manual_time              29.6 us         29.7 us        23561
cpu-omp-sse/16/manual_time             32.8 us         33.0 us        17798
cpu-omp-sse/32/manual_time             45.8 us         46.0 us        15213
cpu-omp-sse/64/manual_time             85.4 us         85.7 us         8423
cpu-omp-sse/128/manual_time             151 us          151 us         4608
cpu-omp-sse/256/manual_time             219 us          219 us         3199
cpu-omp-sse/512/manual_time            1268 us         1268 us          552
cpu-omp-sse/1024/manual_time           9430 us         9429 us           74
cpu-omp-sse/2048/manual_time         109966 us        88884 us            6
cpu-omp-sse/4096/manual_time         779271 us       779244 us            1
cpu-omp-sse/manual_time_BigO      153221.06 N     151285.94 N    
cpu-omp-sse/manual_time_RMS             108 %           116 %    
cpu-omp-avx/8/manual_time              30.9 us         31.0 us        20176
cpu-omp-avx/16/manual_time             34.2 us         34.4 us        18321
cpu-omp-avx/32/manual_time             46.7 us         46.8 us        13417
cpu-omp-avx/64/manual_time             88.2 us         88.5 us         7982
cpu-omp-avx/128/manual_time             176 us          176 us         3909
cpu-omp-avx/256/manual_time             172 us          172 us         4105
cpu-omp-avx/512/manual_time             761 us          761 us          925
cpu-omp-avx/1024/manual_time           7904 us         7583 us          121
cpu-omp-avx/2048/manual_time          72523 us        72517 us           11
cpu-omp-avx/4096/manual_time         757608 us       757583 us            1
cpu-omp-avx/manual_time_BigO      145744.49 N     145724.66 N    
cpu-omp-avx/manual_time_RMS             122 %           122 %    
cpu-omp-avx512/8/manual_time           30.0 us         30.1 us        20418
cpu-omp-avx512/16/manual_time          34.3 us         34.5 us        18607
cpu-omp-avx512/32/manual_time          52.7 us         53.0 us        13019
cpu-omp-avx512/64/manual_time           107 us          107 us         6643
cpu-omp-avx512/128/manual_time          207 us          207 us         2892
cpu-omp-avx512/256/manual_time          183 us          183 us         3861
cpu-omp-avx512/512/manual_time          994 us          989 us          792
cpu-omp-avx512/1024/manual_time        6948 us         6947 us          100
cpu-omp-avx512/2048/manual_time       63709 us        63704 us           11
cpu-omp-avx512/4096/manual_time      779112 us       779040 us            1
cpu-omp-avx512/manual_time_RMS          126 %           126 %    
cuda-naive/8/manual_time               6.55 us         6.57 us       106011
cuda-naive/16/manual_time              7.02 us         7.04 us        97804
cuda-naive/32/manual_time              7.98 us         8.01 us        75329
cuda-naive/64/manual_time              10.1 us         10.1 us        68300
cuda-naive/128/manual_time             14.5 us         14.6 us        48024
cuda-naive/256/manual_time             25.0 us         25.0 us        27947
cuda-naive/512/manual_time              161 us          161 us         4362
cuda-naive/1024/manual_time            1250 us         1250 us          562
cuda-naive/2048/manual_time           10571 us        10571 us           67
cuda-naive/4096/manual_time           84485 us        84483 us            8
cuda-naive/8192/manual_time          702178 us       702151 us            1
cuda-naive/manual_time_BigO        68411.08 N      68408.48 N    
cuda-naive/manual_time_RMS              119 %           119 %    
cuda-tile/8/manual_time                7.18 us         7.20 us        95355
cuda-tile/16/manual_time               7.25 us         7.27 us        94390
cuda-tile/32/manual_time               7.37 us         7.40 us        96173
cuda-tile/64/manual_time               9.06 us         9.08 us        76692
cuda-tile/128/manual_time              12.6 us         12.6 us        55037
cuda-tile/256/manual_time              22.4 us         22.4 us        31162
cuda-tile/512/manual_time               133 us          133 us         5251
cuda-tile/1024/manual_time             1016 us         1016 us          690
cuda-tile/2048/manual_time             8188 us         8187 us           85
cuda-tile/4096/manual_time            65086 us        65083 us           11
cuda-tile/8192/manual_time           573187 us       573120 us            1
cuda-tile/manual_time_BigO         55656.10 N      55649.83 N    
cuda-tile/manual_time_RMS               121 %           121 %    
cuda-unroll/8/manual_time              8.85 us         8.88 us        77857
cuda-unroll/16/manual_time             8.87 us         8.90 us        77694
cuda-unroll/32/manual_time             8.88 us         8.91 us        78817
cuda-unroll/64/manual_time             12.4 us         12.4 us        56070
cuda-unroll/128/manual_time            19.2 us         19.3 us        36495
cuda-unroll/256/manual_time            32.6 us         32.7 us        21448
cuda-unroll/512/manual_time            69.6 us         69.6 us        10043
cuda-unroll/1024/manual_time            517 us          517 us         1360
cuda-unroll/2048/manual_time           4102 us         4102 us          170
cuda-unroll/4096/manual_time          32010 us        32007 us           19
cuda-unroll/8192/manual_time         275340 us       275314 us            2
cuda-unroll/manual_time_BigO       26773.73 N      26771.26 N    
cuda-unroll/manual_time_RMS             120 %           120 %    
cublas-sgemm/8/manual_time             7.29 us         7.31 us        95903
cublas-sgemm/16/manual_time            7.19 us         7.22 us        95263
cublas-sgemm/32/manual_time            8.57 us         8.60 us        76672
cublas-sgemm/64/manual_time            9.02 us         9.04 us        7713
cublas-sgemm/128/manual_time           9.94 us         9.96 us        70510
cublas-sgemm/256/manual_time           15.9 us         15.9 us        43980
cublas-sgemm/512/manual_time           32.1 us         32.1 us        21864
cublas-sgemm/1024/manual_time           179 us          179 us         3986
cublas-sgemm/2048/manual_time          1378 us         1378 us          518
cublas-sgemm/4096/manual_time         10903 us        10902 us           60
cublas-sgemm/8192/manual_time         85924 us        85919 us            7
cublas-sgemm/manual_time_BigO       8399.52 N       8399.04 N    
cublas-sgemm/manual_time_RMS            116 %           116 %    
```

![matmul_benchmark](figures/matmul_benchmark.png)

