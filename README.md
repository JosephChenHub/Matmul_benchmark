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
For matmul operation : `C=AB` (`A:MxN, B: NxK`, the `GFlops` is given by
```
2*M*N*K * 1e-9 / run_time (second)
```

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
cpu-naive/8/manual_time               0.390 us        0.420 us      1869724
cpu-naive/16/manual_time               2.79 us         2.82 us       251394
cpu-naive/32/manual_time               25.7 us         25.8 us        26373
cpu-naive/64/manual_time                231 us          231 us         3109
cpu-naive/128/manual_time              2245 us         2244 us          312
cpu-naive/256/manual_time             20423 us        20423 us           34
cpu-naive/512/manual_time            188038 us       188030 us            4
cpu-naive/1024/manual_time          3238616 us      3238507 us            1
cpu-naive/2048/manual_time         46238192 us     46235348 us            1
cpu-naive/manual_time_BigO       17544211.35 N    17543149.19 N    
cpu-naive/manual_time_RMS               125 %           125 %    
cpu-trans/8/manual_time                15.4 us         15.5 us        45895
cpu-trans/16/manual_time               20.2 us         20.2 us        35292
cpu-trans/32/manual_time               58.4 us         58.5 us        11951
cpu-trans/64/manual_time                383 us          383 us         1830
cpu-trans/128/manual_time              3045 us         3045 us          230
cpu-trans/256/manual_time             22049 us        22049 us           33
cpu-trans/512/manual_time            168616 us       168610 us            4
cpu-trans/1024/manual_time          1370930 us      1370866 us            1
cpu-trans/2048/manual_time         11121680 us     11120935 us            1
cpu-trans/manual_time_BigO       4340443.12 N    4340158.15 N    
cpu-trans/manual_time_RMS               106 %           106 %    
cpu-trans-block/8/manual_time          16.5 us         16.6 us        42354
cpu-trans-block/16/manual_time         20.8 us         20.8 us        33758
cpu-trans-block/32/manual_time         54.2 us         54.4 us        12863
cpu-trans-block/64/manual_time          321 us          321 us         2180
cpu-trans-block/128/manual_time        2471 us         2471 us          283
cpu-trans-block/256/manual_time       19319 us        19003 us           46
cpu-trans-block/512/manual_time      109829 us       109826 us            6
cpu-trans-block/1024/manual_time     855524 us       855509 us            1
cpu-trans-block/2048/manual_time    6650865 us      6650436 us            1
cpu-trans-block/manual_time_BigO 2603280.58 N    2603106.16 N    
cpu-trans-block/manual_time_RMS         104 %           104 %    
cpu-openmp/8/manual_time               15.5 us         15.6 us        44942
cpu-openmp/16/manual_time              65.7 us         65.9 us        16021
cpu-openmp/32/manual_time              69.3 us         69.5 us         9717
cpu-openmp/64/manual_time              58.7 us         59.0 us        11457
cpu-openmp/128/manual_time              114 us          114 us         6135
cpu-openmp/256/manual_time              756 us          756 us          922
cpu-openmp/512/manual_time             6548 us         6548 us          104
cpu-openmp/1024/manual_time          236012 us       235362 us            3
cpu-openmp/2048/manual_time         1980963 us      1949639 us            1
cpu-openmp/manual_time_BigO       769306.31 N     757716.10 N    
cpu-openmp/manual_time_RMS              110 %           110 %    
cpu-omp-sse/8/manual_time              31.7 us         31.8 us        22098
cpu-omp-sse/16/manual_time             37.1 us         37.4 us        20792
cpu-omp-sse/32/manual_time             47.1 us         47.3 us        13938
cpu-omp-sse/64/manual_time             86.6 us         86.9 us         7909
cpu-omp-sse/128/manual_time             157 us          158 us         4360
cpu-omp-sse/256/manual_time             224 us          224 us         3147
cpu-omp-sse/512/manual_time            1284 us         1284 us          542
cpu-omp-sse/1024/manual_time           9541 us         9541 us           73
cpu-omp-sse/2048/manual_time          86432 us        86431 us            8
cpu-omp-sse/manual_time_BigO      180488.00 N     180422.83 N    
cpu-omp-sse/manual_time_RMS             123 %           123 %    
cpu-omp-avx/8/manual_time              34.6 us         34.7 us        22029
cpu-omp-avx/16/manual_time             33.7 us         34.1 us        20889
cpu-omp-avx/32/manual_time             45.4 us         45.6 us        15208
cpu-omp-avx/64/manual_time             85.0 us         85.3 us         8343
cpu-omp-avx/128/manual_time             160 us          160 us         4432
cpu-omp-avx/256/manual_time             178 us          179 us         3903
cpu-omp-avx/512/manual_time             758 us          758 us          921
cpu-omp-avx/1024/manual_time           5714 us         5714 us          123
cpu-omp-avx/2048/manual_time          63397 us        63395 us           11
cpu-omp-avx/manual_time_BigO      141489.89 N     141466.05 N    
cpu-omp-avx/manual_time_RMS             125 %           125 %    
cpu-omp-avx512/8/manual_time           31.8 us         31.8 us        22052
cpu-omp-avx512/16/manual_time          33.8 us         34.1 us        20735
cpu-omp-avx512/32/manual_time          50.8 us         51.2 us        13488
cpu-omp-avx512/64/manual_time           102 us          102 us         7010
cpu-omp-avx512/128/manual_time          199 us          199 us         3537
cpu-omp-avx512/256/manual_time          186 us          186 us         3843
cpu-omp-avx512/512/manual_time          874 us          874 us          799
cpu-omp-avx512/1024/manual_time        7238 us         7198 us           98
cpu-omp-avx512/2048/manual_time       63706 us        63702 us           11
cpu-omp-avx512/manual_time_BigO   180888.67 N     146896.16 N    
cpu-omp-avx512/manual_time_RMS          132 %           126 %    
cuda-naive/8/manual_time               6.89 us         6.92 us       100320
cuda-naive/16/manual_time              7.51 us         7.54 us        93253
cuda-naive/32/manual_time              8.42 us         8.45 us        71243
cuda-naive/64/manual_time              10.4 us         10.5 us        67177
cuda-naive/128/manual_time             15.0 us         15.0 us        46744
cuda-naive/256/manual_time             25.4 us         25.5 us        27522
cuda-naive/512/manual_time              162 us          162 us         4319
cuda-naive/1024/manual_time            1259 us         1259 us          559
cuda-naive/2048/manual_time           10636 us        10635 us           67
cuda-naive/4096/manual_time           85093 us        85013 us            8
cuda-naive/8192/manual_time          709016 us       708374 us            1
cuda-naive/manual_time_BigO        69066.48 N      69004.06 N    
cuda-naive/manual_time_RMS              119 %           119 %    
cuda-tile/8/manual_time                7.60 us         7.62 us        91717
cuda-tile/16/manual_time               7.63 us         7.66 us        91558
cuda-tile/32/manual_time               7.71 us         7.73 us        91013
cuda-tile/64/manual_time               9.39 us         9.42 us        74608
cuda-tile/128/manual_time              12.9 us         13.0 us        54036
cuda-tile/256/manual_time              22.8 us         22.8 us        30710
cuda-tile/512/manual_time               134 us          134 us         5238
cuda-tile/1024/manual_time             1022 us         1022 us          684
cuda-tile/2048/manual_time             8215 us         8215 us           85
cuda-tile/4096/manual_time            71314 us        71312 us           11
cuda-tile/8192/manual_time           568552 us       568528 us            1
cuda-tile/manual_time_BigO         55517.57 N      55515.30 N    
cuda-tile/manual_time_RMS               117 %           117 %    
cuda-unroll/8/manual_time              9.63 us         9.66 us        72503
cuda-unroll/16/manual_time             9.69 us         9.72 us        72353
cuda-unroll/32/manual_time             9.64 us         9.67 us        72704
cuda-unroll/64/manual_time             13.2 us         13.2 us        53160
cuda-unroll/128/manual_time            20.0 us         20.0 us        34973
cuda-unroll/256/manual_time            33.5 us         33.5 us        20899
cuda-unroll/512/manual_time            70.4 us         70.4 us         9929
cuda-unroll/1024/manual_time            521 us          521 us         1351
cuda-unroll/2048/manual_time           4152 us         4151 us          169
cuda-unroll/4096/manual_time          32437 us        32435 us           18
cuda-unroll/8192/manual_time         273728 us       273711 us            2
cuda-unroll/manual_time_BigO       26646.96 N      26645.32 N    
cuda-unroll/manual_time_RMS             119 %           119 %    
cuda-comopt/8/manual_time          10.9 us         11.0 us        53702
cuda-comopt/16/manual_time         10.4 us         10.5 us        66745
cuda-comopt/32/manual_time         14.0 us         14.0 us        50052
cuda-comopt/64/manual_time         21.0 us         21.1 us        33278
cuda-comopt/128/manual_time        36.0 us         36.0 us        19437
cuda-comopt/256/manual_time        65.0 us         65.0 us        10778
cuda-comopt/512/manual_time         123 us          123 us         5727
cuda-comopt/1024/manual_time        267 us          267 us         2626
cuda-comopt/2048/manual_time       1878 us         1878 us          379
cuda-comopt/4096/manual_time      14784 us        14783 us           43
cuda-comopt/8192/manual_time     117838 us       117833 us            5
cuda-comopt/manual_time_BigO   11512.14 N      11511.71 N    
cuda-comopt/manual_time_RMS         116 %           116 % 
cublas-sgemm/8/manual_time             8.34 us         8.37 us        84221
cublas-sgemm/16/manual_time            8.43 us         8.45 us        85636
cublas-sgemm/32/manual_time            10.3 us         10.3 us        66270
cublas-sgemm/64/manual_time            10.3 us         10.3 us        64935
cublas-sgemm/128/manual_time           11.1 us         11.2 us        62988
cublas-sgemm/256/manual_time           17.6 us         17.6 us        39835
cublas-sgemm/512/manual_time           33.4 us         33.4 us        20954
cublas-sgemm/1024/manual_time           182 us          182 us         3914
cublas-sgemm/2048/manual_time          1388 us         1388 us          513
cublas-sgemm/4096/manual_time         10981 us        10980 us           57
cublas-sgemm/8192/manual_time         83791 us        83782 us            7
cublas-sgemm/manual_time_BigO       8208.07 N       8207.22 N    
cublas-sgemm/manual_time_RMS            115 %           115 %    
```

![matmul_benchmark](figures/matmul_benchmark.png)

