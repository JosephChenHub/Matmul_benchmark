#pragma once



void cublas_sgemm(const float* d_A, const float *d_B, float* d_C, const int A_ROW, const int A_COL, const int B_COL);

template <typename Dtype>
void matmul_naive_kernel(const Dtype*, const Dtype* , Dtype*, 
        const int, const int, const int );

template <typename Dtype>
void matmul_tile_kernel(const Dtype*, const Dtype* , Dtype*, 
        const int, const int, const int );

template <typename Dtype>
void matmul_unroll_kernel(const Dtype*, const Dtype* , Dtype*, 
        const int, const int, const int );


//template <typename Dtype>
//void matmul_vectorize_kernel(const Dtype*, const Dtype*, Dtype*,
//        const int, const int, const int );
