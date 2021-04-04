#pragma once



template <typename Dtype>
void matmul_naive_kernel(const Dtype*, const Dtype* , Dtype*, 
        const int, const int, const int );

template <typename Dtype>
void matmul_shared_kernel(const Dtype*, const Dtype* , Dtype*, 
        const int, const int, const int );
