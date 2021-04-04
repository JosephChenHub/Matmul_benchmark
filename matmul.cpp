#include <iostream>

#include "matmul.hpp"


template <typename Dtype>
void matmul_naive(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    assert (A.rows() == C.rows() && A.cols() == B.rows() && B.cols() == C.cols());
    for(int i = 0; i < C.rows(); ++i) {
        for(int j = 0; j < C.cols(); ++j) {
            Dtype tmp(0);
            for(int k = 0; k < A.cols(); ++k) {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
        }
    }
}


template <typename Dtype>
void matmul_openmp(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    assert (A.rows() == C.rows() && A.cols() == B.rows() && B.cols() == C.cols());
    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for(i = 0; i < C.rows(); ++i) {
        for(j = 0; j < C.cols(); ++j) {
            Dtype tmp(0);
            for(k = 0; k < A.cols(); ++k) {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
        }
    }
}




template <typename Dtype>
void matmul_cuda_naive(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    matmul_naive_kernel<Dtype>(
            reinterpret_cast<Dtype*>(A.gpu_data()), 
            reinterpret_cast<Dtype*>(B.gpu_data()), 
            reinterpret_cast<Dtype*>(C.gpu_data()), 
            A.rows(), A.cols(), B.cols());
}

template <typename Dtype>
void matmul_cuda_shared(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    matmul_shared_kernel<Dtype>(
            reinterpret_cast<Dtype*>(A.gpu_data()), 
            reinterpret_cast<Dtype*>(B.gpu_data()), 
            reinterpret_cast<Dtype*>(C.gpu_data()), 
            A.rows(), A.cols(), B.cols());
}





template void matmul_naive(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_naive(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_naive(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_openmp(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_openmp(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_openmp(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_cuda_naive(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_cuda_naive(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_cuda_naive(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_cuda_shared(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_cuda_shared(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_cuda_shared(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );
