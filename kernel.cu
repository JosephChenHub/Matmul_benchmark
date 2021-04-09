#include <cuda_runtime_api.h>
#include "kernel.hpp"

#include <cublas_v2.h>
#include <stdio.h>
#include <cassert>

// cublas
const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

cublasHandle_t blas_handle() {
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    const int n = 0;
    //cudaError_t status = cudaGetDevice(&n);
    if(!init[n]) {
        cublasStatus_t st = cublasCreate(&handle[n]);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("blas_handle create failed! %s:%d, code:%s\n", __FILE__, __LINE__, cublasGetErrorString(st));
        }
        init[n] = 1;
    }
    return handle[n];
}

void cublas_sgemm(const float* d_A, const float *d_B, float* d_C, const int A_ROW, const int A_COL, const int B_COL) {
    cublasHandle_t handle = blas_handle();
    float alpha = 1.0, beta = 0.0;
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_COL, A_ROW, A_COL,&alpha,
            d_B, B_COL, d_A, A_COL, &beta, d_C, B_COL);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm error occurred! %s : %d, error_code:%s\n", __FILE__, __LINE__, 
                cublasGetErrorString(st));
        exit(-1);
    }
}

template<typename Dtype>
__device__ __forceinline__ void set(Dtype* data, const int rows, const int cols,
		const int i, const int j, const Dtype value) {
   if(i < rows && j < cols) {
       data[i*cols+j] = value;
   }
}	

template <typename Dtype>
__device__ __forceinline__ Dtype fetch(const Dtype* data, const int rows, const int cols,
		const int i, const int j) {
    return (i < rows && j < cols) ? data[i * cols + j] : static_cast<Dtype>(0);
}	

// naive version 
template <typename Dtype>
__global__ void matmul_naive(const Dtype* A, 
            const Dtype* B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    size_t tid = xIndex + yIndex * col_B;
    if(xIndex >= col_B || yIndex >= row_A) return;

    Dtype tmp(0);
    for(int i = 0; i < col_A; ++i) {
        tmp += A[yIndex * col_A + i] * B[i*col_B + xIndex];
    }
    C[tid] = tmp;
}

#define DIM 32
#define IPAD 0

__device__ __host__ __forceinline__ int div_up(const int a, const int b) {
    return (a + b - 1) / b;
}	

template <typename Dtype>
__global__ void matmul_tile(const Dtype* A, 
            const Dtype* B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    __shared__ Dtype sA[DIM][DIM+IPAD];
    __shared__ Dtype sB[DIM][DIM+IPAD];
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x; // column x of C/B 
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y; // row y of C/A

    Dtype tmp(0);
    for(int i = 0; i < div_up(col_A, DIM); ++i) {
	auto xA = DIM * i + threadIdx.x; 
	auto yB = DIM * i + threadIdx.y;
        // load
	sA[threadIdx.y][threadIdx.x] = fetch(A, row_A, col_A, yIndex, xA);
	sB[threadIdx.y][threadIdx.x] = fetch(B, col_A, col_B, yB, xIndex);

        __syncthreads();
        // partial matmul
        for(int j = 0; j < DIM; ++j) {
            tmp += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    set(C, row_A, col_B, yIndex, xIndex, tmp);
}



template <typename Dtype>
__global__ void matmul_unroll(const Dtype* A, 
            const Dtype* B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    __shared__ Dtype sA[DIM*2][DIM+IPAD];
    __shared__ Dtype sB[DIM*2][DIM+IPAD];
    const int xIndex = threadIdx.x + 2*blockIdx.x * blockDim.x; // column x of C/B 
    const int yIndex = threadIdx.y + 2*blockIdx.y * blockDim.y; // row y of C/A
    Dtype tmp[4] = {0, 0, 0, 0};

    for(int i = 0; i < div_up(col_A, DIM); ++i) {
	auto xA = DIM * i + threadIdx.x; 
	auto yB = DIM * i + threadIdx.y;
        // load
	sA[threadIdx.y][threadIdx.x] = fetch(A, row_A, col_A, yIndex, xA);
	sA[threadIdx.y+DIM][threadIdx.x] = fetch(A, row_A, col_A, DIM+yIndex, xA);
        sB[threadIdx.y][threadIdx.x] =  fetch(B, col_A, col_B, yB, xIndex); 
        sB[threadIdx.y+DIM][threadIdx.x] = fetch(B, col_A, col_B, yB, xIndex+DIM); 
        __syncthreads();
        for(int j = 0; j < DIM; ++j) {
	    auto aj1 = sA[threadIdx.y][j];
	    auto aj2 = sA[threadIdx.y+DIM][j];
            auto bj1 = sB[j][threadIdx.x];
	    auto bj2 = sB[j+DIM][threadIdx.x];

            tmp[0] += aj1 * bj1; 
	    tmp[1] += aj1 * bj2; 
	    tmp[2] += aj2 * bj1; 
	    tmp[3] += aj2 * bj2;
        }
        __syncthreads();
    }
    set(C, row_A, col_B, yIndex, xIndex, tmp[0]);
    set(C, row_A, col_B, yIndex, xIndex+DIM, tmp[1]);
    set(C, row_A, col_B, yIndex+DIM, xIndex, tmp[2]);
    set(C, row_A, col_B, yIndex+DIM, xIndex+DIM, tmp[3]);
}





template <typename Dtype>
void matmul_naive_kernel(const Dtype* A, const Dtype* B, Dtype* C,
        const int m, const int n, const int k) {
    dim3 block(DIM, DIM);
    int grid_x = div_up(k, block.x);
    int grid_y = div_up(m, block.y);
    dim3 grid(grid_x, grid_y);

    matmul_naive<Dtype><<<grid, block>>>(
            A, B, C, m, n, k);
}

template <typename Dtype>
void matmul_tile_kernel(const Dtype* A, const Dtype* B, Dtype* C,
        const int m, const int n, const int k) {
    dim3 block(DIM, DIM);
    int grid_x = div_up(k, block.x);
    int grid_y = div_up(m, block.y);
    dim3 grid(grid_x, grid_y);

    matmul_tile<Dtype><<<grid, block>>>(
            A, B, C, m, n, k);
}


template <typename Dtype>
void matmul_unroll_kernel(const Dtype* A, const Dtype* B, Dtype* C,
        const int m, const int n, const int k) {
    dim3 block(DIM, DIM);
    int grid_x = div_up(k, block.x*2);
    int grid_y = div_up(m, block.y*2);
    dim3 grid(grid_x, grid_y);

    matmul_unroll<Dtype><<<grid, block>>>(
            A, B, C, m, n, k);
}



template void matmul_naive_kernel(const float*, const float* ,float*, 
        const int, const int, const int );
template void matmul_naive_kernel(const double*, const double* , double*, 
        const int, const int, const int );
template void matmul_naive_kernel(const int*, const int* , int*, 
        const int, const int, const int );

template void matmul_tile_kernel(const float*, const float* ,float*, 
        const int, const int, const int );
template void matmul_tile_kernel(const double*, const double* , double*, 
        const int, const int, const int );
template void matmul_tile_kernel(const int*, const int* , int*, 
        const int, const int, const int );

template void matmul_unroll_kernel(const float*, const float* ,float*, 
        const int, const int, const int );
template void matmul_unroll_kernel(const double*, const double* , double*, 
        const int, const int, const int );
template void matmul_unroll_kernel(const int*, const int* , int*, 
        const int, const int, const int );


