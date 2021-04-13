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
__device__ __forceinline__ Dtype fetch(const Dtype* __restrict__ data, const int rows, const int cols,
    const int i, const int j) {
    return (i < rows && j < cols) ? data[i * cols + j] : static_cast<Dtype>(0);
}

// naive version 
template <typename Dtype>
__global__ void matmul_naive(const Dtype* __restrict__ A, 
            const Dtype* __restrict__ B, Dtype* C,  
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

// 32x32: no bank conflicts
template <typename Dtype>
__global__ void matmul_tile(const Dtype* __restrict__ A, 
            const Dtype* __restrict__ B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    __shared__ Dtype sA[DIM][DIM];
    __shared__ Dtype sB[DIM][DIM];
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
__global__ void matmul_unroll(const Dtype* __restrict__ A, 
            const Dtype* __restrict__ B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    __shared__ Dtype sA[DIM][DIM*2]; // 32x32, 32x32
    __shared__ Dtype sB[DIM][DIM*2];  
    const int xIndex = threadIdx.x + 2*blockIdx.x * blockDim.x; // column x of C/B 
    const int yIndex = threadIdx.y + 2*blockIdx.y * blockDim.y; // row y of C/A
    Dtype tmp[4] = {0, 0, 0, 0};

    for(int i = 0; i < div_up(col_A, DIM); ++i) {
        auto xA = DIM * i + threadIdx.x; 
        auto yB = DIM * i + threadIdx.y;
        // load
        sA[threadIdx.y][threadIdx.x] = fetch(A, row_A, col_A, yIndex, xA);
        sA[threadIdx.y][threadIdx.x+DIM] = fetch(A, row_A, col_A, DIM+yIndex, xA);

        sB[threadIdx.y][threadIdx.x] =  fetch(B, col_A, col_B, yB, xIndex); 
        sB[threadIdx.y][threadIdx.x+DIM] = fetch(B, col_A, col_B, yB, xIndex+DIM);
        __syncthreads();
        for(int j = 0; j < DIM; ++j) {
            auto aj1 = sA[threadIdx.y][j];
            auto aj2 = sA[threadIdx.y][j+DIM];
            auto bj1 = sB[j][threadIdx.x];
            auto bj2 = sB[j][threadIdx.x+DIM];

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

#define _TILE_WIDTH 128 
#define _BLOCK_THREADS 16 
#define _REDUCE 8
#define _X_EXPAND _BLOCK_THREADS
#define _Y_EXPAND (_BLOCK_THREADS * _BLOCK_THREADS / _TILE_WIDTH)

// block:16x16, each block computes a 128x128 tile
template <typename Dtype>
__global__ void matmul_comopt(const Dtype* __restrict__ A, 
            const Dtype* __restrict__ B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    const int xIndex = threadIdx.x + blockIdx.x * _TILE_WIDTH; // column x of C/B 
    const int yIndex = threadIdx.y + blockIdx.y * _TILE_WIDTH; // row y of C/A
    __shared__ Dtype sA[_TILE_WIDTH][_BLOCK_THREADS];  // 128x16
    __shared__ Dtype sB[_BLOCK_THREADS][_TILE_WIDTH];  // 16x128
    Dtype tmp[_REDUCE*_REDUCE];
    for(int i = 0; i < _REDUCE*_REDUCE; ++i) tmp[i] = 0;

    const int local_id = threadIdx.x + blockDim.x * threadIdx.y;
    const int local_ax = local_id % _BLOCK_THREADS;
    const int local_ay = local_id / _BLOCK_THREADS;
    const int local_bx = local_id % _TILE_WIDTH;
    const int local_by = local_id / _TILE_WIDTH;

    int ii, jj;
    for(int i = 0; i < div_up(col_A, _BLOCK_THREADS); ++i) {
        // load
        #pragma unroll
        for(ii = 0; ii < _REDUCE; ++ii) {
            sA[local_ay+ii*_X_EXPAND][local_ax] =  fetch(A, row_A, col_A, blockIdx.y*_TILE_WIDTH + ii*_X_EXPAND + local_ay, local_ax + _BLOCK_THREADS*i);
            sB[local_by+ii*_Y_EXPAND][local_bx] =  fetch(B, col_A, col_B, _BLOCK_THREADS*i+local_by+ii*_Y_EXPAND, local_bx+blockIdx.x*_TILE_WIDTH);
        }
        __syncthreads(); 
        for(int j = 0; j < _BLOCK_THREADS; ++j) {
            #pragma unroll 
            for(ii = 0; ii < _REDUCE; ++ii) {
                for(jj = 0; jj < _REDUCE; ++jj) {
                    tmp[ii*_REDUCE+jj] += sA[threadIdx.y+blockDim.y*ii][j] * sB[j][threadIdx.x + blockDim.x * jj];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(ii = 0; ii < _REDUCE; ++ii) {
        for(jj = 0; jj < _REDUCE; ++jj) {
            set(C, row_A, col_B, yIndex+blockDim.y*ii , xIndex+blockDim.x*jj, tmp[ii*_REDUCE+jj]);
        }   
    }
}




#define TRANS_X 32
#define TRANS_Y 32
#define TRANS_PAD 0
template <typename Dtype>
__global__ void transpose_unroll(const Dtype* __restrict__ src, Dtype* __restrict__ dst,
    const int rows, const int cols) {
    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x * 2;
    const int yIndex = threadIdx.y + blockIdx.y * blockDim.y; 
    __shared__ Dtype tile[TRANS_Y][TRANS_X*2 + TRANS_PAD];

    tile[threadIdx.y][threadIdx.x] = fetch(src, rows, cols, yIndex, xIndex);
    tile[threadIdx.y][threadIdx.x+TRANS_X] = fetch(src, rows, cols, yIndex, xIndex+TRANS_X);
    __syncthreads();

    set(dst, cols, rows, xIndex, yIndex, tile[threadIdx.y][threadIdx.x]);
    set(dst, cols, rows, xIndex+TRANS_X, yIndex, tile[threadIdx.y][threadIdx.x+TRANS_X]);
}

template <typename Dtype>
void cuda_transpose(const Dtype* __restrict__ A, Dtype* __restrict__ B, 
    const int rows, const int cols) {

    dim3 block(TRANS_X, TRANS_Y);
    dim3 grid(div_up(cols, TRANS_X*2), div_up(rows, TRANS_Y));

    transpose_unroll<Dtype><<<grid, block>>>(A, B, rows, cols);
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
    cudaFuncSetCacheConfig<Dtype>((Dtype*)&matmul_unroll<Dtype>, cudaFuncCachePreferShared);
    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    matmul_unroll<Dtype><<<grid, block>>>(
            A, B, C, m, n, k);
}

template <typename Dtype>
void matmul_comopt_kernel(const Dtype* A, const Dtype* B, Dtype* C,
        const int rows_A, const int cols_A, const int cols_B) {
    dim3 block(_BLOCK_THREADS, _BLOCK_THREADS);
    int grid_x = div_up(cols_B, _TILE_WIDTH);
    int grid_y = div_up(rows_A, _TILE_WIDTH);
    dim3 grid(grid_x, grid_y);
    cudaFuncSetCacheConfig<Dtype>((Dtype*)&matmul_comopt<Dtype>, cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    matmul_comopt<Dtype><<<grid, block>>>(
            A, B, C, rows_A, cols_A, cols_B);
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

template void matmul_comopt_kernel(const float*, const float* ,float*, 
        const int, const int, const int );
template void matmul_comopt_kernel(const double*, const double* , double*, 
        const int, const int, const int );
template void matmul_comopt_kernel(const int*, const int* , int*, 
        const int, const int, const int );

template void cuda_transpose(const float*, float*, const int, const int);
template void cuda_transpose(const double*, double*, const int, const int);
template void cuda_transpose(const int*, int*, const int, const int);
