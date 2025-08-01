#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h> // For uint16_t
#include "comp_table.h"

#define TILE_SIZE 16

typedef union {
    __nv_bfloat16 bf16;
    uint16_t u16;
} bf16_union;

__device__ uint16_t bfloat16_to_uint16(__nv_bfloat16 a) {
    bf16_union u;
    u.bf16 = a;
    return u.u16;
}

__device__ __nv_bfloat16 uint16_to_bfloat16(uint16_t a) {
    bf16_union u;
    u.u16 = a;
    return u.bf16;
}

__device__ __nv_bfloat16 FPMA_approx_kernel_mul_bf16(__nv_bfloat16 a, __nv_bfloat16 b, uint16_t comp_table[COMP_TABLE_SIZE][COMP_TABLE_SIZE]) {
    uint16_t a_bits = bfloat16_to_uint16(a);
    uint16_t b_bits = bfloat16_to_uint16(b);
    
    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint8_t comp_table_index_a = (a_bits >> 4) & 0x07;
    uint8_t comp_table_index_b = (b_bits >> 4) & 0x07;
    // uint8_t comp_table_index_a = (a_bits >> 3) & 0x0F;
    // uint8_t comp_table_index_b = (b_bits >> 3) & 0x0F;
    uint16_t comp = comp_table[comp_table_index_a][comp_table_index_b];
    // BF16 Baseline: 0x3F80
    uint16_t threshold = 0x3F80;
    threshold = threshold - comp;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;
    
    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t adjusted_mantissa = diff & (~is_negative_mask); 
    
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) + sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF; 
    // }

    result_bits = is_zero ? 0 : result_bits;
    
    return uint16_to_bfloat16(result_bits);
}

// __global__ void FPMA_approx_kernel_bf16(
//     const __nv_bfloat16* __restrict__ A,
//     const __nv_bfloat16* __restrict__ B,
//     __nv_bfloat16* __restrict__ C,
//     const int M, const int N, const int K) {
    
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < M && col < N) {
//         __nv_bfloat16 sum = uint16_to_bfloat16(0);
//         for (int k = 0; k < K; k++) {
//             __nv_bfloat16 a = A[row * K + k];
//             __nv_bfloat16 b = B[k * N + col];
//             // sum += a * b;
//             sum += FPMA_approx_kernel_mul_bf16(a, b);
//         }
//         C[row * N + col] = sum;
//     }
// }

// Optimized GEMM kernel with tiling, shared memory, and loop unrolling
template <int BLOCK, int STRIDE>
__global__ void FPMA_approx_kernel_bf16_optimized(
    __nv_bfloat16*  A,
    __nv_bfloat16*  B,
    __nv_bfloat16*  C,
    int M, int N, int K) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_bf16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    
    constexpr int STEP = BLOCK * STRIDE * 2;

    int tx = threadIdx.x * STRIDE;
    int ty = threadIdx.y * STRIDE;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    __nv_bfloat16 *begin_A = A + bx * STEP * K;
    __nv_bfloat16 *begin_B = B + by * STEP * K;
    __nv_bfloat16 *end_a = begin_A + K;
    __nv_bfloat16 *end_b = begin_B + K;

    __shared__ __nv_bfloat16 shared_A[STEP][STEP + 1];
    __shared__ __nv_bfloat16 shared_B[STEP + 1][STEP];

    float sum0_fp32[STRIDE][STRIDE] = {0.0f};
    float sum1_fp32[STRIDE][STRIDE] = {0.0f};
    float sum2_fp32[STRIDE][STRIDE] = {0.0f};
    float sum3_fp32[STRIDE][STRIDE] = {0.0f};

    for (__nv_bfloat16* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=STEP, b_ptr+=STEP) {
        __nv_bfloat16 sum0[STRIDE][STRIDE] = {__float2bfloat16(0.0f)};
        __nv_bfloat16 sum1[STRIDE][STRIDE] = {__float2bfloat16(0.0f)};
        __nv_bfloat16 sum2[STRIDE][STRIDE] = {__float2bfloat16(0.0f)};
        __nv_bfloat16 sum3[STRIDE][STRIDE] = {__float2bfloat16(0.0f)};

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                shared_A[tx + i][ty + j] = (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) ? a_ptr[(tx + i) * K + (ty + j)] : __float2bfloat16(0.0f);
                shared_A[tx + i][ty + j + 32] = (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) ? a_ptr[(tx + i) * K + (ty + j + 32)] : __float2bfloat16(0.0f);
                shared_A[tx + i + 32][ty + j] = (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) ? a_ptr[(tx + i + 32) * K + (ty + j)] : __float2bfloat16(0.0f);
                shared_A[tx + i + 32][ty + j + 32] = (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) ? a_ptr[(tx + i + 32) * K + (ty + j + 32)] : __float2bfloat16(0.0f);

                shared_B[ty + j][tx + i] = (b_ptr + tx + i < end_b && by * STEP + ty + j < N) ? b_ptr[(ty + j) * K + (tx + i)] : __float2bfloat16(0.0f);
                shared_B[ty + j][tx + i + 32] = (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) ? b_ptr[(ty + j) * K + (tx + i + 32)] : __float2bfloat16(0.0f);
                shared_B[ty + j + 32][tx + i] = (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) ? b_ptr[(ty + j + 32) * K + (tx + i)] : __float2bfloat16(0.0f);
                shared_B[ty + j + 32][tx + i + 32] = (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) ? b_ptr[(ty + j + 32) * K + (tx + i + 32)] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();


        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    // sum0[i][j] = __hadd(sum0[i][j], __hmul(shared_A[k][tx + i], shared_B[ty + j][k]));
                    // sum1[i][j] = __hadd(sum1[i][j], __hmul(shared_A[k][tx + i], shared_B[ty + j + 32][k]));
                    // sum2[i][j] = __hadd(sum2[i][j], __hmul(shared_A[k][tx + i + 32], shared_B[ty + j][k]));
                    // sum3[i][j] = __hadd(sum3[i][j], __hmul(shared_A[k][tx + i + 32], shared_B[ty + j + 32][k]));
                    sum0[i][j] = __hadd(sum0[i][j], FPMA_approx_kernel_mul_bf16(shared_A[tx + i][k], shared_B[ty + j][k], comp_table_shared));
                    sum1[i][j] = __hadd(sum1[i][j], FPMA_approx_kernel_mul_bf16(shared_A[tx + i][k], shared_B[ty + j + 32][k], comp_table_shared));
                    sum2[i][j] = __hadd(sum2[i][j], FPMA_approx_kernel_mul_bf16(shared_A[tx + i + 32][k], shared_B[ty + j][k], comp_table_shared));
                    sum3[i][j] = __hadd(sum3[i][j], FPMA_approx_kernel_mul_bf16(shared_A[tx + i + 32][k], shared_B[ty + j + 32][k], comp_table_shared));
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                sum0_fp32[i][j] += __bfloat162float(sum0[i][j]);
                sum1_fp32[i][j] += __bfloat162float(sum1[i][j]);
                sum2_fp32[i][j] += __bfloat162float(sum2[i][j]);
                sum3_fp32[i][j] += __bfloat162float(sum3[i][j]);
            }
        }
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx + i;
            int col = STEP * by + ty + j;
            C[row * N + col] = (row < M && col < N) ? __float2bfloat16(sum0_fp32[i][j]) : __float2bfloat16(0.0f);
            C[row * N + col + 32] = (row < M && col + 32 < N) ? __float2bfloat16(sum1_fp32[i][j]) : __float2bfloat16(0.0f);
            C[(row + 32) * N + col] = (row + 32 < M && col < N) ? __float2bfloat16(sum2_fp32[i][j]) : __float2bfloat16(0.0f);
            C[(row + 32) * N + col + 32] = (row + 32 < M && col + 32 < N) ? __float2bfloat16(sum3_fp32[i][j]) : __float2bfloat16(0.0f);
        }
    }
}

__global__ void FPMA_approx_kernel_bf16_optimized_batched(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K, 
    const int batch_size) {

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_bf16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) return;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int row = block_row * TILE_SIZE + thread_row;
    int col = block_col * TILE_SIZE + thread_col;

    __shared__ __nv_bfloat16 shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 shared_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int A_offset = batch_idx * M * K;
    int B_offset = batch_idx * K * N;
    int C_offset = batch_idx * M * N;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load data into shared memory with boundary checks
        if ((row < M) && (t * TILE_SIZE + thread_col) < K) {
            shared_A[thread_row][thread_col] = A[A_offset + row * K + t * TILE_SIZE + thread_col];
        } else {
            shared_A[thread_row][thread_col] = uint16_to_bfloat16(0);
        }

        if ((t * TILE_SIZE + thread_row) < K && (col < N)) {
            shared_B[thread_row][thread_col] = B[B_offset + (t * TILE_SIZE + thread_row) * N + col];
        } else {
            shared_B[thread_row][thread_col] = uint16_to_bfloat16(0);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            __nv_bfloat16 a = shared_A[thread_row][k];
            __nv_bfloat16 b = shared_B[k][thread_col];

            __nv_bfloat16 product_bf16 = FPMA_approx_kernel_mul_bf16(a, b, comp_table_shared);
            float product = __bfloat162float(product_bf16);
            sum += product;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        __nv_bfloat16 sum_bf16 = __float2bfloat16(sum);
        C[C_offset + row * N + col] = sum_bf16;
    }
}

void launch_FPMA_approx_kernel_bf16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    dim3 threads_per_block(BLOCK, BLOCK);
    dim3 blocks_per_grid(
        (M + BLOCK*STRIDE*2 - 1) / BLOCK / STRIDE / 2,  
        (N + BLOCK*STRIDE*2 - 1) / BLOCK / STRIDE / 2 
    );
    FPMA_approx_kernel_bf16_optimized<BLOCK, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);

    // Check for CUDA errors
    cudaError_t errlast = cudaGetLastError();
    if (errlast != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(errlast));
    }
}

void launch_FPMA_approx_kernel_bf16_batched(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K, int batch_size) {
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y, batch_size);

    FPMA_approx_kernel_bf16_optimized_batched<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K, batch_size);

    // Check for CUDA errors
    cudaError_t errlast = cudaGetLastError();
    if (errlast != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(errlast));
    }
}

__global__ void FPMA_elementwisemul_single_kernel_bf16(
    const __nv_bfloat16* __restrict__ A, 
    const __nv_bfloat16* __restrict__ B, 
    __nv_bfloat16* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col) {

    __shared__ __nv_bfloat16 shared_A[TILE_SIZE][TILE_SIZE];
    __nv_bfloat16 reg_B = B[0];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_bfloat16(0);
    }
    
    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_bf16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        __nv_bfloat16 reg_A = shared_A[threadIdx.y][threadIdx.x];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = FPMA_approx_kernel_mul_bf16(reg_A, reg_B, comp_table_shared);
    }
}

__global__ void FPMA_elementwisemul_all_kernel_bf16(
    const __nv_bfloat16* __restrict__ A, 
    const __nv_bfloat16* __restrict__ B, 
    __nv_bfloat16* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col) {

    __shared__ __nv_bfloat16 shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 shared_B[TILE_SIZE][TILE_SIZE];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_bfloat16(0);
    }

    if (A_row_idx < B_row && A_col_idx < B_col) {
        shared_B[threadIdx.y][threadIdx.x] = B[A_row_idx * B_col + A_col_idx] ;
    } else {
        shared_B[threadIdx.y][threadIdx.x] = uint16_to_bfloat16(0);
    }

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_bf16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        __nv_bfloat16 reg_A = shared_A[threadIdx.y][threadIdx.x];
        __nv_bfloat16 reg_B = shared_B[threadIdx.y][threadIdx.x];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = FPMA_approx_kernel_mul_bf16(reg_A, reg_B, comp_table_shared);
    }
}

__global__ void FPMA_elementwisemul_rowwise_kernel_bf16(
    const __nv_bfloat16* __restrict__ A, 
    const __nv_bfloat16* __restrict__ B, 
    __nv_bfloat16* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col) {

    __shared__ __nv_bfloat16 shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 shared_B[TILE_SIZE];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_bfloat16(0);
    }

    if (threadIdx.x == 0) {
        shared_B[threadIdx.y] = B[A_row_idx];
    }

    __shared__ uint16_t comp_table_shared[COMP_TABLE_SIZE][COMP_TABLE_SIZE];
    if (threadIdx.x < COMP_TABLE_SIZE && threadIdx.y < COMP_TABLE_SIZE) {
        comp_table_shared[threadIdx.x][threadIdx.y] = comp_table_const_bf16_8x8[threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        __nv_bfloat16 reg_A = shared_A[threadIdx.y][threadIdx.x];
        __nv_bfloat16 reg_B = shared_B[threadIdx.y];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = FPMA_approx_kernel_mul_bf16(reg_A, reg_B, comp_table_shared);
    }
}

void launch_FPMA_elementwisemul_kernel_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int A_row, int A_col, int B_row, int B_col) {
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid((A_col + threads_per_block.x - 1) / threads_per_block.x, (A_row + threads_per_block.y - 1) / threads_per_block.y);

    if (B_row == 1 && B_col == 1) {
        FPMA_elementwisemul_single_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else if (B_row == A_row && B_col == A_col) {
        FPMA_elementwisemul_all_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else if (B_row == A_row && B_col == 1) {
        FPMA_elementwisemul_rowwise_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else {
        throw std::runtime_error("Unsupported elementwise operation");
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}