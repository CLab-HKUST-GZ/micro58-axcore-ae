#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h> // For uint16_t

// #define BLOCK_SIZE 16

// 1.3s 1x
template <int BLOCK>
__global__ void sgemm_1(const half* A, const half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    int row = blockIdx.x * BLOCK + threadIdx.x;
    int col = blockIdx.y * BLOCK + threadIdx.y;
    if (row < M and col < N) {
        half sum = __float2half(0.0f);
        #pragma unroll
        for (int i = 0; i < K; i++) {
        sum = __hadd(sum, __hmul(A[row * K + i], B[col * K + i]));
        }
        C[row * N + col] = sum;
    }
}

// 115ms 12x
template <int BLOCK>
__global__ void sgemm_2(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.x * BLOCK + threadIdx.x;
    int col = blockIdx.y * BLOCK + threadIdx.y;
    
    half *begin_A = A + blockIdx.x * BLOCK * K;
    half *begin_B = B + blockIdx.y * BLOCK * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    half sum = __float2half(0.0f);
    for (half* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=BLOCK, b_ptr+=BLOCK) {
        __shared__ half shared_A[BLOCK][BLOCK];
        __shared__ half shared_B[BLOCK][BLOCK];

        if (a_ptr + ty < end_a) {
            shared_A[ty][tx] = a_ptr[tx * K + ty];
        } else {
            shared_A[ty][tx] = __float2half(0.0f);
        }
        if (b_ptr + tx < end_b) {
            shared_B[ty][tx] = b_ptr[ty * K + tx];
        } else {
            shared_B[ty][tx] = __float2half(0.0f);
        }
        __syncthreads();

        for (int i=0; i<BLOCK; i++) {
            sum = __hadd(sum, __hmul(shared_A[i][tx], shared_B[ty][i]));
        }
        __syncthreads();
    }
    if (row < M and col < N) {
        C[row * N + col] = sum;
    }
}

// BLOCK 16 STRIDE 2 70ms 19x
// BLOCK 8 STRIDE 4 53ms 25x
template <int BLOCK, int STRIDE>
__global__ void sgemm_3(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    constexpr int STEP = BLOCK * STRIDE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    half *begin_A = A + bx * STEP * K;
    half *begin_B = B + by * STEP * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    half sum[STRIDE][STRIDE] = {__float2half(0.0f)};
    for (half* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=STEP, b_ptr+=STEP) {
        __shared__ half shared_A[STEP][STEP];
        __shared__ half shared_B[STEP][STEP];

        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                if (a_ptr + ty * STRIDE + j < end_a && bx * STEP + tx * STRIDE + i < M) {
                    // shared_A[tx * STRIDE + i][ty * STRIDE + j] = a_ptr[(tx * STRIDE + i) * K + (ty * STRIDE + j)];
                    shared_A[ty * STRIDE + j][tx * STRIDE + i] = a_ptr[(tx * STRIDE + i) * K + (ty * STRIDE + j)];
                } else {
                    shared_A[ty * STRIDE + j][tx * STRIDE + i] = __float2half(0.0f);
                }
                if (b_ptr + tx * STRIDE + i < end_b && by * STEP + ty * STRIDE + j < N) {
                    // shared_B[tx * STRIDE + i][ty * STRIDE + j] = b_ptr[(ty * STRIDE + j) * K + (tx * STRIDE + i)];
                    shared_B[ty * STRIDE + j][tx * STRIDE + i] = b_ptr[(ty * STRIDE + j) * K + (tx * STRIDE + i)];
                } else {
                    shared_B[ty * STRIDE + j][tx * STRIDE + i] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();


        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    sum[i][j] = __hadd(sum[i][j], __hmul(shared_A[k][tx * STRIDE + i], shared_B[ty * STRIDE + j][k]));
                }
            }
        }
        __syncthreads();
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx * STRIDE + i;
            int col = STEP * by + ty * STRIDE + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

// BLOCK 8 STRIDE 4 58ms
// BLOCK 16 STRIDE 2 70ms
template <int BLOCK, int STRIDE>
__global__ void sgemm_4(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    constexpr int STEP = BLOCK * STRIDE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    half *begin_A = A + bx * STEP * K;
    half *begin_B = B + by * STEP * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    half sum[STRIDE][STRIDE] = {__float2half(0.0f)};
    for (half* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=2*STEP, b_ptr+=2*STEP) {
        __shared__ half shared_A[STEP * 2][STEP];
        __shared__ half shared_B[STEP][STEP * 2];

        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                if (a_ptr + ty * STRIDE + j < end_a && bx * STEP + tx * STRIDE + i < M) {
                    // shared_A[tx * STRIDE + i][ty * STRIDE + j] = a_ptr[(tx * STRIDE + i) * K + (ty * STRIDE + j)];
                    shared_A[ty * STRIDE + j][tx * STRIDE + i] = a_ptr[(tx * STRIDE + i) * K + (ty * STRIDE + j)];
                } else {
                    shared_A[ty * STRIDE + j][tx * STRIDE + i] = __float2half(0.0f);
                }
                if (a_ptr + ty * STRIDE + j + STEP < end_a && bx * STEP + tx * STRIDE + i < M) {
                    shared_A[ty * STRIDE + j + STEP][tx * STRIDE + i] = a_ptr[(tx * STRIDE + i) * K + (ty * STRIDE + j + STEP)];
                } else {
                    shared_A[ty * STRIDE + j + STEP][tx * STRIDE + i] = __float2half(0.0f);
                }

                if (b_ptr + tx * STRIDE + i < end_b && by * STEP + ty * STRIDE + j < N) {
                    shared_B[ty * STRIDE + j][tx * STRIDE + i] = b_ptr[(ty * STRIDE + j) * K + (tx * STRIDE + i)];
                } else {
                    shared_B[ty * STRIDE + j][tx * STRIDE + i] = __float2half(0.0f);
                }
                if (b_ptr + tx * STRIDE + i < end_b && by * STEP + ty * STRIDE + j < N) {
                    shared_B[ty * STRIDE + j][tx * STRIDE + i + STEP] = b_ptr[(ty * STRIDE + j) * K + (tx * STRIDE + i + STEP)];
                } else {
                    shared_B[ty * STRIDE + j][tx * STRIDE + i + STEP] = __float2half(0.0f);
                }

            }
        }
        __syncthreads();


        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < 2*STEP; k++) {
                    sum[i][j] = __hadd(sum[i][j], __hmul(shared_A[k][tx * STRIDE + i], shared_B[ty * STRIDE + j][k]));
                }
            }
        }
        __syncthreads();
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx * STRIDE + i;
            int col = STEP * by + ty * STRIDE + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

typedef union {
    half fp16;
    uint16_t u16;
} fp16_union;

__device__ uint16_t float16_to_uint16(half a) {
    fp16_union u;
    u.fp16 = a;
    return u.u16;
}

__device__ half uint16_to_float16(uint16_t a) {
    fp16_union u;
    u.u16 = a;
    return u.fp16;
}

__device__ half baseline_approx_kernel_mul_fp16(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;
    
    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;
    // FP16 Baseline: 0x3C00
    uint16_t threshold = 0x3C00;
    
    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t adjusted_mantissa = diff & (~is_negative_mask); 
    // uint16_t adjusted_mantissa = diff;
    
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;

    result_bits = is_zero ? 0 : result_bits;
    
    return uint16_to_float16(result_bits);
}

// BLOCK 16 STRIDE 2 33ms 39x
template <int BLOCK, int STRIDE>
__global__ void sgemm_5(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    constexpr int STEP = BLOCK * STRIDE * 2;

    int tx = threadIdx.x * STRIDE;
    int ty = threadIdx.y * STRIDE;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    half *begin_A = A + bx * STEP * K;
    half *begin_B = B + by * STEP * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    __shared__ half shared_A[STEP][STEP + 1];
    __shared__ half shared_B[STEP + 1][STEP];

    half sum0[STRIDE][STRIDE] = {__float2half(0.0f)};
    half sum1[STRIDE][STRIDE] = {__float2half(0.0f)};
    half sum2[STRIDE][STRIDE] = {__float2half(0.0f)};
    half sum3[STRIDE][STRIDE] = {__float2half(0.0f)};
    for (half* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=STEP, b_ptr+=STEP) {
        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                if (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) {
                    shared_A[ty + j][tx + i] = a_ptr[(tx + i) * K + (ty + j)];
                } else {
                    shared_A[ty + j][tx + i] = __float2half(0.0f);
                }
                if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) {
                    shared_A[ty + j + 32][tx + i] = a_ptr[(tx + i) * K + (ty + j + 32)];
                } else {
                    shared_A[ty + j + 32][tx + i] = __float2half(0.0f);
                }
                if (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) {
                    shared_A[ty + j][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j)];
                } else {
                    shared_A[ty + j][tx + i + 32] = __float2half(0.0f);
                }
                if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) {
                    shared_A[ty + j + 32][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j + 32)];
                } else {
                    shared_A[ty + j + 32][tx + i + 32] = __float2half(0.0f);
                }


                if (b_ptr + tx + i < end_b && by * STEP + ty + j < N) {
                    shared_B[ty + j][tx + i] = b_ptr[(ty + j) * K + (tx + i)];
                } else {
                    shared_B[ty + j][tx + i] = __float2half(0.0f);
                }
                if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) {
                    shared_B[ty + j][tx + i + 32] = b_ptr[(ty + j) * K + (tx + i + 32)];
                } else {
                    shared_B[ty + j][tx + i + 32] = __float2half(0.0f);
                }
                if (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) {
                    shared_B[ty + j + 32][tx + i] = b_ptr[(ty + j + 32) * K + (tx + i)];
                } else {
                    shared_B[ty + j + 32][tx + i] = __float2half(0.0f);
                }
                if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) {
                    shared_B[ty + j + 32][tx + i + 32] = b_ptr[(ty + j + 32) * K + (tx + i + 32)];
                } else {
                    shared_B[ty + j + 32][tx + i + 32] = __float2half(0.0f);
                }


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

                    sum0[i][j] = __hadd(sum0[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i], shared_B[ty + j][k]));
                    sum1[i][j] = __hadd(sum1[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i], shared_B[ty + j + 32][k]));
                    sum2[i][j] = __hadd(sum2[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i + 32], shared_B[ty + j][k]));
                    sum3[i][j] = __hadd(sum3[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i + 32], shared_B[ty + j + 32][k]));
                }
            }
        }
        __syncthreads();
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx + i;
            int col = STEP * by + ty + j;
            if (row < M && col < N) {
                C[row * N + col] = sum0[i][j];
                C[row * N + col + 32] = sum1[i][j];
                C[(row + 32) * N + col] = sum2[i][j];
                C[(row + 32) * N + col + 32] = sum3[i][j];
            }
        }
    }
}

// BLOCK 16 STRIDE 2 49ms
template <int BLOCK, int STRIDE>
__global__ void sgemm_6(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    constexpr int STEP = BLOCK * STRIDE * 2;

    int tx = threadIdx.x * STRIDE;
    int ty = threadIdx.y * STRIDE;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    half *begin_A = A + bx * STEP * K;
    half *begin_B = B + by * STEP * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    __shared__ half shared_A[2][STEP][STEP];
    __shared__ half shared_B[2][STEP][STEP];

    half sum0[STRIDE][STRIDE] = {__float2half(0.0f)};
    half sum1[STRIDE][STRIDE] = {__float2half(0.0f)};
    half sum2[STRIDE][STRIDE] = {__float2half(0.0f)};
    half sum3[STRIDE][STRIDE] = {__float2half(0.0f)};
    half* a_ptr=begin_A, *b_ptr=begin_B;

    #pragma unroll  
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            if (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) {
                shared_A[0][ty + j][tx + i] = a_ptr[(tx + i) * K + (ty + j)];
            } else {
                shared_A[0][ty + j][tx + i] = __float2half(0.0f);
            }
            if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) {
                shared_A[0][ty + j + 32][tx + i] = a_ptr[(tx + i) * K + (ty + j + 32)];
            } else {
                shared_A[0][ty + j + 32][tx + i] = __float2half(0.0f);
            }
            if (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) {
                shared_A[0][ty + j][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j)];
            } else {
                shared_A[0][ty + j][tx + i + 32] = __float2half(0.0f);
            }
            if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) {
                shared_A[0][ty + j + 32][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j + 32)];
            } else {
                shared_A[0][ty + j + 32][tx + i + 32] = __float2half(0.0f);
            }

            if (b_ptr + tx + i < end_b && by * STEP + ty + j < N) {
                shared_B[0][ty + j][tx + i] = b_ptr[(ty + j) * K + (tx + i)];
            } else {
                shared_B[0][ty + j][tx + i] = __float2half(0.0f);
            }
            if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) {
                shared_B[0][ty + j][tx + i + 32] = b_ptr[(ty + j) * K + (tx + i + 32)];
            } else {
                shared_B[0][ty + j][tx + i + 32] = __float2half(0.0f);
            }
            if (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) {
                shared_B[0][ty + j + 32][tx + i] = b_ptr[(ty + j + 32) * K + (tx + i)];
            } else {
                shared_B[0][ty + j + 32][tx + i] = __float2half(0.0f);
            }
            if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) {
                shared_B[0][ty + j + 32][tx + i + 32] = b_ptr[(ty + j + 32) * K + (tx + i + 32)];
            } else {
                shared_B[0][ty + j + 32][tx + i + 32] = __float2half(0.0f);
            }
        }
    }
    for (; a_ptr<end_a;) {
        __syncthreads();
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                if (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) {
                    shared_A[1][ty + j][tx + i] = a_ptr[(tx + i) * K + (ty + j)];
                } else {
                    shared_A[1][ty + j][tx + i] = __float2half(0.0f);
                }
                if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) {
                    shared_A[1][ty + j + 32][tx + i] = a_ptr[(tx + i) * K + (ty + j + 32)];
                } else {
                    shared_A[1][ty + j + 32][tx + i] = __float2half(0.0f);
                }
                if (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) {
                    shared_A[1][ty + j][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j)];
                } else {
                    shared_A[1][ty + j][tx + i + 32] = __float2half(0.0f);
                }
                if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) {
                    shared_A[1][ty + j + 32][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j + 32)];
                } else {
                    shared_A[1][ty + j + 32][tx + i + 32] = __float2half(0.0f);
                }

                if (b_ptr + tx + i < end_b && by * STEP + ty + j < N) {
                    shared_B[1][ty + j][tx + i] = b_ptr[(ty + j) * K + (tx + i)];
                } else {
                    shared_B[1][ty + j][tx + i] = __float2half(0.0f);
                }
                if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) {
                    shared_B[1][ty + j][tx + i + 32] = b_ptr[(ty + j) * K + (tx + i + 32)];
                } else {
                    shared_B[1][ty + j][tx + i + 32] = __float2half(0.0f);
                }
                if (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) {
                    shared_B[1][ty + j + 32][tx + i] = b_ptr[(ty + j + 32) * K + (tx + i)];
                } else {
                    shared_B[1][ty + j + 32][tx + i] = __float2half(0.0f);
                }
                if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) {
                    shared_B[1][ty + j + 32][tx + i + 32] = b_ptr[(ty + j + 32) * K + (tx + i + 32)];
                } else {
                    shared_B[1][ty + j + 32][tx + i + 32] = __float2half(0.0f);
                }
            }
        }
        a_ptr += STEP; b_ptr += STEP;

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    sum0[i][j] = __hadd(sum0[i][j], __hmul(shared_A[0][k][tx + i], shared_B[0][ty + j][k]));
                    sum1[i][j] = __hadd(sum1[i][j], __hmul(shared_A[0][k][tx + i], shared_B[0][ty + j + 32][k]));
                    sum2[i][j] = __hadd(sum2[i][j], __hmul(shared_A[0][k][tx + i + 32], shared_B[0][ty + j][k]));
                    sum3[i][j] = __hadd(sum3[i][j], __hmul(shared_A[0][k][tx + i + 32], shared_B[0][ty + j + 32][k]));
                }
            }
        }
        __syncthreads();
        if (a_ptr < end_a) {
            for (int i=0; i<STRIDE; i++) {
                for (int j=0; j<STRIDE; j++) {
                    if (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) {
                        shared_A[0][ty + j][tx + i] = a_ptr[(tx + i) * K + (ty + j)];
                    } else {
                        shared_A[0][ty + j][tx + i] = __float2half(0.0f);
                    }
                    if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) {
                        shared_A[0][ty + j + 32][tx + i] = a_ptr[(tx + i) * K + (ty + j + 32)];
                    } else {
                        shared_A[0][ty + j + 32][tx + i] = __float2half(0.0f);
                    }
                    if (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) {
                        shared_A[0][ty + j][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j)];
                    } else {
                        shared_A[0][ty + j][tx + i + 32] = __float2half(0.0f);
                    }
                    if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) {
                        shared_A[0][ty + j + 32][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j + 32)];
                    } else {
                        shared_A[0][ty + j + 32][tx + i + 32] = __float2half(0.0f);
                    }

                    if (b_ptr + tx + i < end_b && by * STEP + ty + j < N) {
                        shared_B[0][ty + j][tx + i] = b_ptr[(ty + j) * K + (tx + i)];
                    } else {
                        shared_B[0][ty + j][tx + i] = __float2half(0.0f);
                    }
                    if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) {
                        shared_B[0][ty + j][tx + i + 32] = b_ptr[(ty + j) * K + (tx + i + 32)];
                    } else {
                        shared_B[0][ty + j][tx + i + 32] = __float2half(0.0f);
                    }
                    if (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) {
                        shared_B[0][ty + j + 32][tx + i] = b_ptr[(ty + j + 32) * K + (tx + i)];
                    } else {
                        shared_B[0][ty + j + 32][tx + i] = __float2half(0.0f);
                    }
                    if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) {
                        shared_B[0][ty + j + 32][tx + i + 32] = b_ptr[(ty + j + 32) * K + (tx + i + 32)];
                    } else {
                        shared_B[0][ty + j + 32][tx + i + 32] = __float2half(0.0f);
                    }
                }
            }
            a_ptr += STEP; b_ptr += STEP;
        }
        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    sum0[i][j] = __hadd(sum0[i][j], __hmul(shared_A[1][k][tx + i], shared_B[1][ty + j][k]));
                    sum1[i][j] = __hadd(sum1[i][j], __hmul(shared_A[1][k][tx + i], shared_B[1][ty + j + 32][k]));
                    sum2[i][j] = __hadd(sum2[i][j], __hmul(shared_A[1][k][tx + i + 32], shared_B[1][ty + j][k]));
                    sum3[i][j] = __hadd(sum3[i][j], __hmul(shared_A[1][k][tx + i + 32], shared_B[1][ty + j + 32][k]));
                }
            }
        }
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx + i;
            int col = STEP * by + ty + j;
            if (row < M && col < N) {
                C[row * N + col] = sum0[i][j];
                C[row * N + col + 32] = sum1[i][j];
                C[(row + 32) * N + col] = sum2[i][j];
                C[(row + 32) * N + col + 32] = sum3[i][j];
            }
        }
    }
}

// BLOCK 16 STRIDE 2 34ms 39x (only 1ms slower than sgemm_5)
template <int BLOCK, int STRIDE>
__global__ void sgemm_7(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    constexpr int STEP = BLOCK * STRIDE * 2;

    int tx = threadIdx.x * STRIDE;
    int ty = threadIdx.y * STRIDE;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    half *begin_A = A + bx * STEP * K;
    half *begin_B = B + by * STEP * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    __shared__ half shared_A[STEP][STEP + 1];
    __shared__ half shared_B[STEP + 1][STEP];

    float sum0_fp32[STRIDE][STRIDE] = {0.0f};
    float sum1_fp32[STRIDE][STRIDE] = {0.0f};
    float sum2_fp32[STRIDE][STRIDE] = {0.0f};
    float sum3_fp32[STRIDE][STRIDE] = {0.0f};
    for (half* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=STEP, b_ptr+=STEP) {
        half sum0[STRIDE][STRIDE] = {__float2half(0.0f)};
        half sum1[STRIDE][STRIDE] = {__float2half(0.0f)};
        half sum2[STRIDE][STRIDE] = {__float2half(0.0f)};
        half sum3[STRIDE][STRIDE] = {__float2half(0.0f)};

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                if (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) {
                    shared_A[ty + j][tx + i] = a_ptr[(tx + i) * K + (ty + j)];
                } else {
                    shared_A[ty + j][tx + i] = __float2half(0.0f);
                }
                if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) {
                    shared_A[ty + j + 32][tx + i] = a_ptr[(tx + i) * K + (ty + j + 32)];
                } else {
                    shared_A[ty + j + 32][tx + i] = __float2half(0.0f);
                }
                if (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) {
                    shared_A[ty + j][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j)];
                } else {
                    shared_A[ty + j][tx + i + 32] = __float2half(0.0f);
                }
                if (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) {
                    shared_A[ty + j + 32][tx + i + 32] = a_ptr[(tx + i + 32) * K + (ty + j + 32)];
                } else {
                    shared_A[ty + j + 32][tx + i + 32] = __float2half(0.0f);
                }


                if (b_ptr + tx + i < end_b && by * STEP + ty + j < N) {
                    shared_B[ty + j][tx + i] = b_ptr[(ty + j) * K + (tx + i)];
                } else {
                    shared_B[ty + j][tx + i] = __float2half(0.0f);
                }
                if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) {
                    shared_B[ty + j][tx + i + 32] = b_ptr[(ty + j) * K + (tx + i + 32)];
                } else {
                    shared_B[ty + j][tx + i + 32] = __float2half(0.0f);
                }
                if (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) {
                    shared_B[ty + j + 32][tx + i] = b_ptr[(ty + j + 32) * K + (tx + i)];
                } else {
                    shared_B[ty + j + 32][tx + i] = __float2half(0.0f);
                }
                if (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) {
                    shared_B[ty + j + 32][tx + i + 32] = b_ptr[(ty + j + 32) * K + (tx + i + 32)];
                } else {
                    shared_B[ty + j + 32][tx + i + 32] = __float2half(0.0f);
                }


            }
        }
        __syncthreads();


        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    sum0[i][j] = __hadd(sum0[i][j], __hmul(shared_A[k][tx + i], shared_B[ty + j][k]));
                    sum1[i][j] = __hadd(sum1[i][j], __hmul(shared_A[k][tx + i], shared_B[ty + j + 32][k]));
                    sum2[i][j] = __hadd(sum2[i][j], __hmul(shared_A[k][tx + i + 32], shared_B[ty + j][k]));
                    sum3[i][j] = __hadd(sum3[i][j], __hmul(shared_A[k][tx + i + 32], shared_B[ty + j + 32][k]));
                    // sum0[i][j] = __hadd(sum0[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i], shared_B[ty + j][k]));
                    // sum1[i][j] = __hadd(sum1[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i], shared_B[ty + j + 32][k]));
                    // sum2[i][j] = __hadd(sum2[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i + 32], shared_B[ty + j][k]));
                    // sum3[i][j] = __hadd(sum3[i][j], baseline_approx_kernel_mul_fp16(shared_A[k][tx + i + 32], shared_B[ty + j + 32][k]));
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                sum0_fp32[i][j] += __half2float(sum0[i][j]);
                sum1_fp32[i][j] += __half2float(sum1[i][j]);
                sum2_fp32[i][j] += __half2float(sum2[i][j]);
                sum3_fp32[i][j] += __half2float(sum3[i][j]);
            }
        }
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx + i;
            int col = STEP * by + ty + j;
            if (row < M && col < N) {
                // C[row * N + col] = sum0[i][j];
                // C[row * N + col + 32] = sum1[i][j];
                // C[(row + 32) * N + col] = sum2[i][j];
                // C[(row + 32) * N + col + 32] = sum3[i][j];
                C[row * N + col] = __float2half(sum0_fp32[i][j]);
                C[row * N + col + 32] = __float2half(sum1_fp32[i][j]);
                C[(row + 32) * N + col] = __float2half(sum2_fp32[i][j]);
                C[(row + 32) * N + col + 32] = __float2half(sum3_fp32[i][j]);
            }
        }
    }
}

// BLOCK 16 STRIDE 2 26ms 50x
template <int BLOCK, int STRIDE>
__global__ void sgemm_8(half* A, half* B, half* C, int M, int N, int K) {
    // A [M, K], B [N, K], C [M, N]
    constexpr int STEP = BLOCK * STRIDE * 2;

    int tx = threadIdx.x * STRIDE;
    int ty = threadIdx.y * STRIDE;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    half *begin_A = A + bx * STEP * K;
    half *begin_B = B + by * STEP * K;
    half *end_a = begin_A + K;
    half *end_b = begin_B + K;

    __shared__ half shared_A[STEP][STEP + 1];
    __shared__ half shared_B[STEP + 1][STEP];

    float sum0_fp32[STRIDE][STRIDE] = {0.0f};
    float sum1_fp32[STRIDE][STRIDE] = {0.0f};
    float sum2_fp32[STRIDE][STRIDE] = {0.0f};
    float sum3_fp32[STRIDE][STRIDE] = {0.0f};
    for (half* a_ptr=begin_A, *b_ptr=begin_B; a_ptr<end_a; a_ptr+=STEP, b_ptr+=STEP) {
        half sum0[STRIDE][STRIDE] = {__float2half(0.0f)};
        half sum1[STRIDE][STRIDE] = {__float2half(0.0f)};
        half sum2[STRIDE][STRIDE] = {__float2half(0.0f)};
        half sum3[STRIDE][STRIDE] = {__float2half(0.0f)};

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                shared_A[tx + i][ty + j] = (a_ptr + ty + j < end_a && bx * STEP + tx + i < M) ? a_ptr[(tx + i) * K + (ty + j)] : __float2half(0.0f);
                shared_A[tx + i][ty + j + 32] = (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i < M) ? a_ptr[(tx + i) * K + (ty + j + 32)] : __float2half(0.0f);
                shared_A[tx + i + 32][ty + j] = (a_ptr + ty + j < end_a && bx * STEP + tx + i + 32 < M) ? a_ptr[(tx + i + 32) * K + (ty + j)] : __float2half(0.0f);
                shared_A[tx + i + 32][ty + j + 32] = (a_ptr + ty + j + 32 < end_a && bx * STEP + tx + i + 32 < M) ? a_ptr[(tx + i + 32) * K + (ty + j + 32)] : __float2half(0.0f);

                shared_B[ty + j][tx + i] = (b_ptr + tx + i < end_b && by * STEP + ty + j < N) ? b_ptr[(ty + j) * K + (tx + i)] : __float2half(0.0f);
                shared_B[ty + j][tx + i + 32] = (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j < N) ? b_ptr[(ty + j) * K + (tx + i + 32)] : __float2half(0.0f);
                shared_B[ty + j + 32][tx + i] = (b_ptr + tx + i < end_b && by * STEP + ty + j + 32 < N) ? b_ptr[(ty + j + 32) * K + (tx + i)] : __float2half(0.0f);
                shared_B[ty + j + 32][tx + i + 32] = (b_ptr + tx + i + 32 < end_b && by * STEP + ty + j + 32 < N) ? b_ptr[(ty + j + 32) * K + (tx + i + 32)] : __float2half(0.0f);
            }
        }
        __syncthreads();


        #pragma unroll 4
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    sum0[i][j] = __hadd(sum0[i][j], __hmul(shared_A[tx + i][k], shared_B[ty + j][k]));
                    sum1[i][j] = __hadd(sum1[i][j], __hmul(shared_A[tx + i][k], shared_B[ty + j + 32][k]));
                    sum2[i][j] = __hadd(sum2[i][j], __hmul(shared_A[tx + i + 32][k], shared_B[ty + j][k]));
                    sum3[i][j] = __hadd(sum3[i][j], __hmul(shared_A[tx + i + 32][k], shared_B[ty + j + 32][k]));
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                sum0_fp32[i][j] += __half2float(sum0[i][j]);
                sum1_fp32[i][j] += __half2float(sum1[i][j]);
                sum2_fp32[i][j] += __half2float(sum2[i][j]);
                sum3_fp32[i][j] += __half2float(sum3[i][j]);
            }
        }
    }
    
    for (int i=0; i<STRIDE; i++) {
        for (int j=0; j<STRIDE; j++) {
            int row = STEP * bx + tx + i;
            int col = STEP * by + ty + j;
            C[row * N + col] = (row < M && col < N) ? __float2half(sum0_fp32[i][j]) : __float2half(0.0f);
            C[row * N + col + 32] = (row < M && col + 32 < N) ? __float2half(sum1_fp32[i][j]) : __float2half(0.0f);
            C[(row + 32) * N + col] = (row + 32 < M && col < N) ? __float2half(sum2_fp32[i][j]) : __float2half(0.0f);
            C[(row + 32) * N + col + 32] = (row + 32 < M && col + 32 < N) ? __float2half(sum3_fp32[i][j]) : __float2half(0.0f);
        }
    }
}



template <int BM, int BN, int BK, int TM, int TN>
__global__ void hgemm_9(half* dA, half* dB, half* dC, int M, int N, int K) {
    // A [M. K], B [K, N], C [M, N]
    constexpr int SUB_K = 128;
    __shared__ half SA[BM * BK];
    __shared__ half SB[BK * BN];
    int indA = TM * blockIdx.x * blockDim.x;
    int indB = TN * blockIdx.y * blockDim.y;
    int width = (K + BK - 1) / BK;
    float tmp_fp32[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid % 128;
    int smem_a_k = tid / 128;
    int smem_b_k = tid % 8;
    int smem_b_n = tid / 8;

    int accumulated_k = 0;
    half tmp[TM * TN] = {__float2half(0.0f)};
    for (int ph = 0; ph < width; ph++) {
        
        if (indA + smem_a_m < M && smem_a_k + ph * BK < K) {
            SA[smem_a_m * BK + smem_a_k] = dA[(indA + smem_a_m) * K + smem_a_k + ph * BK];
        }
        else {
            SA[smem_a_m * BK + smem_a_k] = __float2half(0.0f);
        }
        if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {
            SB[smem_b_k * BN + smem_b_n] = dB[(smem_b_k + ph * BK) * N + smem_b_n + indB];
        }
        else {
            SB[smem_b_k * BN + smem_b_n] = __float2half(0.0f);
        }
        __syncthreads();

        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    tmp[index_m * TN + index_n] = __hadd(__hmul(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                }
            }

            accumulated_k++;
            if (accumulated_k == SUB_K) {
                #pragma unroll
                for (int i = 0; i < TM*TN; i++) {
                    tmp_fp32[i] += __half2float(tmp[i]);
                    tmp[i] = __float2half(0.0f);
                }
                accumulated_k = 0;
            }
        }
        __syncthreads();
    }

    if (accumulated_k > 0) {
        #pragma unroll
        for (int i = 0; i < TM*TN; i++) {
            tmp_fp32[i] += __half2float(tmp[i]);
        }
    }

    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            if (indA + index_m < M && indB + index_n < N) {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = __float2half(tmp_fp32[index_m * TN + index_n]);
                // dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_m * TN + index_n];
            }
        }
    }
    
}

// 10.92.254.206 123456
void launch_fp_gemm_kernel_fp16(half* A, half* B, half* C, int M, int N, int K) {
    constexpr int BLOCK_SIZE = 16;
    // dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 blocks_per_grid(
    //     (M + BLOCK_SIZE - 1) / BLOCK_SIZE,  
    //     (N + BLOCK_SIZE - 1) / BLOCK_SIZE 
    // );

    // constexpr int K_till = 32;
    // constexpr int MN_till = 8;
    constexpr int STRIDE = 2;
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid(
        (M + BLOCK_SIZE*STRIDE*2 - 1) / BLOCK_SIZE / STRIDE / 2,  
        (N + BLOCK_SIZE*STRIDE*2 - 1) / BLOCK_SIZE / STRIDE / 2 
    );
    // dim3 threads_per_block(K_till, MN_till);
    // dim3 blocks_per_grid(
    //     (M + MN_till*STRIDE - 1) / MN_till / STRIDE,  
    //     (N + MN_till*STRIDE - 1) / MN_till / STRIDE
    // );
    // baseline_approx_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // sgemm_2<BLOCK_SIZE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // sgemm_3<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // sgemm_4<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // sgemm_5<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // sgemm_6<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // sgemm_7<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    sgemm_8<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    // dim3 threads_per_block(8, 8);
    // dim3 blocks_per_grid((M + 127) / 128, (N + 127) / 128);
    // sgemm_10<BLOCK_SIZE, STRIDE><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void launch_group_gemm_kernel(half* A, half* B, half* C, int M, int N, int K) {
    const int TM = 4;
    const int TN = 4;
    const int BLOCK_DIM_x = 32;
    const int BLOCK_DIM_y = 32;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 8;

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    hgemm_9<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}