#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath> // for isfinite
#include "AxCore_utils.cuh"

__device__ half dequantize_approx_kernel_fp16(half a, half b, int16_t comp) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;
    
    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);
    
    uint16_t threshold = 0x3C00 + comp;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask); 

    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    
    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

__device__ half ablation_approx_kernel_mul_fp16_e2m1(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t is_subnormal = (b_bits & 0x0C00) == 0;  // for FP16-FP4

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    a_bits = a_bits & 0x7FFF;
    b_bits = b_bits & 0x7FFF;

    uint16_t mantissa_sum = a_bits + b_bits;
    
    // FP16 Baseline: 0x3C00
    uint16_t threshold = 0x0400; // for FP16-FP4
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask); 
    // adjusted_mantissa = diff;
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

__device__ half ablation_approx_kernel_mul_fp16_e2m1_S(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t is_subnormal = (b_bits & 0x0C00) == 0;  // for FP16-FP4

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    a_bits = a_bits & 0x7FFF;
    b_bits = (is_subnormal) ? 0 : (b_bits & 0x7FFF);

    uint16_t mantissa_sum = a_bits + b_bits;
    
    // FP16 Baseline: 0x3C00
    uint16_t threshold = 0x0400; // for FP16-FP4
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask); 
    // adjusted_mantissa = diff;
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

__device__ half ablation_approx_kernel_mul_fp16_e2m1_S_C(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t is_subnormal = (b_bits & 0x0C00) == 0;  // for FP16-FP4

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    a_bits = a_bits & 0x7FFF;
    b_bits = (is_subnormal) ? 0 : (b_bits & 0x7FFF);
    // b_bits = b_bits & 0x7FFF;

    uint16_t mantissa_sum = a_bits + b_bits;
    
    // FP16 Baseline: 0x3C00
    uint16_t threshold = 0x0400 - 43; // for FP16-FP4
    // uint16_t threshold = 0x0400; // for FP16-FP4
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask); 
    // adjusted_mantissa = diff;
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

// =============================================================
//  New kernel: group_GEmpFPMA_GEMMMM_typed
//  Supports different optimizations in ablation study
//  opt: 0: baseline, 1: subnormal handling, 2: subnormal handling + comp
// =============================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void mpFPMA_GEMM(half* dA, half* dB, half* dC, half* dS, int M, int N, int K, int opt) {
    // A [M. K], B [K, N], C [M, N], S [K//g, N]
    constexpr int SUB_K = 128;
    constexpr int g = SUB_K;
    __shared__ half SA[BM * BK];
    __shared__ half SB[BK * BN];
    __shared__ half SS[1 * BN];
    
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
    // half tmp_scaled[TM * TN] = {__float2half(0.0f)};
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
        if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {
            int global_k = ph * BK + smem_b_k;
            int s_index = (global_k)/g;
            SS[smem_b_n] = dS[s_index * N + smem_b_n + indB];
        }
        else {
            SS[smem_b_n] = __float2half(0.0f);
        }
        __syncthreads();

        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    if (opt == 0) {
                        tmp[index_m * TN + index_n] = __hadd(ablation_approx_kernel_mul_fp16_e2m1(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                    }
                    else if (opt == 1) {
                        tmp[index_m * TN + index_n] = __hadd(ablation_approx_kernel_mul_fp16_e2m1_S(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                    }
                    else if (opt == 2) {
                        tmp[index_m * TN + index_n] = __hadd(ablation_approx_kernel_mul_fp16_e2m1_S_C(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                    }
                }
            }
            accumulated_k++;
            if (accumulated_k == SUB_K) {
                #pragma unroll
                for (int i = 0; i < TM*TN; i++) {
                    int col_idx = threadIdx.y * TN + (i % TN);
                    if (opt == 0 || opt == 1) {
                        tmp[i] = __hmul(tmp[i], SS[col_idx]);
                    }
                    else if (opt == 2) {
                        tmp[i] = dequantize_approx_kernel_fp16(tmp[i], SS[col_idx], 0);
                    }
                    tmp_fp32[i] += __half2float(tmp[i]);
                    // tmp_fp32[i] += __half2float(tmp[i]);
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
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                dC[global_m * N + global_n] =
                    __float2half(tmp_fp32[index_m * TN + index_n]);
            }
        }
    }
}

void launch_mpFPMA_gemm_kernel_fp16(half* A, half* B, half* C, half* S, int M, int N, int K, int opt, int device_id) {
    cudaSetDevice(device_id);
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
    mpFPMA_GEMM<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, S, M, N, K, opt);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
