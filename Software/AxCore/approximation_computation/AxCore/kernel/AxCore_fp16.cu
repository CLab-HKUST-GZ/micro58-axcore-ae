#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath> // for isfinite

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


__device__ half baseline_approx_kernel_mul_fp16_e2m1(half a, half b) {
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

__device__ half baseline_approx_kernel_mul_fp16_e1m2(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;

    uint16_t is_subnormal = (b_bits & 0x0400) == 0;  // for FP16-FP4
    uint16_t snc1 = is_subnormal && (b_bits & 0x0300) == 0x0100;
    uint16_t snc2 = is_subnormal && (b_bits & 0x0300) == 0x0200;
    uint16_t snc3 = is_subnormal && (b_bits & 0x0300) == 0x0300;

    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    a_bits = a_bits & 0x7FFF;
    b_bits = (is_subnormal) ? (snc1 ? 0x0000 : (snc2 ? 0x0000 : (snc3 ? 0x0200 : 0))) : (b_bits & 0x7FFF);
    // b_bits = b_bits & 0x7FFF;

    uint16_t mantissa_sum = a_bits + b_bits;
    
    // FP16 Baseline: 0x3C00
    uint16_t threshold = 0x0000; // for FP16-FP4
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold + 54 - (snc1 ? 0x0400 : 0); 
    uint16_t is_negative_mask = -(mantissa_sum < threshold - 54 + (snc1 ? 0x0400 : 0));
    adjusted_mantissa = diff & (~is_negative_mask);

    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;

    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

__device__ half dequantize_approx_kernel_mul_fp16(half a, half b, int16_t comp) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;
    uint16_t is_zero = is_zero_a | is_zero_b;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;
    
    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);
    
    // FP16 Baseline: 0x3C00
    // uint16_t threshold = 0x3C00;
    // uint16_t threshold = 0x3C00 - 20;
    uint16_t threshold = 0x3C00 + comp;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask); 
    // adjusted_mantissa = diff;
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FE; 
    // }
    
    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

// 43
template <int BM, int BN, int BK, int TM, int TN>
__global__ void group_GEMM(half* dA, half* dB, half* dC, half* dS, int M, int N, int K) {
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
                    tmp[index_m * TN + index_n] = __hadd(baseline_approx_kernel_mul_fp16_e1m2(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                    // tmp[index_m * TN + index_n] = __hadd(baseline_approx_kernel_mul_fp16_e2m1(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                    // tmp[index_m * TN + index_n] = __hadd(__hmul(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                }
            }

            accumulated_k++;
            if (accumulated_k == SUB_K) {
                #pragma unroll
                for (int i = 0; i < TM*TN; i++) {
                    int col_idx = threadIdx.y * TN + (i % TN);
                    // tmp[i] = dequantize_approx_kernel_mul_fp16(tmp[i], SS[col_idx], 0);
                    tmp[i] = __hmul(tmp[i], SS[col_idx]);
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
            // if (indA + index_m < M && indB + index_n < N) {
            //     dC[(indA + reg_c_m) * N + indB + reg_c_n] = __float2half(tmp_fp32[index_m * TN + index_n]);
            //     // dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_m * TN + index_n];
            // }
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                dC[global_m * N + global_n] =
                    __float2half(tmp_fp32[index_m * TN + index_n]);
            }
        }
    }
}

void launch_AxCore_group_gemm_kernel_fp16(half* A, half* B, half* C, half* S, int M, int N, int K, int device_id) {
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
    group_GEMM<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, S, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// =============================================================
//  New kernel: group_GEMM_typed
//  Supports per-tile datatype selection (e3m0 / e2m1 / e1m2)
//  dT: unsigned char matrix of shape [K/G, N/64] (value 0,1,2)
//  G: group size (64 or 128)
// =============================================================
template <int BM, int BN, int BK, int TM, int TN, int G>
__global__ void group_GEMM_typed(half* dA, half* dB, half* dC,
                                 half* dS, unsigned char* dT,
                                 int M, int N, int K, bool use_approx_dequant = true) {
    constexpr int SUB_K = G;                  // accumulate G fp16 MACs before fp32 cast (simulate HW MAC array)
    constexpr int g      = SUB_K;
    constexpr int n_tile = 64;                // column group size for dT

    // shared memory tiles
    __shared__ half           SA[BM * BK];
    __shared__ half           SB[BK * BN];
    __shared__ half           SS[1 * BN];     // scaling factors (per col)
    __shared__ unsigned char  ST[BN];         // datatype flag (per col)

    // global tile indices
    int indA  = TM * blockIdx.x * blockDim.x;   // row start in A/C
    int indB  = TN * blockIdx.y * blockDim.y;   // col start in B/C,S,T

    int width = (K + BK - 1) / BK;              // #k-tiles

    // thread id decomposition (same as original kernel)
    int tid        = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m   = tid % 128;
    int smem_a_k_base = tid / 128;   // 0..3 when 512 threads, need two loads (k and k+4)
    int smem_b_k   = tid % 8;
    int smem_b_n   = tid / 8;

    float tmp_fp32[TM * TN] = {0.0f};           // accumulation buffer in fp32

    // low-precision accumulation buffer (fp16) and counter
    half  tmp[TM * TN] = {__float2half(0.0f)};
    int   accumulated_k = 0;

    // iterate over k tiles
    for (int ph = 0; ph < width; ++ph) {
        // === load A tile (each thread loads two K elements to cover BK=8 with 512 threads) ===
        #pragma unroll
        for (int offs = 0; offs < 2; ++offs) {
            int smem_a_k = smem_a_k_base + offs * 4;   // 0..7
            if (smem_a_k < BK) {
                if (indA + smem_a_m < M && smem_a_k + ph * BK < K) {
                    SA[smem_a_m * BK + smem_a_k] =
                        dA[(indA + smem_a_m) * K + smem_a_k + ph * BK];
                } else {
                    SA[smem_a_m * BK + smem_a_k] = __float2half(0.0f);
                }
            }
        }
        // === load B tile ===
        if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {
            SB[smem_b_k * BN + smem_b_n] =
                dB[(smem_b_k + ph * BK) * N + smem_b_n + indB];
        } else {
            SB[smem_b_k * BN + smem_b_n] = __float2half(0.0f);
        }
        // === load S and T ===
        if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {
            int global_k  = ph * BK + smem_b_k;
            int s_index   = global_k / g;                    // row in S/T
            SS[smem_b_n]  = dS[s_index * N + smem_b_n + indB];
            int col_global = indB + smem_b_n;
            int t_col      = col_global / n_tile;            // column in T
            ST[smem_b_n]  = dT[s_index * (N / n_tile) + t_col];
        } else {
            SS[smem_b_n] = __float2half(0.0f);
            ST[smem_b_n] = 2; // default to e3m0
        }
        __syncthreads();

        // === compute BK inner product ===
        #pragma unroll
        for (int index_k = 0; index_k < BK; ++index_k) {
            #pragma unroll
            for (int index_m = 0; index_m < TM; ++index_m) {
                #pragma unroll
                for (int index_n = 0; index_n < TN; ++index_n) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    half a_val  = SA[reg_c_m * BK + index_k];
                    half b_val  = SB[index_k * BN + reg_c_n];
                    unsigned char type_id = ST[reg_c_n];
                    half prod;
                    if (type_id == 0) {
                        prod = __hmul(a_val, b_val); 
                    } else if (type_id == 1) {
                        // prod = __hmul(a_val, b_val);
                        prod = baseline_approx_kernel_mul_fp16_e2m1(a_val, b_val);
                    } else {
                        // prod = __hmul(a_val, b_val);
                        prod = baseline_approx_kernel_mul_fp16_e1m2(a_val, b_val);
                    }
                    // accumulate in fp16 first (un-scaled)
                    tmp[index_m * TN + index_n] = __hadd(prod, tmp[index_m * TN + index_n]);
                }
            }

            accumulated_k++;
            if (accumulated_k == SUB_K) {
                #pragma unroll
                for (int i = 0; i < TM * TN; ++i) {
                    int col_idx = threadIdx.y * TN + (i % TN);
                    half scaled = use_approx_dequant ? dequantize_approx_kernel_mul_fp16(tmp[i], SS[col_idx], 0) : __hmul(tmp[i], SS[col_idx]);
                    // half scaled = use_approx_dequant ? dequantize_approx_kernel_mul_fp16(tmp[i], SS[col_idx], 0) : dequantize_approx_kernel_mul_fp16(tmp[i], SS[col_idx], -58);
                    tmp_fp32[i] += __half2float(scaled);
                    tmp[i] = __float2half(0.0f);
                }
                accumulated_k = 0;
            }
        }
        __syncthreads();
    }

    // flush the remainder (<SUB_K) for this k-tile if any
    if (accumulated_k > 0) {
        #pragma unroll
        for (int i = 0; i < TM * TN; ++i) {
            int col_idx = threadIdx.y * TN + (i % TN);
            // half scaled = __hmul(tmp[i], SS[col_idx]);
            // half scaled = dequantize_approx_kernel_mul_fp16(tmp[i], SS[col_idx], -58);
            // tmp_fp32[i] += __half2float(scaled);
            tmp_fp32[i] += __half2float(tmp[i]);
            tmp[i] = __float2half(0.0f);
        }
        accumulated_k = 0;
    }

    // write C
    #pragma unroll
    for (int index_m = 0; index_m < TM; ++index_m) {
        #pragma unroll
        for (int index_n = 0; index_n < TN; ++index_n) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            int global_m = indA + reg_c_m;
            int global_n = indB + reg_c_n;
            if (global_m < M && global_n < N) {
                dC[global_m * N + global_n] = __float2half(tmp_fp32[index_m * TN + index_n]);
            }
        }
    }
}

// -------------------------------------------------------------
// Host launcher for the typed kernel
// -------------------------------------------------------------
void launch_AxCore_group_gemm_typed_kernel_fp16(
        half* A, half* B, half* C, half* S, unsigned char* T, int g,
        int M, int N, int K, int device_id, bool use_approx_dequant = true) {
    cudaSetDevice(device_id);
    const int TM = 4;
    const int TN = 4;
    const int BLOCK_DIM_x = 32;
    const int BLOCK_DIM_y = 16;  // reduce threads per block to 512 (32x16) to stay within register budget
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 8;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim((M + BM - 1) / BM, (N + BN - 1) / BN, 1);

    // Dispatch to appropriate kernel based on group size
    if (g == 64) {
        group_GEMM_typed<BM, BN, BK, TM, TN, 64><<<grid_dim, block_dim>>>(
            A, B, C, S, T, M, N, K, use_approx_dequant);
    } else if (g == 128) {
        group_GEMM_typed<BM, BN, BK, TM, TN, 128><<<grid_dim, block_dim>>>(
            A, B, C, S, T, M, N, K, use_approx_dequant);
    } else {
        throw std::runtime_error("Unsupported group size. Only g=64 and g=128 are supported.");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}