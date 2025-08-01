#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h> // For uint16_t

#define BLOCK_SIZE 32
#define TILE_SIZE 32


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
    
    uint16_t mantissa_sum = (a_bits & 0x7FFF) + (b_bits & 0x7FFF);
    
    // FP16 Baseline: 0x3C00
    // uint16_t threshold = 0x3C00;
    uint16_t threshold = 0x3C00 - 58;
    // uint16_t threshold = 0x3C00;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_sum - threshold; 
    uint16_t is_negative_mask = -(mantissa_sum < threshold);
    adjusted_mantissa = diff & (~is_negative_mask); 
    // adjusted_mantissa = diff;
    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    // uint16_t exponent_fp16 = (result_bits >> 10) & 0x001F;

    // if (exponent_fp16 == 0x001F) {
    //     result_bits = sign | (0x001E << 10) | 0x03FF; 
    // }
    
    result_bits = is_zero ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

__device__ half baseline_approx_kernel_div_fp16(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    // uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;

    // if (is_zero_b) {
    //     return uint16_to_float16(a_bits);
    // }

    // uint16_t s1 = a_bits & 0x8000;
    // uint16_t s2 = b_bits & 0x8000;
    // uint16_t sign = s1 ^ s2;
    
    // uint16_t mantissa_div = (a_bits & 0x7FFF) - (b_bits & 0x7FFF);
    uint16_t mantissa_div = a_bits - b_bits;

    // FP16 Baseline: 0x3C00
    // uint16_t threshold = 0x3C00;
    // uint16_t threshold = 0x3C00 - 20;
    uint16_t threshold = 0x3C00;
    uint16_t adjusted_mantissa;
    uint16_t diff = mantissa_div + threshold; 
    adjusted_mantissa = diff;

    // Combine the sign and adjusted mantissa
    // uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;
    uint16_t result_bits = adjusted_mantissa;
    
    result_bits = is_zero_a ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

__device__ half baseline_approx_kernel_add_without_subnormal_rtz(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);
    
    // if ((a_bits & 0x7FFF) == 0) {
    //     return b;
    // } else if ((b_bits & 0x7FFF) == 0) {
    //     return a;
    // }
    
    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t diff_sign = s1 ^ s2;
    uint16_t e1 = (a_bits >> 10) & 0x001F;
    uint16_t e2 = (b_bits >> 10) & 0x001F;
    uint16_t m1 = (a_bits & 0x03FF) | 0x0400;
    uint16_t m2 = (b_bits & 0x03FF) | 0x0400;
    
    uint16_t mask = e2 > e1;
    uint16_t e_mask = e2 == e1;
    uint16_t m_mask = m2 > m1;
    uint16_t e_max = mask ? e2 : e1;
    uint16_t e_min = mask ? e1 : e2;
    uint16_t s_max = e_mask ? (m_mask ? s2 : s1) : mask ? s2 : s1;
    uint16_t m_max = e_mask ? (m_mask ? m2 : m1) : (mask ? m2 : m1);
    uint16_t m_min = e_mask ? (m_mask ? m1 : m2) : (mask ? m1 : m2);
    

    uint16_t delta_e = e_max - e_min;
    // uint16_t aligned_m_min = (delta_e < 11) ? (m_min >> delta_e) : 0;
    uint16_t aligned_m_min = m_min >> delta_e;

    uint16_t sum_m = diff_sign ? (m_max - aligned_m_min) : (m_max + aligned_m_min);

    uint16_t result_sign = diff_sign ? s_max : s1;

    if (diff_sign) {
        if (sum_m == 0) {
            return uint16_to_float16(0);
        }
        int leading_zeros = __clz((unsigned int)sum_m) - 16;
        int norm_shift = leading_zeros - 5;
        if (norm_shift > 0) {
            norm_shift = (norm_shift > e_max) ? e_max : norm_shift;
            sum_m <<= norm_shift;
            e_max -= norm_shift;
        }
        // if (e_max == 0) {
        //     return uint16_to_float16(result_sign);
        // }
        return uint16_to_float16(result_sign | (e_max << 10) | (sum_m & 0x03FF));
    } else {
        if (sum_m > 0x07FF) {
            sum_m >>= 1;
            e_max += 1;
        }
        if (e_max >= 31) {
            return uint16_to_float16(result_sign | (0x001E << 10) | 0x03FF);
        }
        return uint16_to_float16(result_sign | (e_max << 10) | (sum_m & 0x03FF));
    }
}

__device__ half baseline_approx_kernel_add_without_subnormal_rtn(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);
    
    if ((a_bits & 0x7FFF) == 0) {
        return b;
    } else if ((b_bits & 0x7FFF) == 0) {
        return a;
    }
    
    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t diff_sign = s1 ^ s2;
    uint16_t e1 = (a_bits >> 10) & 0x001F;
    uint16_t e2 = (b_bits >> 10) & 0x001F;
    uint16_t m1 = ((a_bits & 0x03FF) | 0x0400);
    uint16_t m2 = ((b_bits & 0x03FF) | 0x0400);
    
    uint16_t mask = e2 > e1;
    uint16_t e_mask = e2 == e1;
    uint16_t m_mask = m2 > m1;
    uint16_t e_max = mask ? e2 : e1;
    uint16_t e_min = mask ? e1 : e2;
    uint16_t s_max = e_mask ? (m_mask ? s2 : s1) : mask ? s2 : s1;
    uint16_t m_max = e_mask ? (m_mask ? m2 : m1) : (mask ? m2 : m1);
    uint16_t m_min = e_mask ? (m_mask ? m1 : m2) : (mask ? m1 : m2);    
    m_max <<= 1;
    m_min <<= 1;
    

    uint16_t delta_e = e_max - e_min;
    // uint16_t aligned_m_min = (delta_e < 11) ? (m_min >> delta_e) : 0;
    uint16_t aligned_m_min = m_min >> delta_e;

    uint16_t sum_m = diff_sign ? (m_max - aligned_m_min) : (m_max + aligned_m_min);
    
    uint16_t result_sign = diff_sign ? s_max : s1;

    if (diff_sign) {
        if (sum_m == 0) {
            return uint16_to_float16(0);
        }
        // int leading_zeros = __clz((unsigned int)sum_m) - 16;
        // int norm_shift = leading_zeros - 4;
        int norm_shift = __clz((unsigned int)sum_m) - 20;
        if (norm_shift > 0) {
            norm_shift = (norm_shift > e_max) ? e_max : norm_shift;
            sum_m <<= norm_shift;
            e_max -= norm_shift;
        }
        // if (e_max == 0) {
        //     return uint16_to_float16(result_sign);
        // }
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x07FE) >> 1));
    } else {
        sum_m = (sum_m & 0x0001) ? (sum_m + 1) : sum_m; // round to nearest
        if (sum_m > 0x0FFF) {
            sum_m >>= 1;
            e_max += 1;
        }
        if (e_max >= 31) {
            return uint16_to_float16(result_sign | (0x001E << 10) | 0x03FF);
        }
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x07FE) >> 1));
    }
}

// __device__ half baseline_approx_kernel_add_without_subnormal_efn(half a, half b) {
//     uint16_t a_bits = float16_to_uint16(a);
//     uint16_t b_bits = float16_to_uint16(b);
    
//     // if ((a_bits & 0x7FFF) == 0) {
//     //     return b;
//     // } else if ((b_bits & 0x7FFF) == 0) {
//     //     return a;
//     // }
    
//     uint16_t s1 = a_bits & 0x8000;
//     uint16_t s2 = b_bits & 0x8000;
//     uint16_t diff_sign = s1 ^ s2;
//     uint16_t e1 = (a_bits >> 10) & 0x001F;
//     uint16_t e2 = (b_bits >> 10) & 0x001F;
//     uint16_t m1 = ((a_bits & 0x03FF) | 0x0400);
//     uint16_t m2 = ((b_bits & 0x03FF) | 0x0400);
    
//     uint16_t mask = e2 > e1;
//     uint16_t e_mask = e2 == e1;
//     uint16_t m_mask = m2 > m1;
//     uint16_t e_max = mask ? e2 : e1;
//     uint16_t e_min = mask ? e1 : e2;
//     uint16_t s_max = e_mask ? (m_mask ? s2 : s1) : mask ? s2 : s1;
//     uint16_t m_max = e_mask ? (m_mask ? m2 : m1) : (mask ? m2 : m1);
//     uint16_t m_min = e_mask ? (m_mask ? m1 : m2) : (mask ? m1 : m2);
//     m_max <<= 2; // 13bits
//     m_min <<= 2;
    

//     uint16_t delta_e = e_max - e_min;
//     // uint16_t aligned_m_min = (delta_e < 11) ? (m_min >> delta_e) : 0;
//     uint16_t aligned_m_min = m_min >> delta_e;

//     uint16_t sum_m = diff_sign ? (m_max - aligned_m_min) : (m_max + aligned_m_min); // 14 bits
    
//     // efficient rounding
//     uint16_t A = sum_m & 0x000C;
//     uint16_t B = sum_m & 0x0003;
    // uint16_t A_mask = A == 0x000C;
    // A = (A_mask) ? A : (B & 0x0002) ? (A + 4) : A;
    // sum_m = (sum_m & 0xFFF0) | A;

//     uint16_t result_sign = diff_sign ? s_max : s1;

//     if (diff_sign) {
//         if (sum_m == 0) {
//             return uint16_to_float16(0);
//         }
//         int leading_zeros = __clz((unsigned int)sum_m) - 16;
//         int norm_shift = leading_zeros - 3;
//         if (norm_shift > 0) {
//             norm_shift = (norm_shift > e_max) ? e_max : norm_shift;
//             sum_m <<= norm_shift;
//             e_max -= norm_shift;
//         }
//         // if (e_max == 0) {
//         //     return uint16_to_float16(result_sign);
//         // }
//         return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x0FFC) >> 2));
//     } else {
        
//         if (sum_m > 0x1FFF) {
//             sum_m >>= 1;
//             e_max += 1;
//         }
//         if (e_max >= 31) {
//             return uint16_to_float16(result_sign | (0x001E << 10) | 0x03FF);
//         }
//         return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x0FFC) >> 2));
//     }
// }

__device__ half baseline_approx_kernel_add_without_subnormal_efn(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);
    
    // if ((a_bits & 0x7FFF) == 0) {
    //     return b;
    // } else if ((b_bits & 0x7FFF) == 0) {
    //     return a;
    // }
    
    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t diff_sign = s1 ^ s2;
    uint16_t e1 = (a_bits >> 10) & 0x001F;
    uint16_t e2 = (b_bits >> 10) & 0x001F;
    uint16_t m1 = ((a_bits & 0x03FF) | 0x0400);
    uint16_t m2 = ((b_bits & 0x03FF) | 0x0400);
    
    uint16_t mask = e2 > e1;
    uint16_t e_mask = e2 == e1;
    uint16_t m_mask = m2 > m1;
    uint16_t e_max = mask ? e2 : e1;
    uint16_t e_min = mask ? e1 : e2;
    uint16_t s_max = e_mask ? (m_mask ? s2 : s1) : mask ? s2 : s1;
    uint16_t m_max = e_mask ? (m_mask ? m2 : m1) : (mask ? m2 : m1);
    uint16_t m_min = e_mask ? (m_mask ? m1 : m2) : (mask ? m1 : m2);
    m_max <<= 1; // 12bits
    m_min <<= 1;
    

    uint16_t delta_e = e_max - e_min;
    // uint16_t aligned_m_min = (delta_e < 11) ? (m_min >> delta_e) : 0;
    uint16_t aligned_m_min = m_min >> delta_e;

    uint16_t sum_m = diff_sign ? (m_max - aligned_m_min) : (m_max + aligned_m_min); // 13 bits
    uint16_t A = sum_m & 0x0002;
    uint16_t B = sum_m & 0x0001;
    // efficient rounding
    // sum_m = sum_m | A | (B << 1);
    sum_m = sum_m | A | (B << 1);
    uint16_t result_sign = diff_sign ? s_max : s1;

    if (diff_sign) {
        if (sum_m == 0) {
            return uint16_to_float16(0);
        }
        int leading_zeros = __clz((unsigned int)sum_m) - 16;
        int norm_shift = leading_zeros - 4;
        if (norm_shift > 0) {
            norm_shift = (norm_shift > e_max) ? e_max : norm_shift;
            sum_m <<= norm_shift;
            e_max -= norm_shift;
        }
        // if (e_max == 0) {
        //     return uint16_to_float16(result_sign);
        // }
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x07FE) >> 1));
    } else {
        
        if (sum_m > 0x0FFF) {
            sum_m >>= 1;
            e_max += 1;
        }
        if (e_max >= 31) {
            return uint16_to_float16(result_sign | (0x001E << 10) | 0x03FF);
        }
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x07FE) >> 1));
    }
}

__device__ half baseline_approx_kernel_add_without_subnormal_tte(half a, half b) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);
    # define shift_bits 3
    if ((a_bits & 0x7FFF) == 0) {
        return b;
    } else if ((b_bits & 0x7FFF) == 0) {
        return a;
    }
    
    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t diff_sign = s1 ^ s2;
    uint16_t e1 = (a_bits >> 10) & 0x001F;
    uint16_t e2 = (b_bits >> 10) & 0x001F;
    uint16_t m1 = ((a_bits & 0x03FF) | 0x0400);
    uint16_t m2 = ((b_bits & 0x03FF) | 0x0400);
    
    uint16_t mask = e2 > e1;
    uint16_t e_mask = e2 == e1;
    uint16_t m_mask = m2 > m1;
    uint16_t e_max = mask ? e2 : e1;
    uint16_t e_min = mask ? e1 : e2;
    uint16_t s_max = e_mask ? (m_mask ? s2 : s1) : mask ? s2 : s1;
    uint16_t m_max = e_mask ? (m_mask ? m2 : m1) : (mask ? m2 : m1);
    uint16_t m_min = e_mask ? (m_mask ? m1 : m2) : (mask ? m1 : m2);
    m_max <<= shift_bits;
    m_min <<= shift_bits;
    

    uint16_t delta_e = e_max - e_min;
    // uint16_t aligned_m_min = (delta_e < 11) ? (m_min >> delta_e) : 0;
    uint16_t aligned_m_min = m_min >> delta_e;

    uint16_t sum_m = diff_sign ? (m_max - aligned_m_min) : (m_max + aligned_m_min);
    uint16_t rounding_mask = sum_m & 0x0007;
    uint16_t odd = sum_m & 0x0008;
    sum_m = (rounding_mask > 4) ? (sum_m + 8) : (rounding_mask < 4) ? sum_m : (odd) ? (sum_m + 8) : sum_m; // round to nearest
    uint16_t result_sign = diff_sign ? s_max : s1;

    if (diff_sign) {
        if (sum_m == 0) {
            return uint16_to_float16(0);
        }
        int leading_zeros = __clz((unsigned int)sum_m) - 16;
        int norm_shift = leading_zeros - 5 + shift_bits;
        if (norm_shift > 0) {
            norm_shift = (norm_shift > e_max) ? e_max : norm_shift;
            sum_m <<= norm_shift;
            e_max -= norm_shift;
        }
        // if (e_max == 0) {
        //     return uint16_to_float16(result_sign);
        // }
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x1FF8) >> shift_bits));
    } else {
        if (sum_m > 0x3FFF) {
            sum_m >>= 1;
            e_max += 1;
        }
        if (e_max >= 31) {
            return uint16_to_float16(result_sign | (0x001E << 10) | 0x03FF);
        }
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x1FF8) >> shift_bits));
    }
}


// Optimized GEMM kernel with tiling, shared memory, and loop unrolling
template <int BLOCK, int STRIDE>
__global__ void baseline_approx_kernel_fp16_optimized(
    half*  A, // A [M, K]
    half*  B, // B [N, K]
    half*  C,
    const int M, const int N, const int K) {
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


        #pragma unroll
        for (int i=0; i<STRIDE; i++) {
            for (int j=0; j<STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    // sum0[i][j] = __hadd(sum0[i][j], __hmul(shared_A[k][tx + i], shared_B[ty + j][k]));
                    // sum1[i][j] = __hadd(sum1[i][j], __hmul(shared_A[k][tx + i], shared_B[ty + j + 32][k]));
                    // sum2[i][j] = __hadd(sum2[i][j], __hmul(shared_A[k][tx + i + 32], shared_B[ty + j][k]));
                    // sum3[i][j] = __hadd(sum3[i][j], __hmul(shared_A[k][tx + i + 32], shared_B[ty + j + 32][k]));
                    sum0[i][j] = __hadd(sum0[i][j], baseline_approx_kernel_mul_fp16(shared_A[tx + i][k], shared_B[ty + j][k]));
                    sum1[i][j] = __hadd(sum1[i][j], baseline_approx_kernel_mul_fp16(shared_A[tx + i][k], shared_B[ty + j + 32][k]));
                    sum2[i][j] = __hadd(sum2[i][j], baseline_approx_kernel_mul_fp16(shared_A[tx + i + 32][k], shared_B[ty + j][k]));
                    sum3[i][j] = __hadd(sum3[i][j], baseline_approx_kernel_mul_fp16(shared_A[tx + i + 32][k], shared_B[ty + j + 32][k]));
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
__global__ void group_GEMM(half* dA, half* dB, half* dC, int M, int N, int K) {
    // A [M. K], B [K, N], C [M, N]
    constexpr int SUB_K = 128;
    constexpr int g = SUB_K;
    __shared__ half SA[BM * BK];
    __shared__ half SB[BK * BN];
    // __shared__ half SS[1 * BN];
    
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
        __syncthreads();

        for (int index_k = 0; index_k < BK; index_k++) {
            for (int index_m = 0; index_m < TM; index_m++) {
                for (int index_n = 0; index_n < TN; index_n++) {
                    int reg_c_m = threadIdx.x * TM + index_m;
                    int reg_c_n = threadIdx.y * TN + index_n;
                    // tmp[index_m * TN + index_n] = __hadd(baseline_approx_kernel_mul_fp16(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                    tmp[index_m * TN + index_n] = __hadd(__hmul(SA[reg_c_m * BK + index_k], SB[index_k * BN + reg_c_n]), tmp[index_m * TN + index_n]);
                }
            }

            // accumulated_k++;
            // if (accumulated_k == SUB_K) {
            //     #pragma unroll
            //     for (int i = 0; i < TM*TN; i++) {
            //         int col_idx = threadIdx.y * TN + (i % TN);
            //         // tmp[i] = dequantize_approx_kernel_mul_fp16(tmp[i], SS[col_idx], 0);
            //         // tmp[i] = __hmul(tmp[i], SS[col_idx]);
            //         tmp_fp32[i] += __half2float(tmp[i]);
            //         // tmp_fp32[i] += __half2float(tmp[i]);
            //         tmp[i] = __float2half(0.0f);
            //     }
            //     accumulated_k = 0;
            // }
        }
        __syncthreads();
    }

    // if (accumulated_k > 0) {
    //     #pragma unroll
    //     for (int i = 0; i < TM*TN; i++) {
    //         tmp_fp32[i] += __half2float(tmp[i]);
    //     }
    // }

    for (int index_m = 0; index_m < TM; index_m++) {
        for (int index_n = 0; index_n < TN; index_n++) {
            int reg_c_m = threadIdx.x * TM + index_m;
            int reg_c_n = threadIdx.y * TN + index_n;
            if (indA + index_m < M && indB + index_n < N) {
                // dC[(indA + reg_c_m) * N + indB + reg_c_n] = __float2half(tmp_fp32[index_m * TN + index_n]);
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_m * TN + index_n];
                // dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_m * TN + index_n];
            }
        }
    }
}

__global__ void baseline_approx_kernel_fp16_optimized_batched(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K, 
    const int batch_size) {
        
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) return;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int row = block_row * TILE_SIZE + thread_row;
    int col = block_col * TILE_SIZE + thread_col;

    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE][TILE_SIZE];

    half sum = uint16_to_float16(0);
    // float sum = 0.0f;

    int A_offset = batch_idx * M * K;
    int B_offset = batch_idx * K * N;
    int C_offset = batch_idx * M * N;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load data into shared memory with boundary checks
        if ((row < M) && (t * TILE_SIZE + thread_col) < K) {
            shared_A[thread_row][thread_col] = A[A_offset + row * K + t * TILE_SIZE + thread_col];
        } else {
            shared_A[thread_row][thread_col] = uint16_to_float16(0);
        }

        if ((t * TILE_SIZE + thread_row) < K && (col < N)) {
            shared_B[thread_row][thread_col] = B[B_offset + (t * TILE_SIZE + thread_row) * N + col];
        } else {
            shared_B[thread_row][thread_col] = uint16_to_float16(0);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            half a = shared_A[thread_row][k];
            half b = shared_B[k][thread_col];

            half product_fp16 = baseline_approx_kernel_mul_fp16(a, b);
            // half product_fp16 = __hmul(a, b);
            // float product_fp32 = __half2float(product_fp16); 

            sum = __hadd(sum, product_fp16); // mse error 12.x
            // sum = baseline_approx_kernel_add_without_subnormal_rtz(sum, product_fp16); // mse error 18.x
            // sum = baseline_approx_kernel_add_without_subnormal_rtn(sum, product_fp16); // mse error 12.x
            // sum = baseline_approx_kernel_add_without_subnormal_efn(sum, product_fp16); // mse error 15.x
            // sum = baseline_approx_kernel_add_without_subnormal_tte(sum, product_fp16); // mse error 12.x
            // sum += product_fp32; // mse error 11.x
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[C_offset + row * N + col] = sum;
    }
}

void launch_baseline_approx_kernel_fp16(half* A, half* B, half* C, int M, int N, int K, int device_id) {
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
    group_GEMM<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void launch_baseline_approx_kernel_fp16_batched(const half* A, const half* B, half* C, int M, int N, int K, int batch_size, int device_id) {
    cudaSetDevice(device_id);
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y, batch_size);

    baseline_approx_kernel_fp16_optimized_batched<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K, batch_size);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

struct BaselineMul {
    __device__ half operator()(half a, half b) const {
        return baseline_approx_kernel_mul_fp16(a, b);
    }
};

struct BaselineDiv {
    __device__ half operator()(half a, half b) const {
        return baseline_approx_kernel_div_fp16(a, b);
    }
};

template <typename APOp>
__global__ void baseline_elementwise_single_kernel_fp16(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col) {
    APOp apop;

    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    half reg_B = B[0];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        half reg_A = shared_A[threadIdx.y][threadIdx.x];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = apop(reg_A, reg_B);
    }
}

template <typename APOp>
__global__ void baseline_elementwise_all_kernel_fp16(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col) {

    APOp apop;

    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE][TILE_SIZE];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    if (A_row_idx < B_row && A_col_idx < B_col) {
        shared_B[threadIdx.y][threadIdx.x] = B[A_row_idx * B_col + A_col_idx] ;
    } else {
        shared_B[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        half reg_A = shared_A[threadIdx.y][threadIdx.x];
        half reg_B = shared_B[threadIdx.y][threadIdx.x];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = apop(reg_A, reg_B);
    }
}

template <typename APOp>
__global__ void baseline_elementwise_rowwise_kernel_fp16(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col) {

    APOp apop;
    
    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    if (threadIdx.x == 0) {
        shared_B[threadIdx.y] = B[A_row_idx];
    }

    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        half reg_A = shared_A[threadIdx.y][threadIdx.x];
        half reg_B = shared_B[threadIdx.y];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = apop(reg_A, reg_B);
    }
}

void launch_baseline_elementwisemul_kernel_fp16(const half* A, const half* B, half* C, int A_row, int A_col, int B_row, int B_col, int device_id) {
    cudaSetDevice(device_id);
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid((A_col + threads_per_block.x - 1) / threads_per_block.x, (A_row + threads_per_block.y - 1) / threads_per_block.y);

    if (B_row == 1 && B_col == 1) {
        baseline_elementwise_single_kernel_fp16<BaselineMul><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else if (B_row == A_row && B_col == A_col) {
        baseline_elementwise_all_kernel_fp16<BaselineMul><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else if (B_row == A_row && B_col == 1) {
        baseline_elementwise_rowwise_kernel_fp16<BaselineMul><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else {
        throw std::runtime_error("Unsupported elementwise operation");
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void launch_baseline_elementwisediv_kernel_fp16(const half* A, const half* B, half* C, int A_row, int A_col, int B_row, int B_col, int device_id) {
    cudaSetDevice(device_id);
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid((A_col + threads_per_block.x - 1) / threads_per_block.x, (A_row + threads_per_block.y - 1) / threads_per_block.y);

    if (B_row == 1 && B_col == 1) {
        baseline_elementwise_single_kernel_fp16<BaselineDiv><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else if (B_row == A_row && B_col == A_col) {
        baseline_elementwise_all_kernel_fp16<BaselineDiv><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else if (B_row == A_row && B_col == 1) {
        baseline_elementwise_rowwise_kernel_fp16<BaselineDiv><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col);
    } else {
        throw std::runtime_error("Unsupported elementwise operation");
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
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

__device__ half quantize_approx_kernel_div_fp16(half a, half b, int16_t comp) {
    uint16_t a_bits = float16_to_uint16(a);
    uint16_t b_bits = float16_to_uint16(b);

    uint16_t is_zero_a = (a_bits & 0x7FFF) == 0;
    // uint16_t is_zero_b = (b_bits & 0x7FFF) == 0;

    uint16_t s1 = a_bits & 0x8000;
    uint16_t s2 = b_bits & 0x8000;
    uint16_t sign = s1 ^ s2;

    // uint16_t mantissa_div = a_bits - b_bits;;

    // FP16 Baseline: 0x3C00
    // uint16_t threshold = 0x3C00;
    // uint16_t threshold = 0x3C00 - 20;
    uint16_t threshold = 0x3C00 + comp;
    uint16_t mantissa_sum = (a_bits & 0x7FFF) + threshold; // add threshold at first to prevent underflow
    uint16_t adjusted_mantissa;
    // uint16_t diff = mantissa_div + threshold; 
    uint16_t diff = mantissa_sum - (b_bits & 0x7FFF); 
    uint16_t is_negative_mask = -(mantissa_sum < (b_bits & 0x7FFF));
    adjusted_mantissa = diff & (~is_negative_mask); 
    

    // Combine the sign and adjusted mantissa
    uint16_t result_bits = (adjusted_mantissa & 0x7FFF) | sign;

    result_bits = is_zero_a ? 0 : result_bits;

    return uint16_to_float16(result_bits);
}

struct DequantizeMul {
    __device__ half operator()(half a, half b, int16_t comp) const {
        return dequantize_approx_kernel_mul_fp16(a, b, comp);
    }
};

struct QuantizeDiv {
    __device__ half operator()(half a, half b, int16_t comp) const {
        return quantize_approx_kernel_div_fp16(a, b, comp);
    }
};

template <typename APOp>
__global__ void Qelementwise_single_kernel_fp16(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col, int16_t comp) {
    APOp apop;

    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    half reg_B = B[0];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        half reg_A = shared_A[threadIdx.y][threadIdx.x];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = apop(reg_A, reg_B, comp);
    }
}

template <typename APOp>
__global__ void Qelementwise_all_kernel_fp16(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col, int16_t comp) {

    APOp apop;

    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE][TILE_SIZE];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    if (A_row_idx < B_row && A_col_idx < B_col) {
        shared_B[threadIdx.y][threadIdx.x] = B[A_row_idx * B_col + A_col_idx] ;
    } else {
        shared_B[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        half reg_A = shared_A[threadIdx.y][threadIdx.x];
        half reg_B = shared_B[threadIdx.y][threadIdx.x];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = apop(reg_A, reg_B, comp);
    }
}

template <typename APOp>
__global__ void Qelementwise_rowwise_kernel_fp16(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int A_row, int A_col, int B_row, int B_col, int16_t comp) {

    APOp apop;
    
    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE];

    int A_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int A_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (A_row_idx < A_row && A_col_idx < A_col) {
        shared_A[threadIdx.y][threadIdx.x] = A[A_row_idx * A_col + A_col_idx] ;
    } else {
        shared_A[threadIdx.y][threadIdx.x] = uint16_to_float16(0);
    }

    if (threadIdx.x == 0) {
        shared_B[threadIdx.y] = B[A_row_idx];
    }

    __syncthreads();

    if (A_row_idx < A_row && A_col_idx < A_col) {
        half reg_A = shared_A[threadIdx.y][threadIdx.x];
        half reg_B = shared_B[threadIdx.y];
        // C[A_row_idx * A_col + A_col_idx] = __hmul(reg_A, reg_B);
        C[A_row_idx * A_col + A_col_idx] = apop(reg_A, reg_B, comp);
    }
}

// void launch_dequantize_elementwisemul_kernel_fp16(const half* A, const half* B, half* C, int A_row, int A_col, int B_row, int B_col, int16_t comp, int device_id) {
//     cudaSetDevice(device_id);
//     dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 blocks_per_grid((A_col + threads_per_block.x - 1) / threads_per_block.x, (A_row + threads_per_block.y - 1) / threads_per_block.y);

//     if (B_row == 1 && B_col == 1) {
//         Qelementwise_single_kernel_fp16<DequantizeMul><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col, comp);
//     } else if (B_row == A_row && B_col == A_col) {
//         Qelementwise_all_kernel_fp16<DequantizeMul><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col, comp);
//     } else if (B_row == A_row && B_col == 1) {
//         Qelementwise_rowwise_kernel_fp16<DequantizeMul><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col, comp);
//     } else {
//         throw std::runtime_error("Unsupported elementwise operation");
//     }

//     // Check for CUDA errors
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(err));
//     }
// }

void launch_dequantize_elementwisemul_kernel_fp16(const half* A, const half* B, half* C, int A_row, int A_col, int B_row, int B_col, int16_t comp, int device_id) {
    cudaSetDevice(device_id);
    
    // CUDA grid dimensions are limited (typically max 65535 in any dimension)
    // For very large tensors, we need to limit the grid size
    const int MAX_GRID_DIM = 65535;
    
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    
    // Calculate grid dimensions with limits
    int grid_x = (A_col + threads_per_block.x - 1) / threads_per_block.x;
    int grid_y = (A_row + threads_per_block.y - 1) / threads_per_block.y;
    
    // Limit grid dimensions to maximum allowed
    grid_x = grid_x > MAX_GRID_DIM ? MAX_GRID_DIM : grid_x;
    grid_y = grid_y > MAX_GRID_DIM ? MAX_GRID_DIM : grid_y;
    
    dim3 blocks_per_grid(grid_x, grid_y);
    
    // Process data in chunks if necessary
    for (int row_offset = 0; row_offset < A_row; row_offset += grid_y * threads_per_block.y) {
        int current_rows = min(grid_y * threads_per_block.y, A_row - row_offset);
        
        if (B_row == 1 && B_col == 1) {
            Qelementwise_single_kernel_fp16<DequantizeMul><<<blocks_per_grid, threads_per_block>>>(
                A + row_offset * A_col, 
                B, 
                C + row_offset * A_col, 
                current_rows, A_col, B_row, B_col, comp
            );
        } else if (B_row == A_row && B_col == A_col) {
            Qelementwise_all_kernel_fp16<DequantizeMul><<<blocks_per_grid, threads_per_block>>>(
                A + row_offset * A_col, 
                B + row_offset * B_col, 
                C + row_offset * A_col, 
                current_rows, A_col, current_rows, B_col, comp
            );
        } else if (B_row == A_row && B_col == 1) {
            Qelementwise_rowwise_kernel_fp16<DequantizeMul><<<blocks_per_grid, threads_per_block>>>(
                A + row_offset * A_col, 
                B + row_offset, 
                C + row_offset * A_col, 
                current_rows, A_col, current_rows, B_col, comp
            );
        } else {
            throw std::runtime_error("Unsupported elementwise operation");
        }
        
        // Check for CUDA errors after each kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}

// void launch_quantize_elementwisediv_kernel_fp16(const half* A, const half* B, half* C, int A_row, int A_col, int B_row, int B_col, int16_t comp, int device_id) {
//     cudaSetDevice(device_id);
//     dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 blocks_per_grid((A_col + threads_per_block.x - 1) / threads_per_block.x, (A_row + threads_per_block.y - 1) / threads_per_block.y);

//     if (B_row == 1 && B_col == 1) {
//         Qelementwise_single_kernel_fp16<QuantizeDiv><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col, comp);
//     } else if (B_row == A_row && B_col == A_col) {
//         Qelementwise_all_kernel_fp16<QuantizeDiv><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col, comp);
//     } else if (B_row == A_row && B_col == 1) {
//         Qelementwise_rowwise_kernel_fp16<QuantizeDiv><<<blocks_per_grid, threads_per_block>>>(A, B, C, A_row, A_col, B_row, B_col, comp);
//     } else {
//         throw std::runtime_error("Unsupported elementwise operation");
//     }

//     // Check for CUDA errors
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(err));
//     }
// }

void launch_quantize_elementwisediv_kernel_fp16(const half* A, const half* B, half* C, int A_row, int A_col, int B_row, int B_col, int16_t comp, int device_id) {
    cudaSetDevice(device_id);
    
    // CUDA grid dimensions are limited (typically max 65535 in any dimension)
    // For very large tensors, we need to limit the grid size
    const int MAX_GRID_DIM = 65535;
    
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    
    // Calculate grid dimensions with limits
    int grid_x = (A_col + threads_per_block.x - 1) / threads_per_block.x;
    int grid_y = (A_row + threads_per_block.y - 1) / threads_per_block.y;
    
    // Limit grid dimensions to maximum allowed
    grid_x = grid_x > MAX_GRID_DIM ? MAX_GRID_DIM : grid_x;
    grid_y = grid_y > MAX_GRID_DIM ? MAX_GRID_DIM : grid_y;
    
    dim3 blocks_per_grid(grid_x, grid_y);
    
    // Process data in chunks if necessary
    for (int row_offset = 0; row_offset < A_row; row_offset += grid_y * threads_per_block.y) {
        int current_rows = min(grid_y * threads_per_block.y, A_row - row_offset);
        
        if (B_row == 1 && B_col == 1) {
            Qelementwise_single_kernel_fp16<QuantizeDiv><<<blocks_per_grid, threads_per_block>>>(
                A + row_offset * A_col, 
                B, 
                C + row_offset * A_col, 
                current_rows, A_col, B_row, B_col, comp
            );
        } else if (B_row == A_row && B_col == A_col) {
            Qelementwise_all_kernel_fp16<QuantizeDiv><<<blocks_per_grid, threads_per_block>>>(
                A + row_offset * A_col, 
                B + row_offset * B_col, 
                C + row_offset * A_col, 
                current_rows, A_col, current_rows, B_col, comp
            );
        } else if (B_row == A_row && B_col == 1) {
            Qelementwise_rowwise_kernel_fp16<QuantizeDiv><<<blocks_per_grid, threads_per_block>>>(
                A + row_offset * A_col, 
                B + row_offset, 
                C + row_offset * A_col, 
                current_rows, A_col, current_rows, B_col, comp
            );
        } else {
            throw std::runtime_error("Unsupported elementwise operation");
        }
        
        // Check for CUDA errors after each kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}