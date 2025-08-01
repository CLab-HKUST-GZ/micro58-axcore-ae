#include <stdexcept>

__device__ float baseline_approx_kernel_mul_fp32(float a, float b) {
    // float result;
    // unsigned int a_bits, b_bits, s1, s2, r0, pred, temp;

    // asm volatile(
    //     "mov.b32 %1, %6;"
    //     "mov.b32 %2, %7;"
    //     "and.b32 %3, %1, 0x80000000;"
    //     "and.b32 %4, %2, 0x80000000;"
    //     "and.b32 %1, %1, 0x7FFFFFFF;"
    //     "and.b32 %2, %2, 0x7FFFFFFF;"
    //     "xor.b32 %3, %3, %4;"
    //     "add.u32 %5, %1, %2;"
    //     "sub.u32 %5, %5, 0x3F780000;"
    //     "and.b32 %5, %5, 0x7FFFFFFF;"
    //     "add.u32 %5, %5, %3;"
    //     "mov.b32 %0, %5;"
    //     : "=f"(result), "=r"(a_bits), "=r"(b_bits), "=r"(s1), "=r"(s2), "=r"(r0)
    //     : "f"(a), "f"(b)
    // );
    // return result;
    if (a == 0 || b == 0) {
        return 0;
    }

    unsigned int a_bits = __float_as_uint(a);
    unsigned int b_bits = __float_as_uint(b);

    
    unsigned int s1 = a_bits & 0x80000000;
    unsigned int s2 = b_bits & 0x80000000;
    unsigned int sign = s1 ^ s2;
    
    unsigned int mantissa_sum = (a_bits & 0x7FFFFFFF) + (b_bits & 0x7FFFFFFF);
    
    // Ensure mantissa_sum does not underflow below 0x3F780000
    unsigned int threshold = 0x3F780000;
    unsigned int adjusted_mantissa = mantissa_sum - threshold;
    // unsigned int adjusted_mantissa = mantissa_sum >= threshold ? mantissa_sum - threshold : 0;
    
    // Combine the sign and adjusted mantissa
    unsigned int result_bits = (adjusted_mantissa & 0x7FFFFFFF) + sign;
    
    return __uint_as_float(result_bits);
}

__global__ void baseline_approx_kernel_fp32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[row * K + k];
            float b = B[k * N + col];
            // sum += a * b;
            sum += baseline_approx_kernel_mul_fp32(a, b);
        }
        C[row * N + col] = sum;
    }
}

void launch_baseline_approx_kernel_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);
    baseline_approx_kernel_fp32<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

