#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void launch_LMul_approx_kernel_fp16(
    half* A,
    half* B,
    half* C,
    int M, int N, int K);

void launch_LMul_approx_kernel_fp16_batched(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    int batch_size);

void launch_LMul_elementwisemul_kernel_fp16(const half* A, 
    const half* B, 
    half* C, 
    int A_row, int A_col, int B_row, int B_col);