#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void launch_FPMA_approx_kernel_fp16(
    half* A,
    half* B,
    half* C,
    int M, int N, int K,
    int device_id);

void launch_FPMA_approx_kernel_fp16_batched(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    int batch_size);

void launch_FPMA_elementwisemul_kernel_fp16(const half* A, 
    const half* B, 
    half* C, 
    int A_row, int A_col, int B_row, int B_col,
    int device_id);