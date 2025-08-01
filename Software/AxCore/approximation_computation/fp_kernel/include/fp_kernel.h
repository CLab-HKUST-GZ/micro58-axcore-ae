#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>

void launch_wmma_gemm_kernel_fp16(const half* A, const half* B, half* C, int M, int N, int K);

// void cutlass_gemm(const half* A, const half* B, half* C, int M, int N, int K);

void launch_fp_gemm_kernel_fp16(half* A, half* B, half* C, int M, int N, int K);

void launch_group_gemm_kernel(half* A, half* B, half* C, int M, int N, int K);
