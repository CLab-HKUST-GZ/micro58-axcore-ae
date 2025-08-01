#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>

void launch_AxCore_group_gemm_kernel_fp16(half* A, half* B, half* C, half* S, int M, int N, int K, int device_id);

void launch_AxCore_group_gemm_typed_kernel_fp16(
    half* A, 
    half* B, 
    half* C, 
    half* S, 
    unsigned char* T, 
    int g,
    int M, int N, int K, int device_id,
    bool use_approx_dequant);
