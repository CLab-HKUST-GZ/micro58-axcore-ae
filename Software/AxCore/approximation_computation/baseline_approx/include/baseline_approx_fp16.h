#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>

void launch_baseline_approx_kernel_fp16(
    half* A,
    half* B,
    half* C,
    int M, int N, int K,
    int device_id);

void launch_baseline_approx_kernel_fp16_batched(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    int batch_size,
    int device_id);

void launch_baseline_elementwisemul_kernel_fp16(
    const half* A, 
    const half* B, 
    half* C, 
    int A_row, int A_col, int B_row, int B_col,
    int device_id);

void launch_baseline_elementwisediv_kernel_fp16(
    const half* A, 
    const half* B, 
    half* C, 
    int A_row, int A_col, int B_row, int B_col,
    int device_id);

void launch_quantize_elementwisediv_kernel_fp16(
    const half* A, 
    const half* B, 
    half* C, 
    int A_row, int A_col, int B_row, int B_col,
    int16_t comp,
    int device_id);

void launch_dequantize_elementwisemul_kernel_fp16(
    const half* A, 
    const half* B, 
    half* C, 
    int A_row, int A_col, int B_row, int B_col,
    int16_t comp,
    int device_id);