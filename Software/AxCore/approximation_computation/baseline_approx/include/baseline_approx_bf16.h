#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void launch_baseline_approx_kernel_bf16(
    __nv_bfloat16* A,
    __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K);

void launch_baseline_approx_kernel_bf16_batched(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    int batch_size);

void launch_baseline_elementwisemul_kernel_bf16(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int A_row, int A_col, int B_row, int B_col);

void launch_baseline_elementwisediv_kernel_bf16(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int A_row, int A_col, int B_row, int B_col);

void launch_quantize_elementwisediv_kernel_bf16(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int A_row, int A_col, int B_row, int B_col,
    int16_t comp,
    int device_id);

void launch_dequantize_elementwisemul_kernel_bf16(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int A_row, int A_col, int B_row, int B_col,
    int16_t comp,
    int device_id);