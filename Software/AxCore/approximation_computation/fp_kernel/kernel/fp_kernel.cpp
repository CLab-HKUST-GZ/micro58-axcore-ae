#include <torch/extension.h>
#include "fp_kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void torch_launch_wmma_gemm_kernel_fp16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {
    
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_wmma_gemm_kernel_fp16(
        (const half*) A.data_ptr(),
        (const half*) B.data_ptr(),
        (half*) C.data_ptr(),
        M, N, K
    );
}

// void torch_launch_cutlass_gemm_kernel_fp16(
//     const torch::Tensor &A,
//     const torch::Tensor &B,
//     torch::Tensor &C,
//     int M, int N, int K) {
    
//     // Validate tensor types
//     if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
//         throw std::invalid_argument("All tensors must be CUDA tensors.");
//     }
//     if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
//         throw std::invalid_argument("Tensors must be of type half.");
//     }

//     cutlass_gemm(
//         (const half*) A.data_ptr(),
//         (const half*) B.data_ptr(),
//         (half*) C.data_ptr(),
//         M, N, K
//     );
// }

void torch_launch_fp_gemm_kernel_fp16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {
    
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_fp_gemm_kernel_fp16(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        M, N, K
    );
}

void torch_launch_group_gemm_kernel(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {
    
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_group_gemm_kernel(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        M, N, K
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_wmma_gemm_kernel_fp16", &torch_launch_wmma_gemm_kernel_fp16, "Launch WMMA GEMM kernel (fp16)");
    m.def("torch_launch_fp_gemm_kernel_fp16", &torch_launch_fp_gemm_kernel_fp16, "Launch FP GEMM kernel (fp16)");
    m.def("torch_launch_group_gemm_kernel", &torch_launch_group_gemm_kernel, "Launch group GEMM kernel (fp16)");
}