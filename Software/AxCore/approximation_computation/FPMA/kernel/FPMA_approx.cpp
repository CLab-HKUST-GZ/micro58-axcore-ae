#include <torch/extension.h>
#include "FPMA_approx_bf16.h"
#include "FPMA_approx_fp16.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void torch_launch_FPMA_approx_kernel_bf16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {
    
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kBFloat16 || B.dtype() != torch::kBFloat16 || C.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("Tensors must be of type bfloat16.");
    }

    launch_FPMA_approx_kernel_bf16(
        (__nv_bfloat16*) A.data_ptr(),
        (__nv_bfloat16*) B.data_ptr(),
        (__nv_bfloat16*) C.data_ptr(),
        M, N, K
    );
}

void torch_launch_FPMA_approx_kernel_bf16_batched(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K,
    int batch_size) {
    
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kBFloat16 || B.dtype() != torch::kBFloat16 || C.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("Tensors must be of type bfloat16.");
    }

    launch_FPMA_approx_kernel_bf16_batched(
        (const __nv_bfloat16*) A.contiguous().data_ptr(),
        (const __nv_bfloat16*) B.contiguous().data_ptr(),
        (__nv_bfloat16*) C.contiguous().data_ptr(),
        M, N, K,
        batch_size
    ); 
}

torch::Tensor torch_launch_FPMA_elementwisemul_kernel_bf16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    int A_row, int A_col, int B_row, int B_col) {

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kBFloat16 || B.dtype() != torch::kBFloat16) {
            throw std::invalid_argument("Tensors must be of type bfloat16.");
        }
    torch::Tensor C = torch::empty({A_row, A_col}, A.options());

    launch_FPMA_elementwisemul_kernel_bf16(
        (const __nv_bfloat16*) A.contiguous().data_ptr(),
        (const __nv_bfloat16*) B.contiguous().data_ptr(),
        (__nv_bfloat16*) C.contiguous().data_ptr(),
        A_row, A_col, B_row, B_col
    );
    return C;
} 

void torch_launch_FPMA_approx_kernel_fp16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {
    
    int device_id = A.device().index();
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_FPMA_approx_kernel_fp16(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        M, N, K, 
        device_id
    );
}

void torch_launch_FPMA_approx_kernel_fp16_batched(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K,
    int batch_size) {

    // Validate tensor types  
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_FPMA_approx_kernel_fp16_batched(
        (const half*) A.contiguous().data_ptr(),
        (const half*) B.contiguous().data_ptr(),
        (half*) C.contiguous().data_ptr(),
        M, N, K, 
        batch_size
    );
}

torch::Tensor torch_launch_FPMA_elementwisemul_kernel_fp16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    int A_row, int A_col, int B_row, int B_col) {

    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf) {
            throw std::invalid_argument("Tensors must be of type half.");
        }
    torch::Tensor C = torch::empty({A_row, A_col}, A.options());

    launch_FPMA_elementwisemul_kernel_fp16(
        (const half*) A.contiguous().data_ptr(),
        (const half*) B.contiguous().data_ptr(),
        (half*) C.contiguous().data_ptr(),
        A_row, A_col, B_row, B_col,
        device_id
    );
    return C;
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_FPMA_approx_kernel_bf16",
          &torch_launch_FPMA_approx_kernel_bf16,
          "FPMA_approx kernel warpper");
    m.def("torch_launch_FPMA_approx_kernel_bf16_batched",
          &torch_launch_FPMA_approx_kernel_bf16_batched,
          "FPMA_approx kernel warpper");
    m.def("torch_launch_FPMA_elementwisemul_kernel_bf16",
          &torch_launch_FPMA_elementwisemul_kernel_bf16,
          "FPMA_approx kernel warpper");
    m.def("torch_launch_FPMA_approx_kernel_fp16",
          &torch_launch_FPMA_approx_kernel_fp16,
          "FPMA_approx kernel warpper");
    m.def("torch_launch_FPMA_approx_kernel_fp16_batched",
          &torch_launch_FPMA_approx_kernel_fp16_batched,
          "FPMA_approx kernel warpper");
    m.def("torch_launch_FPMA_elementwisemul_kernel_fp16",
          &torch_launch_FPMA_elementwisemul_kernel_fp16,
          "FPMA_approx kernel warpper");
}