#include <torch/extension.h>
#include "AxCore.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


void torch_launch_AxCore_group_gemm_kernel(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    torch::Tensor &S,
    int M, int N, int K) {
    
    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_AxCore_group_gemm_kernel_fp16(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        (half*) S.data_ptr(),
        M, N, K,
        device_id
    );
}

void torch_launch_AxCore_group_gemm_typed_kernel(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    torch::Tensor &S,
    torch::Tensor &T,
    int g,
    int M, int N, int K,
    bool use_approx_dequant = true) {
    
    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    TORCH_CHECK(T.dtype() == torch::kUInt8 , "Types tensor must be torch.uint8");

    launch_AxCore_group_gemm_typed_kernel_fp16(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        (half*) S.data_ptr(),
        (unsigned char*) T.data_ptr(),
        g,
        M, N, K,
        device_id,
        use_approx_dequant
    );
}

void torch_launch_mpFPMA_gemm_kernel_fp16(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    torch::Tensor &S,
    int M, int N, int K,
    int opt) {
    
    int device_id = A.device().index();

    TORCH_CHECK(A.device() == B.device(), "Tensors A and B must be on the same device");
    if (A.dtype() != torch::kHalf || B.dtype() != torch::kHalf || C.dtype() != torch::kHalf) {
        throw std::invalid_argument("Tensors must be of type half.");
    }

    launch_mpFPMA_gemm_kernel_fp16(
        (half*) A.data_ptr(),
        (half*) B.data_ptr(),
        (half*) C.data_ptr(),
        (half*) S.data_ptr(),
        M, N, K,
        opt,
        device_id
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_AxCore_group_gemm_kernel", &torch_launch_AxCore_group_gemm_kernel, "AxCore group GEMM kernel (fp16)");
    m.def("torch_launch_AxCore_group_gemm_typed_kernel", &torch_launch_AxCore_group_gemm_typed_kernel, "AxCore group GEMM typed kernel (fp16)");
    m.def("torch_launch_mpFPMA_gemm_kernel_fp16", &torch_launch_mpFPMA_gemm_kernel_fp16, "mpFPMA GEMM kernel (fp16)");
}
