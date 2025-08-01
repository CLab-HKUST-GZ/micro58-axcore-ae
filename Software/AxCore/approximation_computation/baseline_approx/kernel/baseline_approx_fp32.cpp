#include <torch/extension.h>
#include "baseline_approx_fp32.h"

void torch_launch_baseline_approx_kernel_fp32(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int M, int N, int K) {
    
    // Validate tensor types
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("All tensors must be CUDA tensors.");
    }
    if (A.dtype() != torch::kFloat32 || B.dtype() != torch::kFloat32 || C.dtype() != torch::kFloat32) {
        throw std::invalid_argument("Tensors must be of type float32.");
    }

    launch_baseline_approx_kernel_fp32(
        (const float*) A.data_ptr(),
        (const float*) B.data_ptr(),
        (float*) C.data_ptr(),
        M, N, K
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_baseline_approx_kernel_fp32",
          &torch_launch_baseline_approx_kernel_fp32,
          "baseline_approx kernel warpper");
}