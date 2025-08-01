// #include "cutlass/gemm/device/gemm.h"

// #define BLOCK_SIZE 16

// using Gemm = cutlass::gemm::device::Gemm<
//     half, cutlass::layout::RowMajor,
//     half, cutlass::layout::RowMajor,
//     half, cutlass::layout::RowMajor,
//     float
// >;

// void cutlass_gemm(const half* A, const half* B, half* C,
//                       int M, int N, int K) {
//     cutlass::gemm::GemmCoord problem_size(M, N, K);
//     half alpha = __float2half(1.0f);
//     half beta = __float2half(0.0f);
//     Gemm gemm_op;

//     cutlass::Status status = gemm_op(
//         problem_size,
//         {alpha, beta},
//         A, K,  
//         B, N,  
//         C, N,  
//         C, N   
//     );
//     if (status != cutlass::Status::kSuccess) {
//         throw std::runtime_error("CUTLASS GEMM failed");
//     }
// }
