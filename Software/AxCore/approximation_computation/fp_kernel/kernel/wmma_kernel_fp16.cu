#include <stdexcept>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define BLOCK_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_kernel(const half *A, const half *B, half *C,
                                   int M, int N, int K) {

    int warpM = (blockIdx.y * (blockDim.y) + threadIdx.y);
    int warpN = (blockIdx.x * (blockDim.x) + threadIdx.x);
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) 
    return;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;

    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += WMMA_K) {
         int aRow = warpM * WMMA_M;
         int aCol = k;
         int bRow = k;
         int bCol = warpN * WMMA_N;

         wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
         wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}

void launch_wmma_gemm_kernel_fp16(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 threads_per_block(8, 4);
    dim3 blocks_per_grid(
        (N + WMMA_N - 1) / WMMA_N,  // ceil(N/16)
        (M + WMMA_M - 1) / WMMA_M   // ceil(M/16)
    );
    // baseline_approx_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    wmma_gemm_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}