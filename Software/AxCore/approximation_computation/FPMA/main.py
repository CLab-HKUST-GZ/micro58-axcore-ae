import torch
from torch.utils.cpp_extension import load

cuda_module = load(name="FPMA_approx",
                   extra_include_paths=["include"],
                   sources=["kernel/FPMA_approx.cpp", "kernel/FPMA_approx_kernel_bf16.cu", "kernel/FPMA_approx_kernel_fp16.cu"],
                   extra_cflags=["-O0"],  # 降低C++优化等级
                   extra_cuda_cflags=["-O0"],  # 降低CUDA优化等级
                   verbose=True)

class FPMAApproxFunctionBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C, M, N, K):
        cuda_module.torch_launch_FPMA_approx_kernel_bf16(A, B, C, M, N, K)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None
    
class FPMAApproxBF16(torch.nn.Module):
    def forward(self, A, B, C, M, N, K):
        return FPMAApproxFunctionBF16.apply(A, B, C, M, N, K)
    
class FPMAApproxFunctionFP16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C, M, N, K):
        cuda_module.torch_launch_FPMA_approx_kernel_fp16(A, B, C, M, N, K)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None

class FPMAApproxFP16(torch.nn.Module):
    def forward(self, A, B, C, M, N, K):
        return FPMAApproxFunctionFP16.apply(A, B, C, M, N, K)
    

def main():
    M, N, K = 10, 1000, 5120
    # a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    # b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    # c = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')
    c = torch.randn(M, N, dtype=torch.float16, device='cuda')
    approx = FPMAApproxFP16()
    a[0, 0] = 0
    # b[:, 0] /= 10000000000
    # checkpoint = torch.load("x_weight.pt")
    # a = checkpoint["x"]
    # b = checkpoint["weight"]
    # b = b.t()
    # S = 5
    # a = a[:S, :S].to(torch.float32).to('cuda')
    # b = b[:S, :S].to(torch.float32).to('cuda')
    a = a.contiguous()
    b = b.contiguous()
    # a = torch.Tensor([[1.7500, 1.1875, 1.4375, 2.7500, 0.4219],
    #                   [2.5000, 0.8125, 1.5000, 1.1250, 1.0625],
    #                   [1.2500, 1.7500, 2.5000, 0.8438, 0.2188],
    #                   [0.1250, 1.0625, 1.6875, 0.3906, 0.1250],
    #                   [1.3750, 1.3125, 0.9375, 1.6875, 1.1250]]).to(torch.float32).to('cuda')
    # b = torch.Tensor([[-0.0156, -0.0781, -0.0312, -0.0625, -0.0625],
    #                   [-0.0781, -0.0938, -0.0625, -0.0000,  0.0625],
    #                   [-0.0469,  0.0156, -0.0312, -0.0156, -0.0469],
    #                   [-0.0156,  0.1406,  0.2500,  0.1562,  0.1875],
    #                   [ 0.0156, -0.0312,  0.0156,  0.0156, -0.0312]]).to(torch.float32).to('cuda')
    
    # c = torch.randn(a, b, dtype=torch.float32, device='cuda')
    print(a.shape, b.shape)
    print(a, a.max(), a.min())
    print(b, b.max(), b.min())
    
    # c_approx = approx(a, b, c, M, N, K)
    c_approx = approx(a, b, c, M, N, K)

    c_torch = torch.matmul(a, b)
    
    c_linear = torch.nn.functional.linear(a, b.t())
    max_diff = torch.max(torch.abs(c_approx - c_torch))
    # Mean square errors 
    mse = torch.mean((c_approx - c_torch) ** 2)
    print("######torch.randn test result######")
    print(f"Maximum difference between FPMA and normal matmul: {max_diff}")
    print(f"Mean square error between FPMA and normal matmul: {mse}")
    print(f"FPMA: {c_approx}, shape: {c_approx.shape}")
    print(f"torch.matmul: {c_torch}, shape: {c_torch.shape}")
    # print(f"c_linear: {c_linear}")
if __name__ == "__main__":
    main()