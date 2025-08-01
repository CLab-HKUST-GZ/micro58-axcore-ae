import torch
from torch.utils.cpp_extension import load

import os

# Get the project root directory dynamically (2 levels up from current file)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.normpath(project_root)

cuda_module = load(name="FPMA_approx",
                   extra_include_paths=[os.path.join(project_root, "approximation_computation", "FPMA", "include")],
                   sources=[os.path.join(project_root, "approximation_computation", "FPMA", "kernel", "FPMA_approx.cpp"), 
                            os.path.join(project_root, "approximation_computation", "FPMA", "kernel", "FPMA_approx_kernel_bf16.cu"),
                            os.path.join(project_root, "approximation_computation", "FPMA", "kernel", "FPMA_approx_kernel_fp16.cu")],
                   extra_cflags=["-O3"],  # 降低C++优化等级
                   extra_cuda_cflags=["-O3"],  # 降低CUDA优化等级
                   verbose=True)

class FPMAApproxMatmulBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            A (Tensor): A tensor of shape (*, M, K) with dtype torch.bfloat16.
            B (Tensor): B tensor of shape (*, K, N) with dtype torch.bfloat16.
        
        Returns:
            Tensor: Output tensor of shape (*, N) with dtype torch.bfloat16.
        """
        A = A.to(torch.bfloat16) # shape: [*, M, K]
        B = B.to(torch.bfloat16) # shape: [*, K, N]
        
        # # Ensure input tensors are contiguous
        A = A.contiguous()
        B = B.contiguous() # y = xW^T # shape: (in_features, out_features)
        
        original_shape = A.shape[:-2]  # e.g., [1, 32]
        M, K = A.shape[-2], A.shape[-1]
        K_w, N = B.shape[-2], B.shape[-1]
        
        if K != K_w:
            raise ValueError(
                f"Incompatible dimensions: input has {K} features, but weight has {K_w} features."
            )
        
        batch_size = A.shape[:-2].numel()
        
        A_flat = A.view(batch_size, M, K)
        B_flat = B.view(batch_size, K, N)
        
        # Allocate output tensor
        flattened_output = torch.empty(
            (batch_size, M, N), dtype=torch.bfloat16, device=A.device
        )
        
        # Launch the CUDA kernel
        cuda_module.torch_launch_FPMA_approx_kernel_bf16_batched(
            A_flat, # [B, M, K]
            B_flat, # [B, K, N]
            flattened_output, # [B, M, N]
            M, N, K,
            batch_size
        )
            
        output = flattened_output.view(*original_shape, M, N)
        del A_flat, B_flat, flattened_output
        torch.cuda.empty_cache()
        return output

class FPMAApproxFunctionBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.bfloat16.
            weight (Tensor): Weight tensor of shape (out_features, in_features) with dtype torch.bfloat16.
            bias (Tensor, optional): Bias tensor of shape (out_features,) with dtype torch.bfloat16.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features) with dtype torch.bfloat16.
        """
        
        original_shape = input.shape[:-1]
        in_features = input.shape[-1]
        out_features, in_features_w = weight.shape
        
        if in_features != in_features_w:
            raise ValueError(f"Incompatible dimensions: input has {in_features} features, but weight has {in_features_w} features.")
        
        flattened_input = input.view(-1, in_features)
        
        # Allocate output tensor
        flattened_output = torch.empty((flattened_input.shape[0], out_features), dtype=torch.bfloat16, device=input.device)
        
        # Launch the CUDA kernel
        cuda_module.torch_launch_FPMA_approx_kernel_bf16(
            flattened_input, weight, flattened_output, flattened_input.shape[0], out_features, in_features
        )
        
        # If bias is provided, add it to the output
        if bias is not None:
            flattened_output += bias  # Broadcast bias to (M, out_features)
            
        output_shape = original_shape + (out_features,)
        output = flattened_output.view(*output_shape)
        del flattened_input, flattened_output
        torch.cuda.empty_cache()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass is not implemented as this is a custom approximation.
        To enable training, you would need to implement gradient computations.
        """
        # To enable backpropagation, implement gradient computations here
        # For simplicity, we'll assume gradients are not needed
        return None, None, None

    
class FPMAApproxLinearBF16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dev='cuda'):
        """
        Initializes the BaselineApproxBF16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(FPMAApproxLinearBF16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter
        self.register_buffer('weight', torch.zeros(self.out_features,
                                                   self.in_features, dtype=torch.bfloat16, requires_grad=False, device=dev))
        
        # Initialize bias parameter if required
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.bfloat16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        Performs the forward pass.
        
        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features) with dtype torch.bfloat16.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features) with dtype torch.bfloat16.
        """
        return FPMAApproxFunctionBF16.apply(input, self.weight, self.bias)
    
class FPMAApproxMatmulFP16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            A (Tensor): A tensor of shape (*, M, K) with dtype torch.bfloat16.
            B (Tensor): B tensor of shape (*, K, N) with dtype torch.bfloat16.
        
        Returns:
            Tensor: Output tensor of shape (*, N) with dtype torch.bfloat16.
        """
        A = A.to(torch.float16) # shape: [*, M, K]
        B = B.to(torch.float16) # shape: [*, K, N]
        
        # # Ensure input tensors are contiguous
        A = A.contiguous()
        B = B.contiguous() # y = xW^T # shape: (in_features, out_features)
        
        original_shape = A.shape[:-2]  # e.g., [1, 32]
        M, K = A.shape[-2], A.shape[-1]
        K_w, N = B.shape[-2], B.shape[-1]
        
        if K != K_w:
            raise ValueError(
                f"Incompatible dimensions: input has {K} features, but weight has {K_w} features."
            )
        
        batch_size = A.shape[:-2].numel()
        
        A_flat = A.view(batch_size, M, K)
        B_flat = B.view(batch_size, K, N)
        
        # Allocate output tensor
        flattened_output = torch.empty(
            (batch_size, M, N), dtype=torch.float16, device=A.device
        )
        
        # Launch the CUDA kernel
        cuda_module.torch_launch_FPMA_approx_kernel_fp16_batched(
            A_flat, # [B, M, K]
            B_flat, # [B, K, N]
            flattened_output, # [B, M, N]
            M, N, K,
            batch_size
        )
            
        output = flattened_output.view(*original_shape, M, N)
        del A_flat, B_flat, flattened_output
        torch.cuda.empty_cache()
        return output

class FPMAApproxFunctionFP16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.float16.
            weight (Tensor): Weight tensor of shape (out_features, in_features) with dtype torch.float16.
            bias (Tensor, optional): Bias tensor of shape (out_features,) with dtype torch.float16.
        
        Returns:
            Tensor: Output tensor of shape (*, out_features) with dtype torch.float16.
        """
        # input = input.to(torch.float16) # shape: [M, K]
        # weight = weight.to(torch.float16) # shape: [N, K]
        # if bias is not None:
        #     bias = bias.to(torch.float16)
        
        original_shape = input.shape[:-1]
        in_features = input.shape[-1]
        out_features, in_features_w = weight.shape
        
        if in_features != in_features_w:
            raise ValueError(f"Incompatible dimensions: input has {in_features} features, but weight has {in_features_w} features.")
        
        flattened_input = input.view(-1, in_features)
        
        # Allocate output tensor
        flattened_output = torch.empty((flattened_input.shape[0], out_features), dtype=torch.float16, device=input.device)
        
        # Launch the CUDA kernel
        cuda_module.torch_launch_FPMA_approx_kernel_fp16(
            flattened_input, weight, flattened_output, flattened_input.shape[0], out_features, in_features
        )
        # flattened_output = torch.matmul(flattened_input, weight)
        
        # If bias is provided, add it to the output
        if bias is not None:
            flattened_output += bias  # Broadcast bias to (M, out_features)
            
        output_shape = original_shape + (out_features,)
        output = flattened_output.view(*output_shape)
        del flattened_input, flattened_output
        torch.cuda.empty_cache()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass is not implemented as this is a custom approximation.
        To enable training, you would need to implement gradient computations.
        """
        # To enable backpropagation, implement gradient computations here
        # For simplicity, we'll assume gradients are not needed
        return None, None, None
    
class FPMAApproxLinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dev='cuda'):
        """
        Initializes the BaselineApproxBF16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(FPMAApproxLinearFP16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter
        self.register_buffer('weight', torch.zeros(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        
        # Initialize bias parameter if required
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        Performs the forward pass.
        
        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features) with dtype torch.float16.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features) with dtype torch.float16.
        """
        return FPMAApproxFunctionFP16.apply(input, self.weight, self.bias)

class FPMAApproxElementwiseMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        if B.dim() != 2:
            raise ValueError("B must be a 2D tensor")
        A_row, A_col = A.shape
        B_row, B_col = B.shape
        
        if A.dtype == torch.float16:
            C = cuda_module.torch_launch_FPMA_elementwisemul_kernel_fp16(A, B, A_row, A_col, B_row, B_col)
        else:
            C = cuda_module.torch_launch_FPMA_elementwisemul_kernel_bf16(A, B, A_row, A_col, B_row, B_col)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass is not implemented as this is a custom approximation.
        To enable training, you would need to implement gradient computations.
        """
        # To enable backpropagation, implement gradient computations here
        # For simplicity, we'll assume gradients are not needed
        return None, None, None

# Example usage
if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dtype = torch.float16
    n_bits = 4
    zero_point = True
    batch = 200
    max_diff_array = []
    mse_array = []
    for i in range(10000):
        W = torch.randn(batch, 64, dtype=dtype, device=device)
        S = torch.randn(batch, 1, dtype=dtype, device=device)
        Z = torch.ones(batch, 1, dtype=dtype, device=device)
        W *= 0.05
        
        if zero_point:
            max_val = W.amax(dim=1, keepdim=True)
            min_val = W.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            S = (max_val - min_val + 1e-3).clamp(min=1e-5) / max_int
            # S = BaselineApproxElementwiseDiv.apply((max_val - min_val).clamp(min=1e-5), torch.tensor(max_int, dtype=dtype, device=device))
            Z = (-torch.round(min_val / S)).clamp_(min_int, max_int)
            # Z = (-torch.round(BaselineApproxElementwiseDiv.apply(min_val, S))).clamp_(min_int, max_int)
        else:
            max_val = W.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            S = max_val / max_int
            # S = BaselineApproxElementwiseDiv.apply(max_val, torch.tensor(max_int, dtype=dtype, device=device))
            Z = 0
            
        # print(f"W: {W}")
        # W_q = BaselineApproxElementwiseDiv.apply(W, S)
        W_q = (W / S)
        # print(f"W_q: {W_q}")
        # if zero_point:
        #     print(f"mx: {W_q.max()}, mn: {W_q.min()}, mxz: {Z.max()}")
        # else:
        #     print(f"mx: {W_q.max()}, mn: {W_q.min()}")
        W_q = torch.round(W_q) + Z
        # W_q = W_q + Z
        # print(f"W_q before clamp: {W_q}")
        # print(f"mx: {W_q.max()}, mn: {W_q.min()}")
        W_q = torch.clamp(W_q, min_int, max_int)
        # print(f"W_q after clamp: {W_q}")
        W_r = FPMAApproxElementwiseMul.apply(W_q - Z, S)
        # W_r = (W_q - Z) * S
        # print(f"W_r: {W_r}")
        max_diff_array.append(torch.max(torch.abs(W_r - W)))
        mse_array.append(torch.mean((W_r - W) ** 2))
    
    # array to tensor
    max_diff = torch.mean(torch.stack(max_diff_array))
    mse = torch.mean(torch.stack(mse_array))
    print(f"Maximum difference between W_r and W: {max_diff}")
    print(f"Mean square error between W_r and W: {mse}")
    # # Model parameters
    # input_size = 256
    # hidden_size = 128
    # output_size = 64
    # batch_size = 32
    # sequence_length = 10  # Example for 3D input

    # # Initialize the model
    # # model = FPMAApproxLinearBF16(input_size, hidden_size, bias=True).to(device)
    # model = FPMAApproxLinearFP16(input_size, hidden_size, bias=True).to(device)
    
    # # Create dummy input with shape [batch_size, sequence_length, input_size]
    # # input_tensor = torch.randn(batch_size, sequence_length, input_size, dtype=torch.bfloat16, device=device)
    # input_tensor = torch.randn(batch_size, sequence_length, input_size, dtype=torch.float16, device=device)
    
    # # Forward pass
    # output_FPMA = model(input_tensor)
    # output_linear = torch.nn.functional.linear(input_tensor, model.weight, model.bias)
    # print(f"Output shape: {output_FPMA.shape}")  # Expected: torch.Size([32, 10, 128])
    
    # max_diff = torch.max(torch.abs(output_FPMA - output_linear))
    # mse = torch.mean((output_FPMA - output_linear) ** 2)
    # print("######torch.randn test result######")
    # print(f"Maximum difference between FPMA and normal matmul: {max_diff}")
    # print(f"Mean square error between FPMA and normal matmul: {mse}")
    # print(f"output_FPMA: {output_FPMA}")
    