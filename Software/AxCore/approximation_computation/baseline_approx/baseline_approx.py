from requests import head
import torch
from torch.utils.cpp_extension import load

import os

# Get the project root directory dynamically (2 levels up from current file)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.normpath(project_root)

cuda_module = load(name="baseline_approx",
                   extra_include_paths=[os.path.join(project_root, "approximation_computation", "baseline_approx", "include")],
                   sources=[os.path.join(project_root, "approximation_computation", "baseline_approx", "kernel", "baseline_approx.cpp"), 
                            os.path.join(project_root, "approximation_computation", "baseline_approx", "kernel", "baseline_approx_kernel_bf16.cu"),
                            os.path.join(project_root, "approximation_computation", "baseline_approx", "kernel", "baseline_approx_kernel_fp16.cu")],
                   extra_cflags=["-O3"], 
                   extra_cuda_cflags=["-O3"],  
                   verbose=True)

class BaselineApproxMatmulBF16(torch.autograd.Function):
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
        # A = A.contiguous()
        # B = B.contiguous() # y = xW^T # shape: (in_features, out_features)
        
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
        cuda_module.torch_launch_baseline_approx_kernel_bf16_batched(
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
    
class BaselineApproxFunctionBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.bfloat16.
            weight (Tensor): Weight tensor of shape (out_features, in_features) with dtype torch.bfloat16.
            bias (Tensor, optional): Bias tensor of shape (out_features,) with dtype torch.bfloat16.
        
        Returns:
            Tensor: Output tensor of shape (*, out_features) with dtype torch.bfloat16.
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
        cuda_module.torch_launch_baseline_approx_kernel_bf16(
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

    
class BaselineApproxLinearBF16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dev='cuda'):
        """
        Initializes the BaselineApproxBF16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(BaselineApproxLinearBF16, self).__init__()
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
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.bfloat16.
        
        Returns:
            Tensor: Output tensor of shape (*, out_features) with dtype torch.bfloat16.
        """
        return BaselineApproxFunctionBF16.apply(input, self.weight, self.bias)


class BaselineApproxMatmulFP16(torch.autograd.Function):
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
        
        # Ensure input tensors are contiguous
        # A = A.contiguous()
        # B = B.contiguous()
        
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
        cuda_module.torch_launch_baseline_approx_kernel_fp16_batched(
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

class BaselineApproxFunctionFP16(torch.autograd.Function):
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
        
        # Ensure input tensors are contiguous
        # input = input.contiguous()
        # weight = weight.t().contiguous() # y = xW^T # shape: (in_features, out_features)
        # if bias is not None:
        #     bias = bias.contiguous()
        
        original_shape = input.shape[:-1]
        in_features = input.shape[-1]
        in_features_w, out_features = weight.shape
        # out_features, in_features_w = weight.shape
        
        if in_features != in_features_w:
            raise ValueError(f"Incompatible dimensions: input has {in_features} features, but weight has {in_features_w} features.")
        
        flattened_input = input.view(-1, in_features)
        
        # Allocate output tensor
        flattened_output = torch.empty((flattened_input.shape[0], out_features), dtype=torch.float16, device=input.device)
        
        # Launch the CUDA kernel
        cuda_module.torch_launch_baseline_approx_kernel_fp16(
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


class BaselineApproxLinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        """
        Initializes the BaselineApproxFP16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(BaselineApproxLinearFP16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter
        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        
        # Initialize bias parameter if required
        if bias:
            self.register_buffer('bias', torch.empty(
                (1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        Performs the forward pass.
        
        Args:
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.float16.
        
        Returns:
            Tensor: Output tensor of shape (*, out_features) with dtype torch.float16.
        """
        return BaselineApproxFunctionFP16.apply(input, self.weight.t().contiguous(), self.bias)

class BaselineApproxElementwiseMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        if B.dim() != 2:
            raise ValueError("B must be a 2D tensor")
        A_row, A_col = A.shape
        B_row, B_col = B.shape
        if A.dtype == torch.float16:
            C = cuda_module.torch_launch_baseline_elementwisemul_kernel_fp16(A, B, A_row, A_col, B_row, B_col)
        else:
            C = cuda_module.torch_launch_baseline_elementwisemul_kernel_bf16(A, B, A_row, A_col, B_row, B_col)
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
    
class BaselineApproxElementwiseDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        if B.dim() == 0:
            B = B.unsqueeze(0)
            B = B.unsqueeze(0)
        if B.dim() == 1:
            B = B.unsqueeze(1)
        A_row, A_col = A.shape
        B_row, B_col = B.shape
        if A.dtype == torch.float16:
            C = cuda_module.torch_launch_baseline_elementwisediv_kernel_fp16(A, B, A_row, A_col, B_row, B_col)
        else:
            C = cuda_module.torch_launch_baseline_elementwisediv_kernel_bf16(A, B, A_row, A_col, B_row, B_col)
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

class DequantizeElementwiseMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, comp):
        if B.dim() == 0:
            B = B.unsqueeze(0)
            B = B.unsqueeze(0)
        if B.dim() == 1:
            B = B.unsqueeze(1)
        A_row, A_col = A.shape
        B_row, B_col = B.shape
        if A.dtype == torch.float16:
            C = cuda_module.torch_launch_dequantize_elementwisemul_kernel_fp16(A, B, A_row, A_col, B_row, B_col, comp)
        else:
            C = cuda_module.torch_launch_dequantize_elementwisemul_kernel_bf16(A, B, A_row, A_col, B_row, B_col, comp)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass is not implemented as this is a custom approximation.
        To enable training, you would need to implement gradient computations.
        """
        # To enable backpropagation, implement gradient computations here
        # For simplicity, we'll assume gradients are not needed
        return None, None, None, None, None, None, None

class QuantizeElementwiseDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, comp):
        if B.dim() == 0:
            B = B.unsqueeze(0)
            B = B.unsqueeze(0)
        if B.dim() == 1:
            B = B.unsqueeze(1)
        A_row, A_col = A.shape
        B_row, B_col = B.shape
        # print("A", A.shape)
        # print("B", B.shape)
        if A.dtype == torch.float16:
            C = cuda_module.torch_launch_quantize_elementwisediv_kernel_fp16(A, B, A_row, A_col, B_row, B_col, comp)
        else:
            C = cuda_module.torch_launch_quantize_elementwisediv_kernel_bf16(A, B, A_row, A_col, B_row, B_col, comp)
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
    
    import torch.nn.functional as F
    def safe_kl_div(p, q, eps=1e-8):
        """安全KL散度计算"""
        p = F.softmax(p, dim=1) + eps
        q = F.softmax(q, dim=1) + eps
        log_p = torch.log(p)
        log_q = torch.log(q)
        return (p * (log_p - log_q)).sum(dim=1).mean()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    n_bits = 4
    mantissa_bit = 1
    qint = False
    zero_point = False
    # batch = 200
    batch = 2
    max_diff_array = []
    mse_array = []
    cos_sim_array = []
    kl_div_array = []
    comp_arrary = []
    W = torch.randn(batch, 64, dtype=dtype, device=device)
    S = torch.randn(batch, 1, dtype=dtype, device=device)
    Z = torch.ones(batch, 1, dtype=dtype, device=device)
    W *= 0.03
    for i in range(1):
        if qint:
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
        else:
            M = mantissa_bit
            E = n_bits - 1 - M
            bias = 2 ** (E - 1) - 1
            max_float = (2 - 2 ** (-M)) * 2 ** (
                    2**E - 1 - bias
                )
            print(f"max_float: {max_float}")
            min_float = -max_float
            max_val = W.abs().amax(dim=1, keepdim=True)
            S = max_val / max_float
            W_unscaled = (W / S)
            # W_unscaled = BaselineApproxElementwiseDiv.apply(W, S)
            # W_unscaled = QuantizeElementwiseDiv.apply(W, S, i-100)
            W_unscaled = torch.clamp(W_unscaled, min_float, max_float)
            print(f"W_unscaled: {W_unscaled}")
            W_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(W_unscaled)) + bias)).detach(), 1.0)
            scales = 2.0 ** (W_log_scales - M - bias)
            W_q = (W_unscaled / scales).round()
            print(f"W_q: {W_q}")
            W_q = W_q * scales
            print(f"W_q: {W_q}")
            Z = 0
    
        # print(f"W: {W}")
        # W_q = BaselineApproxElementwiseDiv.apply(W, S)
        # W_q = (W / S)
        # W_q = torch.round(W_q) + Z
        # W_q = torch.clamp(W_q, min_int, max_int)
        # W_r = BaselineApproxElementwiseMul.apply(W_q - Z, S)
        W_r = DequantizeElementwiseMul.apply(W_q - Z, S, i-100)
        # W_r = (W_q - Z) * S
        # print(f"S: {S}")
        # print(f"W_r: {W_r}")
        max_diff_array.append(torch.max(torch.abs(W_r - W)))
        mse_array.append(torch.mean((W_r - W) ** 2))
        kl_div = safe_kl_div(W, W_r)
        kl_div_array.append(kl_div.detach())
        cos_sim = torch.nn.functional.cosine_similarity(W_r.flatten(1), W.flatten(1), dim=1)
        cos_sim_array.append(cos_sim.mean().detach())
        comp_arrary.append(i-100)
    
    # array to tensor
    mse_tensor = torch.stack(mse_array)
    # min_mse = torch.min(mse_tensor)
    # min_idx = torch.argmin(mse_tensor)
    cos_sim_tensor = torch.stack(cos_sim_array)
    cos_sim = torch.max(cos_sim_tensor)
    min_idx = torch.argmax(cos_sim_tensor)
    best_comp = comp_arrary[min_idx]
    print(f"Minimum MSE: {mse_array[min_idx].item():.4e} at index {min_idx.item()} (comp={best_comp}), cos_sim={cos_sim_array[min_idx].item():.4e}, max_diff={max_diff_array[min_idx].item():.4e}")
    # max_diff = torch.mean(torch.stack(max_diff_array))
    # mse = torch.mean(torch.stack(mse_array))
    # kl_div = torch.mean(torch.stack(kl_div_array)).item()
    # cos_sim = torch.mean(torch.stack(cos_sim_array)).item()
    # print(f"Maximum difference between W_r and W: {max_diff}")
    # print(f"Mean square error between W_r and W: {mse}")
    # print(f"KL divergence between W and W_r: {kl_div}")
    # print(f"Cosine similarity between W and W_r: {cos_sim}")
    
    # def print_fp16_binary(tensor, name='Tensor'):
    #     import numpy as np
    #     """打印半精度浮点张量的二进制表示"""
    #     # 转换为CPU上的numpy数组并转为uint16
    #     np_tensor = tensor.cpu().view(torch.uint16).numpy()
        
    #     print(f'{name} binary representation:')
    #     for i in np.ndindex(np_tensor.shape):
    #         # 转换为16位二进制字符串，前面补零
    #         binary_str = format(np_tensor[i], '016b')
    #         print(f'Element {i}: {binary_str[:1]} {binary_str[1:6]} {binary_str[6:]}')
    
    # A = torch.ones(2, 2, dtype=dtype, device=device)
    # B = torch.ones(2, 2, dtype=dtype, device=device)
    # A *= 0.1
    # B *= 0.1
    
    # C = BaselineApproxElementwiseMul.apply(A, B)
    # D = A * B
    
    # assert torch.isnan(C).sum() == 0
    
    # print(f"C: {C}, D: {D}")
    
    # print_fp16_binary(A, name='A')
    # print_fp16_binary(B, name='B')
    # print_fp16_binary(C, name='C')
    # print_fp16_binary(D, name='D')
    
    # # Model parameters
    # input_size = 782
    # hidden_size = 1024
    # output_size = 64
    # batch_size = 32
    # sequence_length = 256  # Example for 3D input
    
    # batch_size = 2
    # num_heads = 32
    # length = 4096
    # head_dim = 128

    # Initialize the model
    # model = BaselineApproxLinearBF16(input_size, hidden_size, bias=True).to(device)
    
    
    # Create dummy input with shape [batch_size, sequence_length, input_size]
    # input_tensor = torch.randn(batch_size, sequence_length, input_size, dtype=torch.bfloat16, device=device)
    # input_tensor = torch.randn(batch_size, sequence_length, input_size, dtype=torch.float16, device=device)
    # A = torch.randn(batch_size, num_heads, length, head_dim, dtype=torch.float16, device=device)
    # B = torch.randn(batch_size, num_heads, length, head_dim, dtype=torch.float16, device=device)
    # B = B.transpose(2, 3)
    
    
    # # Forward pass
    # model = torch.nn.Linear(input_size, hidden_size, bias=True).to(device).to(torch.float16)
    # output_linear = model(input_tensor)
    # model_ap = BaselineApproxLinearFP16(input_size, hidden_size, bias=True).to(device).to(torch.float16)
    # model_ap.weight.copy_(model.weight)
    # if model_ap.bias is not None:
    #     model_ap.bias.copy_(model.bias)
    # output_baseline = model_ap(input_tensor)
    # output_baseline = BaselineApproxMatmulFP16.apply(A, B)
    # output_linear = torch.matmul(A, B)
    
    # print(f"Output shape: {output_baseline.shape}")  # Expected: torch.Size([32, 10, 128])
    
    # max_diff = torch.max(torch.abs(output_baseline - output_linear))
    # mse = torch.mean((output_baseline - output_linear) ** 2)
    # print("######torch.randn test result######")
    # print(f"Maximum difference between baseline and normal matmul: {max_diff}")
    # print(f"Mean square error between baseline and normal matmul: {mse}")
    # print(f"output_linear: {output_linear}")
    # print(f"output_baseline: {output_baseline}")