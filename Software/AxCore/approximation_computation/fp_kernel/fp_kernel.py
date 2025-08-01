from requests import head
import torch
from torch.utils.cpp_extension import load

path = '/data-ssd/home/jiaxiang/codes/FPMA_TENSOR_CORE'
cuda_module = load(name="fp_kernel",
                   extra_include_paths=[f"{path}/approximation_computation/fp_kernel/include"],
                   sources=[f"{path}/approximation_computation/fp_kernel/kernel/fp_kernel.cpp", 
                            f"{path}/approximation_computation/fp_kernel/kernel/wmma_kernel_fp16.cu",
                            f"{path}/approximation_computation/fp_kernel/kernel/cublas_kernel_fp16.cu",
                            f"{path}/approximation_computation/fp_kernel/kernel/fp_gemm.cu"],
                   extra_cflags=["-O3"],  # 降低C++优化等级
                   extra_cuda_cflags=["-O3"],  # 降低CUDA优化等级
                   verbose=True)

class WMMAGEMMFunctionFP16(torch.autograd.Function):
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
        input = input.to(torch.float16) # shape: [M, K]
        weight = weight.to(torch.float16) # shape: [N, K]
        if bias is not None:
            bias = bias.to(torch.float16)
        
        original_shape = input.shape[:-1]
        in_features = input.shape[-1]
        # out_features, in_features_w = weight.shape
        in_features_w, out_features = weight.shape
        
        if in_features != in_features_w:
            raise ValueError(f"Incompatible dimensions: input has {in_features} features, but weight has {in_features_w} features.")
        
        flattened_input = input.view(-1, in_features)
        
        # Allocate output tensor
        flattened_output = torch.empty((flattened_input.shape[0], out_features), dtype=torch.float16, device=input.device)
        
        # Launch the CUDA kernel
        cuda_module.torch_launch_wmma_gemm_kernel_fp16(
            flattened_input, weight, flattened_output, flattened_input.shape[0], out_features, in_features
        )
        
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
    
class WMMALinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        """
        Initializes the WMMALinearFP16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(WMMALinearFP16, self).__init__()
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
        return WMMAGEMMFunctionFP16.apply(input, self.weight.t(), self.bias)
    
class GEMMFunctionFP16(torch.autograd.Function):
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
        # out_features, in_features_w = weight.shape
        in_features_w, out_features = weight.shape

        if in_features != in_features_w:
            raise ValueError(f"Incompatible dimensions: input has {in_features} features, but weight has {in_features_w} features.")
        
        flattened_input = input.view(-1, in_features)
        
        # Allocate output tensor
        flattened_output = torch.empty((flattened_input.shape[0], out_features), dtype=torch.float16, device=input.device)
        
        # Launch the CUDA kernel
        # cuda_module.torch_launch_fp_gemm_kernel_fp16(
        #     flattened_input, weight, flattened_output, flattened_input.shape[0], out_features, in_features
        # )
        cuda_module.torch_launch_group_gemm_kernel(
            flattened_input, weight, flattened_output, flattened_input.shape[0], out_features, in_features
        )
        
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
    
class GEMMLinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        """
        Initializes the GEMMLinearFP16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(GEMMLinearFP16, self).__init__()
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
        return GEMMFunctionFP16.apply(input, self.weight.T.contiguous(), self.bias)
    
# Example usage
if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 1024
    hidden_size = 1024
    output_size = 1024
    batch_size = 10
    sequence_length = 1024  # Example for 3D input
    
    
    # Create dummy input with shape [batch_size, sequence_length, input_size]
    input_tensor = torch.randn(batch_size, sequence_length, input_size, dtype=torch.float16, device=device)

    # # Forward pass
    model = torch.nn.Linear(input_size, hidden_size, bias=False).to(device).to(torch.float16)
    output_linear = model(input_tensor)
    model_ap = GEMMLinearFP16(input_size, hidden_size, bias=False).to(device).to(torch.float16)
    model_ap.weight.copy_(model.weight)
    if model_ap.bias is not None:
        model_ap.bias.copy_(model.bias)
    output_baseline = model_ap(input_tensor)
    
    print(f"Output shape: {output_baseline.shape}")  # Expected: torch.Size([32, 10, 128])
    # print(f"output_linear: {output_linear}")
    # print(f"output_baseline: {output_baseline}")
    
    max_diff = torch.max(torch.abs(output_baseline - output_linear))
    mse = torch.mean((output_baseline - output_linear) ** 2)
    print("######torch.randn test result######")
    print(f"Maximum difference between baseline and normal matmul: {max_diff}")
    print(f"Mean square error between baseline and normal matmul: {mse}")
    # print(f"output_linear: {output_linear}")
    # print(f"output_wmma: {output_baseline}")