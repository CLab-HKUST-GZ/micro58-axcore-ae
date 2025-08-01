from requests import head
import torch
from torch.utils.cpp_extension import load
import os

# Get the project root directory dynamically (2 levels up from current file)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.normpath(project_root)

axcore_cuda_module = load(name="axcore_gemm",
                   extra_include_paths=[os.path.join(project_root, "approximation_computation", "AxCore", "include")],
                   sources=[os.path.join(project_root, "approximation_computation", "AxCore", "kernel", "AxCore.cpp"), 
                            os.path.join(project_root, "approximation_computation", "AxCore", "kernel", "AxCore_fp16.cu")],
                   extra_cflags=["-O3"],  
                   extra_cuda_cflags=["-O3", "--ptxas-options=-v"], 
                   verbose=True)


class AxCoreFunctionFP16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scales, bias=None):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.float16.
            weight (Tensor): Weight tensor of shape (out_features, in_features) with dtype torch.float16.
            bias (Tensor, optional): Bias tensor of shape (out_features,) with dtype torch.float16.
        
        Returns:
            Tensor: Output tensor of shape (*, out_features) with dtype torch.float16.
        """
        
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
        axcore_cuda_module.torch_launch_AxCore_group_gemm_kernel(
            flattened_input, weight, flattened_output, scales, flattened_input.shape[0], out_features, in_features
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



class AxCoreLinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        """
        Initializes the BaselineApproxFP16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(AxCoreLinearFP16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter
        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        
        self.register_buffer('scales', torch.empty(self.out_features, self.in_features // 128, dtype=torch.float16, requires_grad=False, device=dev))
        
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
        return AxCoreFunctionFP16.apply(input, self.weight.T.contiguous(), self.scales.T.contiguous(), self.bias)
    

class AxCoreTypedFunctionFP16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scales, types, g, use_approx_dequant, bias=None):
        """
        Performs the forward pass using the custom CUDA kernel.
        
        Args:
            input (Tensor): Input tensor of shape (*, in_features) with dtype torch.float16.
            weight (Tensor): Weight tensor of shape (out_features, in_features) with dtype torch.float16.
            scales (Tensor): Scales tensor of shape (out_features, in_features // 128) with dtype torch.float16.
            types (Tensor): Types tensor of shape (out_features // 64, in_features // 128) with dtype torch.uint8.
            bias (Tensor, optional): Bias tensor of shape (out_features,) with dtype torch.float16.
        
        Returns:
            Tensor: Output tensor of shape (*, out_features) with dtype torch.float16.
        """
        
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
        axcore_cuda_module.torch_launch_AxCore_group_gemm_typed_kernel(
            flattened_input, weight, flattened_output, scales, types, g, flattened_input.shape[0], out_features, in_features,
            use_approx_dequant
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



class AxCoreTypedLinearFP16(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dev='cuda'):
        """
        Initializes the BaselineApproxFP16 module.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        """
        super(AxCoreTypedLinearFP16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter
        self.register_buffer('weight', torch.empty(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        
        self.register_buffer('scales', torch.empty(self.out_features, self.in_features // 128, dtype=torch.float16, requires_grad=False, device=dev))
        
        self.register_buffer('types', torch.empty(self.out_features // 64, self.in_features // 128, dtype=torch.uint8, requires_grad=False, device=dev))
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
        return AxCoreTypedFunctionFP16.apply(input, self.weight.T.contiguous(), self.scales.T.contiguous(), self.types.T.contiguous(), self.bias)


if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    import random
    import gc
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from useful_for_rmse_test import *
    from approximation_computation.quantization.APAQ.converter import *
    
    # Enforce deterministic CUDA operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.set_printoptions(threshold=float('inf'))
    np.set_printoptions(suppress=True)
    
    size = 512
    rmse_array = []
    snr_array = []
    for i in range(1000):
        # Set seeds for reproducibility for both torch and numpy
        seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) # Seed Python's random module as well

        print(f"\n--- Iteration {i} (seed={seed}) ---")
        
        # fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        #                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float16, device='cuda')
        fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                                  -0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5], dtype=torch.float16, device='cuda')
        random_indices = torch.randint(0, len(fp4_values), (size, size), device='cuda') 
        random_tensor_FP4 = fp4_values[random_indices]
        # print(random_tensor_FP4)
        print("Unique elements in random_tensor_FP4:", torch.unique(random_tensor_FP4))

        # Assuming random_tensor_gen uses either torch or numpy for random numbers
        # Pass the seed to the function to ensure it's used
        random_tensor_FP16 = random_tensor_gen(expo_width=5, mant_width=10, custom_bias=None, row=32, col=size, seed=seed)
        random_tensor_FP16 = random_tensor_FP16.half().to('cuda')
        # random_tensor_FP16 = torch.randn(32, 4096, device='cuda').half()
        # print(random_tensor_FP16)

        # Check for inf/nan in inputs
        if torch.any(torch.isinf(random_tensor_FP16)) or torch.any(torch.isnan(random_tensor_FP16)):
            print("!!! Inf or NaN found in random_tensor_FP16")
        if torch.any(torch.isinf(random_tensor_FP4)) or torch.any(torch.isnan(random_tensor_FP4)):
            print("!!! Inf or NaN found in random_tensor_FP4")
        
        S = torch.ones((size // 128, size), device='cuda', dtype=torch.float16)
        # print(S)
        
        D_w = 1 * random_tensor_FP4
        # print(D_w)
        
        O_r = torch.matmul(random_tensor_FP16, D_w)
        # print(O_r)


        
        # random_tensor_FP4_converted = fp4_class_convert(random_tensor_FP4)
        random_tensor_FP4_converted = fp4_e1m2_class_convert(random_tensor_FP4)
        # print(random_tensor_FP4_converted)
        
        O_a = AxCoreFunctionFP16.apply(random_tensor_FP16, random_tensor_FP4_converted.contiguous(), S.T.contiguous(), None)
        # print(O_a)

        O_a = O_a.to(torch.float32)
        O_r = O_r.to(torch.float32)
        rmse = torch.sqrt(torch.mean((O_a - O_r) ** 2))

        diff = torch.abs(O_a - O_r)
        snr = -10*torch.log10(rmse/torch.sqrt(torch.mean(O_r ** 2)))

        print(f"Max difference: {torch.max(diff)}")
        print(f"RMSE: {rmse.item() if isinstance(rmse, torch.Tensor) else rmse}")
        print(f"SNR: {snr.item() if isinstance(snr, torch.Tensor) else snr}")
        
        rmse_array.append(rmse.item() if isinstance(rmse, torch.Tensor) else rmse)
        snr_array.append(snr.item() if isinstance(snr, torch.Tensor) else snr)
        
        torch.cuda.synchronize()
        del O_a, O_r, random_tensor_FP4_converted, random_tensor_FP16, S, D_w, fp4_values, random_indices, random_tensor_FP4
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nFinal RMSE array:")
    print(rmse_array)
    # print("\nFinal Percentage array:")
    # print(percentage_array)
    # Use numpy's nanmean to ignore nans in the average calculation
    print(f"Average RMSE: {np.nanmean(rmse_array)}")
    print(f"Average SNR: {np.nanmean(snr_array)}")