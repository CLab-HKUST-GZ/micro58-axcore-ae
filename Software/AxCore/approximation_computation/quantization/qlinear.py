import torch
import torch.nn as nn
from functools import partial
# Use absolute imports to avoid package issues
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approximation_computation.quantization.APAQ.converter import (
        fp4_class_convert, fp4_e1m2_class_convert
    )

from ..AxCore.AxCore import *

try:
    # Try relative imports first (when imported as a package)
    from .quant_funcs import (
        fp_scale,
        fp_scale_ap,
        quantize_weight_per_channel_absmax,
        quantize_weight_per_tensor_absmax,
        pseudo_quantize_tensor,
        pseudo_quantize_tensor_dse
    )
    from ..baseline_approx.baseline_approx import *
except ImportError:
    # Fall back to absolute imports (when run directly)
    from approximation_computation.quantization.quant_funcs import (
        fp_scale,
        fp_scale_ap,
        quantize_weight_per_channel_absmax,
        quantize_weight_per_tensor_absmax,
        pseudo_quantize_tensor,
        pseudo_quantize_tensor_dse
    )
    from approximation_computation.baseline_approx.baseline_approx import *

class WLinear(nn.Module):
    """
    Quantized weight Linear module - without activation quantization.
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=8, dev='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit

        self.register_buffer('weight', torch.zeros(self.out_features,
                                                  self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_buffer('bias', None)

    def to(self, *args, **kwargs):
        super(WLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

    @staticmethod
    def from_float(module, w_bit=4, weight_group=128):
        assert isinstance(module, torch.nn.Linear)
        new_module = WLinear(
            module.in_features, module.out_features, module.bias is not None, w_bit=w_bit, dev=module.weight.device)
        
        new_module.weight = pseudo_quantize_tensor(module.weight, n_bits=w_bit, q_group_size=weight_group, inplace=True)
        
        if module.bias is not None:
            new_module.bias = module.bias
        del module
        return new_module

    def __repr__(self):
        return 'W{}Linear'.format(self.w_bit)


class DSEWLinear(WLinear):
    """
    Weight Quantized Linear module with DSE for optimal mantissa bit configuration
    based on calibration data and layer output MSE.
    Simplified without activation quantization.
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=8, weight_group=128, dev='cuda'):
        super().__init__(in_features, out_features, bias, w_bit, dev)
        # # Store the original weight for reference
        # self.register_buffer('orig_weight', torch.zeros(self.out_features,
        #                                                self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        
    @staticmethod
    def from_float(module, w_bit=4, weight_group=128, calibration_data=None):
        """
        Create a DSEWLinear module from a floating point module.
        calibration_data: Tensor of shape [batch_size, in_features] used to find optimal mantissa bit config
        """
        assert isinstance(module, torch.nn.Linear)
        assert calibration_data is not None, "Calibration data is required for DSEWLinear"
        
        device = module.weight.device
        new_module = DSEWLinear(
            module.in_features, module.out_features, module.bias is not None, 
            w_bit=w_bit, weight_group=weight_group, dev=device)
        
        # # Keep original weights for reference
        # new_module.orig_weight = module.weight.clone()
        
        # Perform DSE to find optimal mantissa bit configuration
        new_module.weight, M_counts = DSEWLinear._find_optimal_weight_format(
            module, calibration_data, w_bit, weight_group, device)
        
        if module.bias is not None:
            new_module.bias = module.bias
        
        del module
        return new_module, M_counts
    
    @staticmethod
    @torch.no_grad()
    def _find_optimal_weight_format(module, calibration_data, w_bit, weight_group, device, out_block_size=64):
        """
        Find the optimal mantissa bit configuration for each weight block by minimizing
        the MSE between original output and quantized output.
        
        Implements a hybrid quantization approach:
        - Block quantization with blocks of size [out_block_size, weight_group] for data type selection
        - Group quantization along in_channels for scale factors
        
        GPU-optimized version with maximized parallel operations for faster DSE.
        
        Args:
            module: The module to quantize
            calibration_data: Calibration data for optimization
            w_bit: Bit width for weights
            weight_group: Group size for input channels
            device: Device to use
            out_block_size: Size of output channel blocks, defaults to weight_group if None
        """
        apmul = DequantizeElementwiseMul
        apdiv = QuantizeElementwiseDiv
        comp = 0
        
        # Original weights and shape information
        orig_weight = module.weight
        org_weight_shape = orig_weight.shape  # [out_features, in_features]
        
        # Prepare calibration data
        if len(calibration_data.shape) == 3:  # batch, seq_len, hidden
            calib_flattened = calibration_data.reshape(-1, calibration_data.shape[-1])
        else:
            calib_flattened = calibration_data
            
        # Check dimensions for block quantization
        if weight_group > 0:
            # Set out_block_size to weight_group if not provided
            if out_block_size is None:
                out_block_size = weight_group
                
            # Check if dimensions are compatible
            assert org_weight_shape[0] % out_block_size == 0, f"out_features ({org_weight_shape[0]}) must be divisible by out_block_size ({out_block_size})"
            assert org_weight_shape[1] % weight_group == 0, f"in_features ({org_weight_shape[1]}) must be divisible by weight_group ({weight_group})"
            
            # Calculate block dimensions
            num_out_blocks = org_weight_shape[0] // out_block_size
            num_in_blocks = org_weight_shape[1] // weight_group
            total_blocks = num_out_blocks * num_in_blocks
            
            # Pre-compute parameters for all mantissa bit configurations
            mantissa_bits = torch.tensor([0, 1, 2], device=device)  # M = 0, 1, 2
            Es = torch.tensor([w_bit - 1 - m for m in mantissa_bits], device=device)
            biases = torch.tensor([2 ** (e - 1) - 1 for e in Es], device=device)
            max_floats = torch.tensor([(2 - 2 ** (-m)) * 2 ** (2**e - 1 - b) for m, e, b in zip(mantissa_bits, Es, biases)], device=device)
            min_floats = -max_floats
            
            # Counter for mantissa bit distribution
            M_counts = torch.zeros(3, dtype=torch.long, device=device)
            
            # Final quantized weights
            best_weight = orig_weight.clone()
            
            # Process blocks in batches to avoid GPU memory issues
            batch_size = 32  # Adjust based on GPU memory
            num_batches = (total_blocks + batch_size - 1) // batch_size
            
            # Create flattened block index list for parallel processing
            block_indices = [(i, j) for i in range(num_out_blocks) for j in range(num_in_blocks)]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_blocks)
                batch_indices = block_indices[start_idx:end_idx]
                batch_size_actual = len(batch_indices)
                
                # Pre-allocate tensors for this batch
                weight_blocks = torch.zeros((batch_size_actual, out_block_size, weight_group), device=device, dtype=orig_weight.dtype)
                original_block_outputs = torch.zeros((batch_size_actual, calib_flattened.shape[0], out_block_size), device=device, dtype=orig_weight.dtype)
                block_inputs = torch.zeros((batch_size_actual, calib_flattened.shape[0], weight_group), device=device, dtype=orig_weight.dtype)
                
                # Extract all blocks and inputs for this batch
                for batch_block_idx, (out_block_idx, in_block_idx) in enumerate(batch_indices):
                    out_start = out_block_idx * out_block_size
                    out_end = out_start + out_block_size
                    in_start = in_block_idx * weight_group
                    in_end = in_start + weight_group
                    
                    # Extract the weight block
                    weight_blocks[batch_block_idx] = orig_weight[out_start:out_end, in_start:in_end]
                    
                    # Extract calibration data for this block's input channels
                    block_inputs[batch_block_idx] = calib_flattened[:, in_start:in_end]
                    
                    # Compute original output contribution from this block
                    original_block_outputs[batch_block_idx] = torch.matmul(block_inputs[batch_block_idx], weight_blocks[batch_block_idx].t())
                
                # Process all mantissa options in parallel
                all_block_mses = torch.zeros((batch_size_actual, 3), device=device)
                
                for m_idx, (M, E, bias, max_float, min_float) in enumerate(zip(mantissa_bits, Es, biases, max_floats, min_floats)):
                    # Create tensor to hold reconstructed blocks for this mantissa config
                    reconstructed_blocks = torch.zeros_like(weight_blocks)
                    
                    for batch_block_idx in range(batch_size_actual):
                        weight_block = weight_blocks[batch_block_idx]
                        
                        # Compute row-wise max values (per output channel)
                        row_max_vals = weight_block.abs().amax(dim=1, keepdim=True)  # [out_block_size, 1]
                        
                        # Scale factors for each row
                        S = apdiv.apply(row_max_vals, torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device), comp)
                        
                        # Unscale using row-level scale factors - vectorized operation
                        unscaled_block = apdiv.apply(weight_block, S, comp)
                        
                        # Clamp the unscaled values
                        unscaled_block = torch.clamp(unscaled_block, min_float, max_float)
                        
                        # Handle zeros properly in log2
                        abs_w = torch.abs(unscaled_block)
                        eps = 1e-10
                        abs_w = torch.where(abs_w > 0, abs_w, torch.tensor(eps, device=device))
                        
                        # Log-domain quantization
                        weight_log_scales = torch.clamp((torch.floor(torch.log2(abs_w) + bias)).detach(), 1.0)
                        scales = 2.0 ** (weight_log_scales - M - bias)
                        block_q = torch.round(unscaled_block / scales) * scales
                        
                        # Scale back using row-level scale factors - vectorized operation
                        reconstructed_blocks[batch_block_idx] = apmul.apply(block_q, S, comp)
                    
                    # Compute quantized outputs for all blocks in this batch for this mantissa option
                    quant_outputs = torch.zeros_like(original_block_outputs)
                    for batch_block_idx in range(batch_size_actual):
                        quant_outputs[batch_block_idx] = torch.matmul(
                            block_inputs[batch_block_idx], 
                            reconstructed_blocks[batch_block_idx].t()
                        )
                    
                    # Calculate MSE for each block with this mantissa option
                    all_block_mses[:, m_idx] = torch.mean((original_block_outputs - quant_outputs) ** 2, dim=(1, 2))
                
                # Find best mantissa configuration for each block in batch
                best_m_indices = torch.argmin(all_block_mses, dim=1)
                
                # Update M_counts based on best choices
                for m in range(3):
                    M_counts[m] += torch.sum(best_m_indices == m).item()
                
                # Update the best weight matrix with the optimal mantissa config for each block
                for batch_block_idx, (out_block_idx, in_block_idx) in enumerate(batch_indices):
                    best_m = best_m_indices[batch_block_idx].item()
                    out_start = out_block_idx * out_block_size
                    out_end = out_start + out_block_size
                    in_start = in_block_idx * weight_group
                    in_end = in_start + weight_group
                    
                    # Recompute the best blocks (we could store them but this saves memory)
                    weight_block = weight_blocks[batch_block_idx]
                    M, E, bias = mantissa_bits[best_m], Es[best_m], biases[best_m]
                    max_float, min_float = max_floats[best_m], min_floats[best_m]
                    
                    # Compute row-wise max values (per output channel)
                    row_max_vals = weight_block.abs().amax(dim=1, keepdim=True)
                    
                    # Scale factors for each row
                    S = apdiv.apply(row_max_vals, torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device), comp)
                    
                    # Unscale using row-level scale factors - vectorized operation
                    unscaled_block = apdiv.apply(weight_block, S, comp)
                    
                    # Clamp the unscaled values
                    unscaled_block = torch.clamp(unscaled_block, min_float, max_float)
                    
                    # Handle zeros properly in log2
                    abs_w = torch.abs(unscaled_block)
                    eps = 1e-10
                    abs_w = torch.where(abs_w > 0, abs_w, torch.tensor(eps, device=device))
                    
                    # Log-domain quantization
                    weight_log_scales = torch.clamp((torch.floor(torch.log2(abs_w) + bias)).detach(), 1.0)
                    scales = 2.0 ** (weight_log_scales - M - bias)
                    block_q = torch.round(unscaled_block / scales) * scales
                    
                    # Scale back using row-level scale factors - vectorized operation
                    block_r = apmul.apply(block_q, S, comp)
                    
                    # Update the best weight matrix
                    best_weight[out_start:out_end, in_start:in_end] = block_r
        
        else:
            raise NotImplementedError("weight group size must be > 0")
            
        # Print statistics about chosen formats
        print(f'M bit distribution: M=0: {M_counts[0]}, M=1: {M_counts[1]}, M=2: {M_counts[2]}')
        
        return best_weight, M_counts
    
    def __repr__(self):
        return 'DSE-W{}Linear'.format(self.w_bit)


# -------------------------------------------------------------
# AxCoreDSEWLinear integrates AxCore typed GEMM kernel to handle
# per-tile data-type selection (M bits 0/1/2). It stores:
#   weight  : fp16   [out, in]
#   scales  : fp16   [out, in//g]
#   types   : uint8  [out//64, in//g]  (value 0/1/2)
# -------------------------------------------------------------
class AxCoreDSEWLinear(torch.nn.Module):
    """
    Weight Quantized Linear module with DSE for optimal mantissa bit configuration
    based on calibration data and layer output MSE.
    Simplified without activation quantization.
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=8,
                 weight_group=128, use_approx_dequant=True, dev='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.weight_group = weight_group
        self.use_approx_dequant = use_approx_dequant

        self.register_buffer('weight', torch.zeros(out_features, in_features,
                                                   dtype=torch.float16, requires_grad=False, device=dev))

        # per-group scale along input dim (K//128)
        self.register_buffer('scale_buf', torch.zeros(out_features, in_features // weight_group,
                                                   dtype=torch.float16, requires_grad=False, device=dev))

        # per-tile types (N//64 × K//128)
        self.register_buffer('types', torch.zeros(out_features // 64, in_features // weight_group,
                                                  dtype=torch.uint8, requires_grad=False, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((1, out_features), dtype=torch.float16,
                                                      requires_grad=False, device=dev))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def from_float(module, w_bit=4, weight_group=128, use_approx_dequant=True, calibration_data=None):
        """
        Create a AxCoreDSEWLinear module from a floating point module.
        calibration_data: Tensor of shape [batch_size, in_features] used to find optimal mantissa bit config
        """
        assert isinstance(module, torch.nn.Linear)
        assert calibration_data is not None, "Calibration data is required for DSEWLinear"
        
        device = module.weight.device
        new_module = AxCoreDSEWLinear(module.in_features, module.out_features,
                                      module.bias is not None, w_bit,
                                      weight_group, use_approx_dequant, dev=device)

        # Run DSE and obtain weight / scales / types
        q_w, q_s, q_t, M_counts = AxCoreDSEWLinear._find_optimal_weight_format(
            module, calibration_data, w_bit, weight_group, use_approx_dequant, device)

        new_module.weight.copy_(q_w)
        new_module.scale_buf.copy_(q_s)
        new_module.types.copy_(q_t)

        if module.bias is not None:
            new_module.bias = module.bias.to(device)
         
        del module
        return new_module, M_counts

    @staticmethod
    @torch.no_grad()
    def _find_optimal_weight_format(module, calibration_data, w_bit, weight_group, use_approx_dequant,
                                    device,
                                    out_block_size=64):
        """
        Find the optimal mantissa bit configuration for each weight block by minimizing
        the MSE between original output and quantized output.
        
        Implements a hybrid quantization approach:
        - Block quantization with blocks of size [out_block_size, weight_group] for data type selection
        - Group quantization along in_channels for scale factors
        
        GPU-optimized version with maximized parallel operations for faster DSE.
        
        Args:
            module: The module to quantize
            calibration_data: Calibration data for optimization
            w_bit: Bit width for weights
            weight_group: Group size for input channels
            device: Device to use
            out_block_size: Size of output channel blocks, defaults to weight_group if None
        """
        apmul = DequantizeElementwiseMul
        apdiv = QuantizeElementwiseDiv
        comp = 0
        
        clip = False
        
        # Original weights and shape information
        orig_weight = module.weight
        org_weight_shape = orig_weight.shape  # [out_features, in_features]
        
        # Prepare calibration data
        if len(calibration_data.shape) == 3:  # batch, seq_len, hidden
            calib_flattened = calibration_data.reshape(-1, calibration_data.shape[-1])
        else:
            calib_flattened = calibration_data
            
        # Check dimensions for block quantization
        if weight_group > 0:
            # Set out_block_size to weight_group if not provided
            if out_block_size is None:
                out_block_size = weight_group
                
            # Check if dimensions are compatible
            assert org_weight_shape[0] % out_block_size == 0, f"out_features ({org_weight_shape[0]}) must be divisible by out_block_size ({out_block_size})"
            assert org_weight_shape[1] % weight_group == 0, f"in_features ({org_weight_shape[1]}) must be divisible by weight_group ({weight_group})"
            
            # Calculate block dimensions
            num_out_blocks = org_weight_shape[0] // out_block_size
            num_in_blocks = org_weight_shape[1] // weight_group
            total_blocks = num_out_blocks * num_in_blocks
            
            # Pre-compute parameters for all mantissa bit configurations
            mantissa_bits = torch.tensor([0, 1, 2], dtype=torch.float16, device=device)  # M = 0, 1, 2
            Es = torch.tensor([w_bit - 1 - m for m in mantissa_bits], dtype=torch.float16, device=device)
            biases = torch.tensor([2 ** (e - 1) - 1 for e in Es], dtype=torch.float16, device=device)
            max_floats = torch.tensor([(2 - 2 ** (-m)) * 2 ** (2**e - 1 - b) for m, e, b in zip(mantissa_bits, Es, biases)], dtype=torch.float16, device=device)
            min_floats = -max_floats
            
            # Counter & outputs
            M_counts = torch.zeros(3, dtype=torch.long, device=device)

            scale_buf = torch.zeros(org_weight_shape[0], org_weight_shape[1] // weight_group,
                                 dtype=torch.float16, device=device)
            types_map = torch.zeros(org_weight_shape[0] // 64, org_weight_shape[1] // weight_group,
                                    dtype=torch.uint8, device=device)
            
            # Final quantized weights
            best_weight = orig_weight.clone()
            
            # Process blocks in batches to avoid GPU memory issues
            batch_size = 32  # Adjust based on GPU memory
            num_batches = (total_blocks + batch_size - 1) // batch_size
            
            # Create flattened block index list for parallel processing
            block_indices = [(i, j) for i in range(num_out_blocks) for j in range(num_in_blocks)]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_blocks)
                batch_indices = block_indices[start_idx:end_idx]
                batch_size_actual = len(batch_indices)
                
                # Pre-allocate tensors for this batch
                weight_blocks = torch.zeros((batch_size_actual, out_block_size, weight_group), device=device, dtype=orig_weight.dtype)
                original_block_outputs = torch.zeros((batch_size_actual, calib_flattened.shape[0], out_block_size), device=device, dtype=orig_weight.dtype)
                block_inputs = torch.zeros((batch_size_actual, calib_flattened.shape[0], weight_group), device=device, dtype=orig_weight.dtype)
                
                # Extract all blocks and inputs for this batch
                for batch_block_idx, (out_block_idx, in_block_idx) in enumerate(batch_indices):
                    out_start = out_block_idx * out_block_size
                    out_end = out_start + out_block_size
                    in_start = in_block_idx * weight_group
                    in_end = in_start + weight_group
                    
                    # Extract the weight block
                    weight_blocks[batch_block_idx] = orig_weight[out_start:out_end, in_start:in_end]
                    
                    # Extract calibration data for this block's input channels
                    block_inputs[batch_block_idx] = calib_flattened[:, in_start:in_end]
                    
                    # Compute original output contribution from this block
                    original_block_outputs[batch_block_idx] = torch.matmul(block_inputs[batch_block_idx], weight_blocks[batch_block_idx].t())

                
                # Process all mantissa options in parallel
                all_block_mses = torch.zeros((batch_size_actual, 3), device=device)
                
                for m_idx, (M, E, bias, max_float, min_float) in enumerate(zip(mantissa_bits, Es, biases, max_floats, min_floats)):
                    # Create tensor to hold reconstructed blocks for this mantissa config
                    reconstructed_blocks = torch.zeros_like(weight_blocks)
                    # List to store per-block scale factors (each of shape [out_block_size,1])
                    scale_blocks = [None] * batch_size_actual
                    
                    for batch_block_idx in range(batch_size_actual):
                        weight_block = weight_blocks[batch_block_idx]
                        
                        # Compute row-wise max values (per output channel)
                        row_max_vals = weight_block.abs().amax(dim=1, keepdim=True)  # [out_block_size, 1]
                        
                        # Scale factors for each row
                        # max_float_tensor = torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device)
                        if use_approx_dequant:
                            max_float_tensor = torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device)
                            S = apdiv.apply(row_max_vals, max_float_tensor, comp)
                            block_q = fp_scale_ap(weight_block, S, M, bias, max_float, min_float, apdiv, comp)
                        else:
                            S = row_max_vals / max_float
                            block_q = fp_scale(weight_block, S, M, bias, max_float, min_float)
                          
                        best_S = S.clone()  # Ensure best_S is always defined for later use (even when clip is False)
                        
                        if clip:  
                            best_err = torch.full([weight_block.shape[0]], float('inf'), device=weight_block.device, dtype=weight_block.dtype)
                            for i in range(int(0.15 * 100)):
                                p = 1 - i / 100
                                row_max_vals_clip = p * weight_block.abs().amax(dim=1, keepdim=True)
                                if use_approx_dequant:
                                    max_float_tensor = torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device)
                                    S_clip = apdiv.apply(row_max_vals_clip, max_float_tensor, comp)
                                    block_q = fp_scale_ap(weight_block, S_clip, M, bias, max_float, min_float, apdiv, comp)
                                    block_reconstruct = apmul.apply(block_q, S_clip, comp)
                                else:
                                    S_clip = row_max_vals_clip / max_float
                                    block_q = fp_scale(weight_block, S_clip, M, bias, max_float, min_float)
                                    block_reconstruct = block_q * S_clip
                                
                                err = (block_reconstruct - weight_block).abs().pow_(2).sum(dim=1)
                                
                                improve_mask = err < best_err
                                if torch.any(improve_mask):
                                    best_err[improve_mask] = err[improve_mask]
                                    best_S[improve_mask] = S_clip[improve_mask]
                                
                            if use_approx_dequant:
                                block_q = fp_scale_ap(weight_block, best_S, M, bias, max_float, min_float, apdiv, comp)
                            else:
                                block_q = fp_scale(weight_block, best_S, M, bias, max_float, min_float)

                        # store quantized (still unscaled) block; kernel will re-scale using block-specific S
                        reconstructed_blocks[batch_block_idx] = block_q
                        scale_blocks[batch_block_idx] = best_S.clone()
                    
                    # Compute quantized outputs (approximate GEMM) for this mantissa option
                    N_tile = out_block_size  # 64 by default
                    quant_outputs = torch.zeros_like(original_block_outputs)
                    for batch_block_idx in range(batch_size_actual):
                        # ----- convert weight to kernel-friendly representation according to m_idx -----
                        w_block_conv = reconstructed_blocks[batch_block_idx]
                        if m_idx == 0:
                            w_block_conv = w_block_conv
                        elif m_idx == 1:
                            w_block_conv = fp4_class_convert(w_block_conv)
                        else:
                            w_block_conv = fp4_e1m2_class_convert(w_block_conv)
                        
                        # kernel expects: weight^T (K,N), scales (K//g , N), types (N//64, K//g)
                        # Kernel expects scales layout (K//g , N). For one input group (g) this is (1,64)
                        S_block = scale_blocks[batch_block_idx].t().contiguous().half()  # (1,64)
                        type_cur = torch.full((N_tile // 64, 1), m_idx, dtype=torch.uint8, device=device)  # (N//64,1)
                        quant_outputs[batch_block_idx] = AxCoreTypedFunctionFP16.apply(
                            block_inputs[batch_block_idx],                     # [B, K]
                            w_block_conv.t().contiguous(),                    # [K, N]
                            S_block, type_cur, weight_group, use_approx_dequant, None)
                        test_outputs = torch.matmul(block_inputs[batch_block_idx], w_block_conv.t().contiguous())
                        test_outputs = test_outputs * S_block
                        
                        torch.cuda.synchronize()
                        
                    # Calculate MSE for each block with this mantissa option
                    all_block_mses[:, m_idx] = torch.mean((original_block_outputs - quant_outputs) ** 2, dim=(1, 2))
                
                # Find best mantissa configuration for each block in batch
                best_m_indices = torch.argmin(all_block_mses, dim=1)
                
                # Update M_counts based on best choices
                for m in range(3):
                    M_counts[m] += torch.sum(best_m_indices == m).item()
                
                # Update the best weight matrix with the optimal mantissa config for each block
                for batch_block_idx, (out_block_idx, in_block_idx) in enumerate(batch_indices):
                    best_m = best_m_indices[batch_block_idx].item()
                    out_start = out_block_idx * out_block_size
                    out_end = out_start + out_block_size
                    in_start = in_block_idx * weight_group
                    in_end = in_start + weight_group
                    
                    # Recompute the best blocks (we could store them but this saves memory)
                    weight_block = weight_blocks[batch_block_idx]
                    M, E, bias = mantissa_bits[best_m], Es[best_m], biases[best_m]
                    max_float, min_float = max_floats[best_m], min_floats[best_m]
                    
                    # Compute row-wise max values (per output channel)
                    row_max_vals = weight_block.abs().amax(dim=1, keepdim=True)

                    # Scale factors for each row
                    if use_approx_dequant:
                        max_float_tensor = torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device)
                        S = apdiv.apply(row_max_vals, max_float_tensor, comp)
                        block_q = fp_scale_ap(weight_block, S, M, bias, max_float, min_float, apdiv, comp)
                    else:
                        S = row_max_vals / max_float
                        block_q = fp_scale(weight_block, S, M, bias, max_float, min_float)
                        
                    best_S = S.clone()
                    if clip:  
                        best_err = torch.full([weight_block.shape[0]], float('inf'), device=weight_block.device, dtype=weight_block.dtype)
                        for i in range(int(0.15 * 100)):
                            p = 1 - i / 100
                            row_max_vals_clip = p * weight_block.abs().amax(dim=1, keepdim=True)
                            if use_approx_dequant:
                                max_float_tensor = torch.tensor(max_float, dtype=row_max_vals.dtype, device=row_max_vals.device)
                                S_clip = apdiv.apply(row_max_vals_clip, max_float_tensor, comp)
                                block_q = fp_scale_ap(weight_block, S_clip, M, bias, max_float, min_float, apdiv, comp)
                                block_reconstruct = apmul.apply(block_q, S_clip, comp)
                            else:
                                S_clip = row_max_vals_clip / max_float
                                block_q = fp_scale(weight_block, S_clip, M, bias, max_float, min_float)
                                block_reconstruct = block_q * S_clip
                            
                            err = (block_reconstruct - weight_block).abs().pow_(2).sum(dim=1)
                            
                            improve_mask = err < best_err
                            if torch.any(improve_mask):
                                best_err[improve_mask] = err[improve_mask]
                                best_S[improve_mask] = S_clip[improve_mask]
                            
                        if use_approx_dequant:
                            block_q = fp_scale_ap(weight_block, best_S, M, bias, max_float, min_float, apdiv, comp)
                        else:
                            block_q = fp_scale(weight_block, best_S, M, bias, max_float, min_float)
                    
                    # Re-scale back to original range
                    block_r = block_q
                    S = best_S  # best_S is always valid now
                    # Map to kernel-friendly fp4 class values according to chosen mantissa bits
                    if best_m == 0:       # e1m2
                        block_r = block_r
                    elif best_m == 1:     # e2m1
                        block_r = fp4_class_convert(block_r)
                        # w_block_conv = w_block_conv
                    elif best_m == 2:     # e3m0
                        block_r = fp4_e1m2_class_convert(block_r)
                        # block_r = block_r
                    # best_m==2 uses raw values (e3m0)
                     
                    # Update the best weight matrix
                    best_weight[out_start:out_end, in_start:in_end] = block_r

                    # store scales per row (out_block_size) for this in_block (group)
                    scale_buf[out_start:out_end, in_block_idx] = S.squeeze(1).half()

                    # store type id per 64-row tile
                    t_row = out_start // 64
                    types_map[t_row, in_block_idx] = best_m  # 0/1/2
        
        else:
            raise NotImplementedError("weight group size must be > 0")
            
        # Print statistics about chosen formats
        print(f'M bit distribution: M=0: {M_counts[0]}, M=1: {M_counts[1]}, M=2: {M_counts[2]}')
        
        return best_weight, scale_buf, types_map, M_counts
    
    def forward(self, inp: torch.Tensor):
        return AxCoreTypedFunctionFP16.apply(inp, self.weight.T.contiguous(),
                                             self.scale_buf.T.contiguous(),
                                             self.types.T.contiguous(), self.weight_group, self.use_approx_dequant, self.bias)

    def __repr__(self):
        return f'AxCoreDSEWLinear(w_bit={self.w_bit}, weight_group={self.weight_group})'