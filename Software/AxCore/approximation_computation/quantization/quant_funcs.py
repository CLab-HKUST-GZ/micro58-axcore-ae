import torch
from ..baseline_approx.baseline_approx import *
from ..L_Mul.LMul_approx import *
from ..FPMA.FPMA_approx import *
from ..quantization.APAQ.converter import *

@torch.no_grad()
def fp_scale(tensor, S, M, bias, max_float, min_float):
    tensor_unscaled = (tensor / S)
    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    return tensor_q

@torch.no_grad()
def fp_scale_ap(tensor, S, M, bias, max_float, min_float, apdiv, comp):
    tensor_unscaled = apdiv.apply(tensor, S, comp)
    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    return tensor_q

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False, fpq=False, mantissa_bit=-1):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if fpq:
        M = mantissa_bit
        E = n_bits - 1 - M
        bias = 2 ** (E - 1) - 1
        max_float = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        min_float = -max_float
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        S = max_val / max_float
        tensor_unscaled = (tensor / S)
        # W_unscaled = BaselineApproxElementwiseDiv.apply(W, S)
        tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
        tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
        scales = 2.0 ** (tensor_log_scales - M - bias)
        tensor_q = (tensor_unscaled / scales).round()
        tensor_q = tensor_q * scales
        tensor = tensor_q * S
    else:
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            scales = max_val / max_int
            zeros = 0

        if inplace:
            (
                (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
            ).mul_(scales)
        else:
            tensor = (
                torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
            ) * scales

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    # return the quantized tonsor, the scaling factor and the zero point value
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor

@torch.no_grad()
def pseudo_quantize_vcache(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False, fpq=False, mantissa_bit=-1):
    """
    The basic quantization function for weight, activation and KV cache.
    Quantizes vcache along the seq_len dimension with groups of q_group_size tokens.
    vcache dimension: [batch, head, seq_len, head_dim]
    """
    org_tensor_shape = tensor.shape
    
    # Extract dimensions
    batch, head, seq_len, head_dim = org_tensor_shape
    
    # Calculate padding needed if seq_len is not divisible by q_group_size
    remainder = seq_len % q_group_size
    padding_size = 0
    if remainder > 0:
        padding_size = q_group_size - remainder
    
    # Pad the tensor if needed
    if padding_size > 0:
        padding = torch.zeros((batch, head, padding_size, head_dim), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=2)  # Pad along seq_len dimension
        padded_seq_len = seq_len + padding_size
    else:
        padded_seq_len = seq_len
    
    # Reshape to prepare for quantization along seq_len groups
    # New shape: [batch * head * (padded_seq_len // q_group_size), q_group_size * head_dim]
    tensor = tensor.reshape(batch, head, padded_seq_len // q_group_size, q_group_size, head_dim)
    tensor = tensor.permute(0, 1, 2, 4, 3).reshape(batch * head * (padded_seq_len // q_group_size), q_group_size * head_dim)
    
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    
    if fpq:
        M = mantissa_bit
        E = n_bits - 1 - M
        bias = 2 ** (E - 1) - 1
        max_float = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        min_float = -max_float
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        S = max_val / max_float
        tensor_unscaled = (tensor / S)
        # W_unscaled = BaselineApproxElementwiseDiv.apply(W, S)
        tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
        tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
        scales = 2.0 ** (tensor_log_scales - M - bias)
        tensor_q = (tensor_unscaled / scales).round()
        tensor_q = tensor_q * scales
        tensor = tensor_q * S
    else:
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            scales = max_val / max_int
            zeros = 0

        if inplace:
            (
                (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
            ).mul_(scales)
        else:
            tensor = (
                torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
            ) * scales
    
    assert torch.isnan(tensor).sum() == 0
    
    # Reshape back to the original shape with the proper permutation
    tensor = tensor.reshape(batch, head, padded_seq_len // q_group_size, head_dim, q_group_size)
    tensor = tensor.permute(0, 1, 2, 4, 3).reshape(batch, head, padded_seq_len, head_dim)
    
    # Remove padding if it was added
    if padding_size > 0:
        tensor = tensor[:, :, :seq_len, :]
        
    # Return the quantized tensor reshaped to the original shape
    return tensor


@torch.no_grad()
def pseudo_quantize_tensor_dse(tensor, n_bits=8, q_group_size=-1, per_tensor=False):
    """
    The basic quantization function for weight, activation and KV cache.
    Searches for the optimal mantissa bit configuration at block level.
    Implements a hybrid quantization approach:
    - Block quantization with blocks of size [group, group] for data type selection
    - Group quantization along in_channels for scale factors
    Fully GPU-optimized version with vectorized operations while maintaining exact result consistency.
    """
    apmul = DequantizeElementwiseMul
    apdiv = QuantizeElementwiseDiv
    comp = 0
    
    org_tensor_shape = tensor.shape # [out_channels, in_channels]
    
    # Check dimensions for block quantization
    if q_group_size > 0:
        assert org_tensor_shape[0] % q_group_size == 0, f"out_channels ({org_tensor_shape[0]}) must be divisible by q_group_size ({q_group_size})"
        assert org_tensor_shape[1] % q_group_size == 0, f"in_channels ({org_tensor_shape[1]}) must be divisible by q_group_size ({q_group_size})"
        
        # Reshape tensor to blocks: [num_out_blocks, num_in_blocks, q_group_size, q_group_size]
        num_out_blocks = org_tensor_shape[0] // q_group_size
        num_in_blocks = org_tensor_shape[1] // q_group_size
        tensor = tensor.reshape(num_out_blocks, q_group_size, num_in_blocks, q_group_size)
        tensor = tensor.permute(0, 2, 1, 3)  # [num_out_blocks, num_in_blocks, q_group_size, q_group_size]
        
        # Prepare for group-level scaling while maintaining block structure
        tensor_blocks = tensor  # Shape: [num_out_blocks, num_in_blocks, q_group_size, q_group_size]
        
        # Flatten each block to [num_blocks, q_group_size*q_group_size] for block-level data type selection
        tensor = tensor.reshape(-1, q_group_size * q_group_size)
    
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    
    assert tensor.dim() == 2
    # tensor = tensor.to("cpu")
    
    num_blocks = tensor.shape[0]
    device = tensor.device
    
    # Pre-compute parameters for all mantissa bit configurations
    mantissa_bits = [0, 1, 2]  # M = 0, 1, 2
    Es = [n_bits - 1 - m for m in mantissa_bits]
    biases = [2 ** (e - 1) - 1 for e in Es]
    max_floats = [(2 - 2 ** (-m)) * 2 ** (2**e - 1 - b) for m, e, b in zip(mantissa_bits, Es, biases)]
    min_floats = [-max_f for max_f in max_floats]
    
    # Each mantissa bit option (m=0,1,2) will have a tensor of MSEs for all blocks
    all_block_mses = []  # Will be a list of [num_blocks] tensors
    all_reconstructed = []  # Will be a list of [num_blocks, q_group_size*q_group_size] tensors
    
    # For each mantissa bit configuration, compute MSE for all blocks at once
    for m_idx, (M, E, bias, max_float, min_float) in enumerate(zip(mantissa_bits, Es, biases, max_floats, min_floats)):
        # Process with group-level scaling factors while maintaining block structure
        if q_group_size > 0:
            # Create a tensor to hold reconstructed blocks
            reconstructed_blocks = torch.zeros_like(tensor_blocks)
            
            # Process each block
            for b_out in range(num_out_blocks):
                for b_in in range(num_in_blocks):
                    # Extract current block
                    block = tensor_blocks[b_out, b_in]  # Shape: [q_group_size, q_group_size]
                    
                    # Compute group-level max values (along rows/in_channels)
                    # Each out_channel (row) has a vector of q_group_size values, each for one group of in_channels
                    group_max_vals = block.abs().amax(dim=1, keepdim=True)  # [q_group_size, 1]
                    
                    # Compute scale factors for each group
                    S = apdiv.apply(group_max_vals, torch.tensor(max_float, dtype=group_max_vals.dtype, device=group_max_vals.device), comp)
                    # S = group_max_vals / max_float
                    
                    # Unscale and clamp using group-level scale factors
                    # Each row will be divided by its own scale factor
                    unscaled_block = torch.zeros_like(block)
                    for i in range(q_group_size):
                        unscaled_block[i] = apdiv.apply(block[i].unsqueeze(0), S[i], comp)
                        # unscaled_block[i] = block[i] / S[i]
                    
                    # Clamp the unscaled values
                    unscaled_block = torch.clamp(unscaled_block, min_float, max_float)
                    
                    # Apply log-domain quantization
                    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(unscaled_block)) + bias)).detach(), 1.0)
                    scales = 2.0 ** (log_scales - M - bias)
                    block_q = torch.round(unscaled_block / scales) * scales
                    
                    # Scale back using group-level scale factors
                    block_r = torch.zeros_like(block_q)
                    for i in range(q_group_size):
                        block_r[i] = apmul.apply(block_q[i].unsqueeze(0), S[i], comp)
                        # block_r[i] = block_q[i] * S[i]
                    
                    # Store the reconstructed block
                    reconstructed_blocks[b_out, b_in] = block_r
            
            # Reshape reconstructed blocks to match the flattened tensor shape for MSE calculation
            tensor_r = reconstructed_blocks.reshape(-1, q_group_size * q_group_size)
        else:
            # If not using group quantization, fall back to original block-level quantization
            # Compute max values for all blocks at once (batched)
            max_vals = tensor.abs().amax(dim=1, keepdim=True)  # [num_blocks, 1]
            
            # Scale factors for all blocks
            # S = apdiv.apply(max_vals, torch.tensor(max_float, dtype=max_vals.dtype, device=max_vals.device), comp)
            S = max_vals / max_float
            
            # Unscale and clamp all blocks at once
            # tensor_unscaled = apdiv.apply(tensor, S, comp)
            tensor_unscaled = tensor / S
            tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
            
            # Log scales, clamping, and quantization
            tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
            scales = 2.0 ** (tensor_log_scales - M - bias)
            tensor_q = torch.round(tensor_unscaled / scales) * scales  # Round and rescale
            # tensor_r = apmul.apply(tensor_q, S, comp)
            tensor_r = tensor_q * S
        
        # Compute MSE for each block
        block_mses = torch.mean((tensor_r - tensor) ** 2, dim=1)  # [num_blocks]
        all_block_mses.append(block_mses)
        all_reconstructed.append(tensor_r)
    
    # Stack all MSEs: [3, num_blocks]
    stacked_mses = torch.stack(all_block_mses)
    
    # Find best mantissa bit configuration for each block
    best_m_indices = torch.argmin(stacked_mses, dim=0)  # [num_blocks]
    
    # Stack all reconstructed tensors: [3, num_blocks, q_group_size*q_group_size]
    stacked_reconstructed = torch.stack(all_reconstructed)
    
    # Create batch indices for selecting from stacked_reconstructed
    batch_indices = torch.arange(num_blocks, device=device)
    
    # Use advanced indexing to gather the best reconstruction for each block
    # This eliminates the need for a for-loop
    best_tensor_r = stacked_reconstructed[best_m_indices, batch_indices]
    
    # Print statistics about chosen formats
    M_counts = torch.bincount(best_m_indices, minlength=3)
    print(f'M bit distribution: M=0: {M_counts[0]}, M=1: {M_counts[1]}, M=2: {M_counts[2]}')
    
    # Reshape back to original shape
    if q_group_size > 0:
        # Reshape from [num_blocks, q_group_size*q_group_size] to [num_out_blocks, num_in_blocks, q_group_size, q_group_size]
        best_tensor_r = best_tensor_r.reshape(num_out_blocks, num_in_blocks, q_group_size, q_group_size)
        # Permute back to original order
        best_tensor_r = best_tensor_r.permute(0, 2, 1, 3)
        # Reshape to original shape [out_channels, in_channels]
        best_tensor_r = best_tensor_r.reshape(org_tensor_shape)
    else:
        best_tensor_r = best_tensor_r.reshape(org_tensor_shape)
    
    # best_tensor_r = best_tensor_r.to('cuda')
    
    return best_tensor_r


@torch.no_grad()
def pseudo_quantize_tensor_approx(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, fpq=False, mantissa_bit=-1):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    apmul = DequantizeElementwiseMul
    
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    comp = 0
    if fpq:
        M = mantissa_bit
        E = n_bits - 1 - M
        bias = 2 ** (E - 1) - 1
        max_float = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        min_float = -max_float
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        S = max_val / max_float
        tensor_unscaled = (tensor / S)
        # W_unscaled = BaselineApproxElementwiseDiv.apply(W, S)
        tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
        tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
        scales = 2.0 ** (tensor_log_scales - M - bias)
        tensor_q = (tensor_unscaled / scales).round()
        tensor_q = tensor_q * scales
        tensor = apmul.apply(tensor_q, S, comp)
    else:
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            scales = max_val / max_int
            zeros = 0

        W_q = torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
        # print(f'W_q.shape: {W_q.shape}, scales.shape: {scales.shape}')
        tensor = apmul.apply(W_q, scales, comp)

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    # return the quantized tonsor, the scaling factor and the zero point value
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor

@torch.no_grad()
def pseudo_quantize_tensor_approx_dse(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, fpq=False, mantissa_bit=-1):
    """
    The basic quantization function for weight, activation and KV cache.
    """

    apmul = DequantizeElementwiseMul
    apdiv = QuantizeElementwiseDiv
    
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if fpq:
        best_comp = 0
        M = mantissa_bit
        E = n_bits - 1 - M
        bias = 2 ** (E - 1) - 1
        max_float = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        min_float = -max_float
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        # from scipy.stats import shapiro
        # x = tensor.flatten().float().cpu().numpy()
        # print(f"{'Not Gaussian' if shapiro(x)[1]<0.05 else 'Gaussian'} {shapiro(x)}")
        # max_val = max_val.clamp(min=1e-5)
        # S = max_val / max_float
        
        #-------- DSE search --------# (not ready now)
        # max_diff_array = []
        # mse_array = []
        # cos_sim_array = []
        # comp_arrary = []
        
        # for i in range(200):
        #     S = apdiv.apply(max_val, torch.tensor(max_float, dtype=max_val.dtype, device=max_val.device), i-100)
        #     # tensor_unscaled = (tensor / S)
        #     tensor_unscaled = apdiv.apply(tensor, S, i-100)
        #     tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
        #     tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
        #     scales = 2.0 ** (tensor_log_scales - M - bias)
        #     tensor_q = (tensor_unscaled / scales).round()
        #     tensor_q = tensor_q * scales
        #     # tensor_r = tensor_q * S
        #     tensor_r = apmul.apply(tensor_q, S, i-100)
            
        #     max_diff_array.append(torch.max(torch.abs(tensor_r - tensor)))
        #     mse_array.append(torch.mean((tensor_r - tensor) ** 2))
        #     cos_sim = torch.nn.functional.cosine_similarity(tensor_r.flatten(1), tensor.flatten(1), dim=1)
        #     cos_sim_array.append(cos_sim.mean().detach())
        #     comp_arrary.append(i-100)
        # # find the best comp
        # max_diff_tensor = torch.stack(max_diff_array)
        # mse_tensor = torch.stack(mse_array)
        # cos_sim_tensor = torch.stack(cos_sim_array)
        # min_mse = torch.min(mse_tensor)
        # idx = torch.argmin(mse_tensor)
        # # cos_sim_max = torch.max(cos_sim_tensor)
        # # idx = torch.argmax(cos_sim_tensor)
        # best_comp = comp_arrary[idx]
        # print(f"Minimum MSE: {mse_array[idx].item():.4e} at index {idx.item()} (comp={best_comp}), cos_sim={cos_sim_array[idx].item():.4e}")
        #--------- DSE search --------# (end)
        # apply the best comp
        S = apdiv.apply(max_val, torch.tensor(max_float, dtype=max_val.dtype, device=max_val.device), best_comp)
        tensor_unscaled = apdiv.apply(tensor, S, best_comp)
        tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
        tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
        scales = 2.0 ** (tensor_log_scales - M - bias)
        tensor_q = (tensor_unscaled / scales).round()
        tensor_q = tensor_q * scales
        # tensor = tensor_q * S
        tensor = apmul.apply(tensor_q, S, best_comp)
    else:
        best_comp = 0
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            # scales = max_val / max_int
            scales = apdiv.apply(max_val, torch.tensor(max_int, dtype=max_val.dtype, device=max_val.device), best_comp)
            zeros = 0
        
        #-------- DSE search --------# (not ready now)
        # max_diff_array = []
        # mse_array = []
        # cos_sim_array = []
        # comp_arrary = []
        
        # for i in range(200):
        #     tensor_q = torch.clamp(torch.round(apdiv.apply(tensor, scales, i-100)) + zeros, min_int, max_int) - zeros
        #     # tensor_r = tensor_q * S
        #     tensor_r = apmul.apply(tensor_q, scales, i-100)
            
        #     max_diff_array.append(torch.max(torch.abs(tensor_r - tensor)))
        #     mse_array.append(torch.mean((tensor_r - tensor) ** 2))
        #     cos_sim = torch.nn.functional.cosine_similarity(tensor_r.flatten(1), tensor.flatten(1), dim=1)
        #     cos_sim_array.append(cos_sim.mean().detach())
        #     comp_arrary.append(i-100)
        # # find the best comp
        # max_diff_tensor = torch.stack(max_diff_array)
        # mse_tensor = torch.stack(mse_array)
        # cos_sim_tensor = torch.stack(cos_sim_array)
        # min_mse = torch.min(mse_tensor)
        # idx = torch.argmin(mse_tensor)
        # # cos_sim_max = torch.max(cos_sim_tensor)
        # # idx = torch.argmax(cos_sim_tensor)
        # best_comp = comp_arrary[idx]
        # print(f"Minimum MSE: {mse_array[idx].item():.4e} at index {idx.item()} (comp={best_comp}), cos_sim={cos_sim_array[idx].item():.4e}")
        #--------- DSE search --------# (end)
        W_q = torch.clamp(torch.round(apdiv.apply(tensor, scales, best_comp)) + zeros, min_int, max_int) - zeros
        
        # W_q = torch.round(apdiv.apply(tensor, scales, best_comp)) + zeros - zeros
        # print(f'W_q.shape: {W_q.shape}, scales.shape: {scales.shape}')
        tensor = apmul.apply(W_q, scales, best_comp)

    assert torch.isnan(tensor).sum() == 0
    
    tensor = tensor.reshape(org_tensor_shape)

    # return the quantized tonsor, the scaling factor and the zero point value
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor

@torch.no_grad()
def pseudo_quantize_tensor_approx_mxAP(tensor, n_bits=8, q_group_size=-1, per_tensor=False, mantissa_bit=-1):
    """
    The basic quantization function for weight, activation and KV cache.
    """

    apmul = DequantizeElementwiseMul
    apdiv = QuantizeElementwiseDiv
    
    org_tensor_shape = tensor.shape

    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
        org_scale_shape = org_tensor_shape[:-1] + (org_tensor_shape[-1]//q_group_size,)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    
    assert tensor.dim() == 2
    
    best_comp = 0
    M = mantissa_bit
    E = n_bits - 1 - M
    bias = 2 ** (E - 1) - 1
    max_float = (2 - 2 ** (-M)) * 2 ** (
            2**E - 1 - bias
        )
    min_float = -max_float
    max_val = tensor.abs().amax(dim=1, keepdim=True)
    S = max_val / max_float
    # apply the best comp
    # S = apdiv.apply(max_val, torch.tensor(max_float, dtype=max_val.dtype, device=max_val.device), best_comp)
    # tensor_unscaled = apdiv.apply(tensor, S, best_comp)
    tensor_unscaled = tensor / S
    
    #--------- to fp4 --------#
    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    #--------- to fp4 --------# (end)
    
    # tensor = tensor_q * S
    tensor_q = fp4_class_convert(tensor_q)
        
    tensor = tensor_q.reshape(org_tensor_shape)
    # tensor = tensor.reshape(org_tensor_shape)
    S = S.reshape(org_scale_shape)
    
    # return the quantized tonsor, the scaling factor and the zero point value
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor, S

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=True, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)
