from approximation_computation.baseline_approx.baseline_approx import *

if __name__ == "__main__":
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    n_bits = 4
    mantissa_bit = 1
    qint = False
    zero_point = True
    # batch = 200
    batch = 64
    max_diff_array = []
    mse_array = []
    cos_sim_array = []
    kl_div_array = []
    comp_arrary = []
    comp = 0
    for i in range(1000):
        W = torch.randn(batch, 64, dtype=dtype, device=device)
        W_d = torch.randn(batch, 64, dtype=dtype, device=device)
        S = torch.randn(batch, 1, dtype=dtype, device=device)
        Z = torch.ones(batch, 1, dtype=dtype, device=device)
        W = W - W_d
        # W = W * 0.05 - W_d * 0.1
        print(f"W.max: {W.amax()}, W.min: {W.amin()}")
        # W *= 0.05
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
            min_float = -max_float
            max_val = W.abs().amax(dim=1, keepdim=True)
            # S = max_val / max_float
            S = QuantizeElementwiseDiv.apply(max_val, torch.tensor(max_float, dtype=dtype, device=device), comp)
            # W_unscaled = (W / S)
            W_unscaled = QuantizeElementwiseDiv.apply(W, S, comp)
            print(f"W_unscaled.max: {W_unscaled.amax()}, W_unscaled.min: {W_unscaled.amin()}")
            W_unscaled = torch.clamp(W_unscaled, min_float, max_float)
            W_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(W_unscaled)) + bias)).detach(), 1.0)
            scales = 2.0 ** (W_log_scales - M - bias)
            W_q = (W_unscaled / scales).round()
            W_q = W_q * scales
            Z = 0

        # print(f"W: {W}")
        # W_q = QuantizeElementwiseDiv.apply(W, S, comp)
        # W_q = (W / S)
        # if zero_point:
        #     print(f"W_q.max: {W_q.amax()}, W_q.min: {W_q.amin()}, Z.max: {Z.amax()}, Z.min: {Z.amin()}")
        # else:
        #     print(f"W_q.max: {W_q.amax()}, W_q.min: {W_q.amin()}")
        # W_q = torch.round(W_q) + Z
        # print(f"W_q.max: {W_q.amax()}, W_q.min: {W_q.amin()}")
        # W_q = torch.clamp(W_q, 0, 15)
        W_r = DequantizeElementwiseMul.apply(W_q - Z, S, comp)
        # W_r = (W_q - Z) * S
        # print(f"S: {S}")
        # print(f"W_r: {W_r}")
        max_diff_array.append(torch.max(torch.abs(W_r - W)))
        mse_array.append(torch.mean((W_r - W) ** 2))
        cos_sim = torch.nn.functional.cosine_similarity(W_r.flatten(1), W.flatten(1), dim=1)
        cos_sim_array.append(cos_sim.mean().detach())
        # comp_arrary.append(i-100)
    
    # array to tensor
    # mse_tensor = torch.stack(mse_array)
    # # min_mse = torch.min(mse_tensor)
    # # min_idx = torch.argmin(mse_tensor)
    # cos_sim_tensor = torch.stack(cos_sim_array)
    # cos_sim = torch.max(cos_sim_tensor)
    # min_idx = torch.argmax(cos_sim_tensor)
    # best_comp = comp_arrary[min_idx]
    
    max_diff = torch.mean(torch.stack(max_diff_array))
    mse = torch.mean(torch.stack(mse_array))
    cos_sim = torch.mean(torch.stack(cos_sim_array)).item()
    print(f"Maximum difference between W_r and W: {max_diff}")
    print(f"Mean square error between W_r and W: {mse}")
    print(f"Cosine similarity between W and W_r: {cos_sim}")
    
    print(2**(17))