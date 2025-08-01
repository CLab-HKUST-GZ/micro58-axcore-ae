
import torch


FP4_CLASS_BIN = {
    0.0   : '0000000000000000',
    0.5   : '0000001000000000', 
    1.0   : '0000010000000000',
    1.5   : '0000011000000000',
    2.0   : '0000100000000000',
    3.0   : '0000101000000000',
    4.0   : '0000110000000000',
    6.0   : '0000111000000000',
    -0.0  : '1000000000000000',
    -0.5  : '1000001000000000',
    -1.0  : '1000010000000000',
    -1.5  : '1000011000000000',
    -2.0  : '1000100000000000',
    -3.0  : '1000101000000000',
    -4.0  : '1000110000000000',
    -6.0  : '1000111000000000',
}

FP4_CLASS = {
    0.0   : 0.0,
    0.5   : 2**-15, 
    1.0   : 2**-14,
    1.5   : 2**-14*1.5,
    2.0   : 2**-13,
    3.0   : 2**-13*1.5,
    4.0   : 2**-12,
    6.0   : 2**-12*1.5,
    -0.0  : -0.0,
    -0.5  : -2**-15,
    -1.0  : -2**-14,
    -1.5  : -2**-14*1.5,
    -2.0  : -2**-13,
    -3.0  : -2**-13*1.5,
    -4.0  : -2**-12,
    -6.0  : -2**-12*1.5,
}

FP4_E1M2_CLASS_BIN = {
    0.0   : '0000000000000000',
    0.5   : '0000000100000000', 
    1.0   : '0000001000000000',
    1.5   : '0000001100000000',
    2.0   : '0000010000000000',
    2.5   : '0000010100000000',
    3.0   : '0000011000000000',
    3.5   : '0000011100000000',
    -0.0  : '1000000000000000',
    -0.5  : '1000000100000000',
    -1.0  : '1000001000000000',
    -1.5  : '1000001100000000',
    -2.0  : '1000010000000000',
    -2.5  : '1000010100000000',
    -3.0  : '1000011000000000',
    -3.5  : '1000011100000000',
}

FP4_E1M2_CLASS = {
    0.0   : 0.0,
    0.5   : 2**-16, 
    1.0   : 2**-15,
    1.5   : 2**-15*1.5,
    2.0   : 2**-14,
    2.5   : 2**-14*1.25,
    3.0   : 2**-14*1.5,
    3.5   : 2**-14*1.75,
    -0.0  : -0.0,
    -0.5  : -2**-16,
    -1.0  : -2**-15,
    -1.5  : -2**-15*1.5,
    -2.0  : -2**-14,
    -2.5  : -2**-14*1.25,
    -3.0  : -2**-14*1.5,
    -3.5  : -2**-14*1.75,
}

FP4_E3M0_CLASS_BIN = {
    0.0   : '0000000000000000',
    0.25   : '0000000100000000', 
    0.5   : '0000001000000000',
    1.0   : '0000001100000000',
    2.0   : '0000010000000000',
    4.0   : '0000010100000000',
    8.0   : '0000011000000000',
    16.0   : '0000011100000000',
    -0.0  : '1000000000000000',
    -0.25  : '1000000100000000',
    -0.5  : '1000001000000000',
    -1.0  : '1000001100000000',
    -2.0  : '1000010000000000',
    -4.0  : '1000010100000000',
    -8.0  : '1000011000000000',
    -16.0  : '1000011100000000',
}

def fp16_to_binary(value):
    """
    将一个 fp16 数值转换为二进制表示。
    
    参数:
        value (float): 输入的浮点数值。
    
    返回:
        str: 该数值的二进制表示。
    """
    # 将输入值转换为 fp16 格式
    fp16_value = torch.tensor(value, dtype=torch.float16)
    
    # 将 fp16 转换为字节表示（小端序）
    byte_representation = fp16_value.numpy().tobytes()
    
    # 将字节转换为整数
    int_representation = int.from_bytes(byte_representation, byteorder='little', signed=False)
    
    # 将整数转换为 16 位二进制字符串
    binary_representation = f"{int_representation:016b}"
    
    return binary_representation

# Function to convert a 16-bit binary string to its float16 value
def binary_string_to_fp16_value(binary_s):
    if not isinstance(binary_s, str) or len(binary_s) != 16:
        raise ValueError("Input must be a 16-bit binary string.")
    if not all(c in '01' for c in binary_s):
        raise ValueError("Input string must contain only '0' or '1'.")

    sign_bit = int(binary_s[0])
    exponent_bits = binary_s[1:6]
    mantissa_bits = binary_s[6:]

    exponent_val = int(exponent_bits, 2)
    mantissa_val = int(mantissa_bits, 2)

    sign = -1.0 if sign_bit == 1 else 1.0
    
    # IEEE 754 half-precision parameters
    exponent_bias = 15
    mantissa_len = 10
    max_exponent_val = (2**5 - 1) # All 1s for 5 bits (31 decimal)

    if exponent_val == max_exponent_val: # Exponent is all 1s
        if mantissa_val == 0:
            return math.inf * sign # Infinity
        else:
            return math.nan # NaN
    elif exponent_val == 0: # Exponent is all 0s
        # Subnormal number
        if mantissa_val == 0:
            return 0.0 * sign # Signed zero
        else:
            # Formula for subnormal: sign * 2^(1 - bias) * (mantissa / 2^mantissa_len)
            return sign * (2**(1 - exponent_bias)) * (mantissa_val / (2**mantissa_len))
    else: # Exponent is not all 0s or all 1s
        # Normalized number
        # Formula for normalized: sign * 2^(exponent - bias) * (1 + mantissa / 2^mantissa_len)
        return sign * (2**(exponent_val - exponent_bias)) * (1 + mantissa_val / (2**mantissa_len))


def fp4_class_convert(tensor: torch.Tensor) -> torch.Tensor:
    # 确保所有张量在同一设备
    device = tensor.device
    
    # 预注册查找表（按设备隔离）
    if not hasattr(fp4_class_convert, 'device_mapping'):
        fp4_class_convert.device_mapping = {}
        
    if device not in fp4_class_convert.device_mapping:
        keys = torch.tensor(list(FP4_CLASS.keys()), dtype=torch.float16, device=device)
        values = torch.tensor(list(FP4_CLASS.values()), dtype=torch.float16, device=device)
        sorted_keys, indices = torch.sort(keys)
        sorted_values = values[indices]
        fp4_class_convert.device_mapping[device] = (sorted_keys, sorted_values)
    
    sorted_keys, sorted_values = fp4_class_convert.device_mapping[device]
    
    # 显式设备转换
    flat_tensor = tensor.to(device).view(-1).to(torch.float16)
    
    # 二分查找
    search_idx = torch.searchsorted(sorted_keys, flat_tensor)
    search_idx.clamp_(max=len(sorted_keys)-1)
    
    # 精确匹配
    mask = torch.isclose(sorted_keys[search_idx], flat_tensor, rtol=1e-3, atol=1e-5)
    
    return torch.where(mask, sorted_values[search_idx], flat_tensor).view(tensor.shape)

def fp4_e1m2_class_convert(tensor: torch.Tensor) -> torch.Tensor:
    # 确保所有张量在同一设备
    device = tensor.device
    
    # 预注册查找表（按设备隔离）
    if not hasattr(fp4_e1m2_class_convert, 'device_mapping'):
        fp4_e1m2_class_convert.device_mapping = {}
        
    if device not in fp4_e1m2_class_convert.device_mapping:
        keys = torch.tensor(list(FP4_E1M2_CLASS.keys()), dtype=torch.float16, device=device)
        values = torch.tensor(list(FP4_E1M2_CLASS.values()), dtype=torch.float16, device=device)
        sorted_keys, indices = torch.sort(keys)
        sorted_values = values[indices]
        fp4_e1m2_class_convert.device_mapping[device] = (sorted_keys, sorted_values)
    
    sorted_keys, sorted_values = fp4_e1m2_class_convert.device_mapping[device]
    
    # 显式设备转换
    flat_tensor = tensor.to(device).view(-1).to(torch.float16)
    
    # 二分查找
    search_idx = torch.searchsorted(sorted_keys, flat_tensor)
    search_idx.clamp_(max=len(sorted_keys)-1)
    
    # 精确匹配
    mask = torch.isclose(sorted_keys[search_idx], flat_tensor, rtol=1e-3, atol=1e-5)
    
    return torch.where(mask, sorted_values[search_idx], flat_tensor).view(tensor.shape)

if __name__ == "__main__":
    # for fp4_num, value in FP4_E1M2_CLASS.items():
    #     binary = fp16_to_binary(value)
    #     print(f"fp4_num: {fp4_num}, binary: {binary}")
    #     true_bin = FP4_E1M2_CLASS_BIN[fp4_num]
    #     if true_bin == binary:
    #         print("PASS")
    #     else:
    #         print(f"FAIL, the true bin is {true_bin}")
    input_tensor = torch.tensor([[0.0, 0.5, 1.0, 1.5],
                                [2.0, 2.5, 3.0, 3.5]], dtype=torch.float16)
    
    output = fp4_e1m2_class_convert(input_tensor)
    output_1 = output[1][3].item()
    print(f"binary : {fp16_to_binary(output_1)}")