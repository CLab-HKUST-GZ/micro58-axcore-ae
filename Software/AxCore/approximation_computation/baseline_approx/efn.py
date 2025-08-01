import numpy as np
import struct

def float16_to_uint16(x):
    """Convert float16 to uint16 representation"""
    return np.frombuffer(struct.pack('<e', x), dtype=np.uint16)[0]

def uint16_to_float16(x):
    """Convert uint16 representation back to float16"""
    return struct.unpack('<e', np.array(x, dtype=np.uint16).tobytes())[0]

def baseline_approx_add_without_subnormal(a, b):
    # Convert inputs to uint16 bit representation
    a_bits = float16_to_uint16(a)
    b_bits = float16_to_uint16(b)
    
    # Extract sign bits
    s1 = a_bits & 0x8000
    s2 = b_bits & 0x8000
    diff_sign = s1 ^ s2
    
    # Extract exponents
    e1 = (a_bits >> 10) & 0x001F
    e2 = (b_bits >> 10) & 0x001F
    
    # Extract mantissas (add implicit leading 1)
    m1 = ((a_bits & 0x03FF) | 0x0400)
    m2 = ((b_bits & 0x03FF) | 0x0400)
    
    # Determine max/min based on exponent and mantissa
    mask = e2 > e1
    e_mask = e2 == e1
    m_mask = m2 > m1
    
    e_max = e2 if mask else e1
    e_min = e1 if mask else e2
    s_max = s2 if (not e_mask and mask) else (s1 if not e_mask else (s2 if m_mask else s1))
    m_max = m2 if (not e_mask and mask) else (m1 if not e_mask else (m2 if m_mask else m1))
    m_min = m1 if (not e_mask and mask) else (m2 if not e_mask else (m1 if m_mask else m2))
    
    # Left shift mantissas
    m_max <<= 2  # 13 bits
    m_min <<= 2
    
    # Align mantissas
    delta_e = e_max - e_min
    aligned_m_min = m_min >> delta_e
    
    # Compute sum of mantissas
    sum_m = m_max - aligned_m_min if diff_sign else m_max + aligned_m_min
    
    result_sign = s_max if diff_sign else s1
    
    if diff_sign:
        if sum_m == 0:
            return uint16_to_float16(0)
        
        # Count leading zeros (simulate __clz)
        leading_zeros = 32 - int(sum_m).bit_length() - 16
        norm_shift = leading_zeros - 3
        
        if norm_shift > 0:
            norm_shift = min(norm_shift, e_max)
            sum_m <<= norm_shift
            e_max -= norm_shift
        
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x0FFC) >> 2))
    else:
        # Rounding logic
        # sum_m = sum_m + 1 if (sum_m & 0x0003) else sum_m
        # uint16_t A = sum_m & 0x000C;
        # uint16_t B = sum_m & 0x0003;
        # A = (A == 0x000C) ? A : (B & 0x0002) ? (A + 4) : A;
        # sum_m = (sum_m & 0xFFF0) | A;
        A = sum_m & 0x000C
        B = sum_m & 0x0003
        A = A if (A == 0x000C) else (A + 4 if (B & 0x0002) else A)
        sum_m = (sum_m & 0xFFF0) | A
        if sum_m > 0x1FFF:
            sum_m >>= 1
            e_max += 1
        
        if e_max >= 31:
            return uint16_to_float16(result_sign | (0x001E << 10) | 0x03FF)
        
        return uint16_to_float16(result_sign | (e_max << 10) | ((sum_m & 0x0FFC) >> 2))

# Example usage
a = np.float16(34560)
b = np.float16(-34563.1)
result = baseline_approx_add_without_subnormal(a, b)
print(f"{a} + {b} = {result}")