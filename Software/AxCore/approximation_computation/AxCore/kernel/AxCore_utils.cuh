#ifndef AXCORE_UTILS_CUH
#define AXCORE_UTILS_CUH

#include <cuda_fp16.h>
#include <stdint.h>

typedef union {
    half fp16;
    uint16_t u16;
} fp16_union;

__device__ __forceinline__ uint16_t float16_to_uint16(half a) {
    fp16_union u;
    u.fp16 = a;
    return u.u16;
}

__device__ __forceinline__ half uint16_to_float16(uint16_t a) {
    fp16_union u;
    u.u16 = a;
    return u.fp16;
}

#endif // AXCORE_UTILS_CUH
