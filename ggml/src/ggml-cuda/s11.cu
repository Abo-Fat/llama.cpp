#include "s11.cuh"
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdlib>
#include <cstring>

static __device__ unsigned long long s11_completed_assignments;

static __device__ float s11_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

static __device__ float s11_scale(const unsigned char * block, int type) {
    const int bytes = type <= GGML_TYPE_S11_W2_BF16 ? 32 : type == GGML_TYPE_S11_W4 ? 64 : 128;
    return *reinterpret_cast<const float *>(block + bytes);
}

static __device__ int s11_code(const unsigned char * block, int k, int type) {
    if (type <= GGML_TYPE_S11_W2_BF16) {
        return ((block[k / 4] >> (2 * (k % 4))) & 3) - 1;
    }
    if (type == GGML_TYPE_S11_W4) {
        return (((block[k / 2] >> (4 * (k % 2))) & 15) ^ 8) - 8;
    }
    return reinterpret_cast<const signed char *>(block)[k];
}

static __device__ __forceinline__ int s11_unpack4_w2_byte(uint32_t b) {
    const uint32_t u = b * 0x00001001u;
    const uint32_t u_shift = u << 6;
    uint32_t raw4;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;\n"
                 : "=r"(raw4)
                 : "r"(u_shift), "n"(0x03000300u), "r"(u & 0x00030003u));
    return int((raw4 + 0x7f7f7f7fu) ^ 0x80808080u);
}

static __device__ __forceinline__ int s11_unpack4_w2(const unsigned char * block, int lane) {
    return s11_unpack4_w2_byte(block[lane]);
}

static __device__ __forceinline__ int s11_unpack4_w4_u16(uint32_t w16) {
    const uint32_t u = (w16 & 0x00ffu) | ((w16 & 0xff00u) << 8);
    const uint32_t u_shift = u << 4;
    uint32_t raw4;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;\n"
                 : "=r"(raw4)
                 : "r"(u_shift), "n"(0x0f000f00u), "r"(u & 0x000f000fu));
    return int(((raw4 ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u);
}

static __device__ __forceinline__ int s11_unpack4_w4(const unsigned char * block, int lane) {
    return s11_unpack4_w4_u16(*reinterpret_cast<const uint16_t *>(block + lane * 2));
}

static __global__ void s11_quantize(
        const char * input, signed char * q, float * scales, float * fake,
        int k, int rows_per_token, int tokens, size_t slot_stride, size_t token_stride) {
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    const float * x = reinterpret_cast<const float *>(
        input + (row % rows_per_token) * slot_stride + (row / rows_per_token) * token_stride);
    float values[4];
    float maximum = 1e-5f;
    for (int i = 0; i < 4; ++i) {
        values[i] = s11_bf16(x[group * 128 + lane + 32 * i]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    for (int offset = 16; offset; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float inverse = __fmul_rn(__fdiv_rn(1.0f, maximum), 127.0f);
    if (lane == 0) {
        scales[row * (k / 128) + group] = __fdiv_rn(1.0f, inverse);
    }
    for (int i = 0; i < 4; ++i) {
        const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(values[i], inverse))));
        const int index = row * k + group * 128 + lane + 32 * i;
        q[index] = quant;
        if (fake) {
            fake[index] = s11_bf16(__fdiv_rn(float(quant), inverse));
        }
    }
}

static __global__ void s11_projection(
        const char * weights, const char * input, const signed char * q,
        const float * activation_scales, const float * fake, const char * ids,
        char * output, int type, int k, int n, int slots, int tokens, int input_slots,
        size_t weight_row, size_t weight_expert, size_t input_slot, size_t input_token,
        size_t ids_slot, size_t ids_token, size_t output_slot, size_t output_token) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int assignment = blockIdx.y;
    const int slot = assignment % slots;
    const int token = assignment / slots;
    if (row >= n || token >= tokens) {
        return;
    }
    const int expert = ids ? *reinterpret_cast<const int *>(ids + slot * ids_slot + token * ids_token) : 0;
    if (expert < 0) {
        if (!lane) {
            *reinterpret_cast<float *>(output + row * sizeof(float) + slot * output_slot + token * output_token) = 0;
        }
        return;
    }
    const int source_row = token * input_slots + (input_slots == 1 ? 0 : slot);
    if (row == 0 && lane == 0 && type == GGML_TYPE_S11_W2 && k == 512 && n == 2048) {
        atomicAdd(&s11_completed_assignments, 1ULL);
    }
    const int block_bytes = type <= GGML_TYPE_S11_W2_BF16 ? 36 : type == GGML_TYPE_S11_W4 ? 68 : 132;
    const char * weight = weights + expert * weight_expert + row * weight_row;
    float sum = 0;
    for (int group = 0; group < k / 128; ++group) {
        const unsigned char * block = reinterpret_cast<const unsigned char *>(weight + group * block_bytes);
        const float scale = s11_scale(block, type);
        if (type == GGML_TYPE_S11_W2) {
            int dot = 0;
            for (int i = 0; i < 4; ++i) {
                const int index = lane + 32 * i;
                dot += int(q[source_row * k + group * 128 + index]) * s11_code(block, index, type);
            }
            for (int offset = 16; offset; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            const float partial = __fmul_rn(__fmul_rn(float(dot), activation_scales[source_row * (k / 128) + group]), scale);
            sum = __fadd_rn(sum, partial);
        } else {
            float dot = 0;
            for (int i = 0; i < 4; ++i) {
                const int index = lane + 32 * i;
                const float a = type == GGML_TYPE_S11_I8
                    ? s11_bf16(reinterpret_cast<const float *>(input + token * input_token + (input_slots == 1 ? 0 : slot) * input_slot)[group * 128 + index])
                    : fake[source_row * k + group * 128 + index];
                const float restored = type == GGML_TYPE_S11_I8
                    ? __fdiv_rn(float(s11_code(block, index, type)), scale)
                    : __fmul_rn(float(s11_code(block, index, type)), scale);
                dot = __fadd_rn(dot, __fmul_rn(a, s11_bf16(restored)));
            }
            for (int offset = 16; offset; offset /= 2) {
                dot = __fadd_rn(dot, __shfl_down_sync(0xffffffff, dot, offset));
            }
            if (!lane) {
                sum = __fadd_rn(sum, dot);
            }
        }
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + row * sizeof(float) + slot * output_slot + token * output_token) = s11_bf16(sum);
    }
}

template<int TYPE, int COLS>
static __global__ void s11_projection_packed(
        const char * weights, const signed char * q, const float * activation_scales,
        const float * fake, const char * input, const char * ids, char * output,
        int k, int n, int slots, int tokens, int input_slots,
        size_t weight_row, size_t weight_expert, size_t input_slot, size_t input_token,
        size_t ids_slot, size_t ids_token, size_t output_slot, size_t output_token) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= n) {
        return;
    }
    const int first = blockIdx.y * COLS;
    const int expert = ids ? *reinterpret_cast<const int *>(ids + (first % slots) * ids_slot + (first / slots) * ids_token) : 0;
    if (expert < 0) {
        if (!lane) {
            *reinterpret_cast<float *>(output + row * sizeof(float) + (first % slots) * output_slot + (first / slots) * output_token) = 0;
        }
        return;
    }
    if (row == 0 && lane == 0 && TYPE == GGML_TYPE_S11_W2 && k == 512 && n == 2048) {
        atomicAdd(&s11_completed_assignments, 1ULL);
    }
    constexpr int bytes = TYPE <= GGML_TYPE_S11_W2_BF16 ? 36 : TYPE == GGML_TYPE_S11_W4 ? 68 : 132;
    const char * weight = weights + expert * weight_expert + row * weight_row;
    float sums[COLS] = {};
    if constexpr (TYPE <= GGML_TYPE_S11_W2_BF16) {
        if ((k % 512) == 0) {
            const int quarter = lane / 8;
            const int sub = lane % 8;
            #pragma unroll 4
            for (int g4 = 0; g4 < k / 512; ++g4) {
                const int group = g4 * 4 + quarter;
                const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
                const float scale = s11_scale(block, TYPE);
                const uint32_t w32 = *reinterpret_cast<const uint32_t *>(block + sub * 4);
                const int p0 = s11_unpack4_w2_byte(w32 & 0xffu);
                const int p1 = s11_unpack4_w2_byte((w32 >> 8) & 0xffu);
                const int p2 = s11_unpack4_w2_byte((w32 >> 16) & 0xffu);
                const int p3 = s11_unpack4_w2_byte(w32 >> 24);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int assignment = first + c;
                    const int token = (TYPE == GGML_TYPE_S11_W2) ? (assignment / slots) : assignment;
                    const int slot = (TYPE == GGML_TYPE_S11_W2) ? (assignment % slots) : 0;
                    if (token < tokens) {
                        const int source = (TYPE == GGML_TYPE_S11_W2) ? (token * input_slots + (input_slots == 1 ? 0 : slot)) : token;
                        const int4 act4 = *reinterpret_cast<const int4 *>(q + size_t(source) * k + group * 128 + sub * 16);
                        int dot = __dp4a(p0, act4.x, 0);
                        dot = __dp4a(p1, act4.y, dot);
                        dot = __dp4a(p2, act4.z, dot);
                        dot = __dp4a(p3, act4.w, dot);
                        const float act_s = activation_scales[size_t(source) * (k / 128) + group];
                        sums[c] = __fadd_rn(sums[c], __fmul_rn(float(dot), __fmul_rn(act_s, scale)));
                    }
                }
            }
        } else {
            for (int group = 0; group < k / 128; ++group) {
                const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
                const float scale = s11_scale(block, TYPE);
                const int packed = s11_unpack4_w2(block, lane);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int assignment = first + c;
                    const int token = (TYPE == GGML_TYPE_S11_W2) ? (assignment / slots) : assignment;
                    const int slot = (TYPE == GGML_TYPE_S11_W2) ? (assignment % slots) : 0;
                    if (token < tokens) {
                        const int source = (TYPE == GGML_TYPE_S11_W2) ? (token * input_slots + (input_slots == 1 ? 0 : slot)) : token;
                        const int activation = *reinterpret_cast<const int *>(q + size_t(source) * k + group * 128 + lane * 4);
                        const int dot = __dp4a(packed, activation, 0);
                        const float act_s = activation_scales[size_t(source) * (k / 128) + group];
                        sums[c] = __fadd_rn(sums[c], __fmul_rn(float(dot), __fmul_rn(act_s, scale)));
                    }
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_S11_W4) {
        if ((k % 512) == 0) {
            const int quarter = lane / 8;
            const int sub = lane % 8;
            #pragma unroll 4
            for (int g4 = 0; g4 < k / 512; ++g4) {
                const int group = g4 * 4 + quarter;
                const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
                const float scale = s11_scale(block, TYPE);
                const uint32_t w_lo = *reinterpret_cast<const uint32_t *>(block + sub * 8);
                const uint32_t w_hi = *reinterpret_cast<const uint32_t *>(block + sub * 8 + 4);
                const int p0 = s11_unpack4_w4_u16(w_lo & 0xffffu);
                const int p1 = s11_unpack4_w4_u16(w_lo >> 16);
                const int p2 = s11_unpack4_w4_u16(w_hi & 0xffffu);
                const int p3 = s11_unpack4_w4_u16(w_hi >> 16);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int token = first + c;
                    if (token < tokens) {
                        const int4 act4 = *reinterpret_cast<const int4 *>(q + size_t(token) * k + group * 128 + sub * 16);
                        int dot = __dp4a(p0, act4.x, 0);
                        dot = __dp4a(p1, act4.y, dot);
                        dot = __dp4a(p2, act4.z, dot);
                        dot = __dp4a(p3, act4.w, dot);
                        const float act_s = activation_scales[size_t(token) * (k / 128) + group];
                        sums[c] = __fadd_rn(sums[c], __fmul_rn(float(dot), __fmul_rn(act_s, scale)));
                    }
                }
            }
        } else if ((k % 256) == 0) {
            const int half_w = lane / 16;
            const int sub = lane % 16;
            #pragma unroll 4
            for (int g2 = 0; g2 < k / 256; ++g2) {
                const int group = g2 * 2 + half_w;
                const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
                const float scale = s11_scale(block, TYPE);
                const uint32_t w32 = *reinterpret_cast<const uint32_t *>(block + sub * 4);
                const int p0 = s11_unpack4_w4_u16(w32 & 0xffffu);
                const int p1 = s11_unpack4_w4_u16(w32 >> 16);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int token = first + c;
                    if (token < tokens) {
                        const int2 act2 = *reinterpret_cast<const int2 *>(q + size_t(token) * k + group * 128 + sub * 8);
                        int dot = __dp4a(p0, act2.x, 0);
                        dot = __dp4a(p1, act2.y, dot);
                        const float act_s = activation_scales[size_t(token) * (k / 128) + group];
                        sums[c] = __fadd_rn(sums[c], __fmul_rn(float(dot), __fmul_rn(act_s, scale)));
                    }
                }
            }
        } else {
            for (int group = 0; group < k / 128; ++group) {
                const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
                const float scale = s11_scale(block, TYPE);
                const int packed = s11_unpack4_w4(block, lane);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int token = first + c;
                    if (token < tokens) {
                        const int activation = *reinterpret_cast<const int *>(q + size_t(token) * k + group * 128 + lane * 4);
                        const int dot = __dp4a(packed, activation, 0);
                        const float act_s = activation_scales[size_t(token) * (k / 128) + group];
                        sums[c] = __fadd_rn(sums[c], __fmul_rn(float(dot), __fmul_rn(act_s, scale)));
                    }
                }
            }
        }
    } else {
        for (int group = 0; group < k / 128; ++group) {
            const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
            const float scale = s11_scale(block, TYPE);
            float dots[COLS] = {};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int index = lane + 32 * i;
                const float restored = __fdiv_rn(float(s11_code(block, index, TYPE)), scale);
                const float w = s11_bf16(restored);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int assignment = first + c, token = assignment / slots, slot = assignment % slots;
                    if (token < tokens) {
                        const float a = s11_bf16(reinterpret_cast<const float *>(input + token * input_token + (input_slots == 1 ? 0 : slot) * input_slot)[group * 128 + index]);
                        dots[c] = __fadd_rn(dots[c], __fmul_rn(a, w));
                    }
                }
            }
            for (int offset = 16; offset; offset /= 2) {
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    dots[c] = __fadd_rn(dots[c], __shfl_down_sync(0xffffffff, dots[c], offset));
                }
            }
            if (!lane) {
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    sums[c] = __fadd_rn(sums[c], dots[c]);
                }
            }
        }
    }
    if constexpr (TYPE <= GGML_TYPE_S11_W4) {
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                sums[c] = __fadd_rn(sums[c], __shfl_down_sync(0xffffffff, sums[c], offset));
            }
        }
    }
    if (!lane) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int assignment = first + c, token = assignment / slots;
            if (token < tokens) {
                *reinterpret_cast<float *>(output + row * sizeof(float) + (assignment % slots) * output_slot + token * output_token) = s11_bf16(sums[c]);
            }
        }
    }
}

template<int TYPE>
static __global__ void __launch_bounds__(256) s11_projection_packed_m1(
        const char * __restrict__ weights, const signed char * __restrict__ q,
        const float * __restrict__ activation_scales, char * __restrict__ output,
        int k, int n, size_t weight_row) {
    __shared__ __align__(16) signed char q_sh[4096];
    __shared__ float scales_sh[32];
    const int t = threadIdx.x;
    const int num_groups = k / 128;
    if (t < num_groups) {
        scales_sh[t] = activation_scales[t];
    }
    const int4 * q4_in = reinterpret_cast<const int4 *>(q);
    int4 * q4_sh = reinterpret_cast<int4 *>(q_sh);
    for (int idx = t; idx < k / 16; idx += 256) {
        q4_sh[idx] = q4_in[idx];
    }
    __syncthreads();

    const int lane = t % 32;
    const int row = blockIdx.x * 8 + t / 32;
    if (row >= n) {
        return;
    }
    constexpr int bytes = TYPE <= GGML_TYPE_S11_W2_BF16 ? 36 : 68;
    const char * weight = weights + size_t(row) * weight_row;
    float sum = 0.0f;
    const int quarter = lane / 8;
    const int sub = lane % 8;
    const int num_g4 = k / 512;
    if constexpr (TYPE <= GGML_TYPE_S11_W2_BF16) {
        for (int g4 = 0; g4 < num_g4; ++g4) {
            const int group = g4 * 4 + quarter;
            const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
            const float scale = s11_scale(block, TYPE);
            const uint32_t w32 = *reinterpret_cast<const uint32_t *>(block + sub * 4);
            const int p0 = s11_unpack4_w2_byte(w32 & 0xffu);
            const int p1 = s11_unpack4_w2_byte((w32 >> 8) & 0xffu);
            const int p2 = s11_unpack4_w2_byte((w32 >> 16) & 0xffu);
            const int p3 = s11_unpack4_w2_byte(w32 >> 24);
            const int4 act4 = *reinterpret_cast<const int4 *>(q_sh + group * 128 + sub * 16);
            int dot = __dp4a(p0, act4.x, 0);
            dot = __dp4a(p1, act4.y, dot);
            dot = __dp4a(p2, act4.z, dot);
            dot = __dp4a(p3, act4.w, dot);
            sum = __fadd_rn(sum, __fmul_rn(float(dot), __fmul_rn(scales_sh[group], scale)));
        }
    } else {
        for (int g4 = 0; g4 < num_g4; ++g4) {
            const int group = g4 * 4 + quarter;
            const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
            const float scale = s11_scale(block, TYPE);
            const uint32_t w_lo = *reinterpret_cast<const uint32_t *>(block + sub * 8);
            const uint32_t w_hi = *reinterpret_cast<const uint32_t *>(block + sub * 8 + 4);
            const int p0 = s11_unpack4_w4_u16(w_lo & 0xffffu);
            const int p1 = s11_unpack4_w4_u16(w_lo >> 16);
            const int p2 = s11_unpack4_w4_u16(w_hi & 0xffffu);
            const int p3 = s11_unpack4_w4_u16(w_hi >> 16);
            const int4 act4 = *reinterpret_cast<const int4 *>(q_sh + group * 128 + sub * 16);
            int dot = __dp4a(p0, act4.x, 0);
            dot = __dp4a(p1, act4.y, dot);
            dot = __dp4a(p2, act4.z, dot);
            dot = __dp4a(p3, act4.w, dot);
            sum = __fadd_rn(sum, __fmul_rn(float(dot), __fmul_rn(scales_sh[group], scale)));
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + size_t(row) * sizeof(float)) = s11_bf16(sum);
    }
}

static __global__ void __launch_bounds__(256) s11_w4_multi_projection_packed_m1(
        const char * __restrict__ w0, const char * __restrict__ w1, const char * __restrict__ w2,
        char * __restrict__ out0, char * __restrict__ out1, char * __restrict__ out2,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        int n0, int n1, int n2, size_t wr0, size_t wr1, size_t wr2) {
    __shared__ __align__(16) signed char q_sh[2048];
    __shared__ float scales_sh[16];
    const int t = threadIdx.x;
    if (t < 16) {
        scales_sh[t] = activation_scales[t];
    }
    const int4 * q4_in = reinterpret_cast<const int4 *>(q);
    int4 * q4_sh = reinterpret_cast<int4 *>(q_sh);
    if (t < 128) {
        q4_sh[t] = q4_in[t];
    }
    __syncthreads();

    const int lane = t % 32;
    const int b = blockIdx.x;
    const int b0 = n0 / 8;
    const int b1 = n1 / 8;
    const char * weights = nullptr;
    char * output = nullptr;
    int row = 0;
    size_t weight_row = 0;
    if (b < b0) {
        weights = w0;
        output = out0;
        row = b * 8 + t / 32;
        weight_row = wr0;
    } else if (b < b0 + b1) {
        weights = w1;
        output = out1;
        row = (b - b0) * 8 + t / 32;
        weight_row = wr1;
    } else {
        weights = w2;
        output = out2;
        row = (b - b0 - b1) * 8 + t / 32;
        weight_row = wr2;
    }
    const char * weight = weights + size_t(row) * weight_row;
    float sum = 0.0f;
    const int quarter = lane / 8;
    const int sub = lane % 8;
    #pragma unroll 4
    for (int g4 = 0; g4 < 4; ++g4) {
        const int group = g4 * 4 + quarter;
        const auto * block = reinterpret_cast<const unsigned char *>(weight + group * 68);
        const float scale = s11_scale(block, GGML_TYPE_S11_W4);
        const uint32_t w_lo = *reinterpret_cast<const uint32_t *>(block + sub * 8);
        const uint32_t w_hi = *reinterpret_cast<const uint32_t *>(block + sub * 8 + 4);
        const int p0 = s11_unpack4_w4_u16(w_lo & 0xffffu);
        const int p1 = s11_unpack4_w4_u16(w_lo >> 16);
        const int p2 = s11_unpack4_w4_u16(w_hi & 0xffffu);
        const int p3 = s11_unpack4_w4_u16(w_hi >> 16);
        const int4 act4 = *reinterpret_cast<const int4 *>(q_sh + group * 128 + sub * 16);
        int dot = __dp4a(p0, act4.x, 0);
        dot = __dp4a(p1, act4.y, dot);
        dot = __dp4a(p2, act4.z, dot);
        dot = __dp4a(p3, act4.w, dot);
        sum = __fadd_rn(sum, __fmul_rn(float(dot), __fmul_rn(scales_sh[group], scale)));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + size_t(row) * sizeof(float)) = s11_bf16(sum);
    }
}

template <int COLS>
static __global__ void __launch_bounds__(128) s11_w4_multi_projection_packed(
        const char * __restrict__ w0, const char * __restrict__ w1, const char * __restrict__ w2,
        char * __restrict__ out0, char * __restrict__ out1, char * __restrict__ out2,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        int n0, int n1, int n2, int tokens,
        size_t wr0, size_t wr1, size_t wr2,
        size_t ot0, size_t ot1, size_t ot2) {
    const int lane = threadIdx.x % 32;
    const int b = blockIdx.x;
    const int b0 = n0 / 4;
    const int b1 = n1 / 4;
    const char * weights = nullptr;
    char * output = nullptr;
    int row = 0;
    size_t weight_row = 0;
    size_t output_token = 0;
    if (b < b0) {
        weights = w0; output = out0; row = b * 4 + threadIdx.x / 32; weight_row = wr0; output_token = ot0;
    } else if (b < b0 + b1) {
        weights = w1; output = out1; row = (b - b0) * 4 + threadIdx.x / 32; weight_row = wr1; output_token = ot1;
    } else {
        weights = w2; output = out2; row = (b - b0 - b1) * 4 + threadIdx.x / 32; weight_row = wr2; output_token = ot2;
    }
    const int first = blockIdx.y * COLS;
    const char * weight = weights + size_t(row) * weight_row;
    float sums[COLS] = {};
    const int quarter = lane / 8;
    const int sub = lane % 8;
    #pragma unroll 4
    for (int g4 = 0; g4 < 4; ++g4) {
        const int group = g4 * 4 + quarter;
        const auto * block = reinterpret_cast<const unsigned char *>(weight + group * 68);
        const float scale = s11_scale(block, GGML_TYPE_S11_W4);
        const uint32_t w_lo = *reinterpret_cast<const uint32_t *>(block + sub * 8);
        const uint32_t w_hi = *reinterpret_cast<const uint32_t *>(block + sub * 8 + 4);
        const int p0 = s11_unpack4_w4_u16(w_lo & 0xffffu);
        const int p1 = s11_unpack4_w4_u16(w_lo >> 16);
        const int p2 = s11_unpack4_w4_u16(w_hi & 0xffffu);
        const int p3 = s11_unpack4_w4_u16(w_hi >> 16);
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int token = first + c;
            if (token < tokens) {
                const int4 act4 = *reinterpret_cast<const int4 *>(q + size_t(token) * 2048 + group * 128 + sub * 16);
                int dot = __dp4a(p0, act4.x, 0);
                dot = __dp4a(p1, act4.y, dot);
                dot = __dp4a(p2, act4.z, dot);
                dot = __dp4a(p3, act4.w, dot);
                const float act_s = activation_scales[size_t(token) * 16 + group];
                sums[c] = __fadd_rn(sums[c], __fmul_rn(float(dot), __fmul_rn(act_s, scale)));
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            sums[c] = __fadd_rn(sums[c], __shfl_down_sync(0xffffffff, sums[c], offset));
        }
    }
    if (!lane) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int token = first + c;
            if (token < tokens) {
                *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(token) * output_token) = s11_bf16(sums[c]);
            }
        }
    }
}

template <int M_TILES = 1>
static __global__ void __launch_bounds__(128) s11_w4_multi_wmma_k128_kernel(
        const char * __restrict__ w0, const char * __restrict__ w1, const char * __restrict__ w2,
        char * __restrict__ out0, char * __restrict__ out1, char * __restrict__ out2,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        int n0, int n1, int n2, int tokens,
        size_t wr0, size_t wr1, size_t wr2,
        size_t ot0, size_t ot1, size_t ot2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int bytes = 68;
    constexpr int kStride = 144;
    const int t = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int first_token = blockIdx.y * (16 * M_TILES);
    if (first_token >= tokens) {
        return;
    }
    const int b = blockIdx.x;
    const int b0 = n0 / 64;
    const int b1 = n1 / 64;
    const char * weights = nullptr;
    char * output = nullptr;
    int first_row = 0;
    int n = 0;
    size_t weight_row = 0;
    size_t output_token = 0;
    if (b < b0) {
        weights = w0; output = out0; first_row = b * 64; n = n0; weight_row = wr0; output_token = ot0;
    } else if (b < b0 + b1) {
        weights = w1; output = out1; first_row = (b - b0) * 64; n = n1; weight_row = wr1; output_token = ot1;
    } else {
        weights = w2; output = out2; first_row = (b - b0 - b1) * 64; n = n2; weight_row = wr2; output_token = ot2;
    }

    __shared__ __align__(32) signed char a_sh[16 * M_TILES * kStride];
    __shared__ __align__(32) signed char b_sh[64 * kStride];
    __shared__ __align__(32) int dots_sh[4][16 * 16];
    __shared__ float a_scales_sh[16 * M_TILES];
    __shared__ float w_scales_sh[64];

    float sums[M_TILES][8] = {};

    for (int group = 0; group < 16; ++group) {
        if (t < 16 * M_TILES) {
            const int tok = first_token + t;
            a_scales_sh[t] = (tok < tokens) ? activation_scales[size_t(tok) * 16 + group] : 0.0f;
        }
        if (t < 64) {
            const int r = first_row + t;
            w_scales_sh[t] = (r < n)
                ? s11_scale(reinterpret_cast<const unsigned char *>(weights + size_t(r) * weight_row + group * bytes), GGML_TYPE_S11_W4)
                : 0.0f;
        }
        #pragma unroll
        for (int idx = t; idx < 16 * M_TILES * 32; idx += 128) {
            const int tok_local = idx / 32;
            const int col4 = idx % 32;
            const int tok = first_token + tok_local;
            *reinterpret_cast<int *>(&a_sh[tok_local * kStride + col4 * 4]) = (tok < tokens)
                ? *reinterpret_cast<const int *>(q + size_t(tok) * 2048 + group * 128 + col4 * 4)
                : 0;
        }
        #pragma unroll
        for (int idx = t; idx < 64 * 32; idx += 128) {
            const int r_local = idx / 32;
            const int col4 = idx % 32;
            const int r = first_row + r_local;
            int packed = 0;
            if (r < n) {
                const auto * block = reinterpret_cast<const unsigned char *>(weights + size_t(r) * weight_row + group * bytes);
                packed = s11_unpack4_w4(block, col4);
            }
            *reinterpret_cast<int *>(&b_sh[r_local * kStride + col4 * 4]) = packed;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[M_TILES];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            wmma::fill_fragment(acc[mt], 0);
        }
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(bf, b_sh + (warp * 16) * kStride + ks, kStride);
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
                wmma::load_matrix_sync(af, a_sh + (mt * 16) * kStride + ks, kStride);
                wmma::mma_sync(acc[mt], af, bf, acc[mt]);
            }
        }
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            wmma::store_matrix_sync(dots_sh[warp], acc[mt], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                const int idx = lane + p * 32;
                const int tok_local = mt * 16 + idx / 16;
                const int row_in_warp = idx % 16;
                const int dot = dots_sh[warp][idx];
                const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_scales_sh[warp * 16 + row_in_warp]);
                sums[mt][p] = __fadd_rn(sums[mt][p], __fmul_rn(float(dot), scale_prod));
            }
            __syncwarp();
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok = first_token + mt * 16 + idx / 16;
            const int row = first_row + warp * 16 + idx % 16;
            if (tok < tokens && row < n) {
                *reinterpret_cast<float *>(output + size_t(tok) * output_token + size_t(row) * sizeof(float)) = s11_bf16(sums[mt][p]);
            }
        }
    }
#endif
}

template <int TYPE, int M_TILES = 1>
static __global__ void __launch_bounds__(128) s11_wmma_k128_kernel(
        const char * __restrict__ weights, const signed char * __restrict__ q,
        const float * __restrict__ activation_scales, char * __restrict__ output,
        int k, int n, int tokens, size_t weight_row, size_t output_token) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int bytes = TYPE <= GGML_TYPE_S11_W2_BF16 ? 36 : 68;
    constexpr int kStride = 144;
    const int t = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int first_row = blockIdx.x * 64;
    const int first_token = blockIdx.y * (16 * M_TILES);
    if (first_row >= n || first_token >= tokens) {
        return;
    }

    __shared__ __align__(32) signed char a_sh[16 * M_TILES * kStride];
    __shared__ __align__(32) signed char b_sh[64 * kStride];
    __shared__ __align__(32) int dots_sh[4][16 * 16];
    __shared__ float a_scales_sh[16 * M_TILES];
    __shared__ float w_scales_sh[64];

    float sums[M_TILES][8] = {};

    for (int group = 0; group < k / 128; ++group) {
        if (t < 16 * M_TILES) {
            const int tok = first_token + t;
            a_scales_sh[t] = (tok < tokens) ? activation_scales[size_t(tok) * (k / 128) + group] : 0.0f;
        }
        if (t < 64) {
            const int r = first_row + t;
            w_scales_sh[t] = (r < n)
                ? s11_scale(reinterpret_cast<const unsigned char *>(weights + size_t(r) * weight_row + group * bytes), TYPE)
                : 0.0f;
        }
        #pragma unroll
        for (int idx = t; idx < 16 * M_TILES * 32; idx += 128) {
            const int tok_local = idx / 32;
            const int col4 = idx % 32;
            const int tok = first_token + tok_local;
            *reinterpret_cast<int *>(&a_sh[tok_local * kStride + col4 * 4]) = (tok < tokens)
                ? *reinterpret_cast<const int *>(q + size_t(tok) * k + group * 128 + col4 * 4)
                : 0;
        }
        #pragma unroll
        for (int idx = t; idx < 64 * 32; idx += 128) {
            const int r_local = idx / 32;
            const int col4 = idx % 32;
            const int r = first_row + r_local;
            int packed = 0;
            if (r < n) {
                const auto * block = reinterpret_cast<const unsigned char *>(weights + size_t(r) * weight_row + group * bytes);
                packed = (TYPE <= GGML_TYPE_S11_W2_BF16)
                    ? s11_unpack4_w2(block, col4)
                    : s11_unpack4_w4(block, col4);
            }
            *reinterpret_cast<int *>(&b_sh[r_local * kStride + col4 * 4]) = packed;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[M_TILES];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            wmma::fill_fragment(acc[mt], 0);
        }
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(bf, b_sh + (warp * 16) * kStride + ks, kStride);
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
                wmma::load_matrix_sync(af, a_sh + (mt * 16) * kStride + ks, kStride);
                wmma::mma_sync(acc[mt], af, bf, acc[mt]);
            }
        }
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            wmma::store_matrix_sync(dots_sh[warp], acc[mt], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                const int idx = lane + p * 32;
                const int tok_local = mt * 16 + idx / 16;
                const int row_in_warp = idx % 16;
                const int dot = dots_sh[warp][idx];
                const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_scales_sh[warp * 16 + row_in_warp]);
                sums[mt][p] = __fadd_rn(sums[mt][p], __fmul_rn(float(dot), scale_prod));
            }
            __syncwarp();
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok = first_token + mt * 16 + idx / 16;
            const int row = first_row + warp * 16 + idx % 16;
            if (tok < tokens && row < n) {
                *reinterpret_cast<float *>(output + size_t(tok) * output_token + size_t(row) * sizeof(float)) = s11_bf16(sums[mt][p]);
            }
        }
    }
#endif
}

static __global__ void __launch_bounds__(128) s11_shexp_up_gate_swiglu_wmma_k128_kernel(
        const char * __restrict__ up_weights, const char * __restrict__ gate_weights,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        char * __restrict__ output, int k, int n, int tokens, size_t weight_row, size_t output_token) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int bytes = 36;
    constexpr int kStride = 144;
    const int t = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int first_row = blockIdx.x * 64;
    const int first_token = blockIdx.y * 16;
    if (first_row >= n || first_token >= tokens) {
        return;
    }

    __shared__ __align__(32) signed char a_sh[16 * kStride];
    __shared__ __align__(32) signed char b_up_sh[64 * kStride];
    __shared__ __align__(32) signed char b_gate_sh[64 * kStride];
    __shared__ __align__(32) int dots_sh[4][16 * 16];
    __shared__ float a_scales_sh[16];
    __shared__ float w_up_scales_sh[64];
    __shared__ float w_gate_scales_sh[64];

    float sums_up[8] = {};
    float sums_gate[8] = {};

    for (int group = 0; group < k / 128; ++group) {
        if (t < 16) {
            const int tok = first_token + t;
            a_scales_sh[t] = (tok < tokens) ? activation_scales[size_t(tok) * (k / 128) + group] : 0.0f;
        }
        #pragma unroll
        for (int idx = t; idx < 16 * 32; idx += 128) {
            const int tok_local = idx / 32;
            const int col4 = idx % 32;
            const int tok = first_token + tok_local;
            *reinterpret_cast<int *>(&a_sh[tok_local * kStride + col4 * 4]) = (tok < tokens)
                ? *reinterpret_cast<const int *>(q + size_t(tok) * k + group * 128 + col4 * 4)
                : 0;
        }
        if (t < 64) {
            const int r = first_row + t;
            if (r < n) {
                w_up_scales_sh[t] = s11_scale(reinterpret_cast<const unsigned char *>(up_weights + size_t(r) * weight_row + group * bytes), GGML_TYPE_S11_W2_BF16);
                w_gate_scales_sh[t] = s11_scale(reinterpret_cast<const unsigned char *>(gate_weights + size_t(r) * weight_row + group * bytes), GGML_TYPE_S11_W2_BF16);
            } else {
                w_up_scales_sh[t] = 0.0f;
                w_gate_scales_sh[t] = 0.0f;
            }
        }
        #pragma unroll
        for (int idx = t; idx < 64 * 32; idx += 128) {
            const int r_local = idx / 32;
            const int col4 = idx % 32;
            const int r = first_row + r_local;
            int packed_up = 0;
            int packed_gate = 0;
            if (r < n) {
                packed_up = s11_unpack4_w2(reinterpret_cast<const unsigned char *>(up_weights + size_t(r) * weight_row + group * bytes), col4);
                packed_gate = s11_unpack4_w2(reinterpret_cast<const unsigned char *>(gate_weights + size_t(r) * weight_row + group * bytes), col4);
            }
            *reinterpret_cast<int *>(&b_up_sh[r_local * kStride + col4 * 4]) = packed_up;
            *reinterpret_cast<int *>(&b_gate_sh[r_local * kStride + col4 * 4]) = packed_gate;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_up;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_gate;
        wmma::fill_fragment(acc_up, 0);
        wmma::fill_fragment(acc_gate, 0);
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf_up;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf_gate;
            wmma::load_matrix_sync(af, a_sh + ks, kStride);
            wmma::load_matrix_sync(bf_up, b_up_sh + (warp * 16) * kStride + ks, kStride);
            wmma::load_matrix_sync(bf_gate, b_gate_sh + (warp * 16) * kStride + ks, kStride);
            wmma::mma_sync(acc_up, af, bf_up, acc_up);
            wmma::mma_sync(acc_gate, af, bf_gate, acc_gate);
        }
        wmma::store_matrix_sync(dots_sh[warp], acc_up, 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok_local = idx / 16;
            const int row_in_warp = idx % 16;
            const int dot = dots_sh[warp][idx];
            const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_up_scales_sh[warp * 16 + row_in_warp]);
            sums_up[p] = __fadd_rn(sums_up[p], __fmul_rn(float(dot), scale_prod));
        }
        __syncwarp();
        wmma::store_matrix_sync(dots_sh[warp], acc_gate, 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok_local = idx / 16;
            const int row_in_warp = idx % 16;
            const int dot = dots_sh[warp][idx];
            const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_gate_scales_sh[warp * 16 + row_in_warp]);
            sums_gate[p] = __fadd_rn(sums_gate[p], __fmul_rn(float(dot), scale_prod));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = lane + p * 32;
        const int tok = first_token + idx / 16;
        const int row = first_row + warp * 16 + idx % 16;
        if (tok < tokens && row < n) {
            const float u = s11_bf16(sums_up[p]);
            const float g = s11_bf16(sums_gate[p]);
            const float silu_g = g / (1.0f + expf(-g));
            *reinterpret_cast<float *>(output + size_t(tok) * output_token + size_t(row) * sizeof(float)) = __fmul_rn(silu_g, u);
        }
    }
#endif
}

static __global__ void __launch_bounds__(256) s11_build_routed_plan_kernel(
        const char * __restrict__ ids, int slots, int tokens, size_t ids_slot, size_t ids_token,
        int32_t * __restrict__ tile_experts, int32_t * __restrict__ tile_starts,
        int32_t * __restrict__ tile_counts, int32_t * __restrict__ sorted_assignments,
        int32_t * __restrict__ num_tiles_out) {
    __shared__ int hash_keys[512];
    __shared__ int bucket_counts[512];
    __shared__ int bucket_starts[512];
    __shared__ int bucket_tile_starts[512];
    const int t = threadIdx.x;
    const int m = tokens * slots;
    hash_keys[t] = -1;
    hash_keys[t + 256] = -1;
    bucket_counts[t] = 0;
    bucket_counts[t + 256] = 0;
    __syncthreads();

    int my_bucket[16];
    int my_pos[16];
    #pragma unroll
    for (int step = 0; step < 16; ++step) {
        const int a = t + step * 256;
        my_bucket[step] = -1;
        my_pos[step] = 0;
        if (a < m) {
            const int slot = a % slots;
            const int token = a / slots;
            const int expert = *reinterpret_cast<const int *>(ids + slot * ids_slot + token * ids_token);
            if (expert >= 0) {
                uint32_t h = (uint32_t(expert) * 2654435761u) & 511u;
                while (true) {
                    int prev = hash_keys[h];
                    if (prev == expert) break;
                    if (prev == -1) {
                        prev = atomicCAS(&hash_keys[h], -1, expert);
                        if (prev == -1 || prev == expert) break;
                    }
                    h = (h + 1) & 511u;
                }
                my_bucket[step] = int(h);
                my_pos[step] = atomicAdd(&bucket_counts[h], 1);
            }
        }
    }
    __syncthreads();

    if (t == 0) {
        int row_off = 0, tile_off = 0;
        for (int h = 0; h < 512; ++h) {
            const int cnt = bucket_counts[h];
            if (cnt > 0) {
                bucket_starts[h] = row_off;
                bucket_tile_starts[h] = tile_off;
                row_off += cnt;
                tile_off += (cnt + 15) / 16;
            }
        }
        num_tiles_out[0] = tile_off;
    }
    __syncthreads();

    for (int h = t; h < 512; h += 256) {
        const int cnt = bucket_counts[h];
        if (cnt > 0) {
            const int expert = hash_keys[h];
            const int b_start = bucket_starts[h];
            const int t_start = bucket_tile_starts[h];
            for (int r = 0; r < cnt; r += 16) {
                const int tile = t_start + r / 16;
                tile_experts[tile] = expert;
                tile_starts[tile] = b_start + r;
                tile_counts[tile] = min(16, cnt - r);
            }
        }
    }
    #pragma unroll
    for (int step = 0; step < 16; ++step) {
        const int a = t + step * 256;
        const int h = my_bucket[step];
        if (a < m && h >= 0) {
            sorted_assignments[bucket_starts[h] + my_pos[step]] = a;
        }
    }
}

static __global__ void __launch_bounds__(128) s11_routed_wmma_k128_kernel(
        const char * __restrict__ weights, const signed char * __restrict__ q,
        const float * __restrict__ activation_scales,
        const int32_t * __restrict__ tile_experts, const int32_t * __restrict__ tile_starts,
        const int32_t * __restrict__ tile_counts, const int32_t * __restrict__ sorted_assignments,
        const int32_t * __restrict__ num_tiles_in, char * __restrict__ output,
        int k, int n, int slots, int input_slots,
        size_t weight_row, size_t weight_expert, size_t output_slot, size_t output_token) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int tile = blockIdx.y;
    if (tile >= num_tiles_in[0]) {
        return;
    }
    const int expert = tile_experts[tile];
    const int start = tile_starts[tile];
    const int count = tile_counts[tile];
    const int t = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int first_row = blockIdx.x * 64;
    if (first_row >= n || count <= 0) {
        return;
    }
    if (first_row == 0 && t == 0 && k == 512 && n == 2048) {
        atomicAdd(&s11_completed_assignments, static_cast<unsigned long long>(count));
    }

    constexpr int kStride = 144;
    __shared__ __align__(32) signed char a_sh[16 * kStride];
    __shared__ __align__(32) signed char b_sh[64 * kStride];
    __shared__ __align__(32) int dots_sh[4][16 * 16];
    __shared__ int assign_sh[16];
    __shared__ int source_sh[16];
    __shared__ float a_scales_sh[16];
    __shared__ float w_scales_sh[64];

    if (t < 16) {
        const int a = (t < count) ? sorted_assignments[start + t] : 0;
        assign_sh[t] = a;
        const int slot = a % slots;
        const int token = a / slots;
        source_sh[t] = token * input_slots + (input_slots == 1 ? 0 : slot);
    }
    __syncthreads();

    const char * expert_weights = weights + size_t(expert) * weight_expert;
    float sums[8] = {};

    for (int group = 0; group < k / 128; ++group) {
        if (t < 16) {
            a_scales_sh[t] = (t < count) ? activation_scales[size_t(source_sh[t]) * (k / 128) + group] : 0.0f;
        }
        if (t < 64) {
            const int r = first_row + t;
            w_scales_sh[t] = (r < n)
                ? s11_scale(reinterpret_cast<const unsigned char *>(expert_weights + size_t(r) * weight_row + group * 36), GGML_TYPE_S11_W2)
                : 0.0f;
        }
        #pragma unroll
        for (int idx = t; idx < 16 * 32; idx += 128) {
            const int tok_local = idx / 32;
            const int col4 = idx % 32;
            *reinterpret_cast<int *>(&a_sh[tok_local * kStride + col4 * 4]) = (tok_local < count)
                ? *reinterpret_cast<const int *>(q + size_t(source_sh[tok_local]) * k + group * 128 + col4 * 4)
                : 0;
        }
        #pragma unroll
        for (int idx = t; idx < 64 * 32; idx += 128) {
            const int r_local = idx / 32;
            const int col4 = idx % 32;
            const int r = first_row + r_local;
            int packed = 0;
            if (r < n) {
                const auto * block = reinterpret_cast<const unsigned char *>(expert_weights + size_t(r) * weight_row + group * 36);
                packed = s11_unpack4_w2(block, col4);
            }
            *reinterpret_cast<int *>(&b_sh[r_local * kStride + col4 * 4]) = packed;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
        wmma::fill_fragment(acc, 0);
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a_sh + ks, kStride);
            wmma::load_matrix_sync(bf, b_sh + (warp * 16) * kStride + ks, kStride);
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(dots_sh[warp], acc, 16, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok_local = idx / 16;
            const int row_in_warp = idx % 16;
            const int dot = dots_sh[warp][idx];
            const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_scales_sh[warp * 16 + row_in_warp]);
            sums[p] = __fadd_rn(sums[p], __fmul_rn(float(dot), scale_prod));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = lane + p * 32;
        const int tok_local = idx / 16;
        const int row = first_row + warp * 16 + idx % 16;
        if (tok_local < count && row < n) {
            const int a = assign_sh[tok_local];
            const int slot = a % slots;
            const int token = a / slots;
            *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(slot) * output_slot + size_t(token) * output_token) = s11_bf16(sums[p]);
        }
    }
#endif
}

struct s11_quant_slot_state {
    const void * last_x_data = nullptr;
    int last_k = 0;
    int last_tokens = 0;
    int last_input_slots = 0;
};

struct s11_persistent_state {
    signed char * q_buf[2] = {nullptr, nullptr};
    float * s_buf[2] = {nullptr, nullptr};
    size_t q_cap[2] = {0, 0};
    size_t s_cap[2] = {0, 0};
    const void * last_x_data[2] = {nullptr, nullptr};
    int last_k[2] = {0, 0};
    int last_tokens[2] = {0, 0};
    int last_input_slots[2] = {0, 0};

    int32_t * plan_buf = nullptr;
    size_t plan_cap = 0;
    const void * last_ids_data = nullptr;
    int last_ids_tokens = 0;
    int last_ids_slots = 0;
    size_t last_ids_nb0 = 0;
    size_t last_ids_nb1 = 0;

    std::unordered_map<const void *, s11_quant_slot_state> graph_exit_state;
};

static s11_persistent_state g_s11_state;

static void s11_ensure_persistent_buffers() {
    for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
        const size_t req_q = 512ULL * (buf_idx == 0 ? 1 : 8) * 4096;
        const size_t req_s = 512ULL * (buf_idx == 0 ? 1 : 8) * 32;
        if (g_s11_state.q_cap[buf_idx] < req_q) {
            if (g_s11_state.q_buf[buf_idx]) CUDA_CHECK(cudaFree(g_s11_state.q_buf[buf_idx]));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.q_buf[buf_idx]), req_q * sizeof(signed char)));
            g_s11_state.q_cap[buf_idx] = req_q;
        }
        if (g_s11_state.s_cap[buf_idx] < req_s) {
            if (g_s11_state.s_buf[buf_idx]) CUDA_CHECK(cudaFree(g_s11_state.s_buf[buf_idx]));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.s_buf[buf_idx]), req_s * sizeof(float)));
            g_s11_state.s_cap[buf_idx] = req_s;
        }
    }
    constexpr size_t kPlanInts = 4096 * 4 + 16;
    if (g_s11_state.plan_cap < kPlanInts) {
        if (g_s11_state.plan_buf) CUDA_CHECK(cudaFree(g_s11_state.plan_buf));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.plan_buf), kPlanInts * sizeof(int32_t)));
        g_s11_state.plan_cap = kPlanInts;
    }
}

void ggml_cuda_s11_reset_graph_state() {
    g_s11_state.last_x_data[0] = nullptr;
    g_s11_state.last_x_data[1] = nullptr;
    g_s11_state.last_ids_data = nullptr;
    s11_ensure_persistent_buffers();
}

void ggml_cuda_s11_begin_split(const ggml_cgraph * cgraph) {
    g_s11_state.last_x_data[1] = nullptr;
    g_s11_state.last_ids_data = nullptr;
    s11_ensure_persistent_buffers();

    if (g_s11_state.last_x_data[0] == nullptr || cgraph == nullptr || cgraph->n_nodes <= 0) {
        g_s11_state.last_x_data[0] = nullptr;
        return;
    }
    const int limit = std::min(cgraph->n_nodes, 8);
    for (int i = 0; i < limit; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || ggml_is_empty(node) || node->op == GGML_OP_NONE ||
            node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if ((node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MUL_MAT) &&
            node->src[0] && ggml_cuda_is_s11(node->src[0]->type)) {
            const ggml_tensor * x = node->src[1];
            const int input_slots = node->op == GGML_OP_MUL_MAT_ID ? (int) x->ne[1] : 1;
            const int tokens = node->op == GGML_OP_MUL_MAT_ID ? (int) node->ne[2] : (int) node->ne[1];
            const int k = (int) node->src[0]->ne[0];
            const bool contiguous = x && x->type == GGML_TYPE_F32 &&
                (x->view_src == nullptr || (x->view_offs == 0 && ggml_is_contiguous(x)));
            if (!contiguous || input_slots != 1 || g_s11_state.last_input_slots[0] != 1 ||
                x->data != g_s11_state.last_x_data[0] || k != g_s11_state.last_k[0] ||
                tokens != g_s11_state.last_tokens[0]) {
                g_s11_state.last_x_data[0] = nullptr;
            }
            return;
        }
        if (node->data == g_s11_state.last_x_data[0] ||
            (node->view_src && node->view_src->data == g_s11_state.last_x_data[0])) {
            g_s11_state.last_x_data[0] = nullptr;
            return;
        }
    }
    g_s11_state.last_x_data[0] = nullptr;
}

void ggml_cuda_s11_record_graph_exit_state(const void * graph_key) {
    if (!graph_key) return;
    g_s11_state.graph_exit_state[graph_key] = s11_quant_slot_state{
        g_s11_state.last_x_data[0],
        g_s11_state.last_k[0],
        g_s11_state.last_tokens[0],
        g_s11_state.last_input_slots[0],
    };
}

void ggml_cuda_s11_restore_graph_exit_state(const void * graph_key) {
    if (!graph_key) {
        g_s11_state.last_x_data[0] = nullptr;
        return;
    }
    const auto it = g_s11_state.graph_exit_state.find(graph_key);
    if (it != g_s11_state.graph_exit_state.end()) {
        g_s11_state.last_x_data[0] = it->second.last_x_data;
        g_s11_state.last_k[0] = it->second.last_k;
        g_s11_state.last_tokens[0] = it->second.last_tokens;
        g_s11_state.last_input_slots[0] = it->second.last_input_slots;
    } else {
        g_s11_state.last_x_data[0] = nullptr;
    }
}

void ggml_cuda_s11_invalidate_if_written(const void * data_ptr) {
    if (!data_ptr) return;
    if (g_s11_state.last_x_data[0] == data_ptr) g_s11_state.last_x_data[0] = nullptr;
    if (g_s11_state.last_x_data[1] == data_ptr) g_s11_state.last_x_data[1] = nullptr;
    if (g_s11_state.last_ids_data == data_ptr) g_s11_state.last_ids_data = nullptr;
}

static bool s11_is_hybrid_mode() {
    static const bool hybrid = [] {
        const char * mode = std::getenv("S11_EXPERT_MODE");
        return mode && std::strcmp(mode, "hybrid") == 0;
    }();
    return hybrid;
}

static int s11_optimization() {
    static const int mode = [] {
        const char * value = std::getenv("S11_CUDA_OPT");
        if (!value || std::strcmp(value, "reference") == 0) { return 0; }
        if (std::strcmp(value, "dp4a") == 0) { return 1; }
        if (std::strcmp(value, "tiled") == 0) { return 2; }
        GGML_ABORT("Invalid S11_CUDA_OPT: %s", value);
    }();
    return mode;
}

bool ggml_cuda_s11_bf16_head(const ggml_tensor * tensor) {
    static const bool enabled = [] {
        const char * value = std::getenv("S11_HEAD_LAYOUT");
        if (!value || std::strcmp(value, "packed") == 0) { return false; }
        if (std::strcmp(value, "bf16") == 0) { return true; }
        GGML_ABORT("Invalid S11_HEAD_LAYOUT: %s", value);
    }();
    return enabled && tensor->type == GGML_TYPE_S11_I8 &&
        std::strcmp(tensor->name, "output.weight") == 0 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static __global__ void s11_restore_head(const unsigned char * packed, nv_bfloat16 * output, int k, int rows) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < k * rows) {
        const int row = index / k, col = index % k;
        const unsigned char * block = packed + row * (k / 128) * 132 + (col / 128) * 132;
        output[index] = __float2bfloat16_rn(__fdiv_rn(float(reinterpret_cast<const signed char *>(block)[col % 128]), s11_scale(block, GGML_TYPE_S11_I8)));
    }
}

void ggml_cuda_s11_upload_head(ggml_tensor * tensor, const void * canonical, cudaStream_t stream) {
    constexpr int tile_rows = 128;
    unsigned char * staging = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&staging), tile_rows * tensor->nb[1]));
    for (int64_t row = 0; row < tensor->ne[1]; row += tile_rows) {
        const int rows = std::min<int64_t>(tile_rows, tensor->ne[1] - row);
        CUDA_CHECK(cudaMemcpyAsync(staging, static_cast<const char *>(canonical) + row * tensor->nb[1],
                                   rows * tensor->nb[1], cudaMemcpyHostToDevice, stream));
        s11_restore_head<<<(rows * tensor->ne[0] + 255) / 256, 256, 0, stream>>>(
            staging, static_cast<nv_bfloat16 *>(tensor->data) + row * tensor->ne[0], tensor->ne[0], rows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
}

static __global__ void s11_head_bf16(const nv_bfloat16 * weights, const char * input, char * output,
                                    int k, int n, int tokens, size_t input_stride, size_t output_stride, bool fast_reduce) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32, token = blockIdx.y;
    if (row >= n || token >= tokens) { return; }
    const float * x = reinterpret_cast<const float *>(input + token * input_stride);
    float sum = 0;
    if (fast_reduce && (k % 256 == 0)) {
        const nv_bfloat16 * w_row = weights + size_t(row) * k;
        #pragma unroll 4
        for (int base = 0; base < k; base += 256) {
            const int col = base + lane * 8;
            const uint4 w8 = *reinterpret_cast<const uint4 *>(w_row + col);
            const float4 x_lo = *reinterpret_cast<const float4 *>(x + col);
            const float4 x_hi = *reinterpret_cast<const float4 *>(x + col + 4);
            const nv_bfloat162 * w2 = reinterpret_cast<const nv_bfloat162 *>(&w8);
            const float2 f0 = __bfloat1622float2(w2[0]);
            const float2 f1 = __bfloat1622float2(w2[1]);
            const float2 f2 = __bfloat1622float2(w2[2]);
            const float2 f3 = __bfloat1622float2(w2[3]);
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_lo.x), f0.x));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_lo.y), f0.y));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_lo.z), f1.x));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_lo.w), f1.y));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_hi.x), f2.x));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_hi.y), f2.y));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_hi.z), f3.x));
            sum = __fadd_rn(sum, __fmul_rn(s11_bf16(x_hi.w), f3.y));
        }
        for (int offset = 16; offset; offset /= 2) {
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
    } else {
        for (int group = 0; group < k / 128; ++group) {
            float dot = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = group * 128 + lane + i * 32;
                dot = __fadd_rn(dot, __fmul_rn(s11_bf16(x[col]), __bfloat162float(weights[size_t(row) * k + col])));
            }
            for (int offset = 16; offset; offset /= 2) {
                dot = __fadd_rn(dot, __shfl_down_sync(0xffffffff, dot, offset));
            }
            if (!lane) { sum = __fadd_rn(sum, dot); }
        }
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + token * output_stride + row * sizeof(float)) = s11_bf16(sum);
    }
}

static __global__ void __launch_bounds__(256) s11_head_bf16_m1(
        const nv_bfloat16 * __restrict__ weights, const char * __restrict__ input,
        char * __restrict__ output, int n) {
    __shared__ __align__(16) float x_sh[2048];
    const int t = threadIdx.x;
    const float4 * x_in4 = reinterpret_cast<const float4 *>(input);
    float4 * x_sh4 = reinterpret_cast<float4 *>(x_sh);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx4 = t + i * 256;
        float4 v = x_in4[idx4];
        v.x = s11_bf16(v.x);
        v.y = s11_bf16(v.y);
        v.z = s11_bf16(v.z);
        v.w = s11_bf16(v.w);
        x_sh4[idx4] = v;
    }
    __syncthreads();

    const int lane = t % 32;
    const int row = blockIdx.x * 8 + t / 32;
    if (row >= n) {
        return;
    }
    const nv_bfloat16 * w_row = weights + size_t(row) * 2048;
    float sum = 0.0f;
    #pragma unroll 4
    for (int base = 0; base < 2048; base += 256) {
        const int col = base + lane * 8;
        const uint4 w8 = *reinterpret_cast<const uint4 *>(w_row + col);
        const float4 x_lo = *reinterpret_cast<const float4 *>(x_sh + col);
        const float4 x_hi = *reinterpret_cast<const float4 *>(x_sh + col + 4);
        const nv_bfloat162 * w2 = reinterpret_cast<const nv_bfloat162 *>(&w8);
        const float2 f0 = __bfloat1622float2(w2[0]);
        const float2 f1 = __bfloat1622float2(w2[1]);
        const float2 f2 = __bfloat1622float2(w2[2]);
        const float2 f3 = __bfloat1622float2(w2[3]);
        sum = __fadd_rn(sum, __fmul_rn(x_lo.x, f0.x));
        sum = __fadd_rn(sum, __fmul_rn(x_lo.y, f0.y));
        sum = __fadd_rn(sum, __fmul_rn(x_lo.z, f1.x));
        sum = __fadd_rn(sum, __fmul_rn(x_lo.w, f1.y));
        sum = __fadd_rn(sum, __fmul_rn(x_hi.x, f2.x));
        sum = __fadd_rn(sum, __fmul_rn(x_hi.y, f2.y));
        sum = __fadd_rn(sum, __fmul_rn(x_hi.z, f3.x));
        sum = __fadd_rn(sum, __fmul_rn(x_hi.w, f3.y));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + size_t(row) * sizeof(float)) = s11_bf16(sum);
    }
}

void ggml_cuda_s11_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->op == GGML_OP_MUL_MAT_ID ? dst->src[2] : nullptr;
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(w->ne[0] % 128 == 0 && w->ne[3] == 1 && x->ne[3] == 1);
    const int k = w->ne[0];
    const int tokens = ids ? dst->ne[2] : dst->ne[1];
    const int slots = ids ? dst->ne[1] : 1;
    const int input_slots = ids ? x->ne[1] : 1;
    GGML_ASSERT(!ids || input_slots == 1 || input_slots == slots);
    const int optimization = s11_optimization();
    if (ggml_cuda_s11_bf16_head(w)) {
        GGML_ASSERT(!ids && !w->view_src);
        if (tokens == 1 && optimization >= 2 && k == 2048) {
            s11_head_bf16_m1<<<dim3((w->ne[1] + 7) / 8, 1), 256, 0, ctx.stream()>>>(
                static_cast<const nv_bfloat16 *>(w->data), static_cast<const char *>(x->data),
                static_cast<char *>(dst->data), w->ne[1]);
        } else {
            s11_head_bf16<<<dim3((w->ne[1] + 3) / 4, tokens), 128, 0, ctx.stream()>>>(
                static_cast<const nv_bfloat16 *>(w->data), static_cast<const char *>(x->data), static_cast<char *>(dst->data),
                k, w->ne[1], tokens, x->nb[1], dst->nb[1], optimization >= 2);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool need_fake = w->type != GGML_TYPE_S11_I8 && !optimization;
    ggml_cuda_pool_alloc<float> fake(ctx.pool(), need_fake ? size_t(tokens) * input_slots * k : 1);

    signed char * q_ptr = nullptr;
    float * scales_ptr = nullptr;
    ggml_cuda_pool_alloc<signed char> q_fallback(ctx.pool());
    ggml_cuda_pool_alloc<float> scales_fallback(ctx.pool());

    if (w->type != GGML_TYPE_S11_I8) {
        const int buf_idx = (input_slots == 1) ? 0 : 1;
        const size_t req_q = std::max<size_t>(size_t(tokens) * input_slots * k, 512ULL * (buf_idx == 0 ? 1 : 8) * 4096);
        const size_t req_s = std::max<size_t>(size_t(tokens) * input_slots * (k / 128), 512ULL * (buf_idx == 0 ? 1 : 8) * 32);
        if (optimization == 2 && !need_fake) {
            if (g_s11_state.q_cap[buf_idx] < req_q) {
                if (g_s11_state.q_buf[buf_idx]) CUDA_CHECK(cudaFree(g_s11_state.q_buf[buf_idx]));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.q_buf[buf_idx]), req_q * sizeof(signed char)));
                g_s11_state.q_cap[buf_idx] = req_q;
                g_s11_state.last_x_data[buf_idx] = nullptr;
            }
            if (g_s11_state.s_cap[buf_idx] < req_s) {
                if (g_s11_state.s_buf[buf_idx]) CUDA_CHECK(cudaFree(g_s11_state.s_buf[buf_idx]));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.s_buf[buf_idx]), req_s * sizeof(float)));
                g_s11_state.s_cap[buf_idx] = req_s;
                g_s11_state.last_x_data[buf_idx] = nullptr;
            }
            q_ptr = g_s11_state.q_buf[buf_idx];
            scales_ptr = g_s11_state.s_buf[buf_idx];
            const bool can_reuse = (x->view_src == nullptr || (x->view_offs == 0 && ggml_is_contiguous(x))) &&
                                   (g_s11_state.last_x_data[buf_idx] == x->data) &&
                                   (g_s11_state.last_k[buf_idx] == k) &&
                                   (g_s11_state.last_tokens[buf_idx] == tokens) &&
                                   (g_s11_state.last_input_slots[buf_idx] == input_slots);
            if (!can_reuse) {
                s11_quantize<<<dim3(k / 128, tokens * input_slots), 32, 0, ctx.stream()>>>(
                    static_cast<const char *>(x->data), q_ptr, scales_ptr, nullptr, k,
                    input_slots, tokens, ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1]);
                CUDA_CHECK(cudaGetLastError());
                g_s11_state.last_x_data[buf_idx] = x->data;
                g_s11_state.last_k[buf_idx] = k;
                g_s11_state.last_tokens[buf_idx] = tokens;
                g_s11_state.last_input_slots[buf_idx] = input_slots;
            }
        } else {
            q_fallback.alloc(size_t(tokens) * input_slots * k);
            scales_fallback.alloc(size_t(tokens) * input_slots * (k / 128));
            q_ptr = q_fallback.get();
            scales_ptr = scales_fallback.get();
            s11_quantize<<<dim3(k / 128, tokens * input_slots), 32, 0, ctx.stream()>>>(
                static_cast<const char *>(x->data), q_ptr, scales_ptr, need_fake ? fake.get() : nullptr, k,
                input_slots, tokens, ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1]);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (optimization) {
        #define S11_PACKED_LAUNCH(TYPE, COLS) \
            s11_projection_packed<TYPE, COLS><<<dim3((w->ne[1] + 3) / 4, (tokens * slots + COLS - 1) / COLS), 128, 0, ctx.stream()>>>( \
                static_cast<const char *>(w->data), q_ptr, scales_ptr, fake.get(), static_cast<const char *>(x->data), \
                ids ? static_cast<const char *>(ids->data) : nullptr, static_cast<char *>(dst->data), \
                k, w->ne[1], slots, tokens, input_slots, w->nb[1], w->nb[2], \
                ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1], ids ? ids->nb[0] : 0, ids ? ids->nb[1] : 0, \
                ids ? dst->nb[1] : 0, ids ? dst->nb[2] : dst->nb[1])
        switch (w->type) {
            case GGML_TYPE_S11_W2:
                if (optimization == 2 && ids && tokens >= 8 && tokens * slots <= 4096 && !s11_is_hybrid_mode()) {
                    constexpr size_t kPlanInts = 4096 * 4 + 16;
                    if (g_s11_state.plan_cap < kPlanInts) {
                        if (g_s11_state.plan_buf) CUDA_CHECK(cudaFree(g_s11_state.plan_buf));
                        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.plan_buf), kPlanInts * sizeof(int32_t)));
                        g_s11_state.plan_cap = kPlanInts;
                        g_s11_state.last_ids_data = nullptr;
                    }
                    int32_t * tile_experts = g_s11_state.plan_buf;
                    int32_t * tile_starts  = g_s11_state.plan_buf + 4096;
                    int32_t * tile_counts  = g_s11_state.plan_buf + 8192;
                    int32_t * sorted_assn  = g_s11_state.plan_buf + 12288;
                    int32_t * num_tiles    = g_s11_state.plan_buf + 16384;
                    const bool reuse_plan = (g_s11_state.last_ids_data == ids->data) &&
                                            (g_s11_state.last_ids_tokens == tokens) &&
                                            (g_s11_state.last_ids_slots == slots) &&
                                            (g_s11_state.last_ids_nb0 == ids->nb[0]) &&
                                            (g_s11_state.last_ids_nb1 == ids->nb[1]);
                    if (!reuse_plan) {
                        s11_build_routed_plan_kernel<<<1, 256, 0, ctx.stream()>>>(
                            static_cast<const char *>(ids->data), slots, tokens, ids->nb[0], ids->nb[1],
                            tile_experts, tile_starts, tile_counts, sorted_assn, num_tiles);
                        CUDA_CHECK(cudaGetLastError());
                        g_s11_state.last_ids_data = ids->data;
                        g_s11_state.last_ids_tokens = tokens;
                        g_s11_state.last_ids_slots = slots;
                        g_s11_state.last_ids_nb0 = ids->nb[0];
                        g_s11_state.last_ids_nb1 = ids->nb[1];
                    }
                    const int m = tokens * slots;
                    const int max_tiles = std::min(m, (m + 15) / 16 + std::min(m, 256));
                    s11_routed_wmma_k128_kernel<<<dim3((w->ne[1] + 63) / 64, max_tiles), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr,
                        tile_experts, tile_starts, tile_counts, sorted_assn, num_tiles,
                        static_cast<char *>(dst->data), k, w->ne[1], slots, input_slots,
                        w->nb[1], w->nb[2], dst->nb[1], dst->nb[2]);
                } else {
                    S11_PACKED_LAUNCH(GGML_TYPE_S11_W2, 1);
                }
                break;
            case GGML_TYPE_S11_W2_BF16:
                if (optimization == 2 && !ids && tokens >= 32) {
                    s11_wmma_k128_kernel<GGML_TYPE_S11_W2_BF16, 2><<<dim3((w->ne[1] + 63) / 64, (tokens + 31) / 32), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], tokens, w->nb[1], dst->nb[1]);
                } else if (optimization == 2 && !ids && tokens >= 16) {
                    s11_wmma_k128_kernel<GGML_TYPE_S11_W2_BF16, 1><<<dim3((w->ne[1] + 63) / 64, (tokens + 15) / 16), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], tokens, w->nb[1], dst->nb[1]);
                } else if (optimization == 2 && !ids && tokens >= 8) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 8); }
                else if (optimization == 2 && !ids && tokens >= 4) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 4); }
                else if (optimization == 2 && !ids && tokens == 1 && (k % 512 == 0) && k <= 4096) {
                    s11_projection_packed_m1<GGML_TYPE_S11_W2_BF16><<<dim3((w->ne[1] + 7) / 8, 1), 256, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], w->nb[1]);
                } else { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 1); }
                break;
            case GGML_TYPE_S11_W4:
                if (optimization == 2 && !ids && tokens >= 32) {
                    s11_wmma_k128_kernel<GGML_TYPE_S11_W4, 2><<<dim3((w->ne[1] + 63) / 64, (tokens + 31) / 32), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], tokens, w->nb[1], dst->nb[1]);
                } else if (optimization == 2 && !ids && tokens >= 16) {
                    s11_wmma_k128_kernel<GGML_TYPE_S11_W4, 1><<<dim3((w->ne[1] + 63) / 64, (tokens + 15) / 16), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], tokens, w->nb[1], dst->nb[1]);
                } else if (optimization == 2 && !ids && tokens >= 8) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 8); }
                else if (optimization == 2 && !ids && tokens >= 4) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 4); }
                else if (optimization == 2 && !ids && tokens == 1 && (k % 512 == 0) && k <= 4096) {
                    s11_projection_packed_m1<GGML_TYPE_S11_W4><<<dim3((w->ne[1] + 7) / 8, 1), 256, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], w->nb[1]);
                } else { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 1); }
                break;
            case GGML_TYPE_S11_I8: S11_PACKED_LAUNCH(GGML_TYPE_S11_I8, 1); break;
            default: GGML_ABORT("Unsupported S11 packed type");
        }
        #undef S11_PACKED_LAUNCH
    } else {
        s11_projection<<<dim3((w->ne[1] + 3) / 4, tokens * slots), 128, 0, ctx.stream()>>>(
        static_cast<const char *>(w->data), static_cast<const char *>(x->data), q_ptr, scales_ptr, fake.get(),
        ids ? static_cast<const char *>(ids->data) : nullptr, static_cast<char *>(dst->data),
        w->type, k, w->ne[1], slots, tokens, input_slots, w->nb[1], w->nb[2],
        ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1],
        ids ? ids->nb[0] : 0, ids ? ids->nb[1] : 0,
        ids ? dst->nb[1] : 0, ids ? dst->nb[2] : dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void __launch_bounds__(256) s11_rms_norm_offset_bf16_kernel(
        const char * __restrict__ x, const nv_bfloat16 * __restrict__ mw, float * __restrict__ dst,
        signed char * __restrict__ q_out, float * __restrict__ s_out,
        int k, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        float eps, float scale, float bias) {
    const int row = blockIdx.x;
    const int i1 = row % ne1;
    const int i2 = (row / ne1) % ne2;
    const int i3 = row / (ne1 * ne2);
    const float * x_row = reinterpret_cast<const float *>(x + size_t(i1) * nb1 + size_t(i2) * nb2 + size_t(i3) * nb3);
    float * y_row = dst + size_t(row) * k;

    __shared__ float x_sh[2048];
    __shared__ float warp_sums[8];

    if (k == 2048) {
        float sum_sq = 0.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int col = threadIdx.x + 256 * m;
            const float v = x_row[col];
            x_sh[col] = v;
            sum_sq = __fadd_rn(sum_sq, __fmul_rn(v, v));
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
        }
        if ((threadIdx.x & 31) == 0) {
            warp_sums[threadIdx.x / 32] = sum_sq;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
            for (int offset = 4; offset > 0; offset /= 2) {
                total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
            }
            if (threadIdx.x == 0) {
                warp_sums[0] = rsqrtf(total / 2048.0f + eps);
            }
        }
        __syncthreads();
        const float inv_rms = warp_sums[0];
        if (q_out && s_out) {
            const int warp = threadIdx.x / 32;
            const int lane = threadIdx.x % 32;
            #pragma unroll
            for (int g_sub = 0; g_sub < 2; ++g_sub) {
                const int group = warp * 2 + g_sub;
                float values[4];
                float maximum = 1e-5f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int col = group * 128 + lane + 32 * i;
                    const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
                    const float val = s11_bf16(__fmul_rn(__fmul_rn(x_sh[col], inv_rms), w));
                    y_row[col] = val;
                    values[i] = val;
                    maximum = fmaxf(maximum, fabsf(val));
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
                }
                const float inverse = __fmul_rn(__fdiv_rn(1.0f, maximum), 127.0f);
                if (lane == 0) {
                    s_out[size_t(row) * 16 + group] = __fdiv_rn(1.0f, inverse);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int col = group * 128 + lane + 32 * i;
                    const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(values[i], inverse))));
                    q_out[size_t(row) * 2048 + col] = static_cast<signed char>(quant);
                }
            }
        } else {
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                const int col = threadIdx.x + 256 * m;
                const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
                const float val = __fmul_rn(__fmul_rn(x_sh[col], inv_rms), w);
                y_row[col] = s11_bf16(val);
            }
        }
        return;
    }

    if (k == 128) {
        const float v_reg = (threadIdx.x < 128) ? x_row[threadIdx.x] : 0.0f;
        float sum_sq = (threadIdx.x < 128) ? __fmul_rn(v_reg, v_reg) : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
        }
        if ((threadIdx.x & 31) == 0) {
            warp_sums[threadIdx.x / 32] = sum_sq;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
            for (int offset = 4; offset > 0; offset /= 2) {
                total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
            }
            if (threadIdx.x == 0) {
                warp_sums[0] = rsqrtf(total / 128.0f + eps);
            }
        }
        __syncthreads();
        if (threadIdx.x < 128) {
            const float inv_rms = warp_sums[0];
            const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[threadIdx.x]), scale), bias);
            y_row[threadIdx.x] = s11_bf16(__fmul_rn(__fmul_rn(v_reg, inv_rms), w));
        }
        return;
    }

    if (false && k == 256) {
        const float v_reg = x_row[threadIdx.x];
        float sum_sq = __fadd_rn(0.0f, __fmul_rn(v_reg, v_reg));
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
        }
        if ((threadIdx.x & 31) == 0) {
            warp_sums[threadIdx.x / 32] = sum_sq;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
            for (int offset = 4; offset > 0; offset /= 2) {
                total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
            }
            if (threadIdx.x == 0) {
                warp_sums[0] = rsqrtf(total / 256.0f + eps);
            }
        }
        __syncthreads();
        const float inv_rms = warp_sums[0];
        const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[threadIdx.x]), scale), bias);
        y_row[threadIdx.x] = s11_bf16(__fmul_rn(__fmul_rn(v_reg, inv_rms), w));
        return;
    }

    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < k; col += 256) {
        const float v = x_row[col];
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(v, v));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x / 32] = sum_sq;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
        }
        if (threadIdx.x == 0) {
            warp_sums[0] = rsqrtf(total / float(k) + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];
    for (int col = threadIdx.x; col < k; col += 256) {
        const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
        const float val = __fmul_rn(__fmul_rn(x_row[col], inv_rms), w);
        y_row[col] = s11_bf16(val);
    }
}

static __global__ void __launch_bounds__(256) s11_residual_rms_norm_quant_fused_kernel(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ res_dst,
        const nv_bfloat16 * __restrict__ mw, float * __restrict__ norm_dst,
        signed char * __restrict__ q_out, float * __restrict__ s_out,
        float eps, float scale, float bias) {
    const int row = blockIdx.x;
    const size_t base = size_t(row) * 2048;
    __shared__ float x_sh[2048];
    __shared__ float warp_sums[8];

    float sum_sq = 0.0f;
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int col = threadIdx.x + 256 * m;
        const float v = s11_bf16(__fadd_rn(a[base + col], b[base + col]));
        x_sh[col] = v;
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(v, v));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x / 32] = sum_sq;
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int col = threadIdx.x + 256 * m;
        res_dst[base + col] = x_sh[col];
    }
    if (threadIdx.x < 32) {
        float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
        }
        if (threadIdx.x == 0) {
            warp_sums[0] = rsqrtf(total / 2048.0f + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];
    if (q_out && s_out) {
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        #pragma unroll
        for (int g_sub = 0; g_sub < 2; ++g_sub) {
            const int group = warp * 2 + g_sub;
            float values[4];
            float maximum = 1e-5f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = group * 128 + lane + 32 * i;
                const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
                const float val = s11_bf16(__fmul_rn(__fmul_rn(x_sh[col], inv_rms), w));
                norm_dst[base + col] = val;
                values[i] = val;
                maximum = fmaxf(maximum, fabsf(val));
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
            }
            const float inverse = __fmul_rn(__fdiv_rn(1.0f, maximum), 127.0f);
            if (lane == 0) {
                s_out[size_t(row) * 16 + group] = __fdiv_rn(1.0f, inverse);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = group * 128 + lane + 32 * i;
                const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(values[i], inverse))));
                q_out[base + col] = static_cast<signed char>(quant);
            }
        }
    } else {
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int col = threadIdx.x + 256 * m;
            const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
            norm_dst[base + col] = s11_bf16(__fmul_rn(__fmul_rn(x_sh[col], inv_rms), w));
        }
    }
}

static __global__ void __launch_bounds__(256) s11_shexp_gate_epilogue_rms_norm_quant_fused_kernel(
        const nv_bfloat16 * __restrict__ gate_w,
        const float * __restrict__ attn_post_norm,
        const float * __restrict__ shared_gate,
        const float * __restrict__ ffn_shexp,
        const float * __restrict__ moe_out,
        const float * __restrict__ ffn_residual,
        float * __restrict__ res_dst,
        const nv_bfloat16 * __restrict__ mw,
        float * __restrict__ norm_dst,
        signed char * __restrict__ q_out,
        float * __restrict__ s_out,
        float eps, float scale, float bias) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const size_t base = size_t(row) * 2048;
    __shared__ float raw_g_sh;
    __shared__ float x_sh[2048];
    __shared__ float warp_sums[8];

    if (gate_w && attn_post_norm) {
        if (t < 32) {
            float sum = 0.0f;
            for (int index = t; index < 2048; index += 32) {
                sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(gate_w[index]), attn_post_norm[base + index]));
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
            }
            if (t == 0) {
                raw_g_sh = sum;
            }
        }
    } else if (t == 0) {
        raw_g_sh = shared_gate[row];
    }
    __syncthreads();
    const float sig_g = s11_bf16(1.0f / (1.0f + expf(-raw_g_sh)));

    float sum_sq = 0.0f;
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int col = t + 256 * m;
        const size_t idx = base + col;
        const float gated = s11_bf16(__fmul_rn(ffn_shexp[idx], sig_g));
        const float sum1  = s11_bf16(__fadd_rn(moe_out[idx], gated));
        const float v     = s11_bf16(__fadd_rn(sum1, ffn_residual[idx]));
        x_sh[col] = v;
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(v, v));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
    if ((t & 31) == 0) {
        warp_sums[t / 32] = sum_sq;
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int col = t + 256 * m;
        res_dst[base + col] = x_sh[col];
    }
    if (t < 32) {
        float total = (t < 8) ? warp_sums[t] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
        }
        if (t == 0) {
            warp_sums[0] = rsqrtf(total / 2048.0f + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];
    if (q_out && s_out) {
        const int warp = t / 32;
        const int lane = t % 32;
        #pragma unroll
        for (int g_sub = 0; g_sub < 2; ++g_sub) {
            const int group = warp * 2 + g_sub;
            float values[4];
            float maximum = 1e-5f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = group * 128 + lane + 32 * i;
                const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
                const float val = s11_bf16(__fmul_rn(__fmul_rn(x_sh[col], inv_rms), w));
                norm_dst[base + col] = val;
                values[i] = val;
                maximum = fmaxf(maximum, fabsf(val));
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
            }
            const float inverse = __fmul_rn(__fdiv_rn(1.0f, maximum), 127.0f);
            if (lane == 0) {
                s_out[size_t(row) * 16 + group] = __fdiv_rn(1.0f, inverse);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = group * 128 + lane + 32 * i;
                const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(values[i], inverse))));
                q_out[base + col] = static_cast<signed char>(quant);
            }
        }
    } else {
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int col = t + 256 * m;
            const float w = __fadd_rn(__fmul_rn(__bfloat162float(mw[col]), scale), bias);
            norm_dst[base + col] = s11_bf16(__fmul_rn(__fmul_rn(x_sh[col], inv_rms), w));
        }
    }
}

static __global__ void __launch_bounds__(128) s11_ssm_conv_silu_l2norm_fused_kernel(
        const float * __restrict__ src0, const nv_bfloat16 * __restrict__ w_bf16,
        int src0_nb1, int src0_nb2,
        float * __restrict__ conv_mix_dst,
        float * __restrict__ q_dst,
        float * __restrict__ k_dst,
        float eps, float q_scale, float k_scale, int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x; // sequence
    const int bidy = blockIdx.y; // 0..63 (128 channels each)
    const float * x_block = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src0) + size_t(bidx) * src0_nb2 + size_t(bidy) * 128 * src0_nb1);
    const nv_bfloat16 * w_block = w_bf16 + size_t(bidy) * 128 * 4;
    const int stride_x = src0_nb1 / sizeof(float);

    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float w[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = __bfloat162float(w_block[tid * 4 + j]);
    }
    __shared__ float warp_sums[4];

    for (int64_t i = 0; i < n_t; ++i) {
        if (i == 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) & 3] = x_block[tid * stride_x + i + 3];
        }
        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf = __fadd_rn(sumf, __fmul_rn(x[(i + j) & 3], w[j]));
        }
        const float silu_v = sumf / (1.0f + expf(-sumf));
        const float v = s11_bf16(silu_v);
        const size_t token = size_t(bidx) * size_t(n_t) + size_t(i);
        conv_mix_dst[token * 8192 + size_t(bidy) * 128 + tid] = v;
        if (bidy < 32) {
            float sum_sq = __fmul_rn(v, v);
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
            }
            if ((tid & 31) == 0) {
                warp_sums[tid / 32] = sum_sq;
            }
            __syncthreads();
            if (tid < 32) {
                float total = (tid < 4) ? warp_sums[tid] : 0.0f;
                for (int offset = 16; offset > 0; offset /= 2) {
                    total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
                }
                if (tid == 0) {
                    warp_sums[0] = rsqrtf(total / 128.0f + eps);
                }
            }
            __syncthreads();
            const float inv_rms = warp_sums[0];
            const float sc = (bidy < 16) ? q_scale : k_scale;
            const float out_val = s11_bf16(__fmul_rn(__fmul_rn(v, inv_rms), sc));
            if (bidy < 16) {
                q_dst[token * 2048 + size_t(bidy) * 128 + tid] = out_val;
            } else {
                k_dst[token * 2048 + size_t(bidy - 16) * 128 + tid] = out_val;
            }
            __syncthreads();
        }
    }
}

static __global__ void __launch_bounds__(128) s11_ssm_conv_long_silu_l2norm_fused_kernel(
        const float * __restrict__ src0, const nv_bfloat16 * __restrict__ w_bf16,
        int src0_nb0, int src0_nb1, int src0_nb2,
        float * __restrict__ conv_mix_dst,
        float * __restrict__ q_dst,
        float * __restrict__ k_dst,
        float eps, float q_scale, float k_scale, int64_t n_t) {
    constexpr int split_d_inner = 128;
    constexpr int d_conv = 4;
    constexpr int64_t split_n_t = 32;
    constexpr int n_cols = d_conv - 1 + split_n_t; // 35

    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src0) +
        size_t(bidx) * src0_nb2 + size_t(bidy) * split_d_inner * src0_nb1 + size_t(bidz) * split_n_t * src0_nb0);
    const nv_bfloat16 * w_block = w_bf16 + size_t(bidy) * split_d_inner * d_conv;
    const int stride_x = src0_nb1 / sizeof(float);
    const int64_t local_n_t = min(split_n_t, n_t - int64_t(bidz) * split_n_t);

    __shared__ float smem[split_d_inner * n_cols];
    __shared__ float warp_sums[4];

    constexpr int load_cols   = n_cols;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
    #pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }
        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    float w[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = __bfloat162float(w_block[tid * 4 + j]);
    }

    for (int64_t i = 0; i < local_n_t; ++i) {
        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf = __fadd_rn(sumf, __fmul_rn(smem[tid * n_cols + i + j], w[j]));
        }
        const float silu_v = sumf / (1.0f + expf(-sumf));
        const float v = s11_bf16(silu_v);
        const size_t token = size_t(bidx) * size_t(n_t) + size_t(bidz) * size_t(split_n_t) + size_t(i);
        conv_mix_dst[token * 8192 + size_t(bidy) * 128 + tid] = v;
        if (bidy < 32) {
            float sum_sq = __fmul_rn(v, v);
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
            }
            if ((tid & 31) == 0) {
                warp_sums[tid / 32] = sum_sq;
            }
            __syncthreads();
            if (tid < 32) {
                float total = (tid < 4) ? warp_sums[tid] : 0.0f;
                for (int offset = 16; offset > 0; offset /= 2) {
                    total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
                }
                if (tid == 0) {
                    warp_sums[0] = rsqrtf(total / 128.0f + eps);
                }
            }
            __syncthreads();
            const float inv_rms = warp_sums[0];
            const float sc = (bidy < 16) ? q_scale : k_scale;
            const float out_val = s11_bf16(__fmul_rn(__fmul_rn(v, inv_rms), sc));
            if (bidy < 16) {
                q_dst[token * 2048 + size_t(bidy) * 128 + tid] = out_val;
            } else {
                k_dst[token * 2048 + size_t(bidy - 16) * 128 + tid] = out_val;
            }
            __syncthreads();
        }
    }
}

static __global__ void __launch_bounds__(128) s11_gdn_conv_l2norm_epilogue_kernel(
        const float * __restrict__ conv_silu_raw,
        float * __restrict__ conv_mix_dst,
        float * __restrict__ q_dst,
        float * __restrict__ k_dst,
        float eps, float q_scale, float k_scale) {
    const int head_block = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t mix_idx = size_t(token) * 8192 + size_t(head_block) * 128 + tid;
    const float v = s11_bf16(conv_silu_raw[mix_idx]);
    conv_mix_dst[mix_idx] = v;
    if (head_block >= 32) {
        return;
    }
    float sum_sq = __fmul_rn(v, v);
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
    __shared__ float warp_sums[4];
    if ((tid & 31) == 0) {
        warp_sums[tid / 32] = sum_sq;
    }
    __syncthreads();
    if (tid < 32) {
        float total = (tid < 4) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
        }
        if (tid == 0) {
            warp_sums[0] = rsqrtf(total / 128.0f + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];
    const float sc = (head_block < 16) ? q_scale : k_scale;
    const float out_val = s11_bf16(__fmul_rn(__fmul_rn(v, inv_rms), sc));
    if (head_block < 16) {
        q_dst[size_t(token) * 2048 + size_t(head_block) * 128 + tid] = out_val;
    } else {
        k_dst[size_t(token) * 2048 + size_t(head_block - 16) * 128 + tid] = out_val;
    }
}

static __global__ void __launch_bounds__(256) s11_unary_round_bf16_f32_kernel(
        const float * __restrict__ src, float * __restrict__ dst, int64_t n, bool is_silu) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float x = src[idx];
        const float y = is_silu ? (x / (1.0f + expf(-x))) : (1.0f / (1.0f + expf(-x)));
        dst[idx] = s11_bf16(y);
    }
}

static __global__ void __launch_bounds__(256) s11_moe_reduce8_round_bf16_kernel(
        const char * __restrict__ down_exps, size_t nb1, size_t nb2, float * __restrict__ moe_out) {
    const int c0 = (blockIdx.x * 256 + threadIdx.x) * 4;
    const int token = blockIdx.y;
    const char * tok_base = down_exps + size_t(token) * nb2 + size_t(c0) * sizeof(float);
    float4 acc = *reinterpret_cast<const float4 *>(tok_base);
    #pragma unroll
    for (int slot = 1; slot < 8; ++slot) {
        const float4 val = *reinterpret_cast<const float4 *>(tok_base + size_t(slot) * nb1);
        acc.x = __fadd_rn(acc.x, val.x);
        acc.y = __fadd_rn(acc.y, val.y);
        acc.z = __fadd_rn(acc.z, val.z);
        acc.w = __fadd_rn(acc.w, val.w);
    }
    acc.x = s11_bf16(acc.x);
    acc.y = s11_bf16(acc.y);
    acc.z = s11_bf16(acc.z);
    acc.w = s11_bf16(acc.w);
    *reinterpret_cast<float4 *>(moe_out + size_t(token) * 2048 + c0) = acc;
}

static __global__ void __launch_bounds__(128) s11_norm_gated_fused_kernel(
        const float * __restrict__ input, const void * __restrict__ weights, int w_type,
        const float * __restrict__ gate, float * __restrict__ dst,
        signed char * __restrict__ q_out, float * __restrict__ s_out, float eps) {
    const int row = blockIdx.x;
    const int t = threadIdx.x; // 0..127 (k = 128)
    const float x = s11_bf16(input[size_t(row) * 128 + t]);
    float sum_sq = __fmul_rn(x, x);
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
    __shared__ float warp_sums[4];
    if ((t & 31) == 0) {
        warp_sums[t / 32] = sum_sq;
    }
    __syncthreads();
    if (t < 4) {
        float total = warp_sums[t];
        total = __fadd_rn(total, __shfl_down_sync(0x0000000f, total, 2));
        total = __fadd_rn(total, __shfl_down_sync(0x0000000f, total, 1));
        if (t == 0) {
            warp_sums[0] = rsqrtf(total * (1.0f / 128.0f) + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];
    const float w = (w_type == GGML_TYPE_BF16)
        ? __bfloat162float(static_cast<const nv_bfloat16 *>(weights)[t])
        : static_cast<const float *>(weights)[t];
    const float g = gate[size_t(row) * 128 + t];
    const float silu_g = g / (1.0f + expf(-g));
    const float norm_val = __fmul_rn(__fmul_rn(x, inv_rms), w);
    const float out_val = s11_bf16(__fmul_rn(norm_val, silu_g));
    dst[size_t(row) * 128 + t] = out_val;
    if (q_out && s_out) {
        __shared__ float warp_max[4];
        float maximum = fmaxf(1e-5f, fabsf(out_val));
        for (int offset = 16; offset > 0; offset /= 2) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        }
        if ((t & 31) == 0) {
            warp_max[t / 32] = maximum;
        }
        __syncthreads();
        if (t < 4) {
            float m = warp_max[t];
            m = fmaxf(m, __shfl_down_sync(0x0000000f, m, 2));
            m = fmaxf(m, __shfl_down_sync(0x0000000f, m, 1));
            if (t == 0) {
                warp_max[0] = m;
            }
        }
        __syncthreads();
        const float inverse = __fmul_rn(__fdiv_rn(1.0f, warp_max[0]), 127.0f);
        if (t == 0) {
            s_out[row] = __fdiv_rn(1.0f, inverse);
        }
        const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(out_val, inverse))));
        q_out[size_t(row) * 128 + t] = static_cast<signed char>(quant);
    }
}

static __global__ void __launch_bounds__(256) s11_gdn_gate_fused_kernel(
        const float * __restrict__ alpha, const void * __restrict__ dt_bias, int dt_type,
        const void * __restrict__ ssm_a, int a_type, float * __restrict__ dst,
        int heads, int total, bool round_bf16) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int h = idx % heads;
    const float db = (dt_type == GGML_TYPE_BF16)
        ? __bfloat162float(static_cast<const nv_bfloat16 *>(dt_bias)[h])
        : static_cast<const float *>(dt_bias)[h];
    const float a_val = (a_type == GGML_TYPE_BF16)
        ? __bfloat162float(static_cast<const nv_bfloat16 *>(ssm_a)[h])
        : static_cast<const float *>(ssm_a)[h];
    const float x = alpha[idx] + db;
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    const float sp_val = round_bf16 ? s11_bf16(sp) : sp;
    const float decay = -expf(a_val);
    const float prod = sp_val * decay;
    dst[idx] = round_bf16 ? s11_bf16(prod) : prod;
}

static __global__ void __launch_bounds__(256) s11_round_bf16_f32_kernel(
        const float * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = s11_bf16(src[idx]);
    }
}

static __global__ void __launch_bounds__(256) s11_add_round_bf16_f32_kernel(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst, int64_t n) {
    if (gridDim.x == 1 && n == 2048) {
        const int t = threadIdx.x;
        float out[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int idx = t + m * 256;
            out[m] = s11_bf16(__fadd_rn(a[idx], b[idx]));
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            dst[t + m * 256] = out[m];
        }
        return;
    }
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = s11_bf16(__fadd_rn(a[idx], b[idx]));
    }
}

static __global__ void __launch_bounds__(256) s11_shexp_gate_epilogue_fused_kernel(
        const nv_bfloat16 * __restrict__ gate_w,
        const float * __restrict__ attn_post_norm,
        const float * __restrict__ ffn_shexp,
        const float * __restrict__ moe_out,
        const float * __restrict__ ffn_residual,
        float * __restrict__ dst) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const size_t base = size_t(row) * 2048;
    __shared__ float raw_g_sh;

    if (t < 32) {
        float sum = 0.0f;
        for (int index = t; index < 2048; index += 32) {
            sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(gate_w[index]), attn_post_norm[base + index]));
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        if (t == 0) {
            raw_g_sh = sum;
        }
    }
    __syncthreads();
    const float sig_g = s11_bf16(1.0f / (1.0f + expf(-raw_g_sh)));
    float out[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int idx = t + m * 256;
        const float gated = s11_bf16(__fmul_rn(ffn_shexp[base + idx], sig_g));
        const float sum1  = s11_bf16(__fadd_rn(moe_out[base + idx], gated));
        out[m]            = s11_bf16(__fadd_rn(sum1, ffn_residual[base + idx]));
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        dst[base + t + m * 256] = out[m];
    }
}

template <int COLS>
static __global__ void __launch_bounds__(128) s11_gdn_alpha_beta_gate_fused_kernel(
        const nv_bfloat16 * __restrict__ w_alpha, const nv_bfloat16 * __restrict__ w_beta,
        const float * __restrict__ x, const void * __restrict__ dt_bias, int dt_type,
        const void * __restrict__ ssm_a, int a_type,
        float * __restrict__ gate_dst, float * __restrict__ beta_dst, int tokens) {
    const int lane = threadIdx.x % 32;
    const bool is_beta = (blockIdx.x >= 8);
    const int row = (is_beta ? (blockIdx.x - 8) : blockIdx.x) * 4 + threadIdx.x / 32;
    const int first_col = blockIdx.y * COLS;
    const nv_bfloat16 * w = (is_beta ? w_beta : w_alpha) + size_t(row) * 2048;

    __shared__ __align__(16) float x_sh[COLS][512];
    float sums[COLS] = {};

    if constexpr (COLS == 1) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x + size_t(first_col) * 2048);
        #pragma unroll
        for (int base_k = 0; base_k < 2048; base_k += 512) {
            reinterpret_cast<float4 *>(x_sh[0])[threadIdx.x] = x4[(base_k / 4) + threadIdx.x];
            __syncthreads();
            if (row < 32) {
                for (int idx = lane; idx < 512; idx += 32) {
                    sums[0] = __fadd_rn(sums[0], __fmul_rn(__bfloat162float(w[base_k + idx]), x_sh[0][idx]));
                }
            }
            __syncthreads();
        }
    } else {
        #pragma unroll
        for (int base_k = 0; base_k < 2048; base_k += 512) {
            const int total_f4 = COLS * 128;
            #pragma unroll
            for (int idx = threadIdx.x; idx < total_f4; idx += 128) {
                const int c = idx / 128;
                const int k4 = idx % 128;
                const int col = first_col + c;
                if (col < tokens) {
                    const float4 * x4 = reinterpret_cast<const float4 *>(x + size_t(col) * 2048 + base_k);
                    reinterpret_cast<float4 *>(x_sh[c])[k4] = x4[k4];
                } else {
                    reinterpret_cast<float4 *>(x_sh[c])[k4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            __syncthreads();

            if (row < 32) {
                #pragma unroll 8
                for (int idx = lane; idx < 512; idx += 32) {
                    const float w_val = __bfloat162float(w[base_k + idx]);
                    #pragma unroll
                    for (int c = 0; c < COLS; ++c) {
                        sums[c] = __fadd_rn(sums[c], __fmul_rn(w_val, x_sh[c][idx]));
                    }
                }
            }
            __syncthreads();
        }
    }
    if (row >= 32) {
        return;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            sums[c] = __fadd_rn(sums[c], __shfl_down_sync(0xffffffff, sums[c], offset));
        }
    }
    if (lane == 0) {
        if (!is_beta) {
            const float db = (dt_type == GGML_TYPE_BF16)
                ? __bfloat162float(static_cast<const nv_bfloat16 *>(dt_bias)[row])
                : static_cast<const float *>(dt_bias)[row];
            const float a_val = (a_type == GGML_TYPE_BF16)
                ? __bfloat162float(static_cast<const nv_bfloat16 *>(ssm_a)[row])
                : static_cast<const float *>(ssm_a)[row];
            const float decay = -expf(a_val);
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int tok = first_col + c;
                if (tok < tokens) {
                    const float v = sums[c] + db;
                    const float sp = (v > 20.0f) ? v : logf(1.0f + expf(v));
                    gate_dst[size_t(tok) * 32 + row] = sp * decay;
                }
            }
        } else {
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int tok = first_col + c;
                if (tok < tokens) {
                    const float sig = 1.0f / (1.0f + expf(-sums[c]));
                    beta_dst[size_t(tok) * 32 + row] = s11_bf16(sig);
                }
            }
        }
    }
}

static __global__ void __launch_bounds__(256) s11_fa_kv_round_set_rows_f16_kernel(
        const float * __restrict__ k_cur, const float * __restrict__ v_cur,
        float * __restrict__ k_round_dst,
        char * __restrict__ k_cache, char * __restrict__ v_cache,
        const int64_t * __restrict__ k_idxs, const int64_t * __restrict__ v_idxs,
        size_t k_cache_nb1, size_t v_cache_nb1) {
    const int is_v = blockIdx.x;
    const int token = blockIdx.y;
    const int t = threadIdx.x; // 0..255, 2 elements per thread = 512 elements per token
    const int c0 = t * 2;
    const size_t src_idx = size_t(token) * 512 + c0;
    if (is_v == 0) {
        const float x0 = s11_bf16(k_cur[src_idx + 0]);
        const float x1 = s11_bf16(k_cur[src_idx + 1]);
        if (k_round_dst) {
            k_round_dst[src_idx + 0] = x0;
            k_round_dst[src_idx + 1] = x1;
        }
        const int64_t dst_row = k_idxs[token];
        half2 * dst_h2 = reinterpret_cast<half2 *>(k_cache + size_t(dst_row) * k_cache_nb1 + size_t(c0) * sizeof(half));
        *dst_h2 = __floats2half2_rn(x0, x1);
    } else {
        const float x0 = v_cur[src_idx + 0];
        const float x1 = v_cur[src_idx + 1];
        const int64_t dst_row = v_idxs[token];
        half2 * dst_h2 = reinterpret_cast<half2 *>(v_cache + size_t(dst_row) * v_cache_nb1 + size_t(c0) * sizeof(half));
        *dst_h2 = __floats2half2_rn(x0, x1);
    }
}

static __global__ void __launch_bounds__(128) s11_fa_gate_strided_quant_fused_kernel(
        const float * __restrict__ cur, const char * __restrict__ qcur_full_base,
        size_t gate_nb1, size_t gate_nb2, size_t gate_offset,
        float * __restrict__ dst, signed char * __restrict__ q_out, float * __restrict__ s_out) {
    const int group_row = blockIdx.x; // 0..(tokens * 32 - 1), each row is 128 elements
    const int t = threadIdx.x; // 0..127
    const size_t idx = size_t(group_row) * 128 + t;
    const int token = group_row >> 5;
    const int col = ((group_row & 31) << 7) + t; // 0..4095
    const int head = col >> 8; // 0..15 (n_embd_head = 256)
    const int dim = col & 255; // 0..255
    const float g = *reinterpret_cast<const float *>(
        qcur_full_base + gate_offset + size_t(token) * gate_nb2 + size_t(head) * gate_nb1 + size_t(dim) * sizeof(float));
    const float sig_g = s11_bf16(1.0f / (1.0f + expf(-g)));
    const float c = s11_bf16(cur[idx]);
    const float out_val = __fmul_rn(c, sig_g);
    dst[idx] = out_val;
    if (q_out && s_out) {
        __shared__ float warp_max[4];
        float maximum = fmaxf(1e-5f, fabsf(out_val));
        for (int offset = 16; offset > 0; offset /= 2) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        }
        if ((t & 31) == 0) {
            warp_max[t / 32] = maximum;
        }
        __syncthreads();
        if (t < 4) {
            float m = warp_max[t];
            m = fmaxf(m, __shfl_down_sync(0x0000000f, m, 2));
            m = fmaxf(m, __shfl_down_sync(0x0000000f, m, 1));
            if (t == 0) {
                warp_max[0] = m;
            }
        }
        __syncthreads();
        const float inverse = __fmul_rn(__fdiv_rn(1.0f, warp_max[0]), 127.0f);
        if (t == 0) {
            s_out[group_row] = __fdiv_rn(1.0f, inverse);
        }
        const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(out_val, inverse))));
        q_out[idx] = static_cast<signed char>(quant);
    }
}

static __global__ void __launch_bounds__(128) s11_fa_gate_quant_fused_kernel(
        const float * __restrict__ cur, const float * __restrict__ gate,
        float * __restrict__ dst, signed char * __restrict__ q_out, float * __restrict__ s_out) {
    const int group_row = blockIdx.x; // 0..(tokens * 32 - 1), each row is 128 elements
    const int t = threadIdx.x; // 0..127
    const size_t idx = size_t(group_row) * 128 + t;
    const float g = gate[idx];
    const float sig_g = s11_bf16(1.0f / (1.0f + expf(-g)));
    const float c = s11_bf16(cur[idx]);
    const float out_val = __fmul_rn(c, sig_g);
    dst[idx] = out_val;
    if (q_out && s_out) {
        __shared__ float warp_max[4];
        float maximum = fmaxf(1e-5f, fabsf(out_val));
        for (int offset = 16; offset > 0; offset /= 2) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        }
        if ((t & 31) == 0) {
            warp_max[t / 32] = maximum;
        }
        __syncthreads();
        if (t < 4) {
            float m = warp_max[t];
            m = fmaxf(m, __shfl_down_sync(0x0000000f, m, 2));
            m = fmaxf(m, __shfl_down_sync(0x0000000f, m, 1));
            if (t == 0) {
                warp_max[0] = m;
            }
        }
        __syncthreads();
        const float inverse = __fmul_rn(__fdiv_rn(1.0f, warp_max[0]), 127.0f);
        if (t == 0) {
            s_out[group_row] = __fdiv_rn(1.0f, inverse);
        }
        const int quant = max(-128, min(127, __float2int_rn(__fmul_rn(out_val, inverse))));
        q_out[idx] = static_cast<signed char>(quant);
    }
}

static __global__ void __launch_bounds__(256) s11_shexp_epilogue_fused_kernel(
        const float * __restrict__ shared_gate,
        const float * __restrict__ ffn_shexp,
        const float * __restrict__ moe_out,
        const float * __restrict__ ffn_residual,
        float * __restrict__ dst,
        int k, int64_t total) {
    if (gridDim.x == 1 && k == 2048 && total == 2048) {
        const int t = threadIdx.x;
        const float raw_g = shared_gate[0];
        const float sig_g = s11_bf16(1.0f / (1.0f + expf(-raw_g)));
        float out[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int idx = t + m * 256;
            const float gated = s11_bf16(__fmul_rn(ffn_shexp[idx], sig_g));
            const float sum1  = s11_bf16(__fadd_rn(moe_out[idx], gated));
            out[m]            = s11_bf16(__fadd_rn(sum1, ffn_residual[idx]));
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            dst[t + m * 256] = out[m];
        }
        return;
    }
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int token = int(idx / k);
        const float raw_g = shared_gate[token];
        const float sig_g = s11_bf16(1.0f / (1.0f + expf(-raw_g)));
        const float gated = s11_bf16(__fmul_rn(ffn_shexp[idx], sig_g));
        const float sum1  = s11_bf16(__fadd_rn(moe_out[idx], gated));
        dst[idx]          = s11_bf16(__fadd_rn(sum1, ffn_residual[idx]));
    }
}

static __global__ void __launch_bounds__(128) s11_routed_up_gate_swiglu_wmma_k128_kernel(
        const char * __restrict__ up_weights, const char * __restrict__ gate_weights,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        const int32_t * __restrict__ tile_experts, const int32_t * __restrict__ tile_starts,
        const int32_t * __restrict__ tile_counts, const int32_t * __restrict__ sorted_assignments,
        const int32_t * __restrict__ num_tiles_in, const char * __restrict__ prob_weights,
        char * __restrict__ output,
        int k, int n, int slots, int input_slots,
        size_t weight_row, size_t weight_expert,
        size_t prob_slot, size_t prob_token,
        size_t output_slot, size_t output_token) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int tile = blockIdx.y;
    if (tile >= num_tiles_in[0]) {
        return;
    }
    const int expert = tile_experts[tile];
    const int start = tile_starts[tile];
    const int count = tile_counts[tile];
    const int t = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int first_row = blockIdx.x * 64;
    if (first_row >= n || count <= 0) {
        return;
    }

    constexpr int kStride = 144;
    __shared__ __align__(32) signed char a_sh[16 * kStride];
    __shared__ __align__(32) signed char b_up_sh[64 * kStride];
    __shared__ __align__(32) signed char b_gate_sh[64 * kStride];
    __shared__ __align__(32) int dots_sh[4][16 * 16];
    __shared__ int assign_sh[16];
    __shared__ int source_sh[16];
    __shared__ float a_scales_sh[16];
    __shared__ float w_up_scales_sh[64];
    __shared__ float w_gate_scales_sh[64];

    if (t < 16) {
        const int a = (t < count) ? sorted_assignments[start + t] : 0;
        assign_sh[t] = a;
        const int slot = a % slots;
        const int token = a / slots;
        source_sh[t] = token * input_slots + (input_slots == 1 ? 0 : slot);
    }
    __syncthreads();

    const char * exp_up   = up_weights   + size_t(expert) * weight_expert;
    const char * exp_gate = gate_weights + size_t(expert) * weight_expert;
    float sums_up[8]   = {};
    float sums_gate[8] = {};

    for (int group = 0; group < k / 128; ++group) {
        if (t < 16) {
            a_scales_sh[t] = (t < count) ? activation_scales[size_t(source_sh[t]) * (k / 128) + group] : 0.0f;
        }
        #pragma unroll
        for (int idx = t; idx < 16 * 32; idx += 128) {
            const int tok_local = idx / 32;
            const int col4 = idx % 32;
            *reinterpret_cast<int *>(&a_sh[tok_local * kStride + col4 * 4]) = (tok_local < count)
                ? *reinterpret_cast<const int *>(q + size_t(source_sh[tok_local]) * k + group * 128 + col4 * 4)
                : 0;
        }
        if (t < 64) {
            const int r = first_row + t;
            if (r < n) {
                w_up_scales_sh[t] = s11_scale(reinterpret_cast<const unsigned char *>(exp_up + size_t(r) * weight_row + group * 36), GGML_TYPE_S11_W2);
                w_gate_scales_sh[t] = s11_scale(reinterpret_cast<const unsigned char *>(exp_gate + size_t(r) * weight_row + group * 36), GGML_TYPE_S11_W2);
            } else {
                w_up_scales_sh[t] = 0.0f;
                w_gate_scales_sh[t] = 0.0f;
            }
        }
        #pragma unroll
        for (int idx = t; idx < 64 * 32; idx += 128) {
            const int r_local = idx / 32;
            const int col4 = idx % 32;
            const int r = first_row + r_local;
            int packed_up = 0;
            int packed_gate = 0;
            if (r < n) {
                packed_up = s11_unpack4_w2(reinterpret_cast<const unsigned char *>(exp_up + size_t(r) * weight_row + group * 36), col4);
                packed_gate = s11_unpack4_w2(reinterpret_cast<const unsigned char *>(exp_gate + size_t(r) * weight_row + group * 36), col4);
            }
            *reinterpret_cast<int *>(&b_up_sh[r_local * kStride + col4 * 4]) = packed_up;
            *reinterpret_cast<int *>(&b_gate_sh[r_local * kStride + col4 * 4]) = packed_gate;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_up;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_gate;
        wmma::fill_fragment(acc_up, 0);
        wmma::fill_fragment(acc_gate, 0);
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf_up;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf_gate;
            wmma::load_matrix_sync(af, a_sh + ks, kStride);
            wmma::load_matrix_sync(bf_up, b_up_sh + (warp * 16) * kStride + ks, kStride);
            wmma::load_matrix_sync(bf_gate, b_gate_sh + (warp * 16) * kStride + ks, kStride);
            wmma::mma_sync(acc_up, af, bf_up, acc_up);
            wmma::mma_sync(acc_gate, af, bf_gate, acc_gate);
        }
        wmma::store_matrix_sync(dots_sh[warp], acc_up, 16, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok_local = idx / 16;
            const int row_in_warp = idx % 16;
            const int dot = dots_sh[warp][idx];
            const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_up_scales_sh[warp * 16 + row_in_warp]);
            sums_up[p] = __fadd_rn(sums_up[p], __fmul_rn(float(dot), scale_prod));
        }
        __syncwarp();

        wmma::store_matrix_sync(dots_sh[warp], acc_gate, 16, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int idx = lane + p * 32;
            const int tok_local = idx / 16;
            const int row_in_warp = idx % 16;
            const int dot = dots_sh[warp][idx];
            const float scale_prod = __fmul_rn(a_scales_sh[tok_local], w_gate_scales_sh[warp * 16 + row_in_warp]);
            sums_gate[p] = __fadd_rn(sums_gate[p], __fmul_rn(float(dot), scale_prod));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = lane + p * 32;
        const int tok_local = idx / 16;
        const int row = first_row + warp * 16 + idx % 16;
        if (tok_local < count && row < n) {
            const int a = assign_sh[tok_local];
            const int slot = a % slots;
            const int token = a / slots;
            const float u = s11_bf16(sums_up[p]);
            const float g = s11_bf16(sums_gate[p]);
            const float silu_g = g / (1.0f + expf(-g));
            float val = silu_g * u;
            if (prob_weights) {
                const float prob = *reinterpret_cast<const float *>(prob_weights + size_t(slot) * prob_slot + size_t(token) * prob_token);
                val = s11_bf16(val * prob);
            }
            *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(slot) * output_slot + size_t(token) * output_token) = val;
        }
    }
#endif
}

template <int TYPE>
static __global__ void __launch_bounds__(256) s11_fused_up_gate_swiglu_kernel_m1(
        const char * __restrict__ up_weights, const char * __restrict__ gate_weights,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        const char * __restrict__ ids, const char * __restrict__ prob_weights,
        char * __restrict__ output,
        int k, int n, int slots, int input_slots,
        size_t weight_row, size_t weight_expert,
        size_t ids_slot, size_t prob_slot, size_t output_slot) {
    __shared__ __align__(16) signed char q_sh[2048];
    __shared__ float scales_sh[16];
    __shared__ int expert_sh;
    __shared__ float prob_sh;
    const int t = threadIdx.x;
    const int slot = (TYPE == GGML_TYPE_S11_W2) ? int(blockIdx.y) : 0;
    const int source = (TYPE == GGML_TYPE_S11_W2 && input_slots > 1) ? slot : 0;
    if (t == 0) {
        expert_sh = ids ? *reinterpret_cast<const int *>(ids + size_t(slot) * ids_slot) : 0;
        prob_sh = prob_weights ? *reinterpret_cast<const float *>(prob_weights + size_t(slot) * prob_slot) : 1.0f;
    }
    if (t < 16) {
        scales_sh[t] = activation_scales[size_t(source) * 16 + t];
    }
    const int4 * q4_in = reinterpret_cast<const int4 *>(q + size_t(source) * 2048);
    int4 * q4_sh = reinterpret_cast<int4 *>(q_sh);
    if (t < 128) {
        q4_sh[t] = q4_in[t];
    }
    __syncthreads();

    const int lane = t % 32;
    const int row = blockIdx.x * 8 + t / 32;
    if (row >= n) {
        return;
    }
    const int expert = expert_sh;
    if (expert < 0) {
        if (!lane) {
            *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(slot) * output_slot) = 0.0f;
        }
        return;
    }
    const char * w_up = up_weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
    const char * w_gate = gate_weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
    float sum_up = 0.0f;
    float sum_gate = 0.0f;

    const int quarter = lane / 8;
    const int sub = lane % 8;
    #pragma unroll
    for (int g4 = 0; g4 < 4; ++g4) {
        const int group = g4 * 4 + quarter;
        const auto * block_up = reinterpret_cast<const unsigned char *>(w_up + group * 36);
        const auto * block_gate = reinterpret_cast<const unsigned char *>(w_gate + group * 36);
        const float scale_up = s11_scale(block_up, TYPE);
        const float scale_gate = s11_scale(block_gate, TYPE);
        const uint32_t u32 = *reinterpret_cast<const uint32_t *>(block_up + sub * 4);
        const uint32_t g32 = *reinterpret_cast<const uint32_t *>(block_gate + sub * 4);
        const int4 act4 = *reinterpret_cast<const int4 *>(q_sh + group * 128 + sub * 16);
        int dot_up = __dp4a(s11_unpack4_w2_byte(u32 & 0xffu), act4.x, 0);
        dot_up = __dp4a(s11_unpack4_w2_byte((u32 >> 8) & 0xffu), act4.y, dot_up);
        dot_up = __dp4a(s11_unpack4_w2_byte((u32 >> 16) & 0xffu), act4.z, dot_up);
        dot_up = __dp4a(s11_unpack4_w2_byte(u32 >> 24), act4.w, dot_up);
        int dot_gate = __dp4a(s11_unpack4_w2_byte(g32 & 0xffu), act4.x, 0);
        dot_gate = __dp4a(s11_unpack4_w2_byte((g32 >> 8) & 0xffu), act4.y, dot_gate);
        dot_gate = __dp4a(s11_unpack4_w2_byte((g32 >> 16) & 0xffu), act4.z, dot_gate);
        dot_gate = __dp4a(s11_unpack4_w2_byte(g32 >> 24), act4.w, dot_gate);
        const float act_s = scales_sh[group];
        sum_up = __fadd_rn(sum_up, __fmul_rn(float(dot_up), __fmul_rn(act_s, scale_up)));
        sum_gate = __fadd_rn(sum_gate, __fmul_rn(float(dot_gate), __fmul_rn(act_s, scale_gate)));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_up = __fadd_rn(sum_up, __shfl_down_sync(0xffffffff, sum_up, offset));
        sum_gate = __fadd_rn(sum_gate, __shfl_down_sync(0xffffffff, sum_gate, offset));
    }
    if (!lane) {
        const float u = s11_bf16(sum_up);
        const float g = s11_bf16(sum_gate);
        const float silu_g = g / (1.0f + expf(-g));
        float val = silu_g * u;
        if (prob_weights) {
            val = s11_bf16(val * prob_sh);
        }
        *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(slot) * output_slot) = val;
    }
}

static __global__ void __launch_bounds__(256) s11_fused_down_exps_reduce_kernel_m1(
        const char * __restrict__ down_weights, const signed char * __restrict__ q,
        const float * __restrict__ activation_scales, const char * __restrict__ ids,
        char * __restrict__ output, int n, int slots,
        size_t weight_row, size_t weight_expert, size_t ids_slot) {
    __shared__ __align__(16) signed char q_sh[4096];
    __shared__ float scales_sh[32];
    __shared__ int ids_sh[8];
    const int t = threadIdx.x;
    if (t < 8) {
        ids_sh[t] = *reinterpret_cast<const int *>(ids + size_t(t) * ids_slot);
    }
    if (t < 32) {
        scales_sh[t] = activation_scales[t];
    }
    const int4 * q4_in = reinterpret_cast<const int4 *>(q);
    int4 * q4_sh = reinterpret_cast<int4 *>(q_sh);
    q4_sh[t] = q4_in[t];
    __syncthreads();

    const int lane = t % 32;
    const int row = blockIdx.x * 8 + t / 32;
    if (row >= n) {
        return;
    }
    if (row == 0 && lane == 0 && n == 2048) {
        atomicAdd(&s11_completed_assignments, static_cast<unsigned long long>(slots));
    }
    const int group = lane / 8;
    const int sub = lane % 8;
    float total_sum = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int expert = ids_sh[slot];
        if (expert < 0) continue;
        const char * w_down = down_weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
        const auto * block = reinterpret_cast<const unsigned char *>(w_down + group * 36);
        const float scale = s11_scale(block, GGML_TYPE_S11_W2);
        const uint32_t w32 = *reinterpret_cast<const uint32_t *>(block + sub * 4);
        const int4 act4 = *reinterpret_cast<const int4 *>(q_sh + size_t(slot) * 512 + group * 128 + sub * 16);
        int dot = __dp4a(s11_unpack4_w2_byte(w32 & 0xffu), act4.x, 0);
        dot = __dp4a(s11_unpack4_w2_byte((w32 >> 8) & 0xffu), act4.y, dot);
        dot = __dp4a(s11_unpack4_w2_byte((w32 >> 16) & 0xffu), act4.z, dot);
        dot = __dp4a(s11_unpack4_w2_byte(w32 >> 24), act4.w, dot);
        const float act_s = scales_sh[slot * 4 + group];
        float sum_s = __fmul_rn(float(dot), __fmul_rn(act_s, scale));
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_s = __fadd_rn(sum_s, __shfl_down_sync(0xffffffff, sum_s, offset));
        }
        if (!lane) {
            total_sum = __fadd_rn(total_sum, s11_bf16(sum_s));
        }
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + size_t(row) * sizeof(float)) = s11_bf16(total_sum);
    }
}

template <int TYPE>
static __global__ void __launch_bounds__(128) s11_fused_up_gate_swiglu_kernel(
        const char * __restrict__ up_weights, const char * __restrict__ gate_weights,
        const signed char * __restrict__ q, const float * __restrict__ activation_scales,
        const char * __restrict__ ids, const char * __restrict__ prob_weights,
        char * __restrict__ output,
        int k, int n, int slots, int tokens, int input_slots,
        size_t weight_row, size_t weight_expert,
        size_t ids_slot, size_t ids_token,
        size_t prob_slot, size_t prob_token,
        size_t output_slot, size_t output_token) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= n) {
        return;
    }
    const int assignment = blockIdx.y;
    const int slot = (TYPE == GGML_TYPE_S11_W2) ? (assignment % slots) : 0;
    const int token = (TYPE == GGML_TYPE_S11_W2) ? (assignment / slots) : assignment;
    if (token >= tokens) {
        return;
    }
    const int expert = ids ? *reinterpret_cast<const int *>(ids + slot * ids_slot + token * ids_token) : 0;
    if (expert < 0) {
        if (!lane) {
            *reinterpret_cast<float *>(output + row * sizeof(float) + slot * output_slot + token * output_token) = 0.0f;
        }
        return;
    }
    const int source = (TYPE == GGML_TYPE_S11_W2) ? (token * input_slots + (input_slots == 1 ? 0 : slot)) : token;
    const char * w_up = up_weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
    const char * w_gate = gate_weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
    float sum_up = 0.0f;
    float sum_gate = 0.0f;

    const int quarter = lane / 8;
    const int sub = lane % 8;
    #pragma unroll
    for (int g4 = 0; g4 < k / 512; ++g4) {
        const int group = g4 * 4 + quarter;
        const auto * block_up = reinterpret_cast<const unsigned char *>(w_up + group * 36);
        const auto * block_gate = reinterpret_cast<const unsigned char *>(w_gate + group * 36);
        const float scale_up = s11_scale(block_up, TYPE);
        const float scale_gate = s11_scale(block_gate, TYPE);
        const uint32_t u32 = *reinterpret_cast<const uint32_t *>(block_up + sub * 4);
        const uint32_t g32 = *reinterpret_cast<const uint32_t *>(block_gate + sub * 4);
        const int4 act4 = *reinterpret_cast<const int4 *>(q + size_t(source) * k + group * 128 + sub * 16);
        int dot_up = __dp4a(s11_unpack4_w2_byte(u32 & 0xffu), act4.x, 0);
        dot_up = __dp4a(s11_unpack4_w2_byte((u32 >> 8) & 0xffu), act4.y, dot_up);
        dot_up = __dp4a(s11_unpack4_w2_byte((u32 >> 16) & 0xffu), act4.z, dot_up);
        dot_up = __dp4a(s11_unpack4_w2_byte(u32 >> 24), act4.w, dot_up);
        int dot_gate = __dp4a(s11_unpack4_w2_byte(g32 & 0xffu), act4.x, 0);
        dot_gate = __dp4a(s11_unpack4_w2_byte((g32 >> 8) & 0xffu), act4.y, dot_gate);
        dot_gate = __dp4a(s11_unpack4_w2_byte((g32 >> 16) & 0xffu), act4.z, dot_gate);
        dot_gate = __dp4a(s11_unpack4_w2_byte(g32 >> 24), act4.w, dot_gate);
        const float act_s = activation_scales[size_t(source) * (k / 128) + group];
        sum_up = __fadd_rn(sum_up, __fmul_rn(float(dot_up), __fmul_rn(act_s, scale_up)));
        sum_gate = __fadd_rn(sum_gate, __fmul_rn(float(dot_gate), __fmul_rn(act_s, scale_gate)));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_up = __fadd_rn(sum_up, __shfl_down_sync(0xffffffff, sum_up, offset));
        sum_gate = __fadd_rn(sum_gate, __shfl_down_sync(0xffffffff, sum_gate, offset));
    }
    if (!lane) {
        const float u = s11_bf16(sum_up);
        const float g = s11_bf16(sum_gate);
        const float silu_g = g / (1.0f + expf(-g));
        float val = silu_g * u;
        if (prob_weights) {
            const float prob = *reinterpret_cast<const float *>(prob_weights + size_t(slot) * prob_slot + size_t(token) * prob_token);
            val = s11_bf16(val * prob);
        }
        *reinterpret_cast<float *>(output + row * sizeof(float) + slot * output_slot + token * output_token) = val;
    }
}

static __global__ void __launch_bounds__(128) s11_fused_down_exps_reduce_kernel(
        const char * __restrict__ down_weights, const signed char * __restrict__ q,
        const float * __restrict__ activation_scales, const char * __restrict__ ids,
        char * __restrict__ output, int k, int n, int slots, int tokens,
        size_t weight_row, size_t weight_expert, size_t ids_slot, size_t ids_token, size_t output_token) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int token = blockIdx.y;
    if (row >= n || token >= tokens) {
        return;
    }
    if (row == 0 && lane == 0 && k == 512 && n == 2048) {
        atomicAdd(&s11_completed_assignments, static_cast<unsigned long long>(slots));
    }
    const int group = lane / 8;
    const int sub = lane % 8;
    float total_sum = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int expert = *reinterpret_cast<const int *>(ids + size_t(slot) * ids_slot + size_t(token) * ids_token);
        if (expert < 0) continue;
        const int source = token * slots + slot;
        const char * w_down = down_weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
        const auto * block = reinterpret_cast<const unsigned char *>(w_down + group * 36);
        const float scale = s11_scale(block, GGML_TYPE_S11_W2);
        const uint32_t w32 = *reinterpret_cast<const uint32_t *>(block + sub * 4);
        const int4 act4 = *reinterpret_cast<const int4 *>(q + size_t(source) * k + group * 128 + sub * 16);
        int dot = __dp4a(s11_unpack4_w2_byte(w32 & 0xffu), act4.x, 0);
        dot = __dp4a(s11_unpack4_w2_byte((w32 >> 8) & 0xffu), act4.y, dot);
        dot = __dp4a(s11_unpack4_w2_byte((w32 >> 16) & 0xffu), act4.z, dot);
        dot = __dp4a(s11_unpack4_w2_byte(w32 >> 24), act4.w, dot);
        const float act_s = activation_scales[size_t(source) * 4 + group];
        float sum_s = __fmul_rn(float(dot), __fmul_rn(act_s, scale));
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_s = __fadd_rn(sum_s, __shfl_down_sync(0xffffffff, sum_s, offset));
        }
        if (!lane) {
            total_sum = __fadd_rn(total_sum, s11_bf16(sum_s));
        }
    }
    if (!lane) {
        *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(token) * output_token) = s11_bf16(total_sum);
    }
}

static bool s11_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b->data);
    const uintptr_t b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

static __global__ void __launch_bounds__(32) s11_copy_ids_kernel(
        const char * __restrict__ ids, int32_t * __restrict__ dst_ids,
        int slots, int tokens, size_t ids_slot, size_t ids_token) {
    const int idx = threadIdx.x;
    const int total = slots * tokens;
    for (int i = idx; i < total; i += 32) {
        const int slot = i % slots;
        const int token = i / slots;
        dst_ids[i] = *reinterpret_cast<const int32_t *>(ids + size_t(slot) * ids_slot + size_t(token) * ids_token);
    }
}

static bool s11_has_external_consumer(ggml_cgraph * cgraph, int start_after, const ggml_tensor * target,
                                      ggml_cuda_s11_fusion_scope * scope) {
    for (int j = start_after; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * consumer = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (consumer->src[s] == target) {
                return true;
            }
        }
    }
    if (scope) {
        const auto * full_graph = scope->graph;
        const auto start = cgraph->nodes - full_graph->nodes;
        const auto end = start + cgraph->n_nodes;
        if (!scope->indexed) {
            scope->consumers.reserve(full_graph->n_nodes);
            for (int i = 0; i < full_graph->n_nodes; ++i) {
                const auto * consumer = full_graph->nodes[i];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const auto * source = consumer->src[s];
                    if (source && source != consumer) {
                        auto entry = scope->consumers.emplace(source, std::make_pair(i, i));
                        entry.first->second.second = i;
                    }
                }
            }
            scope->indexed = true;
        }
        const auto it = scope->consumers.find(target);
        if (it != scope->consumers.end() &&
            (it->second.first < start || it->second.second >= end)) {
            return true;
        }
    }
    return false;
}

int ggml_cuda_s11_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
                          ggml_cuda_s11_fusion_scope * scope) {
    const auto s11_has_external_consumer = [scope](ggml_cgraph * graph, int start, const ggml_tensor * target) {
        return ::s11_has_external_consumer(graph, start, target, scope);
    };
    if (s11_optimization() < 2) {
        return 0;
    }
    ggml_tensor * n0 = cgraph->nodes[i];

    // Pattern A: 6-node S11 RMSNorm: RMS_NORM -> CPY(mw, F32) -> SCALE(1,1) -> MUL -> CPY(BF16) -> CPY(F32)
    if (n0->op == GGML_OP_RMS_NORM && i + 5 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        if (n1->op == GGML_OP_CPY && n2->op == GGML_OP_SCALE && n3->op == GGML_OP_MUL &&
            n4->op == GGML_OP_CPY && n5->op == GGML_OP_CPY &&
            n1->src[0] && n1->src[0]->type == GGML_TYPE_BF16 && n1->type == GGML_TYPE_F32 &&
            n2->src[0] == n1 && n3->src[0] == n0 && n3->src[1] == n2 &&
            n4->src[0] == n3 && n4->type == GGML_TYPE_BF16 &&
            n5->src[0] == n4 && n5->type == GGML_TYPE_F32 &&
            n0->src[0]->type == GGML_TYPE_F32 && n0->src[0]->nb[0] == sizeof(float) &&
            ggml_is_contiguous(n1->src[0]) && ggml_is_contiguous(n5) &&
            ggml_nelements(n1->src[0]) == n0->ne[0] &&
            (!s11_tensors_overlap(n5, n0->src[0]) || (ggml_is_contiguous(n0->src[0]) && n5->data == n0->src[0]->data))) {
            if (!s11_has_external_consumer(cgraph, i + 6, n0) &&
                !s11_has_external_consumer(cgraph, i + 6, n1) &&
                !s11_has_external_consumer(cgraph, i + 6, n2) &&
                !s11_has_external_consumer(cgraph, i + 6, n3) &&
                !s11_has_external_consumer(cgraph, i + 6, n4)) {
                float eps = 1e-6f, scale = 1.0f, bias = 1.0f;
                std::memcpy(&eps, n0->op_params, sizeof(float));
                std::memcpy(&scale, reinterpret_cast<const char *>(n2->op_params) + 0, sizeof(float));
                std::memcpy(&bias, reinterpret_cast<const char *>(n2->op_params) + 4, sizeof(float));
                const ggml_tensor * x = n0->src[0];
                const int k = int(n0->ne[0]);
                const int nrows = int(ggml_nrows(n0));
                ggml_cuda_s11_invalidate_if_written(n5->data);
                const bool fuse_quant = (k == 2048) &&
                                        (size_t(nrows) * 2048 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows) * 16   <= g_s11_state.s_cap[0]);
                s11_rms_norm_offset_bf16_kernel<<<nrows, 256, 0, ctx.stream()>>>(
                    static_cast<const char *>(x->data),
                    static_cast<const nv_bfloat16 *>(n1->src[0]->data),
                    static_cast<float *>(n5->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    k, int(x->ne[1]), int(x->ne[2]), x->nb[1], x->nb[2], x->nb[3],
                    eps, scale, bias);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n5->data;
                    g_s11_state.last_k[0]           = 2048;
                    g_s11_state.last_tokens[0]      = nrows;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 5;
            }
        }
    }

    // Pattern B-Prologue: 12-node GDN beta + sigmoid + alpha + gate prologue sharing input cur
    if (n0->op == GGML_OP_MUL_MAT && i + 11 < cgraph->n_nodes &&
        n0->src[0] && n0->src[0]->type == GGML_TYPE_BF16 &&
        n0->src[0]->ne[0] == 2048 && n0->src[0]->ne[1] == 32 &&
        n0->src[1] && n0->src[1]->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0->src[1]) &&
        n0->ne[1] * n0->ne[2] * n0->ne[3] == 1) {
        ggml_tensor * b_res  = cgraph->nodes[i + 1];
        ggml_tensor * b_sig  = cgraph->nodes[i + 2];
        ggml_tensor * b_c1   = cgraph->nodes[i + 3];
        ggml_tensor * b_c2   = cgraph->nodes[i + 4];
        ggml_tensor * a_mm   = cgraph->nodes[i + 5];
        ggml_tensor * a_res  = cgraph->nodes[i + 6];
        ggml_tensor * dt_cpy = cgraph->nodes[i + 7];
        ggml_tensor * a_add  = cgraph->nodes[i + 8];
        ggml_tensor * a_sp   = cgraph->nodes[i + 9];
        ggml_tensor * g_mul  = cgraph->nodes[i + 10];
        ggml_tensor * g_res  = cgraph->nodes[i + 11];
        if (b_res->op == GGML_OP_RESHAPE && b_res->src[0] == n0 &&
            b_sig->op == GGML_OP_UNARY && ggml_get_unary_op(b_sig) == GGML_UNARY_OP_SIGMOID && b_sig->src[0] == b_res &&
            b_c1->op == GGML_OP_CPY && b_c1->src[0] == b_sig && b_c1->type == GGML_TYPE_BF16 &&
            b_c2->op == GGML_OP_CPY && b_c2->src[0] == b_c1 && b_c2->type == GGML_TYPE_F32 &&
            a_mm->op == GGML_OP_MUL_MAT && a_mm->src[1] == n0->src[1] &&
            a_mm->src[0] && a_mm->src[0]->type == GGML_TYPE_BF16 &&
            a_mm->src[0]->ne[0] == 2048 && a_mm->src[0]->ne[1] == 32 &&
            ggml_is_contiguous(a_mm->src[0]) &&
            a_res->op == GGML_OP_RESHAPE && a_res->src[0] == a_mm &&
            dt_cpy->op == GGML_OP_CPY && dt_cpy->src[0] && ggml_nelements(dt_cpy) == 32 &&
            a_add->op == GGML_OP_ADD && a_add->src[0] == a_res && a_add->src[1] == dt_cpy &&
            a_sp->op == GGML_OP_UNARY && ggml_get_unary_op(a_sp) == GGML_UNARY_OP_SOFTPLUS && a_sp->src[0] == a_add &&
            g_mul->op == GGML_OP_MUL && g_mul->src[0] == a_sp && g_mul->src[1] &&
            g_mul->src[1]->op == GGML_OP_SCALE && g_mul->src[1]->src[0] &&
            g_mul->src[1]->src[0]->op == GGML_OP_UNARY && ggml_get_unary_op(g_mul->src[1]->src[0]) == GGML_UNARY_OP_EXP &&
            g_mul->src[1]->src[0]->src[0] && g_mul->src[1]->src[0]->src[0]->op == GGML_OP_CPY &&
            g_mul->src[1]->src[0]->src[0]->src[0] &&
            g_res->op == GGML_OP_RESHAPE && g_res->src[0] == g_mul && g_res->data == g_mul->data &&
            ggml_is_contiguous(b_c2) && ggml_is_contiguous(g_res) &&
            !s11_tensors_overlap(b_c2, n0->src[1]) && !s11_tensors_overlap(g_res, n0->src[1]) &&
            !s11_tensors_overlap(b_c2, g_res)) {
            bool external = false;
            for (const int idx : {0, 1, 2, 3, 5, 6, 7, 8, 9, 10}) {
                if (s11_has_external_consumer(cgraph, i + 12, cgraph->nodes[i + idx])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const ggml_tensor * ssm_a_src = g_mul->src[1]->src[0]->src[0]->src[0];
                const int tokens = int(n0->ne[1] * n0->ne[2] * n0->ne[3]);
                ggml_cuda_s11_invalidate_if_written(b_c2->data);
                ggml_cuda_s11_invalidate_if_written(g_res->data);
                if (tokens >= 8) {
                    s11_gdn_alpha_beta_gate_fused_kernel<8><<<dim3(16, (tokens + 7) / 8), 128, 0, ctx.stream()>>>(
                        static_cast<const nv_bfloat16 *>(a_mm->src[0]->data),
                        static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                        static_cast<const float *>(n0->src[1]->data),
                        dt_cpy->src[0]->data, dt_cpy->src[0]->type,
                        ssm_a_src->data, ssm_a_src->type,
                        static_cast<float *>(g_res->data),
                        static_cast<float *>(b_c2->data),
                        tokens);
                } else {
                    s11_gdn_alpha_beta_gate_fused_kernel<1><<<dim3(16, tokens), 128, 0, ctx.stream()>>>(
                        static_cast<const nv_bfloat16 *>(a_mm->src[0]->data),
                        static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                        static_cast<const float *>(n0->src[1]->data),
                        dt_cpy->src[0]->data, dt_cpy->src[0]->type,
                        ssm_a_src->data, ssm_a_src->type,
                        static_cast<float *>(g_res->data),
                        static_cast<float *>(b_c2->data),
                        tokens);
                }
                CUDA_CHECK(cudaGetLastError());
                return 11;
            }
        }
    }

    // Pattern B-Prologue: 15-node GDN alpha + gate + beta prologue sharing input cur
    if (false && n0->op == GGML_OP_MUL_MAT && i + 14 < cgraph->n_nodes &&
        n0->src[0] && n0->src[0]->type == GGML_TYPE_BF16 &&
        n0->src[0]->ne[0] == 2048 && n0->src[0]->ne[1] == 32 &&
        n0->src[1] && n0->src[1]->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0->src[1])) {
        ggml_tensor * n1  = cgraph->nodes[i + 1];
        ggml_tensor * n2  = cgraph->nodes[i + 2];
        ggml_tensor * n3  = cgraph->nodes[i + 3];
        ggml_tensor * n4  = cgraph->nodes[i + 4];
        ggml_tensor * n5  = cgraph->nodes[i + 5];
        ggml_tensor * n6  = cgraph->nodes[i + 6];
        ggml_tensor * n7  = cgraph->nodes[i + 7];
        ggml_tensor * n8  = cgraph->nodes[i + 8];
        ggml_tensor * n9  = cgraph->nodes[i + 9];
        ggml_tensor * n10 = cgraph->nodes[i + 10];
        ggml_tensor * n11 = cgraph->nodes[i + 11];
        ggml_tensor * n12 = cgraph->nodes[i + 12];
        ggml_tensor * n13 = cgraph->nodes[i + 13];
        ggml_tensor * n14 = cgraph->nodes[i + 14];
        if (n1->op == GGML_OP_RESHAPE && n1->src[0] == n0 &&
            n2->op == GGML_OP_CPY && n2->src[0] && ggml_nelements(n2) == 32 &&
            n3->op == GGML_OP_ADD && n3->src[0] == n1 && n3->src[1] == n2 &&
            n4->op == GGML_OP_UNARY && ggml_get_unary_op(n4) == GGML_UNARY_OP_SOFTPLUS && n4->src[0] == n3 &&
            n5->op == GGML_OP_CPY && n5->src[0] && ggml_nelements(n5) == 32 &&
            n6->op == GGML_OP_UNARY && ggml_get_unary_op(n6) == GGML_UNARY_OP_EXP && n6->src[0] == n5 &&
            n7->op == GGML_OP_SCALE && n7->src[0] == n6 &&
            n8->op == GGML_OP_MUL && n8->src[0] == n4 && n8->src[1] == n7 &&
            n9->op == GGML_OP_RESHAPE && n9->src[0] == n8 && n9->data == n8->data &&
            n10->op == GGML_OP_MUL_MAT && n10->src[1] == n0->src[1] &&
            n10->src[0] && n10->src[0]->type == GGML_TYPE_BF16 &&
            n10->src[0]->ne[0] == 2048 && n10->src[0]->ne[1] == 32 &&
            ggml_is_contiguous(n10->src[0]) &&
            n11->op == GGML_OP_RESHAPE && n11->src[0] == n10 &&
            n12->op == GGML_OP_UNARY && ggml_get_unary_op(n12) == GGML_UNARY_OP_SIGMOID && n12->src[0] == n11 &&
            n13->op == GGML_OP_CPY && n13->src[0] == n12 && n13->type == GGML_TYPE_BF16 &&
            n14->op == GGML_OP_CPY && n14->src[0] == n13 && n14->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n9) && ggml_is_contiguous(n14) &&
            !s11_tensors_overlap(n9, n0->src[1]) && !s11_tensors_overlap(n14, n0->src[1]) &&
            !s11_tensors_overlap(n9, n14)) {
            bool external = false;
            for (int m = 0; m < 14; ++m) {
                if (m == 9) continue;
                if (s11_has_external_consumer(cgraph, i + 15, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int tokens = int(n0->ne[1] * n0->ne[2] * n0->ne[3]);
                ggml_cuda_s11_invalidate_if_written(n9->data);
                ggml_cuda_s11_invalidate_if_written(n14->data);
                if (tokens >= 8) {
                    s11_gdn_alpha_beta_gate_fused_kernel<8><<<dim3(16, (tokens + 7) / 8), 128, 0, ctx.stream()>>>(
                        static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                        static_cast<const nv_bfloat16 *>(n10->src[0]->data),
                        static_cast<const float *>(n0->src[1]->data),
                        n2->src[0]->data, n2->src[0]->type,
                        n5->src[0]->data, n5->src[0]->type,
                        static_cast<float *>(n9->data),
                        static_cast<float *>(n14->data),
                        tokens);
                } else {
                    s11_gdn_alpha_beta_gate_fused_kernel<1><<<dim3(16, tokens), 128, 0, ctx.stream()>>>(
                        static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                        static_cast<const nv_bfloat16 *>(n10->src[0]->data),
                        static_cast<const float *>(n0->src[1]->data),
                        n2->src[0]->data, n2->src[0]->type,
                        n5->src[0]->data, n5->src[0]->type,
                        static_cast<float *>(n9->data),
                        static_cast<float *>(n14->data),
                        tokens);
                }
                CUDA_CHECK(cudaGetLastError());
                return 14;
            }
        }
    }

    // Pattern B1: 7-node S11 GDN gate subgraph (no intermediate BF16 casts)
    if (n0->op == GGML_OP_CPY && i + 6 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        ggml_tensor * n6 = cgraph->nodes[i + 6];
        if (n1->op == GGML_OP_ADD && n2->op == GGML_OP_UNARY && ggml_get_unary_op(n2) == GGML_UNARY_OP_SOFTPLUS &&
            n3->op == GGML_OP_CPY && n4->op == GGML_OP_UNARY && ggml_get_unary_op(n4) == GGML_UNARY_OP_EXP &&
            n5->op == GGML_OP_SCALE && n6->op == GGML_OP_MUL &&
            n1->src[1] == n0 && n2->src[0] == n1 &&
            n4->src[0] == n3 && n5->src[0] == n4 && n6->src[0] == n2 && n6->src[1] == n5 &&
            ggml_is_contiguous(n1->src[0]) && ggml_is_contiguous(n6) &&
            ggml_nelements(n0) == n1->ne[0] && ggml_nelements(n3) == n1->ne[0] &&
            !s11_tensors_overlap(n6, n1->src[0])) {
            bool external = false;
            for (int m = 0; m < 6; ++m) {
                if (s11_has_external_consumer(cgraph, i + 7, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int heads = int(n1->ne[0]);
                const int total = int(ggml_nelements(n6));
                ggml_cuda_s11_invalidate_if_written(n6->data);
                s11_gdn_gate_fused_kernel<<<(total + 255) / 256, 256, 0, ctx.stream()>>>(
                    static_cast<const float *>(n1->src[0]->data),
                    n0->src[0]->data, n0->src[0]->type,
                    n3->src[0]->data, n3->src[0]->type,
                    static_cast<float *>(n6->data), heads, total, false);
                CUDA_CHECK(cudaGetLastError());
                return 6;
            }
        }
    }

    // Pattern B2-11: 11-node in-place build_norm_gated subgraph in GDN layers (absorbing z MUL_MAT + z_2d RESHAPE)
    if (n0->op == GGML_OP_CPY && i + 10 < cgraph->n_nodes) {
        ggml_tensor * n1    = cgraph->nodes[i + 1];
        ggml_tensor * n2    = cgraph->nodes[i + 2];
        ggml_tensor * n3    = cgraph->nodes[i + 3];
        ggml_tensor * n4    = cgraph->nodes[i + 4];
        ggml_tensor * n_z   = cgraph->nodes[i + 5];
        ggml_tensor * n_z2d = cgraph->nodes[i + 6];
        ggml_tensor * n5    = cgraph->nodes[i + 7];
        ggml_tensor * n6    = cgraph->nodes[i + 8];
        ggml_tensor * n7    = cgraph->nodes[i + 9];
        ggml_tensor * n8    = cgraph->nodes[i + 10];
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            n2->op == GGML_OP_RMS_NORM && n2->src[0] == n1 && n2->ne[0] == 128 &&
            n3->op == GGML_OP_CPY && n3->src[0] && n3->type == GGML_TYPE_F32 && ggml_nelements(n3) == 128 &&
            n4->op == GGML_OP_MUL && n4->src[0] == n2 && n4->src[1] == n3 &&
            n_z->op == GGML_OP_MUL_MAT && n_z->src[0] && ggml_cuda_is_s11(n_z->src[0]->type) &&
            n_z2d->op == GGML_OP_RESHAPE && n_z2d->src[0] == n_z && n_z2d->data == n_z->data &&
            n5->op == GGML_OP_UNARY && ggml_get_unary_op(n5) == GGML_UNARY_OP_SILU && n5->src[0] == n_z2d &&
            n6->op == GGML_OP_MUL && n6->src[0] == n4 && n6->src[1] == n5 &&
            n7->op == GGML_OP_CPY && n7->src[0] == n6 && n7->type == GGML_TYPE_BF16 &&
            n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n_z2d) && ggml_is_contiguous(n8) &&
            !s11_tensors_overlap(n_z, n0->src[0]) && !s11_tensors_overlap(n_z, n3->src[0]) &&
            !s11_tensors_overlap(n8, n0->src[0]) && !s11_tensors_overlap(n8, n_z2d)) {
            bool external = false;
            for (int m = 0; m < 10; ++m) {
                if (s11_has_external_consumer(cgraph, i + 11, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                ggml_cuda_s11_invalidate_if_written(n_z->data);
                ggml_cuda_s11_mul_mat(ctx, n_z);
                float eps = 1e-6f;
                std::memcpy(&eps, n2->op_params, sizeof(float));
                const int nrows = int(ggml_nrows(n8));
                ggml_cuda_s11_invalidate_if_written(n8->data);
                const bool fuse_quant = (nrows % 32 == 0) &&
                                        (size_t(nrows) * 128 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows)       <= g_s11_state.s_cap[0]);
                s11_norm_gated_fused_kernel<<<nrows, 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    n3->src[0]->data, n3->src[0]->type,
                    static_cast<const float *>(n_z2d->data),
                    static_cast<float *>(n8->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    eps);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n8->data;
                    g_s11_state.last_k[0]           = 4096;
                    g_s11_state.last_tokens[0]      = nrows / 32;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 10;
            }
        }
    }

    // Pattern B2-10: 10-node in-place build_norm_gated subgraph when z is pre-expanded at build_qkvz
    if (n0->op == GGML_OP_CPY && i + 9 < cgraph->n_nodes) {
        ggml_tensor * n1    = cgraph->nodes[i + 1];
        ggml_tensor * n2    = cgraph->nodes[i + 2];
        ggml_tensor * n3    = cgraph->nodes[i + 3];
        ggml_tensor * n4    = cgraph->nodes[i + 4];
        ggml_tensor * n_z2d = cgraph->nodes[i + 5];
        ggml_tensor * n5    = cgraph->nodes[i + 6];
        ggml_tensor * n6    = cgraph->nodes[i + 7];
        ggml_tensor * n7    = cgraph->nodes[i + 8];
        ggml_tensor * n8    = cgraph->nodes[i + 9];
        auto non_overlapping_or_exact_inplace = [](const ggml_tensor * dst, const ggml_tensor * src) {
            return !s11_tensors_overlap(dst, src) || (dst->data == src->data && ggml_nbytes(dst) == ggml_nbytes(src));
        };
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            n2->op == GGML_OP_RMS_NORM && n2->src[0] == n1 && n2->ne[0] == 128 &&
            n3->op == GGML_OP_CPY && n3->src[0] && n3->type == GGML_TYPE_F32 && ggml_nelements(n3) == 128 &&
            n4->op == GGML_OP_MUL && n4->src[0] == n2 && n4->src[1] == n3 &&
            n_z2d->op == GGML_OP_RESHAPE && n_z2d->src[0] && n_z2d->src[0]->type == GGML_TYPE_F32 &&
            n5->op == GGML_OP_UNARY && ggml_get_unary_op(n5) == GGML_UNARY_OP_SILU && n5->src[0] == n_z2d &&
            n6->op == GGML_OP_MUL && n6->src[0] == n4 && n6->src[1] == n5 &&
            n7->op == GGML_OP_CPY && n7->src[0] == n6 && n7->type == GGML_TYPE_BF16 &&
            n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n_z2d) && ggml_is_contiguous(n8) &&
            non_overlapping_or_exact_inplace(n8, n0->src[0]) &&
            non_overlapping_or_exact_inplace(n8, n_z2d) &&
            !s11_tensors_overlap(n8, n3->src[0])) {
            bool external = false;
            for (int m = 0; m < 9; ++m) {
                if (s11_has_external_consumer(cgraph, i + 10, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                float eps = 1e-6f;
                std::memcpy(&eps, n2->op_params, sizeof(float));
                const int nrows = int(ggml_nrows(n8));
                ggml_cuda_s11_invalidate_if_written(n8->data);
                const bool fuse_quant = (nrows % 32 == 0) &&
                                        (size_t(nrows) * 128 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows)       <= g_s11_state.s_cap[0]);
                s11_norm_gated_fused_kernel<<<nrows, 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    n3->src[0]->data, n3->src[0]->type,
                    static_cast<const float *>(n_z2d->data),
                    static_cast<float *>(n8->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    eps);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n8->data;
                    g_s11_state.last_k[0]           = 4096;
                    g_s11_state.last_tokens[0]      = nrows / 32;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 9;
            }
        }
    }

    // Pattern B2: 9-node build_norm_gated subgraph in GDN layers
    if (n0->op == GGML_OP_CPY && i + 8 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        ggml_tensor * n6 = cgraph->nodes[i + 6];
        ggml_tensor * n7 = cgraph->nodes[i + 7];
        ggml_tensor * n8 = cgraph->nodes[i + 8];
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            n2->op == GGML_OP_RMS_NORM && n2->src[0] == n1 && n2->ne[0] == 128 &&
            n3->op == GGML_OP_CPY && n3->src[0] && n3->type == GGML_TYPE_F32 && ggml_nelements(n3) == 128 &&
            n4->op == GGML_OP_MUL && n4->src[0] == n2 && n4->src[1] == n3 &&
            n5->op == GGML_OP_UNARY && ggml_get_unary_op(n5) == GGML_UNARY_OP_SILU &&
            n6->op == GGML_OP_MUL && n6->src[0] == n4 && n6->src[1] == n5 &&
            n7->op == GGML_OP_CPY && n7->src[0] == n6 && n7->type == GGML_TYPE_BF16 &&
            n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n5->src[0]) && ggml_is_contiguous(n8) &&
            !s11_tensors_overlap(n8, n0->src[0]) && !s11_tensors_overlap(n8, n5->src[0])) {
            bool external = false;
            for (int m = 0; m < 8; ++m) {
                if (s11_has_external_consumer(cgraph, i + 9, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                float eps = 1e-6f;
                std::memcpy(&eps, n2->op_params, sizeof(float));
                const int nrows = int(ggml_nrows(n8));
                ggml_cuda_s11_invalidate_if_written(n8->data);
                const bool fuse_quant = (nrows % 32 == 0) &&
                                        (size_t(nrows) * 128 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows)       <= g_s11_state.s_cap[0]);
                s11_norm_gated_fused_kernel<<<nrows, 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    n3->src[0]->data, n3->src[0]->type,
                    static_cast<const float *>(n5->src[0]->data),
                    static_cast<float *>(n8->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    eps);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n8->data;
                    g_s11_state.last_k[0]           = 4096;
                    g_s11_state.last_tokens[0]      = nrows / 32;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 8;
            }
        }
    }

    // Pattern B: 11-node S11 GDN gate subgraph
    if (n0->op == GGML_OP_CPY && i + 10 < cgraph->n_nodes) {
        ggml_tensor * n1  = cgraph->nodes[i + 1];
        ggml_tensor * n2  = cgraph->nodes[i + 2];
        ggml_tensor * n3  = cgraph->nodes[i + 3];
        ggml_tensor * n4  = cgraph->nodes[i + 4];
        ggml_tensor * n5  = cgraph->nodes[i + 5];
        ggml_tensor * n6  = cgraph->nodes[i + 6];
        ggml_tensor * n7  = cgraph->nodes[i + 7];
        ggml_tensor * n8  = cgraph->nodes[i + 8];
        ggml_tensor * n9  = cgraph->nodes[i + 9];
        ggml_tensor * n10 = cgraph->nodes[i + 10];
        if (n1->op == GGML_OP_ADD && n2->op == GGML_OP_UNARY && ggml_get_unary_op(n2) == GGML_UNARY_OP_SOFTPLUS &&
            n3->op == GGML_OP_CPY && n3->type == GGML_TYPE_BF16 &&
            n4->op == GGML_OP_CPY && n4->type == GGML_TYPE_F32 &&
            n5->op == GGML_OP_CPY && n6->op == GGML_OP_UNARY && ggml_get_unary_op(n6) == GGML_UNARY_OP_EXP &&
            n7->op == GGML_OP_SCALE && n8->op == GGML_OP_MUL &&
            n9->op == GGML_OP_CPY && n9->type == GGML_TYPE_BF16 &&
            n10->op == GGML_OP_CPY && n10->type == GGML_TYPE_F32 &&
            n1->src[1] == n0 && n2->src[0] == n1 && n3->src[0] == n2 && n4->src[0] == n3 &&
            n6->src[0] == n5 && n7->src[0] == n6 && n8->src[0] == n4 && n8->src[1] == n7 &&
            n9->src[0] == n8 && n10->src[0] == n9 &&
            ggml_is_contiguous(n1->src[0]) && ggml_is_contiguous(n10) &&
            ggml_nelements(n0) == n1->ne[0] && ggml_nelements(n5) == n1->ne[0]) {
            bool external = false;
            for (int m = 0; m < 10; ++m) {
                if (s11_has_external_consumer(cgraph, i + 11, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int heads = int(n1->ne[0]);
                const int total = int(ggml_nelements(n10));
                ggml_cuda_s11_invalidate_if_written(n10->data);
                s11_gdn_gate_fused_kernel<<<(total + 255) / 256, 256, 0, ctx.stream()>>>(
                    static_cast<const float *>(n1->src[0]->data),
                    n0->src[0]->data, n0->src[0]->type,
                    n5->src[0]->data, n5->src[0]->type,
                    static_cast<float *>(n10->data), heads, total, true);
                CUDA_CHECK(cudaGetLastError());
                return 10;
            }
        }
    }

    // Pattern G13+A: 19-node ffn_gate_inp_shexp (n=1) + Shared-Expert + MoE + Residual Epilogue + Next-Layer RMSNorm + Quantize
    if (n0->op == GGML_OP_MUL_MAT && i + 18 < cgraph->n_nodes &&
        n0->src[0] && n0->src[0]->type == GGML_TYPE_BF16 && n0->src[0]->ne[0] == 2048 &&
        n0->src[1] && n0->src[1]->type == GGML_TYPE_F32 && n0->ne[0] == 1 &&
        ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0->src[1])) {
        ggml_tensor * g0  = cgraph->nodes[i + 1];
        ggml_tensor * g1  = cgraph->nodes[i + 2];
        ggml_tensor * g2  = cgraph->nodes[i + 3];
        ggml_tensor * g3  = cgraph->nodes[i + 4];
        ggml_tensor * g4  = cgraph->nodes[i + 5];
        ggml_tensor * g5  = cgraph->nodes[i + 6];
        ggml_tensor * g6  = cgraph->nodes[i + 7];
        ggml_tensor * g7  = cgraph->nodes[i + 8];
        ggml_tensor * g8  = cgraph->nodes[i + 9];
        ggml_tensor * g9  = cgraph->nodes[i + 10];
        ggml_tensor * g10 = cgraph->nodes[i + 11];
        ggml_tensor * g11 = cgraph->nodes[i + 12];
        ggml_tensor * a0  = cgraph->nodes[i + 13];
        ggml_tensor * a1  = cgraph->nodes[i + 14];
        ggml_tensor * a2  = cgraph->nodes[i + 15];
        ggml_tensor * a3  = cgraph->nodes[i + 16];
        ggml_tensor * a4  = cgraph->nodes[i + 17];
        ggml_tensor * a5  = cgraph->nodes[i + 18];
        auto safe_alias = [](const ggml_tensor * dst, const ggml_tensor * src) {
            return !s11_tensors_overlap(dst, src) || dst->data == src->data;
        };
        if (g0->op == GGML_OP_UNARY && ggml_get_unary_op(g0) == GGML_UNARY_OP_SIGMOID && g0->src[0] == n0 &&
            g1->op == GGML_OP_CPY && g1->src[0] == g0 && g1->type == GGML_TYPE_BF16 &&
            g2->op == GGML_OP_CPY && g2->src[0] == g1 && g2->type == GGML_TYPE_F32 &&
            g3->op == GGML_OP_MUL && g3->src[1] == g2 && g3->src[0] && g3->src[0]->type == GGML_TYPE_F32 &&
            g4->op == GGML_OP_CPY && g4->src[0] == g3 && g4->type == GGML_TYPE_BF16 &&
            g5->op == GGML_OP_CPY && g5->src[0] == g4 && g5->type == GGML_TYPE_F32 &&
            g6->op == GGML_OP_ADD && g6->src[1] == g5 && g6->src[0] && g6->src[0]->type == GGML_TYPE_F32 &&
            g7->op == GGML_OP_CPY && g7->src[0] == g6 && g7->type == GGML_TYPE_BF16 &&
            g8->op == GGML_OP_CPY && g8->src[0] == g7 && g8->type == GGML_TYPE_F32 &&
            g9->op == GGML_OP_ADD && g9->src[0] == g8 && g9->src[1] && g9->src[1]->type == GGML_TYPE_F32 &&
            g10->op == GGML_OP_CPY && g10->src[0] == g9 && g10->type == GGML_TYPE_BF16 &&
            g11->op == GGML_OP_CPY && g11->src[0] == g10 && g11->type == GGML_TYPE_F32 &&
            a0->op == GGML_OP_RMS_NORM && a0->src[0] == g11 && a0->ne[0] == 2048 &&
            a1->op == GGML_OP_CPY && a1->src[0] && a1->src[0]->type == GGML_TYPE_BF16 && a1->type == GGML_TYPE_F32 &&
            a2->op == GGML_OP_SCALE && a2->src[0] == a1 &&
            a3->op == GGML_OP_MUL && a3->src[0] == a0 && a3->src[1] == a2 &&
            a4->op == GGML_OP_CPY && a4->src[0] == a3 && a4->type == GGML_TYPE_BF16 &&
            a5->op == GGML_OP_CPY && a5->src[0] == a4 && a5->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(g3->src[0]) && ggml_is_contiguous(g6->src[0]) &&
            ggml_is_contiguous(g9->src[1]) && ggml_is_contiguous(g11) &&
            ggml_is_contiguous(a1->src[0]) && ggml_is_contiguous(a5) &&
            ggml_nelements(a1->src[0]) == 2048 &&
            g11->ne[0] == 2048 && n0->ne[1] == g11->ne[1] &&
            ggml_are_same_shape(n0->src[1], g11) &&
            ggml_are_same_shape(g3->src[0], g6->src[0]) &&
            ggml_are_same_shape(g6->src[0], g9->src[1]) &&
            ggml_are_same_shape(g9->src[1], g11) &&
            ggml_are_same_shape(g11, a5) &&
            !s11_tensors_overlap(g11, a5) &&
            !s11_tensors_overlap(g11, n0->src[0]) && !s11_tensors_overlap(a5, n0->src[0]) &&
            !s11_tensors_overlap(g11, a1->src[0]) && !s11_tensors_overlap(a5, a1->src[0]) &&
            safe_alias(g11, n0->src[1]) && safe_alias(g11, g3->src[0]) &&
            safe_alias(g11, g6->src[0]) && safe_alias(g11, g9->src[1]) &&
            safe_alias(a5, n0->src[1]) && safe_alias(a5, g3->src[0]) &&
            safe_alias(a5, g6->src[0]) && safe_alias(a5, g9->src[1])) {
            bool external = false;
            for (int m = 0; m < 18; ++m) {
                if (m == 12) continue;
                if (s11_has_external_consumer(cgraph, i + 19, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                float eps = 1e-6f, scale = 1.0f, bias = 1.0f;
                std::memcpy(&eps, a0->op_params, sizeof(float));
                std::memcpy(&scale, reinterpret_cast<const char *>(a2->op_params) + 0, sizeof(float));
                std::memcpy(&bias, reinterpret_cast<const char *>(a2->op_params) + 4, sizeof(float));
                const int nrows = int(ggml_nrows(a0));
                ggml_cuda_s11_invalidate_if_written(g11->data);
                ggml_cuda_s11_invalidate_if_written(a5->data);
                const bool fuse_quant = (size_t(nrows) * 2048 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows) * 16   <= g_s11_state.s_cap[0]);
                s11_shexp_gate_epilogue_rms_norm_quant_fused_kernel<<<nrows, 256, 0, ctx.stream()>>>(
                    static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                    static_cast<const float *>(n0->src[1]->data),
                    nullptr,
                    static_cast<const float *>(g3->src[0]->data),
                    static_cast<const float *>(g6->src[0]->data),
                    static_cast<const float *>(g9->src[1]->data),
                    static_cast<float *>(g11->data),
                    static_cast<const nv_bfloat16 *>(a1->src[0]->data),
                    static_cast<float *>(a5->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    eps, scale, bias);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = a5->data;
                    g_s11_state.last_k[0]           = 2048;
                    g_s11_state.last_tokens[0]      = nrows;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 18;
            }
        }
    }

    // Pattern G-13: 13-node ffn_gate_inp_shexp (n=1) + Shared-Expert + MoE + Residual Epilogue
    if (n0->op == GGML_OP_MUL_MAT && i + 12 < cgraph->n_nodes &&
        n0->src[0] && n0->src[0]->type == GGML_TYPE_BF16 && n0->src[0]->ne[0] == 2048 &&
        n0->src[1] && n0->src[1]->type == GGML_TYPE_F32 && n0->ne[0] == 1 &&
        ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0->src[1])) {
        ggml_tensor * g0  = cgraph->nodes[i + 1];
        ggml_tensor * g1  = cgraph->nodes[i + 2];
        ggml_tensor * g2  = cgraph->nodes[i + 3];
        ggml_tensor * g3  = cgraph->nodes[i + 4];
        ggml_tensor * g4  = cgraph->nodes[i + 5];
        ggml_tensor * g5  = cgraph->nodes[i + 6];
        ggml_tensor * g6  = cgraph->nodes[i + 7];
        ggml_tensor * g7  = cgraph->nodes[i + 8];
        ggml_tensor * g8  = cgraph->nodes[i + 9];
        ggml_tensor * g9  = cgraph->nodes[i + 10];
        ggml_tensor * g10 = cgraph->nodes[i + 11];
        ggml_tensor * g11 = cgraph->nodes[i + 12];
        auto safe_alias = [](const ggml_tensor * dst, const ggml_tensor * src) {
            return !s11_tensors_overlap(dst, src) || dst->data == src->data;
        };
        if (g0->op == GGML_OP_UNARY && ggml_get_unary_op(g0) == GGML_UNARY_OP_SIGMOID && g0->src[0] == n0 &&
            g1->op == GGML_OP_CPY && g1->src[0] == g0 && g1->type == GGML_TYPE_BF16 &&
            g2->op == GGML_OP_CPY && g2->src[0] == g1 && g2->type == GGML_TYPE_F32 &&
            g3->op == GGML_OP_MUL && g3->src[1] == g2 && g3->src[0] && g3->src[0]->type == GGML_TYPE_F32 &&
            g4->op == GGML_OP_CPY && g4->src[0] == g3 && g4->type == GGML_TYPE_BF16 &&
            g5->op == GGML_OP_CPY && g5->src[0] == g4 && g5->type == GGML_TYPE_F32 &&
            g6->op == GGML_OP_ADD && g6->src[1] == g5 && g6->src[0] && g6->src[0]->type == GGML_TYPE_F32 &&
            g7->op == GGML_OP_CPY && g7->src[0] == g6 && g7->type == GGML_TYPE_BF16 &&
            g8->op == GGML_OP_CPY && g8->src[0] == g7 && g8->type == GGML_TYPE_F32 &&
            g9->op == GGML_OP_ADD && g9->src[0] == g8 && g9->src[1] && g9->src[1]->type == GGML_TYPE_F32 &&
            g10->op == GGML_OP_CPY && g10->src[0] == g9 && g10->type == GGML_TYPE_BF16 &&
            g11->op == GGML_OP_CPY && g11->src[0] == g10 && g11->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(g3->src[0]) && ggml_is_contiguous(g6->src[0]) &&
            ggml_is_contiguous(g9->src[1]) && ggml_is_contiguous(g11) &&
            g11->ne[0] == 2048 && n0->ne[1] == g11->ne[1] &&
            ggml_are_same_shape(n0->src[1], g11) &&
            ggml_are_same_shape(g3->src[0], g6->src[0]) &&
            ggml_are_same_shape(g6->src[0], g9->src[1]) &&
            ggml_are_same_shape(g9->src[1], g11) &&
            !s11_tensors_overlap(g11, n0->src[0]) &&
            safe_alias(g11, n0->src[1]) && safe_alias(g11, g3->src[0]) &&
            safe_alias(g11, g6->src[0]) && safe_alias(g11, g9->src[1])) {
            bool external = false;
            for (int m = 0; m < 12; ++m) {
                if (s11_has_external_consumer(cgraph, i + 13, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int tokens = int(g11->ne[1] * g11->ne[2] * g11->ne[3]);
                ggml_cuda_s11_invalidate_if_written(g11->data);
                s11_shexp_gate_epilogue_fused_kernel<<<tokens, 256, 0, ctx.stream()>>>(
                    static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                    static_cast<const float *>(n0->src[1]->data),
                    static_cast<const float *>(g3->src[0]->data),
                    static_cast<const float *>(g6->src[0]->data),
                    static_cast<const float *>(g9->src[1]->data),
                    static_cast<float *>(g11->data));
                CUDA_CHECK(cudaGetLastError());
                return 12;
            }
        }
    }

    // Pattern FA-Gate: 6-node post-FlashAttention gate sigmoid + BF16 casts + mul + A128 quantize
    if (n0->op == GGML_OP_UNARY && ggml_get_unary_op(n0) == GGML_UNARY_OP_SIGMOID &&
        i + 5 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_BF16 &&
            n2->op == GGML_OP_CPY && n2->src[0] == n1 && n2->type == GGML_TYPE_F32 &&
            n3->op == GGML_OP_CPY && n3->src[0] && n3->src[0]->type == GGML_TYPE_F32 && n3->type == GGML_TYPE_BF16 &&
            n4->op == GGML_OP_CPY && n4->src[0] == n3 && n4->type == GGML_TYPE_F32 &&
            n5->op == GGML_OP_MUL && n5->src[0] == n4 && n5->src[1] == n2 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n3->src[0]) && ggml_is_contiguous(n5) &&
            ggml_are_same_shape(n0->src[0], n3->src[0]) && ggml_are_same_shape(n3->src[0], n5) &&
            n5->ne[0] == 4096 &&
            (!s11_tensors_overlap(n5, n0->src[0]) || n5->data == n0->src[0]->data) &&
            (!s11_tensors_overlap(n5, n3->src[0]) || n5->data == n3->src[0]->data)) {
            bool external = false;
            for (int m = 0; m < 5; ++m) {
                if (s11_has_external_consumer(cgraph, i + 6, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int tokens = int(n5->ne[1] * n5->ne[2] * n5->ne[3]);
                const int group_rows = tokens * 32;
                ggml_cuda_s11_invalidate_if_written(n5->data);
                const bool fuse_quant = (size_t(group_rows) * 128 <= g_s11_state.q_cap[0]) &&
                                        (size_t(group_rows)       <= g_s11_state.s_cap[0]);
                s11_fa_gate_quant_fused_kernel<<<group_rows, 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(n3->src[0]->data),
                    static_cast<const float *>(n0->src[0]->data),
                    static_cast<float *>(n5->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n5->data;
                    g_s11_state.last_k[0]           = 4096;
                    g_s11_state.last_tokens[0]      = tokens;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 5;
            }
        }
    }

    // Pattern G+A: 18-node Shared-Expert + MoE + Residual Epilogue + Next-Layer RMSNorm + Quantize
    if (n0->op == GGML_OP_UNARY && ggml_get_unary_op(n0) == GGML_UNARY_OP_SIGMOID &&
        i + 17 < cgraph->n_nodes) {
        ggml_tensor * n1  = cgraph->nodes[i + 1];
        ggml_tensor * n2  = cgraph->nodes[i + 2];
        ggml_tensor * n3  = cgraph->nodes[i + 3];
        ggml_tensor * n4  = cgraph->nodes[i + 4];
        ggml_tensor * n5  = cgraph->nodes[i + 5];
        ggml_tensor * n6  = cgraph->nodes[i + 6];
        ggml_tensor * n7  = cgraph->nodes[i + 7];
        ggml_tensor * n8  = cgraph->nodes[i + 8];
        ggml_tensor * n9  = cgraph->nodes[i + 9];
        ggml_tensor * n10 = cgraph->nodes[i + 10];
        ggml_tensor * n11 = cgraph->nodes[i + 11];
        ggml_tensor * n12 = cgraph->nodes[i + 12];
        ggml_tensor * n13 = cgraph->nodes[i + 13];
        ggml_tensor * n14 = cgraph->nodes[i + 14];
        ggml_tensor * n15 = cgraph->nodes[i + 15];
        ggml_tensor * n16 = cgraph->nodes[i + 16];
        ggml_tensor * n17 = cgraph->nodes[i + 17];
        auto safe_alias = [](const ggml_tensor * dst, const ggml_tensor * src) {
            return !s11_tensors_overlap(dst, src) || dst->data == src->data;
        };
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->ne[0] == 1 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_BF16 &&
            n2->op == GGML_OP_CPY && n2->src[0] == n1 && n2->type == GGML_TYPE_F32 &&
            n3->op == GGML_OP_MUL && n3->src[1] == n2 && n3->src[0] && n3->src[0]->type == GGML_TYPE_F32 &&
            n4->op == GGML_OP_CPY && n4->src[0] == n3 && n4->type == GGML_TYPE_BF16 &&
            n5->op == GGML_OP_CPY && n5->src[0] == n4 && n5->type == GGML_TYPE_F32 &&
            n6->op == GGML_OP_ADD && n6->src[1] == n5 && n6->src[0] && n6->src[0]->type == GGML_TYPE_F32 &&
            n7->op == GGML_OP_CPY && n7->src[0] == n6 && n7->type == GGML_TYPE_BF16 &&
            n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
            n9->op == GGML_OP_ADD && n9->src[0] == n8 && n9->src[1] && n9->src[1]->type == GGML_TYPE_F32 &&
            n10->op == GGML_OP_CPY && n10->src[0] == n9 && n10->type == GGML_TYPE_BF16 &&
            n11->op == GGML_OP_CPY && n11->src[0] == n10 && n11->type == GGML_TYPE_F32 &&
            n12->op == GGML_OP_RMS_NORM && n12->src[0] == n11 && n12->ne[0] == 2048 &&
            n13->op == GGML_OP_CPY && n13->src[0] && n13->src[0]->type == GGML_TYPE_BF16 && n13->type == GGML_TYPE_F32 &&
            n14->op == GGML_OP_SCALE && n14->src[0] == n13 &&
            n15->op == GGML_OP_MUL && n15->src[0] == n12 && n15->src[1] == n14 &&
            n16->op == GGML_OP_CPY && n16->src[0] == n15 && n16->type == GGML_TYPE_BF16 &&
            n17->op == GGML_OP_CPY && n17->src[0] == n16 && n17->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n3->src[0]) &&
            ggml_is_contiguous(n6->src[0]) && ggml_is_contiguous(n9->src[1]) &&
            ggml_is_contiguous(n11) && ggml_is_contiguous(n13->src[0]) && ggml_is_contiguous(n17) &&
            ggml_nelements(n13->src[0]) == 2048 &&
            ggml_are_same_shape(n3->src[0], n6->src[0]) &&
            ggml_are_same_shape(n6->src[0], n9->src[1]) &&
            ggml_are_same_shape(n9->src[1], n11) &&
            ggml_are_same_shape(n11, n17) &&
            n0->src[0]->ne[1] == n11->ne[1] &&
            !s11_tensors_overlap(n11, n17) &&
            !s11_tensors_overlap(n11, n0->src[0]) && !s11_tensors_overlap(n17, n0->src[0]) &&
            !s11_tensors_overlap(n11, n13->src[0]) && !s11_tensors_overlap(n17, n13->src[0]) &&
            safe_alias(n11, n3->src[0]) && safe_alias(n11, n6->src[0]) && safe_alias(n11, n9->src[1]) &&
            safe_alias(n17, n3->src[0]) && safe_alias(n17, n6->src[0]) && safe_alias(n17, n9->src[1])) {
            bool external = false;
            for (int m = 0; m < 17; ++m) {
                if (m == 11) continue;
                if (s11_has_external_consumer(cgraph, i + 18, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                float eps = 1e-6f, scale = 1.0f, bias = 1.0f;
                std::memcpy(&eps, n12->op_params, sizeof(float));
                std::memcpy(&scale, reinterpret_cast<const char *>(n14->op_params) + 0, sizeof(float));
                std::memcpy(&bias, reinterpret_cast<const char *>(n14->op_params) + 4, sizeof(float));
                const int nrows = int(ggml_nrows(n12));
                ggml_cuda_s11_invalidate_if_written(n11->data);
                ggml_cuda_s11_invalidate_if_written(n17->data);
                const bool fuse_quant = (size_t(nrows) * 2048 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows) * 16   <= g_s11_state.s_cap[0]);
                s11_shexp_gate_epilogue_rms_norm_quant_fused_kernel<<<nrows, 256, 0, ctx.stream()>>>(
                    nullptr, nullptr,
                    static_cast<const float *>(n0->src[0]->data),
                    static_cast<const float *>(n3->src[0]->data),
                    static_cast<const float *>(n6->src[0]->data),
                    static_cast<const float *>(n9->src[1]->data),
                    static_cast<float *>(n11->data),
                    static_cast<const nv_bfloat16 *>(n13->src[0]->data),
                    static_cast<float *>(n17->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    eps, scale, bias);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n17->data;
                    g_s11_state.last_k[0]           = 2048;
                    g_s11_state.last_tokens[0]      = nrows;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 17;
            }
        }
    }

    // Pattern G: 12-node Shared-Expert + MoE + Residual Epilogue
    if (n0->op == GGML_OP_UNARY && ggml_get_unary_op(n0) == GGML_UNARY_OP_SIGMOID &&
        i + 11 < cgraph->n_nodes) {
        ggml_tensor * n1  = cgraph->nodes[i + 1];
        ggml_tensor * n2  = cgraph->nodes[i + 2];
        ggml_tensor * n3  = cgraph->nodes[i + 3];
        ggml_tensor * n4  = cgraph->nodes[i + 4];
        ggml_tensor * n5  = cgraph->nodes[i + 5];
        ggml_tensor * n6  = cgraph->nodes[i + 6];
        ggml_tensor * n7  = cgraph->nodes[i + 7];
        ggml_tensor * n8  = cgraph->nodes[i + 8];
        ggml_tensor * n9  = cgraph->nodes[i + 9];
        ggml_tensor * n10 = cgraph->nodes[i + 10];
        ggml_tensor * n11 = cgraph->nodes[i + 11];
        auto safe_alias = [](const ggml_tensor * dst, const ggml_tensor * src) {
            return !s11_tensors_overlap(dst, src) || dst->data == src->data;
        };
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->ne[0] == 1 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_BF16 &&
            n2->op == GGML_OP_CPY && n2->src[0] == n1 && n2->type == GGML_TYPE_F32 &&
            n3->op == GGML_OP_MUL && n3->src[1] == n2 && n3->src[0] && n3->src[0]->type == GGML_TYPE_F32 &&
            n4->op == GGML_OP_CPY && n4->src[0] == n3 && n4->type == GGML_TYPE_BF16 &&
            n5->op == GGML_OP_CPY && n5->src[0] == n4 && n5->type == GGML_TYPE_F32 &&
            n6->op == GGML_OP_ADD && n6->src[1] == n5 && n6->src[0] && n6->src[0]->type == GGML_TYPE_F32 &&
            n7->op == GGML_OP_CPY && n7->src[0] == n6 && n7->type == GGML_TYPE_BF16 &&
            n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
            n9->op == GGML_OP_ADD && n9->src[0] == n8 && n9->src[1] && n9->src[1]->type == GGML_TYPE_F32 &&
            n10->op == GGML_OP_CPY && n10->src[0] == n9 && n10->type == GGML_TYPE_BF16 &&
            n11->op == GGML_OP_CPY && n11->src[0] == n10 && n11->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n3->src[0]) &&
            ggml_is_contiguous(n6->src[0]) && ggml_is_contiguous(n9->src[1]) &&
            ggml_is_contiguous(n11) &&
            ggml_are_same_shape(n3->src[0], n6->src[0]) &&
            ggml_are_same_shape(n6->src[0], n9->src[1]) &&
            ggml_are_same_shape(n9->src[1], n11) &&
            n0->src[0]->ne[1] == n11->ne[1] &&
            !s11_tensors_overlap(n11, n0->src[0]) &&
            ((n11->ne[0] == 2048 && n11->ne[1] == 1) ||
             (safe_alias(n11, n3->src[0]) && safe_alias(n11, n6->src[0]) && safe_alias(n11, n9->src[1])))) {
            bool external = false;
            for (int m = 0; m < 11; ++m) {
                if (s11_has_external_consumer(cgraph, i + 12, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int k = int(n11->ne[0]);
                const int64_t total = ggml_nelements(n11);
                const int blocks = (k == 2048 && total == 2048) ? 1 : int((total + 255) / 256);
                ggml_cuda_s11_invalidate_if_written(n11->data);
                s11_shexp_epilogue_fused_kernel<<<blocks, 256, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    static_cast<const float *>(n3->src[0]->data),
                    static_cast<const float *>(n6->src[0]->data),
                    static_cast<const float *>(n9->src[1]->data),
                    static_cast<float *>(n11->data), k, total);
                CUDA_CHECK(cudaGetLastError());
                return 11;
            }
        }
    }

    // Pattern F+A: 9-node ADD + BF16 Round + RMSNorm + Quantize
    if (n0->op == GGML_OP_ADD && i + 8 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        ggml_tensor * n6 = cgraph->nodes[i + 6];
        ggml_tensor * n7 = cgraph->nodes[i + 7];
        ggml_tensor * n8 = cgraph->nodes[i + 8];
        auto safe_alias = [](const ggml_tensor * dst, const ggml_tensor * src) {
            return !s11_tensors_overlap(dst, src) || dst->data == src->data;
        };
        if (n0->src[0] && n0->src[1] &&
            n0->src[0]->type == GGML_TYPE_F32 && n0->src[1]->type == GGML_TYPE_F32 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_BF16 &&
            n2->op == GGML_OP_CPY && n2->src[0] == n1 && n2->type == GGML_TYPE_F32 &&
            n3->op == GGML_OP_RMS_NORM && n3->src[0] == n2 && n3->ne[0] == 2048 &&
            n4->op == GGML_OP_CPY && n4->src[0] && n4->src[0]->type == GGML_TYPE_BF16 && n4->type == GGML_TYPE_F32 &&
            n5->op == GGML_OP_SCALE && n5->src[0] == n4 &&
            n6->op == GGML_OP_MUL && n6->src[0] == n3 && n6->src[1] == n5 &&
            n7->op == GGML_OP_CPY && n7->src[0] == n6 && n7->type == GGML_TYPE_BF16 &&
            n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(n0->src[0], n0->src[1]) &&
            ggml_are_same_shape(n0->src[0], n2) &&
            ggml_are_same_shape(n2, n8) &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0->src[1]) &&
            ggml_is_contiguous(n2) && ggml_is_contiguous(n4->src[0]) && ggml_is_contiguous(n8) &&
            ggml_nelements(n4->src[0]) == 2048 &&
            !s11_tensors_overlap(n2, n8) &&
            !s11_tensors_overlap(n2, n4->src[0]) && !s11_tensors_overlap(n8, n4->src[0]) &&
            safe_alias(n2, n0->src[0]) && safe_alias(n2, n0->src[1]) &&
            safe_alias(n8, n0->src[0]) && safe_alias(n8, n0->src[1])) {
            bool external = false;
            for (int m = 0; m < 8; ++m) {
                if (m == 2) continue;
                if (s11_has_external_consumer(cgraph, i + 9, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                float eps = 1e-6f, scale = 1.0f, bias = 1.0f;
                std::memcpy(&eps, n3->op_params, sizeof(float));
                std::memcpy(&scale, reinterpret_cast<const char *>(n5->op_params) + 0, sizeof(float));
                std::memcpy(&bias, reinterpret_cast<const char *>(n5->op_params) + 4, sizeof(float));
                const int nrows = int(ggml_nrows(n3));
                ggml_cuda_s11_invalidate_if_written(n2->data);
                ggml_cuda_s11_invalidate_if_written(n8->data);
                const bool fuse_quant = (size_t(nrows) * 2048 <= g_s11_state.q_cap[0]) &&
                                        (size_t(nrows) * 16   <= g_s11_state.s_cap[0]);
                s11_residual_rms_norm_quant_fused_kernel<<<nrows, 256, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    static_cast<const float *>(n0->src[1]->data),
                    static_cast<float *>(n2->data),
                    static_cast<const nv_bfloat16 *>(n4->src[0]->data),
                    static_cast<float *>(n8->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr,
                    eps, scale, bias);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n8->data;
                    g_s11_state.last_k[0]           = 2048;
                    g_s11_state.last_tokens[0]      = nrows;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 8;
            }
        }
    }

    // Pattern E-Prefill: 9-compute-node post-down_exps 8-slot reduction + BF16 round (tokens >= 8)
    if (n0->op == GGML_OP_ADD && i + 8 < cgraph->n_nodes) {
        const ggml_tensor * v0 = n0->src[0];
        const ggml_tensor * v1 = n0->src[1];
        const ggml_tensor * down_exps = (v0 && v0->op == GGML_OP_VIEW) ? v0->src[0] : nullptr;
        if (down_exps && down_exps->type == GGML_TYPE_F32 &&
            down_exps->ne[0] == 2048 && down_exps->ne[1] == 8 && down_exps->ne[2] >= 8 &&
            down_exps->nb[0] == sizeof(float) &&
            v0->op == GGML_OP_VIEW && v0->src[0] == down_exps && v0->view_offs == 0 &&
            v0->nb[0] == sizeof(float) && v0->nb[1] == down_exps->nb[2] &&
            v1 && v1->op == GGML_OP_VIEW && v1->src[0] == down_exps && v1->view_offs == down_exps->nb[1] &&
            v1->nb[0] == sizeof(float) && v1->nb[1] == down_exps->nb[2]) {
            bool match_reduce = true;
            for (int a = 1; a < 7; ++a) {
                ggml_tensor * an = cgraph->nodes[i + a];
                const ggml_tensor * vk = an ? an->src[1] : nullptr;
                if (!an || an->op != GGML_OP_ADD || an->src[0] != cgraph->nodes[i + a - 1] ||
                    !vk || vk->op != GGML_OP_VIEW || vk->src[0] != down_exps ||
                    vk->view_offs != size_t(a + 1) * down_exps->nb[1] ||
                    vk->nb[0] != sizeof(float) || vk->nb[1] != down_exps->nb[2]) {
                    match_reduce = false;
                    break;
                }
            }
            ggml_tensor * n7 = cgraph->nodes[i + 7];
            ggml_tensor * n8 = cgraph->nodes[i + 8];
            if (match_reduce &&
                n7->op == GGML_OP_CPY && n7->src[0] == cgraph->nodes[i + 6] && n7->type == GGML_TYPE_BF16 &&
                n8->op == GGML_OP_CPY && n8->src[0] == n7 && n8->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(n8) && n8->ne[0] == 2048 && n8->ne[1] == down_exps->ne[2] &&
                !s11_tensors_overlap(n8, down_exps)) {
                for (int m = 0; m < 8; ++m) {
                    if (s11_has_external_consumer(cgraph, i + 9, cgraph->nodes[i + m])) {
                        match_reduce = false;
                        break;
                    }
                }
                if (match_reduce) {
                    const int tokens = int(down_exps->ne[2]);
                    ggml_cuda_s11_invalidate_if_written(n8->data);
                    s11_moe_reduce8_round_bf16_kernel<<<dim3(2, tokens), 256, 0, ctx.stream()>>>(
                        static_cast<const char *>(down_exps->data),
                        down_exps->nb[1], down_exps->nb[2],
                        static_cast<float *>(n8->data));
                    CUDA_CHECK(cudaGetLastError());
                    return 8;
                }
            }
        }
    }

    // Pattern F: 3-node ADD + CPY(F32 -> BF16) + CPY(BF16 -> F32)
    if (n0->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        if (n0->src[0] && n0->src[1] &&
            n0->src[0]->type == GGML_TYPE_F32 && n0->src[1]->type == GGML_TYPE_F32 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_BF16 &&
            n2->op == GGML_OP_CPY && n2->src[0] == n1 && n2->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(n0->src[0], n0->src[1]) &&
            ggml_are_same_shape(n0->src[0], n2) &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0->src[1]) &&
            ggml_is_contiguous(n0) && ggml_is_contiguous(n1) && ggml_is_contiguous(n2) &&
            (ggml_nelements(n2) == 2048 ||
             ((!s11_tensors_overlap(n2, n0->src[0]) || n2->data == n0->src[0]->data) &&
              (!s11_tensors_overlap(n2, n0->src[1]) || n2->data == n0->src[1]->data))) &&
            !s11_has_external_consumer(cgraph, i + 3, n0) &&
            !s11_has_external_consumer(cgraph, i + 3, n1)) {
            const int64_t count = ggml_nelements(n2);
            const int blocks = (count == 2048) ? 1 : int((count + 255) / 256);
            ggml_cuda_s11_invalidate_if_written(n2->data);
            s11_add_round_bf16_f32_kernel<<<blocks, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(n0->src[0]->data),
                static_cast<const float *>(n0->src[1]->data),
                static_cast<float *>(n2->data), count);
            CUDA_CHECK(cudaGetLastError());
            return 2;
        }
    }

    // Pattern H-Conv: 15-node CPY(ssm_conv1d, F32) + SSM_CONV + SILU + Pattern H pipeline
    if (n0->op == GGML_OP_CPY && i + 14 < cgraph->n_nodes &&
        n0->src[0] && n0->src[0]->type == GGML_TYPE_BF16 && n0->type == GGML_TYPE_F32 &&
        n0->ne[0] == 4 && n0->ne[1] == 8192 && ggml_is_contiguous(n0->src[0])) {
        ggml_tensor * c_conv = cgraph->nodes[i + 1];
        ggml_tensor * c_silu = cgraph->nodes[i + 2];
        ggml_tensor * h0  = cgraph->nodes[i + 3];
        ggml_tensor * h1  = cgraph->nodes[i + 4];
        ggml_tensor * h2  = cgraph->nodes[i + 5];
        ggml_tensor * h3  = cgraph->nodes[i + 6];
        ggml_tensor * h4  = cgraph->nodes[i + 7];
        ggml_tensor * h5  = cgraph->nodes[i + 8];
        ggml_tensor * h6  = cgraph->nodes[i + 9];
        ggml_tensor * h7  = cgraph->nodes[i + 10];
        ggml_tensor * h8  = cgraph->nodes[i + 11];
        ggml_tensor * h9  = cgraph->nodes[i + 12];
        ggml_tensor * h10 = cgraph->nodes[i + 13];
        ggml_tensor * h11 = cgraph->nodes[i + 14];
        if (c_conv->op == GGML_OP_SSM_CONV && c_conv->src[1] == n0 && c_conv->src[0] &&
            c_conv->src[0]->type == GGML_TYPE_F32 && c_conv->src[0]->nb[0] == sizeof(float) &&
            c_silu->op == GGML_OP_UNARY && ggml_get_unary_op(c_silu) == GGML_UNARY_OP_SILU && c_silu->src[0] == c_conv &&
            h0->op == GGML_OP_CPY && h0->src[0] == c_silu && h0->type == GGML_TYPE_BF16 &&
            h1->op == GGML_OP_CPY && h1->src[0] == h0 && h1->type == GGML_TYPE_F32 &&
            h1->ne[0] == 8192 && ggml_is_contiguous(h1) &&
            h2->op == GGML_OP_VIEW && h2->src[0] == h1 && h2->ne[0] == 128 && h2->ne[1] == 16 &&
            h2->view_offs == 0 && h2->nb[0] == sizeof(float) && h2->nb[1] == 128 * sizeof(float) &&
            h2->nb[2] == 8192 * sizeof(float) &&
            h3->op == GGML_OP_RMS_NORM && h3->src[0] == h2 &&
            h4->op == GGML_OP_SCALE && h4->src[0] == h3 &&
            h5->op == GGML_OP_CPY && h5->src[0] == h4 && h5->type == GGML_TYPE_BF16 &&
            h6->op == GGML_OP_CPY && h6->src[0] == h5 && h6->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(h6) &&
            h7->op == GGML_OP_VIEW && h7->src[0] == h1 && h7->ne[0] == 128 && h7->ne[1] == 16 &&
            h7->view_offs == 2048 * sizeof(float) && h7->nb[0] == sizeof(float) &&
            h7->nb[1] == 128 * sizeof(float) && h7->nb[2] == 8192 * sizeof(float) &&
            h8->op == GGML_OP_RMS_NORM && h8->src[0] == h7 &&
            h9->op == GGML_OP_SCALE && h9->src[0] == h8 &&
            h10->op == GGML_OP_CPY && h10->src[0] == h9 && h10->type == GGML_TYPE_BF16 &&
            h11->op == GGML_OP_CPY && h11->src[0] == h10 && h11->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(h11) &&
            !s11_tensors_overlap(h1, c_conv->src[0]) &&
            !s11_tensors_overlap(h6, c_conv->src[0]) && !s11_tensors_overlap(h11, c_conv->src[0]) &&
            !s11_tensors_overlap(h6, h1) && !s11_tensors_overlap(h11, h1) &&
            !s11_tensors_overlap(h6, h11)) {
            bool external = false;
            for (const int idx : {0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13}) {
                if (s11_has_external_consumer(cgraph, i + 15, cgraph->nodes[i + idx])) {
                    external = true;
                    break;
                }
            }
            float eps_q = 1e-6f, eps_k = 1e-6f, q_scale = 1.0f, k_scale = 1.0f;
            float q_bias = 0.0f, k_bias = 0.0f;
            std::memcpy(&eps_q, h3->op_params, sizeof(float));
            std::memcpy(&eps_k, h8->op_params, sizeof(float));
            std::memcpy(&q_scale, reinterpret_cast<const char *>(h4->op_params) + 0, sizeof(float));
            std::memcpy(&q_bias,  reinterpret_cast<const char *>(h4->op_params) + 4, sizeof(float));
            std::memcpy(&k_scale, reinterpret_cast<const char *>(h9->op_params) + 0, sizeof(float));
            std::memcpy(&k_bias,  reinterpret_cast<const char *>(h9->op_params) + 4, sizeof(float));
            if (!external && eps_q == eps_k && q_bias == 0.0f && k_bias == 0.0f) {
                const ggml_tensor * src0 = c_conv->src[0];
                const int64_t n_t = c_conv->ne[1];
                const int64_t n_s = c_conv->ne[2];
                ggml_cuda_s11_invalidate_if_written(h1->data);
                ggml_cuda_s11_invalidate_if_written(h6->data);
                ggml_cuda_s11_invalidate_if_written(h11->data);
                if (n_t <= 32) {
                    s11_ssm_conv_silu_l2norm_fused_kernel<<<dim3(n_s, 64, 1), 128, 0, ctx.stream()>>>(
                        static_cast<const float *>(src0->data),
                        static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                        int(src0->nb[1]), int(src0->nb[2]),
                        static_cast<float *>(h1->data),
                        static_cast<float *>(h6->data),
                        static_cast<float *>(h11->data),
                        eps_q, q_scale, k_scale, n_t);
                } else {
                    s11_ssm_conv_long_silu_l2norm_fused_kernel<<<dim3(n_s, 64, (n_t + 31) / 32), 128, 0, ctx.stream()>>>(
                        static_cast<const float *>(src0->data),
                        static_cast<const nv_bfloat16 *>(n0->src[0]->data),
                        int(src0->nb[0]), int(src0->nb[1]), int(src0->nb[2]),
                        static_cast<float *>(h1->data),
                        static_cast<float *>(h6->data),
                        static_cast<float *>(h11->data),
                        eps_q, q_scale, k_scale, n_t);
                }
                CUDA_CHECK(cudaGetLastError());
                return 14;
            }
        }
    }

    // Pattern H: 12-node GDN post-conv SiLU round + Q/K L2-norm + BF16 round
    if (n0->op == GGML_OP_CPY && i + 11 < cgraph->n_nodes) {
        ggml_tensor * n1  = cgraph->nodes[i + 1];
        ggml_tensor * n2  = cgraph->nodes[i + 2];
        ggml_tensor * n3  = cgraph->nodes[i + 3];
        ggml_tensor * n4  = cgraph->nodes[i + 4];
        ggml_tensor * n5  = cgraph->nodes[i + 5];
        ggml_tensor * n6  = cgraph->nodes[i + 6];
        ggml_tensor * n7  = cgraph->nodes[i + 7];
        ggml_tensor * n8  = cgraph->nodes[i + 8];
        ggml_tensor * n9  = cgraph->nodes[i + 9];
        ggml_tensor * n10 = cgraph->nodes[i + 10];
        ggml_tensor * n11 = cgraph->nodes[i + 11];
        const ggml_tensor * conv_silu_raw = n0->src[0];
        if (conv_silu_raw && conv_silu_raw->type == GGML_TYPE_F32 &&
            n0->type == GGML_TYPE_BF16 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            n1->ne[0] == 8192 && ggml_is_contiguous(conv_silu_raw) && ggml_is_contiguous(n1) &&
            n2->op == GGML_OP_VIEW && n2->src[0] == n1 && n2->ne[0] == 128 && n2->ne[1] == 16 &&
            n2->view_offs == 0 && n2->nb[0] == sizeof(float) && n2->nb[1] == 128 * sizeof(float) &&
            n2->nb[2] == 8192 * sizeof(float) &&
            n3->op == GGML_OP_RMS_NORM && n3->src[0] == n2 &&
            n4->op == GGML_OP_SCALE && n4->src[0] == n3 &&
            n5->op == GGML_OP_CPY && n5->src[0] == n4 && n5->type == GGML_TYPE_BF16 &&
            n6->op == GGML_OP_CPY && n6->src[0] == n5 && n6->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n6) &&
            n7->op == GGML_OP_VIEW && n7->src[0] == n1 && n7->ne[0] == 128 && n7->ne[1] == 16 &&
            n7->view_offs == 2048 * sizeof(float) && n7->nb[0] == sizeof(float) &&
            n7->nb[1] == 128 * sizeof(float) && n7->nb[2] == 8192 * sizeof(float) &&
            n8->op == GGML_OP_RMS_NORM && n8->src[0] == n7 &&
            n9->op == GGML_OP_SCALE && n9->src[0] == n8 &&
            n10->op == GGML_OP_CPY && n10->src[0] == n9 && n10->type == GGML_TYPE_BF16 &&
            n11->op == GGML_OP_CPY && n11->src[0] == n10 && n11->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n11) &&
            (!s11_tensors_overlap(n1, conv_silu_raw) || n1->data == conv_silu_raw->data) &&
            !s11_tensors_overlap(n6, conv_silu_raw) && !s11_tensors_overlap(n11, conv_silu_raw) &&
            !s11_tensors_overlap(n6, n1) && !s11_tensors_overlap(n11, n1) &&
            !s11_tensors_overlap(n6, n11)) {
            bool external = false;
            for (const int idx : {0, 2, 3, 4, 5, 7, 8, 9, 10}) {
                if (s11_has_external_consumer(cgraph, i + 12, cgraph->nodes[i + idx])) {
                    external = true;
                    break;
                }
            }
            float eps_q = 1e-6f, eps_k = 1e-6f, q_scale = 1.0f, k_scale = 1.0f;
            float q_bias = 0.0f, k_bias = 0.0f;
            std::memcpy(&eps_q, n3->op_params, sizeof(float));
            std::memcpy(&eps_k, n8->op_params, sizeof(float));
            std::memcpy(&q_scale, reinterpret_cast<const char *>(n4->op_params) + 0, sizeof(float));
            std::memcpy(&q_bias,  reinterpret_cast<const char *>(n4->op_params) + 4, sizeof(float));
            std::memcpy(&k_scale, reinterpret_cast<const char *>(n9->op_params) + 0, sizeof(float));
            std::memcpy(&k_bias,  reinterpret_cast<const char *>(n9->op_params) + 4, sizeof(float));
            if (!external && eps_q == eps_k && q_bias == 0.0f && k_bias == 0.0f) {
                const int tokens = int(n1->ne[1] * n1->ne[2] * n1->ne[3]);
                ggml_cuda_s11_invalidate_if_written(n1->data);
                ggml_cuda_s11_invalidate_if_written(n6->data);
                ggml_cuda_s11_invalidate_if_written(n11->data);
                s11_gdn_conv_l2norm_epilogue_kernel<<<dim3(64, tokens), 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(conv_silu_raw->data),
                    static_cast<float *>(n1->data),
                    static_cast<float *>(n6->data),
                    static_cast<float *>(n11->data),
                    eps_q, q_scale, k_scale);
                CUDA_CHECK(cudaGetLastError());
                return 11;
            }
        }
    }

    // Pattern C-Unary: 3-node UNARY(SIGMOID/SILU) + CPY(F32 -> BF16) + CPY(BF16 -> F32)
    if (n0->op == GGML_OP_UNARY &&
        (ggml_get_unary_op(n0) == GGML_UNARY_OP_SIGMOID || ggml_get_unary_op(n0) == GGML_UNARY_OP_SILU) &&
        i + 2 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_F32 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_BF16 &&
            n2->op == GGML_OP_CPY && n2->src[0] == n1 && n2->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0) &&
            ggml_is_contiguous(n1) && ggml_is_contiguous(n2) &&
            (!s11_tensors_overlap(n2, n0->src[0]) || n2->data == n0->src[0]->data) &&
            !s11_has_external_consumer(cgraph, i + 3, n0) &&
            !s11_has_external_consumer(cgraph, i + 3, n1)) {
            const int64_t count = ggml_nelements(n2);
            const bool is_silu = (ggml_get_unary_op(n0) == GGML_UNARY_OP_SILU);
            ggml_cuda_s11_invalidate_if_written(n2->data);
            s11_unary_round_bf16_f32_kernel<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(n0->src[0]->data),
                static_cast<float *>(n2->data), count, is_silu);
            CUDA_CHECK(cudaGetLastError());
            return 2;
        }
    }

    // Pattern FA-KV-SetRows: 6-node Kcur BF16 round + K/V FP16 cache SET_ROWS
    if (n0->op == GGML_OP_CPY && i + 5 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            n2->op == GGML_OP_VIEW && n2->src[0] == n1 && n2->view_offs == 0 &&
            n3->op == GGML_OP_SET_ROWS && n3->src[0] == n2 && n3->src[1] && n3->src[2] &&
            n3->type == GGML_TYPE_F16 && n3->src[1]->type == GGML_TYPE_I64 &&
            n4->op == GGML_OP_VIEW && n4->src[0] && n4->src[0]->type == GGML_TYPE_F32 && n4->view_offs == 0 &&
            n5->op == GGML_OP_SET_ROWS && n5->src[0] == n4 && n5->src[1] && n5->src[2] &&
            n5->type == GGML_TYPE_F16 && n5->src[1]->type == GGML_TYPE_I64 &&
            n2->ne[0] == 512 && n2->ne[2] == 1 && n2->ne[3] == 1 &&
            n4->ne[0] == 512 && n4->ne[1] == n2->ne[1] && n4->ne[2] == 1 && n4->ne[3] == 1 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n1) && ggml_is_contiguous(n4->src[0]) &&
            (!s11_tensors_overlap(n1, n0->src[0]) || n1->data == n0->src[0]->data) &&
            !s11_tensors_overlap(n1, n4->src[0])) {
            bool external = false;
            for (const int idx : {0, 2, 4}) {
                if (s11_has_external_consumer(cgraph, i + 6, cgraph->nodes[i + idx])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int tokens = int(n2->ne[1]);
                const bool write_k_round = s11_has_external_consumer(cgraph, i + 6, n1);
                if (write_k_round) {
                    ggml_cuda_s11_invalidate_if_written(n1->data);
                }
                s11_fa_kv_round_set_rows_f16_kernel<<<dim3(2, tokens), 256, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    static_cast<const float *>(n4->src[0]->data),
                    write_k_round ? static_cast<float *>(n1->data) : nullptr,
                    static_cast<char *>(n3->data),
                    static_cast<char *>(n5->data),
                    static_cast<const int64_t *>(n3->src[1]->data),
                    static_cast<const int64_t *>(n5->src[1]->data),
                    n3->nb[1], n5->nb[1]);
                CUDA_CHECK(cudaGetLastError());
                return 5;
            }
        }
    }

    // Pattern FA-Gate-8: 8-node post-FlashAttention cur BF16 round + strided gate view/cont + sigmoid + BF16 round + mul + A128 quantize
    if (n0->op == GGML_OP_CPY && i + 7 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        ggml_tensor * n2 = cgraph->nodes[i + 2];
        ggml_tensor * n3 = cgraph->nodes[i + 3];
        ggml_tensor * n4 = cgraph->nodes[i + 4];
        ggml_tensor * n5 = cgraph->nodes[i + 5];
        ggml_tensor * n6 = cgraph->nodes[i + 6];
        ggml_tensor * n7 = cgraph->nodes[i + 7];
        if (n0->src[0] && n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->op == GGML_OP_CPY && n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            n2->op == GGML_OP_VIEW && n2->src[0] && n2->src[0]->type == GGML_TYPE_F32 &&
            n2->ne[0] == 256 && n2->ne[1] == 16 && n2->nb[0] == sizeof(float) &&
            n3->op == GGML_OP_CONT && n3->src[0] == n2 && n3->type == GGML_TYPE_F32 &&
            n4->op == GGML_OP_UNARY && ggml_get_unary_op(n4) == GGML_UNARY_OP_SIGMOID && n4->src[0] == n3 &&
            n5->op == GGML_OP_CPY && n5->src[0] == n4 && n5->type == GGML_TYPE_BF16 &&
            n6->op == GGML_OP_CPY && n6->src[0] == n5 && n6->type == GGML_TYPE_F32 &&
            n7->op == GGML_OP_MUL && n7->src[0] == n1 && n7->src[1] == n6 &&
            n7->ne[0] == 4096 && ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n7) &&
            ggml_are_same_shape(n0->src[0], n7) &&
            (!s11_tensors_overlap(n7, n0->src[0]) || n7->data == n0->src[0]->data) &&
            !s11_tensors_overlap(n7, n2->src[0]) &&
            !s11_tensors_overlap(n0->src[0], n2->src[0])) {
            bool external = false;
            for (int m = 0; m < 7; ++m) {
                if (s11_has_external_consumer(cgraph, i + 8, cgraph->nodes[i + m])) {
                    external = true;
                    break;
                }
            }
            if (!external) {
                const int tokens = int(n7->ne[1] * n7->ne[2] * n7->ne[3]);
                const int group_rows = tokens * 32;
                ggml_cuda_s11_invalidate_if_written(n7->data);
                const bool fuse_quant = (size_t(group_rows) * 128 <= g_s11_state.q_cap[0]) &&
                                        (size_t(group_rows)       <= g_s11_state.s_cap[0]);
                s11_fa_gate_strided_quant_fused_kernel<<<group_rows, 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    static_cast<const char *>(n2->src[0]->data),
                    n2->nb[1], n2->nb[2], n2->view_offs,
                    static_cast<float *>(n7->data),
                    fuse_quant ? g_s11_state.q_buf[0] : nullptr,
                    fuse_quant ? g_s11_state.s_buf[0] : nullptr);
                CUDA_CHECK(cudaGetLastError());
                if (fuse_quant) {
                    g_s11_state.last_x_data[0]      = n7->data;
                    g_s11_state.last_k[0]           = 4096;
                    g_s11_state.last_tokens[0]      = tokens;
                    g_s11_state.last_input_slots[0] = 1;
                }
                return 7;
            }
        }
    }

    // Pattern C: 2-node CPY(F32 -> BF16) + CPY(BF16 -> F32)
    if (n0->op == GGML_OP_CPY && i + 1 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        if (n1->op == GGML_OP_CPY && n0->src[0] &&
            n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0) && ggml_is_contiguous(n1) &&
            (!s11_tensors_overlap(n1, n0->src[0]) || n1->data == n0->src[0]->data) &&
            !s11_has_external_consumer(cgraph, i + 2, n0)) {
            const int64_t count = ggml_nelements(n1);
            ggml_cuda_s11_invalidate_if_written(n1->data);
            s11_round_bf16_f32_kernel<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(n0->src[0]->data),
                static_cast<float *>(n1->data), count);
            CUDA_CHECK(cudaGetLastError());
            return 1;
        }
    }

    // Pattern E: 18-node Down-Exps + 8 Views + 7 Adds + BF16 round in Decode (tokens < 8)
    if (n0->op == GGML_OP_MUL_MAT_ID && n0->src[0]->type == GGML_TYPE_S11_W2 &&
        n0->src[0]->ne[0] == 512 && n0->src[0]->ne[1] == 2048 &&
        n0->ne[1] == 8 && n0->ne[2] < 8 && !s11_is_hybrid_mode() &&
        i + 17 < cgraph->n_nodes) {
        bool match_down = true;
        for (int v = 0; v < 8; ++v) {
            ggml_tensor * vn = cgraph->nodes[i + 1 + v];
            if (vn->op != GGML_OP_VIEW || vn->src[0] != n0) { match_down = false; break; }
        }
        for (int a = 0; a < 7 && match_down; ++a) {
            ggml_tensor * an = cgraph->nodes[i + 9 + a];
            if (an->op != GGML_OP_ADD) { match_down = false; break; }
            const ggml_tensor * expected_lhs = (a == 0) ? cgraph->nodes[i + 1] : cgraph->nodes[i + 8 + a];
            const ggml_tensor * expected_rhs = cgraph->nodes[i + 2 + a];
            if (an->src[0] != expected_lhs || an->src[1] != expected_rhs) { match_down = false; break; }
        }
        ggml_tensor * c1 = cgraph->nodes[i + 16];
        ggml_tensor * c2 = cgraph->nodes[i + 17];
        if (match_down &&
            c1->op == GGML_OP_CPY && c1->src[0] == cgraph->nodes[i + 15] && c1->type == GGML_TYPE_BF16 &&
            c2->op == GGML_OP_CPY && c2->src[0] == c1 && c2->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(c2)) {
            for (int m = 0; m < 17; ++m) {
                if (s11_has_external_consumer(cgraph, i + 18, cgraph->nodes[i + m])) {
                    match_down = false;
                    break;
                }
            }
            if (match_down) {
                const ggml_tensor * w = n0->src[0];
                const ggml_tensor * x = n0->src[1];
                const ggml_tensor * ids = n0->src[2];
                const int k = 512, n = 2048, slots = 8, tokens = int(n0->ne[2]);
                signed char * q_ptr = g_s11_state.q_buf[1];
                float * scales_ptr = g_s11_state.s_buf[1];
                int32_t * safe_ids = g_s11_state.plan_buf;
                s11_quantize<<<dim3(k / 128, tokens * slots), 32, 0, ctx.stream()>>>(
                    static_cast<const char *>(x->data), q_ptr, scales_ptr, nullptr, k,
                    slots, tokens, x->nb[1], x->nb[2]);
                const char * ids_ptr = static_cast<const char *>(ids->data);
                size_t ids_nb0 = ids->nb[0];
                size_t ids_nb1 = ids->nb[1];
                if (s11_tensors_overlap(c2, ids)) {
                    s11_copy_ids_kernel<<<1, 32, 0, ctx.stream()>>>(
                        ids_ptr, safe_ids, slots, tokens, ids_nb0, ids_nb1);
                    ids_ptr = reinterpret_cast<const char *>(safe_ids);
                    ids_nb0 = sizeof(int32_t);
                    ids_nb1 = size_t(slots) * sizeof(int32_t);
                }
                CUDA_CHECK(cudaGetLastError());
                g_s11_state.last_x_data[1] = x->data;
                g_s11_state.last_k[1] = k;
                g_s11_state.last_tokens[1] = tokens;
                g_s11_state.last_input_slots[1] = slots;
                g_s11_state.last_ids_data = nullptr;
                ggml_cuda_s11_invalidate_if_written(c2->data);
                if (tokens == 1) {
                    s11_fused_down_exps_reduce_kernel_m1<<<dim3((n + 7) / 8, 1), 256, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr,
                        ids_ptr, static_cast<char *>(c2->data),
                        n, slots, w->nb[1], w->nb[2], ids_nb0);
                } else {
                    s11_fused_down_exps_reduce_kernel<<<dim3((n + 3) / 4, tokens), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr,
                        ids_ptr, static_cast<char *>(c2->data),
                        k, n, slots, tokens, w->nb[1], w->nb[2], ids_nb0, ids_nb1, c2->nb[1]);
                }
                CUDA_CHECK(cudaGetLastError());
                return 17;
            }
        }
    }

    // Pattern W4-Multi: 2 or 3 consecutive S11_W4 MUL_MAT nodes sharing identical input x (k == 2048)
    if (n0->op == GGML_OP_MUL_MAT && i + 1 < cgraph->n_nodes &&
        n0->src[0] && n0->src[0]->type == GGML_TYPE_S11_W4 &&
        n0->src[0]->ne[0] == 2048 && (n0->src[0]->ne[1] % 64 == 0) &&
        n0->src[1] && n0->src[1]->type == GGML_TYPE_F32 && n0->src[1]->nb[0] == sizeof(float) &&
        n0->src[0]->ne[2] == 1 && n0->src[0]->ne[3] == 1 &&
        n0->src[1]->ne[2] == 1 && n0->src[1]->ne[3] == 1) {
        ggml_tensor * m0 = n0;
        ggml_tensor * m1 = cgraph->nodes[i + 1];
        const ggml_tensor * x = m0->src[1];
        if (m1->op == GGML_OP_MUL_MAT && m1->src[1] == x &&
            m1->src[0] && m1->src[0]->type == GGML_TYPE_S11_W4 &&
            m1->src[0]->ne[0] == 2048 && (m1->src[0]->ne[1] % 64 == 0) &&
            m1->src[0]->ne[2] == 1 && m1->src[0]->ne[3] == 1 &&
            m1->ne[1] == m0->ne[1] &&
            !s11_tensors_overlap(m0, x) && !s11_tensors_overlap(m1, x) &&
            !s11_tensors_overlap(m0, m1)) {
            ggml_tensor * m2 = nullptr;
            if (i + 2 < cgraph->n_nodes) {
                ggml_tensor * cand = cgraph->nodes[i + 2];
                if (cand->op == GGML_OP_MUL_MAT && cand->src[1] == x &&
                    cand->src[0] && cand->src[0]->type == GGML_TYPE_S11_W4 &&
                    cand->src[0]->ne[0] == 2048 && (cand->src[0]->ne[1] % 64 == 0) &&
                    cand->src[0]->ne[2] == 1 && cand->src[0]->ne[3] == 1 &&
                    cand->ne[1] == m0->ne[1] &&
                    !s11_tensors_overlap(cand, x) &&
                    !s11_tensors_overlap(cand, m0) && !s11_tensors_overlap(cand, m1)) {
                    m2 = cand;
                }
            }
            const int tokens = int(m0->ne[1]);
            const int n0_rows = int(m0->src[0]->ne[1]);
            const int n1_rows = int(m1->src[0]->ne[1]);
            const int n2_rows = m2 ? int(m2->src[0]->ne[1]) : 0;
            signed char * q_ptr = g_s11_state.q_buf[0];
            float * scales_ptr = g_s11_state.s_buf[0];
            const bool can_reuse = (x->view_src == nullptr || (x->view_offs == 0 && ggml_is_contiguous(x))) &&
                                   (g_s11_state.last_x_data[0] == x->data) &&
                                   (g_s11_state.last_k[0] == 2048) &&
                                   (g_s11_state.last_tokens[0] == tokens) &&
                                   (g_s11_state.last_input_slots[0] == 1);
            if (!can_reuse) {
                s11_quantize<<<dim3(16, tokens), 32, 0, ctx.stream()>>>(
                    static_cast<const char *>(x->data), q_ptr, scales_ptr, nullptr, 2048,
                    1, tokens, 0, x->nb[1]);
                CUDA_CHECK(cudaGetLastError());
            }
            ggml_cuda_s11_invalidate_if_written(m0->data);
            ggml_cuda_s11_invalidate_if_written(m1->data);
            if (m2) ggml_cuda_s11_invalidate_if_written(m2->data);
            if (x->view_src == nullptr || (x->view_offs == 0 && ggml_is_contiguous(x))) {
                g_s11_state.last_x_data[0] = x->data;
                g_s11_state.last_k[0] = 2048;
                g_s11_state.last_tokens[0] = tokens;
                g_s11_state.last_input_slots[0] = 1;
            }

            const char * w0_ptr = static_cast<const char *>(m0->src[0]->data);
            const char * w1_ptr = static_cast<const char *>(m1->src[0]->data);
            const char * w2_ptr = m2 ? static_cast<const char *>(m2->src[0]->data) : nullptr;
            char * out0_ptr = static_cast<char *>(m0->data);
            char * out1_ptr = static_cast<char *>(m1->data);
            char * out2_ptr = m2 ? static_cast<char *>(m2->data) : nullptr;
            const size_t wr0 = m0->src[0]->nb[1];
            const size_t wr1 = m1->src[0]->nb[1];
            const size_t wr2 = m2 ? m2->src[0]->nb[1] : 0;
            const size_t ot0 = m0->nb[1];
            const size_t ot1 = m1->nb[1];
            const size_t ot2 = m2 ? m2->nb[1] : 0;

            if (tokens >= 32) {
                const int total_b = (n0_rows + n1_rows + n2_rows) / 64;
                s11_w4_multi_wmma_k128_kernel<2><<<dim3(total_b, (tokens + 31) / 32), 128, 0, ctx.stream()>>>(
                    w0_ptr, w1_ptr, w2_ptr, out0_ptr, out1_ptr, out2_ptr,
                    q_ptr, scales_ptr, n0_rows, n1_rows, n2_rows, tokens,
                    wr0, wr1, wr2, ot0, ot1, ot2);
            } else if (tokens >= 16) {
                const int total_b = (n0_rows + n1_rows + n2_rows) / 64;
                s11_w4_multi_wmma_k128_kernel<1><<<dim3(total_b, (tokens + 15) / 16), 128, 0, ctx.stream()>>>(
                    w0_ptr, w1_ptr, w2_ptr, out0_ptr, out1_ptr, out2_ptr,
                    q_ptr, scales_ptr, n0_rows, n1_rows, n2_rows, tokens,
                    wr0, wr1, wr2, ot0, ot1, ot2);
            } else if (tokens == 1) {
                const int total_b8 = (n0_rows + n1_rows + n2_rows) / 8;
                s11_w4_multi_projection_packed_m1<<<dim3(total_b8, 1), 256, 0, ctx.stream()>>>(
                    w0_ptr, w1_ptr, w2_ptr, out0_ptr, out1_ptr, out2_ptr,
                    q_ptr, scales_ptr, n0_rows, n1_rows, n2_rows, wr0, wr1, wr2);
            } else if (tokens >= 8) {
                const int total_b4 = (n0_rows + n1_rows + n2_rows) / 4;
                s11_w4_multi_projection_packed<8><<<dim3(total_b4, (tokens + 7) / 8), 128, 0, ctx.stream()>>>(
                    w0_ptr, w1_ptr, w2_ptr, out0_ptr, out1_ptr, out2_ptr,
                    q_ptr, scales_ptr, n0_rows, n1_rows, n2_rows, tokens,
                    wr0, wr1, wr2, ot0, ot1, ot2);
            } else if (tokens >= 4) {
                const int total_b4 = (n0_rows + n1_rows + n2_rows) / 4;
                s11_w4_multi_projection_packed<4><<<dim3(total_b4, (tokens + 3) / 4), 128, 0, ctx.stream()>>>(
                    w0_ptr, w1_ptr, w2_ptr, out0_ptr, out1_ptr, out2_ptr,
                    q_ptr, scales_ptr, n0_rows, n1_rows, n2_rows, tokens,
                    wr0, wr1, wr2, ot0, ot1, ot2);
            } else {
                const int total_b4 = (n0_rows + n1_rows + n2_rows) / 4;
                s11_w4_multi_projection_packed<1><<<dim3(total_b4, tokens), 128, 0, ctx.stream()>>>(
                    w0_ptr, w1_ptr, w2_ptr, out0_ptr, out1_ptr, out2_ptr,
                    q_ptr, scales_ptr, n0_rows, n1_rows, n2_rows, tokens,
                    wr0, wr1, wr2, ot0, ot1, ot2);
            }
            CUDA_CHECK(cudaGetLastError());
            return m2 ? 2 : 1;
        }
    }

    // Pattern D: Fused Up + Gate + SwiGLU (+ optional ProbWeight + BF16 round)
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * first_mm = cgraph->nodes[i];
    ggml_tensor * second_mm = cgraph->nodes[i + 1];
    ggml_tensor * glu_node = cgraph->nodes[i + 2];
    if ((first_mm->op != GGML_OP_MUL_MAT && first_mm->op != GGML_OP_MUL_MAT_ID) ||
        second_mm->op != first_mm->op || glu_node->op != GGML_OP_GLU) {
        return 0;
    }
    if (ggml_get_glu_op(glu_node) != GGML_GLU_OP_SWIGLU) {
        return 0;
    }
    ggml_tensor * gate_node = nullptr;
    ggml_tensor * up_node = nullptr;
    if (glu_node->src[0] == first_mm && glu_node->src[1] == second_mm) {
        gate_node = first_mm;
        up_node = second_mm;
    } else if (glu_node->src[0] == second_mm && glu_node->src[1] == first_mm) {
        gate_node = second_mm;
        up_node = first_mm;
    } else {
        return 0;
    }
    const ggml_tensor * w_up = up_node->src[0];
    const ggml_tensor * w_gate = gate_node->src[0];
    const ggml_tensor * x = up_node->src[1];
    const ggml_tensor * ids = up_node->op == GGML_OP_MUL_MAT_ID ? up_node->src[2] : nullptr;
    if (w_up->type != w_gate->type ||
        (w_up->type != GGML_TYPE_S11_W2 && w_up->type != GGML_TYPE_S11_W2_BF16)) {
        return 0;
    }
    if (gate_node->src[1] != x || (ids && gate_node->src[2] != ids)) {
        return 0;
    }
    if (!ggml_are_same_shape(w_up, w_gate) || !ggml_are_same_shape(up_node, gate_node) ||
        !ggml_are_same_shape(up_node, glu_node)) {
        return 0;
    }
    if (w_up->nb[1] != w_gate->nb[1] || w_up->nb[2] != w_gate->nb[2]) {
        return 0;
    }
    const int tokens = ids ? glu_node->ne[2] : glu_node->ne[1];
    const int slots = ids ? glu_node->ne[1] : 1;
    if (tokens >= 8 && !(w_up->type == GGML_TYPE_S11_W2 && ids && tokens * slots <= 4096 && !s11_is_hybrid_mode()) &&
        !(w_up->type == GGML_TYPE_S11_W2_BF16 && !ids)) {
        return 0;
    }

    int skip_nodes = 2;
    ggml_tensor * out_node = glu_node;
    const ggml_tensor * prob_weights = nullptr;

    // Check if followed by MUL(glu, prob_weights) -> CPY(BF16) -> CPY(F32)
    if (w_up->type == GGML_TYPE_S11_W2 && ids && i + 5 < cgraph->n_nodes) {
        ggml_tensor * mul_node = cgraph->nodes[i + 3];
        ggml_tensor * c1_node  = cgraph->nodes[i + 4];
        ggml_tensor * c2_node  = cgraph->nodes[i + 5];
        if (mul_node->op == GGML_OP_MUL && mul_node->src[0] == glu_node &&
            mul_node->src[1]->type == GGML_TYPE_F32 &&
            mul_node->src[1]->ne[0] == 1 && mul_node->src[1]->ne[1] == glu_node->ne[1] &&
            mul_node->src[1]->ne[2] == glu_node->ne[2] &&
            c1_node->op == GGML_OP_CPY && c1_node->src[0] == mul_node && c1_node->type == GGML_TYPE_BF16 &&
            c2_node->op == GGML_OP_CPY && c2_node->src[0] == c1_node && c2_node->type == GGML_TYPE_F32 &&
            !s11_has_external_consumer(cgraph, i + 6, glu_node) &&
            !s11_has_external_consumer(cgraph, i + 6, mul_node) &&
            !s11_has_external_consumer(cgraph, i + 6, c1_node) &&
            !s11_tensors_overlap(c2_node, ids) &&
            !s11_tensors_overlap(c2_node, mul_node->src[1])) {
            prob_weights = mul_node->src[1];
            out_node = c2_node;
            skip_nodes = 5;
        }
    }

    if (s11_has_external_consumer(cgraph, i + skip_nodes + 1, up_node) ||
        s11_has_external_consumer(cgraph, i + skip_nodes + 1, gate_node)) {
        return 0;
    }

    const int k = w_up->ne[0];
    const int n = w_up->ne[1];
    const int input_slots = ids ? x->ne[1] : 1;
    const int buf_idx = (input_slots == 1) ? 0 : 1;
    signed char * q_ptr = g_s11_state.q_buf[buf_idx];
    float * scales_ptr = g_s11_state.s_buf[buf_idx];
    const bool can_reuse = (x->view_src == nullptr || (x->view_offs == 0 && ggml_is_contiguous(x))) &&
                           (g_s11_state.last_x_data[buf_idx] == x->data) &&
                           (g_s11_state.last_k[buf_idx] == k) &&
                           (g_s11_state.last_tokens[buf_idx] == tokens) &&
                           (g_s11_state.last_input_slots[buf_idx] == input_slots);
    if (!can_reuse) {
        s11_quantize<<<dim3(k / 128, tokens * input_slots), 32, 0, ctx.stream()>>>(
            static_cast<const char *>(x->data), q_ptr, scales_ptr, nullptr, k,
            input_slots, tokens, ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1]);
        CUDA_CHECK(cudaGetLastError());
        g_s11_state.last_x_data[buf_idx] = x->data;
        g_s11_state.last_k[buf_idx] = k;
        g_s11_state.last_tokens[buf_idx] = tokens;
        g_s11_state.last_input_slots[buf_idx] = input_slots;
    }

    ggml_cuda_s11_invalidate_if_written(out_node->data);

    if (w_up->type == GGML_TYPE_S11_W2) {
        if (tokens >= 8) {
            constexpr size_t kPlanInts = 4096 * 4 + 16;
            if (g_s11_state.plan_cap < kPlanInts) {
                if (g_s11_state.plan_buf) CUDA_CHECK(cudaFree(g_s11_state.plan_buf));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_s11_state.plan_buf), kPlanInts * sizeof(int32_t)));
                g_s11_state.plan_cap = kPlanInts;
                g_s11_state.last_ids_data = nullptr;
            }
            int32_t * tile_experts = g_s11_state.plan_buf;
            int32_t * tile_starts  = g_s11_state.plan_buf + 4096;
            int32_t * tile_counts  = g_s11_state.plan_buf + 8192;
            int32_t * sorted_assn  = g_s11_state.plan_buf + 12288;
            int32_t * num_tiles    = g_s11_state.plan_buf + 16384;
            const bool reuse_plan = (g_s11_state.last_ids_data == ids->data) &&
                                    (g_s11_state.last_ids_tokens == tokens) &&
                                    (g_s11_state.last_ids_slots == slots) &&
                                    (g_s11_state.last_ids_nb0 == ids->nb[0]) &&
                                    (g_s11_state.last_ids_nb1 == ids->nb[1]);
            if (!reuse_plan) {
                s11_build_routed_plan_kernel<<<1, 256, 0, ctx.stream()>>>(
                    static_cast<const char *>(ids->data), slots, tokens, ids->nb[0], ids->nb[1],
                    tile_experts, tile_starts, tile_counts, sorted_assn, num_tiles);
                CUDA_CHECK(cudaGetLastError());
                g_s11_state.last_ids_data = ids->data;
                g_s11_state.last_ids_tokens = tokens;
                g_s11_state.last_ids_slots = slots;
                g_s11_state.last_ids_nb0 = ids->nb[0];
                g_s11_state.last_ids_nb1 = ids->nb[1];
            }
            const int m = tokens * slots;
            const int max_tiles = std::min(m, (m + 15) / 16 + std::min(m, 256));
            s11_routed_up_gate_swiglu_wmma_k128_kernel<<<dim3((n + 63) / 64, max_tiles), 128, 0, ctx.stream()>>>(
                static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
                q_ptr, scales_ptr, tile_experts, tile_starts, tile_counts, sorted_assn, num_tiles,
                prob_weights ? static_cast<const char *>(prob_weights->data) : nullptr,
                static_cast<char *>(out_node->data), k, n, slots, input_slots,
                w_up->nb[1], w_up->nb[2],
                prob_weights ? prob_weights->nb[1] : 0, prob_weights ? prob_weights->nb[2] : 0,
                out_node->nb[1], out_node->nb[2]);
        } else {
            if (prob_weights) {
                g_s11_state.last_ids_data = ids->data;
                g_s11_state.last_ids_tokens = tokens;
                g_s11_state.last_ids_slots = slots;
                g_s11_state.last_ids_nb0 = ids->nb[0];
                g_s11_state.last_ids_nb1 = ids->nb[1];
            }
            if (tokens == 1 && k == 2048) {
                s11_fused_up_gate_swiglu_kernel_m1<GGML_TYPE_S11_W2><<<dim3((n + 7) / 8, slots), 256, 0, ctx.stream()>>>(
                    static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
                    q_ptr, scales_ptr, ids ? static_cast<const char *>(ids->data) : nullptr,
                    prob_weights ? static_cast<const char *>(prob_weights->data) : nullptr,
                    static_cast<char *>(out_node->data), k, n, slots, input_slots,
                    w_up->nb[1], w_up->nb[2], ids ? ids->nb[0] : 0,
                    prob_weights ? prob_weights->nb[1] : 0, out_node->nb[1]);
            } else {
                s11_fused_up_gate_swiglu_kernel<GGML_TYPE_S11_W2><<<dim3((n + 3) / 4, tokens * slots), 128, 0, ctx.stream()>>>(
                    static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
                    q_ptr, scales_ptr, ids ? static_cast<const char *>(ids->data) : nullptr,
                    prob_weights ? static_cast<const char *>(prob_weights->data) : nullptr,
                    static_cast<char *>(out_node->data), k, n, slots, tokens, input_slots,
                    w_up->nb[1], w_up->nb[2], ids ? ids->nb[0] : 0, ids ? ids->nb[1] : 0,
                    prob_weights ? prob_weights->nb[1] : 0, prob_weights ? prob_weights->nb[2] : 0,
                    out_node->nb[1], out_node->nb[2]);
            }
        }
    } else {
        if (tokens >= 16) {
            s11_shexp_up_gate_swiglu_wmma_k128_kernel<<<dim3((n + 63) / 64, (tokens + 15) / 16), 128, 0, ctx.stream()>>>(
                static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
                q_ptr, scales_ptr, static_cast<char *>(out_node->data),
                k, n, tokens, w_up->nb[1], out_node->nb[1]);
        } else if (tokens == 1 && k == 2048) {
            s11_fused_up_gate_swiglu_kernel_m1<GGML_TYPE_S11_W2_BF16><<<dim3((n + 7) / 8, 1), 256, 0, ctx.stream()>>>(
                static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
                q_ptr, scales_ptr, nullptr, nullptr,
                static_cast<char *>(out_node->data), k, n, 1, 1,
                w_up->nb[1], w_up->nb[2], 0, 0, 0);
        } else {
            s11_fused_up_gate_swiglu_kernel<GGML_TYPE_S11_W2_BF16><<<dim3((n + 3) / 4, tokens * slots), 128, 0, ctx.stream()>>>(
                static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
                q_ptr, scales_ptr, nullptr, nullptr,
                static_cast<char *>(out_node->data), k, n, 1, tokens, 1,
                w_up->nb[1], w_up->nb[2], 0, 0, 0, 0,
                0, out_node->nb[1]);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return skip_nodes;
}

static __global__ void s11_rows(const char * weights, const int * ids, float * output, int k, int rows, size_t row_stride) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * k) {
        return;
    }
    const int column = index % k;
    const unsigned char * block = reinterpret_cast<const unsigned char *>(weights + ids[index / k] * row_stride + (column / 128) * 132);
    output[index] = s11_bf16(__fdiv_rn(float(reinterpret_cast<const signed char *>(block)[column % 128]), s11_scale(block, GGML_TYPE_S11_I8)));
}

void ggml_cuda_s11_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    GGML_ASSERT(weights->type == GGML_TYPE_S11_I8 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_cuda_s11_bf16_head(weights) && "S11 derived head supports immutable matmul, not embedding lookup");
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ggml_is_contiguous(dst));
    GGML_ASSERT(weights->ne[2] == 1 && weights->ne[3] == 1);
    const int count = ggml_nelements(ids);
    s11_rows<<<(count * weights->ne[0] + 255) / 256, 256, 0, ctx.stream()>>>(
        static_cast<const char *>(weights->data), static_cast<const int *>(ids->data),
        static_cast<float *>(dst->data), weights->ne[0], count, weights->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

uint64_t ggml_cuda_s11_assignments() {
    unsigned long long value = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&value, s11_completed_assignments, sizeof(value)));
    return value;
}

static __device__ float s11_read_float(const char * pointer, int type) {
    return type == GGML_TYPE_BF16 ? __bfloat162float(*reinterpret_cast<const nv_bfloat16 *>(pointer))
        : type == GGML_TYPE_F16 ? __half2float(*reinterpret_cast<const half *>(pointer))
        : *reinterpret_cast<const float *>(pointer);
}

static __global__ void __launch_bounds__(128) s11_dense_kernel(
        const char * a, const char * b, char * output,
        int atype, int btype, int k, int n, int m, int heads, int a_heads, int a_sequences,
        size_t a0, size_t a1, size_t a2, size_t a3,
        size_t b0, size_t b1, size_t b2, size_t b3,
        size_t d1, size_t d2, size_t d3) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int column = blockIdx.y % m;
    const int head = (blockIdx.y / m) % heads;
    const int sequence = blockIdx.y / (m * heads);
    const char * x = b + column * b1 + head * b2 + sequence * b3;

    if (k == 2048 && btype == GGML_TYPE_F32 && b0 == sizeof(float)) {
        __shared__ __align__(16) float x_sh[2048];
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        float4 * x_sh4 = reinterpret_cast<float4 *>(x_sh);
        #pragma unroll
        for (int i = threadIdx.x; i < 512; i += 128) {
            x_sh4[i] = x4[i];
        }
        __syncthreads();
        if (row >= n) {
            return;
        }
        const char * w = a + row * a1 + (head / (heads / a_heads)) * a2 + (sequence / a_sequences) * a3;
        float sum = 0.0f;
        if (atype == GGML_TYPE_BF16 && a0 == sizeof(nv_bfloat16)) {
            const nv_bfloat16 * w_bf16 = reinterpret_cast<const nv_bfloat16 *>(w);
            #pragma unroll 8
            for (int index = lane; index < 2048; index += 32) {
                sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(w_bf16[index]), x_sh[index]));
            }
        } else {
            for (int index = lane; index < 2048; index += 32) {
                sum = __fadd_rn(sum, __fmul_rn(s11_read_float(w + index * a0, atype), x_sh[index]));
            }
        }
        for (int offset = 16; offset; offset /= 2) {
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        if (lane == 0) {
            *reinterpret_cast<float *>(output + row * sizeof(float) + column * d1 + head * d2 + sequence * d3) = sum;
        }
        return;
    }

    if (row >= n) {
        return;
    }
    const char * w = a + row * a1 + (head / (heads / a_heads)) * a2 + (sequence / a_sequences) * a3;
    float sum = 0;
    for (int index = lane; index < k; index += 32) {
        sum = __fadd_rn(sum, __fmul_rn(s11_read_float(w + index * a0, atype), s11_read_float(x + index * b0, btype)));
    }
    for (int offset = 16; offset; offset /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (lane == 0) {
        *reinterpret_cast<float *>(output + row * sizeof(float) + column * d1 + head * d2 + sequence * d3) = sum;
    }
}

template <int COLS>
static __global__ void __launch_bounds__(128) s11_dense_tiled_kernel(
        const char * __restrict__ a, const char * __restrict__ b, char * __restrict__ output,
        int atype, int btype, int k, int n, int m, int heads, int a_heads, int a_sequences,
        size_t a0, size_t a1, size_t a2, size_t a3,
        size_t b0, size_t b1, size_t b2, size_t b3,
        size_t d1, size_t d2, size_t d3) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int tile_idx = blockIdx.y;
    const int m_tiles = (m + COLS - 1) / COLS;
    const int first_col = (tile_idx % m_tiles) * COLS;
    const int head = (tile_idx / m_tiles) % heads;
    const int sequence = tile_idx / (m_tiles * heads);
    const char * w = (row < n)
        ? (a + size_t(row) * a1 + size_t(head / (heads / a_heads)) * a2 + size_t(sequence / a_sequences) * a3)
        : nullptr;
    const char * x_base = b + size_t(head) * b2 + size_t(sequence) * b3;
    float sums[COLS] = {};

    if (k == 2048 && btype == GGML_TYPE_F32 && b0 == sizeof(float)) {
        __shared__ float x_sh[COLS][512];
        for (int base_k = 0; base_k < 2048; base_k += 512) {
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int col = first_col + c;
                if (col < m) {
                    const float * x_col = reinterpret_cast<const float *>(x_base + size_t(col) * b1) + base_k;
                    #pragma unroll
                    for (int idx = threadIdx.x; idx < 512; idx += 128) {
                        x_sh[c][idx] = x_col[idx];
                    }
                } else {
                    #pragma unroll
                    for (int idx = threadIdx.x; idx < 512; idx += 128) {
                        x_sh[c][idx] = 0.0f;
                    }
                }
            }
            __syncthreads();
            if (row < n) {
                if (atype == GGML_TYPE_BF16 && a0 == sizeof(nv_bfloat16)) {
                    const nv_bfloat16 * w_bf16 = reinterpret_cast<const nv_bfloat16 *>(w) + base_k;
                    #pragma unroll 8
                    for (int idx = lane; idx < 512; idx += 32) {
                        const float w_val = __bfloat162float(w_bf16[idx]);
                        #pragma unroll
                        for (int c = 0; c < COLS; ++c) {
                            sums[c] = __fadd_rn(sums[c], __fmul_rn(w_val, x_sh[c][idx]));
                        }
                    }
                } else {
                    for (int idx = lane; idx < 512; idx += 32) {
                        const float w_val = s11_read_float(w + size_t(base_k + idx) * a0, atype);
                        #pragma unroll
                        for (int c = 0; c < COLS; ++c) {
                            sums[c] = __fadd_rn(sums[c], __fmul_rn(w_val, x_sh[c][idx]));
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (row >= n) {
            return;
        }
    } else {
        if (row >= n) {
            return;
        }
        for (int index = lane; index < k; index += 32) {
            const float w_val = s11_read_float(w + size_t(index) * a0, atype);
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int col = first_col + c;
                if (col < m) {
                    const float x_val = s11_read_float(x_base + size_t(col) * b1 + size_t(index) * b0, btype);
                    sums[c] = __fadd_rn(sums[c], __fmul_rn(w_val, x_val));
                }
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            sums[c] = __fadd_rn(sums[c], __shfl_down_sync(0xffffffff, sums[c], offset));
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int col = first_col + c;
            if (col < m) {
                *reinterpret_cast<float *>(output + size_t(row) * sizeof(float) + size_t(col) * d1 + size_t(head) * d2 + size_t(sequence) * d3) = sums[c];
            }
        }
    }
}

void ggml_cuda_s11_dense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_BF16 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0);
    const int m = int(dst->ne[1]);
    const int heads = int(dst->ne[2]);
    const int seqs = int(dst->ne[3]);
    if (s11_optimization() >= 2 && m >= 8) {
        const int m_tiles = (m + 7) / 8;
        const int y_blocks = m_tiles * heads * seqs;
        if (y_blocks <= 65535) {
            s11_dense_tiled_kernel<8><<<dim3((dst->ne[0] + 3) / 4, y_blocks), 128, 0, ctx.stream()>>>(
                static_cast<const char *>(a->data), static_cast<const char *>(b->data), static_cast<char *>(dst->data),
                a->type, b->type, a->ne[0], dst->ne[0], m, heads, a->ne[2], b->ne[3] / a->ne[3],
                a->nb[0], a->nb[1], a->nb[2], a->nb[3], b->nb[0], b->nb[1], b->nb[2], b->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const int columns = dst->ne[1] * dst->ne[2] * dst->ne[3];
    GGML_ASSERT(columns <= 65535);
    s11_dense_kernel<<<dim3((dst->ne[0] + 3) / 4, columns), 128, 0, ctx.stream()>>>(
        static_cast<const char *>(a->data), static_cast<const char *>(b->data), static_cast<char *>(dst->data),
        a->type, b->type, a->ne[0], dst->ne[0], dst->ne[1], dst->ne[2], a->ne[2], b->ne[3] / a->ne[3],
        a->nb[0], a->nb[1], a->nb[2], a->nb[3], b->nb[0], b->nb[1], b->nb[2], b->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
