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

template <int TYPE>
static __global__ void __launch_bounds__(128) s11_wmma_k128_kernel(
        const char * __restrict__ weights, const signed char * __restrict__ q,
        const float * __restrict__ activation_scales, char * __restrict__ output,
        int k, int n, int tokens, size_t weight_row, size_t output_token) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int bytes = TYPE <= GGML_TYPE_S11_W2_BF16 ? 36 : 68;
    const int t = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int first_row = blockIdx.x * 64;
    const int first_token = blockIdx.y * 16;
    if (first_row >= n || first_token >= tokens) {
        return;
    }

    __shared__ __align__(32) signed char a_sh[16 * 128];
    __shared__ __align__(32) signed char b_sh[64 * 128];
    __shared__ __align__(32) int dots_sh[4][16 * 16];
    __shared__ float a_scales_sh[16];
    __shared__ float w_scales_sh[64];

    float sums[8] = {};

    for (int group = 0; group < k / 128; ++group) {
        if (t < 16) {
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
        for (int idx = t; idx < 16 * 32; idx += 128) {
            const int tok_local = idx / 32;
            const int col4 = idx % 32;
            const int tok = first_token + tok_local;
            *reinterpret_cast<int *>(&a_sh[tok_local * 128 + col4 * 4]) = (tok < tokens)
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
            *reinterpret_cast<int *>(&b_sh[r_local * 128 + col4 * 4]) = packed;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
        wmma::fill_fragment(acc, 0);
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a_sh + ks, 128);
            wmma::load_matrix_sync(bf, b_sh + (warp * 16) * 128 + ks, 128);
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
        const int tok = first_token + idx / 16;
        const int row = first_row + warp * 16 + idx % 16;
        if (tok < tokens && row < n) {
            *reinterpret_cast<float *>(output + size_t(tok) * output_token + size_t(row) * sizeof(float)) = s11_bf16(sums[p]);
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

    __shared__ __align__(32) signed char a_sh[16 * 128];
    __shared__ __align__(32) signed char b_sh[64 * 128];
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
            *reinterpret_cast<int *>(&a_sh[tok_local * 128 + col4 * 4]) = (tok_local < count)
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
            *reinterpret_cast<int *>(&b_sh[r_local * 128 + col4 * 4]) = packed;
        }
        __syncthreads();

        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
        wmma::fill_fragment(acc, 0);
        #pragma unroll
        for (int ks = 0; ks < 128; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a_sh + ks, 128);
            wmma::load_matrix_sync(bf, b_sh + (warp * 16) * 128 + ks, 128);
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
};

static s11_persistent_state g_s11_state;

void ggml_cuda_s11_reset_graph_state() {
    g_s11_state.last_x_data[0] = nullptr;
    g_s11_state.last_x_data[1] = nullptr;
    g_s11_state.last_ids_data = nullptr;
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
        s11_head_bf16<<<dim3((w->ne[1] + 3) / 4, tokens), 128, 0, ctx.stream()>>>(
            static_cast<const nv_bfloat16 *>(w->data), static_cast<const char *>(x->data), static_cast<char *>(dst->data),
            k, w->ne[1], tokens, x->nb[1], dst->nb[1], optimization >= 2);
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
            const bool can_reuse = (x->view_src == nullptr) &&
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
                    const bool reuse_plan = (ids->view_src == nullptr) &&
                                            (g_s11_state.last_ids_data == ids->data) &&
                                            (g_s11_state.last_ids_tokens == tokens) &&
                                            (g_s11_state.last_ids_slots == slots);
                    if (!reuse_plan) {
                        s11_build_routed_plan_kernel<<<1, 256, 0, ctx.stream()>>>(
                            static_cast<const char *>(ids->data), slots, tokens, ids->nb[0], ids->nb[1],
                            tile_experts, tile_starts, tile_counts, sorted_assn, num_tiles);
                        CUDA_CHECK(cudaGetLastError());
                        g_s11_state.last_ids_data = ids->data;
                        g_s11_state.last_ids_tokens = tokens;
                        g_s11_state.last_ids_slots = slots;
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
                if (optimization == 2 && !ids && tokens >= 16) {
                    s11_wmma_k128_kernel<GGML_TYPE_S11_W2_BF16><<<dim3((w->ne[1] + 63) / 64, (tokens + 15) / 16), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], tokens, w->nb[1], dst->nb[1]);
                } else if (optimization == 2 && !ids && tokens >= 8) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 8); }
                else if (optimization == 2 && !ids && tokens >= 4) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 4); }
                else { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 1); }
                break;
            case GGML_TYPE_S11_W4:
                if (optimization == 2 && !ids && tokens >= 16) {
                    s11_wmma_k128_kernel<GGML_TYPE_S11_W4><<<dim3((w->ne[1] + 63) / 64, (tokens + 15) / 16), 128, 0, ctx.stream()>>>(
                        static_cast<const char *>(w->data), q_ptr, scales_ptr, static_cast<char *>(dst->data),
                        k, w->ne[1], tokens, w->nb[1], dst->nb[1]);
                } else if (optimization == 2 && !ids && tokens >= 8) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 8); }
                else if (optimization == 2 && !ids && tokens >= 4) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 4); }
                else { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 1); }
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
        int k, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        float eps, float scale, float bias) {
    const int row = blockIdx.x;
    const int i1 = row % ne1;
    const int i2 = (row / ne1) % ne2;
    const int i3 = row / (ne1 * ne2);
    const float * x_row = reinterpret_cast<const float *>(x + size_t(i1) * nb1 + size_t(i2) * nb2 + size_t(i3) * nb3);
    float * y_row = dst + size_t(row) * k;

    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < k; col += 256) {
        const float v = x_row[col];
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(v, v));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
    __shared__ float warp_sums[8];
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

static __global__ void __launch_bounds__(128) s11_norm_gated_fused_kernel(
        const float * __restrict__ input, const void * __restrict__ weights, int w_type,
        const float * __restrict__ gate, float * __restrict__ dst, float eps) {
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
    dst[size_t(row) * 128 + t] = s11_bf16(__fmul_rn(norm_val, silu_g));
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

static bool s11_has_external_consumer(ggml_cgraph * cgraph, int start_after, const ggml_tensor * target) {
    for (int j = start_after; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * consumer = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (consumer->src[s] == target) {
                return true;
            }
        }
    }
    return false;
}

int ggml_cuda_s11_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
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
                s11_rms_norm_offset_bf16_kernel<<<nrows, 256, 0, ctx.stream()>>>(
                    static_cast<const char *>(x->data),
                    static_cast<const nv_bfloat16 *>(n1->src[0]->data),
                    static_cast<float *>(n5->data),
                    k, int(x->ne[1]), int(x->ne[2]), x->nb[1], x->nb[2], x->nb[3],
                    eps, scale, bias);
                CUDA_CHECK(cudaGetLastError());
                return 5;
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
                s11_norm_gated_fused_kernel<<<nrows, 128, 0, ctx.stream()>>>(
                    static_cast<const float *>(n0->src[0]->data),
                    n3->src[0]->data, n3->src[0]->type,
                    static_cast<const float *>(n5->src[0]->data),
                    static_cast<float *>(n8->data), eps);
                CUDA_CHECK(cudaGetLastError());
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

    // Pattern C: 2-node CPY(F32 -> BF16) + CPY(BF16 -> F32)
    if (n0->op == GGML_OP_CPY && i + 1 < cgraph->n_nodes) {
        ggml_tensor * n1 = cgraph->nodes[i + 1];
        if (n1->op == GGML_OP_CPY && n0->src[0] &&
            n0->src[0]->type == GGML_TYPE_F32 && n0->type == GGML_TYPE_BF16 &&
            n1->src[0] == n0 && n1->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(n0->src[0]) && ggml_is_contiguous(n0) && ggml_is_contiguous(n1) &&
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
                s11_copy_ids_kernel<<<1, 32, 0, ctx.stream()>>>(
                    static_cast<const char *>(ids->data), safe_ids,
                    slots, tokens, ids->nb[0], ids->nb[1]);
                CUDA_CHECK(cudaGetLastError());
                g_s11_state.last_x_data[1] = x->data;
                g_s11_state.last_k[1] = k;
                g_s11_state.last_tokens[1] = tokens;
                g_s11_state.last_input_slots[1] = slots;
                g_s11_state.last_ids_data = nullptr;
                ggml_cuda_s11_invalidate_if_written(c2->data);
                s11_fused_down_exps_reduce_kernel<<<dim3((n + 3) / 4, tokens), 128, 0, ctx.stream()>>>(
                    static_cast<const char *>(w->data), q_ptr, scales_ptr,
                    reinterpret_cast<const char *>(safe_ids), static_cast<char *>(c2->data),
                    k, n, slots, tokens, w->nb[1], w->nb[2], sizeof(int32_t), size_t(slots) * sizeof(int32_t), c2->nb[1]);
                CUDA_CHECK(cudaGetLastError());
                return 17;
            }
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
    if (tokens >= 8) {
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
    const int slots = ids ? glu_node->ne[1] : 1;
    const int input_slots = ids ? x->ne[1] : 1;
    const int buf_idx = (input_slots == 1) ? 0 : 1;
    signed char * q_ptr = g_s11_state.q_buf[buf_idx];
    float * scales_ptr = g_s11_state.s_buf[buf_idx];
    const bool can_reuse = (x->view_src == nullptr) &&
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
        if (prob_weights) {
            g_s11_state.last_ids_data = ids->data;
            g_s11_state.last_ids_tokens = tokens;
            g_s11_state.last_ids_slots = slots;
        }
        s11_fused_up_gate_swiglu_kernel<GGML_TYPE_S11_W2><<<dim3((n + 3) / 4, tokens * slots), 128, 0, ctx.stream()>>>(
            static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
            q_ptr, scales_ptr, ids ? static_cast<const char *>(ids->data) : nullptr,
            prob_weights ? static_cast<const char *>(prob_weights->data) : nullptr,
            static_cast<char *>(out_node->data), k, n, slots, tokens, input_slots,
            w_up->nb[1], w_up->nb[2], ids ? ids->nb[0] : 0, ids ? ids->nb[1] : 0,
            prob_weights ? prob_weights->nb[1] : 0, prob_weights ? prob_weights->nb[2] : 0,
            out_node->nb[1], out_node->nb[2]);
    } else {
        s11_fused_up_gate_swiglu_kernel<GGML_TYPE_S11_W2_BF16><<<dim3((n + 3) / 4, tokens * slots), 128, 0, ctx.stream()>>>(
            static_cast<const char *>(w_up->data), static_cast<const char *>(w_gate->data),
            q_ptr, scales_ptr, nullptr, nullptr,
            static_cast<char *>(out_node->data), k, n, 1, tokens, 1,
            w_up->nb[1], w_up->nb[2], 0, 0, 0, 0,
            0, out_node->nb[1]);
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

static __global__ void s11_dense_kernel(
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
    if (row >= n) {
        return;
    }
    const char * w = a + row * a1 + (head / (heads / a_heads)) * a2 + (sequence / a_sequences) * a3;
    const char * x = b + column * b1 + head * b2 + sequence * b3;
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
    if (row >= n) {
        return;
    }
    const int tile_idx = blockIdx.y;
    const int m_tiles = (m + COLS - 1) / COLS;
    const int first_col = (tile_idx % m_tiles) * COLS;
    const int head = (tile_idx / m_tiles) % heads;
    const int sequence = tile_idx / (m_tiles * heads);
    const char * w = a + size_t(row) * a1 + size_t(head / (heads / a_heads)) * a2 + size_t(sequence / a_sequences) * a3;
    const char * x_base = b + size_t(head) * b2 + size_t(sequence) * b3;
    float sums[COLS] = {};

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
