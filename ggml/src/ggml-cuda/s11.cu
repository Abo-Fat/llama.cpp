#include "s11.cuh"
#include <cuda_bf16.h>
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
    for (int group = 0; group < k / 128; ++group) {
        const auto * block = reinterpret_cast<const unsigned char *>(weight + group * bytes);
        const float scale = s11_scale(block, TYPE);
        if constexpr (TYPE == GGML_TYPE_S11_W2) {
            unsigned packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packed |= unsigned(static_cast<unsigned char>(s11_code(block, lane * 4 + j, TYPE))) << (8 * j);
            }
            const int source = (first / slots) * input_slots + (input_slots == 1 ? 0 : first % slots);
            const int activation = *reinterpret_cast<const int *>(q + source * k + group * 128 + lane * 4);
            int dot = __dp4a(int(packed), activation, 0);
            for (int offset = 16; offset; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            sums[0] = __fadd_rn(sums[0], __fmul_rn(__fmul_rn(float(dot), activation_scales[source * (k / 128) + group]), scale));
        } else {
            float dots[COLS] = {};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int index = lane + 32 * i;
                const float restored = TYPE == GGML_TYPE_S11_I8
                    ? __fdiv_rn(float(s11_code(block, index, TYPE)), scale)
                    : __fmul_rn(float(s11_code(block, index, TYPE)), scale);
                const float w = s11_bf16(restored);
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int assignment = first + c, token = assignment / slots, slot = assignment % slots;
                    if (token < tokens) {
                        const int source = token * input_slots + (input_slots == 1 ? 0 : slot);
                        const float a = TYPE == GGML_TYPE_S11_I8
                            ? s11_bf16(reinterpret_cast<const float *>(input + token * input_token + (input_slots == 1 ? 0 : slot) * input_slot)[group * 128 + index])
                            : fake[source * k + group * 128 + index];
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
                                    int k, int n, int tokens, size_t input_stride, size_t output_stride) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32, token = blockIdx.y;
    if (row >= n || token >= tokens) { return; }
    const float * x = reinterpret_cast<const float *>(input + token * input_stride);
    float sum = 0;
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
    if (ggml_cuda_s11_bf16_head(w)) {
        GGML_ASSERT(!ids && !w->view_src);
        s11_head_bf16<<<dim3((w->ne[1] + 3) / 4, tokens), 128, 0, ctx.stream()>>>(
            static_cast<const nv_bfloat16 *>(w->data), static_cast<const char *>(x->data), static_cast<char *>(dst->data),
            k, w->ne[1], tokens, x->nb[1], dst->nb[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    ggml_cuda_pool_alloc<signed char> q(ctx.pool(), w->type == GGML_TYPE_S11_I8 ? 1 : size_t(tokens) * input_slots * k);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(), w->type == GGML_TYPE_S11_I8 ? 1 : size_t(tokens) * input_slots * (k / 128));
    const int optimization = s11_optimization();
    const bool need_fake = w->type != GGML_TYPE_S11_I8 && (w->type != GGML_TYPE_S11_W2 || !optimization);
    ggml_cuda_pool_alloc<float> fake(ctx.pool(), need_fake ? size_t(tokens) * input_slots * k : 1);
    if (w->type != GGML_TYPE_S11_I8) {
        s11_quantize<<<dim3(k / 128, tokens * input_slots), 32, 0, ctx.stream()>>>(
            static_cast<const char *>(x->data), q.get(), scales.get(), need_fake ? fake.get() : nullptr, k,
            input_slots, tokens, ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1]);
        CUDA_CHECK(cudaGetLastError());
    }
    if (optimization) {
        #define S11_PACKED_LAUNCH(TYPE, COLS) \
            s11_projection_packed<TYPE, COLS><<<dim3((w->ne[1] + 3) / 4, (tokens * slots + COLS - 1) / COLS), 128, 0, ctx.stream()>>>( \
                static_cast<const char *>(w->data), q.get(), scales.get(), fake.get(), static_cast<const char *>(x->data), \
                ids ? static_cast<const char *>(ids->data) : nullptr, static_cast<char *>(dst->data), \
                k, w->ne[1], slots, tokens, input_slots, w->nb[1], w->nb[2], \
                ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1], ids ? ids->nb[0] : 0, ids ? ids->nb[1] : 0, \
                ids ? dst->nb[1] : 0, ids ? dst->nb[2] : dst->nb[1])
        switch (w->type) {
            case GGML_TYPE_S11_W2: S11_PACKED_LAUNCH(GGML_TYPE_S11_W2, 1); break;
            case GGML_TYPE_S11_W2_BF16:
                if (optimization == 2 && !ids && tokens >= 4) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 4); }
                else { S11_PACKED_LAUNCH(GGML_TYPE_S11_W2_BF16, 1); }
                break;
            case GGML_TYPE_S11_W4:
                if (optimization == 2 && !ids && tokens >= 4) { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 4); }
                else { S11_PACKED_LAUNCH(GGML_TYPE_S11_W4, 1); }
                break;
            case GGML_TYPE_S11_I8: S11_PACKED_LAUNCH(GGML_TYPE_S11_I8, 1); break;
            default: GGML_ABORT("Unsupported S11 packed type");
        }
        #undef S11_PACKED_LAUNCH
    } else {
        s11_projection<<<dim3((w->ne[1] + 3) / 4, tokens * slots), 128, 0, ctx.stream()>>>(
        static_cast<const char *>(w->data), static_cast<const char *>(x->data), q.get(), scales.get(), fake.get(),
        ids ? static_cast<const char *>(ids->data) : nullptr, static_cast<char *>(dst->data),
        w->type, k, w->ne[1], slots, tokens, input_slots, w->nb[1], w->nb[2],
        ids ? x->nb[1] : 0, ids ? x->nb[2] : x->nb[1],
        ids ? ids->nb[0] : 0, ids ? ids->nb[1] : 0,
        ids ? dst->nb[1] : 0, ids ? dst->nb[2] : dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
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

void ggml_cuda_s11_dense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_BF16 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0);
    const int columns = dst->ne[1] * dst->ne[2] * dst->ne[3];
    GGML_ASSERT(columns <= 65535);
    s11_dense_kernel<<<dim3((dst->ne[0] + 3) / 4, columns), 128, 0, ctx.stream()>>>(
        static_cast<const char *>(a->data), static_cast<const char *>(b->data), static_cast<char *>(dst->data),
        a->type, b->type, a->ne[0], dst->ne[0], dst->ne[1], dst->ne[2], a->ne[2], b->ne[3] / a->ne[3],
        a->nb[0], a->nb[1], a->nb[2], a->nb[3], b->nb[0], b->nb[1], b->nb[2], b->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
