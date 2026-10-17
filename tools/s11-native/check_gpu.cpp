#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "../../ggml/src/ggml-cpu/s11-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int check_roundtrip_alias() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "CUDA backend initialization failed\n");
        return 1;
    }
    bool passed = true;
    for (int shift : {0, 128, -128, 16384}) {
        constexpr int count = 8192;
        ggml_context * ctx = ggml_init({size_t(2) << 20, nullptr, true});
        ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count + 16384 + 128);
        const size_t src_offset = shift < 0 ? 128 * sizeof(float) : 0;
        const size_t dst_offset = size_t(shift < 0 ? 0 : shift) * sizeof(float);
        ggml_tensor * input = ggml_view_1d(ctx, storage, count, src_offset);
        ggml_tensor * bf16 = ggml_cast(ctx, input, GGML_TYPE_BF16);
        ggml_tensor * output = ggml_cast(ctx, bf16, GGML_TYPE_F32);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            std::fprintf(stderr, "Roundtrip alias buffer allocation failed\n");
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
        // Reproduce a legal allocator layout without an extra view node between the copies.
        output->data = static_cast<char *>(storage->data) + dst_offset;
        std::vector<float> values(count), actual(count);
        for (int i = 0; i < count; ++i) {
            values[i] = float(i * 17 % 4093 - 2046) / 37.0f;
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                passed = false;
                break;
            }
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            for (int i = 0; i < count; ++i) {
                if (actual[i] != ggml_bf16_to_fp32(ggml_fp32_to_bf16(values[i]))) {
                    std::fprintf(stderr, "BF16 roundtrip alias mismatch: shift=%d repeat=%d index=%d\n",
                                 shift, repeat, i);
                    passed = false;
                    break;
                }
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::printf("BF16 roundtrip alias: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--roundtrip-alias") == 0) {
        return check_roundtrip_alias();
    }
    const bool cpu = argc == 2 && std::strcmp(argv[1], "--cpu") == 0;
    ggml_backend_t backend = cpu ? ggml_backend_cpu_init() : ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "Requested backend initialization failed\n");
        return 1;
    }
    bool passed = true;
    for (ggml_type type : {GGML_TYPE_S11_W2, GGML_TYPE_S11_W2_BF16, GGML_TYPE_S11_W4, GGML_TYPE_S11_I8}) {
        for (int tokens : {1, 3, 17, 128}) {
            const int k = 2048, n = 64;
            ggml_context * context = ggml_init({size_t(8) << 20, nullptr, true});
            ggml_tensor * weights = ggml_new_tensor_2d(context, type, k, n);
            if (type == GGML_TYPE_S11_I8) {
                ggml_set_name(weights, "output.weight");
            }
            ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, tokens);
            ggml_tensor * output = ggml_mul_mat(context, weights, input);
            ggml_cgraph * graph = ggml_new_graph(context);
            ggml_build_forward_expand(graph, output);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
            if (!buffer) {
                std::fprintf(stderr, "GPU test buffer allocation failed\n");
                return 1;
            }
            const size_t block = ggml_type_size(type);
            const size_t codes = block - sizeof(float);
            std::vector<unsigned char> raw(ggml_nbytes(weights));
            std::mt19937 random(3011 + type);
            for (int row = 0; row < n; ++row) {
                for (int group = 0; group < k / 128; ++group) {
                    unsigned char * p = raw.data() + row * weights->nb[1] + group * block;
                    for (int i = 0; i < 128; ++i) {
                        const int value = int(random() % 3) - 1;
                        if (type <= GGML_TYPE_S11_W2_BF16) {
                            p[i / 4] |= (value + 1) << (2 * (i % 4));
                        } else if (type == GGML_TYPE_S11_W4) {
                            p[i / 2] |= ((int(random() % 15) - 7) & 15) << (4 * (i % 2));
                        } else {
                            p[i] = static_cast<unsigned char>(int(random() % 255) - 127);
                        }
                    }
                    const float scale = type == GGML_TYPE_S11_I8 ? 32.0f + random() % 128 : 0.001f * (1 + random() % 80);
                    std::memcpy(p + codes, &scale, sizeof(scale));
                }
            }
            std::vector<float> x(k * tokens);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = ggml_bf16_to_fp32(ggml_fp32_to_bf16((float(int(random() % 2001) - 1000) / 1000.0f)));
            }
            ggml_backend_tensor_set(weights, raw.data(), 0, raw.size());
            if (type == GGML_TYPE_S11_I8) {
                std::vector<unsigned char> restored(raw.size());
                ggml_backend_tensor_get(weights, restored.data(), 0, restored.size());
                if (restored != raw) {
                    std::fprintf(stderr, "I8 head canonical values changed\n");
                    return 1;
                }
            }
            ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "GPU graph execution failed\n");
                return 1;
            }
            std::vector<float> actual(n * tokens);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            float maximum = 0;
            for (int token = 0; token < tokens; ++token) {
                for (int row = 0; row < n; ++row) {
                    float expected;
                    char error[256];
                    if (s11_cpu_vec_dot(type, raw.data() + row * weights->nb[1], x.data() + token * k, k, &expected, error, sizeof(error)) != S11_CPU_OK) {
                        std::fprintf(stderr, "CPU reference failed: %s\n", error);
                        return 1;
                    }
                    const float value = actual[token * n + row];
                    maximum = std::fmax(maximum, std::fabs(value - expected));
                    if (!std::isfinite(value) || std::fabs(value - expected) > 0.002f + 0.02f * std::fabs(expected)) {
                        passed = false;
                    }
                    if (type == GGML_TYPE_S11_W2 && value != expected) {
                        passed = false;
                    }
                }
            }
            std::printf("{\"type\":%d,\"tokens\":%d,\"maximum_absolute_error\":%.9g}\n", type, tokens, maximum);
            ggml_backend_buffer_free(buffer);
            ggml_free(context);
        }
    }
    for (int tokens : {1, 5}) {
        const int k = 2048, n = 64, experts = 3, slots = 8;
        ggml_context * context = ggml_init({size_t(8) << 20, nullptr, true});
        ggml_tensor * weights = ggml_new_tensor_3d(context, GGML_TYPE_S11_W2, k, n, experts);
        ggml_tensor * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, k, 1, tokens);
        ggml_tensor * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, slots, tokens);
        ggml_tensor * output = ggml_mul_mat_id(context, weights, input, ids);
        ggml_cgraph * graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) {
            std::fprintf(stderr, "Routed test allocation failed\n");
            return 1;
        }
        std::vector<unsigned char> raw(ggml_nbytes(weights));
        std::mt19937 random(1763);
        for (size_t first = 0; first < raw.size(); first += 36) {
            for (int column = 0; column < 128; ++column) {
                raw[first + column / 4] |= (random() % 3) << (2 * (column % 4));
            }
            const float scale = 0.0005f * (1 + random() % 50);
            std::memcpy(raw.data() + first + 32, &scale, sizeof(scale));
        }
        std::vector<float> x(tokens * k);
        for (float & value : x) {
            value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(float(int(random() % 201)-100) / 100));
        }
        std::vector<int> selected(tokens * slots);
        for (int & value : selected) {
            value = random() % experts;
        }
        ggml_backend_tensor_set(weights, raw.data(), 0, raw.size());
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return 1;
        }
        std::vector<float> actual(n * tokens * slots);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                for (int row = 0; row < n; ++row) {
                    float expected;
                    char error[256];
                    const unsigned char * weight = raw.data() + selected[token * slots + slot] * weights->nb[2] + row * weights->nb[1];
                    if (s11_cpu_vec_dot(S11_CPU_TYPE_W2, weight, x.data()+token*k, k, &expected, error, sizeof(error)) != S11_CPU_OK ||
                        actual[(token*slots+slot)*n+row] != expected) {
                        std::fprintf(stderr, "Native routed CPU/GPU group arithmetic mismatch\n");
                        return 1;
                    }
                }
            }
        }
        std::printf("{\"routed_tokens\":%d,\"slots\":8,\"experts\":3,\"bitwise_equal\":true}\n", tokens);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
