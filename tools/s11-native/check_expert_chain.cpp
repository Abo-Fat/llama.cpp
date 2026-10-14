#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "../../ggml/src/ggml-cpu/s11-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 1;
    }
    ggml_context * inventory = nullptr;
    gguf_context * file = gguf_init_from_file(argv[1], {true, &inventory});
    if (!file) {
        return 1;
    }
    int fd = open(argv[1], O_RDONLY);
    struct stat info;
    if (fd < 0 || fstat(fd, &info) != 0) {
        return 1;
    }
    void * mapping = mmap(nullptr, info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        return 1;
    }
    const int e = 4, h = 2048, f = 512;
    const char * names[3] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
    const uint8_t * sources[3];
    size_t sizes[3];
    for (int i = 0; i < 3; ++i) {
        ggml_tensor * tensor = ggml_get_tensor(inventory, names[i]);
        const int64_t index = gguf_find_tensor(file, names[i]);
        if (!tensor || index < 0 || tensor->type != GGML_TYPE_S11_W2) {
            return 1;
        }
        sources[i] = static_cast<const uint8_t *>(mapping) + gguf_get_data_offset(file) + gguf_get_tensor_offset(file, index);
        sizes[i] = tensor->nb[2] * e;
    }
    const int cpus[8] = {0,1,2,3,4,5,6,7};
    char error[256];
    s11_cpu_pool * pool = nullptr;
    s11_cpu_executor * executor = nullptr;
    s11_cpu_weights weights{S11_CPU_ABI_VERSION, e, h, f, sources[0], sizes[0], sources[1], sizes[1], sources[2], sizes[2]};
    if (s11_cpu_pool_create(8, cpus, 8, &pool, error, sizeof(error)) != 0 ||
        s11_cpu_executor_create(pool, &weights, 5, 8, 16*1024*1024, &executor, error, sizeof(error)) != 0) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    for (int tokens : {1, 5}) {
        ggml_context * ctx = ggml_init({size_t(8)<<20, nullptr, true});
        ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, GGML_TYPE_S11_W2, h, f, e);
        ggml_tensor * up_w = ggml_new_tensor_3d(ctx, GGML_TYPE_S11_W2, h, f, e);
        ggml_tensor * down_w = ggml_new_tensor_3d(ctx, GGML_TYPE_S11_W2, f, h, e);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h, 1, tokens);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, tokens);
        ggml_tensor * probabilities = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 8, tokens);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_w, x, ids);
        ggml_tensor * up = ggml_mul_mat_id(ctx, up_w, x, ids);
        ggml_tensor * middle = ggml_mul(ctx, ggml_swiglu_split(ctx, gate, up), probabilities);
        middle = ggml_cast(ctx, ggml_cast(ctx, middle, GGML_TYPE_BF16), GGML_TYPE_F32);
        ggml_tensor * y = ggml_mul_mat_id(ctx, down_w, middle, ids);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            return 1;
        }
        ggml_backend_tensor_set(gate_w, sources[0], 0, sizes[0]);
        ggml_backend_tensor_set(up_w, sources[1], 0, sizes[1]);
        ggml_backend_tensor_set(down_w, sources[2], 0, sizes[2]);
        std::mt19937 random(881);
        std::vector<float> inputs(tokens*h), scores(tokens*8);
        std::vector<int32_t> routes(tokens*8);
        for (float & value : inputs) {
            value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(float(int(random()%2001)-1000)/500));
        }
        for (int token = 0; token < tokens; ++token) {
            float total = 0;
            for (int slot = 0; slot < 8; ++slot) {
                routes[token*8+slot] = random()%e;
                total += scores[token*8+slot] = float(1+random()%100);
            }
            for (int slot = 0; slot < 8; ++slot) {
                scores[token*8+slot] /= total;
            }
        }
        ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size()*sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size()*sizeof(int32_t));
        ggml_backend_tensor_set(probabilities, scores.data(), 0, scores.size()*sizeof(float));
        std::vector<uint16_t> cpu(tokens*8*h);
        if (s11_cpu_execute(executor, inputs.data(), routes.data(), scores.data(), nullptr, tokens, 8,
                            cpu.data(), nullptr, error, sizeof(error)) != 0 ||
            ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s\n", error);
            return 1;
        }
        std::vector<float> gpu(cpu.size());
        ggml_backend_tensor_get(y, gpu.data(), 0, gpu.size()*sizeof(float));
        float maximum = 0;
        double error_squared = 0, norm = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            const float expected = ggml_bf16_to_fp32(ggml_bf16_t{cpu[i]});
            if (!std::isfinite(gpu[i])) {
                return 1;
            }
            const float difference = gpu[i]-expected;
            maximum = std::fmax(maximum, std::fabs(difference));
            error_squared += double(difference)*difference;
            norm += double(expected)*expected;
            if (std::fabs(difference) > 0.002f + 0.04f*std::fabs(expected)) {
                std::fprintf(stderr, "Complete CPU/GPU expert chain mismatch\n");
                return 1;
            }
        }
        std::printf("{\"tokens\":%d,\"real_experts\":4,\"relative_l2\":%.9g,\"max_absolute\":%.9g}\n",
                    tokens, std::sqrt(error_squared/std::max(norm,1e-30)), maximum);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    s11_cpu_executor_destroy(executor);
    s11_cpu_pool_destroy(pool);
    munmap(mapping, info.st_size);
    close(fd);
    ggml_free(inventory);
    gguf_free(file);
    return 0;
}
