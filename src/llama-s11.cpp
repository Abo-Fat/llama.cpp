#include "llama-s11.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "llama-moe-cache.h"
#include "llama-context.h"
#include "../ggml/src/ggml-cpu/s11-cpu.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <sched.h>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

class llama_s11_cpu_head {
public:
    std::vector<ggml_bf16_t> values;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * weight = nullptr;
    explicit llama_s11_cpu_head(const ggml_tensor * source) {
        if (source->type != GGML_TYPE_S11_I8 || !source->buffer ||
            !ggml_backend_buffer_is_host(source->buffer) || !ggml_is_contiguous(source)) {
            throw std::invalid_argument("S11 CPU BF16 head requires immutable host I8 weight");
        }
        values.resize(ggml_nelements(source) + 32);
        auto * data = reinterpret_cast<ggml_bf16_t *>((reinterpret_cast<uintptr_t>(values.data()) + 63) & ~uintptr_t(63));
        context.reset(ggml_init({ggml_tensor_overhead() * 2, nullptr, true}));
        if (!context) { throw std::runtime_error("Cannot allocate CPU head metadata"); }
        weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_BF16, source->ne[0], source->ne[1]);
        ggml_set_name(weight, "s11_cpu_head_bf16");
        buffer.reset(ggml_backend_cpu_buffer_from_ptr(data, ggml_nelements(source) * sizeof(ggml_bf16_t)));
        if (!buffer) { throw std::runtime_error("Cannot bind derived CPU head buffer"); }
        ggml_backend_tensor_alloc(buffer.get(), weight, data);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> row(source->ne[0]);
        char error[256];
        for (int64_t r = 0; r < source->ne[1]; ++r) {
            if (s11_cpu_dequantize(S11_CPU_TYPE_I8, static_cast<const char *>(source->data) + r * source->nb[1],
                                   source->ne[0], row.data(), error, sizeof(error))) {
                throw std::runtime_error(error);
            }
            for (int64_t k = 0; k < source->ne[0]; ++k) {
                data[r * source->ne[0] + k] = ggml_fp32_to_bf16(row[k]);
            }
        }
    }
};

ggml_tensor * llama_s11_cpu_head_weight(const llama_model & model) {
    const char * layout = std::getenv("S11_CPU_HEAD");
    if (!layout || std::strcmp(layout, "packed") == 0) { return model.output; }
    if (std::strcmp(layout, "bf16") != 0) { throw std::invalid_argument("Invalid S11_CPU_HEAD"); }
    const char * matmul = std::getenv("S11_CPU_MATMUL");
    if (!matmul || (std::strcmp(matmul, "prepared") != 0 && std::strcmp(matmul, "tiled") != 0)) {
        throw std::invalid_argument("S11 CPU BF16 head requires prepared or tiled CPU matmul");
    }
    if (!model.s11_cpu_head) { model.s11_cpu_head = std::make_shared<llama_s11_cpu_head>(model.output); }
    return model.s11_cpu_head->weight;
}

class llama_s11_cpu {
public:
    struct executor {
        s11_cpu_executor * handle = nullptr;
        std::vector<uint16_t> output;
        std::vector<uint8_t> miss;
        executor() = default;
        executor(const executor &) = delete;
        executor & operator=(const executor &) = delete;
        ~executor() { s11_cpu_executor_destroy(handle); }
    };
    s11_cpu_pool * pool = nullptr;
    std::array<executor, 40> layers;
    std::mutex mutex;
    std::atomic<uint64_t> assignments{0};
    const int capacity = 256;

    explicit llama_s11_cpu(const llama_model & model) {
        cpu_set_t allowed;
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
            throw std::runtime_error("Cannot read S11 CPU affinity");
        }
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &allowed)) { cpus.push_back(cpu); }
        }
        if (cpus.empty() || cpus.size() > 8) {
            throw std::runtime_error("S11 CPU executor requires at most eight allowed CPUs");
        }
        int threads = cpus.size();
        if (const char * value = std::getenv("S11_CPU_THREADS")) {
            char * end;
            const long selected = std::strtol(value, &end, 10);
            if (!*value || *end || selected < 1 || selected > threads) {
                throw std::runtime_error("Invalid S11_CPU_THREADS");
            }
            threads = selected;
        }
        char error[256];
        if (s11_cpu_pool_create(threads, cpus.data(), cpus.size(), &pool, error, sizeof(error)) != S11_CPU_OK) {
            throw std::runtime_error(error);
        }
        try {
            for (int layer = 0; layer < 40; ++layer) {
                const auto & block = model.layers[layer];
                const ggml_tensor * tensors[3] = {block.ffn_gate_exps, block.ffn_up_exps, block.ffn_down_exps};
                for (const ggml_tensor * tensor : tensors) {
                    if (!tensor || tensor->type != GGML_TYPE_S11_W2 || !tensor->buffer ||
                        !ggml_backend_buffer_is_host(tensor->buffer) || !ggml_is_contiguous(tensor)) {
                        throw std::runtime_error("S11 native CPU mode requires --cpu-moe, separate resident gate/up/down banks");
                    }
                }
                s11_cpu_weights weights{
                    S11_CPU_ABI_VERSION, 256, 2048, 512,
                    static_cast<const uint8_t *>(tensors[0]->data), ggml_nbytes(tensors[0]),
                    static_cast<const uint8_t *>(tensors[1]->data), ggml_nbytes(tensors[1]),
                    static_cast<const uint8_t *>(tensors[2]->data), ggml_nbytes(tensors[2]),
                };
                if (s11_cpu_executor_create(pool, &weights, capacity, 8, 16 * 1024 * 1024,
                                           &layers[layer].handle, error, sizeof(error)) != S11_CPU_OK) {
                    throw std::runtime_error(error);
                }
                layers[layer].output.resize(size_t(capacity) * 8 * 2048);
                layers[layer].miss.resize(size_t(capacity) * 8);
            }
        } catch (...) {
            for (auto & layer : layers) {
                s11_cpu_executor_destroy(layer.handle);
                layer.handle = nullptr;
            }
            s11_cpu_pool_destroy(pool);
            pool = nullptr;
            throw;
        }
    }
    ~llama_s11_cpu() {
        for (auto & layer : layers) {
            s11_cpu_executor_destroy(layer.handle);
            layer.handle = nullptr;
        }
        s11_cpu_pool_destroy(pool);
    }
};

struct s11_callback {
    std::shared_ptr<llama_s11_cpu> owner;
    int layer;
    const llama_moe_cache * cache = nullptr;
};

static void execute_s11_misses(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    auto & callback = *static_cast<s11_callback *>(userdata);
    auto & owner = *callback.owner;
    std::lock_guard<std::mutex> lock(owner.mutex);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * probabilities = dst->src[2];
    const int tokens = input->ne[1];
    auto & layer = owner.layers[callback.layer];
    GGML_ASSERT(tokens > 0 && tokens <= owner.capacity && ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(probabilities));
    const ggml_tensor * map = callback.cache ? callback.cache->get_slot_map(callback.layer, tokens, 8) : nullptr;
    const int32_t * lookup = map ? static_cast<const int32_t *>(map->data) : nullptr;
    const int32_t * selected = static_cast<const int32_t *>(ids->data);
    uint64_t misses = 0;
    for (int row = 0; row < tokens * 8; ++row) {
        GGML_ASSERT(selected[row] >= 0 && selected[row] < 256);
        layer.miss[row] = !lookup || lookup[selected[row]] < 0;
        misses += layer.miss[row];
    }
    char error[256];
    if (s11_cpu_execute(layer.handle, static_cast<const float *>(input->data), selected,
                        static_cast<const float *>(probabilities->data), layer.miss.data(),
                        tokens, 8, layer.output.data(), nullptr, error, sizeof(error)) != S11_CPU_OK) {
        GGML_ABORT("S11 CPU cache miss: %s", error);
    }
    owner.assignments.fetch_add(misses, std::memory_order_relaxed);
    float * output = static_cast<float *>(dst->data);
    for (size_t i = 0; i < size_t(tokens) * 8 * 2048; ++i) {
        output[i] = ggml_bf16_to_fp32(ggml_bf16_t{layer.output[i]});
    }
}

static void execute_s11_cpu(ggml_tensor * dst, const ggml_tensor * input, const ggml_tensor * ids,
                           const ggml_tensor * probabilities, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    auto & callback = *static_cast<s11_callback *>(userdata);
    auto & owner = *callback.owner;
    std::lock_guard<std::mutex> lock(owner.mutex);
    const int tokens = input->ne[1];
    GGML_ASSERT(tokens > 0 && tokens <= owner.capacity);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && probabilities->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(ids) && ggml_is_contiguous(probabilities));
    auto & layer = owner.layers[callback.layer];
    char error[256];
    const int status = s11_cpu_execute(
        layer.handle, static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
        static_cast<const float *>(probabilities->data), nullptr, tokens, 8, layer.output.data(), nullptr, error, sizeof(error));
    if (status != S11_CPU_OK) {
        GGML_ABORT("S11 expert executor: %s", error);
    }
    owner.assignments.fetch_add(uint64_t(tokens) * 8, std::memory_order_relaxed);
    float * output = static_cast<float *>(dst->data);
    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < 2048; ++column) {
            float sum = 0;
            for (int slot = 0; slot < 8; ++slot) {
                ggml_bf16_t value{layer.output[(token * 8 + slot) * 2048 + column]};
                sum += ggml_bf16_to_fp32(value);
            }
            output[token * 2048 + column] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(sum));
        }
    }
}

ggml_tensor * llama_s11_cpu_experts(ggml_context * ctx, const llama_model & model, int layer,
                                  ggml_tensor * input, ggml_tensor * ids, ggml_tensor * probabilities) {
    if (!model.s11_cpu) {
        model.s11_cpu = std::make_shared<llama_s11_cpu>(model);
    }

    // Callback records live with the model, not with a reusable graph arena.
    if (layer < 0 || layer >= 40) {
        throw std::invalid_argument("S11 expert layer outside model");
    }
    if (!model.s11_callbacks[layer]) {
        model.s11_callbacks[layer] = std::make_shared<s11_callback>(s11_callback{model.s11_cpu, layer});
    }
    return ggml_map_custom3(ctx, ggml_cont(ctx, input), ggml_cont(ctx, ids), ggml_cont(ctx, probabilities),
                           execute_s11_cpu, 1, model.s11_callbacks[layer].get());
}

uint64_t llama_s11_cpu_assignments(const llama_model & model) {
    return model.s11_cpu ? model.s11_cpu->assignments.load(std::memory_order_relaxed) : 0;
}

uint64_t llama_model_s11_cpu_assignments(const llama_model * model) {
    return model ? llama_s11_cpu_assignments(*model) : 0;
}

llama_s11_cache_stats llama_s11_cache_statistics(const llama_context * ctx) {
    const auto * cache = ctx ? ctx->get_moe_cache() : nullptr;
    if (!cache) {
        return {0, 0, 0};
    }
    const auto values = cache->s11_statistics();
    return {values[0], values[1], values[2]};
}

llama_s11_prefetch_stats llama_s11_prefetch_statistics(const llama_context * ctx) {
    const auto * cache = ctx ? ctx->get_moe_cache() : nullptr;
    if (!cache) {
        return {0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    const auto values = cache->prefetch_statistics();
    return {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]};
}

ggml_tensor * llama_s11_cpu_misses(ggml_context * ctx, const llama_model & model,
                                 const llama_moe_cache * cache, int layer,
                                 ggml_tensor * input, ggml_tensor * ids, ggml_tensor * probabilities) {
    if (!model.s11_cpu) {
        model.s11_cpu = std::make_shared<llama_s11_cpu>(model);
    }
    if (layer < 0 || layer >= 40) {
        throw std::invalid_argument("S11 miss layer outside model");
    }
    if (!model.s11_callbacks[layer]) {
        model.s11_callbacks[layer] = std::make_shared<s11_callback>(s11_callback{model.s11_cpu, layer, cache});
    }
    model.s11_callbacks[layer]->cache = cache;
    ggml_tensor * sources[3] = {ggml_cont(ctx, input), ggml_cont(ctx, ids), ggml_cont(ctx, probabilities)};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 2048, 8, input->ne[1], 1, sources, 3,
                          execute_s11_misses, 1, model.s11_callbacks[layer].get());
}

void llama_s11_release_cpu(llama_model & model) {
    for (auto & callback : model.s11_callbacks) { callback.reset(); }
    model.s11_cpu_head.reset();
    model.s11_cpu.reset();
}
