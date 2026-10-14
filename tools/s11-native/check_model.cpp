#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <string>

struct layer_trace {
    std::array<std::vector<float>, 40> last;
    static bool observe(ggml_tensor * tensor, bool ask, void * data) {
        const std::string name(tensor->name);
        if (name.rfind("l_out-", 0) != 0) {
            return false;
        }
        if (ask) {
            return true;
        }
        const int layer = std::stoi(name.substr(6));
        if (layer < 0 || layer >= 40 || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 2048) {
            return false;
        }
        auto & values = static_cast<layer_trace *>(data)->last[layer];
        values.resize(2048);
        ggml_backend_tensor_get(tensor, values.data(), (tensor->ne[1]-1)*tensor->nb[1], 2048*sizeof(float));
        return true;
    }
};

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "Usage: check-model MODEL.gguf [gpu] [plain]\n");
        return 1;
    }
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    const bool gpu = argc >= 3 && std::strcmp(argv[2], "gpu") == 0;
    const char * resident_value = std::getenv("S11_RESIDENT");
    const bool resident = resident_value && std::strcmp(resident_value, "1") == 0;
    if (resident_value && (!resident || !gpu)) {
        std::fprintf(stderr, "S11_RESIDENT=1 requires GPU qualification\n");
        return 1;
    }
    const bool diagnostic = argc < 4;
    mp.n_gpu_layers = gpu ? 99 : 0;
    std::vector<llama_model_tensor_buft_override> overrides = {
        {"\\.ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type()},
    };
    const char * embedding = std::getenv("S11_EMBEDDING");
    if (embedding && std::strcmp(embedding, "cpu") == 0) {
        overrides.push_back({"^token_embd\\.weight$", ggml_backend_cpu_buffer_type()});
    }
    overrides.push_back({nullptr, nullptr});
    mp.tensor_buft_overrides = resident ? nullptr : overrides.data();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    const char * expert_mode = std::getenv("S11_EXPERT_MODE");
    cp.moe_cache_size = gpu && !resident && (!expert_mode || std::strcmp(expert_mode, "cpu") != 0) ? size_t(3072) * 1024 * 1024 : 0;
    cp.type_k = GGML_TYPE_BF16;
    cp.type_v = GGML_TYPE_BF16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    layer_trace trace;
    if (diagnostic) {
        cp.cb_eval = layer_trace::observe;
        cp.cb_eval_user_data = &trace;
    }
    std::array<std::vector<float>, 40> reference_layers;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> expected;
    std::vector<llama_token> ids(33, 1000);
    for (int request = 0; request < 2; ++request) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(33, 0, 1);
        batch.n_tokens = 33;
        for (int i = 0; i < 33; ++i) {
            batch.token[i] = ids[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == 32;
        }
        const uint64_t before = llama_model_s11_cpu_assignments(model);
        const uint64_t gpu_before = gpu ? ggml_cuda_s11_assignments() : 0;
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "Native decode failed\n");
            return 1;
        }
        const float * output = llama_get_logits_ith(ctx, -1);
        if (!output || std::any_of(output, output + vocab, [](float v) { return !std::isfinite(v); })) {
            std::fprintf(stderr, "Native model logits missing or nonfinite\n");
            return 1;
        }
        if (request == 0) {
            expected.assign(output, output + vocab);
            reference_layers = trace.last;
        } else if (!std::equal(expected.begin(), expected.end(), output)) {
            std::fprintf(stderr, "Native raw logits changed across reset\n");
            return 1;
        }
        const uint64_t work = llama_model_s11_cpu_assignments(model) - before;
        const uint64_t gpu_work = gpu ? ggml_cuda_s11_assignments() - gpu_before : 0;
        if (work + gpu_work != uint64_t(33) * 40 * 8) {
            std::fprintf(stderr, "Native CPU expert workload mismatch: %llu\n", static_cast<unsigned long long>(work));
            return 1;
        }
        std::printf("{\"request\":%d,\"tokens\":33,\"finite_logits\":%d,\"cpu_assignments\":%llu,\"gpu_assignments\":%llu,\"reset_logits_equal\":true}\n",
                    request, vocab, static_cast<unsigned long long>(work), static_cast<unsigned long long>(gpu_work));
        llama_batch_free(batch);
    }
    for (int chunk : {17, 1}) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const uint64_t cpu_before = llama_model_s11_cpu_assignments(model);
        const uint64_t gpu_before = gpu ? ggml_cuda_s11_assignments() : 0;
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        for (int first = 0; first < 33; first += chunk) {
            batch.n_tokens = std::min(chunk, 33-first);
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = ids[first+i];
                batch.pos[i] = first+i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = first+i == 32;
            }
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "Native chunked state update failed\n");
                return 1;
            }
        }
        const float * actual = llama_get_logits_ith(ctx, -1);
        double error_squared = 0, norm_squared = 0;
        float maximum = 0;
        for (int i = 0; i < vocab; ++i) {
            if (!std::isfinite(actual[i])) {
                return 1;
            }
            const float error = actual[i] - expected[i];
            maximum = std::max(maximum, std::fabs(error));
            error_squared += double(error)*error;
            norm_squared += double(expected[i])*expected[i];
        }
        const double relative = std::sqrt(error_squared/std::max(norm_squared, 1e-30));
        for (int layer = 0; layer < 40; ++layer) {
            if (!diagnostic) {
                break;
            }
            if (reference_layers[layer].size() != 2048 || trace.last[layer].size() != 2048) {
                std::fprintf(stderr, "Missing native diagnostic layer boundary\n");
                return 1;
            }
            double e = 0, norm = 0;
            float max_error = 0;
            for (int i = 0; i < 2048; ++i) {
                const float difference = trace.last[layer][i]-reference_layers[layer][i];
                e += double(difference)*difference;
                norm += double(reference_layers[layer][i])*reference_layers[layer][i];
                max_error = std::max(max_error, std::fabs(difference));
            }
            if (e > 0) {
                std::printf("{\"chunk\":%d,\"layer\":%d,\"hidden_relative_l2\":%.9g,\"hidden_max\":%.9g}\n",
                            chunk, layer, std::sqrt(e/std::max(norm,1e-30)), max_error);
            }
        }
        const uint64_t work = llama_model_s11_cpu_assignments(model)-cpu_before +
            (gpu ? ggml_cuda_s11_assignments()-gpu_before : 0);
        if (relative > 0.02 || llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != 32 || work != uint64_t(33)*40*8) {
            std::fprintf(stderr, "Native chunk/incremental state qualification failed: relative=%g max=%g work=%llu\n",
                         relative, maximum, static_cast<unsigned long long>(work));
            return 1;
        }
        std::printf("{\"chunk\":%d,\"finite_logits\":%d,\"relative_l2\":%.9g,\"max_absolute\":%.9g,\"position\":32,\"assignments\":%llu}\n",
                    chunk, vocab, relative, maximum, static_cast<unsigned long long>(work));
        llama_batch_free(batch);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
