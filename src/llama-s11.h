#pragma once

#include "ggml.h"
#include <memory>

struct llama_model;
class llama_moe_cache;
class llama_s11_cpu;

ggml_tensor * llama_s11_cpu_experts(ggml_context * ctx, const llama_model & model,
                                  int layer, ggml_tensor * input, ggml_tensor * ids,
                                  ggml_tensor * probabilities);
uint64_t llama_s11_cpu_assignments(const llama_model & model);
ggml_tensor * llama_s11_cpu_misses(ggml_context * ctx, const llama_model & model,
                                 const llama_moe_cache * cache, int layer,
                                 ggml_tensor * input, ggml_tensor * ids, ggml_tensor * probabilities);
