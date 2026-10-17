#pragma once
#include "common.cuh"

inline bool ggml_cuda_is_s11(ggml_type type) {
    return type >= GGML_TYPE_S11_W2 && type <= GGML_TYPE_S11_I8;
}

void ggml_cuda_s11_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_s11_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_s11_dense(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_s11_bf16_head(const ggml_tensor * tensor);
void ggml_cuda_s11_upload_head(ggml_tensor * tensor, const void * canonical, cudaStream_t stream);
void ggml_cuda_s11_reset_graph_state();
void ggml_cuda_s11_begin_split(const ggml_cgraph * cgraph);
void ggml_cuda_s11_record_graph_exit_state(const void * graph_key);
void ggml_cuda_s11_restore_graph_exit_state(const void * graph_key);
void ggml_cuda_s11_invalidate_if_written(const void * data_ptr);
struct ggml_cuda_s11_fusion_scope {
    const ggml_cgraph * graph;
    bool indexed = false;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> consumers;
};
int ggml_cuda_s11_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
                          ggml_cuda_s11_fusion_scope * scope = nullptr);
