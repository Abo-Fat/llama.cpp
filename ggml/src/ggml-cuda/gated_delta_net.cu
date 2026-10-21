#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_cols_per_warp = 4;

// base+tip recurrent rollback (llama S11_RS_REPLAY): op_params[GDN_RS_MODE_OP_PARAM] holds the mode bits, src[6] the
// control {n_rep, parity, base_row, tip_row, sel_row}, src[7] the input ring and src[8] the SSM cache
#define GDN_RS_MODE_OP_PARAM  15
#define GDN_RS_MODE_REPLAY    1 // load the state row sel_row of the cache, replay n_rep ring rows before the ubatch
#define GDN_RS_MODE_KEEP_RING 2 // store the rows to the other ring half, the replayed base and the tip to the cache

struct gdn_rs_args {
    const int32_t * ctl;
    float *         ring;      // [row_elems, ring_rows, 2]: k (hk heads) | v (H heads) | raw g (H) | beta (H)
    float *         states;    // SSM cache rows of H * S_v * S_v
    int64_t         row_elems;
    int64_t         ring_rows;
    int64_t         hk;
    int             mode;
};

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const gdn_rs_args rs) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    __shared__ float q_sh[S_v];
    __shared__ float k_sh[S_v];
    __shared__ float g_sh[KDA ? S_v : 1];
    __shared__ float beta_sh;
    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int cta_threads = blockDim.x * blockDim.y;

    ggml_cuda_pdl_sync();

    // base+tip recurrent rollback: the state row comes straight from the cache, and the n_rep accepted rows of the
    // previous keep ubatch are replayed from the input ring first through a separate prologue loop
    int           n_rep   = 0;
    const float * ring_rd = nullptr;
    float *       ring_wr = nullptr;
    float *       base_st = nullptr;
    float *       tip_st  = nullptr;
    if (rs.mode != 0) {
        const int64_t D      = H * S_v * S_v;
        const int64_t half   = rs.ring_rows * rs.row_elems;
        const int64_t parity = rs.ctl[1] & 1;
        const int64_t head_o = (int64_t) h_idx * S_v * S_v;
        n_rep      = rs.ctl[0];
        ring_rd    = rs.ring + parity * half;
        curr_state = rs.states + rs.ctl[4] * D + head_o + col * S_v;
        if (rs.mode & GDN_RS_MODE_KEEP_RING) {
            if (n_tokens > 2) {
                ring_wr = rs.ring + (parity ^ 1) * half;
            }
            tip_st  = rs.states + rs.ctl[3] * D + head_o;
            if (n_rep > 0 || n_tokens == 2) {
                base_st = rs.states + rs.ctl[2] * D + head_o;
            }
        }
    }
    const int64_t rs_off_v = rs.hk * S_v;
    const int64_t rs_off_g = rs_off_v + H * S_v;
    const int64_t rs_off_b = rs_off_g + H;
    // one writer per ring element: k by the first column block of each k head, g and beta by its thread 0
    const bool    rs_k_wr  = ring_wr != nullptr && blockIdx.z == 0 && h_idx == iq1;
    const bool    rs_gb_wr = ring_wr != nullptr && blockIdx.z == 0 && tid == 0;

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    if (__builtin_expect(n_rep > 0, 0)) {
        for (int tr = 0; tr < n_rep; tr++) {
            const float * row    = ring_rd + (int64_t) tr * rs.row_elems;
            const float * k_t    = row + (int64_t) iq1 * S_v;
            const float * v_t    = row + rs_off_v + (int64_t) h_idx * S_v;
            const float * g_t    = row + rs_off_g + h_idx;
            const float * beta_t = row + rs_off_b + h_idx;
            for (int idx = tid; idx < S_v; idx += cta_threads) {
                k_sh[idx] = k_t[idx];
            }
            if (tid == 0) {
                beta_sh = *beta_t;
                if constexpr (!KDA) {
                    g_sh[0] = expf(*g_t);
                }
            }
            __syncthreads();
            const float beta_val = beta_sh;
            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                k_reg[r] = k_sh[i];
            }
            if constexpr (!KDA) {
                const float g_val = g_sh[0];
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);
                float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                }
            }
            __syncthreads();
        }
        if (base_st != nullptr && n_tokens > 2) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                base_st[col * S_v + i] = s_shard[r];
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        for (int idx = tid; idx < S_v; idx += cta_threads) {
            q_sh[idx] = q_t[idx];
            k_sh[idx] = k_t[idx];
            if constexpr (KDA) {
                g_sh[idx] = expf(g_t[idx]);
            }
        }
        if (tid == 0) {
            beta_sh = *beta_t;
            if constexpr (!KDA) {
                g_sh[0] = expf(*g_t);
            }
        }
        if (__builtin_expect(ring_wr != nullptr, 0)) {
            float * row_wr = ring_wr + (int64_t) t * rs.row_elems;
            if (rs_k_wr) {
                for (int idx = tid; idx < S_v; idx += cta_threads) {
                    row_wr[(int64_t) iq1 * S_v + idx] = k_t[idx];
                }
            }
            if (rs_gb_wr && !KDA) {
                row_wr[rs_off_g + h_idx] = *g_t;
                row_wr[rs_off_b + h_idx] = *beta_t;
            }
            if (lane_in_col == 0) {
                row_wr[rs_off_v + (int64_t) h_idx * S_v + col] = v_t[col];
            }
        }
        __syncthreads();

        const float beta_val = beta_sh;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_sh[i];
            q_reg[r] = q_sh[i];
        }

        if constexpr (!KDA) {
            const float g_val = g_sh[0];

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += g_sh[i] * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = g_sh[i] * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            if (base_st != nullptr && n_tokens == 2 && t == 0) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    base_st[col * S_v + i] = s_shard[r];
                }
            }
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                // in the replay keep mode the only slot (K == 1) is the tip row of the cache
                float * curr_state = (tip_st != nullptr ? tip_st : state) + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }

        if (n_tokens > 1) {
            __syncthreads();
        }
    }

    if constexpr (!keep_rs_t) {
        float * state_dst = tip_st != nullptr ? tip_st : state;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i              = r * lanes_per_col + lane_in_col;
            state_dst[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const gdn_rs_args & rs, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // base+tip recurrent rollback (see gdn_rs_args): control, input ring and SSM cache ride as extra sources
    gdn_rs_args rs = {};
    rs.mode = ggml_get_op_params_i32(dst, GDN_RS_MODE_OP_PARAM);
    if (rs.mode != 0) {
        const ggml_tensor * src_ctl   = dst->src[6];
        const ggml_tensor * src_ring  = dst->src[7];
        const ggml_tensor * src_cache = dst->src[8];

        GGML_ASSERT(!kda && n_seqs == 1);
        GGML_ASSERT(src_ctl   != nullptr && src_ctl->type   == GGML_TYPE_I32 && ggml_nelements(src_ctl) >= 5);
        GGML_ASSERT(src_ring  != nullptr && src_ring->type  == GGML_TYPE_F32 && ggml_is_contiguous(src_ring) && src_ring->ne[2] == 2);
        GGML_ASSERT(src_cache != nullptr && src_cache->type == GGML_TYPE_F32 && src_cache->ne[0] == S_v * S_v * H);

        rs.ctl       = (const int32_t *) src_ctl->data;
        rs.ring      = (float *) src_ring->data;
        rs.states    = (float *) src_cache->data;
        rs.row_elems = src_ring->ne[0];
        rs.ring_rows = src_ring->ne[1];
        rs.hk        = neqk1;

        GGML_ASSERT(rs.row_elems == S_v * neqk1 + S_v * H + 2 * H);
        GGML_ASSERT(!(rs.mode & GDN_RS_MODE_KEEP_RING) || (K == 1 && n_tokens <= rs.ring_rows));
    }

    // the keep instantiation also runs the replay keep mode, whose single slot (K == 1) is the tip row
    const bool keep_rs = K > 1 || (rs.mode & GDN_RS_MODE_KEEP_RING) != 0;

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
