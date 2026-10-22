#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

struct argmax_value_index {
    float value;
    int32_t index;
};

static __device__ __forceinline__ argmax_value_index argmax_best(argmax_value_index a, argmax_value_index b) {
    return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

static __device__ __forceinline__ argmax_value_index argmax_block_reduce(argmax_value_index best) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        best = argmax_best(best, {
            __shfl_xor_sync(0xFFFFFFFF, best.value, offset, WARP_SIZE),
            __shfl_xor_sync(0xFFFFFFFF, best.index, offset, WARP_SIZE)});
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = best.value;
            shared_argmax[warp_id] = best.index;
        }

        __syncthreads();

        if (warp_id == 0) {
            best = lane_id < n_warps ? argmax_value_index{shared_maxval[lane_id], shared_argmax[lane_id]} :
                argmax_value_index{-INFINITY, INT32_MAX};
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                best = argmax_best(best, {
                    __shfl_xor_sync(0xFFFFFFFF, best.value, offset, WARP_SIZE),
                    __shfl_xor_sync(0xFFFFFFFF, best.index, offset, WARP_SIZE)});
            }
        }
    }

    return best;
}

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;
    argmax_value_index best{-INFINITY, INT32_MAX};
    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        best = argmax_best(best, {x[row * ncols + col], int32_t(col)});
    }
    best = argmax_block_reduce(best);
    if (threadIdx.x == 0) {
        dst[row] = best.value == -INFINITY ? 0 : best.index;
    }
}

static constexpr int ARGMAX_TILE = 1024;

static __global__ void argmax_f32_partial(const float * x, argmax_value_index * partials, int64_t ncols, int tiles) {
    const int64_t start = int64_t(blockIdx.x) * ARGMAX_TILE;
    const int64_t row = blockIdx.y;
    argmax_value_index best{-INFINITY, INT32_MAX};
    for (int64_t col = start + threadIdx.x; col < ncols && col < start + ARGMAX_TILE; col += blockDim.x) {
        best = argmax_best(best, {x[row * ncols + col], int32_t(col)});
    }
    best = argmax_block_reduce(best);
    if (threadIdx.x == 0) {
        partials[row * tiles + blockIdx.x] = best;
    }
}

static __global__ void argmax_f32_finish(const argmax_value_index * partials, int32_t * dst, int tiles) {
    const int row = blockIdx.x;
    argmax_value_index best{-INFINITY, INT32_MAX};
    for (int i = threadIdx.x; i < tiles; i += blockDim.x) {
        best = argmax_best(best, partials[size_t(row) * tiles + i]);
    }
    best = argmax_block_reduce(best);
    if (threadIdx.x == 0) {
        dst[row] = best.value == -INFINITY ? 0 : best.index;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    GGML_ASSERT(ne00 > 0 && ne00 <= INT32_MAX && nrows > 0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    // Wide, short matrices otherwise scan the whole vocabulary on one SM per row.
    if (ne00 >= 16384 && nrows <= 8) {
        const int tiles = int((ne00 + ARGMAX_TILE - 1) / ARGMAX_TILE);
        ggml_cuda_pool_alloc<argmax_value_index> partials(ctx.pool(), size_t(tiles) * nrows);
        argmax_f32_partial<<<dim3(tiles, nrows), 256, 0, stream>>>(src0_d, partials.get(), ne00, tiles);
        CUDA_CHECK(cudaGetLastError());
        argmax_f32_finish<<<nrows, 256, 0, stream>>>(partials.get(), dst_d, tiles);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
