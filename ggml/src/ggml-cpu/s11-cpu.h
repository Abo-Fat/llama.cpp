#ifndef GGML_S11_CPU_H
#define GGML_S11_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    S11_CPU_ABI_VERSION = 1,
    S11_CPU_GROUP = 128,
    S11_CPU_W2_BYTES = 36,
    S11_CPU_W4_BYTES = 68,
    S11_CPU_I8_BYTES = 132,
    S11_CPU_TYPE_W2 = 43,
    S11_CPU_TYPE_W2_BF16 = 44,
    S11_CPU_TYPE_W4 = 45,
    S11_CPU_TYPE_I8 = 46,
};

enum s11_cpu_status {
    S11_CPU_OK = 0,
    S11_CPU_INVALID = 1,
    S11_CPU_BUSY = 2,
    S11_CPU_RESOURCE = 3,
    S11_CPU_NUMERIC = 4,
};

typedef struct s11_cpu_pool s11_cpu_pool;
typedef struct s11_cpu_executor s11_cpu_executor;

/* Little-endian, row-major [expert][row][K/128][block_bytes].
 * W2 has sequential four K codes per byte, low bits first: code = weight + 1.
 * W4 has two signed two's-complement nibbles per byte, low nibble first.
 * Each block ends with the original FP32 scale (I8: reciprocal).
 * Borrowed immutable buffers, including mmap storage, outlive all executors.
 * Gate/up: [E,F,H/128,36], down: [E,H,F/128,36]. */
typedef struct s11_cpu_weights {
    uint32_t abi_version;
    int32_t experts;
    int32_t hidden;
    int32_t intermediate;
    const uint8_t * gate;
    size_t gate_bytes;
    const uint8_t * up;
    size_t up_bytes;
    const uint8_t * down;
    size_t down_bytes;
} s11_cpu_weights;

typedef struct s11_cpu_stats {
    size_t source_bytes;
    size_t workspace_bytes;
    uint64_t calls;
    uint64_t assignments;
    int32_t pool_threads; /* Includes the synchronous calling thread. */
    int32_t max_tokens;
    int32_t max_slots;
} s11_cpu_stats;

/* Diagnostics only: borrowed compact active-route buffers, invalidated by the
 * next execute or destroy. All BF16 boundaries are represented by rounded F32. */
typedef struct s11_cpu_trace {
    int32_t active;
    const float * fc1;       /* [active,2F], gate then up */
    const float * middle;    /* [active,F], SwiGLU * probability, BF16 */
    const int8_t * input_q;  /* [tokens,H] */
    const float * input_s;   /* [tokens,H/128] */
    const int8_t * middle_q; /* [active,F] */
    const float * middle_s;  /* [active,F/128] */
} s11_cpu_trace;

/* All functions return status, optionally write a NUL-terminated error.
 * One pool per runtime, shared by executors. threads-1 workers are created here,
 * never in execute. CPUs must be distinct, allowed by current affinity, <= 8.
 * During execute the calling thread's affinity must be a subset of these CPUs.
 * No concurrent calls on a pool: contention returns BUSY. Destroy only while
 * quiescent, executors before pool. No function changes the caller's affinity.
 * Successful create sets *out; failed create sets it to NULL. */
int s11_cpu_pool_create(int threads, const int * cpus, int n_cpus,
                        s11_cpu_pool ** out, char * error, size_t error_size);
void s11_cpu_pool_destroy(s11_cpu_pool * pool);
int s11_cpu_executor_create(s11_cpu_pool * pool, const s11_cpu_weights * weights,
                            int max_tokens, int max_slots, size_t workspace_limit,
                            s11_cpu_executor ** out, char * error, size_t error_size);
void s11_cpu_executor_destroy(s11_cpu_executor * executor);
int s11_cpu_executor_stats(const s11_cpu_executor * executor, s11_cpu_stats * stats);

/* Synchronous complete expert chain. x: F32[tokens,H] (caller supplies BF16
 * rounded model input). ids/probabilities/miss: [tokens,slots], probability
 * finite in [0,1], miss byte 0/1. NULL miss means all valid IDs run.
 * ID -1 is inactive; other IDs must be in range even when masked out.
 * Output: BF16 uint16_t[tokens,slots,H], inactive slots are zero. No aggregation.
 * No file IO, weight expansion, heap allocation, or worker creation in execute.
 * Buffers must not alias each other, weights or workspace; valid until return.
 * On error output is unspecified. Compact trace order is token then slot. */
int s11_cpu_execute(s11_cpu_executor * executor, const float * x,
                     const int32_t * ids, const float * probabilities,
                     const uint8_t * miss, int tokens, int slots,
                     uint16_t * output, s11_cpu_trace * trace,
                     char * error, size_t error_size);

/* Small checked math entrypoints for offline qualification and ggml adapters.
 * No pool/allocation. These do not implement ggml's callback ABI themselves.
 * vec_dot W2: A128 integer group dot, (dot * act_scale) * weight_scale,
 * left-order FP32 groups, final BF16. W2_BF16/W4: A128 fake BF16 operands.
 * I8: BF16(x) * BF16(code / reciprocal), no activation quantization.
 * dequant returns BF16-restored F32 except integer W2 returns code * scale. */
int s11_cpu_quantize_a128(const float * x, int k, int8_t * q, float * scales,
                          char * error, size_t error_size);
int s11_cpu_dot128(const uint8_t * block, const int8_t * q, int32_t * dot,
                   char * error, size_t error_size);
int s11_cpu_dequantize(int type, const void * packed, int k, float * output,
                       char * error, size_t error_size);
int s11_cpu_vec_dot(int type, const void * packed, const float * x, int k,
                    float * output, char * error, size_t error_size);
/* GGML fast adapter for loader-validated immutable W2_BF16/W4/I8 weights.
 * Preparation validates dynamic input once; dot keeps scalar K accumulation order. */
int s11_cpu_prepare_row(int type, const float * x, int k, float * prepared,
                        char * error, size_t error_size);
int s11_cpu_prepared_dot(int type, const void * packed, const float * prepared, int k,
                         float * output, char * error, size_t error_size);
int s11_cpu_prepared_tile(int type, const void * packed, size_t row_stride,
                          const float * prepared, int k, int rows, int columns,
                          float * output, size_t output_stride, char * error, size_t error_size);
int s11_cpu_bf16_head_dot(const uint16_t * weights, const float * prepared, int k,
                          float * output, char * error, size_t error_size);
void s11_cpu_record_dispatch(int path);
void s11_cpu_dispatch_counts(uint64_t counts[4]);

#ifdef __cplusplus
}
#endif
#endif
