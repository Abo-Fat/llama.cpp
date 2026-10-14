// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// AVX2 unsigned-byte/signed-byte maddubs + madd technique adapted from BitNet
// ggml-cpu/quants.c, local commit 8fb65cf044e7c833ef4545a9fee3b72ab89706ea.
// S11 sequential packing, group offset correction and executor are independent.
// See the repository LICENSE for the MIT permission and warranty notice.
#include "s11-cpu.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int G = S11_CPU_GROUP;
constexpr size_t B = S11_CPU_W2_BYTES;
constexpr int TILE = 16;
constexpr size_t MAX_WORKSPACE = size_t(512) << 20;

int fail(int status, char * error, size_t size, const char * message) {
    if (error && size) {
        std::snprintf(error, size, "%s", message);
    }
    return status;
}

float load_scale(const uint8_t * p, int codes = 32) {
    float s;
    std::memcpy(&s, p + codes, sizeof(s));
    return s;
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fff + ((u >> 16) & 1);
    return uint16_t(u >> 16);
}

float bf16(float f) {
    uint32_t u = uint32_t(bf16_bits(f)) << 16;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

int hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

int sum128(const int8_t * x) {
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < G; j += 32) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + j));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(one8, q), one16));
    }
    return hsum(acc);
}

int dot128(const uint8_t * p, const int8_t * x, int correction) {
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi16(1);
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    const __m256i a = _mm256_and_si256(packed, mask);
    const __m256i b = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
    const __m256i c = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    const __m256i d = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
    const __m256i ab0 = _mm256_unpacklo_epi8(a, b), ab1 = _mm256_unpackhi_epi8(a, b);
    const __m256i cd0 = _mm256_unpacklo_epi8(c, d), cd1 = _mm256_unpackhi_epi8(c, d);
    const __m256i t0 = _mm256_unpacklo_epi16(ab0, cd0), t1 = _mm256_unpackhi_epi16(ab0, cd0);
    const __m256i t2 = _mm256_unpacklo_epi16(ab1, cd1), t3 = _mm256_unpackhi_epi16(ab1, cd1);
    const __m256i w[4] = {
        _mm256_permute2x128_si256(t0, t1, 0x20), _mm256_permute2x128_si256(t2, t3, 0x20),
        _mm256_permute2x128_si256(t0, t1, 0x31), _mm256_permute2x128_si256(t2, t3, 0x31),
    };
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < 4; ++j) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + j * 32));
        // Unsigned codes <=2, signed activations [-128,127]: pair magnitude <=512.
        // maddubs cannot saturate; correction is separate for every K128 group.
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w[j], q), one));
    }
    return hsum(acc) - correction;
}

bool valid_w2(const uint8_t * p) {
    for (int j = 0; j < 32; ++j) {
        if ((p[j] & (p[j] >> 1) & 0x55) != 0) {
            return false;
        }
    }
    const float s = load_scale(p);
    return std::isfinite(s) && s >= 0;
}

bool quantize128(const float * x, int8_t * q, float * scale, int32_t * sum = nullptr) {
    float denom = 1e-5f;
    for (int j = 0; j < G; ++j) {
        denom = std::max(denom, std::abs(x[j]));
    }
    const float inverse = (1.0f / denom) * 127.0f;
    *scale = 1.0f / inverse;
    if (!std::isfinite(*scale) || inverse <= 0) {
        return false;
    }
    for (int j = 0; j < G; j += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_set1_ps(inverse));
        const __m256 r = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256 clamped = _mm256_min_ps(_mm256_max_ps(r, _mm256_set1_ps(-128)), _mm256_set1_ps(127));
        alignas(32) int32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), _mm256_cvttps_epi32(clamped));
        for (int i = 0; i < 8; ++i) {
            q[j + i] = int8_t(lanes[i]);
        }
    }
    if (sum) {
        *sum = sum128(q);
    }
    return true;
}

float linear(const int8_t * q, const float * s, const int32_t * sums, const uint8_t * w, int k) {
    float value = 0;
    for (int g = 0; g < k / G; ++g) {
        const uint8_t * block = w + g * B;
        const float partial = float(dot128(block, q + g * G, sums[g]));
        value = value + (partial * s[g]) * load_scale(block);
    }
    return bf16(value);
}

bool finite(const float * x, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        if (!std::isfinite(x[j])) {
            return false;
        }
    }
    return true;
}

size_t checked_mul(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) {
        throw std::invalid_argument("Size overflow");
    }
    return a * b;
}

struct Route {
    int token, slot, expert;
    float probability;
};
} // namespace

struct s11_cpu_pool {
    using Callback = void (*)(void *, size_t);
    std::mutex submission, mutex;
    std::condition_variable work, complete;
    std::vector<std::thread> workers;
    cpu_set_t cpus;
    std::atomic<size_t> next{0};
    size_t count = 0, generation = 0;
    int pending = 0, threads;
    bool stop = false;
    Callback callback = nullptr;
    void * context = nullptr;

    explicit s11_cpu_pool(int n, const cpu_set_t & mask) : cpus(mask), threads(n) {
        workers.reserve(n - 1);
        try {
            for (int i = 1; i < n; ++i) {
                workers.emplace_back([this] {
                    size_t seen = 0;
                    std::unique_lock<std::mutex> lock(mutex);
                    while (true) {
                        work.wait(lock, [&] { return stop || generation != seen; });
                        if (stop) {
                            return;
                        }
                        seen = generation;
                        lock.unlock();
                        const unsigned old_mxcsr = _mm_getcsr();
                        const int old_round = std::fegetround();
                        std::fesetround(FE_TONEAREST);
                        _mm_setcsr(_mm_getcsr() & ~((1u << 15) | (1u << 6)));
                        consume();
                        std::fesetround(old_round);
                        _mm_setcsr(old_mxcsr);
                        lock.lock();
                        if (--pending == 0) {
                            complete.notify_one();
                        }
                    }
                });
                if (pthread_setaffinity_np(workers.back().native_handle(), sizeof(cpus), &cpus) != 0) {
                    throw std::runtime_error("Cannot pin CPU worker");
                }
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~s11_cpu_pool() { shutdown(); }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        work.notify_all();
        for (auto & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    bool caller_affinity() const {
        cpu_set_t current;
        if (sched_getaffinity(0, sizeof(current), &current) != 0) {
            return false;
        }
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &current) && !CPU_ISSET(i, &cpus)) {
                return false;
            }
        }
        return true;
    }

    void consume() {
        size_t j;
        while ((j = next.fetch_add(1, std::memory_order_relaxed)) < count) {
            callback(context, j);
        }
    }

    void run(size_t jobs, Callback fn, void * ctx) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            count = jobs;
            callback = fn;
            context = ctx;
            next.store(0, std::memory_order_relaxed);
            pending = threads - 1;
            ++generation;
        }
        work.notify_all();
        consume();
        std::unique_lock<std::mutex> lock(mutex);
        complete.wait(lock, [&] { return pending == 0; });
    }
};

struct s11_cpu_executor {
    s11_cpu_pool * pool;
    s11_cpu_weights weights;
    int max_tokens, max_slots, tokens = 0, slots = 0, active = 0;
    size_t bytes, source_bytes;
    uint64_t calls = 0, assignments = 0;
    std::vector<Route> routes;
    std::vector<int8_t> qx, qm;
    std::vector<float> sx, sm, fc1, middle;
    std::vector<int32_t> sums_x, sums_m;
    const float * input = nullptr;
    uint16_t * output = nullptr;
    std::atomic<bool> numeric{false};

    s11_cpu_executor(s11_cpu_pool * p, const s11_cpu_weights & w, int mt, int ms, size_t limit)
        : pool(p), weights(w), max_tokens(mt), max_slots(ms) {
        const size_t r = checked_mul(mt, ms), h = w.hidden, f = w.intermediate;
        bytes = checked_mul(r, sizeof(Route));
        bytes += checked_mul(mt, h + (h / G) * (sizeof(float) + sizeof(int32_t)));
        bytes += checked_mul(r, f + (f / G) * (sizeof(float) + sizeof(int32_t)) + 3 * f * sizeof(float));
        if (bytes > limit || bytes > MAX_WORKSPACE) {
            throw std::invalid_argument("Workspace budget exceeded (maximum 512 MiB)");
        }
        source_bytes = w.gate_bytes + w.up_bytes + w.down_bytes;
        routes.resize(r);
        qx.resize(mt * h); sx.resize(mt * h / G); sums_x.resize(mt * h / G);
        qm.resize(r * f); sm.resize(r * f / G); sums_m.resize(r * f / G);
        fc1.resize(r * 2 * f); middle.resize(r * f);
    }

    static void input_job(void * ctx, size_t job) {
        auto & e = *static_cast<s11_cpu_executor *>(ctx);
        if (!quantize128(e.input + job * G, e.qx.data() + job * G, e.sx.data() + job, e.sums_x.data() + job)) {
            e.numeric.store(true, std::memory_order_relaxed);
        }
    }

    static void fc1_job(void * ctx, size_t job) {
        auto & e = *static_cast<s11_cpu_executor *>(ctx);
        const int h = e.weights.hidden, f = e.weights.intermediate;
        const size_t tiles = (f + TILE - 1) / TILE, r = job / tiles, begin = (job % tiles) * TILE;
        const Route & route = e.routes[r];
        const int8_t * q = e.qx.data() + size_t(route.token) * h;
        const float * s = e.sx.data() + size_t(route.token) * (h / G);
        const int32_t * sums = e.sums_x.data() + size_t(route.token) * (h / G);
        const size_t stride = (h / G) * B;
        for (size_t j = begin; j < std::min(begin + TILE, size_t(f)); ++j) {
            const size_t offset = (size_t(route.expert) * f + j) * stride;
            e.fc1[r * 2 * f + j] = linear(q, s, sums, e.weights.gate + offset, h);
            e.fc1[r * 2 * f + f + j] = linear(q, s, sums, e.weights.up + offset, h);
        }
    }

    static void middle_job(void * ctx, size_t r) {
        auto & e = *static_cast<s11_cpu_executor *>(ctx);
        const int f = e.weights.intermediate;
        float * mid = e.middle.data() + r * f;
        for (int j = 0; j < f; ++j) {
            const float gate = e.fc1[r * 2 * f + j], up = e.fc1[r * 2 * f + f + j];
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
            mid[j] = bf16(((gate * sigmoid) * up) * e.routes[r].probability);
        }
        if (!finite(mid, f)) {
            e.numeric.store(true, std::memory_order_relaxed);
            return;
        }
        for (int g = 0; g < f / G; ++g) {
            const size_t pos = r * (f / G) + g;
            if (!quantize128(mid + g * G, e.qm.data() + pos * G, e.sm.data() + pos, e.sums_m.data() + pos)) {
                e.numeric.store(true, std::memory_order_relaxed);
            }
        }
    }

    static void down_job(void * ctx, size_t job) {
        auto & e = *static_cast<s11_cpu_executor *>(ctx);
        const int h = e.weights.hidden, f = e.weights.intermediate;
        const size_t tiles = (h + TILE - 1) / TILE, r = job / tiles, begin = (job % tiles) * TILE;
        const Route & route = e.routes[r];
        const size_t stride = (f / G) * B;
        uint16_t * out = e.output + (size_t(route.token) * e.slots + route.slot) * h;
        for (size_t j = begin; j < std::min(begin + TILE, size_t(h)); ++j) {
            const float v = linear(e.qm.data() + r * f, e.sm.data() + r * (f / G),
                                   e.sums_m.data() + r * (f / G),
                                   e.weights.down + (size_t(route.expert) * h + j) * stride, f);
            if (!std::isfinite(v)) {
                e.numeric.store(true, std::memory_order_relaxed);
            }
            out[j] = bf16_bits(v);
        }
    }
};

extern "C" int s11_cpu_pool_create(int threads, const int * cpus, int n_cpus,
                                   s11_cpu_pool ** out, char * error, size_t size) {
    if (!out) {
        return fail(S11_CPU_INVALID, error, size, "NULL pool output");
    }
    *out = nullptr;
    cpu_set_t allowed, selected;
    CPU_ZERO(&selected);
    if (!cpus || n_cpus < 1 || n_cpus > 8 || threads < 1 || threads > n_cpus ||
        sched_getaffinity(0, sizeof(allowed), &allowed) != 0 || !__builtin_cpu_supports("avx2")) {
        return fail(S11_CPU_INVALID, error, size, "Require AVX2 and 1..8 allowed CPUs, threads <= CPUs");
    }
    for (int i = 0; i < n_cpus; ++i) {
        const int cpu = cpus[i];
        if (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed) || CPU_ISSET(cpu, &selected)) {
            return fail(S11_CPU_INVALID, error, size, "Invalid, duplicate or disallowed CPU ID");
        }
        CPU_SET(cpu, &selected);
    }
    try {
        *out = new s11_cpu_pool(threads, selected);
        return S11_CPU_OK;
    } catch (const std::exception & ex) {
        return fail(S11_CPU_RESOURCE, error, size, ex.what());
    }
}

extern "C" void s11_cpu_pool_destroy(s11_cpu_pool * pool) { delete pool; }

extern "C" int s11_cpu_executor_create(s11_cpu_pool * pool, const s11_cpu_weights * w,
                                       int mt, int ms, size_t limit, s11_cpu_executor ** out,
                                       char * error, size_t size) {
    if (!out) {
        return fail(S11_CPU_INVALID, error, size, "NULL executor output");
    }
    *out = nullptr;
    if (!pool || !w || w->abi_version != S11_CPU_ABI_VERSION || w->experts < 1 || w->experts > 256 ||
        w->hidden != 2048 || w->intermediate != 512 || mt < 1 || mt > 4096 || ms < 1 || ms > 8 ||
        !w->gate || !w->up || !w->down || !limit) {
        return fail(S11_CPU_INVALID, error, size, "Invalid S11 executor descriptor or capacity (H2048/F512)");
    }
    const size_t gate_bytes = size_t(w->experts) * w->intermediate * (w->hidden / G) * B;
    const size_t down_bytes = size_t(w->experts) * w->hidden * (w->intermediate / G) * B;
    if (w->gate_bytes != gate_bytes || w->up_bytes != gate_bytes || w->down_bytes != down_bytes) {
        return fail(S11_CPU_INVALID, error, size, "Packed weight byte length mismatch");
    }
    for (int matrix = 0; matrix < 3; ++matrix) {
        const uint8_t * p = matrix == 0 ? w->gate : matrix == 1 ? w->up : w->down;
        const size_t n = matrix == 2 ? down_bytes : gate_bytes;
        for (size_t j = 0; j < n; j += B) {
            if (!valid_w2(p + j)) {
                return fail(S11_CPU_INVALID, error, size, "Reserved W2 code 3 or negative/nonfinite scale");
            }
        }
    }
    try {
        *out = new s11_cpu_executor(pool, *w, mt, ms, limit);
        return S11_CPU_OK;
    } catch (const std::invalid_argument & ex) {
        return fail(S11_CPU_INVALID, error, size, ex.what());
    } catch (const std::exception & ex) {
        return fail(S11_CPU_RESOURCE, error, size, ex.what());
    }
}

extern "C" void s11_cpu_executor_destroy(s11_cpu_executor * e) { delete e; }

extern "C" int s11_cpu_executor_stats(const s11_cpu_executor * e, s11_cpu_stats * s) {
    if (!e || !s) {
        return S11_CPU_INVALID;
    }
    *s = {e->source_bytes, e->bytes, e->calls, e->assignments, e->pool->threads, e->max_tokens, e->max_slots};
    return S11_CPU_OK;
}

extern "C" int s11_cpu_execute(s11_cpu_executor * e, const float * x, const int32_t * ids,
                               const float * probabilities, const uint8_t * miss,
                               int tokens, int slots, uint16_t * output, s11_cpu_trace * trace,
                               char * error, size_t size) {
    if (trace) {
        *trace = {};
    }
    if (!e || !x || !ids || !probabilities || !output || tokens < 1 || tokens > e->max_tokens ||
        slots < 1 || slots > e->max_slots) {
        return fail(S11_CPU_INVALID, error, size, "Invalid execution pointers or shape/capacity");
    }
    std::unique_lock<std::mutex> submission(e->pool->submission, std::try_to_lock);
    if (!submission.owns_lock()) {
        return fail(S11_CPU_BUSY, error, size, "CPU pool is busy");
    }
    if (!e->pool->caller_affinity()) {
        return fail(S11_CPU_INVALID, error, size, "Caller affinity exceeds the pool's <=8 CPU set");
    }
    if (std::fegetround() != FE_TONEAREST || (_mm_getcsr() & ((1u << 15) | (1u << 6)))) {
        return fail(S11_CPU_INVALID, error, size, "Require nearest-even FP environment with FTZ/DAZ disabled");
    }
    if (!finite(x, size_t(tokens) * e->weights.hidden)) {
        return fail(S11_CPU_INVALID, error, size, "Nonfinite input activation");
    }
    int active = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            const size_t pos = size_t(t) * slots + s;
            const int id = ids[pos];
            const float probability = probabilities[pos];
            if (id < -1 || id >= e->weights.experts || !std::isfinite(probability) ||
                probability < 0 || probability > 1 || (miss && miss[pos] > 1)) {
                return fail(S11_CPU_INVALID, error, size, "Invalid expert ID, probability or miss byte");
            }
            if (id >= 0 && (!miss || miss[pos])) {
                e->routes[active++] = {t, s, id, probability};
            }
        }
    }
    e->input = x; e->output = output; e->tokens = tokens; e->slots = slots; e->active = active;
    e->numeric.store(false, std::memory_order_relaxed);
    std::memset(output, 0, size_t(tokens) * slots * e->weights.hidden * sizeof(uint16_t));
    if (active) {
        e->pool->run(size_t(tokens) * (e->weights.hidden / G), s11_cpu_executor::input_job, e);
        if (e->numeric.load(std::memory_order_relaxed)) {
            return fail(S11_CPU_NUMERIC, error, size, "A128 input scale overflow");
        }
        e->pool->run(size_t(active) * ((e->weights.intermediate + TILE - 1) / TILE), s11_cpu_executor::fc1_job, e);
        if (!finite(e->fc1.data(), size_t(active) * 2 * e->weights.intermediate)) {
            return fail(S11_CPU_NUMERIC, error, size, "FC1 BF16 overflow");
        }
        e->pool->run(active, s11_cpu_executor::middle_job, e);
        if (e->numeric.load(std::memory_order_relaxed)) {
            return fail(S11_CPU_NUMERIC, error, size, "SwiGLU BF16 or A128 intermediate overflow");
        }
        e->pool->run(size_t(active) * ((e->weights.hidden + TILE - 1) / TILE), s11_cpu_executor::down_job, e);
        if (e->numeric.load(std::memory_order_relaxed)) {
            return fail(S11_CPU_NUMERIC, error, size, "FC2 BF16 overflow");
        }
    }
    ++e->calls;
    e->assignments += active;
    if (trace) {
        *trace = {active, e->fc1.data(), e->middle.data(), e->qx.data(), e->sx.data(), e->qm.data(), e->sm.data()};
    }
    return S11_CPU_OK;
}

extern "C" int s11_cpu_quantize_a128(const float * x, int k, int8_t * q, float * scales,
                                    char * error, size_t size) {
    if (!x || !q || !scales || k <= 0 || k % G || !finite(x, k) || std::fegetround() != FE_TONEAREST) {
        return fail(S11_CPU_INVALID, error, size, "Invalid A128 pointers, K, activation or rounding mode");
    }
    for (int g = 0; g < k / G; ++g) {
        if (!quantize128(x + g * G, q + g * G, scales + g)) {
            return fail(S11_CPU_NUMERIC, error, size, "A128 scale overflow");
        }
    }
    return S11_CPU_OK;
}

extern "C" int s11_cpu_dot128(const uint8_t * p, const int8_t * q, int32_t * dot,
                             char * error, size_t size) {
    if (!p || !q || !dot || !valid_w2(p)) {
        return fail(S11_CPU_INVALID, error, size, "Invalid W2 block, activation or output");
    }
    *dot = dot128(p, q, sum128(q));
    return S11_CPU_OK;
}

namespace {
int block_bytes(int type) {
    return type == S11_CPU_TYPE_W2 || type == S11_CPU_TYPE_W2_BF16 ? 36 :
           type == S11_CPU_TYPE_W4 ? 68 : type == S11_CPU_TYPE_I8 ? 132 : 0;
}

int code(const uint8_t * p, int j, int type) {
    if (type == S11_CPU_TYPE_I8) {
        return int(int8_t(p[j]));
    }
    if (type == S11_CPU_TYPE_W4) {
        const int u = (p[j / 2] >> ((j % 2) * 4)) & 15;
        return (u ^ 8) - 8;
    }
    return int((p[j / 4] >> ((j % 4) * 2)) & 3) - 1;
}

bool valid_block(const uint8_t * p, int type, int bytes) {
    if (type == S11_CPU_TYPE_W2 || type == S11_CPU_TYPE_W2_BF16) {
        return valid_w2(p);
    }
    const float s = load_scale(p, bytes - 4);
    if (!std::isfinite(s) || s < 0 || (type == S11_CPU_TYPE_I8 && s == 0)) {
        return false;
    }
    if (type == S11_CPU_TYPE_I8) {
        for (int j = 0; j < G; ++j) {
            if (p[j] == 128) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

extern "C" int s11_cpu_dequantize(int type, const void * packed, int k, float * out,
                                 char * error, size_t size) {
    const int bytes = block_bytes(type);
    if (!bytes || !packed || !out || k <= 0 || k % G) {
        return fail(S11_CPU_INVALID, error, size, "Invalid dequant type, pointers or K");
    }
    const auto * p = static_cast<const uint8_t *>(packed);
    for (int g = 0; g < k / G; ++g, p += bytes) {
        if (!valid_block(p, type, bytes)) {
            return fail(S11_CPU_INVALID, error, size, "Invalid packed code or scale");
        }
        const float s = load_scale(p, bytes - 4);
        for (int j = 0; j < G; ++j) {
            const float v = type == S11_CPU_TYPE_I8 ? float(code(p, j, type)) / s : float(code(p, j, type)) * s;
            out[g * G + j] = type == S11_CPU_TYPE_W2 ? v : bf16(v);
        }
    }
    return finite(out, k) ? S11_CPU_OK : fail(S11_CPU_NUMERIC, error, size, "Dequant overflow");
}

extern "C" int s11_cpu_vec_dot(int type, const void * packed, const float * x, int k,
                              float * out, char * error, size_t size) {
    const int bytes = block_bytes(type);
    if (!bytes || !packed || !x || !out || k <= 0 || k % G || !finite(x, k) ||
        std::fegetround() != FE_TONEAREST) {
        return fail(S11_CPU_INVALID, error, size, "Invalid vec_dot type, pointers, K, input or rounding");
    }
    const auto * p = static_cast<const uint8_t *>(packed);
    float value = 0;
    for (int g = 0; g < k / G; ++g, p += bytes) {
        if (!valid_block(p, type, bytes)) {
            return fail(S11_CPU_INVALID, error, size, "Invalid packed code or scale");
        }
        alignas(32) int8_t q[G];
        float act_scale = 0;
        if (type != S11_CPU_TYPE_I8 && !quantize128(x + g * G, q, &act_scale)) {
            return fail(S11_CPU_NUMERIC, error, size, "A128 scale overflow");
        }
        const float s = load_scale(p, bytes - 4);
        if (type == S11_CPU_TYPE_W2) {
            value = value + (float(dot128(p, q, sum128(q))) * act_scale) * s;
        } else {
            float absmax = 1e-5f;
            if (type != S11_CPU_TYPE_I8) {
                for (int j = 0; j < G; ++j) { absmax = std::max(absmax, std::abs(x[g * G + j])); }
            }
            const float inverse = (1.0f / absmax) * 127.0f;
            for (int j = 0; j < G; ++j) {
                const float a = bf16(type == S11_CPU_TYPE_I8 ? x[g * G + j] : float(q[j]) / inverse);
                const float w = bf16(type == S11_CPU_TYPE_I8 ? float(code(p, j, type)) / s : float(code(p, j, type)) * s);
                value = value + a * w;
            }
        }
    }
    *out = bf16(value);
    return std::isfinite(*out) ? S11_CPU_OK : fail(S11_CPU_NUMERIC, error, size, "vec_dot BF16 overflow");
}

extern "C" int s11_cpu_prepare_row(int type, const float * x, int k, float * prepared,
                                  char * error, size_t size) {
    if (type < S11_CPU_TYPE_W2_BF16 || type > S11_CPU_TYPE_I8 || !x || !prepared ||
        k <= 0 || k % G || !finite(x, k) || std::fegetround() != FE_TONEAREST) {
        return fail(S11_CPU_INVALID, error, size, "Invalid prepared activation");
    }
    for (int g = 0; g < k / G; ++g) {
        if (type == S11_CPU_TYPE_I8) {
            for (int j = 0; j < G; ++j) { prepared[g * G + j] = bf16(x[g * G + j]); }
        } else {
            alignas(32) int8_t q[G];
            float scale;
            if (!quantize128(x + g * G, q, &scale)) {
                return fail(S11_CPU_NUMERIC, error, size, "Prepared activation overflow");
            }
            float maximum = 1e-5f;
            for (int j = 0; j < G; ++j) { maximum = std::max(maximum, std::abs(x[g * G + j])); }
            const float inverse = (1.0f / maximum) * 127.0f;
            for (int j = 0; j < G; ++j) { prepared[g * G + j] = bf16(float(q[j]) / inverse); }
        }
    }
    return S11_CPU_OK;
}

extern "C" int s11_cpu_prepared_dot(int type, const void * packed, const float * prepared, int k,
                                   float * out, char * error, size_t size) {
    const int bytes = block_bytes(type);
    if (type < S11_CPU_TYPE_W2_BF16 || type > S11_CPU_TYPE_I8 || !packed || !prepared ||
        !out || k <= 0 || k % G) {
        return fail(S11_CPU_INVALID, error, size, "Invalid prepared dot");
    }
    const auto * p = static_cast<const uint8_t *>(packed);
    float value = 0;
    const __m256i mask = _mm256_set1_epi32(int(0xffff0000u));
    for (int g = 0; g < k / G; ++g, p += bytes) {
        const __m256 scale = _mm256_set1_ps(load_scale(p, bytes - 4));
        for (int j = 0; j < G; j += 8) {
            alignas(32) int32_t codes[8];
            for (int i = 0; i < 8; ++i) { codes[i] = code(p, j + i, type); }
            const __m256 c = _mm256_cvtepi32_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(codes)));
            const __m256 restored = type == S11_CPU_TYPE_I8 ? _mm256_div_ps(c, scale) : _mm256_mul_ps(c, scale);
            const __m256i bits = _mm256_castps_si256(restored);
            const __m256i tie = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
            const __m256 rounded = _mm256_castsi256_ps(_mm256_and_si256(
                _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), tie)), mask));
            alignas(32) float products[8];
            _mm256_store_ps(products, _mm256_mul_ps(_mm256_loadu_ps(prepared + g * G + j), rounded));
            for (float product : products) { value = value + product; }
        }
    }
    *out = bf16(value);
    return std::isfinite(*out) ? S11_CPU_OK : fail(S11_CPU_NUMERIC, error, size, "Prepared dot overflow");
}
