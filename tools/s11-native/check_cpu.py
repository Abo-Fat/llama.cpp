#!/usr/bin/env python3
"""Offline RN03-RN05 CPU qualification; no model runtime, CUDA or old runners.

Build (from workspace_qwen):
  g++ -std=c++17 -O3 -mavx2 -fno-fast-math -ffp-contract=off -pthread \
    -Wall -Wextra -Werror -fPIC -shared s11-native/ggml/src/ggml-cpu/s11-cpu.cpp \
    -o baseline-envs/s11-native/libs11-cpu.so
Run with the existing vllm .venv Python and <=8-core taskset affinity.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import numpy as np
from safetensors import safe_open

H, F, G, B = 2048, 512, 128, 36
ROOT = Path(__file__).resolve().parents[3]

NATIVE_SMOKE = r"""
#include "s11-cpu.h"
#include <immintrin.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <vector>
static std::atomic<size_t> allocations{0};
void * operator new(size_t n) {
    ++allocations;
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
int main(int argc, char ** argv) {
    require(argc == 2, "Expected packed source path");
    constexpr size_t matrix_bytes = size_t(3) * 512 * 16 * 36;
    std::vector<uint8_t> source(3 * matrix_bytes);
    std::ifstream file(argv[1], std::ios::binary);
    file.read(reinterpret_cast<char *>(source.data()), source.size());
    require(size_t(file.gcount()) == source.size(), "Truncated source");
    file.close();
    s11_cpu_weights w{1, 3, 2048, 512,
        source.data(), matrix_bytes, source.data() + matrix_bytes, matrix_bytes,
        source.data() + 2 * matrix_bytes, matrix_bytes};
    cpu_set_t mask;
    require(sched_getaffinity(0, sizeof(mask), &mask) == 0, "Cannot read affinity");
    int cpus[8], count = 0;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &mask)) {
        require(count < 8, "Affinity exceeds eight cores"); cpus[count++] = c;
    }
    std::vector<float> x(2048), probabilities(8, 0.125f);
    std::vector<int32_t> ids{0, 0, 1, 2, 1, 0, 2, 2};
    std::vector<uint16_t> output(8 * 2048), expected;
    for (int j = 0; j < 2048; ++j) x[j] = float((j % 31) - 15) / 16.0f;
    char error[256];
    std::printf("{\"direct_cpp_abi\":true,\"weights_read_once\":true,\"benchmarks\":[");
    for (int iteration = 0; iteration < 2; ++iteration) {
        const int threads = iteration == 0 ? 1 : count;
        s11_cpu_pool * pool = nullptr;
        s11_cpu_executor * executor = nullptr, * shared = nullptr;
        require(s11_cpu_pool_create(threads, cpus, count, &pool, error, sizeof(error)) == 0, error);
        require(s11_cpu_executor_create(pool, &w, 17, 8, 64u << 20, &executor, error, sizeof(error)) == 0, error);
        require(s11_cpu_executor_create(pool, &w, 1, 8, 64u << 20, &shared, error, sizeof(error)) == 0, error);
        auto execute = [&](s11_cpu_executor * e) {
            require(s11_cpu_execute(e, x.data(), ids.data(), probabilities.data(), nullptr,
                1, 8, output.data(), nullptr, error, sizeof(error)) == 0, error);
        };
        execute(executor);
        if (iteration == 0) expected = output;
        require(output == expected, "Thread-count changed output bits");
        execute(shared);
        require(output == expected, "Shared pool changed output bits");
        s11_cpu_executor_destroy(shared);
        double p50[2]{}, minimum[2]{};
        for (int cold = 0; cold < 2; ++cold) {
            double times[21];
            for (int run = 0; run < 21; ++run) {
                if (cold) {
                    for (size_t j = 0; j < source.size(); j += 64) _mm_clflush(source.data() + j);
                    _mm_mfence();
                }
                const size_t before = allocations.load();
                auto start = std::chrono::steady_clock::now();
                execute(executor);
                auto stop = std::chrono::steady_clock::now();
                require(allocations.load() == before, "Execute allocated C++ heap memory");
                require(output == expected, "Repeated execution changed output");
                times[run] = std::chrono::duration<double, std::milli>(stop - start).count();
            }
            std::sort(times, times + 21);
            p50[cold] = times[10]; minimum[cold] = times[0];
        }
        s11_cpu_executor_destroy(executor);
        s11_cpu_pool_destroy(pool);
        if (iteration) std::printf(",");
        std::printf("{\"threads\":%d,\"hot_p50_ms\":%.6f,\"hot_min_ms\":%.6f,"
                    "\"source_cold_p50_ms\":%.6f,\"source_cold_min_ms\":%.6f,"
                    "\"request_cpp_allocations\":0,\"shared_pool_exact\":true}",
                    threads, p50[0], minimum[0], p50[1], minimum[1]);
    }
    std::printf("]}\n");
}
"""


def native_smoke(library, packed):
    directory = library.parent
    binary = directory / "s11-cpu-smoke"
    source_path = directory / ".cpu-qualification-packed.bin"
    env = dict(os.environ, TMPDIR=str(directory))
    source_dir = ROOT / "s11-native/ggml/src/ggml-cpu"
    subprocess.run([
        "g++", "-x", "c++", "-", "-std=c++17", "-O3", "-mavx2",
        "-fno-fast-math", "-ffp-contract=off", "-pthread",
        "-I", str(source_dir), "-x", "none", str(library),
        "-Wl,--export-dynamic", "-Wl,-rpath," + str(directory), "-o", str(binary),
    ], input=NATIVE_SMOKE, text=True, check=True, env=env)
    try:
        with source_path.open("wb") as file:
            for a in packed:
                file.write(a.tobytes())
        completed = subprocess.run([str(binary), str(source_path)], text=True, capture_output=True, check=True)
        return json.loads(completed.stdout)
    finally:
        source_path.unlink(missing_ok=True)


class Weights(C.Structure):
    _fields_ = [
        ("abi_version", C.c_uint32), ("experts", C.c_int32),
        ("hidden", C.c_int32), ("intermediate", C.c_int32),
        ("gate", C.c_void_p), ("gate_bytes", C.c_size_t),
        ("up", C.c_void_p), ("up_bytes", C.c_size_t),
        ("down", C.c_void_p), ("down_bytes", C.c_size_t),
    ]


class Stats(C.Structure):
    _fields_ = [
        ("source_bytes", C.c_size_t), ("workspace_bytes", C.c_size_t),
        ("calls", C.c_uint64), ("assignments", C.c_uint64),
        ("pool_threads", C.c_int32), ("max_tokens", C.c_int32),
        ("max_slots", C.c_int32),
    ]


class Trace(C.Structure):
    _fields_ = [("active", C.c_int32)] + [
        (name, C.c_void_p) for name in
        ("fc1", "middle", "input_q", "input_s", "middle_q", "middle_s")
    ]


def ptr(a):
    return C.c_void_p(a.ctypes.data) if a is not None else None


def bf16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def output_f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def a128(x):
    grouped = x.astype(np.float32).reshape(-1, G)
    denom = np.maximum(np.max(np.abs(grouped), axis=1), np.float32(1e-5))
    inverse = (np.float32(1) / denom) * np.float32(127)
    q = np.clip(np.rint(grouped * inverse[:, None]), -128, 127).astype(np.int8)
    return q.reshape(x.shape), (np.float32(1) / inverse).reshape(*x.shape[:-1], x.shape[-1] // G)


def decode_source(words):
    """Checkpoint's byte-strided signed packing, not native W2 packing."""
    shifts = (np.arange(16) % 4) * 8 + (np.arange(16) // 4) * 2
    unsigned = ((words.astype(np.uint32)[..., None] >> shifts) & 3).astype(np.int8)
    assert not np.any(unsigned == 2), "Illegal checkpoint signed code 10"
    return ((unsigned ^ 2) - 2).reshape(*words.shape[:-1], words.shape[-1] * 16)


def repack(codes, scales):
    assert codes.dtype == np.int8 and scales.dtype == np.float32
    groups = (codes.astype(np.int16) + 1).reshape(*scales.shape, 32, 4)
    packed = np.sum(groups.astype(np.uint8) << (np.arange(4, dtype=np.uint8) * 2), axis=-1, dtype=np.uint8)
    out = np.empty((*scales.shape, B), dtype=np.uint8)
    out[..., :32] = packed
    out[..., 32:] = np.ascontiguousarray(scales).view(np.uint8).reshape(*scales.shape, 4)
    restored = (((out[..., :32, None] >> (np.arange(4, dtype=np.uint8) * 2)) & 3).astype(np.int8) - 1)
    assert np.array_equal(restored.reshape(codes.shape), codes)
    assert np.array_equal(out[..., 32:], scales.view(np.uint8).reshape(*scales.shape, 4))
    return np.ascontiguousarray(out)


def projection(q, act_scales, w, scales):
    result = np.zeros((q.shape[0], w.shape[0]), np.float32)
    for g in range(q.shape[1] // G):
        dot = q[:, g * G:(g + 1) * G].astype(np.int32) @ w[:, g * G:(g + 1) * G].astype(np.int32).T
        partial = dot.astype(np.float32) * act_scales[:, g, None]
        result = result + partial * scales[None, :, g]
    return bf16(result)


def reference(x, ids, probabilities, miss, codes, scales):
    qx, sx = a128(x)
    positions = [(t, s) for t in range(len(x)) for s in range(ids.shape[1])
                 if ids[t, s] >= 0 and (miss is None or miss[t, s])]
    active = len(positions)
    fc1 = np.zeros((active, 2 * F), np.float32)
    middle = np.zeros((active, F), np.float32)
    output = np.zeros((*ids.shape, H), np.float32)
    qm = np.zeros((active, F), np.int8)
    sm = np.zeros((active, F // G), np.float32)
    for expert in range(len(codes[0])):
        routes = [r for r, (t, s) in enumerate(positions) if ids[t, s] == expert]
        if not routes:
            continue
        tokens = [positions[r][0] for r in routes]
        gate = projection(qx[tokens], sx[tokens], codes[0][expert], scales[0][expert])
        up = projection(qx[tokens], sx[tokens], codes[1][expert], scales[1][expert])
        fc1[routes] = np.concatenate([gate, up], axis=1)
        probability = np.array([probabilities[positions[r]] for r in routes], np.float32)
        with np.errstate(over="ignore"):
            sigmoid = np.float32(1) / (np.float32(1) + np.exp(-gate))
        mid = bf16(((gate * sigmoid) * up) * probability[:, None])
        middle[routes] = mid
        q, s = a128(mid)
        qm[routes], sm[routes] = q, s
        out = projection(q, s, codes[2][expert], scales[2][expert])
        for i, r in enumerate(routes):
            output[positions[r]] = out[i]
    return output, fc1, middle, qx, sx, qm, sm


def comparison(actual, expected, exact=False):
    assert np.isfinite(actual).all()
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    norm = max(np.linalg.norm(expected.astype(np.float64)), 1e-30)
    rms = max(np.sqrt(np.mean(expected.astype(np.float64) ** 2)), 1e-30)
    result = {
        "bit_equal_fraction": float(np.mean(actual.view(np.uint32) == expected.view(np.uint32))),
        "relative_l2": float(np.linalg.norm(delta) / norm),
        "max_abs": float(np.max(np.abs(delta))),
        "max_abs_over_rms": float(np.max(np.abs(delta)) / rms),
    }
    if exact:
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), result
    else:
        assert result["relative_l2"] <= 0.002 and result["max_abs_over_rms"] <= 0.02, result
    return result


class Native:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.error = C.create_string_buffer(256)
        self.pool = C.c_void_p()
        self.executor = C.c_void_p()
        v, i, z = C.c_void_p, C.c_int, C.c_size_t
        signatures = {
            "pool_create": [i, v, i, v, v, z], "pool_destroy": [v],
            "executor_create": [v, v, i, i, z, v, v, z],
            "executor_destroy": [v], "executor_stats": [v, v],
            "execute": [v, v, v, v, v, i, i, v, v, v, z],
            "quantize_a128": [v, i, v, v, v, z],
            "dot128": [v, v, v, v, z],
            "dequantize": [i, v, i, v, v, z],
            "vec_dot": [i, v, v, i, v, v, z],
            "prepare_row": [i, v, i, v, v, z],
            "prepared_dot": [i, v, v, i, v, v, z],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, "s11_cpu_" + name)
            fn.argtypes = args
            fn.restype = None if name.endswith("destroy") else i

    def call(self, name, *args, expected=0):
        self.error.value = b""
        status = getattr(self.lib, "s11_cpu_" + name)(*args, self.error, len(self.error))
        assert status == expected, (name, status, self.error.value.decode())
        if expected:
            assert self.error.value, name
        return self.error.value.decode()

    def create(self, packed, threads, max_tokens=17):
        cpus = np.array(sorted(os.sched_getaffinity(0)), np.int32)
        self.call("pool_create", threads, ptr(cpus), len(cpus), C.byref(self.pool))
        self.weights = Weights(1, len(packed[0]), H, F, *sum(
            [(ptr(a), a.nbytes) for a in packed], ()))
        self.call("executor_create", self.pool, C.byref(self.weights), max_tokens, 8,
                  64 << 20, C.byref(self.executor))

    def execute(self, x, ids, probs, miss=None, expected=0):
        out = np.full((*ids.shape, H), 0xFFFF, np.uint16)
        trace = Trace()
        self.call("execute", self.executor, ptr(x), ptr(ids), ptr(probs), ptr(miss),
                  len(x), ids.shape[1], ptr(out), C.byref(trace), expected=expected)
        return out, trace

    def close(self):
        self.lib.s11_cpu_executor_destroy(self.executor)
        self.lib.s11_cpu_pool_destroy(self.pool)
        self.executor = C.c_void_p()
        self.pool = C.c_void_p()


def trace_array(address, shape, dtype):
    count = int(np.prod(shape))
    ctype = {np.float32: C.c_float, np.int8: C.c_int8}[dtype]
    return np.ctypeslib.as_array(C.cast(address, C.POINTER(ctype)), shape=(count,)).reshape(shape).copy()


def math_checks(native, rng):
    results = []
    for name, q in [
        ("signed_extremes", np.resize(np.array([-128, 127, -127, 0], np.int8), G)),
        ("all_negative_128", np.full(G, -128, np.int8)),
        ("all_positive_127", np.full(G, 127, np.int8)),
        ("random", rng.integers(-128, 128, G, dtype=np.int8)),
    ]:
        for weight in [-1, 0, 1, None]:
            w = rng.integers(-1, 2, G, dtype=np.int8) if weight is None else np.full(G, weight, np.int8)
            block = repack(w.reshape(1, G), np.ones((1, 1), np.float32)).reshape(-1)
            dot = C.c_int32()
            native.call("dot128", ptr(block), ptr(q), C.byref(dot))
            assert dot.value == int(q.astype(np.int32) @ w.astype(np.int32))
        results.append(name)
    groups = np.array([
        np.zeros(G),
        np.resize(np.array([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]), G),
        np.resize(np.array([1e-30, -1e-30, 1e-5, -1e-5]), G),
        np.resize(np.array([1e30, -1e30, 5e29, -5e29]), G),
        np.resize(np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max]), G),
    ], np.float32)
    q, s = np.empty(groups.shape, np.int8), np.empty((len(groups), 1), np.float32)
    native.call("quantize_a128", ptr(groups), groups.size, ptr(q), ptr(s))
    eq, es = a128(groups)
    assert np.array_equal(q, eq)
    comparison(s, es, exact=True)
    native.call("quantize_a128", ptr(groups), 127, ptr(q), ptr(s), expected=1)
    invalid = repack(np.zeros((1, G), np.int8), np.ones((1, 1), np.float32)).reshape(-1)
    invalid[0] |= 3
    native.call("dot128", ptr(invalid), ptr(q), C.byref(C.c_int32()), expected=1)
    zero_scale = repack(np.ones((1, G), np.int8), np.zeros((1, 1), np.float32)).reshape(-1)
    value = C.c_float()
    native.call("vec_dot", 43, ptr(zero_scale), ptr(groups[1]), G, C.byref(value))
    assert value.value == 0
    overflow_scale = repack(np.ones((1, G), np.int8),
                            np.full((1, 1), np.finfo(np.float32).max, np.float32)).reshape(-1)
    native.call("vec_dot", 43, ptr(overflow_scale), ptr(groups[1]), G, C.byref(value), expected=4)
    results += ["A128_zero_half_ties_tiny_extreme", "K_tail_rejected", "reserved_code_rejected",
                "zero_weight_scale", "finite_scale_output_overflow_rejected"]
    return results


def adapter_checks(native, rng):
    results = {}
    x = bf16(rng.normal(0, 0.75, 2 * G).astype(np.float32))
    q, _ = a128(x[None])
    denom = np.maximum(np.max(np.abs(x.reshape(-1, G)), axis=1), np.float32(1e-5))
    inverse = (np.float32(1) / denom) * np.float32(127)
    fake = bf16(q.reshape(-1, G).astype(np.float32) / inverse[:, None]).reshape(-1)
    for typ in [43, 44, 45, 46]:
        w = rng.integers(-1, 2, x.size, dtype=np.int8) if typ in [43, 44] else rng.integers(
            -8 if typ == 45 else -127, 8 if typ == 45 else 128, x.size, dtype=np.int8)
        scale = np.array([[0.03, 0.07]], np.float32) if typ != 46 else np.array([[31.0, 71.0]], np.float32)
        if typ in [43, 44]:
            packed = repack(w.reshape(1, -1), scale).reshape(-1)
        else:
            nbytes = 68 if typ == 45 else 132
            blocks = np.empty((2, nbytes), np.uint8)
            if typ == 45:
                nibble = (w.reshape(2, G).astype(np.uint8) & 15).reshape(2, 64, 2)
                blocks[:, :64] = nibble[:, :, 0] | (nibble[:, :, 1] << 4)
            else:
                blocks[:, :128] = w.reshape(2, G).view(np.uint8)
            blocks[:, -4:] = scale.reshape(-1).view(np.uint8).reshape(2, 4)
            packed = blocks.reshape(-1)
        restored = w.astype(np.float32).reshape(-1, G)
        restored = restored / scale.reshape(-1, 1) if typ == 46 else restored * scale.reshape(-1, 1)
        if typ != 43:
            restored = bf16(restored)
        restored = restored.reshape(-1)
        out = np.empty_like(x)
        native.call("dequantize", typ, ptr(packed), x.size, ptr(out))
        comparison(out, restored, exact=True)
        dot = C.c_float()
        native.call("vec_dot", typ, ptr(packed), ptr(x), x.size, C.byref(dot))
        if typ == 43:
            expected = projection(q.reshape(1, -1), a128(x[None])[1], w.reshape(1, -1), scale)[0, 0]
        else:
            operands = bf16(x) if typ == 46 else fake
            value = np.float32(0)
            for a, b in zip(operands, restored):
                value = np.float32(value + np.float32(a * b))
            expected = bf16(np.array([value]))[0]
        assert dot.value == float(expected), (typ, dot.value, expected)
        if typ != 43:
            prepared = np.empty_like(x)
            native.call("prepare_row", typ, ptr(x), x.size, ptr(prepared))
            comparison(prepared, operands, exact=True)
            fast = C.c_float()
            native.call("prepared_dot", typ, ptr(packed), ptr(prepared), x.size, C.byref(fast))
            assert fast.value == dot.value, (typ, fast.value, dot.value)
        results[str(typ)] = "dequant_and_dot_exact"
    return results


def invalid_checks(native, packed, x, ids, probabilities):
    cases = {}
    for name, field, value in [
        ("nonfinite_input", "x", np.nan), ("infinite_input", "x", np.inf),
        ("expert_upper_bound", "ids", len(packed[0])), ("expert_negative", "ids", -2),
        ("nonfinite_probability", "probabilities", np.nan),
        ("negative_probability", "probabilities", -0.1),
        ("probability_above_one", "probabilities", 1.1),
    ]:
        values = {"x": x.copy(), "ids": ids.copy(), "probabilities": probabilities.copy()}
        values[field].flat[0] = value
        native.execute(values["x"], values["ids"], values["probabilities"], expected=1)
        cases[name] = "rejected"
    mask = np.ones(ids.shape, np.uint8)
    mask.flat[0] = 2
    native.execute(x, ids, probabilities, mask, expected=1)
    cases["invalid_miss_byte"] = "rejected"
    native.execute(np.repeat(x, 18, axis=0), np.repeat(ids, 18, axis=0),
                   np.repeat(probabilities, 18, axis=0), expected=1)
    cases["token_capacity_exceeded"] = "rejected"
    no_tasks = np.full_like(ids, -1)
    out, trace = native.execute(x, no_tasks, probabilities)
    assert trace.active == 0 and not out.any()
    cases["all_inactive_ids"] = "zero_output"
    bad = Weights.from_buffer_copy(native.weights)
    unused = C.c_void_p()
    for name, mutate in [
        ("wrong_abi", lambda: setattr(bad, "abi_version", 2)),
        ("wrong_shape", lambda: setattr(bad, "hidden", 128)),
        ("wrong_byte_length", lambda: setattr(bad, "gate_bytes", bad.gate_bytes - 1)),
    ]:
        bad = Weights.from_buffer_copy(native.weights)
        mutate()
        native.call("executor_create", native.pool, C.byref(bad), 1, 8, 64 << 20, C.byref(unused), expected=1)
        assert not unused.value
        cases[name] = "rejected"
    native.call("executor_create", native.pool, C.byref(native.weights), 17, 8, 1, C.byref(unused), expected=1)
    cases["workspace_budget"] = "rejected"
    bad_gate = packed[0].copy()
    for name in ["reserved_weight_code", "nonfinite_weight_scale", "negative_weight_scale"]:
        bad_gate = packed[0].copy()
        if "scale" in name:
            bad_gate.reshape(-1)[32:36] = np.array([np.nan if "nonfinite" in name else -1], np.float32).view(np.uint8)
        else:
            bad_gate.reshape(-1)[0] = 3
        bad = Weights.from_buffer_copy(native.weights)
        bad.gate = ptr(bad_gate)
        native.call("executor_create", native.pool, C.byref(bad), 1, 8, 64 << 20, C.byref(unused), expected=1)
        cases[name] = "rejected"
    cpus = np.array(sorted(os.sched_getaffinity(0)), np.int32)
    native.call("pool_create", 9, ptr(cpus), len(cpus), C.byref(unused), expected=1)
    cases["more_than_eight_threads"] = "rejected"
    duplicates = np.array([cpus[0], cpus[0]], np.int32)
    native.call("pool_create", 2, ptr(duplicates), 2, C.byref(unused), expected=1)
    cases["duplicate_affinity_cpu"] = "rejected"
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "baseline-envs/s11-native/libs11-cpu.so")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/bitnet_toy_models/qwen36_w4attn_w158ffn_iter0002000")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/s11-native/20261012-rebuild/cpu-qualification.json")
    args = parser.parse_args()
    assert 1 <= len(os.sched_getaffinity(0)) <= 8, "Run with taskset limiting affinity to <=8 cores"
    rng = np.random.default_rng(20261012)
    sources, words, source_scales = [], [], []
    started = time.perf_counter()
    for layer, expert in [(0, 0), (0, 17), (13, 255)]:
        path = args.checkpoint / f"model-layer-{layer:03d}.safetensors"
        keys = [f"model.layers.{layer}.mlp.experts.{name}" for name in
                ("gate_up_proj", "gate_up_proj_scale", "down_proj", "down_proj_scale")]
        with safe_open(path, framework="numpy") as file:
            arrays = [np.ascontiguousarray(file.get_slice(key)[expert:expert + 1]) for key in keys]
        sources.append({
            "layer": layer, "expert": expert, "file": str(path), "keys": keys,
            "slice_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
        })
        words.append((arrays[0], arrays[2]))
        source_scales.append((arrays[1], arrays[3]))
    load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    gu = np.concatenate([decode_source(a[0]) for a in words])
    down = np.concatenate([decode_source(a[1]) for a in words])
    gs = np.concatenate([a[0] for a in source_scales])
    ds = np.concatenate([a[1] for a in source_scales])
    codes = [np.ascontiguousarray(gu[:, :F]), np.ascontiguousarray(gu[:, F:]), down]
    scales = [np.ascontiguousarray(gs[:, :F]), np.ascontiguousarray(gs[:, F:]), ds]
    packed = [repack(w, s) for w, s in zip(codes, scales)]
    repack_ms = (time.perf_counter() - started) * 1000
    native = Native(args.library)
    result = {
        "schema_version": 1, "scope": "RN03-RN05 offline actual-expert CPU vertical prototype",
        "backend": "AVX2_maddubs_madd_group128_left_fp32_BF16_v1",
        "tolerances_frozen": {"FC1": "bit_exact", "A128": "bit_exact",
                             "chain_relative_l2": 0.002, "chain_max_abs_over_rms": 0.02},
        "source_lifetime": "Caller owns immutable packed buffers until executor destruction; no request weight IO.",
        "sources": sources, "source_load_ms": load_ms, "lossless_repack_ms": repack_ms,
        "layout": {"group": G, "W2_block_bytes": B, "codes_and_scale_bits_roundtrip": True},
        "affinity": sorted(os.sched_getaffinity(0)),
        "simd_provenance": {
            "path": "/home/v-zning/workspace/bitnet-30b-cpu-gpu/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c",
            "commit": "8fb65cf044e7c833ef4545a9fee3b72ab89706ea", "license": "MIT",
            "method": "unsigned W2 bytes * signed A8 maddubs + madd; new sequential layout and per-group subtraction",
        },
        "not_claimed": ["full-model integration", "task accuracy", "GPU equivalence", "full request latency"],
    }
    result["math_checks"] = math_checks(native, rng)
    result["adapter_checks"] = adapter_checks(native, rng)
    cases = []
    for tokens, slots, mode in [(1, 8, "all"), (5, 3, "tail"), (17, 8, "misses"),
                                (5, 8, "none"), (1, 8, "zero"), (3, 8, "probability_zero"),
                                (2, 8, "extreme_groups")]:
        x = bf16(rng.normal(0, 0.75, (tokens, H)).astype(np.float32))
        ids = rng.integers(0, len(packed[0]), (tokens, slots), dtype=np.int32)
        ids[:, :min(2, slots)] = 0
        probabilities = rng.random((tokens, slots), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        miss = None
        if mode == "misses":
            miss = rng.integers(0, 2, ids.shape, dtype=np.uint8)
            ids[0, 0] = -1
        elif mode == "none":
            miss = np.zeros(ids.shape, np.uint8)
        elif mode == "zero":
            x.fill(0)
        elif mode == "probability_zero":
            probabilities[:, :2] = 0
        elif mode == "extreme_groups":
            x[:, :128] = 0
            x[:, 128:256] *= np.float32(128)
            x[:, 256:384] *= np.float32(1e-6)
        expected = reference(x, ids, probabilities, miss, codes, scales)
        cases.append((mode, x, ids, probabilities, miss, expected))
    result["executor"] = []
    for threads in sorted(set([1, min(8, len(os.sched_getaffinity(0)))])):
        before_threads = len(os.listdir("/proc/self/task"))
        started = time.perf_counter()
        native.create(packed, threads)
        prepare_ms = (time.perf_counter() - started) * 1000
        pool_tasks = sorted(int(t) for t in os.listdir("/proc/self/task"))
        assert len(pool_tasks) == before_threads + threads - 1
        for tid in pool_tasks:
            assert os.sched_getaffinity(tid) <= os.sched_getaffinity(0)
        report = {"threads_including_caller": threads, "prepare_ms": prepare_ms, "cases": []}
        for mode, x, ids, probs, miss, expected in cases:
            out, trace = native.execute(x, ids, probs, miss)
            actual = output_f32(out)
            comparison_result = comparison(actual, expected[0])
            if trace.active:
                traced = [
                    trace_array(trace.fc1, (trace.active, 2 * F), np.float32),
                    trace_array(trace.middle, (trace.active, F), np.float32),
                    trace_array(trace.input_q, x.shape, np.int8),
                    trace_array(trace.input_s, (len(x), H // G), np.float32),
                    trace_array(trace.middle_q, (trace.active, F), np.int8),
                    trace_array(trace.middle_s, (trace.active, F // G), np.float32),
                ]
                comparison(traced[0], expected[1], exact=True)
                comparison(traced[1], expected[2])
                assert np.array_equal(traced[2], expected[3])
                comparison(traced[3], expected[4], exact=True)
                assert np.array_equal(traced[4], expected[5])
                comparison(traced[5], expected[6], exact=True)
            times = []
            for _ in range(11):
                t0 = time.perf_counter()
                repeated, _ = native.execute(x, ids, probs, miss)
                times.append((time.perf_counter() - t0) * 1000)
                assert np.array_equal(repeated, out)
            assert len(os.listdir("/proc/self/task")) == len(pool_tasks), "Execute created threads"
            report["cases"].append({
                "name": mode, "tokens": len(x), "slots": ids.shape[1], "active": trace.active,
                "numerics": comparison_result, "repeat_bit_exact": True,
                "native_call_ms_p50": statistics.median(times),
                "native_call_ms_min": min(times), "native_call_ms_all": times,
            })
        mode, x, ids, probs, miss, expected = cases[0]
        if threads == 1:
            result["invalid_checks"] = invalid_checks(native, packed, x, ids, probs)
        stats = Stats()
        assert native.lib.s11_cpu_executor_stats(native.executor, C.byref(stats)) == 0
        report["stats"] = {name: getattr(stats, name) for name, _ in Stats._fields_}
        native.close()
        # /proc task removal can trail pthread_join's clear-child-tid wakeup.
        deadline = time.monotonic() + 0.1
        while len(os.listdir("/proc/self/task")) != before_threads and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(os.listdir("/proc/self/task")) == before_threads, (
            "Pool workers did not exit", before_threads, os.listdir("/proc/self/task"))
        result["executor"].append(report)
    result["native_cpp"] = native_smoke(args.library, packed)
    result["native_cpp"]["cold_method"] = "clflush all source cache lines then mfence, outside timed interval; workspace remains warm"
    result["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["RN03"] = "passed_targeted"
    result["RN04"] = "passed_targeted"
    result["RN05"] = "passed_actual_expert_vertical_prototype"
    result["performance_caveat"] = "Synchronous CPU call includes validation, scatter and ctypes; no H2D/D2H or model-level timing."
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "qualification": str(args.output),
                      "peak_rss_mib": result["peak_rss_mib"],
                      "native_call_ms_p50": {str(r["threads_including_caller"]): {
                          c["name"]: round(c["native_call_ms_p50"], 4) for c in r["cases"]
                      } for r in result["executor"]}}))


if __name__ == "__main__":
    main()
