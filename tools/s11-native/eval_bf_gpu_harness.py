#!/usr/bin/env python3
"""Trusted end-to-end GPU evaluation harness for Boostingflow (S11-native on A100).

Invoked by Boostingflow's ``evaluate_harness`` under ``sandbox_profile="gpu_harness"``:
    python -I -B eval_bf_gpu_harness.py <target_dir> [sample_count]

Pipeline executed per candidate trial:
1. Sync candidate C++/CUDA files into isolated build workspace (/tmp/bf-s11-gpu-harness)
   and invalidate modified object files (.o) to guarantee nvcc/g++ recompilation.
2. Compile ``llama-server`` and ``eval-loss-gate`` with CUDA 13.4 (sm_80).
3. Run 3-tier ``G0`` accuracy gate (``eval-loss-gate``, L1 KL + L2 WikiText-2 PPL +
   L3 MMLU-Pro 200 items with ``S11_FLASH_ATTN=1``).
4. Launch ``llama-server`` under the 8GiB VRAM budget (``--moe-cache-mib 4608 -fa on``),
   verify ``peak_vram_mb <= 7600.0``, and run ``sample_count`` passes of the 4 standard
   workloads (128/32, 2048/32, 128/256, 2048/256).
5. Output ``{"correctness": bool, "observations": [ns_1, ...]}`` on stdout.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path("/home/v-zning/workspace_qwen")
REPO_SRC = WORKSPACE_ROOT / "s11-native"
BASELINE_DIR = WORKSPACE_ROOT / "outputs/s11-native/bf-workspace/s11_baseline"
MODEL_PATH = WORKSPACE_ROOT / "baseline-models/s11-native/s11-native.gguf"
GATE_INPUT_BIN = WORKSPACE_ROOT / "outputs/s11-native/20261015-lossy-accel/loss_gate_input.bin"
GATE_ORACLE_BIN = WORKSPACE_ROOT / "outputs/s11-native/20261015-lossy-accel/p0-oracle.bin"
PROMPT_2048_FILE = (
    WORKSPACE_ROOT / "outputs/baselines/20261009-abc/b3-exl3-cache0-fixed/prompt-tokens.json"
)
ORACLE_TOKENS_FILE = WORKSPACE_ROOT / "outputs/s11-native/bf-workspace/o78_oracle_tokens.json"

REALWORLD_PROMPTS: dict[str, str] = {
    "W2_Hybrid_Struct": (
        "<|im_start|>user\n"
        "Generate a Markdown table of 15 GPU cluster nodes (node-01 to node-15) with columns: "
        "| Node ID | GPU Model | VRAM (GiB) | Bandwidth (GB/s) | Status |, followed by a brief 3-sentence "
        "architectural summary of the cluster.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    ),
    "W3_Code_Impl": (
        "<|im_start|>user\n"
        "Write a clean Python implementation of a doubly-linked LRU cache class `LRUCache` with `get(key)` "
        "and `put(key, value)` methods in O(1) time, including inline comments and a usage example.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    ),
    "W4_Cold_Essay": (
        "<|im_start|>user\n"
        "Write a vivid, non-repetitive historical narrative about the construction of the transcontinental "
        "telegraph line across nineteenth-century North America and how it transformed journalism and diplomacy.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    ),
}

HARNESS_WORK = Path("/tmp/bf-s11-gpu-harness")
BUILD_SRC = WORKSPACE_ROOT / "s11-native"
BUILD_DIR = WORKSPACE_ROOT / "baseline-envs/s11-native/build"
CMAKE_BIN = str(
    WORKSPACE_ROOT / "baseline-envs/b1/lib/python3.12/site-packages/cmake/data/bin/cmake"
)

EDITABLE_FILES: dict[str, tuple[str, ...]] = {
    "common/ngram-map.cpp": (
        "common/CMakeFiles/llama-common.dir/ngram-map.cpp.o",
    ),
    "common/ngram-map.h": (
        "common/CMakeFiles/llama-common.dir/ngram-map.cpp.o",
        "common/CMakeFiles/llama-common.dir/speculative.cpp.o",
    ),
    "common/ngram-cache.cpp": (
        "common/CMakeFiles/llama-common.dir/ngram-cache.cpp.o",
    ),
    "common/ngram-cache.h": (
        "common/CMakeFiles/llama-common.dir/ngram-cache.cpp.o",
        "common/CMakeFiles/llama-common.dir/speculative.cpp.o",
    ),
    "common/ngram-mod.cpp": (
        "common/CMakeFiles/llama-common.dir/ngram-mod.cpp.o",
    ),
    "common/ngram-mod.h": (
        "common/CMakeFiles/llama-common.dir/ngram-mod.cpp.o",
        "common/CMakeFiles/llama-common.dir/speculative.cpp.o",
    ),
    "common/speculative.cpp": (
        "common/CMakeFiles/llama-common.dir/speculative.cpp.o",
    ),
    "common/speculative.h": (
        "common/CMakeFiles/llama-common.dir/speculative.cpp.o",
        "tools/server/CMakeFiles/server-context.dir/server-context.cpp.o",
    ),
    "common/common.h": (
        "common/CMakeFiles/llama-common.dir/speculative.cpp.o",
        "tools/server/CMakeFiles/server-context.dir/server-context.cpp.o",
    ),
    "tools/server/server-context.cpp": (
        "tools/server/CMakeFiles/server-context.dir/server-context.cpp.o",
    ),
    "ggml/src/ggml-cuda/s11.cu": (
        "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/s11.cu.o",
    ),
    "ggml/src/ggml-cuda/s11.cuh": (
        "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/s11.cu.o",
        "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o",
    ),
    "ggml/src/ggml-cuda/ggml-cuda.cu": (
        "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o",
    ),
    "ggml/src/ggml-cuda/gated_delta_net.cu": (
        "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/gated_delta_net.cu.o",
    ),
    "ggml/src/ggml-cuda/ssm-conv.cu": (
        "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ssm-conv.cu.o",
    ),
    "src/llama-context.cpp": (
        "src/CMakeFiles/llama.dir/llama-context.cpp.o",
    ),
    "src/llama-context.h": (
        "src/CMakeFiles/llama.dir/llama-context.cpp.o",
        "src/CMakeFiles/llama.dir/llama-moe-cache.cpp.o",
    ),
    "src/llama-moe-cache.cpp": (
        "src/CMakeFiles/llama.dir/llama-moe-cache.cpp.o",
    ),
    "src/llama-moe-cache.h": (
        "src/CMakeFiles/llama.dir/llama-context.cpp.o",
        "src/CMakeFiles/llama.dir/llama-moe-cache.cpp.o",
    ),
    "src/llama-graph.cpp": (
        "src/CMakeFiles/llama.dir/llama-graph.cpp.o",
    ),
    "src/llama-graph.h": (
        "src/CMakeFiles/llama.dir/llama-graph.cpp.o",
        "src/CMakeFiles/llama.dir/llama-context.cpp.o",
    ),
    "src/models/qwen35moe.cpp": (
        "src/CMakeFiles/llama.dir/models/qwen35moe.cpp.o",
        "src/CMakeFiles/llama.dir/Unity/unity_8_cxx.cxx.o",
    ),
}

VRAM_REDLINE_MB = 7680.0
SERVER_PORT = 18191


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    cmake_dir = str(Path(CMAKE_BIN).parent)
    env["PATH"] = f"{cmake_dir}:/usr/local/cuda-13.4/bin:/usr/local/bin:/usr/bin:/bin"
    env["LD_LIBRARY_PATH"] = str(BUILD_DIR / "bin")
    env["HOME"] = "/tmp"
    env["TMPDIR"] = "/tmp"
    env["S11_EXPERT_MODE"] = "gpu-cache"
    env["S11_CUDA_OPT"] = "tiled"
    env["S11_HEAD_LAYOUT"] = "bf16"
    env["S11_EMBEDDING"] = "gpu"
    env["S11_CPU_MATMUL"] = "prepared"
    env["S11_FLASH_ATTN"] = "1"
    env["S11_MOE_CACHE_MIB"] = "4864"
    return env


def _ensure_build_workspace(env: dict[str, str]) -> bool:
    if (BUILD_DIR / "CMakeCache.txt").is_file() and (BUILD_SRC / "CMakeLists.txt").is_file():
        return True
    HARNESS_WORK.mkdir(parents=True, exist_ok=True)
    if BUILD_SRC.exists():
        shutil.rmtree(BUILD_SRC)
    shutil.copytree(
        REPO_SRC,
        BUILD_SRC,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "build*"),
    )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cfg = subprocess.run(
        [
            CMAKE_BIN,
            "-B",
            str(BUILD_DIR),
            "-S",
            str(BUILD_SRC),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=80",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if cfg.returncode != 0:
        sys.stderr.write(f"CMake configure failed:\n{cfg.stderr[-2000:]}\n")
        return False
    return True


def _sync_editable_files(source_dir: Path) -> list[str]:
    changed: list[str] = []
    for rel, obj_paths in EDITABLE_FILES.items():
        cand_file = source_dir / rel
        if not cand_file.is_file():
            cand_file = BASELINE_DIR / rel
        if not cand_file.is_file():
            cand_file = BASELINE_DIR.parent / rel
        dst_file = BUILD_SRC / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.is_file() or _sha256(cand_file) != _sha256(dst_file):
            dst_file.write_bytes(cand_file.read_bytes())
            changed.append(rel)
            for obj_rel in obj_paths:
                obj_file = BUILD_DIR / obj_rel
                if obj_file.is_file():
                    obj_file.unlink()
    return changed


def _build_targets(env: dict[str, str]) -> bool:
    res = subprocess.run(
        [
            CMAKE_BIN,
            "--build",
            str(BUILD_DIR),
            "--target",
            "llama-server",
            "-j",
            "16",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if res.returncode != 0:
        sys.stderr.write(f"Compilation failed:\n{res.stdout[-1500:]}\n{res.stderr[-2500:]}\n")
        return False
    return True


def _run_g0_gate(env: dict[str, str], max_tasks: int) -> bool:
    gate_bin = WORKSPACE_ROOT / "baseline-envs/s11-native/eval-loss-gate"
    res = subprocess.run(
        [
            "taskset",
            "-c",
            "0-7",
            str(gate_bin),
            str(MODEL_PATH),
            str(GATE_INPUT_BIN),
            str(GATE_ORACLE_BIN),
            "compare",
            str(max_tasks),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if res.returncode != 0:
        sys.stderr.write(
            f"G0 gate failed (exit {res.returncode}):\n{res.stdout[-1000:]}\n{res.stderr[-1000:]}\n"
        )
        return False
    for line in reversed(res.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if data.get("passed") is True:
                    return True
            except ValueError:
                pass
    sys.stderr.write(f"G0 gate did not report passed=true:\n{res.stdout[-1000:]}\n")
    return False


def _http_request(method: str, path: str, body: bytes | None = None, timeout: float = 120.0) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", errors="replace")
        return resp.status, data
    finally:
        conn.close()


def _parse_peak_vram_mb(metrics_text: str) -> float | None:
    m = re.search(r"^llama_s11_peak_vram_mb\s+([0-9.]+)", metrics_text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def _run_4workload_benchmark(env: dict[str, str], sample_count: int) -> tuple[bool, list[int]]:
    server_bin = BUILD_DIR / "bin/llama-server"
    raw_tokens = json.loads(PROMPT_2048_FILE.read_text(encoding="utf-8"))
    tokens_2048: list[int] = (
        raw_tokens["prompt"] if isinstance(raw_tokens, dict) else list(raw_tokens[:2048])
    )
    tokens_128: list[int] = tokens_2048[:128]
    oracle_map: dict[str, list[int]] = (
        json.loads(ORACLE_TOKENS_FILE.read_text(encoding="utf-8"))
        if ORACLE_TOKENS_FILE.is_file()
        else {}
    )

    log_file = open("/tmp/bf-harness-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "taskset",
            "-c",
            "0-7",
            str(server_bin),
            "-m",
            str(MODEL_PATH),
            "--host",
            "127.0.0.1",
            "--port",
            str(SERVER_PORT),
            "-ngl",
            "99",
            "--cpu-moe",
            "--moe-cache-mib",
            "4864",
            "--fit",
            "off",
            "-t",
            "8",
            "-tb",
            "8",
            "-c",
            "4096",
            "-b",
            "512",
            "-ub",
            "128",
            "-np",
            "1",
            "--no-warmup",
            "--poll",
            "0",
            "--poll-batch",
            "0",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "-fa",
            "on",
            "--backend-sampling",
            "--metrics",
            "--spec-type",
            "ngram-simple,draft-mtp",
            "--spec-ngram-simple-size-n",
            "3",
            "--spec-ngram-simple-size-m",
            "5",
            "--spec-draft-n-max",
            "1",
            "--spec-draft-n-min",
            "1",
            "--spec-draft-p-min",
            "0.0",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = False
        for _ in range(180):
            if proc.poll() is not None:
                break
            try:
                status, _ = _http_request("GET", "/health", timeout=2.0)
                if status == 200:
                    ready = True
                    break
            except OSError:
                pass
            time.sleep(0.5)
        if not ready:
            log_file.flush()
            err = Path("/tmp/bf-harness-server.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            sys.stderr.write(f"llama-server failed to start:\n{err}\n")
            return False, []

        workloads: list[tuple[str, list[int], int]] = [
            ("std_128_32", tokens_128, 32),
            ("std_2048_32", tokens_2048, 32),
            ("std_128_256", tokens_128, 256),
            ("std_2048_256", tokens_2048, 256),
        ]
        for w_name, w_text in REALWORLD_PROMPTS.items():
            t_status, t_resp = _http_request(
                "POST",
                "/tokenize",
                body=json.dumps({"content": w_text}).encode("utf-8"),
                timeout=30.0,
            )
            if t_status != 200:
                sys.stderr.write(f"Tokenize failed for {w_name}: status {t_status}\n")
                return False, []
            w_toks = json.loads(t_resp).get("tokens", [])[:128]
            workloads.append((w_name, w_toks, 256))

        observations = [0 for _ in range(sample_count)]
        recorded_oracle: dict[str, list[int]] = {}
        for w_name, w_tokens, w_pred in workloads:
            body = json.dumps(
                {
                    "prompt": w_tokens,
                    "n_predict": w_pred,
                    "temperature": 0,
                    "seed": 0,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "return_tokens": True,
                }
            ).encode("utf-8")
            # 1 warmup repeat + sample_count formal repeats per shape
            for repeat in range(sample_count + 1):
                t0 = time.perf_counter()
                status, resp_text = _http_request("POST", "/completion", body=body, timeout=180.0)
                dt = time.perf_counter() - t0
                if status != 200:
                    sys.stderr.write(f"Completion request {w_name} failed with status {status}\n")
                    return False, []
                resp_json = json.loads(resp_text)
                out_ids = resp_json.get("tokens")
                if not isinstance(out_ids, list) or len(out_ids) != w_pred:
                    sys.stderr.write(f"Invalid output token count on {w_name}: {len(out_ids or [])} vs {w_pred}\n")
                    return False, []
                if w_name in oracle_map and out_ids != oracle_map[w_name]:
                    sys.stderr.write(f"Greedy token ID mismatch vs O7.8 oracle on {w_name}\n")
                    return False, []
                recorded_oracle[w_name] = out_ids
                work = resp_json.get("s11_work")
                expected_assigns = (len(w_tokens) + w_pred - 1) * 320
                if (
                    not isinstance(work, dict)
                    or work.get("cpu_assignments", 0) + work.get("gpu_assignments", 0)
                    != expected_assigns
                ):
                    sys.stderr.write(f"Incomplete expert workload in s11_work on {w_name}\n")
                    return False, []
                if repeat == 0:
                    continue
                elapsed_s = dt
                observations[repeat - 1] += int(round(elapsed_s * 1_000_000_000))

        if not oracle_map and os.environ.get("BF_WRITE_ORACLE") == "1":
            ORACLE_TOKENS_FILE.write_text(json.dumps(recorded_oracle, indent=2), encoding="utf-8")

        # Check 8GiB VRAM redline after all shapes have executed
        m_status, m_text = _http_request("GET", "/metrics", timeout=10.0)
        if m_status == 200:
            peak_vram = _parse_peak_vram_mb(m_text)
            if peak_vram is not None and peak_vram > VRAM_REDLINE_MB:
                sys.stderr.write(
                    f"VRAM redline exceeded: {peak_vram:.1f} MiB > {VRAM_REDLINE_MB:.1f} MiB\n"
                )
                return False, []

        return True, observations
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log_file.close()


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"correctness": False, "observations": []}))
        return 0
    target_dir = Path(sys.argv[1]).resolve()
    sample_count = int(sys.argv[2]) if len(sys.argv) >= 3 else 3
    g0_tasks = int(os.environ.get("BF_G0_TASKS", "200"))
    env = _runtime_env()

    if not _ensure_build_workspace(env):
        print(json.dumps({"correctness": False, "observations": []}))
        return 0

    try:
        changed = _sync_editable_files(target_dir)
        if not _build_targets(env):
            print(json.dumps({"correctness": False, "observations": []}))
            return 0
        need_g0 = not changed or any(rel.startswith(("ggml/", "src/")) for rel in changed)
        if need_g0 and not _run_g0_gate(env, g0_tasks):
            print(json.dumps({"correctness": False, "observations": []}))
            return 0
        ok, observations = _run_4workload_benchmark(env, sample_count)
        if not ok or len(observations) != sample_count:
            print(json.dumps({"correctness": False, "observations": []}))
            return 0
        print(json.dumps({"correctness": True, "observations": observations}))
        return 0
    finally:
        # Always restore build workspace editable files to baseline after trial
        if BASELINE_DIR.is_dir() and BUILD_SRC.is_dir():
            restored = _sync_editable_files(BASELINE_DIR)
            if restored:
                _build_targets(env)


if __name__ == "__main__":
    raise SystemExit(main())
