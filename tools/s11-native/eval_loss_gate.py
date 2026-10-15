#!/usr/bin/env python3
"""Three-tier loss evaluation gate (L1 Logits KL/Cos, L2 Perplexity, L3 200-item deterministic tasks)."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

MAGIC = 0x53313147  # "S11G"
ROOT = Path("/home/v-zning/workspace_qwen")

CALIB_TEXTS = [
    (
        "In computer architecture, a Tensor Core performs matrix multiply-accumulate operations on small tiles "
        "such as 16x16x16 per clock cycle. While standard SIMT execution decodes individual scalar or 4-element "
        "vectors via DP4A instructions, grouped matrix multiplication amortizes weight loads across multiple "
        "tokens routed to the same expert in a Mixture-of-Experts transformer layer."
    ),
    (
        "混合专家模型（MoE）在每一层包含256个路由专家和1个共享专家。在单请求推理的Prefill阶段，"
        "长上下文输入包含上千个Token，不同Token被分配到相同的专家时，如果能够在GPU片上共享内存中将2比特或4比特压缩权重"
        "即时解包为INT8矩阵块并调用Tensor Core，就能够避免冗余的全局显存读取，显著降低首字延迟（TTFT）。"
    ),
    (
        "Let G be a finite group and H a normal subgroup of G. By Lagrange's theorem, the order of H divides "
        "the order of G, and the quotient group G/H has order |G|/|H|. Furthermore, if f: G -> G' is a group "
        "homomorphism with kernel K, the First Isomorphism Theorem states that G/K is isomorphic to the image of f."
    ),
    (
        "def dijkstra_shortest_path(graph, source):\n"
        "    import heapq\n"
        "    dist = {node: float('inf') for node in graph}\n"
        "    dist[source] = 0\n"
        "    pq = [(0, source)]\n"
        "    while pq:\n"
        "        d, u = heapq.heappop(pq)\n"
        "        if d > dist[u]:\n"
        "            continue\n"
        "        for v, weight in graph[u]:\n"
        "            if dist[u] + weight < dist[v]:\n"
        "                dist[v] = dist[u] + weight\n"
        "                heapq.heappush(pq, (dist[v], v))\n"
        "    return dist\n"
    ),
    (
        "template <int BLOCK_SIZE>\n"
        "__global__ void rms_norm_kernel(const float * __restrict__ x, const float * __restrict__ w,\n"
        "                                float * __restrict__ y, int cols, float eps) {\n"
        "    const int row = blockIdx.x;\n"
        "    float sum_sq = 0.0f;\n"
        "    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {\n"
        "        float val = x[row * cols + col];\n"
        "        sum_sq += val * val;\n"
        "    }\n"
        "    // Warp reduction for mean square\n"
        "}\n"
    ),
    (
        "在量化神经网络推理中，对称INT4权重量化通常将每128个连续通道分为一组（group_size=128），"
        "共享一个FP32缩放因子scale。如果激活值同样按128通道动态量化为INT8，则组内点积完全可以在INT32累加器中精确完成，"
        "随后再一次性乘上激活与权重的缩放因子之积，而无需将每个权重还原为BF16浮点数。"
    ),
    (
        "The second law of thermodynamics establishes the concept of entropy as a state function of a thermodynamic "
        "system. In any reversible Carnot cycle operating between two reservoirs at temperatures T_H and T_C, the "
        "maximum possible thermal efficiency is given by eta = 1 - T_C / T_H, independent of the working substance."
    ),
    (
        "Linear attention mechanisms such as Gated DeltaNet (GDN) replace the quadratic key-value cache of standard "
        "softmax self-attention with a fixed-size recurrent matrix state. During autoregressive decoding, each step "
        "updates the state in O(d^2) time, whereas chunkwise parallel forms allow efficient hardware utilization "
        "during prompt prefill."
    ),
    (
        "In distributed systems, the Raft consensus algorithm separates key elements of consensus into leader election, "
        "log replication, and safety. A leader accepts log entries from clients, replicates them across a majority "
        "quorum of followers, and commits an entry only when it has been safely stored on a majority of servers."
    ),
    (
        "Consider a continuous-time Markov chain with generator matrix Q. The Kolmogorov forward equations describe "
        "the time evolution of the transition probability matrix P(t) via dP(t)/dt = P(t)Q, with initial condition "
        "P(0) = I. When the chain is irreducible and positive recurrent, it converges to a unique stationary distribution."
    ),
    (
        "现代编译器的寄存器分配通常将SSA（静态单赋值）形式的中间表示转换为干渉图（Interference Graph），"
        "并通过图着色算法或线性扫描算法将无限虚拟寄存器映射到有限的物理寄存器集合。当寄存器压力过大时，"
        "编译器需要插入Spill指令将变量暂存至栈帧或共享内存。"
    ),
    (
        "In numerical linear algebra, the singular value decomposition (SVD) factors any m-by-n real or complex matrix "
        "A into U * Sigma * V^T, where U and V are orthogonal (or unitary) matrices and Sigma is a non-negative "
        "diagonal matrix of singular values ordered from largest to smallest. The Eckart-Young theorem proves that "
        "truncating the SVD to the top k singular values yields the best rank-k approximation in both Frobenius and spectral norms."
    ),
    (
        "The Kullback-Leibler (KL) divergence D_KL(P || Q) measures how one probability distribution P diverges from "
        "a second candidate distribution Q. By Gibbs' inequality, D_KL(P || Q) is always non-negative and equals zero "
        "if and only if P and Q are identical almost everywhere. In language model quantization, keeping mean token-wise "
        "KL divergence below 1e-3 ensures virtually indistinguishable generation behavior."
    ),
    (
        "操作系统内核通过虚拟内存与页表机制为每个进程提供独立的地址空间。当CPU访问未映射或不在物理内存中的虚拟页时，"
        "内存管理单元（MMU）触发缺页异常（Page Fault），由内核分配物理页框并从磁盘或交换区加载数据；"
        "在GPU异构推理中，专家权重的LRU缓存管理与之高度相似。"
    ),
    (
        "A B-tree of order m is a self-balancing search tree in which every internal node has at most m children and "
        "at least ceil(m/2) children, and all leaf nodes appear at the same depth. Because node sizes are tuned to match "
        "storage block or cache line sizes, B-trees minimize the number of disk I/O operations required for range queries."
    ),
    (
        "When optimizing CUDA kernels for Ampere architecture (SM80), the lop3.b32 instruction can evaluate any arbitrary "
        "three-input bitwise boolean truth table in a single cycle using an 8-bit immediate lookup table (immLut). "
        "Combined with SIMD byte subtraction (__vsubss4), packed 2-bit ternary weights can be unpacked into signed 8-bit "
        "integers in registers without branch divergence or shared memory bank conflicts."
    ),
]


def format_mc_prompt(question: str, choices: list[str]) -> str:
    lines = [
        "Question: " + question,
        *(f"{chr(65 + idx)}. {c}" for idx, c in enumerate(choices)),
        "Answer with a single letter (A, B, C, or D): ",
    ]
    return "\n".join(lines)


def build_200_tasks() -> list[dict]:
    mmlu_path = ROOT / "outputs/baselines/20261009-abc/mmlu-pilot30.json"
    items = []
    if mmlu_path.is_file():
        raw = json.loads(mmlu_path.read_text())["items"]
        for it in raw:
            items.append({
                "prompt": format_mc_prompt(it["question"], it["choices"]),
                "answer": int(it["answer"]),
            })

    # Generate 170 deterministic STEM / math / logic / code-trace / linear algebra problems
    # Category 1: Linear equations & arithmetic word problems (50 items)
    for i in range(50):
        a = 13 + (i * 7) % 29
        b = 19 + (i * 11) % 37
        c = 5 + (i % 9)
        ans_val = a * c + b
        ans_idx = (i * 3 + 1) % 4
        offsets = [-12, -4, 6, 14]
        choices = []
        off_cursor = 0
        for j in range(4):
            if j == ans_idx:
                choices.append(str(ans_val))
            else:
                choices.append(str(ans_val + offsets[off_cursor]))
                off_cursor += 1
        q = (
            f"If a sequence of operations multiplies {a} by {c} and then adds {b}, "
            f"what is the exact resulting integer?"
        )
        items.append({"prompt": format_mc_prompt(q, choices), "answer": ans_idx})

    # Category 2: Modular arithmetic, number theory & combinatorics (40 items)
    for i in range(40):
        n = 6 + (i % 10)
        k = 2 + (i % 3)
        # Compute n * (n - 1) // 2 or modular power
        if i % 2 == 0:
            val = (n * (n - 1)) // 2
            q = f"How many distinct pairs of vertices can be chosen from a complete graph with {n} vertices?"
        else:
            base = 3 + (i % 5)
            mod = 11 + 2 * (i % 4)
            val = pow(base, k + 2, mod)
            q = f"What is the remainder when {base}^{k + 2} is divided by {mod}?"
        ans_idx = (i * 5 + 2) % 4
        deltas = [1, 2, 3] if val < 5 else [-2, 1, 3]
        choices = []
        d_idx = 0
        for j in range(4):
            if j == ans_idx:
                choices.append(str(val))
            else:
                choices.append(str(val + deltas[d_idx]))
                d_idx += 1
        items.append({"prompt": format_mc_prompt(q, choices), "answer": ans_idx})

    # Category 3: Python code execution trace (40 items)
    for i in range(40):
        init_x = 2 + (i % 7)
        step = 3 + (i % 5)
        loops = 3 + (i % 4)
        final_x = init_x
        for r in range(loops):
            final_x = final_x + step * (r + 1)
        ans_idx = (i + 3) % 4
        deltas = [-step, step, 2 * step]
        choices = []
        d_idx = 0
        for j in range(4):
            if j == ans_idx:
                choices.append(str(final_x))
            else:
                choices.append(str(final_x + deltas[d_idx]))
                d_idx += 1
        q = (
            f"What is the final value of `x` after running the following Python snippet?\n"
            f"x = {init_x}\n"
            f"for r in range(1, {loops + 1}):\n"
            f"    x += {step} * r"
        )
        items.append({"prompt": format_mc_prompt(q, choices), "answer": ans_idx})

    # Category 4: Computer Science, Systems & ML architecture concepts (40 items)
    cs_bank = [
        (
            "What is the asymptotic time complexity of searching for a key in a balanced binary search tree with N nodes?",
            ["O(1)", "O(log N)", "O(N)", "O(N log N)"],
            1,
        ),
        (
            "In IEEE 754 floating-point representations, how many exponent bits does BFloat16 (BF16) have?",
            ["5 bits", "7 bits", "8 bits", "10 bits"],
            2,
        ),
        (
            "In a 32-thread CUDA warp, which intrinsic performs a 4-way signed 8-bit integer dot product with 32-bit accumulation?",
            ["__fadd_rn", "__dp4a", "__shfl_xor_sync", "__hmma_m16n8k16"],
            1,
        ),
        (
            "Which sorting algorithm guarantees O(N log N) worst-case time complexity and is stable?",
            ["QuickSort", "MergeSort", "HeapSort", "SelectionSort"],
            1,
        ),
        (
            "In HTTP/1.1, which status code indicates that the server successfully processed the request and is returning content?",
            ["200 OK", "301 Moved Permanently", "404 Not Found", "503 Service Unavailable"],
            0,
        ),
        (
            "In linear algebra, the trace of a square matrix is equal to the sum of its:",
            ["Determinants", "Eigenvalues", "Singular vectors", "Off-diagonal entries"],
            1,
        ),
        (
            "How many bytes are required to store 128 ternary weights packed at 2 bits per weight, plus one 32-bit float scale?",
            ["20 bytes", "32 bytes", "36 bytes", "68 bytes"],
            2,
        ),
        (
            "How many bytes are required to store 128 signed 4-bit weights packed at 2 weights per byte, plus one 32-bit float scale?",
            ["36 bytes", "64 bytes", "68 bytes", "132 bytes"],
            2,
        ),
    ]
    for i in range(40):
        base_q, base_choices, base_ans = cs_bank[i % len(cs_bank)]
        # Rotate choices deterministically by i % 4
        rot = i % 4
        rotated_choices = [base_choices[(j - rot) % 4] for j in range(4)]
        rotated_ans = (base_ans + rot) % 4
        q_text = f"[Case #{i + 1}] {base_q}"
        items.append({"prompt": format_mc_prompt(q_text, rotated_choices), "answer": rotated_ans})

    return items[:200]


def write_input_bin(path: Path) -> None:
    formal_tokens = json.loads(
        (ROOT / "outputs/s11-native/20261013-hn07-hn12/formal-prompt-tokens.json").read_text()
    )
    dev_tokens = json.loads(
        (ROOT / "outputs/s11-native/20261013-hn07-hn12/dev-prompt-tokens.json").read_text()
    )
    anchors = [
        formal_tokens[:128],
        dev_tokens[:128],
        formal_tokens[:512],
        dev_tokens[:512],
    ]
    tasks = build_200_tasks()
    with path.open("wb") as f:
        f.write(struct.pack("<II", MAGIC, len(anchors)))
        for seq in anchors:
            f.write(struct.pack("<I", len(seq)))
            f.write(struct.pack(f"<{len(seq)}i", *seq))
        f.write(struct.pack("<I", len(CALIB_TEXTS)))
        for text in CALIB_TEXTS:
            b = text.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        f.write(struct.pack("<I", len(tasks)))
        for task in tasks:
            b = task["prompt"].encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
            f.write(struct.pack("<i", task["answer"]))


def build_binary(binary_path: Path) -> None:
    src = ROOT / "s11-native/tools/s11-native/eval_loss_gate.cpp"
    lib_dir = ROOT / "baseline-envs/s11-native/build/bin"
    inc_llama = ROOT / "s11-native/include"
    inc_ggml = ROOT / "s11-native/ggml/include"
    cmd = [
        "g++", "-std=c++17", "-O3",
        str(src),
        "-I", str(inc_llama),
        "-I", str(inc_ggml),
        "-L", str(lib_dir),
        f"-Wl,-rpath,{lib_dir}",
        "-lllama", "-lggml", "-lggml-base", "-lggml-cpu", "-lggml-cuda",
        "-o", str(binary_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dump", "compare"], required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--gpu-mib", type=float, default=8192)
    parser.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE env vars")
    args = parser.parse_args()

    out_dir = args.output.resolve()
    work_dir = out_dir.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    input_bin = work_dir / "loss_gate_input.bin"
    if not input_bin.is_file():
        write_input_bin(input_bin)

    binary = ROOT / "baseline-envs/s11-native/eval-loss-gate"
    build_binary(binary)

    env_args = [
        "S11_EXPERT_MODE=gpu-cache",
        "S11_CUDA_OPT=tiled",
        "S11_HEAD_LAYOUT=bf16",
        "S11_EMBEDDING=gpu",
        *args.env,
    ]
    python_bin = ROOT / "baseline-envs/f0/bin/python"
    guard_script = ROOT / "baselines/nvidia-tools/run_guarded.py"
    model_gguf = ROOT / "baseline-models/s11-native/s11-native.gguf"

    cmd = [
        str(python_bin), str(guard_script),
        "--output", str(out_dir),
        "--gpu-mib", str(args.gpu_mib),
        "--ram-gib", "64",
        "--gpu-idle-seconds", "5",
        "--gpu-wait-timeout", "600",
        "--timeout", "1800",
        "--",
        "env", *env_args,
        "taskset", "-c", "8-15",
        str(binary),
        str(model_gguf),
        str(input_bin),
        str(args.oracle.resolve()),
        args.mode,
        str(args.max_tasks),
    ]
    subprocess.run(cmd, check=True)
    log_text = (out_dir / "command.log").read_text()
    json_lines = [line for line in log_text.splitlines() if line.startswith("{") and line.endswith("}")]
    if not json_lines:
        raise RuntimeError(f"No JSON output found in {out_dir / 'command.log'}")
    metrics = json.loads(json_lines[-1])
    guard_summary = json.loads((out_dir / "summary.json").read_text())
    metrics["gpu_peak_mib"] = guard_summary["gpu_peak_mib"]
    metrics["rss_peak_gib"] = guard_summary["rss_peak_gib"]
    (out_dir / "loss_gate_result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
