# S11-Native

This private fork starts at the frozen B1 llama.cpp revision
`6184e92c57dcd34de8a3e381d7641a5e75250d5f`. It uses only the existing S11
checkpoint, not a new quantization of the official model. Historical Q routes
are not runtime dependencies.

## Format and arithmetic

`convert_s11.py` exports source codes/scales losslessly and verifies every
source tensor through inverse layout mapping. The native types are:

| Type | ID | K128 block | Arithmetic |
| --- | ---: | ---: | --- |
| S11_W2 | 43 | 32 code bytes + FP32 scale | Integer routed A128 |
| S11_W2_BF16 | 44 | 32 code bytes + FP32 scale | Shared BF16-restored operands |
| S11_W4 | 45 | 64 code bytes + FP32 scale | Signed W4 BF16-restored operands |
| S11_I8 | 46 | 128 code bytes + FP32 reciprocal | Embedding/head division, no A128 |

W2 codes represent weight + 1, with four adjacent K values per byte,
low bits first. Original source scales remain FP32. No requantization or
float-to-quant generic callback is permitted. Raw norm gamma, A_log and
dt_bias remain unchanged in the container. The model graph applies the
required zero-centered gamma offset, -exp(A_log), and protected-value casts.

The routed chain preserves integer K128 partials, activation then weight
scale, left-order group accumulation, BF16 boundaries, and probability
before FC2 activation quantization. The full CPU executor uses an AVX2
byte-dot implementation, a persistent pool and bounded workspace.

GPU protected/router and unfused attention products use a fixed-order FP32
kernel. Flash Attention is disabled for this initial backend: tested automatic
Prefill/Decode paths changed recurrent results beyond the declared tolerance.
Ordinary CPU llamafile GEMM is disabled for the same shape-consistency reason.
Neither restriction removes the S11 AVX2 expert executor. CUDA float reductions
are a declared native backend, not a bitwise claim against another framework.

HN04/HN05 kernels are available with `S11_CUDA_OPT=dp4a` (typed kernels,
integer K128 DP4A) or `S11_CUDA_OPT=tiled` (also four-column floating projection
weight-restoration reuse). The backend default remains `reference`; the
qualified gpu-cache launcher selects `tiled`. Invalid values are rejected. These kernels keep
the source scales, BF16 boundaries and floating reduction order; they do not
enable Tensor Cores, Flash Attention or a different quantization.

`S11_HEAD_LAYOUT=bf16` is the qualified HN06 physical CUDA head layout.
It keeps logical S11_I8 identity and CPU canonical
codes/scales, replaces GPU packed storage with exactly restored BF16 values,
and uses bounded loading tiles. Backend default `packed` behavior is unchanged;
the gpu-cache launcher now selects `bf16` after canonical roundtrip, state and
four-load request qualification. Derived heads are immutable and do not support
weight views, get_rows or memset; those requests fail explicitly.
`S11_EMBEDDING=cpu` with a CPU weight override requests CPU get_rows.
It was qualified but did not reduce observed lifecycle GPU peak and regressed
development latency, so the launcher retains GPU placement.

## Build

The existing workspace tools can build without installing dependencies:

```bash
root=/home/v-zning/workspace_qwen
cmake="$root/baseline-envs/b1/lib/python3.12/site-packages/cmake/data/bin/cmake"
taskset -c 0-7 "$cmake" -S "$root/s11-native" \
  -B "$root/baseline-envs/s11-native/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_BUILD_SERVER=ON
taskset -c 0-7 "$cmake" --build "$root/baseline-envs/s11-native/build" \
  --target llama-server -j4
```

## Run and roles

Use the resource guard; do not run a second GPU engine in parallel:

```bash
root=/home/v-zning/workspace_qwen
"$root/baseline-envs/f0/bin/python" "$root/baselines/nvidia-tools/run_guarded.py" \
  --output "$root/outputs/s11-native/new-run" \
  --gpu-mib 8192 --ram-gib 64 --gpu-idle-seconds 120 \
  --gpu-wait-timeout 1800 --timeout 14400 -- \
  bash "$root/s11-native/tools/s11-native/run.sh"
```

The launcher defaults to native GPU cache fill with a4.5 GiB target cache,
typed DP4A/four-column kernels and exact BF16 physical head.
The5 GiB/head combination passed the8 GiB hard gate but left only470 MiB
in the full formal campaign; it was rejected against the512 MiB engineering
reserve. The selected configuration peaked at7210 MiB (982 MiB remaining).
Hybrid and cpu modes retain their earlier CUDA reference/packed defaults;
CPU matmul now selects prepared. Hybrid cache stays3 GiB.
`S11_EXPERT_MODE=hybrid` uses native CPU misses for at most eight selected
assignments; larger batches fill the cache. This is a task-size rule, not
an assumption that M1 must be Decode. `S11_EXPERT_MODE=cpu` runs all routed
experts through the persistent native CPU executor, with the GPU trunk.
Cached GPU modes require an explicit cache and microbatches no larger than128.
All choices preserve original top8 slots and expose actual work in completion
responses. The cost-selected default and frozen performance evidence are
recorded in the run report; no CPU participation percentage is forced.

The container stays CPU-backed for routed weights. CPU/GPU views borrow the
source for the model lifetime. Cache eviction observes native copy/compute
dependencies; CPU mask calculation reads the current host slot map.
Single-context, single-slot serving is the initial supported configuration.

`measure_native.py` uses the same real-token HTTP/SSE timing for this native
server and B1. Empty-text tokens count; final events do not count as tokens.
`--request-timeout` defaults to1800 seconds; the provisional CPU-only formal
campaign uses7200 seconds because the conservative CPU trunk is slow.
`guard_cpu.py` enforces8-CPU affinity, group RSS64GiB and zero owned GPU
holders for the provisional CPU-only campaign. It records foreign GPU
activity and host load/steal without terminating unrelated work. This explicit
CPU-only isolation amendment does not weaken the GPU/B1 exclusive-run guard;
the CPU result is not a paired clean stability claim.

HN07/HN08 additions:
- `S11_ROUTE_SYNC=event` uses model-owned route-ready events for S11 cache
  callbacks. Qualification passed, but development latency did not improve;
  `backend` remains the default. Both wait the same current CUDA stream, so
  neither is a claim of barrier elimination or full-token Graph capture.
- `S11_CPU_MATMUL=prepared` prepares dynamic activations once per token and
  uses AVX2 weight restoration/products while retaining scalar K accumulation.
  Only loader-validated immutable2D W2_BF16/W4/I8 weights use this path.
  Checked standalone math APIs still validate inputs; unsupported shapes use
  the existing adapter. Numerical/state and the complete optimized CPU reference
  passed. The launcher selects prepared; `scalar` preserves the old adapter.
- `S11_CPU_THREADS=1..8` selects expert pool workers including the caller
  within inherited allowed affinity (at most8 CPUs). It no longer assumes
  physical CPU IDs0-7, although the fixed main experiment still pins0-7.
- `S11_EXPERT_MODE=gpu-resident` is an independent hardware experiment,
  not the8GiB launcher mode. Use an explicit16GiB guard, omit `--cpu-moe`
  and `--moe-cache-mib`, and qualify the same S11 model first. Every routed
  weight bank must actually be device-resident; host banks/cache are rejected.
  The corrected full-state16GiB qualification and four-load campaigns passed.
  No physical16GiB
  card or different hardware ISA has been validated.

HN07-HN12 completed evidence:
[report](../../../outputs/s11-native/20261013-hn07-hn12/REPORT.md).
The CPU optimization preserves all old CPU output IDs and improves observed
request medians3.25-3.75x against the historical scalar CPU control. The8GiB
native still loses to the same-campaign strong B1. Full routed-expert residency
under an independent16GiB guard eliminates route readbacks/uploads, but does
not deliver a stable all-load latency win or close the B1 gap.
`report_hardware.py OUTPUT` verifies100 formal samples, assets, work counts,
output identity and separate resource budgets.

Independent resident reproduction (never treat this as the8GiB launcher):

```bash
root=/home/v-zning/workspace_qwen
"$root/baseline-envs/f0/bin/python" "$root/baselines/nvidia-tools/run_guarded.py" \
  --output "$root/outputs/s11-native/resident-new-run" \
  --gpu-mib 16384 --ram-gib 64 --gpu-idle-seconds 120 --timeout 3600 -- \
  env S11_EXPERT_MODE=gpu-resident S11_CUDA_OPT=tiled S11_HEAD_LAYOUT=bf16 \
  taskset -c 0-7 "$root/baseline-envs/s11-native/build/bin/llama-server" \
  -m "$root/baseline-models/s11-native/s11-native.gguf" \
  --host 127.0.0.1 --port 18141 -ngl 99 --fit off -t 8 -tb 8 \
  -c 4096 -b 512 -ub 128 -np 1 --no-warmup --poll 0 --poll-batch 0 \
  -ctk bf16 -ctv bf16 -fa off --backend-sampling
```
Use `--require-work` for S11 to verify `(input + output - 1) * 40 * 8`.
Native responses include CPU/GPU assignment and actual CUDA Graph replay
counts. Prefix reuse and speculative decoding are not part of the protocol.

## Qualification tools

- `check_cpu.py`: independent actual-source CPU expert and malformed-input tests.
- `check_gpu.cpp`: native type arithmetic and repeated-expert indexing.
- `check_expert_chain.cpp`: real S11 CPU/CUDA complete expert chains.
- `check_model.cpp`: all-vocabulary finite logits, reset/chunk/incremental state,
  and actual routed task counts; diagnostic boundary capture is optional.
- `qualify_native.py`: guarded CPU-only or outer-guarded GPU model/stream tests.

CPU-only diagnostics are not GPU or formal performance qualification.
Passing qualification does not establish language quality, physical 8 GiB
device performance, or superiority to B1. Preserve negative experiments and
resource failures; never replace the frozen strong B1 denominator.

HN00-HN06 evidence is in
[the report](../../../outputs/s11-native/20261013-hn00-hn06/REPORT.md).
`report_hn.py OUTPUT` recomputes the selected60 formal samples and verifies
assets, exact token/work counts, GPU output identity and resource gates.
The rejected5 GiB candidate's20 samples remain separate; no HN07+ work was run.
