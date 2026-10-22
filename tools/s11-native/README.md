# S11-Native

This private fork starts at the frozen B1 llama.cpp revision
`6184e92c57dcd34de8a3e381d7641a5e75250d5f`. It uses only the existing S11
checkpoint, not a new quantization of the official model. Historical Q routes
are not runtime dependencies.

## Format and arithmetic

CPU and CUDA ARGMAX use the lowest token index for equal maxima, matching
the CPU greedy sampler on token-ID-ordered inputs. CPU sampling sorts ties
by token ID; backend top-k sampling uses ordered ARGSORT instead of
unordered TOP_K. This matters for ties introduced by BF16 head rounding.
O11 uses a separately authorized deterministic oracle, retaining O10 data.
Graph on/off timing alone does not establish a graph correctness defect.

`S11_HEAD_LAYOUT=mixed` keeps the first 81,920 output-head rows in BF16 and
the remaining rows in canonical I8 storage. It requires `k=2048` and
`S11_CUDA_OPT=tiled`. Target scoring still covers the complete vocabulary;
only the physical storage changes. It saves 314.84 MiB relative to the
full BF16 head, without training or requantization. I8 restoration falls
back to correctly rounded division near BF16 rounding boundaries.
`S11_HEAD_CHECK=1` checks restored rows bitwise during derived-head loading.
The complete 508,559,360-element restoration check and the mixed-layout G0
results are recorded in the O11 experiment directory.
The canonical mixed-layout tail is uploaded once, after restoring the prefix.

`S11_HEAD_MMA=1` uses BF16 Tensor Cores for 4-8-row verification heads,
only with the full `bf16` head layout. Single-row prefill and two-row MTP
heads retain the original reduction. It changes accumulation order, so
qualification includes full-vocabulary logit comparisons and target token
IDs, not just throughput. `S11_HEAD_MMA_DUMP` is a one-shot diagnostic
output path and requires CUDA graphs to be disabled.

For single-sequence cached inference, `S11_MOE_PROMPT_ADMISSION=1` releases
previous decode pins when the first prompt chunk changes. Resident weights,
LRU contents and in-flight epoch guards are retained; repeated prompts keep
their pins. It is experimental and off by default: final holdout testing
found a short-prefill regression. This is cache admission, not KV-prefix
reuse or a trained router.
`S11_ROUTE_TRACE_MISSES=1` adds miss IDs to logs when `S11_ROUTE_TRACE` is set;
route diagnostics are not used as performance measurements.

`S11_GREEDY_FAST=1` removes a proven maximum-preserving sampler suffix for
zero-temperature generation without requested probabilities. Logit bias,
penalties and DRY remain; unusual sampler ordering retains the original
chain. Temperature-zero compaction is kept so backend sampling exports one
logit instead of copying the whole vocabulary to the host. Wide CUDA
ARGMAX uses deterministic two-stage reduction. The fast chain is qualified
for the 8GiB recipe; the 16GiB recipe retains its original chain because
one short-prefill workload regressed.

`S11_SCHED_REUSE=1` retains capacity-compatible backend schedulers and compute
buffers when samplers are rebound. It still synchronizes, invalidates cached
graphs and reserves fresh graphs; sampler parameters and probability-output
requests are not reused. A changed graph-node capacity recreates the scheduler.
This avoids repeated pinned-host allocation on the short-prefill path. It is
off by default in the library and enabled by the qualified 16GiB launcher
profile only; the 8GiB profile retains scheduler recreation.

### O11 launch profiles

The launcher defaults to the qualified 8GiB cached layout and accepts
`S11_EXPERT_MODE=gpu-resident` for the 16GiB resident layout. Explicit
environment overrides still take precedence. Flash Attention accepts
`1`/`on` and `0`/`off`. Speculation remains opt-in:

```bash
S11_EXPERT_MODE=gpu-cache bash tools/s11-native/run.sh \
  --spec-type ngram-simple,draft-mtp \
  --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 5 \
  --spec-draft-n-max 1 --spec-draft-n-min 1 --spec-draft-p-min 0.0
```

Use `gpu-resident` instead for 16GiB. The released draft vocabulary cap is
98,304 in both budgets. Earlier resident combinations missed the final
short-prefill gate even with the original 81,920 cap; the final fix reuses
the scheduler rather than attributing the regression solely to vocabulary
size. The larger cap was not readmitted after this fix.
Target scoring always uses all 248,320 rows.
Resource claims apply to the measured single-sequence configurations, not
arbitrary context, parallelism or environment overrides. See the workspace
[O11 results](../../../REPORT.md#16-o11固定预算-prefill--decode-深度优化).

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

## CB CPU baseline (CB00-CB12 completed)

The isolated build `baseline-envs/s11-native-cpu/build` uses `GGML_CUDA=OFF`,
native AVX2, `GGML_OPENMP=OFF` and `GGML_LLAMAFILE=OFF`.
`run_cpu.sh` pins0-7, disables visible GPU devices, selects native CPU experts,
`S11_CPU_MATMUL=tiled`, direct expert layout and packed I8 head. The original
GPU launcher/build and B1 denominator are not replaced. Expert layout
`reference` means the existing direct packed AVX2 executor, not a Python or
full-float model fallback. Run the CPU launcher
under `guard_cpu.py`, not unmonitored:

```bash
root=/home/v-zning/workspace_qwen
"$root/vllm-bitnet-cpu-gpu/.venv/bin/python" \
  "$root/s11-native/tools/s11-native/guard_cpu.py" \
  --output "$root/outputs/s11-native/cpu-new-run" --timeout 64800 -- \
  bash "$root/s11-native/tools/s11-native/run_cpu.sh"
```

The floating4-row/4-column tile shares restored BF16 weights and prepared
inputs while preserving scalar K accumulation. `tiled` uses the prepared GEMV
for fewer than4 input columns and the tiled path for4 or more; this is a shape
rule, not an assumption that M1 is always Decode. The shared restoration helper
is explicitly inlined to avoid innerloop function-call spills.
It supports loader-validated,
immutable, contiguous2D S11 W2_BF16/W4/I8; GGML owns bounded preparation memory.
Set `S11_CPU_MATMUL=prepared` for the prior CPU adapter. Unsupported production
S11 matmul dispatch is observable with `S11_CPU_COVERAGE=1` in `check_model.cpp`;
the initial candidate's complete-state check had0fallback/11474tiled; the final
shape-specific GEMV/GEMM qualification records both counters separately.
Counters are diagnostic only, disabled in formal timing.

Two qualified experiments are not defaults:
- `S11_CPU_EXPERT_LAYOUT=interleaved4` creates executor-owned
  `[expert][row/4][K128][row%4][36B]` weight banks and batches up to4 current
  assignments for the same expert. Canonical source stays immutable, routes and
  output slots stay in their original order, and each group has its own exact
  dot/scale accumulation. Inverse block checks and M1/5/17/128 chains passed.
  It adds8.44GiB RSS and lost on developer long requests; `reference` is chosen.
- `S11_CPU_HEAD=bf16` creates a model-owned aligned BF16 head from canonical I8
  via bounded row restoration. A private marker selects ordered CPU accumulation
  without changing `model.output`'s S11 identity. It requires prepared/tiled CPU
  matmul and host I8 source, has explicit invalid-mode/source errors, and adds
  about970MiB. Math/state passed after an alignment fix; complete developer
  request cost did not improve, so `packed` remains selected.

Model destruction explicitly releases CPU callback/executor/head owners before
the canonical pimpl weight buffers, preserving borrowed-weight lifetime.

This adapts public BitNet row/column/interleaving methods, not its row-scale
quantization recipe. Group128 scales, probability position, A128 and BF16
boundaries remain S11. The caller supplies validated immutable weights to the
trusted prepared/tile/head helpers; checked standalone dequant/vec-dot APIs
retain malformed-input validation. `report_cpu_baseline.py OUTPUT` verifies
40 selected formal C0/candidate samples;20 initial compiler-regressed samples
remain separate (60 retained). The observed four-load request reductions are
27.8/40.0/15.0/34.3%, with unchanged output IDs and exact useful work.
Decode only improves modestly to about2.80-2.84token/s; Prefill is the main gain.
Selected CPU peak RSS11.693GiB, GPU0,8-CPU affinity. No stable or universally
optimal CPU speed is claimed; developer1+1 is only screening.

The [CB report](../../../outputs/s11-native/20261015-cpu-baseline/REPORT.md)
records source/binary identities, negative results and stopping boundary.
The live GPU build/worktree was observed to change concurrently outside this
CPU task. It was not reverted; CPU C0 used a frozen binary snapshot and final
CPU measurements used the isolated CPU-only build. Do not infer a new paired
GPU comparison or an unchanged live GPU binary from these CPU results.

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

- O4.13 rejects shifted-overlap BF16 roundtrip fusion and retains the original
  two-copy path for that allocator layout. `check_gpu.cpp --roundtrip-alias`
  checks exact, shifted (both directions), and disjoint layouts.
- O4.15 keeps S11 resident decode's static segments captured while dynamic
  FlashAttention runs separately. Parent-graph consumer ranges are indexed once
  when fusion is evaluated, not scanned again for every candidate. The default
  is `split` for `S11_EXPERT_MODE=gpu-resident`; `S11_DECODE_GRAPH=warmup`
  restores the old policy. GPU-cache and other models keep their decode policy.
- `S11_GATE_TRACE=PATH` on the rebuilt loss evaluator writes per-task full-logit
  fingerprints and choice logits. It does not change the Oracle or thresholds.
  Evidence and rejected candidates are in
  [the O4 second batch](../../../outputs/s11-native/20261016-o4-r2/).

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
