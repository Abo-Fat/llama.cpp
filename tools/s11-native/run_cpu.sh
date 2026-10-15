#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "$0")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export S11_EXPERT_MODE=cpu
export S11_CPU_MATMUL="${S11_CPU_MATMUL:-tiled}"
export S11_CPU_EXPERT_LAYOUT="${S11_CPU_EXPERT_LAYOUT:-reference}"
export S11_CPU_HEAD="${S11_CPU_HEAD:-packed}"
exec taskset -c 0-7 "$workspace/baseline-envs/s11-native-cpu/build/bin/llama-server" \
  -m "$workspace/baseline-models/s11-native/s11-native.gguf" \
  --device none -ngl 0 --cpu-moe --host 127.0.0.1 --port 18141 \
  --fit off -t 8 -tb 8 -c 4096 -b 512 -ub 128 -np 1 --no-warmup \
  --poll 0 --poll-batch 0 -ctk bf16 -ctv bf16 -fa off --backend-sampling "$@"
