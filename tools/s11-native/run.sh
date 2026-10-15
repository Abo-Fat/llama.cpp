#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "$0")/../../.." && pwd)"
export S11_EXPERT_MODE="${S11_EXPERT_MODE:-gpu-cache}"
case "$S11_EXPERT_MODE" in
  gpu-cache)
    export S11_CUDA_OPT="${S11_CUDA_OPT:-tiled}"
    export S11_HEAD_LAYOUT="${S11_HEAD_LAYOUT:-bf16}"
    cache_args=(--moe-cache-mib 4608)
    if [[ "${S11_FLASH_ATTN:-on}" == "on" ]]; then
      fa_args=(-ctk f16 -ctv f16 -fa on)
    else
      fa_args=(-ctk bf16 -ctv bf16 -fa off)
    fi
    ;;
  hybrid) cache_args=(--moe-cache-mib 3072); fa_args=(-ctk bf16 -ctv bf16 -fa off) ;;
  cpu) cache_args=(); fa_args=(-ctk bf16 -ctv bf16 -fa off) ;;
  *) printf 'Invalid S11_EXPERT_MODE: %s\n' "$S11_EXPERT_MODE" >&2; exit 1 ;;
esac
export S11_EMBEDDING="${S11_EMBEDDING:-gpu}"
export S11_CPU_MATMUL="${S11_CPU_MATMUL:-prepared}"
exec taskset -c 0-7 "$workspace/baseline-envs/s11-native/build/bin/llama-server" \
  -m "$workspace/baseline-models/s11-native/s11-native.gguf" \
  --host 127.0.0.1 --port 18141 -ngl 99 --cpu-moe "${cache_args[@]}" \
  --fit off -t 8 -tb 8 -c 4096 -b 512 -ub 128 -np 1 --no-warmup \
  --poll 0 --poll-batch 0 "${fa_args[@]}" --backend-sampling "$@"
