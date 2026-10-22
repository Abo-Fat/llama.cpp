#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "$0")/../../.." && pwd)"
export S11_EXPERT_MODE="${S11_EXPERT_MODE:-gpu-cache}"
case "$S11_EXPERT_MODE" in
  gpu-cache)
    export S11_CUDA_OPT="${S11_CUDA_OPT:-tiled}"
    export S11_HEAD_LAYOUT="${S11_HEAD_LAYOUT:-mixed}"
    export S11_HEAD_MMA="${S11_HEAD_MMA:-0}"
    export S11_GREEDY_FAST="${S11_GREEDY_FAST:-1}"
    export S11_MOE_PROMPT_ADMISSION="${S11_MOE_PROMPT_ADMISSION:-0}"
    export S11_MTP_VOCAB="${S11_MTP_VOCAB:-98304}"
    cache_args=(--cpu-moe --moe-cache-mib 4864)
    ;;
  gpu-resident)
    export S11_CUDA_OPT="${S11_CUDA_OPT:-tiled}"
    export S11_HEAD_LAYOUT="${S11_HEAD_LAYOUT:-bf16}"
    export S11_HEAD_MMA="${S11_HEAD_MMA:-1}"
    export S11_GREEDY_FAST="${S11_GREEDY_FAST:-0}"
    export S11_SCHED_REUSE="${S11_SCHED_REUSE:-1}"
    export S11_MTP_VOCAB="${S11_MTP_VOCAB:-98304}"
    cache_args=()
    ;;
  hybrid) cache_args=(--cpu-moe --moe-cache-mib 3072); fa_args=(-ctk bf16 -ctv bf16 -fa off) ;;
  cpu) cache_args=(--cpu-moe); fa_args=(-ctk bf16 -ctv bf16 -fa off) ;;
  *) printf 'Invalid S11_EXPERT_MODE: %s\n' "$S11_EXPERT_MODE" >&2; exit 1 ;;
esac
if [[ "$S11_EXPERT_MODE" == gpu-cache || "$S11_EXPERT_MODE" == gpu-resident ]]; then
  case "${S11_FLASH_ATTN:-1}" in
    1|on) export S11_FLASH_ATTN=1; fa_args=(-ctk f16 -ctv f16 -fa on) ;;
    0|off) export S11_FLASH_ATTN=0; fa_args=(-ctk bf16 -ctv bf16 -fa off) ;;
    *) printf 'Invalid S11_FLASH_ATTN: %s\n' "$S11_FLASH_ATTN" >&2; exit 1 ;;
  esac
fi
export S11_EMBEDDING="${S11_EMBEDDING:-gpu}"
export S11_CPU_MATMUL="${S11_CPU_MATMUL:-prepared}"
exec taskset -c 0-7 "$workspace/baseline-envs/s11-native/build/bin/llama-server" \
  -m "$workspace/baseline-models/s11-native/s11-native.gguf" \
  --host 127.0.0.1 --port 18141 -ngl 99 "${cache_args[@]}" \
  --fit off -t 8 -tb 8 -c 4096 -b 512 -ub 128 -np 1 --no-warmup \
  --poll 0 --poll-batch 0 "${fa_args[@]}" --backend-sampling "$@"
