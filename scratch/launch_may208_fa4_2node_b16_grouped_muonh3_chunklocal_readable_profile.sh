#!/usr/bin/env bash
set -euo pipefail

cd /Users/dlwh/.codex/worktrees/079b/marin

RUN_STAMP="$(date -u +%Y%m%d-%H%M)"
RUN_ID="${RUN_ID:-GM2560-MAY208-B16-R2D1E8-GMUONH3-CHUNKLOCAL-PROF-N2-cw-${RUN_STAMP}}"

KUBECONFIG="${KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}" \
LEVANTER_PALLAS_GPU_CUSTOM_BWD_V_BLOCK_SIZE=8192 \
MAY_PROFILER_ENABLE_HLO_PROTO=true \
MAY_PROFILER_HOST_TRACER_LEVEL=1 \
MAY_PROFILER_PYTHON_TRACER_LEVEL=0 \
MAY_UPLOAD_PROFILER_ARTIFACT=true \
experiments/grug/moe/run_cw_may_d2560.sh \
  --submit \
  --run-id "$RUN_ID" \
  --prefix s3://marin-na/tmp/ttl=7d \
  --nodes 2 \
  --data synthetic \
  --checkpoints none \
  --worker-cpu 8 \
  --model-axis 1 \
  --expert-axis 8 \
  --replica-axis 2 \
  --batch 16 \
  --seq-len 4096 \
  --sliding-window 2048 \
  --steps 8 \
  --total-tokens 8000000 \
  --profiler-start 3 \
  --profiler-steps 2 \
  --tracker wandb \
  --watch-interval 0 \
  --log-every 1 \
  --log-jaxprs false \
  --log-xla-hlo false \
  --save-xla-dumps false \
  --xla-flags "--xla_gpu_enable_command_buffer=''" \
  --ce-implementation pallas_gpu \
  --moe-implementation ring \
  --input-embed-sharding replicated \
  --output-proj-sharding replicated \
  --optimizer muonh \
  --ordinary-2d-optimizer muonh \
  --expert-3d-optimizer grouped_muonh \
  --expert-grouped-muonh-group-size 2 \
  --expert-grouped-muonh-packed-entry false \
  --expert-grouped-muonh-chunk-local-boundaries true \
  --muon-backend-steps 3 \
  --muon-orthogonalization-layout stack_batch_4d_sharded \
  --muon-max-grouped-stack-size 512 \
  --muon-ns-compute-dtype bf16 \
  --muon-nesterov true \
  --assert-optimizer-sharding true \
  --match-optimizer-sharding true \
  --live-param-mode param \
  --remat save_moe \
  --attention gpu_fa4_cute \
  --mp params=bfloat16,compute=bfloat16,output=bfloat16 \
  --xla-memory-fraction 0.90 \
  --pallas-ce-autotune-on-miss true
