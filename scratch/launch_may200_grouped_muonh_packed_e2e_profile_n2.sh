#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="$(date -u +%Y%m%d-%H%M)"
RUN_ID="${RUN_ID:-GM2560-MAY-200S4096-W2048-B16-R1-E8M1-PALLASCEV8192-RING-SAVEMOE-FA4GROUPEDMUONH3-PACKEDA2A-PROFILE-N2-cw-${RUN_STAMP}}"

export MAY_PROFILER_ENABLE_HLO_PROTO="${MAY_PROFILER_ENABLE_HLO_PROTO:-true}"
export MAY_PROFILER_HOST_TRACER_LEVEL="${MAY_PROFILER_HOST_TRACER_LEVEL:-1}"
export MAY_PROFILER_PYTHON_TRACER_LEVEL="${MAY_PROFILER_PYTHON_TRACER_LEVEL:-0}"
export MAY_UPLOAD_PROFILER_ARTIFACT="${MAY_UPLOAD_PROFILER_ARTIFACT:-true}"
export MAY_XLA_FLAGS="${MAY_XLA_FLAGS:---xla_gpu_enable_command_buffer=''}"

experiments/grug/moe/run_cw_may_d2560.sh \
  --submit \
  --run-id "$RUN_ID" \
  --prefix s3://marin-na/tmp/ttl=7d \
  --nodes 2 \
  --worker-cpu 8 \
  --expert-axis 8 \
  --replica-axis 1 \
  --model-axis 1 \
  --allow-cross-node-expert-axis false \
  --batch 16 \
  --seq-len 4096 \
  --sliding-window 2048 \
  --steps 5 \
  --total-tokens 8000000 \
  --profiler-start 2 \
  --profiler-steps 2 \
  --tracker wandb \
  --data synthetic \
  --checkpoints none \
  --remat save_moe \
  --input-embed-sharding hidden_batch \
  --output-proj-sharding lm_head \
  --optimizer muonh \
  --muon-backend-steps 3 \
  --muon-ns-compute-dtype bfloat16 \
  --muon-nesterov true \
  --assert-optimizer-sharding true \
  --match-optimizer-sharding true \
  --expert-3d-optimizer grouped_muonh \
  --ordinary-2d-optimizer muonh \
  --expert-grouped-muonh-packed-entry true \
  --expert-grouped-muonh-chunk-local-boundaries false \
  --mp params=bfloat16,compute=bfloat16,output=bfloat16 \
  --live-param-mode param \
  --attention gpu_fa4_cute \
  --ce-implementation pallas_gpu \
  --moe-implementation ring \
  --watch-interval 0 \
  --log-every 1 \
  --xla-memory-fraction 0.95
