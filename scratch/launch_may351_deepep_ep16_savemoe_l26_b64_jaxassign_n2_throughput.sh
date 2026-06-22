#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-MAY351-DEEPEP-EP16-SAVEMOE-JAXASSIGN-FIXEDFFICOLLAPSE-L26-B64-N2-$(date -u +%Y%m%d-%H%M)}"

export LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE="${LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE:-jax}"
export LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE="${LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE:-ffi}"

bash experiments/grug/moe/run_cw_may_d2560.sh \
  --submit \
  --run-id "$RUN_ID" \
  --nodes 2 \
  --worker-cpu 32 \
  --task-image ghcr.io/marin-community/iris-task-cuda-devel:969c0e3 \
  --expert-axis 16 \
  --replica-axis 1 \
  --model-axis 1 \
  --allow-cross-node-expert-axis true \
  --deepep-ranks-per-node 8 \
  --deepep-processes-per-task 8 \
  --batch 64 \
  --seq-len 4096 \
  --sliding-window 2048 \
  --layers 26 \
  --steps 8 \
  --profiler-steps 0 \
  --data synthetic \
  --checkpoints none \
  --remat save_moe \
  --optimizer sgd \
  --mp params=bfloat16,compute=bfloat16,output=bfloat16 \
  --live-param-mode param \
  --attention gpu_fa4_cute \
  --ce-implementation pallas_gpu \
  --moe-implementation deepep_internode \
  --deepep-bootstrap-source true \
  --input-embed-sharding replicated \
  --output-proj-sharding lm_head
