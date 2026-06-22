#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-MAY357-DEEPEP-EP8-OFFLOAD-HIDDEN-L26-B8-N1-$(date -u +%Y%m%d-%H%M)}"

export MAY_DEEPEP_SOURCE_ARCHIVE="${MAY_DEEPEP_SOURCE_ARCHIVE:-s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-source/DeepEP-7febc6e25660af0f54d95dd781ecdcd62265ecca.tgz}"

bash experiments/grug/moe/run_cw_may_d2560.sh \
  --submit \
  --run-id "$RUN_ID" \
  --nodes 1 \
  --worker-cpu 32 \
  --task-image ghcr.io/marin-community/iris-task-cuda-devel:969c0e3 \
  --expert-axis 8 \
  --replica-axis 1 \
  --model-axis 1 \
  --batch 8 \
  --seq-len 4096 \
  --sliding-window 2048 \
  --layers 26 \
  --steps 8 \
  --profiler-steps 0 \
  --data synthetic \
  --checkpoints none \
  --remat offload_moe_hidden \
  --optimizer sgd \
  --mp params=bfloat16,compute=bfloat16,output=bfloat16 \
  --live-param-mode param \
  --attention gpu_fa4_cute \
  --ce-implementation pallas_gpu \
  --moe-implementation deepep \
  --deepep-bootstrap-source true \
  --input-embed-sharding replicated \
  --output-proj-sharding lm_head
