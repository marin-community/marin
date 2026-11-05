#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUN_NAME="${1:?Unique job name required}"
TASK_RECIPE="${2:?Recipe required}"
TASK_GPUS="${3:?GPU count required}"
cd "$TASK_ROOT/marin"
uv run --project /Users/benfeuer/Documents/marin iris \
  --config /Users/benfeuer/Documents/marin/lib/iris/config/cw-rno2a.yaml \
  job run --no-wait --enable-extra-resources --gpu "H100x${TASK_GPUS}" \
  --priority interactive --cpu 16 --memory 128GB --disk 100GB \
  --timeout 900 --max-retries 0 --job-name "$TASK_RUN_NAME" \
  --sync-package marin-core --extra gpu --extra dedup \
  -e JAX_PLATFORMS cuda,cpu -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.85 \
  -e PYTHONPATH . -e NCCL_DEBUG WARN \
  -e XLA_FLAGS '--xla_allow_excess_precision=false --xla_gpu_enable_triton_gemm=false' \
  -e CW_KEY_ID "${CW_KEY_ID:?}" -e CW_KEY_SECRET "${CW_KEY_SECRET:?}" \
  -- bash -c 'uv pip install --python .venv/bin/python nvidia-cufile==1.15.1.6 nvidia-cusparselt-cu13==0.8.0 nvidia-curand==10.4.0.35 && exec .venv/bin/python -m experiments.grug.tied_drafter.profile --recipe "$1" --gpus "$2" --steps 6 --draft-learning-rate 0.001 --output "$IRIS_OUTPUT_DIR"' _ "$TASK_RECIPE" "$TASK_GPUS"
