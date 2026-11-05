#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUN_NAME="${1:?Unique trial name required}"
TASK_MODE="${2:?native or draft required}"
TASK_BENCHMARK_PROTOCOL="experiments/grug/tied-drafter-benchmark-protocol.json"
TASK_PROTOCOL="experiments/grug/tied_drafter/protocol-v1.json"
TASK_EXCHANGE_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
cd "$TASK_ROOT/marin"

submit_role() {
  local task_role="$1"
  local task_gpus="$2"
  shift 2
  uv run --project /Users/benfeuer/Documents/marin iris \
    --config /Users/benfeuer/Documents/marin/lib/iris/config/cw-rno2a.yaml \
    job run --no-wait --enable-extra-resources --gpu "H100x${task_gpus}" \
    --priority interactive --cpu 16 --memory 128GB --disk 100GB \
    --timeout 420 --max-retries 0 --job-name "${TASK_RUN_NAME}-${task_role}" \
    --sync-package marin-core --extra gpu --extra dedup \
    -e JAX_PLATFORMS cuda,cpu -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.85 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e PYTHONPATH . -e NCCL_DEBUG WARN \
    -e XLA_FLAGS '--xla_allow_excess_precision=false --xla_gpu_enable_triton_gemm=false' \
    -e CW_KEY_ID "${CW_KEY_ID:?}" -e CW_KEY_SECRET "${CW_KEY_SECRET:?}" \
    -e TIED_DRAFTER_TOKEN "$TASK_EXCHANGE_TOKEN" \
    -- bash -c 'uv pip install --python .venv/bin/python nvidia-cufile==1.15.1.6 nvidia-cusparselt-cu13==0.8.0 nvidia-curand==10.4.0.35 && exec .venv/bin/python -m "$1" --output "$IRIS_OUTPUT_DIR" "${@:2}"' _ "$@"
}

submit_role hero 8 experiments.grug.tied_drafter_benchmark \
  --protocol "$TASK_PROTOCOL" --benchmark-protocol "$TASK_BENCHMARK_PROTOCOL" --mode "$TASK_MODE"
if [[ "$TASK_MODE" == draft ]]; then
  submit_role draft 4 experiments.grug.tied_drafter.draft_worker \
    --protocol "$TASK_PROTOCOL" --hero-endpoint "/benfeuer/${TASK_RUN_NAME}-hero/tied-drafter"
fi
