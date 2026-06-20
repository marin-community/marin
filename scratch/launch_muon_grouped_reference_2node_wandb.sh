#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="$(date -u +%Y%m%d-%H%M%S)"
export MARIN_PREFIX="${MARIN_PREFIX:-s3://marin-na/tmp/ttl=7d}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}"

export MUON_BENCH_GPU_REPLICAS="${MUON_BENCH_GPU_REPLICAS:-2}"
export MUON_BENCH_LAYERS="${MUON_BENCH_LAYERS:-26}"
# Use the second node as data/FSDP by default. R2D1 replicates the full expert
# params, grads, and grouped MuonH state across nodes and can OOM before timing.
export MUON_BENCH_REPLICA_AXIS="${MUON_BENCH_REPLICA_AXIS:-1}"
export MUON_BENCH_DATA_AXIS="${MUON_BENCH_DATA_AXIS:-2}"
export MUON_BENCH_EXPERT_AXIS="${MUON_BENCH_EXPERT_AXIS:-8}"
export MUON_BENCH_MODEL_AXIS="${MUON_BENCH_MODEL_AXIS:-1}"
# Larger groups pack the grouped-to-FSDP restore into fewer collectives, but G26
# and G13 OOM on the 2-node full-layer benchmark. G8 is the current lazy
# reference because it is the only packed group size in this sweep that ran.
export MUON_BENCH_NS4D_GROUP_SIZE="${MUON_BENCH_NS4D_GROUP_SIZE:-8}"
export MUON_BENCH_NS4D_GROUP_AXIS="${MUON_BENCH_NS4D_GROUP_AXIS:-replica_dcn,data}"
export MUON_BENCH_KINDS="${MUON_BENCH_KINDS:-real_expert_fsdp_grouped_muonh_optimizer_update}"
export MUON_BENCH_SWEEP_BACKEND_STEPS="${MUON_BENCH_SWEEP_BACKEND_STEPS:-3}"
export MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES="${MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES:-512}"
export MUON_BENCH_DTYPE="${MUON_BENCH_DTYPE:-bf16}"
export MUON_BENCH_NS_COMPUTE_DTYPE="${MUON_BENCH_NS_COMPUTE_DTYPE:-bf16}"
export MUON_BENCH_WARMUP="${MUON_BENCH_WARMUP:-1}"
export MUON_BENCH_ITERS="${MUON_BENCH_ITERS:-3}"
export MUON_BENCH_MODE="${MUON_BENCH_MODE:-both}"
export MUON_BENCH_ENABLE_JAX_PROFILE="${MUON_BENCH_ENABLE_JAX_PROFILE:-false}"
export MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES="${MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES:-true}"
export MUON_BENCH_DISABLE_ABSTRACT_MESH="${MUON_BENCH_DISABLE_ABSTRACT_MESH:-true}"
export MUON_BENCH_WORKER_CPU="${MUON_BENCH_WORKER_CPU:-8}"
export MUON_BENCH_WORKER_RAM="${MUON_BENCH_WORKER_RAM:-256g}"

case "${MUON_BENCH_NS4D_GROUP_SIZE}" in
  8)
    ;;
  4|13|26)
    if [[ "${MUON_BENCH_ALLOW_UNFIT_GROUP_SIZE:-false}" != "true" ]]; then
      echo "MUON_BENCH_NS4D_GROUP_SIZE=${MUON_BENCH_NS4D_GROUP_SIZE} is a known-OOM 2-node setting." >&2
      echo "Using known-fit reference group size 8. Set MUON_BENCH_ALLOW_UNFIT_GROUP_SIZE=true to override." >&2
      export MUON_BENCH_NS4D_GROUP_SIZE=8
    fi
    ;;
esac

export MUON_BENCH_TRACKER="${MUON_BENCH_TRACKER:-wandb}"
export MUON_BENCH_WANDB_PROJECT="${MUON_BENCH_WANDB_PROJECT:-${WANDB_PROJECT:-marin_moe}}"
export MUON_BENCH_WANDB_GROUP="${MUON_BENCH_WANDB_GROUP:-grug-moe-cw-muon-update-bench}"

export RUN_ID="${RUN_ID:-MUON-BENCH-D2560-L26-R${MUON_BENCH_REPLICA_AXIS}D${MUON_BENCH_DATA_AXIS}E${MUON_BENCH_EXPERT_AXIS}-G${MUON_BENCH_NS4D_GROUP_SIZE}-H3-N2-cw-${STAMP}}"

if [[ "${MUON_BENCH_READABLE_PROFILE:-false}" == "true" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
else
  export XLA_FLAGS="${XLA_FLAGS:-}"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

echo "Launching ${RUN_ID}"
echo "Grouped MuonH group size: ${MUON_BENCH_NS4D_GROUP_SIZE}"
echo "W&B: ${WANDB_ENTITY:-marin-community}/${MUON_BENCH_WANDB_PROJECT}"
echo "Output prefix: ${MARIN_PREFIX}/experiments/grug-moe-cw/muon-update-bench/${RUN_ID}-*"

if [[ "${MUON_BENCH_DRY_RUN:-false}" == "true" ]]; then
  echo "Dry run only; not launching Iris."
  exit 0
fi

bash scratch/muon_update_bench_fast_loop.sh iris fullprod-r4e8-l26-h3
