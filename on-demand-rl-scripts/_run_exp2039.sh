#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <zone> <tpu_type> <deployment_preset> <shared_bucket>" >&2
  exit 2
fi

DEFAULT_ZONE="$1"
DEFAULT_TPU_TYPE="$2"
DEFAULT_DEPLOYMENT_PRESET="$3"
DEFAULT_SHARED_BUCKET="$4"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE=""
if [[ -f ".marin.yaml" ]]; then
  CONFIG_FILE=".marin.yaml"
elif [[ -f ".levanter.yaml" ]]; then
  CONFIG_FILE=".levanter.yaml"
elif [[ -f ".config" ]]; then
  CONFIG_FILE=".config"
fi

if [[ -n "${CONFIG_FILE}" ]]; then
  echo "Using ${CONFIG_FILE} for launch defaults and env injection."
else
  echo "No .marin.yaml/.levanter.yaml/.config found in ${REPO_ROOT}; using launch defaults only."
fi

RUN_ID="${RUN_ID:-exp2039-$(date +%Y%m%d-%H%M%S)}"
ZONE="${ZONE:-${DEFAULT_ZONE}}"
TPU_TYPE="${TPU_TYPE:-${DEFAULT_TPU_TYPE}}"
DEPLOYMENT_PRESET="${DEPLOYMENT_PRESET:-${DEFAULT_DEPLOYMENT_PRESET}}"
SHARED_BUCKET="${SHARED_BUCKET:-${DEFAULT_SHARED_BUCKET}}"
SHARED_ROOT="${SHARED_ROOT:-gs://${SHARED_BUCKET}/tmp/exp2039/${RUN_ID}}"
ROLLOUT_SHAPE="${ROLLOUT_SHAPE:-exp2039}"
CAPACITY_MODE="${CAPACITY_MODE:-spot}"
RETRIES="${RETRIES:-10}"
SEED="${SEED:-42}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

CAPACITY_ARGS=()
case "${CAPACITY_MODE}" in
  spot | preemptible)
    CAPACITY_ARGS=(--spot)
    ;;
  on-demand)
    CAPACITY_ARGS=(--on-demand)
    ;;
  reserved)
    CAPACITY_ARGS=(--reserved)
    ;;
  best-effort)
    CAPACITY_ARGS=(--capacity_type=best-effort)
    ;;
  *)
    echo "Invalid CAPACITY_MODE=${CAPACITY_MODE}. Use one of: spot, on-demand, reserved, best-effort." >&2
    exit 2
    ;;
esac

echo "Launching exp2039 run_id=${RUN_ID} preset=${DEPLOYMENT_PRESET} zone=${ZONE} tpu_type=${TPU_TYPE} capacity_mode=${CAPACITY_MODE}"
echo "Shared root: ${SHARED_ROOT}"

resolve_tpu_type() {
  local role="$1"
  local preset_type
  preset_type=$(uv run python -c "
from experiments.exp2039_rl_math500 import DEPLOYMENT_PRESETS
p = DEPLOYMENT_PRESETS['${DEPLOYMENT_PRESET}']
print(p.trainer_tpu_type if '${role}' == 'trainer' else p.sampler_tpu_type)
" 2>/dev/null) || true
  echo "${preset_type:-${TPU_TYPE}}"
}

launch_role() {
  local role="$1"
  local role_tpu_type
  role_tpu_type="$(resolve_tpu_type "${role}")"
  echo "Role ${role} using tpu_type=${role_tpu_type}"
  # IMPORTANT: --retries=0 and NO --foreground.
  # --foreground uses docker run -t, which blocks SSH until the container exits.
  # On any failure, launch.py retries and does docker rm -f, DESTROYING crash logs.
  # Without --foreground, launch.py uses docker run -d (detached) and returns immediately.
  # --retries=0 ensures we never auto-retry and destroy evidence.
  uv run python lib/levanter/infra/launch.py \
    --tpu_name="${TPU_PREFIX:-exp2039-nb}-${role}" \
    --tpu_type="${role_tpu_type}" \
    --zone="${ZONE}" \
    "${CAPACITY_ARGS[@]}" \
    --retries=0 \
    -e TPU_BACKEND_TYPE jax \
    -e PJRT_DEVICE TPU \
    -e VLLM_ENABLE_V1_MULTIPROCESSING 0 \
    -- \
    bash /opt/marin/on-demand-rl-scripts/bootstrap_rl.sh /opt/marin/experiments/exp2039_rl_math500.py \
      --mode "${role}" \
      --deployment-preset "${DEPLOYMENT_PRESET}" \
      --run-id "${RUN_ID}" \
      --shared-root "${SHARED_ROOT}" \
      --rollout-shape "${ROLLOUT_SHAPE}" \
      --seed "${SEED}" \
      ${NUM_TRAIN_STEPS:+--num-train-steps "${NUM_TRAIN_STEPS}"}
}

launch_role trainer &
launch_role sampler &
wait
