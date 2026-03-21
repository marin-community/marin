#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing virtualenv at ${SCRIPT_DIR}/.venv. Run 'uv venv' from the repo root first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

: "${IRIS_CONFIG:=lib/iris/examples/marin.yaml}"
: "${IRIS_REGION:=us-east5}"
: "${IRIS_SUBMIT_RETRIES:=30}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before launching." >&2
  exit 1
fi

export UV_CACHE_DIR="${TMPDIR:-/tmp}/marin-uv-cache"
mkdir -p "${UV_CACHE_DIR}"

iris_env_args=(
  -e WANDB_API_KEY "${WANDB_API_KEY}"
  -e HF_TOKEN "${HF_TOKEN_VALUE}"
)

last_rc=0
for attempt in $(seq 1 "${IRIS_SUBMIT_RETRIES}"); do
  job_name="isoflop-moe-adamh-gatednorm-r2-$(date -u +%Y%m%d-%H%M%S)"

  if uv run iris --config "${IRIS_CONFIG}" job run \
    --no-wait \
    --extra marin:tpu \
    --cpu 4 \
    --memory 16GB \
    --disk 20GB \
    --region "${IRIS_REGION}" \
    --job-name "${job_name}" \
    "${iris_env_args[@]}" \
    -- \
    python experiments/grug/moe_scaling_iteration_02/launch_isoflop_moe_adamh_gatednorm.py \
    "$@"; then
    exit 0
  fi

  last_rc=$?
  if [[ "${attempt}" -lt "${IRIS_SUBMIT_RETRIES}" ]]; then
    echo "Attempt ${attempt}/${IRIS_SUBMIT_RETRIES} failed with exit code ${last_rc}; retrying in 5s." >&2
    sleep 5
  fi
done

exit "${last_rc}"
