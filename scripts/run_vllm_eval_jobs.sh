#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RAY_RUN="${REPO_ROOT}/lib/marin/src/marin/run/ray_run.py"
DEFAULT_CLUSTER_CONFIG="${REPO_ROOT}/infra/marin-us-east1-d-vllm.yaml"

WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
HUGGINGFACE_TOKEN_VALUE="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

if [[ -z "${WANDB_API_KEY_VALUE}" ]]; then
  echo "WANDB_API_KEY must be set in the environment." >&2
  exit 1
fi

if [[ -z "${HUGGINGFACE_TOKEN_VALUE}" ]]; then
  echo "HUGGINGFACE_TOKEN or HUGGING_FACE_HUB_TOKEN must be set in the environment." >&2
  exit 1
fi

CLUSTER_CONFIG="${1:-${DEFAULT_CLUSTER_CONFIG}}"

EXPERIMENTS=(
  "experiments/exp905c_vllm_eval_model_marin_8b_base.py"
  "experiments/exp905c_vllm_eval_model_gemma_3_27b_pt.py"
  "experiments/exp905c_vllm_eval_model_llama_3_1_8b.py"
  "experiments/exp905c_vllm_eval_model_olmo3.py"
  "experiments/exp905c_vllm_eval_model_qwen2_5_7b.py"
  "experiments/exp905c_vllm_eval_model_qwen3_8b_base.py"
)

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run python)
else
  PYTHON_RUNNER=(python)
fi

for experiment in "${EXPERIMENTS[@]}"; do
  echo "Submitting ${experiment} to ${CLUSTER_CONFIG}"
  "${PYTHON_RUNNER[@]}" "${RAY_RUN}" --no_wait \
    --env_vars WANDB_API_KEY "${WANDB_API_KEY_VALUE}" \
    --env_vars HUGGING_FACE_HUB_TOKEN "${HUGGINGFACE_TOKEN_VALUE}" \
    --cluster "${CLUSTER_CONFIG}" \
    -- python "${experiment}" --force_run_failed True
done
