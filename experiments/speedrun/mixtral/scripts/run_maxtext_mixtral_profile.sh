#!/usr/bin/env bash
# Example run (from repo root, with tmp dirs on local SSD):
#   TMPDIR=/tmp RAY_TMPDIR=/tmp ./experiments/speedrun/mixtral/scripts/run_maxtext_mixtral_profile.sh
# Run MaxText Mixtral profiling on v5p-64 via Marin's Ray launcher.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH="${REPO_ROOT}/maxtext_marin"
if [[ ! -d "${VENV_PATH}" || ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtualenv at ${VENV_PATH}. Please run 'uv venv --python 3.12 maxtext_marin' first." >&2
  exit 1
fi

if [[ ! -d "submodules/maxtext" ]]; then
  echo "Expected MaxText checkout under submodules/maxtext. Please follow docs/tutorials/co-develop.md." >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"

uv pip install -e lib/marin >/dev/null
uv pip install -e lib/levanter >/dev/null

RUN_SUFFIX="$(date +%Y%m%d%H%M%S)"
RUN_NAME="maxtext_mixtral_profile_${RUN_SUFFIX}"
OUTPUT_GCS="gs://marin-us-central1/maxtext/profiles/${RUN_NAME}"

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
cd submodules/maxtext
export RUN_PREFLIGHT=false
export REPO_ROOT="$(pwd)/.."
export PYTHONPATH="$(pwd)/src:${REPO_ROOT}/lib/marin/src:${REPO_ROOT}/lib/levanter/src:${REPO_ROOT}/experiments"
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=81920 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
REQ_FILE=/tmp/maxtext_profile_requirements.txt
cat <<'REQ' > "${REQ_FILE}"
absl-py==2.3.1
aqtp==0.9.0
datasets==3.6.0
flax==0.10.0
gcsfs==2025.3.0
grain[parquet]==0.2.9
huggingface_hub==0.36.0
jsonlines==4.0.0
ml-collections==1.1.0
ml-goodput-measurement==0.0.15
omegaconf==2.3.0
optax==0.2.6
orbax-checkpoint==0.11.26
pathwaysutils==0.1.3
protobuf==4.23.4
pydantic==2.11.10
sentencepiece==0.2.1
tensorboard==2.15.1
tensorboard-data-server==0.7.2
tensorboard-plugin-profile==2.15.0
tensorboardx==2.6.4
tokenizers==0.22.1
tiktoken==0.12.0
transformers==4.57.1
cloud-tpu-diagnostics==0.1.5
werkzeug==3.1.3
mlperf-logging @ https://github.com/mlcommons/logging/archive/38ab22670527888c8eb7825a4ece176fcc36a95d.zip
REQ
python3 -m pip install --upgrade pip
python3 -m pip uninstall -y cloud-accelerator-diagnostics google-cloud-aiplatform google-cloud-bigquery google-cloud-resource-manager google-cloud-storage shapely || true
python3 -m pip install --force-reinstall \
  numpy==1.26.4 \
  ml_dtypes==0.5.0
python3 -m pip install --force-reinstall \
  "jax[tpu]==0.6.2" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install --requirement "${REQ_FILE}"
python3 - <<'PY'
import jax
print("JAX OK", jax.__version__, "PJRT backend:", jax.default_backend())
try:
    from jaxlib import xla_extension
except ImportError:
    from jax._src.lib import _jax as xla_extension
print("DistributedRuntimeClient?", hasattr(xla_extension, "DistributedRuntimeClient"))
PY
python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=mixtral-8x7b \
  steps=40 \
  per_device_batch_size=32 \
  enable_checkpointing=false \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  max_target_length=1024 \
  base_output_directory=${OUTPUT_PATH} \
  run_name=${RUN_NAME} \
  dataset_type=synthetic \
  reuse_example_batch=1 \
  gcs_metrics=true \
  profiler=xplane \
  skip_first_n_steps_for_profiler=10 \
  profiler_steps=10 \
  upload_all_profiler_results=False \
  attention=dot_product \
  enable_nnx=false \
  sa_block_q=1024 \
  sa_block_q_dkv=2048 \
  sa_block_q_dq=2048
EOF
)

UV_PROJECT_ENV="${VENV_PATH}" uv run python -m marin.run.ray_run \
  --cluster "infra/marin-us-central1.yaml" \
  --extra tpu \
  --env_vars WANDB_MODE online \
  --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" \
  --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" \
  --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" \
  --env_vars HF_TOKEN "${HF_TOKEN:-}" \
  --env_vars PIP_NO_CACHE_DIR 1 \
  --env_vars RAY_TMPDIR /tmp \
  --env_vars TMPDIR /tmp \
  --env_vars MAXTEXT_REPO_ROOT submodules/maxtext \
  --env_vars OUTPUT_PATH "${OUTPUT_GCS}" \
  --env_vars RUN_NAME "${RUN_NAME}" \
  -- \
  bash -lc "${REMOTE_CMD}"
