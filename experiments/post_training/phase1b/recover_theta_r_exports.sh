#!/usr/bin/env bash
set -euo pipefail

STEP=${1:?pass an export step}
MODE=${2:?pass dry-run, submit, or verify-only}
case "$STEP" in 10|30|50|100|125|remaining) ;; *) echo "unexpected export step: $STEP" >&2; exit 2;; esac
case "$MODE" in dry-run|submit|verify-only) ;; *) echo "unexpected mode: $MODE" >&2; exit 2;; esac
EXPORT_JOB_SUFFIX=${EXPORT_JOB_SUFFIX:-guarded-20261109}

cd /app/vendor/skyrl
checkout=/app/vendor/skyrl
IRIS_VENV="${IRIS_VENV:-/tmp/marinskyrl/.iris-phase1b-export-venv}"
IRIS_WORKDIR="${IRIS_WORKDIR:-/app}"
bash /app/vendor/skyrl/cloud/iris/bootstrap_runtime.sh "$checkout" "$IRIS_VENV" /app/vendor/skyrl/.iris-runtime-env megatron-export production
source /app/vendor/skyrl/.iris-runtime-env

python - <<'PY'
from skyrl_train.entrypoints.checkpoint_export import run
from skyrl_train.hf_model_io import has_completed_hf_model_export, verify_completed_hf_model_export
from cloud.iris.export_hf_checkpoint import argument_parser

assert run and has_completed_hf_model_export and verify_completed_hf_model_export
assert any(action.dest == "verify_only" for action in argument_parser()._actions)
print("Validated guarded export runtime and verify-only completion path")
PY

python - <<'PY'
from pathlib import Path

import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

source = "s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/sft-theta_R-lr2e-06-seed0/2026.11.09.3/resolved-launch.yaml"
destination = "/tmp/marinskyrl/phase1b-theta-r-resolved-inner.yaml"
configure_coreweave_s3()
filesystem, path = filesystem_for(source)
with filesystem.open(path) as handle:
    document = yaml.safe_load(handle)
config = document["config"]
assert config["artifacts"]["checkpoint_root"].startswith("s3://marin-us-east-08a/")
assert config["artifacts"]["export_root"].startswith("s3://marin-us-east-08a/")
assert config["iris"]["cluster"] == "cw-rno2a"
assert config["skyrl"]["trainer"]["hf_export_steps"] == [10, 30, 50, 100, 125]
Path(destination).parent.mkdir(parents=True, exist_ok=True)
with open(destination, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print("Validated Phase 1b theta_R recovery config and staged its inner launch document")
PY

checkpoint_root=s3://marin-us-east-08a/tmp/ttl=30d/skyrl/marin/users/benfeuer/checkpoints/phase1b/sft-theta_R-lr2e-06-seed0/2026.11.09.3/checkpoints
export_root=s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/sft-theta_R-lr2e-06-seed0/2026.11.09.3/exports
if [[ "$MODE" == dry-run && "$STEP" == 10 ]]; then
  python /app/experiments/post_training/phase1b/audit_theta_r_checkpoint_metadata.py --checkpoint-root "$checkpoint_root"
fi
if [[ "$STEP" == remaining ]]; then
  steps=(30 50 100 125)
else
  steps=("$STEP")
fi
for step in "${steps[@]}"; do
  args=(
    --request "$checkpoint_root/global_step_$step"
    --launch-config /tmp/marinskyrl/phase1b-theta-r-resolved-inner.yaml
    --gpu-variant H100
    --priority interactive
    --job-name "opd-1b-theta-r-export-s${step}-${EXPORT_JOB_SUFFIX}"
  )
  if [[ "$MODE" == dry-run ]]; then args+=(--dry-run); fi
  if [[ "$MODE" == verify-only ]]; then args+=(--verify-only); fi
  python -m cloud.iris.export_hf_checkpoint "${args[@]}"
  if [[ "$MODE" != dry-run ]]; then
    export CHECKPOINT_ROOT="$checkpoint_root" EXPORT_ROOT="$export_root" EXPORT_STEP="$step"
    python - <<'PY'
import os

from marinskyrl.checkpoint_paths import policy_export_path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from skyrl_train.hf_export import read_hf_export_request
from skyrl_train.hf_export_schema import HFExportStatus
from skyrl_train.hf_model_io import verify_completed_hf_model_export

configure_coreweave_s3()
step = int(os.environ["EXPORT_STEP"])
request = read_hf_export_request(f"{os.environ['CHECKPOINT_ROOT']}/global_step_{step}")
assert request is not None and request.status is HFExportStatus.COMPLETE
policy = policy_export_path(os.environ["EXPORT_ROOT"], step)
verify_completed_hf_model_export(policy)
print(f"Verified completed HF export and request at step {step}: {policy}")
PY
  fi
done
