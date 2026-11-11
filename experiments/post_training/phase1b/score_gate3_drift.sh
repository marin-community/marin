#!/usr/bin/env bash
set -euo pipefail

mode=${1:?pass reference or a condition}
if [[ "$mode" == reference ]]; then
  model_uri=s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192
else
  condition=$mode
  step=${2:?pass a verified export step}
  case "$condition" in theta_R|theta_J|theta_RJ) ;; *) echo "unexpected condition: $condition" >&2; exit 2;; esac
  case "$step" in 10|30|50|100|125) ;; *) echo "unexpected export step: $step" >&2; exit 2;; esac
  version=2026.11.10.14
  if [[ "$condition" == theta_J ]]; then version=2026.11.10.13; fi
  model_uri="s3://marin-rno2a/marin/users/benfeuer/checkpoints/phase1b/sft-${condition}-lr2e-06-seed0/${version}/exports/global_step_${step}/policy"
fi
export MODEL_URI=$model_uri

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python - <<'PY'
import json
import os
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy
from skyrl_train.hf_model_io import verify_completed_hf_model_export

configure_coreweave_s3()
model_uri = os.environ["MODEL_URI"]
if "/exports/global_step_" in model_uri:
    verify_completed_hf_model_export(model_uri)
destination = Path("/tmp/model/drift")
copy(model_uri, str(destination), recursive=True)
with (destination / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
expected = set(index["weight_map"].values())
actual = {path.name for path in destination.glob("*.safetensors")}
if actual != expected:
    raise ValueError(f"incomplete model staging: expected {len(expected)} shards, got {len(actual)}")
print(f"Staged {len(actual)} shards from {model_uri}")
PY

responses=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/drift/inputs
root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/drift
for family in P_chat P_think; do
  if [[ "$mode" == reference ]]; then
    python -m experiments.post_training.phase1b.score_native_responses reference \
      --model-path /tmp/model/drift --reference-set "$responses/$family.jsonl" \
      --out "$root/B/$family"
  else
    python -m experiments.post_training.phase1b.score_native_responses candidate \
      --model-path /tmp/model/drift --reference-set "$responses/$family.jsonl" \
      --candidate-set "$responses/$family.jsonl" --support-root "$root/B/$family" \
      --out "$root/$condition/step_$step/$family"
  fi
done
