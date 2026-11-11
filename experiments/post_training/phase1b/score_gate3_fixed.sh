#!/usr/bin/env bash
set -euo pipefail

condition=${1:?pass theta_R, theta_J, or theta_RJ}
step=${2:?pass a verified export step}
case "$condition" in theta_R|theta_J|theta_RJ) ;; *) echo "unexpected condition: $condition" >&2; exit 2;; esac
case "$step" in 10|30|50|100|125) ;; *) echo "unexpected export step: $step" >&2; exit 2;; esac

version=2026.11.10.14
if [[ "$condition" == theta_J ]]; then version=2026.11.10.13; fi
export MODEL_URI="s3://marin-rno2a/marin/users/benfeuer/checkpoints/phase1b/sft-${condition}-lr2e-06-seed0/${version}/exports/global_step_${step}/policy"

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
verify_completed_hf_model_export(os.environ["MODEL_URI"])
destination = Path("/tmp/model/gate3")
copy(os.environ["MODEL_URI"], str(destination), recursive=True)
with (destination / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
expected = set(index["weight_map"].values())
actual = {path.name for path in destination.glob("*.safetensors")}
if actual != expected:
    raise ValueError(f"incomplete model staging: expected {len(expected)} shards, got {len(actual)}")
print(f"Staged {len(actual)} verified shards from {os.environ['MODEL_URI']}")
PY

root="s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/scores/${condition}/step_${step}"
math="s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.10/candidates/s38"
base="s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b"
python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/gate3 \
  --reference-set "$math/continuations/P_R.jsonl" \
  --candidate-set "$math/aligned-B/P_R.jsonl" \
  --support-root "$math/math-kl/support" \
  --out "$root/P_R"

python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/gate3 \
  --reference-set "$base/2026.11.08.22/probes/P_J-T1.jsonl" \
  --candidate-set "$base/2026.11.08.22/probes/P_J-B.jsonl" \
  --support-root "$base/2026.11.08.36/klfull/T1/P_J/reference" \
  --out "$root/P_J"
