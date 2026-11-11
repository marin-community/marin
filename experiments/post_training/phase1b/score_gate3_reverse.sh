#!/usr/bin/env bash
set -euo pipefail

condition=${1:?pass theta_R, theta_J, or theta_RJ}
step=${2:?pass a verified export step}
phase=${3:-all}
case "$condition" in theta_R|theta_J|theta_RJ) ;; *) echo "unexpected condition: $condition" >&2; exit 2;; esac
case "$step" in 10|30|50|100|125) ;; *) echo "unexpected export step: $step" >&2; exit 2;; esac
case "$phase" in all|math|terminal|terminal_only) ;; *) echo "unexpected phase: $phase" >&2; exit 2;; esac

version=2026.11.10.14
if [[ "$condition" == theta_J ]]; then version=2026.11.10.13; fi
export MODEL_URI="s3://marin-rno2a/marin/users/benfeuer/checkpoints/phase1b/sft-${condition}-lr2e-06-seed0/${version}/exports/global_step_${step}/policy"

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

stage_model() {
  export STAGE_URI=$1
  python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy
from skyrl_train.hf_model_io import verify_completed_hf_model_export

configure_coreweave_s3()
if os.environ["STAGE_URI"] == os.environ["MODEL_URI"]:
    verify_completed_hf_model_export(os.environ["STAGE_URI"])
destination = Path("/tmp/model/reverse")
shutil.rmtree(destination, ignore_errors=True)
copy(os.environ["STAGE_URI"], str(destination), recursive=True)
with (destination / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
expected = set(index["weight_map"].values())
actual = {path.name for path in destination.glob("*.safetensors")}
if actual != expected:
    raise ValueError(f"incomplete model staging: expected {len(expected)} shards, got {len(actual)}")
print(f"Staged {len(actual)} shards from {os.environ['STAGE_URI']}")
PY
}

base="s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/reverse/${condition}/step_${step}"
export BASE=$base
if [[ "$phase" == all || "$phase" == math || "$phase" == terminal_only ]]; then
  stage_model "$MODEL_URI"
fi
if [[ "$phase" == all || "$phase" == math ]]; then
  python -m experiments.post_training.phase1b.score_native_responses reference \
    --model-path /tmp/model/reverse --reference-set "$base/common-S38/theta.jsonl" \
    --out "$base/support/P_R"
fi
if [[ "$phase" == all || "$phase" == terminal_only ]]; then
  python -m experiments.post_training.phase1b.score_native_responses reference \
    --model-path /tmp/model/reverse --reference-set "$base/continuations/P_J.jsonl" \
    --out "$base/support/P_J"
fi

if [[ "$phase" == all || "$phase" == math ]]; then
  stage_model s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models/S38-cfc1d845
  python -m experiments.post_training.phase1b.score_native_responses candidate \
    --model-path /tmp/model/reverse --reference-set "$base/common-S38/theta.jsonl" \
    --candidate-set "$base/common-S38/S38.jsonl" --support-root "$base/support/P_R" \
    --out "$base/kl-theta-to-S38/P_R"
fi
if [[ "$phase" == math ]]; then
  exit 0
fi
if [[ "$phase" == terminal ]]; then
  python - <<'PY'
import json
import os
import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()
root = f"{os.environ['BASE']}/support/P_J"
with fsspec.open(f"{root}/reference.json") as handle:
    marker = json.load(handle)
if marker["rows"] != 1000:
    raise ValueError(f"Incomplete terminal reference support: {marker}")
filesystem, path = fsspec.core.url_to_fs(root)
shards = [item for item in filesystem.ls(path) if item.endswith(".npz")]
if not shards:
    raise ValueError(f"No terminal reference-support shards at {root}")
PY
fi

stage_model s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/T1-b4558699
python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/reverse --reference-set "$base/continuations/P_J.jsonl" \
  --candidate-set "$base/aligned-T1/P_J.jsonl" --support-root "$base/support/P_J" \
  --out "$base/kl-theta-to-T1/P_J"
