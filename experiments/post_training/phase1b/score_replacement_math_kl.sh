#!/usr/bin/env bash
set -euo pipefail

variant=${1:?pass s92, s38, datakit, or antidoom}
pass=${2:?pass reference, B, or T1}
mirror_root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models
result_root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.10/candidates/$variant
case "$variant" in
  s92) expert_uri=$mirror_root/S92-fdc3c3c5 ;;
  s38) expert_uri=$mirror_root/S38-cfc1d845 ;;
  datakit) expert_uri=$mirror_root/Datakit-8c76b22a ;;
  antidoom) expert_uri=$mirror_root/Antidoom-fd07ef0f ;;
  *) echo "unexpected math candidate: $variant" >&2; exit 2 ;;
esac
case "$pass" in
  reference) model_uri=$expert_uri ;;
  B) model_uri=s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192 ;;
  T1) model_uri=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/T1-b4558699 ;;
  *) echo "unexpected math KL pass: $pass" >&2; exit 2 ;;
esac
export MODEL_URI=$model_uri

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
destination = Path("/tmp/model/current")
shutil.rmtree(destination, ignore_errors=True)
copy(os.environ["MODEL_URI"], str(destination), recursive=True)
with (destination / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
expected = set(index["weight_map"].values())
actual = {path.name for path in destination.glob("*.safetensors")}
if actual != expected:
    raise ValueError(f"incomplete model staging: expected {len(expected)} shards, got {len(actual)}")
print(f"Staged {len(actual)} safetensors shards from {os.environ['MODEL_URI']}")
PY

reference_set=$result_root/continuations/P_R.jsonl
support_root=$result_root/math-kl/support
if [[ "$pass" == reference ]]; then
  python -m experiments.post_training.phase1b.score_native_responses reference \
    --model-path /tmp/model/current --reference-set "$reference_set" --out "$support_root"
else
  python -m experiments.post_training.phase1b.score_native_responses candidate \
    --model-path /tmp/model/current --reference-set "$reference_set" \
    --candidate-set "$result_root/aligned-$pass/P_R.jsonl" --support-root "$support_root" \
    --out "$result_root/math-kl/to-$pass"
fi
