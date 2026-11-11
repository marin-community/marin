#!/usr/bin/env bash
set -euo pipefail

pass=${1:?pass reference, baseline, or candidate}
root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.10/candidates/s38/common-pj
case "$pass" in
  reference)
    model_uri=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/T1-b4558699
    ;;
  baseline)
    model_uri=s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192
    candidate=B
    ;;
  candidate)
    model_uri=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models/S38-cfc1d845
    candidate=S38
    ;;
  *) echo "unexpected common-context pass: $pass" >&2; exit 2 ;;
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

if [[ "$pass" == reference ]]; then
  python -m experiments.post_training.phase1b.score_native_responses reference \
    --model-path /tmp/model/current --reference-set "$root/T1.jsonl" --out "$root/support"
else
  python -m experiments.post_training.phase1b.score_native_responses candidate \
    --model-path /tmp/model/current --reference-set "$root/T1.jsonl" \
    --candidate-set "$root/$candidate.jsonl" --support-root "$root/support" \
    --out "$root/kl-t1-to-$candidate"
fi
