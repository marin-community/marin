#!/usr/bin/env bash
set -euo pipefail

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

panel=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/reverse/B-P_R/common-S38
output=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.15/reverse/B-to-S38

stage_model() {
  export MODEL_URI=$1
  python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
destination = Path("/tmp/model/reverse")
shutil.rmtree(destination, ignore_errors=True)
copy(os.environ["MODEL_URI"], str(destination), recursive=True)
with (destination / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
expected = set(index["weight_map"].values())
actual = {path.name for path in destination.glob("*.safetensors")}
if actual != expected:
    raise ValueError(f"incomplete model staging: expected {len(expected)} shards, got {len(actual)}")
print(f"Staged {len(actual)} shards from {os.environ['MODEL_URI']}")
PY
}

stage_model s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192
python -m experiments.post_training.phase1b.score_native_responses reference \
  --model-path /tmp/model/reverse --reference-set "$panel/B.jsonl" --out "$output/support"

stage_model s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models/S38-cfc1d845
python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/reverse --reference-set "$panel/B.jsonl" \
  --candidate-set "$panel/S38.jsonl" --support-root "$output/support" \
  --out "$output/kl-b-to-s38"
