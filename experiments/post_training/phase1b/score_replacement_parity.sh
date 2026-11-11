#!/usr/bin/env bash
set -euo pipefail

variant=${1:?pass s92, s38, datakit, or antidoom}
mirror_root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8
parity_root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.10.9
case "$variant" in
  s92) model_uri=$mirror_root/models/S92-fdc3c3c5 ;;
  s38) model_uri=$mirror_root/models/S38-cfc1d845 ;;
  datakit) model_uri=$mirror_root/models/Datakit-8c76b22a ;;
  antidoom) model_uri=$mirror_root/models/Antidoom-fd07ef0f ;;
  *) echo "unexpected math candidate: $variant" >&2; exit 2 ;;
esac
export MODEL_URI=$model_uri

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python -c 'from experiments.post_training.phase1b.score_train_rows import candidate_pass; assert candidate_pass'
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

input=$parity_root/parity/$variant/short8.jsonl
support=$parity_root/parity/$variant/megatron/support
scores=$parity_root/parity/$variant/megatron/scores
python -m experiments.post_training.phase1b.score_native_responses reference \
  --model-path /tmp/model/current --reference-set "$input" --out "$support"
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$support" --out "$scores"
