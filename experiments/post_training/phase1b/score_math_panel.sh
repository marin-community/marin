#!/usr/bin/env bash
set -euo pipefail

model_uri=${1:?pass the scored model URI}
input=${2:?pass the aligned response panel root}
output=${3:?pass the score output root}
case "$model_uri" in
  s3://marin-rno2a/*|s3://marin-us-east-08a/*) ;;
  *) echo "expected CoreWeave model URI: $model_uri" >&2; exit 2 ;;
esac
for uri in "$input" "$output"; do
  [[ "$uri" == s3://marin-us-east-08a/* ]] || { echo "expected marin-us-east-08a output URI: $uri" >&2; exit 2; }
done

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
uri = os.environ["STAGE_URI"]
if "/exports/global_step_" in uri:
    verify_completed_hf_model_export(uri)
destination = Path("/tmp/model/matched-math")
shutil.rmtree(destination, ignore_errors=True)
copy(uri, str(destination), recursive=True)
with (destination / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
expected = set(index["weight_map"].values())
actual = {path.name for path in destination.glob("*.safetensors")}
if actual != expected:
    raise ValueError(f"incomplete model staging: expected {len(expected)} shards, got {len(actual)}")
print(f"Staged {len(actual)} shards from {uri}")
PY
}

stage_model "$model_uri"
python -m experiments.post_training.phase1b.score_native_responses reference \
  --model-path /tmp/model/matched-math --reference-set "$input/reference.jsonl" \
  --out "$output/support"

stage_model s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.8/models/S38-cfc1d845
python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/matched-math --reference-set "$input/reference.jsonl" \
  --candidate-set "$input/S38.jsonl" --support-root "$output/support" \
  --out "$output/kl-to-S38"
