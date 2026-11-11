#!/usr/bin/env bash
set -euo pipefail

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python -c 'from experiments.post_training.phase1b.score_train_rows import candidate_pass; assert candidate_pass'

root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.6/diagnostics/train200
input=$root/R_train_200.jsonl
support=$root/support/M1

stage_model() {
  export MODEL_URI=$1
  python - <<'PY'
import os
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
destination = Path("/tmp/model/current")
shutil.rmtree(destination, ignore_errors=True)
copy(os.environ["MODEL_URI"], str(destination), recursive=True)
if not (destination / "model.safetensors.index.json").is_file():
    raise ValueError(f"model index missing after staging {os.environ['MODEL_URI']}")
if len(list(destination.glob("*.safetensors"))) != 39:
    raise ValueError(f"expected 39 model shards at {destination}")
print(f"Staged {os.environ['MODEL_URI']}")
PY
}

stage_model s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/M1-bc318121
python -m experiments.post_training.phase1b.score_train_rows reference \
  --model-path /tmp/model/current --input "$input" --out "$support"
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$support" --out "$root/scores/M1"

stage_model s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$support" --out "$root/scores/B"

stage_model s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/T1-b4558699
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$support" --out "$root/scores/T1"

stage_model s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/sft-theta_R-lr2e-06-seed0/2026.11.09.3/exports/global_step_30/policy
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$support" --out "$root/scores/theta_R_s30"

stage_model s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/sft-theta_R-lr2e-06-seed0/2026.11.09.3/exports/global_step_125/policy
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$support" --out "$root/scores/theta_R_s125"
