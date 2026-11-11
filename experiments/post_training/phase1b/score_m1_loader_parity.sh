#!/usr/bin/env bash
set -euo pipefail

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python -c 'from experiments.post_training.phase1b.score_train_rows import candidate_pass; assert candidate_pass'

root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.7/diagnostics/m1-loader-parity
input=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.6/diagnostics/train200/R_train_8_short.jsonl
model=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/M1-bc318121

python - <<'PY'
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
destination = Path("/tmp/model/current")
shutil.rmtree(destination, ignore_errors=True)
copy(
    "s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/M1-bc318121",
    str(destination),
    recursive=True,
)
if len(list(destination.glob("*.safetensors"))) != 39:
    raise ValueError("M1 model staging did not yield all 39 weight shards")
PY

python -m experiments.post_training.phase1b.score_train_rows reference \
  --model-path /tmp/model/current --input "$input" --out "$root/support/M1"
python -m experiments.post_training.phase1b.score_train_rows candidate \
  --model-path /tmp/model/current --input "$input" --support-root "$root/support/M1" --out "$root/scores/M1"
