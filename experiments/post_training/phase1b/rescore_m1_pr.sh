#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?pass reference, B, T1, M1-on-P_J, or a theta_R export step}
root=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.7
reference_input=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.27/math-full/M1/P_R.jsonl
candidate_input=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.43/inputs/M1-P_R/aligned-B/P_R.jsonl
support=$root/klfull/M1/P_R/reference

case "$VARIANT" in
  reference)
    model_uri=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/M1-bc318121
    ;;
  B)
    model_uri=s3://marin-rno2a/marin/checkpoints/stage0-snowball/s3-terminal/2026.10.22/hf/step-1192
    ;;
  T1)
    model_uri=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/T1-b4558699
    candidate_input=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.43/inputs/M1-P_R/aligned-T1/P_R.jsonl
    ;;
  M1-on-P_J)
    model_uri=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.1/models/M1-bc318121
    reference_input=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.22/probes/P_J-T1.jsonl
    candidate_input=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.22/probes/P_J-M1.jsonl
    support=s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.36/klfull/T1/P_J/reference
    ;;
  10|30|50|100|125)
    model_uri=s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/sft-theta_R-lr2e-06-seed0/2026.11.09.3/exports/global_step_${VARIANT}/policy
    ;;
  *)
    echo "unexpected scorer variant: $VARIANT" >&2
    exit 2
    ;;
esac
export MODEL_URI=$model_uri

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python - <<'PY'
import os
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy

configure_coreweave_s3()
destination = Path("/tmp/model/current")
copy(os.environ["MODEL_URI"], str(destination), recursive=True)
if len(list(destination.glob("*.safetensors"))) != 39:
    raise ValueError(f"expected 39 weight shards at {destination}")
PY

if [ "$VARIANT" = reference ]; then
  python -m experiments.post_training.phase1b.score_native_responses reference \
    --model-path /tmp/model/current --reference-set "$reference_input" --out "$support"
else
  if [ "$VARIANT" = M1-on-P_J ]; then
    output=$root/scores/M1-P_J
  else
    output=$root/scores/M1-P_R/$VARIANT
  fi
  python -m experiments.post_training.phase1b.score_native_responses candidate \
    --model-path /tmp/model/current --reference-set "$reference_input" \
    --candidate-set "$candidate_input" --support-root "$support" \
    --out "$output"
fi
