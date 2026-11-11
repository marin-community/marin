#!/usr/bin/env bash
set -euo pipefail

STEP=${1:?pass the verified theta_R export step}
case "$STEP" in 10|30|50|100|125) ;; *) echo "unexpected export step: $STEP" >&2; exit 2;; esac
export STEP

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron development
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app

python - <<'PY'
import os

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy
from skyrl_train.hf_model_io import verify_completed_hf_model_export

step = int(os.environ["STEP"])
model_uri = (
    "s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/"
    f"sft-theta_R-lr2e-06-seed0/2026.11.09.3/exports/global_step_{step}/policy"
)
configure_coreweave_s3()
verify_completed_hf_model_export(model_uri)
copy(model_uri, "/tmp/model/theta_R", recursive=True)
print(f"Staged verified theta_R step {step} from {model_uri}")
PY

output_root="s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.09.5/scores/theta_R/step_${STEP}"
python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/theta_R \
  --reference-set s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.27/math-full/M1/P_R.jsonl \
  --candidate-set s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.43/inputs/M1-P_R/aligned-B/P_R.jsonl \
  --support-root s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.42/klfull/M1/P_R/reference \
  --out "$output_root/P_R"

python -m experiments.post_training.phase1b.score_native_responses candidate \
  --model-path /tmp/model/theta_R \
  --reference-set s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.22/probes/P_J-T1.jsonl \
  --candidate-set s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.22/probes/P_J-B.jsonl \
  --support-root s3://marin-rno2a/marin/users/benfeuer/opd-interference/phase1b/2026.11.08.36/klfull/T1/P_J/reference \
  --out "$output_root/P_J"
