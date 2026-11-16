#!/usr/bin/env bash
# One real synchronous OPD step with a separate fixed vLLM teacher. This runs inside
# one H100x4 Iris task: policy, rollout, and teacher each own one GPU.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-opd-env}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3-0.6B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-1.7B}"
TEACHER_REVISION="${TEACHER_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
TEACHER_SOURCE="${TEACHER_SOURCE:-local_inference}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_opd_nightly}"
LOG="${LOG:-$PWD/opd-nightly-run.log}"
SPEC="${SPEC:-ci/marin_nightly/specs/opd-qwen3-sync.json}"
# The nightly takes one synchronous step. An asynchronous smoke sets MAX_STALENESS_STEPS >= 1 and MAX_STEPS >= 2,
# so some trained groups come from an older policy than the one they train. The 8-row slice allows 4 steps.
MAX_STEPS="${MAX_STEPS:-1}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-0}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$NIGHTLY_RL_ENV" production

cd "$REPOSITORY_ROOT/skyrl-train"
rm -rf skyrl-gym
cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM=0

TEACHER_ARGS=()
teacher_pid=""
cleanup_teacher() {
  if [[ -n "$teacher_pid" ]]; then
    kill "$teacher_pid" >/dev/null 2>&1 || true
    wait "$teacher_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup_teacher EXIT

case "$TEACHER_SOURCE" in
  local_inference)
    TEACHER_ARGS=(--teacher-source local_inference)
    ;;
  openai_compatible)
    TOKENIZER_FINGERPRINT=$("$PYTHON" -c \
      'import sys; from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint; from skyrl_train.tokenizer import create_tokenizer; print(tokenizer_vocabulary_fingerprint(create_tokenizer(sys.argv[1], disable_fast_tokenizer=False)))' \
      "$POLICY_MODEL")
    "$PYTHON" tests/fixtures/opd_http_teacher.py --port 18080 &
    teacher_pid=$!
    "$PYTHON" - <<'PY'
import time
import urllib.error
import urllib.request

for _ in range(100):
    try:
        urllib.request.urlopen("http://127.0.0.1:18080", timeout=1)
    except urllib.error.HTTPError:
        break
    except OSError:
        time.sleep(0.1)
else:
    raise RuntimeError("remote teacher fixture did not start")
PY
    TEACHER_ARGS=(--teacher-source openai_compatible --tokenizer-fingerprint "$TOKENIZER_FINGERPRINT")
    ;;
  *)
    echo "unsupported TEACHER_SOURCE: $TEACHER_SOURCE" >&2
    exit 2
    ;;
esac

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv
test -x "$PYTHON"
"$PYTHON" -c "import torch, vllm; print(f'torch {torch.__version__} | vllm {vllm.__version__}')"

echo "::: preparing a small GSM8K slice"
"$PYTHON" examples/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR"
DATA_DIR="$DATA_DIR" "$PYTHON" - <<'PY'
import os
import pathlib

import polars as pl

data_dir = pathlib.Path(os.environ["DATA_DIR"])
for name, rows in (("train", 8), ("validation", 2)):
    path = data_dir / f"{name}.parquet"
    pl.read_parquet(path).head(rows).write_parquet(path)
PY

echo "::: training ${POLICY_MODEL} for ${MAX_STEPS} teacher-sensitive step(s) with ${TEACHER_MODEL}@${TEACHER_REVISION}"
echo "::: rollouts: max_staleness_steps=${MAX_STALENESS_STEPS}"
START=$(date +%s)
"$PYTHON" -m cloud.iris.telemetry_env -- \
  "$PYTHON" -m ci.marin_nightly.opd \
  --model "$POLICY_MODEL" --data-dir "$DATA_DIR" \
  --teacher-model "$TEACHER_MODEL" --teacher-revision "$TEACHER_REVISION" \
  "${TEACHER_ARGS[@]}" --steps "$MAX_STEPS" --max-staleness-steps "$MAX_STALENESS_STEPS" \
  2>&1 | tee "$LOG"
ELAPSED=$(( $(date +%s) - START ))

echo "::: gating (run took ${ELAPSED}s)"
"$PYTHON" -m ci.marin_nightly.gate \
  --log "$LOG" --spec "$SPEC" --wall-clock-seconds "$ELAPSED"
