#!/usr/bin/env bash
# CatCount on-policy distillation from synthetic expert teachers, inside one Iris H100x4 task.
# Two GPUs train Qwen2.5-0.5B-Instruct with Megatron, two serve vLLM rollouts, and each teacher is a CPU process
# on this host. TEACHER_WORDS="cat" is single-teacher OPD; "cat evens" is MOPD with one expert per route (cat
# repeats a word, evens lists even numbers), and SWAP_ROUTES=1 sends each route to the other's expert. SPEC gates the run (by default the OPD or MOPD spec);
# set SPEC= (empty) to only train, for calibration.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-cat-count-opd-env}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
POLICY_REVISION="${POLICY_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
OUTPUT="${OUTPUT:-$PWD/cat-count-opd}"
LOG="${LOG:-$PWD/cat-count-opd-run.log}"
TEACHER_WORDS="${TEACHER_WORDS:-cat}"
SWAP_ROUTES="${SWAP_ROUTES:-0}"
read -r -a WORDS <<< "$TEACHER_WORDS"
for word in "${WORDS[@]}"; do
  if [[ "$word" != cat && "$word" != evens ]]; then echo "GPU routes are cat and evens; got '$word'" >&2; exit 2; fi
done
if (( ${#WORDS[@]} > 1 )); then DEFAULT_SPEC=cat-evens-mopd-qwen2.5-0.5b-async.json
else DEFAULT_SPEC=cat-count-opd-qwen2.5-0.5b-async.json; fi
SPEC="${SPEC-ci/marin_nightly/specs/$DEFAULT_SPEC}"
MAX_STEPS="${MAX_STEPS:-30}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
SEED="${SEED:-17}"

# An explicitly empty value disables clipping; an unset value uses the calibrated bound.
ADVANTAGE_CLIP="${ADVANTAGE_CLIP-5}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-none}"
TEACHER_JITTER="${TEACHER_JITTER:-0.5}"
TEACHER_FLAGS="${TEACHER_FLAGS:-}"
FIRST_TEACHER_PORT=18080
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$NIGHTLY_RL_ENV" production

cd "$REPOSITORY_ROOT/skyrl-train"
export PYTHONPATH="$REPOSITORY_ROOT/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM=0

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

MODEL_DIR=$("$PYTHON" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' \
  "$POLICY_MODEL" "$POLICY_REVISION")
echo "::: policy ${POLICY_MODEL}@${POLICY_REVISION} at ${MODEL_DIR}"

teacher_pids=()
cleanup_teachers() {
  for pid in "${teacher_pids[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup_teachers EXIT
TEACHER_ARGS=()
for index in "${!WORDS[@]}"; do
  word="${WORDS[$index]}"
  port=$((FIRST_TEACHER_PORT + index))
  # shellcheck disable=SC2086
  if [[ "$word" == evens ]]; then TARGET_ARGS=(--task evens); else TARGET_ARGS=(--task count --word "$word"); fi
  "$PYTHON" examples/cat_count/synthetic_teacher.py --tokenizer "$MODEL_DIR" --port "$port" "${TARGET_ARGS[@]}" \
    --jitter "$TEACHER_JITTER" $TEACHER_FLAGS &
  teacher_pids+=($!)
  "$PYTHON" - "$port" <<'PY'
import sys
import time
import urllib.error
import urllib.request

for _ in range(300):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}", timeout=1)
    except urllib.error.HTTPError:
        break
    except OSError:
        time.sleep(0.1)
else:
    raise RuntimeError("the synthetic teacher did not start")
PY
  TEACHER_ARGS+=(--teacher "${word}=http://127.0.0.1:${port}/v1")
done
if [[ "$SWAP_ROUTES" == 1 ]]; then TEACHER_ARGS+=(--swap-routes); fi

echo "::: distilling for ${MAX_STEPS} steps at lr ${LEARNING_RATE}, advantage clip ${ADVANTAGE_CLIP:-none}; experts ${TEACHER_WORDS} (swap ${SWAP_ROUTES}), jitter ${TEACHER_JITTER} ${TEACHER_FLAGS}"
START=$(date +%s)
# shellcheck disable=SC2086
"$PYTHON" -m cloud.iris.telemetry_env -- \
  "$PYTHON" examples/cat_count/gpu_opd.py \
  --model "$MODEL_DIR" --output "$OUTPUT" \
  "${TEACHER_ARGS[@]}" \
  --teacher-revision "jitter${TEACHER_JITTER}${TEACHER_FLAGS// /}" \
  --steps "$MAX_STEPS" --lr "$LEARNING_RATE" --seed "$SEED" \
  --advantage-clip "$ADVANTAGE_CLIP" \
  2>&1 | tee "$LOG"
ELAPSED=$(( $(date +%s) - START ))

if [[ -n "$SPEC" ]]; then
  echo "::: gating (run took ${ELAPSED}s)"
  "$PYTHON" -m ci.marin_nightly.gate --log "$LOG" --spec "$SPEC" --wall-clock-seconds "$ELAPSED"
else
  echo "::: no SPEC set; run took ${ELAPSED}s"
fi
