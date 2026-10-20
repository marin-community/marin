#!/usr/bin/env bash
set -euo pipefail

PARENT=${1:?parent: v0 or v0.1}
ARM=${2:?arm: plain, rs-multi, relay, recovery or plateau}
ROWS=${3:?production rows prefix}
SHA=${4:?manifest sha256}
PLAIN=${5:?production plain prefix for this parent}
RUN_DATE=${6:?run date YYYYMMDD}
MODE=${7:-}
case "$PARENT" in v0|v0.1) ;; *) echo "Unknown parent: $PARENT" >&2; exit 1 ;; esac
case "$ARM" in plain|rs-multi|relay|recovery|plateau) ;; *) echo "Unknown arm: $ARM" >&2; exit 1 ;; esac
case "$MODE" in ""|--run) ;; *) echo "Seventh argument must be --run or omitted" >&2; exit 1 ;; esac
CLUSTER=${CLUSTER:-cw-rno2a}
case "$CLUSTER" in cw-rno2a|cw-us-east-02a) ;; *) echo "Unsupported cluster: $CLUSTER" >&2; exit 1 ;; esac

cd "$(dirname "$0")/../../.."
unset GOOGLE_APPLICATION_CREDENTIALS
export MARIN_PREFIX=s3://marin-us-east-02a/marin
export NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=
COMMAND=(python -m experiments.post_training.agentic_sweep.arms
  --parent "$PARENT" --run-date "$RUN_DATE"
  --arm-data "$ROWS" --manifest-sha256 "$SHA" --plain-data "$PLAIN"
  --learning-rate 1e-6 --epochs 3 --only train
  --preemption-retries "${PREEMPTION_RETRIES:-2}")
if [[ "$ARM" == plateau ]]; then
  COMMAND+=(--arm plain --plateau)
else
  COMMAND+=(--arm "$ARM")
fi
NAME="arm-train-$ARM-${PARENT//./p}-v7-$RUN_DATE-$CLUSTER"
if [[ -n "${RESUME:-}" ]]; then
  COMMAND+=(--resume-checkpoint "$RESUME")
  NAME="$NAME-from${RESUME##*step-}"
fi
if [[ -z "$MODE" ]]; then
  exec uv run --frozen "${COMMAND[@]}"
fi

test -z "$(git status --porcelain -- experiments lib)"
: "${CW_KEY_ID:?Set credentials in the environment}"
: "${CW_KEY_SECRET:?Set credentials in the environment}"
: "${HF_TOKEN:?Set credentials in the environment}"
: "${WANDB_API_KEY:?Set credentials in the environment}"
date -u
exec uv run --frozen iris --cluster=marin job run \
  --target-cluster "$CLUSTER" --enable-extra-resources \
  --cpu 4 --memory 16GB --disk 32GB --priority interactive --max-retries 0 \
  --job-name "$NAME" --no-wait \
  -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET" \
  -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX "$MARIN_PREFIX" -e NCCL_DEBUG WARN -e NCCL_DEBUG_SUBSYS "" \
  -- "${COMMAND[@]}" --run
