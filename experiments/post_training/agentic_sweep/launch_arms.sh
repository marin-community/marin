#!/usr/bin/env bash
set -euo pipefail

PARENT=${1:?parent: v0 or v0.1}
ARM=${2:?arm: plain, rs-multi, relay, recovery or plateau}
ROWS=${3:?production rows prefix}
SHA=${4:?manifest sha256}
PLAIN=${5:?production plain prefix for this parent}
RUN_DATE=${6:?run date YYYYMMDD}
shift 6
MODE=
DATA_VARIANT=high
CONTEXT_LENGTH=32768
HF_EXPORT_SCHEDULE=epoch2-and-final
while (( $# )); do
  case "$1" in
    --run) MODE=--run; shift ;;
    --data-variant) DATA_VARIANT=${2:?data variant}; shift 2 ;;
    --context-length) CONTEXT_LENGTH=${2:?training context length}; shift 2 ;;
    --hf-export-schedule) HF_EXPORT_SCHEDULE=${2:?HF export schedule}; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
case "$PARENT" in v0|v0.1) ;; *) echo "Unknown parent: $PARENT" >&2; exit 1 ;; esac
case "$ARM" in plain|rs-multi|relay|recovery|plateau) ;; *) echo "Unknown arm: $ARM" >&2; exit 1 ;; esac
case "$DATA_VARIANT" in high|low|matched-relay|contributor|lowshared|contribshared) ;; *) echo "Unknown data variant: $DATA_VARIANT" >&2; exit 1 ;; esac
if [[ "$DATA_VARIANT" == matched-relay && "$ARM" != plain ]]; then
  echo "The matched-relay comparator uses plain training" >&2
  exit 1
fi
CLUSTER=${CLUSTER:-cw-rno2a}
case "$CLUSTER" in cw-rno2a|cw-us-east-02a) ;; *) echo "Unsupported cluster: $CLUSTER" >&2; exit 1 ;; esac

cd "$(dirname "$0")/../../.."
unset GOOGLE_APPLICATION_CREDENTIALS
export MARIN_PREFIX=s3://marin-us-east-02a/marin
export NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=
COMMAND=(python -m experiments.post_training.agentic_sweep.arms
  --parent "$PARENT" --run-date "$RUN_DATE" --data-variant "$DATA_VARIANT"
  --context-length "$CONTEXT_LENGTH"
  --hf-export-schedule "$HF_EXPORT_SCHEDULE"
  --arm-data "$ROWS" --manifest-sha256 "$SHA" --plain-data "$PLAIN"
  --learning-rate 1e-6 --epochs 3 --only train
  --preemption-retries "${PREEMPTION_RETRIES:-2}")
if [[ "$ARM" == plateau ]]; then
  COMMAND+=(--arm plain --plateau)
else
  COMMAND+=(--arm "$ARM")
fi
METHOD=$ARM
if [[ "$DATA_VARIANT" != high ]]; then
  METHOD="$METHOD-$DATA_VARIANT"
fi
NAME="arm-train-$METHOD-${PARENT//./p}-v7-$RUN_DATE-$CLUSTER"
if [[ "$CONTEXT_LENGTH" != 32768 ]]; then
  NAME="$NAME-ctx$CONTEXT_LENGTH"
fi
COMMAND+=(--resume-checkpoint "${RESUME:-auto}")
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
  --cpu 4 --memory 16GB --disk 32GB --max-retries 0 \
  --job-name "$NAME" --no-wait \
  -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET" \
  -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX "$MARIN_PREFIX" -e NCCL_DEBUG WARN -e NCCL_DEBUG_SUBSYS "" \
  -- "${COMMAND[@]}" --run
