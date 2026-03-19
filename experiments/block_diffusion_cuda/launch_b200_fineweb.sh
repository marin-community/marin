#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <baseline|bigdn> [smoke|full] [extra train args...]" >&2
  exit 1
fi

VARIANT="$1"
MODE="${2:-smoke}"

if [[ "$MODE" != "smoke" ]]; then
  shift 2
else
  shift 1
  if [[ $# -gt 0 && "$1" == "smoke" ]]; then
    shift 1
  fi
fi

case "$VARIANT" in
  baseline|bigdn)
    ;;
  *)
    echo "variant must be 'baseline' or 'bigdn', got: $VARIANT" >&2
    exit 1
    ;;
esac

case "$MODE" in
  smoke)
    STEPS="${STEPS:-20}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    D_MODEL="${D_MODEL:-512}"
    N_HEADS="${N_HEADS:-8}"
    GDN_HEADS="${GDN_HEADS:-8}"
    N_LAYERS="${N_LAYERS:-8}"
    ;;
  full)
    STEPS="${STEPS:-1000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    D_MODEL="${D_MODEL:-1024}"
    N_HEADS="${N_HEADS:-16}"
    GDN_HEADS="${GDN_HEADS:-16}"
    N_LAYERS="${N_LAYERS:-16}"
    ;;
  *)
    echo "mode must be 'smoke' or 'full', got: $MODE" >&2
    exit 1
    ;;
esac

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATASET="${DATASET:-fineweb}"
TOKENIZER="${TOKENIZER:-gpt2}"
DEVICE="${DEVICE:-cuda}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
WINDOW_BLOCKS="${WINDOW_BLOCKS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-block-diffusion-bigdn}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${VARIANT}-${MODE}-b200-fineweb}"

exec torchrun --nproc_per_node="$NPROC_PER_NODE" -m experiments.block_diffusion_cuda.train \
  --dataset "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --variant "$VARIANT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --block-size "$BLOCK_SIZE" \
  --window-blocks "$WINDOW_BLOCKS" \
  --d-model "$D_MODEL" \
  --n-heads "$N_HEADS" \
  --gdn-heads "$GDN_HEADS" \
  --n-layers "$N_LAYERS" \
  --num-workers "$NUM_WORKERS" \
  --streaming \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "$@"
