#!/usr/bin/env bash
# Kelp v7 training: overnight_cpu model on GPU with Stack Edu + prompt conditioning.
#
# Strategy: use the small (10M param) model that converges fast, but train on
# a much richer dataset (Stack Edu educational Python) with prompt conditioning.
# On a single GPU this should achieve hundreds of thousands of steps overnight.
#
# Two-phase workflow:
#   Phase 1: Prepare corpus (~10-30 min depending on network)
#   Phase 2: Train model (overnight)
#
# Usage:
#   # Full pipeline (prepare + train):
#   bash experiments/kelp/train_v7.sh
#
#   # Skip corpus prep if already done:
#   bash experiments/kelp/train_v7.sh --skip-prep

set -euo pipefail

CORPUS_FILE="experiments/kelp/corpus_v7.txt"
OUTPUT_DIR="checkpoints/kelp-edit-v7"
STACK_EDU_MAX=50000
MAX_LENGTH=512
STEPS=50000
BATCH_SIZE=16
LR=0.001
CHECKPOINT_INTERVAL=5000
SEED=42
P_PROMPT=0.5
SKIP_PREP=false

for arg in "$@"; do
    case "$arg" in
        --skip-prep) SKIP_PREP=true ;;
    esac
done

# Phase 1: Prepare corpus with Stack Edu.
if [ "$SKIP_PREP" = false ]; then
    echo "=== Phase 1: Preparing corpus with Stack Edu ==="
    uv run python experiments/kelp/prepare_corpus.py \
        --output "$CORPUS_FILE" \
        --max-length "$MAX_LENGTH" \
        --stack-edu-max "$STACK_EDU_MAX" \
        --seed "$SEED"
    echo ""
fi

# Phase 2: Train with prompt conditioning.
echo "=== Phase 2: Training v7 (overnight_cpu model on GPU, prompt conditioning) ==="
uv run python experiments/kelp/train.py \
    --preset overnight_cpu \
    --corpus-file "$CORPUS_FILE" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --augment \
    --prompt-conditioning \
    --p-prompt "$P_PROMPT" \
    --corruption-curriculum linear \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --log-interval 10 \
    --seed "$SEED"
