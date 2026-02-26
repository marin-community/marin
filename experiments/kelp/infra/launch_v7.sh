#!/usr/bin/env bash
# Launch Kelp v7 training + eval on a single A100 via SkyPilot.
#
# This script:
# 1. Launches training on an A100 (corpus prep + 50k steps)
# 2. Runs corpus and MBPP evals on the same cluster
# 3. Downloads checkpoints and results locally
# 4. Tears down the cluster
#
# Prerequisites:
#   pip install "skypilot[lambda]"
#   sky check  # verify Lambda credentials
#
# Usage:
#   bash experiments/kelp/infra/launch_v7.sh
#
# With W&B logging:
#   bash experiments/kelp/infra/launch_v7.sh --wandb
#
# To skip teardown (keep cluster for debugging):
#   bash experiments/kelp/infra/launch_v7.sh --keep

set -euo pipefail

CLUSTER_NAME="kelp-v7"
TRAIN_YAML="experiments/kelp/infra/kelp-v7-train.yaml"
EVAL_YAML="experiments/kelp/infra/kelp-v7-eval.yaml"
LOCAL_CKPT_DIR="checkpoints/kelp-edit-v7"
KEEP_CLUSTER=false
USE_WANDB=false

for arg in "$@"; do
    case "$arg" in
        --keep) KEEP_CLUSTER=true ;;
        --wandb) USE_WANDB=true ;;
    esac
done

# Build env flags for sky launch.
ENV_FLAGS=""
if [ "$USE_WANDB" = true ]; then
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "ERROR: --wandb requires WANDB_API_KEY to be set"
        echo "  export WANDB_API_KEY=your-key"
        exit 1
    fi
    ENV_FLAGS="--env WANDB_API_KEY"
    echo "W&B logging: enabled (project=kelp, run=kelp-v7-prompt-conditioning)"
fi

echo "=============================================="
echo " Kelp v7: SkyPilot Training + Eval Pipeline"
echo "=============================================="
echo ""
echo "Cluster:  ${CLUSTER_NAME}"
echo "Training: ${TRAIN_YAML}"
echo "Eval:     ${EVAL_YAML}"
echo ""

# Step 1: Launch training.
echo ">>> Step 1: Launching training on A100..."
sky launch -c "$CLUSTER_NAME" "$TRAIN_YAML" $ENV_FLAGS --retry-until-up -y

echo ""
echo ">>> Step 1 complete: Training finished."
echo ""

# Step 2: Run eval on the same cluster (checkpoints already on disk).
echo ">>> Step 2: Running evaluations..."
sky exec "$CLUSTER_NAME" "$EVAL_YAML"

echo ""
echo ">>> Step 2 complete: Evaluations finished."
echo ""

# Step 3: Download results (from cluster via rsync, with S3 fallback).
echo ">>> Step 3: Downloading checkpoints and results..."
mkdir -p "$LOCAL_CKPT_DIR"

if rsync -avz "${CLUSTER_NAME}:~/sky_workdir/checkpoints/kelp-edit-v7/" "$LOCAL_CKPT_DIR/" 2>/dev/null; then
    rsync -avz "${CLUSTER_NAME}:~/sky_workdir/experiments/kelp/corpus_v7.txt" "experiments/kelp/corpus_v7.txt" 2>/dev/null || true
    echo ">>> Downloaded from cluster via rsync."
else
    echo ">>> Cluster rsync failed, downloading from S3..."
    aws s3 sync "s3://oa-fomo-outputs/kelp/kelp-edit-v7/" "$LOCAL_CKPT_DIR/"
    aws s3 cp "s3://oa-fomo-outputs/kelp/corpus_v7.txt" "experiments/kelp/corpus_v7.txt" 2>/dev/null || true
    echo ">>> Downloaded from S3."
fi

echo ""
echo ">>> Step 3 complete: Results downloaded to ${LOCAL_CKPT_DIR}/"
echo ">>> Results also persisted at s3://oa-fomo-outputs/kelp/"
echo ""

# Step 4: Teardown (unless --keep).
if [ "$KEEP_CLUSTER" = true ]; then
    echo ">>> Keeping cluster ${CLUSTER_NAME} alive (--keep flag set)."
    echo "    To tear down later: sky down ${CLUSTER_NAME} -y"
else
    echo ">>> Step 4: Tearing down cluster..."
    sky down "$CLUSTER_NAME" -y
    echo ">>> Cluster ${CLUSTER_NAME} terminated."
fi

echo ""
echo "=============================================="
echo " Done! Check results:"
echo "   Local: ${LOCAL_CKPT_DIR}/"
echo "   S3:    s3://oa-fomo-outputs/kelp/kelp-edit-v7/"
echo ""
echo "   Key files:"
echo "     training.log"
echo "     eval.log"
echo "     corpus_eval_results.json"
echo "     mbpp_eval_results.json"
echo "=============================================="
