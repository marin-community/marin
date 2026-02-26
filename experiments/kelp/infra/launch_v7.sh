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
#   pip install "skypilot[gcp]"
#   sky check  # verify GCP credentials
#
# Usage:
#   bash experiments/kelp/infra/launch_v7.sh
#
# To skip teardown (keep cluster for debugging):
#   bash experiments/kelp/infra/launch_v7.sh --keep

set -euo pipefail

CLUSTER_NAME="kelp-v7"
TRAIN_YAML="experiments/kelp/infra/kelp-v7-train.yaml"
EVAL_YAML="experiments/kelp/infra/kelp-v7-eval.yaml"
LOCAL_CKPT_DIR="checkpoints/kelp-edit-v7"
KEEP_CLUSTER=false

for arg in "$@"; do
    case "$arg" in
        --keep) KEEP_CLUSTER=true ;;
    esac
done

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
sky launch -c "$CLUSTER_NAME" "$TRAIN_YAML" -y

echo ""
echo ">>> Step 1 complete: Training finished."
echo ""

# Step 2: Run eval on the same cluster (checkpoints already on disk).
echo ">>> Step 2: Running evaluations..."
sky exec "$CLUSTER_NAME" "$EVAL_YAML"

echo ""
echo ">>> Step 2 complete: Evaluations finished."
echo ""

# Step 3: Download results.
echo ">>> Step 3: Downloading checkpoints and results..."
mkdir -p "$LOCAL_CKPT_DIR"
rsync -avz "${CLUSTER_NAME}:~/sky_workdir/checkpoints/kelp-edit-v7/" "$LOCAL_CKPT_DIR/"

# Also grab the corpus for local eval.
rsync -avz "${CLUSTER_NAME}:~/sky_workdir/experiments/kelp/corpus_v7.txt" "experiments/kelp/corpus_v7.txt"

echo ""
echo ">>> Step 3 complete: Results downloaded to ${LOCAL_CKPT_DIR}/"
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
echo "   ${LOCAL_CKPT_DIR}/training.log"
echo "   ${LOCAL_CKPT_DIR}/eval.log"
echo "   ${LOCAL_CKPT_DIR}/corpus_eval_results.json"
echo "   ${LOCAL_CKPT_DIR}/mbpp_eval_results.json"
echo "=============================================="
