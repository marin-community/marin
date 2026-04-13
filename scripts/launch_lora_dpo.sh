#!/bin/bash
# Launch LoRA-DPO training on TPU pod simpo_worker_2 (v5p-128, 16 workers).
#
# This script:
# 1. Syncs the marin codebase to all TPU workers
# 2. Installs dependencies (uv + Python packages)
# 3. Launches the LoRA-DPO training script in the background on all workers
#
# Usage:
#   bash scripts/launch_lora_dpo.sh

set -euo pipefail

TPU_NAME="simpo_worker_2"
ZONE="us-central1-a"
PROJECT="hai-gcp-models"
NUM_WORKERS=16

# Environment variables for training
WANDB_API_KEY="${WANDB_API_KEY:-3d91078de9092186db48b81253a2e8902563454b}"
HF_TOKEN="${HF_TOKEN:-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}"
WANDB_PROJECT="dpo"

SSH_CMD="gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT}"

echo "=== LoRA-DPO Training Launch ==="
echo "TPU: ${TPU_NAME} (v5p-128, ${NUM_WORKERS} workers)"
echo "Zone: ${ZONE}"
echo "Project: ${PROJECT}"
echo ""

# Step 1: Create a tarball of the codebase (excluding heavy dirs)
echo "[1/4] Creating codebase tarball..."
cd /Users/ahmed/code/marin
tar czf /tmp/marin_code.tar.gz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='node_modules' \
    --exclude='site' \
    --exclude='docs' \
    --exclude='*.egg-info' \
    --exclude='uv.lock' \
    --exclude='lib/levanter/docs' \
    --exclude='data_browser' \
    .

# Step 2: Upload to all workers
echo "[2/4] Uploading codebase to all workers..."
for worker in $(seq 0 $((NUM_WORKERS - 1))); do
    gcloud compute tpus tpu-vm scp /tmp/marin_code.tar.gz \
        ${TPU_NAME}:~/marin_code.tar.gz \
        --zone=${ZONE} --project=${PROJECT} --worker=${worker} &
done
wait
echo "  Upload complete."

# Step 3: Set up environment on all workers
echo "[3/4] Setting up environment on all workers..."
${SSH_CMD} --worker=all --command='
set -ex
cd $HOME

# Extract code
rm -rf marin
mkdir -p marin
cd marin
tar xzf ~/marin_code.tar.gz

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Install Python deps with TPU extras
uv sync --all-packages --extra=tpu --python=3.11 2>&1 | tail -5
echo "Setup complete on $(hostname)"
'

# Step 4: Launch training in background on all workers
echo "[4/4] Launching LoRA-DPO training in background on all workers..."
${SSH_CMD} --worker=all --command="
set -ex
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd \$HOME/marin

# Set environment variables
export WANDB_API_KEY='${WANDB_API_KEY}'
export HF_TOKEN='${HF_TOKEN}'
export WANDB_PROJECT='${WANDB_PROJECT}'
export TPU_STDERR_LOG_LEVEL=2
export TPU_MIN_LOG_LEVEL=2
export TOKENIZERS_PARALLELISM=false

# Kill any existing training processes
pkill -f 'levanter.main.lora_dpo' 2>/dev/null || true
sleep 2

echo 'Starting LoRA-DPO training on '\$(hostname)'...'
nohup uv run python -m levanter.main.lora_dpo \\
    --config_path lib/levanter/config/lora_dpo_llama8b_ultrafeedback.yaml \\
    > ~/lora_dpo_training.log 2>&1 &

echo 'Training process launched with PID:' \$!
echo 'Logs: ~/lora_dpo_training.log'
"

echo ""
echo "=== Training launched successfully ==="
echo ""
echo "To monitor training logs on worker 0:"
echo "  ${SSH_CMD} --worker=0 --command='tail -f ~/lora_dpo_training.log'"
echo ""
echo "To check if training is running:"
echo "  ${SSH_CMD} --worker=0 --command='ps aux | grep lora_dpo'"
echo ""
echo "WandB dashboard: https://wandb.ai/${WANDB_PROJECT}"
echo "Checkpoints: gs://marin-us-central1/checkpoints/lora_dpo_llama8b_ultrafeedback"
