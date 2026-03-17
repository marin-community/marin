#!/bin/bash
# =============================================================================
# Ray Multi-Host vLLM on TPU — Launcher Script
#
# Serves large models (70B-235B) across multiple TPU hosts using Ray PP + TP.
# Applies runtime patches to vllm-tpu:nightly to fix:
#   1. JAX isolation for multi-host (XLA device_id mismatch)
#   2. Weight TP sharding (nnx.get_named_sharding fails under Ray)
#   3. PP parallel state (vLLM PP rank propagation)
#   4. KV cache layer name mapping (PP boundary layers)
#   5. supports_mm_inputs attribute (TPUModelRunner)
#
# Usage:
#   bash launch.sh <TPU_NAME> <ZONE> <MODEL_GCS_PATH> <PP_SIZE> <TP_SIZE> <MAX_MODEL_LEN>
#
# Example (Qwen3-235B on v5p-16):
#   bash launch.sh vllm-qwen3-v5p16 us-central1-a \
#     gs://marin-us-central1/models/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae \
#     2 4 16384
#
# Example (Llama 70B on v6e-16):
#   bash launch.sh vllm-ray-v6e16 us-east1-d \
#     gs://marin-us-east1/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct \
#     4 4 4096
#
# Prerequisites:
#   - TPU already allocated and ACTIVE
#   - gcsfuse installed on all hosts
#   - Model weights in same-region GCS bucket
#   - Docker image: vllm/vllm-tpu:nightly pulled on all hosts
# =============================================================================

set -euo pipefail

TPU_NAME="${1:?Usage: launch.sh <TPU_NAME> <ZONE> <MODEL_GCS_PATH> <PP_SIZE> <TP_SIZE> <MAX_MODEL_LEN>}"
ZONE="${2:?}"
MODEL_GCS_PATH="${3:?}"
PP_SIZE="${4:?}"
TP_SIZE="${5:?}"
MAX_MODEL_LEN="${6:-4096}"
PROJECT="${PROJECT:-hai-gcp-models}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
PORT="${PORT:-8000}"

# Extract bucket and path for gcsfuse
GCS_BUCKET=$(echo "$MODEL_GCS_PATH" | sed 's|gs://||' | cut -d/ -f1)
GCS_SUBPATH=$(echo "$MODEL_GCS_PATH" | sed "s|gs://${GCS_BUCKET}/||")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"

echo "============================================"
echo "Ray Multi-Host vLLM Launcher"
echo "============================================"
echo "TPU:           $TPU_NAME ($ZONE)"
echo "Model:         $MODEL_GCS_PATH"
echo "PP=$PP_SIZE TP=$TP_SIZE max_model_len=$MAX_MODEL_LEN"
echo "GCS bucket:    $GCS_BUCKET"
echo "GCS subpath:   $GCS_SUBPATH"
echo "============================================"

# Get number of workers
NUM_WORKERS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" \
  --format="value(networkEndpoints)" 2>/dev/null | tr ';' '\n' | wc -l)
echo "Workers: $NUM_WORKERS"

# Get head IP
HEAD_IP=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="hostname -I | awk '{print \$1}'" 2>/dev/null)
echo "Head IP: $HEAD_IP"

# Verify PP size matches worker count
if [ "$PP_SIZE" -ne "$NUM_WORKERS" ]; then
  echo "WARNING: PP_SIZE=$PP_SIZE != NUM_WORKERS=$NUM_WORKERS"
  echo "PP_SIZE must equal the number of hosts for Ray multi-host."
  echo "Continuing anyway..."
fi

echo ""
echo "Step 1: Setup gcsfuse on all hosts..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="
    # Install gcsfuse if needed
    which gcsfuse >/dev/null 2>&1 || {
      export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)
      echo \"deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main\" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
      curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
      sudo apt-get update -qq && sudo apt-get install -y -qq gcsfuse
    }
    # Mount
    sudo umount /mnt/gcs-models 2>/dev/null || true
    sudo mkdir -p /mnt/gcs-models
    sudo gcsfuse --implicit-dirs -o allow_other \
      --only-dir '$GCS_SUBPATH' \
      --file-cache-max-size-mb 0 \
      '$GCS_BUCKET' /mnt/gcs-models 2>&1 | grep -E 'mounted|Error'
    ls /mnt/gcs-models/config.json >/dev/null 2>&1 && echo 'MODEL OK' || echo 'MODEL FAIL'
  " 2>/dev/null

echo ""
echo "Step 2: Start Docker containers + Ray cluster..."

# Kill old containers
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="sg docker 'docker rm -f ray-node 2>/dev/null'; echo done" 2>/dev/null

# Start head
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sg docker 'docker run -d --name ray-node --privileged --net=host --shm-size=16g \
    -e TPU_MULTIHOST_BACKEND=ray -e TPU_BACKEND_TYPE=jax -e RAY_DEDUP_LOGS=0 \
    -e HF_TOKEN=\$HF_TOKEN -e HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN -e HUGGINGFACE_HUB_TOKEN=\$HF_TOKEN \
    -v /dev/shm:/dev/shm -v /mnt/gcs-models:/mnt/gcs-models:ro \
    vllm/vllm-tpu:nightly \
    bash -c \"ray start --head --port=6379 && sleep infinity\"'" 2>/dev/null
sleep 5

# Start workers
for w in $(seq 1 $((NUM_WORKERS - 1))); do
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker="$w" \
    --command="sg docker 'docker run -d --name ray-node --privileged --net=host --shm-size=16g \
      -e TPU_MULTIHOST_BACKEND=ray -e TPU_BACKEND_TYPE=jax -e RAY_DEDUP_LOGS=0 \
      -e HF_TOKEN=\$HF_TOKEN -e HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN -e HUGGINGFACE_HUB_TOKEN=\$HF_TOKEN \
      -v /dev/shm:/dev/shm -v /mnt/gcs-models:/mnt/gcs-models:ro \
      vllm/vllm-tpu:nightly \
      bash -c \"ray start --address=$HEAD_IP:6379 --block\"'" 2>/dev/null &
done
wait
sleep 10

# Verify Ray
echo "Ray cluster status:"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sg docker 'docker exec ray-node ray status 2>/dev/null | grep -E \"Active:|TPU\"'" 2>/dev/null

echo ""
echo "Step 3: Apply patches..."

# Copy patches to all workers
for w in $(seq 0 $((NUM_WORKERS - 1))); do
  for p in "$PATCH_DIR"/*.py; do
    gcloud compute tpus tpu-vm scp "$p" "$TPU_NAME:/tmp/$(basename $p)" \
      --zone="$ZONE" --project="$PROJECT" --worker="$w" 2>/dev/null
  done &
done
wait

# Apply patches
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="sg docker '
    for p in /tmp/patch_*.py; do
      docker cp \$p ray-node:/tmp/
      docker exec ray-node python \$p 2>&1 | tail -1
    done
  '" 2>/dev/null | grep -c PATCHED
echo "patches applied across all workers"

echo ""
echo "Step 4: Launch vLLM serve..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sg docker \"docker exec -d ray-node bash -c 'vllm serve /mnt/gcs-models \
    --tensor-parallel-size $TP_SIZE \
    --pipeline-parallel-size $PP_SIZE \
    --distributed-executor-backend ray \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $PORT \
    --trust-remote-code \
    > /tmp/vllm_serve.log 2>&1'\"" 2>/dev/null

echo ""
echo "============================================"
echo "vLLM serve launched!"
echo "============================================"
echo "Tail logs:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command=\"sg docker 'docker exec ray-node tail -f /tmp/vllm_serve.log'\" 2>/dev/null | grep throughput"
echo ""
echo "Check health:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command=\"sg docker 'docker exec ray-node curl -s http://localhost:$PORT/health'\""
echo ""
echo "Weight loading may take 30-60 min for large models (gcsfuse, first load)."
echo "Subsequent restarts within the same container use OS page cache (seconds)."
echo "============================================"
