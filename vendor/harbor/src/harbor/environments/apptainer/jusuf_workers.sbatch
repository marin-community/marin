#!/bin/bash
#SBATCH --job-name=apptainer_workers
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --partition=batch
#SBATCH --account=reformo
#SBATCH --time=12:00:00
#SBATCH --output=/p/scratch/transfernetx/nezhurina1/apptainer_workers_%j.out

# Apptainer Bridge Workers on Jusuf/JUWELS CPU nodes.
# Each node runs multiple worker threads that manage Apptainer instances.
#
# Prerequisites:
#   1. Bridge server running on Jupiter (start_bridge.sh)
#   2. SIF images pre-built (prebuild_sifs.sh)
#   3. Reverse SSH tunnel + relay to Jusuf
#
# Usage:
#   sbatch jusuf_workers.sbatch

set -e

# The bridge server is accessible via relay on Jusuf login node
BRIDGE_URL="${BRIDGE_URL:-http://localhost:9910}"
SIF_CACHE="${SIF_CACHE:-/p/scratch/transfernetx/nezhurina1/sif_cache}"
STAGING_BASE="${STAGING_BASE:-/tmp/apptainer_staging}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-16}"

WORKER_SCRIPT="$(dirname "$0")/worker.py"
# Fallback: check if copied to home dir
if [ ! -f "$WORKER_SCRIPT" ]; then
    WORKER_SCRIPT="/p/home/jusers/${USER}/jusuf/apptainer_bridge_worker.py"
fi
if [ ! -f "$WORKER_SCRIPT" ]; then
    WORKER_SCRIPT="/p/scratch/transfernetx/${USER}/apptainer_bridge/worker.py"
fi

echo "============================================="
echo "Apptainer Bridge Workers"
echo "  Nodes: $SLURM_JOB_NUM_NODES"
echo "  Workers/node: $WORKERS_PER_NODE"
echo "  Bridge: $BRIDGE_URL"
echo "  SIF cache: $SIF_CACHE"
echo "  Staging: $STAGING_BASE"
echo "  Worker script: $WORKER_SCRIPT"
echo "============================================="

if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "ERROR: Worker script not found"
    echo "Copy it: scp harbor_patched/src/harbor/environments/apptainer_bridge/worker.py \\
    ${USER}@jusuf.fz-juelich.de:/p/scratch/transfernetx/${USER}/apptainer_bridge/"
    exit 1
fi

# Verify apptainer is available
if ! command -v apptainer &>/dev/null && ! command -v singularity &>/dev/null; then
    echo "Loading apptainer module..."
    module load Apptainer-Tools 2>/dev/null || module load Singularity 2>/dev/null || true
fi

# Test bridge connectivity
for i in $(seq 1 30); do
    if curl -s "$BRIDGE_URL/status" > /dev/null 2>&1; then
        echo "Bridge reachable!"
        curl -s "$BRIDGE_URL/status" | python3 -m json.tool 2>/dev/null || true
        break
    fi
    echo "Waiting for bridge (attempt $i/30)..."
    sleep 5
done

if ! curl -s "$BRIDGE_URL/status" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach bridge at $BRIDGE_URL"
    echo "Make sure start_bridge.sh has been run on Jupiter."
    exit 1
fi

# Create staging directories
mkdir -p "$STAGING_BASE"

# Launch workers on all nodes
srun --overlap python3 "$WORKER_SCRIPT" \
    --bridge-url "$BRIDGE_URL" \
    --sif-cache "$SIF_CACHE" \
    --staging-base "$STAGING_BASE" \
    --num-workers "$WORKERS_PER_NODE" \
    --poll-interval 0.3
