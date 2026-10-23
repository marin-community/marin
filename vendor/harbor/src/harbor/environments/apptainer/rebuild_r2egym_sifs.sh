#!/bin/bash
#SBATCH --job-name=rebuild_r2egym_sifs
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --partition=devel
#SBATCH --account=projectnucleus
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/transfernetx/nezhurina1/sif_rebuild_%j.out

# Rebuild all R2EGym SIFs to include chardet (pip install).
# Uses existing SIFs as base — no Docker pull needed.
# Distributes work across nodes with GNU parallel.

set -e

SIF_CACHE="${SIF_CACHE:-/p/scratch/transfernetx/nezhurina1/sif_cache}"
PARALLEL_PER_NODE=12

echo "============================================="
echo "Rebuilding R2EGym SIFs with chardet"
echo "  Nodes: $SLURM_JOB_NUM_NODES"
echo "  Parallel/node: $PARALLEL_PER_NODE"
echo "  SIF cache: $SIF_CACHE"
echo "============================================="

# Create the rebuild script that each node will run
REBUILD_SCRIPT="/p/scratch/transfernetx/${USER}/apptainer_bridge/rebuild_one_sif_${SLURM_JOB_ID}.sh"
cat > "$REBUILD_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash
SIF_PATH="$1"
SIF_CACHE="$2"

if [ ! -f "$SIF_PATH" ]; then
    echo "SKIP: $SIF_PATH not found"
    exit 0
fi

BASENAME=$(basename "$SIF_PATH")
TMP_DEF=$(mktemp /p/scratch/transfernetx/${USER}/apptainer_bridge/rebuild_XXXXXX.def)
TMP_SIF="${SIF_PATH}.rebuild.tmp"

cat > "$TMP_DEF" << EOF
Bootstrap: localimage
From: ${SIF_PATH}

%post
    pip install chardet 2>/dev/null || python3 -m pip install chardet 2>/dev/null || true
EOF

if apptainer build --fakeroot "$TMP_SIF" "$TMP_DEF" > /dev/null 2>&1; then
    mv "$TMP_SIF" "$SIF_PATH"
    echo "OK: $BASENAME"
else
    echo "FAIL: $BASENAME"
    rm -f "$TMP_SIF"
fi

rm -f "$TMP_DEF"
SCRIPT_EOF
chmod +x "$REBUILD_SCRIPT"

# Create list of R2EGym SIFs to rebuild
SIF_LIST="/p/scratch/transfernetx/${USER}/apptainer_bridge/r2egym_sifs_${SLURM_JOB_ID}.txt"
ls "$SIF_CACHE"/build_r2egym-*.sif > "$SIF_LIST" 2>/dev/null
TOTAL=$(wc -l < "$SIF_LIST")
echo "Found $TOTAL R2EGym SIFs to rebuild"

# Split list across nodes
LINES_PER_NODE=$(( (TOTAL + SLURM_JOB_NUM_NODES - 1) / SLURM_JOB_NUM_NODES ))

# Create per-node launcher
NODE_LAUNCHER="/p/scratch/transfernetx/${USER}/apptainer_bridge/node_rebuild_${SLURM_JOB_ID}.sh"
cat > "$NODE_LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/bash
SIF_LIST="$1"
REBUILD_SCRIPT="$2"
SIF_CACHE="$3"
PARALLEL_PER_NODE="$4"
LINES_PER_NODE="$5"

NODE=$(hostname -s)
NODE_IDX=$SLURM_PROCID

# Get this node's chunk
START_LINE=$(( NODE_IDX * LINES_PER_NODE + 1 ))
MY_LIST="/p/scratch/transfernetx/nezhurina1/apptainer_bridge/node_${NODE_IDX}_sifs.txt"
sed -n "${START_LINE},$(( START_LINE + LINES_PER_NODE - 1 ))p" "$SIF_LIST" > "$MY_LIST"
MY_COUNT=$(wc -l < "$MY_LIST")

echo "[$NODE] Processing $MY_COUNT SIFs (lines $START_LINE-$(( START_LINE + MY_COUNT - 1 )))"

# Run in parallel using xargs
cat "$MY_LIST" | xargs -P "$PARALLEL_PER_NODE" -I{} bash "$REBUILD_SCRIPT" {} "$SIF_CACHE"

echo "[$NODE] Done"
rm -f "$MY_LIST"
LAUNCHER_EOF
chmod +x "$NODE_LAUNCHER"

export SIF_LIST REBUILD_SCRIPT SIF_CACHE PARALLEL_PER_NODE LINES_PER_NODE

START_TIME=$(date +%s)
srun --export=ALL bash "$NODE_LAUNCHER" "$SIF_LIST" "$REBUILD_SCRIPT" "$SIF_CACHE" "$PARALLEL_PER_NODE" "$LINES_PER_NODE"
END_TIME=$(date +%s)

echo ""
echo "============================================="
echo "Done in $((END_TIME - START_TIME)) seconds"
echo "============================================="

rm -f "$REBUILD_SCRIPT" "$SIF_LIST" "$NODE_LAUNCHER"
