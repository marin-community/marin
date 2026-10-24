#!/bin/bash
#SBATCH --job-name=apptainer_workers_jureca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --partition=dc-cpu
#SBATCH --account=synthlaion
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/synthlaion/nezhurina1/dc_agent_eval/logs/apptainer_workers_%j.out

# Apptainer Bridge Workers on JURECA.
#
# Differences from JUWELS sbatch:
#   - No cross-cluster reverse tunnel; bridge runs on JURECA login.
#   - Bridge URL points to login IB hostname (jrlogin05i).
#   - SIF cache is shared with JUWELS via /p/scratch/transfernetx (same filesystem).
#   - dc-cpu nodes may not have direct internet — proxychains over SSH for SIF builds.
#
# Auto-chain like the JUWELS version. Set MAX_CHAIN=0 to disable.

set -e

BRIDGE_PORT="${BRIDGE_PORT:-9920}"
BRIDGE_LOGIN="${BRIDGE_LOGIN:-jrlogin05i}"
BRIDGE_URL="${BRIDGE_URL:-http://${BRIDGE_LOGIN}:${BRIDGE_PORT}}"

SIF_CACHE="${SIF_CACHE:-/p/scratch/transfernetx/nezhurina1/sif_cache}"
STAGING_BASE="${STAGING_BASE:-/p/scratch/transfernetx/nezhurina1/apptainer_staging}"
APPTAINER_CACHEDIR_BASE="${APPTAINER_CACHEDIR_BASE:-/p/scratch/transfernetx/${USER}/apptainer_cache}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-16}"
MAX_CHAIN="${MAX_CHAIN:-0}"
CHAIN_COUNT="${CHAIN_COUNT:-0}"

HARBOR_PATCHED="${HARBOR_PATCHED:-/p/project1/laionize/marianna/dc_agent/harbor_patched/src}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${HARBOR_PATCHED}/harbor/environments/apptainer_bridge/worker.py}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${HARBOR_PATCHED}/harbor/environments/apptainer_bridge/jureca_workers.sbatch}"

# Apptainer instance metadata is written under $HOME/.apptainer/instances by default,
# but /p/home is quota-tight. Redirect HOME for the worker process to scratch.
APPTAINER_HOME_OVERRIDE="${APPTAINER_HOME_OVERRIDE:-/p/scratch/synthlaion/${USER}/cache/apptainer_home}"
mkdir -p "$APPTAINER_HOME_OVERRIDE"

# Activate conda env (worker.py only uses stdlib but apptainer needs to be in PATH)
export MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
export CONDA_ENV="/p/project1/ccstdl/envs/marianna/py3.12"
source "${MINICONDA_PATH}/bin/activate" "${CONDA_ENV}"

echo "============================================="
echo "Apptainer Bridge Workers on JURECA"
echo "  Nodes:        $SLURM_JOB_NUM_NODES"
echo "  Bridge URL:   $BRIDGE_URL"
echo "  Workers/node: $WORKERS_PER_NODE"
echo "  SIF cache:    $SIF_CACHE  (shared with JUWELS)"
echo "  Staging:      $STAGING_BASE"
echo "  Worker:       $WORKER_SCRIPT"
echo "  Chain:        $CHAIN_COUNT / $MAX_CHAIN"
echo "  Job ID:       $SLURM_JOB_ID"
echo "============================================="

# Auto-chain
NEXT_CHAIN=$((CHAIN_COUNT + 1))
if [ "$NEXT_CHAIN" -le "$MAX_CHAIN" ] && [ -f "$SBATCH_SCRIPT" ]; then
    echo "Submitting chain job $NEXT_CHAIN/$MAX_CHAIN (dependency=afterany:$SLURM_JOB_ID)..."
    NEXT_JOB=$(sbatch \
        --dependency=afterany:${SLURM_JOB_ID} \
        --export=ALL,CHAIN_COUNT=${NEXT_CHAIN},MAX_CHAIN=${MAX_CHAIN} \
        "$SBATCH_SCRIPT" 2>&1 | grep -o '[0-9]*')
    echo "Chain job submitted: $NEXT_JOB"
fi

if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "ERROR: Worker script not found at $WORKER_SCRIPT"
    exit 1
fi

# Verify bridge reachable
for i in $(seq 1 30); do
    if curl -s --connect-timeout 5 "$BRIDGE_URL/status" > /dev/null 2>&1; then
        echo "Bridge reachable at $BRIDGE_URL"
        curl -s "$BRIDGE_URL/status" | python3 -m json.tool 2>/dev/null | head -10 || true
        break
    fi
    echo "Waiting for bridge (attempt $i/30)..."
    sleep 5
done
if ! curl -s --connect-timeout 5 "$BRIDGE_URL/status" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach bridge at $BRIDGE_URL"
    echo "Start the bridge first: bash $HARBOR_PATCHED/harbor/environments/apptainer_bridge/start_bridge_jureca.sh"
    exit 1
fi

# Per-node launcher — must be on a shared FS so all nodes can exec it
LAUNCHER_DIR="${LAUNCHER_DIR:-/p/scratch/synthlaion/${USER}/dc_agent_eval/logs}"
mkdir -p "$LAUNCHER_DIR"
LAUNCHER="${LAUNCHER_DIR}/jureca_worker_launcher_${SLURM_JOB_ID}.sh"
cat > "$LAUNCHER" << 'WORKER_EOF'
#!/bin/bash
set -e
NODE=$(hostname -s)
echo "[$NODE] starting $WORKERS_PER_NODE workers, bridge=$BRIDGE_URL, HOME=$APPTAINER_HOME_OVERRIDE"

mkdir -p "$STAGING_BASE"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR_BASE}/${NODE}"
mkdir -p "$APPTAINER_CACHEDIR"
export HOME="$APPTAINER_HOME_OVERRIDE"

# If a SOCKS proxy is configured (caller exported BRIDGE_PROXY_HOST/PORT), build a
# proxychains config so worker.py can use it for outbound builds. dc-cpu lacks
# direct internet; the eval sbatch will provide proxy details when needed.
if [ -n "$BRIDGE_PROXY_HOST" ] && [ -n "$BRIDGE_PROXY_PORT" ]; then
    PROXYCHAINS_CONF="${STAGING_BASE}/proxychains_${SLURM_JOB_ID}_${SLURM_PROCID}.conf"
    cat > "$PROXYCHAINS_CONF" <<PCEOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${BRIDGE_PROXY_HOST} ${BRIDGE_PROXY_PORT}
PCEOF
    export PROXYCHAINS_CONF_FILE="$PROXYCHAINS_CONF"
    echo "[$NODE] proxychains config -> $PROXYCHAINS_CONF (socks5://$BRIDGE_PROXY_HOST:$BRIDGE_PROXY_PORT)"
fi

python3 "$WORKER_SCRIPT" \
    --bridge-url "$BRIDGE_URL" \
    --sif-cache "$SIF_CACHE" \
    --staging-base "$STAGING_BASE" \
    --num-workers "$WORKERS_PER_NODE" \
    --poll-interval 0.1
WORKER_EOF
chmod +x "$LAUNCHER"

export BRIDGE_URL WORKER_SCRIPT SIF_CACHE STAGING_BASE WORKERS_PER_NODE APPTAINER_CACHEDIR_BASE APPTAINER_HOME_OVERRIDE

srun --export=ALL bash "$LAUNCHER"
