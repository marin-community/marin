#!/bin/bash
#SBATCH --job-name=apptainer_workers
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --partition=batch
#SBATCH --account=transfernetx
#SBATCH --time=24:00:00
#SBATCH --output=/p/scratch/transfernetx/nezhurina1/apptainer_workers_%j.out

# Apptainer Bridge Workers on JUWELS (per-node dispatcher mode)
#
# Architecture:
#   Jupiter: bridge server on localhost:9920 (fast for Harbor client)
#   JUWELS login: reverse SSH tunnel → Jupiter:9920
#   Each compute node: 1 dispatcher polls localhost:9920 (via login node)
#                      N workers consume from local queue (no HTTP polling)
#   Only 32 dispatchers poll through tunnel (not 512 workers)
#
# Auto-chain: submits a successor job before starting, so workers
# are always available. Set MAX_CHAIN=0 to disable.

set -e

BRIDGE_PORT="${BRIDGE_PORT:-9920}"
SIF_CACHE="${SIF_CACHE:-/p/scratch/transfernetx/nezhurina1/sif_cache}"
STAGING_BASE="${STAGING_BASE:-/p/scratch/projectnucleus/nezhurina1/apptainer_staging}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-16}"
MAX_CHAIN="${MAX_CHAIN:-5}"
CHAIN_COUNT="${CHAIN_COUNT:-0}"
RESTART_NODES="${RESTART_NODES:-8}"

WORKER_SCRIPT="/p/scratch/transfernetx/${USER}/apptainer_bridge/worker.py"
SBATCH_SCRIPT="/p/scratch/transfernetx/${USER}/apptainer_bridge/juwels_workers.sbatch"

# Workers connect to bridge via reverse tunnel on login node
# Use login node's internal hostname (compute nodes can't use localhost)
LOGIN_NODE="${LOGIN_NODE:-jwlogin01i}"
BRIDGE_URL="http://${LOGIN_NODE}:${BRIDGE_PORT}"

echo "============================================="
echo "Apptainer Bridge Workers (per-node dispatcher)"
echo "  Nodes: $SLURM_JOB_NUM_NODES"
echo "  Bridge: $BRIDGE_URL (via reverse tunnel to Jupiter)"
echo "  Workers/node: $WORKERS_PER_NODE"
echo "  Dispatchers: $SLURM_JOB_NUM_NODES (1 per node)"
echo "  SIF cache: $SIF_CACHE"
echo "  Chain: $CHAIN_COUNT / $MAX_CHAIN"
echo "  Job ID: $SLURM_JOB_ID"
echo "============================================="

# ---- Auto-chain: submit successor job that starts after this one ends ----
NEXT_CHAIN=$((CHAIN_COUNT + 1))
if [ "$NEXT_CHAIN" -le "$MAX_CHAIN" ] && [ -f "$SBATCH_SCRIPT" ]; then
    echo "Submitting chain job $NEXT_CHAIN/$MAX_CHAIN (dependency=afterany:$SLURM_JOB_ID, nodes=$RESTART_NODES)..."
    NEXT_JOB=$(sbatch \
        --dependency=afterany:${SLURM_JOB_ID} \
        --nodes=${RESTART_NODES} \
        --export=ALL,CHAIN_COUNT=${NEXT_CHAIN},MAX_CHAIN=${MAX_CHAIN},RESTART_NODES=${RESTART_NODES} \
        "$SBATCH_SCRIPT" 2>&1 | grep -o '[0-9]*')
    echo "Chain job submitted: $NEXT_JOB"
else
    echo "Chain limit reached or script not found, no successor submitted."
fi

if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "ERROR: Worker script not found at $WORKER_SCRIPT"
    exit 1
fi

# Verify bridge is reachable via reverse tunnel
for i in $(seq 1 30); do
    if curl -s "$BRIDGE_URL/status" > /dev/null 2>&1; then
        echo "Bridge reachable at $BRIDGE_URL"
        curl -s "$BRIDGE_URL/status" | python3 -m json.tool 2>/dev/null || true
        break
    fi
    echo "Waiting for bridge (attempt $i/30)..."
    sleep 5
done

if ! curl -s "$BRIDGE_URL/status" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach bridge at $BRIDGE_URL"
    echo "Make sure start_bridge.sh was run on Jupiter first (sets up reverse tunnel)."
    exit 1
fi

# Create worker launcher
LAUNCHER="/p/scratch/transfernetx/${USER}/apptainer_bridge/worker_launcher_${SLURM_JOB_ID}.sh"
cat > "$LAUNCHER" << 'WORKER_EOF'
#!/bin/bash
set -e
NODE=$(hostname -s)

# Verify bridge connectivity via reverse tunnel
for i in $(seq 1 30); do
    if curl -s "$BRIDGE_URL/status" > /dev/null 2>&1; then
        echo "[$NODE] Bridge reachable at $BRIDGE_URL"
        break
    fi
    sleep 2
done
if ! curl -s "$BRIDGE_URL/status" > /dev/null 2>&1; then
    echo "[$NODE] ERROR: Cannot reach bridge at $BRIDGE_URL"
    exit 1
fi

mkdir -p "$STAGING_BASE"
export APPTAINER_CACHEDIR="/p/scratch/transfernetx/${USER}/apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR"

# Setup SOCKS proxy for internet access (needed by SweSmith test scripts)
export SSH_KEY="${SSH_KEY:-$HOME/.ssh/docker}"
TUNNEL_PORT=$((7200 + SLURM_PROCID))
NODE_IP=$(nslookup $NODE | grep 'Address' | tail -n1 | awk '{print $2}')

# Detect login node by compute node hostname prefix
if [[ $NODE == jwc* ]]; then
    LOGIN_FOR_PROXY="jwlogin22i"
elif [[ $NODE == jrc* ]]; then
    LOGIN_FOR_PROXY="jrlogin05i"
else
    LOGIN_FOR_PROXY="jwlogin22i"
fi

ssh -g -f -N -D ${TUNNEL_PORT} \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=1000 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=15 \
    -o TCPKeepAlive=no \
    -o ExitOnForwardFailure=yes \
    -o BatchMode=yes \
    -i ${SSH_KEY} ${USER}@${LOGIN_FOR_PROXY} 2>/dev/null || true

export BRIDGE_PROXY_HOST="${NODE_IP}"
export BRIDGE_PROXY_PORT="${TUNNEL_PORT}"

# Create proxychains config
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
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF
export PROXYCHAINS_CONF_FILE="$PROXYCHAINS_CONF"
echo "[$NODE] Proxy tunnel on port $TUNNEL_PORT via $LOGIN_FOR_PROXY"

echo "[$NODE] Starting 1 dispatcher + $WORKERS_PER_NODE workers, bridge=$BRIDGE_URL"
python3 "$WORKER_SCRIPT" \
    --bridge-url "$BRIDGE_URL" \
    --sif-cache "$SIF_CACHE" \
    --staging-base "$STAGING_BASE" \
    --num-workers "$WORKERS_PER_NODE" \
    --poll-interval 0.1
WORKER_EOF
chmod +x "$LAUNCHER"

export BRIDGE_URL WORKER_SCRIPT SIF_CACHE STAGING_BASE WORKERS_PER_NODE

srun --export=ALL bash "$LAUNCHER"
