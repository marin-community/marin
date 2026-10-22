#!/bin/bash
# Start Apptainer Bridge: server on Jupiter, workers on JUWELS via SSH reverse tunnel.
#
# Architecture (server-on-Jupiter, per-node dispatcher):
#   Jupiter: bridge server on 0.0.0.0:9920 (local, fast for Harbor client)
#   JUWELS compute: 1 dispatcher/node polls via SSH reverse tunnel -> Jupiter:9920
#   Workers consume jobs from local in-process queue (no tunnel traffic)
#   Only 32 dispatcher polls traverse SSH (not 512 worker polls)
#
# Usage: bash start_bridge.sh [--jusuf | --juwels]

set -e

CLUSTER="${1:---jusuf}"
BRIDGE_PORT=9920
SSH_KEY="${SSH_KEY:-$HOME/.ssh/docker_jusuf}"
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF_CACHE="/p/scratch/transfernetx/nezhurina1/sif_cache"
SERVER_LOG="/tmp/apptainer_bridge_server.log"
SERVER_SESSION="apptainer_bridge_${BRIDGE_PORT}"

start_server() {
    pkill -f "server.py --port $BRIDGE_PORT" 2>/dev/null || true
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
    sleep 1

    if command -v tmux >/dev/null 2>&1; then
        tmux new-session -d -s "$SERVER_SESSION" \
            "python3 '$BRIDGE_DIR/server.py' --port '$BRIDGE_PORT' --host 0.0.0.0 >> '$SERVER_LOG' 2>&1"
    else
        nohup python3 "$BRIDGE_DIR/server.py" --port "$BRIDGE_PORT" --host 0.0.0.0 \
            >> "$SERVER_LOG" 2>&1 &
    fi
}

if [[ "$CLUSTER" == "--juwels" ]]; then
    REMOTE_HOST="${USER}@juwels01.fz-juelich.de"
    REMOTE_SCRATCH="/p/scratch/transfernetx/${USER}"
    echo "Target cluster: JUWELS-Cluster (pinned to juwels01)"
elif [[ "$CLUSTER" == "--jusuf" ]]; then
    REMOTE_HOST="${USER}@jusuf2.fz-juelich.de"
    REMOTE_SCRATCH="/p/scratch/transfernetx/${USER}"
    echo "Target cluster: Jusuf"
else
    echo "Usage: bash start_bridge.sh [--jusuf | --juwels]"
    exit 1
fi

REMOTE_BRIDGE_DIR="${REMOTE_SCRATCH}/apptainer_bridge"

# ---- 1. Copy worker script to remote (server stays on Jupiter) ----
echo "Syncing worker script to remote..."
ssh -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p $REMOTE_BRIDGE_DIR"
scp -i "$SSH_KEY" "$BRIDGE_DIR/worker.py" "$REMOTE_HOST:$REMOTE_BRIDGE_DIR/worker.py"

# ---- 2. Start bridge server locally on Jupiter ----
echo ""
echo "Starting bridge server on Jupiter (localhost:$BRIDGE_PORT)..."
: > "$SERVER_LOG"
start_server

sleep 2
if curl -s "http://localhost:$BRIDGE_PORT/status" > /dev/null 2>&1; then
    echo "Bridge server started on Jupiter:$BRIDGE_PORT"
    curl -s "http://localhost:$BRIDGE_PORT/status" | python3 -m json.tool 2>/dev/null || true
else
    echo "ERROR: Bridge server failed to start"
    tail -20 "$SERVER_LOG"
    exit 1
fi

# ---- 3. SSH reverse tunnel: remote login:9920 -> Jupiter:9920 ----
pkill -f "ssh.*-R.*${BRIDGE_PORT}:localhost:${BRIDGE_PORT}.*${REMOTE_HOST##*@}" 2>/dev/null || true
sleep 1

echo ""
echo "Starting reverse SSH tunnel (${REMOTE_HOST##*@}:$BRIDGE_PORT -> Jupiter:$BRIDGE_PORT)..."
ssh -i "$SSH_KEY" -fN -R 0.0.0.0:${BRIDGE_PORT}:localhost:${BRIDGE_PORT} "$REMOTE_HOST" \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=10 \
    -o GatewayPorts=yes \
    -o ExitOnForwardFailure=yes

TUNNEL_PID=$(pgrep -f "ssh.*-R.*${BRIDGE_PORT}:localhost:${BRIDGE_PORT}" | head -1)
echo "Reverse tunnel PID: $TUNNEL_PID"

sleep 2
TUNNEL_OK=$(ssh -i "$SSH_KEY" "$REMOTE_HOST" "curl -s http://localhost:$BRIDGE_PORT/status 2>/dev/null" || true)
if echo "$TUNNEL_OK" | grep -q "envs"; then
    echo "Reverse tunnel working! JUWELS can reach Jupiter bridge."
else
    echo "WARNING: Tunnel may not be ready yet. Verify manually:"
    echo "  ssh -i $SSH_KEY $REMOTE_HOST 'curl -s http://localhost:$BRIDGE_PORT/status'"
fi

# ---- 4. Create SIF cache dir & copy sbatch script ----
ssh -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p $SIF_CACHE"
WORKER_SBATCH="juwels_workers.sbatch"
if [[ "$CLUSTER" == "--jusuf" ]]; then
    WORKER_SBATCH="jusuf_workers.sbatch"
fi
scp -i "$SSH_KEY" "$BRIDGE_DIR/$WORKER_SBATCH" "$REMOTE_HOST:$REMOTE_BRIDGE_DIR/$WORKER_SBATCH"

# ---- 5. Start tunnel health monitor ----
pkill -f "tunnel_monitor.sh $CLUSTER" 2>/dev/null || true
sleep 1
echo ""
echo "Starting tunnel health monitor..."
nohup bash "$BRIDGE_DIR/tunnel_monitor.sh" "$CLUSTER" > /tmp/apptainer_tunnel_monitor.log 2>&1 &
MONITOR_PID=$!
echo "Tunnel monitor PID: $MONITOR_PID"

echo ""
echo "============================================="
echo "Apptainer Bridge ready! (server-on-Jupiter, per-node dispatcher)"
echo ""
echo "  Jupiter server:  http://localhost:$BRIDGE_PORT (local)"
echo "  Remote access:   http://localhost:$BRIDGE_PORT (via reverse tunnel)"
echo "  Booster access:  http://10.128.1.1:$BRIDGE_PORT (Jupiter from Booster)"
echo "  Server session:  $SERVER_SESSION"
echo "  Tunnel monitor:  PID $MONITOR_PID (log: /tmp/apptainer_tunnel_monitor.log)"
echo ""
echo "Next steps:"
echo "  1. Submit workers on ${CLUSTER#--} (auto-chains up to MAX_CHAIN=5):"
echo "     ssh -i $SSH_KEY $REMOTE_HOST \\
       'SIF_CACHE=$SIF_CACHE \\
        sbatch ${REMOTE_BRIDGE_DIR}/$WORKER_SBATCH'"
echo ""
echo "  2. In BenSkyRL config, set:"
echo "     export APPTAINER_BRIDGE_URL=http://localhost:$BRIDGE_PORT"
echo "     (or http://10.128.1.1:$BRIDGE_PORT from Booster)"
echo "============================================="
