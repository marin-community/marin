#!/bin/bash
# Start Apptainer Bridge on JURECA. Server + workers both on JURECA (single cluster).
#
# Architecture (JURECA-local, no cross-cluster SSH):
#   JURECA login (or compute): bridge server on 0.0.0.0:9920
#   JURECA dc-cpu/dc-gpu nodes: workers poll the server directly via internal IB network
#
# This differs from JUWELS setup: no reverse SSH tunnel, no separate worker cluster.
# Eval, bridge, and workers all live on JURECA and share /p filesystem with JUWELS,
# so the existing SIF cache at /p/scratch/transfernetx/nezhurina1/sif_cache is reused.
#
# Usage: bash start_bridge_jureca.sh [--port 9920]

set -e

BRIDGE_PORT="${BRIDGE_PORT:-9920}"
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_LOG="${SERVER_LOG:-/tmp/apptainer_bridge_jureca.log}"
SERVER_SESSION="apptainer_bridge_jureca_${BRIDGE_PORT}"

# Parse simple --port flag if provided
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) BRIDGE_PORT="$2"; shift 2;;
        *) shift;;
    esac
done

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

sleep 2
if curl -s "http://localhost:$BRIDGE_PORT/status" > /dev/null 2>&1; then
    echo "Bridge server started on $(hostname):$BRIDGE_PORT"
    curl -s "http://localhost:$BRIDGE_PORT/status" | python3 -m json.tool 2>/dev/null || true
else
    echo "ERROR: Bridge server failed to start"
    tail -20 "$SERVER_LOG" 2>/dev/null
    exit 1
fi

# Resolve internal IB hostname for workers
LOGIN_IB_NAME="$(hostname -s)i"  # jrlogin05 -> jrlogin05i on IB
LOGIN_IB_IP="$(getent ahostsv4 "$LOGIN_IB_NAME" 2>/dev/null | awk '{print $1; exit}')"

cat <<INFO
=============================================
Apptainer Bridge on JURECA ready
  Server:      http://$(hostname):$BRIDGE_PORT (local)
  Workers URL: http://${LOGIN_IB_NAME:-<host>}:$BRIDGE_PORT (${LOGIN_IB_IP:-?})
  Session:     $SERVER_SESSION
  Log:         $SERVER_LOG

Next steps:
  1) Submit workers:
       sbatch $BRIDGE_DIR/jureca_workers.sbatch
     (or set BRIDGE_LOGIN=$LOGIN_IB_NAME inside sbatch env)

  2) In eval sbatch, set:
       export APPTAINER_BRIDGE_URL=http://${LOGIN_IB_NAME:-localhost}:$BRIDGE_PORT
=============================================
INFO
