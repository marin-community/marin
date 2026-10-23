#!/bin/bash
# SSH Tunnel Health Monitor: runs on Jupiter, checks and restarts the
# reverse SSH tunnel to JUWELS if it drops. Also restarts the local bridge
# server if the Jupiter-side server process disappears.
#
# Usage: nohup bash tunnel_monitor.sh [--juwels|--jusuf] &

set -e

CLUSTER="${1:---juwels}"
BRIDGE_PORT=9920
SSH_KEY="${SSH_KEY:-$HOME/.ssh/docker_jusuf}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-100}"
LOG="/tmp/apptainer_tunnel_monitor.log"
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_LOG="/tmp/apptainer_bridge_server.log"
SERVER_SESSION="apptainer_bridge_${BRIDGE_PORT}"

if [[ "$CLUSTER" == "--juwels" ]]; then
    REMOTE_HOST="${USER}@juwels01.fz-juelich.de"
elif [[ "$CLUSTER" == "--jusuf" ]]; then
    REMOTE_HOST="${USER}@jusuf2.fz-juelich.de"
else
    echo "Usage: tunnel_monitor.sh [--juwels|--jusuf]"
    exit 1
fi

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG"
}

restart_server() {
    log "Restarting local bridge server on Jupiter..."
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
        log "Local bridge server restarted successfully"
        return 0
    fi

    log "WARNING: Local bridge server restart failed"
    return 1
}

restart_tunnel() {
    log "Restarting SSH reverse tunnel to ${REMOTE_HOST}..."
    pkill -f "ssh.*-R.*${BRIDGE_PORT}:localhost:${BRIDGE_PORT}.*${REMOTE_HOST##*@}" 2>/dev/null || true
    sleep 2

    ssh -i "$SSH_KEY" -fN \
        -R 0.0.0.0:${BRIDGE_PORT}:localhost:${BRIDGE_PORT} "$REMOTE_HOST" \
        -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=10 \
        -o GatewayPorts=yes \
        -o ExitOnForwardFailure=yes \
        -o ConnectTimeout=15

    sleep 2

    TUNNEL_OK=$(ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_HOST" \
        "curl -s http://localhost:$BRIDGE_PORT/status 2>/dev/null" 2>/dev/null || true)
    if echo "$TUNNEL_OK" | grep -q "envs"; then
        TUNNEL_PID=$(pgrep -f "ssh.*-R.*${BRIDGE_PORT}:localhost:${BRIDGE_PORT}" | head -1)
        log "Tunnel restarted successfully (PID: $TUNNEL_PID)"
        return 0
    fi

    log "WARNING: Tunnel restart may have failed"
    return 1
}

check_tunnel() {
    if ! pgrep -f "ssh.*-R.*${BRIDGE_PORT}:localhost:${BRIDGE_PORT}.*${REMOTE_HOST##*@}" > /dev/null 2>&1; then
        log "Tunnel process not found"
        return 1
    fi

    if ! curl -s "http://localhost:$BRIDGE_PORT/status" > /dev/null 2>&1; then
        log "Bridge server not responding locally"
        return 2
    fi

    TUNNEL_OK=$(ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_HOST" \
        "curl -s -m 5 http://localhost:$BRIDGE_PORT/status 2>/dev/null" 2>/dev/null || true)
    if echo "$TUNNEL_OK" | grep -q "envs"; then
        return 0
    fi

    log "Tunnel not forwarding traffic (remote can't reach bridge)"
    return 1
}

log "Starting tunnel monitor (check every ${CHECK_INTERVAL}s, max ${MAX_RESTARTS} restarts)"
log "Remote: $REMOTE_HOST, Port: $BRIDGE_PORT"

RESTARTS=0
CONSECUTIVE_FAILURES=0

while true; do
    if check_tunnel; then
        CONSECUTIVE_FAILURES=0
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))

        if [ "$CONSECUTIVE_FAILURES" -ge 3 ]; then
            if ! curl -s "http://localhost:$BRIDGE_PORT/status" > /dev/null 2>&1; then
                restart_server || true
            fi

            log "3 consecutive failures, restarting tunnel (restart $((RESTARTS + 1))/$MAX_RESTARTS)..."
            if restart_tunnel; then
                RESTARTS=$((RESTARTS + 1))
                CONSECUTIVE_FAILURES=0
                log "Total restarts: $RESTARTS"
            else
                log "Restart failed, will retry in ${CHECK_INTERVAL}s"
            fi

            if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
                log "ERROR: Max restarts ($MAX_RESTARTS) reached, exiting"
                exit 1
            fi
        else
            log "Failure $CONSECUTIVE_FAILURES/3, will retry before restarting"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
