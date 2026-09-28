#!/bin/bash
# Calvin (2026-09-26): cancel the us-east5 v6e-4 MT-MBPP full run once the europe-west4 full run is running, since the
# EU run covers the same three checkpoints. "Running" means the EU full parent is running with at least one TPU child
# running. Cancels exactly /calvinxu/mt-mbpp-accuracy-v6e4-full-east5-20260926 and its descendants, nothing else.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
E=.agents/projects/mt_mbpp_exec_20260926/euw4
LOG=$E/cancel_east5.log
EU=/calvinxu/mt-mbpp-accuracy-v6e4-full-euw4-20260926
EAST5=/calvinxu/mt-mbpp-accuracy-v6e4-full-east5-20260926
ir() { uv run --no-sync iris --config lib/iris/config/marin.yaml "$@"; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "watcher started (pid $$)"
while :; do
  grep -q "STOP" $E/auto_release.log 2>/dev/null && { log "EU chain stopped; east5 left running"; exit 1; }
  s=$(ir job describe $EU 2>/dev/null | awk '/^State:/{print $2; exit}')
  running=$(ir job list --prefix "$EU/" 2>/dev/null | awk 'NR>1 && $1 ~ /^\// && $2=="running"' | wc -l | tr -d ' ')
  if [[ $s == running && $running -ge 1 ]]; then
    log "EU full running with $running TPU children; cancelling $EAST5"
    ir job cancel --exact --dry-run $EAST5 >> "$LOG" 2>&1
    ir job cancel --exact $EAST5 >> "$LOG" 2>&1 && log "cancelled $EAST5 (state now: $(ir job describe $EAST5 2>/dev/null | awk '/^State:/{print $2; exit}'))" || log "cancel command failed"
    exit 0
  fi
  sleep 120
done
