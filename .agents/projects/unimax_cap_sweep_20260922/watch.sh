#!/bin/bash
# Detached watch of the UniMax cap sweep parent and its children; one line every 10 min, CHANGE lines when states move.
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
D=.agents/projects/unimax_cap_sweep_20260922; LOG=$D/watch.log; JOB=/calvinxu/dm-delphi-unimax-cap-sweep-3e18-20260922-retry1
IRIS="uv run --no-sync iris --config lib/iris/config/marin.yaml"
prev=""
while :; do
  parent=$($IRIS job describe $JOB 2>/dev/null | awk '/^State:/{print $2; exit}')
  kids=$($IRIS job list --prefix "$JOB/" 2>/dev/null | awk 'NR>1 && $1 ~ /^\// {n=$1; sub(".*/", "", n); printf "%s=%s ", n, $2}')
  line="parent=$parent children: $kids"
  if [[ "$line" != "$prev" ]]; then echo "$(date -u +%FT%TZ) CHANGE $line" >> "$LOG"; prev="$line"; else echo "$(date -u +%FT%TZ) HEARTBEAT parent=$parent" >> "$LOG"; fi
  case "$parent" in succeeded|failed|killed) echo "$(date -u +%FT%TZ) DONE parent=$parent" >> "$LOG"; exit 0;; esac
  sleep 600
done
