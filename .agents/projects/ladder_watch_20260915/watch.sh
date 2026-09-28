#!/bin/bash
# Overnight ladder watch: emits a line only when a child's reduced state changes, plus a heartbeat every ~3 h.
cd /Users/calvinxu/Projects/Work/Marin/marin
W=/Users/calvinxu/Projects/Work/Marin/marin/.agents/projects/ladder_watch_20260915
reduce() { awk -F'|' '{ out=$1"|"$2; for (i=3;i<=NF;i++) if ($i !~ /^(p=|run=|pend=)/) out=out"|"$i; print out }'; }
prev=""; i=0
while true; do
  i=$((i+1))
  cur=$(uv run --offline --no-sync python "$W/snapshot.py" --gcs 2>/dev/null | grep -v '^Warning' | reduce | sort)
  if [ -z "$cur" ]; then echo "WATCH-ERROR: empty snapshot at $(date -u +%H:%MZ)"; sleep 600; continue; fi
  printf '%s\n' "$cur" > "$W/last_snapshot.txt"; date -u +%Y-%m-%dT%H:%M:%SZ > "$W/last_snapshot_time.txt"
  if [ -n "$prev" ]; then
    comm -13 <(printf '%s\n' "$prev") <(printf '%s\n' "$cur") | sed "s/^/CHANGE $(date -u +%H:%MZ): /"
  fi
  if (( i % 18 == 1 )); then echo "HEARTBEAT $(date -u +%H:%MZ): $(printf '%s\n' "$cur" | grep -v '|parent$' | grep -v '|eval$' | cut -d'|' -f1-3,5- | tr '\n' ';' | cut -c1-900)"; fi
  prev="$cur"
  sleep 600
done
