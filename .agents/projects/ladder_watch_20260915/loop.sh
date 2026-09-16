#!/bin/bash
# Ladder watch loop: one STATUS line per cycle (~12 min) plus ALERT/EVALOK lines only when they change.
# Dedupe key strips the HH:MMZ stamp so an unchanged alert does not re-fire every cycle.
cd /Users/calvinxu/Projects/Work/Marin/marin
W=/Users/calvinxu/Projects/Work/Marin/marin/.agents/projects/ladder_watch_20260915
strip() { sed -E 's/^(ALERT|EVALOK) [0-9]{2}:[0-9]{2}Z /\1 /'; }
[ -f "$W/prev_alerts.txt" ] || : > "$W/prev_alerts.txt"
while true; do
  out=$(uv run --offline --no-sync python "$W/tick.py" 2>/dev/null | grep -v '^Warning')
  if [ -z "$out" ]; then
    echo "PROBEERR $(date -u +%H:%MZ): empty tick"
  else
    printf '%s\n' "$out" >> "$W/tick_history.log"
    printf '%s\n' "$out" | grep -E '^(STATUS|PROBEERR)' || true
    printf '%s\n' "$out" | grep -E '^(ALERT|EVALOK)' | strip | sort > "$W/cur_alerts.txt"
    comm -13 "$W/prev_alerts.txt" "$W/cur_alerts.txt" | sed "s/^/$(date -u +%H:%MZ) NEW /" || true
    comm -23 "$W/prev_alerts.txt" "$W/cur_alerts.txt" | sed "s/^/$(date -u +%H:%MZ) CLEARED /" || true
    mv "$W/cur_alerts.txt" "$W/prev_alerts.txt"
  fi
  sleep 600
done
