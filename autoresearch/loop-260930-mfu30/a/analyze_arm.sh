#!/usr/bin/env bash
# Download a traced arm's rank-0 xplane and run the stream / carry-stall / copy checks.
#   analyze_arm.sh <run-id> <scratch dir>
set -euo pipefail
RUN="$1"; OUT="$2/$RUN"; mkdir -p "$OUT"
cd "$(git rev-parse --show-toplevel)"
T=autoresearch/loop-260930-mfu30
REMOTE="marin-cw:hero-checkpoints/tmp/ttl=30d/xprof/${RUN}/plugins/profile"
X=$(rclone lsf -R "$REMOTE" | grep 'xplane.pb$' | head -1)
[ -f "$OUT/trace.xplane.pb" ] || rclone copyto "$REMOTE/$X" "$OUT/trace.xplane.pb"
[ -f "$OUT/rows.pkl" ] || uv run python $T/tfop_dump.py "$OUT/trace.xplane.pb" $T "$OUT/rows.pkl" 2>&1 | grep -v warn | tail -1
[ -f "$OUT/opnames.pkl" ] || uv run python $T/hlo_opnames.py "$OUT/trace.xplane.pb" $T "$OUT/opnames.pkl" 2>&1 | grep -v warn | tail -1
echo "== streams"; uv run python $T/a/stream_check.py "$OUT/rows.pkl" 2>&1 | grep -v warn
echo "== carry stall"; uv run python $T/a/carry_stall.py "$OUT/rows.pkl" 2>&1 | grep -v warn
echo "== anatomy"; uv run python $T/anatomy.py "$OUT/rows.pkl" "$OUT/opnames.pkl" 2>&1 | grep -v warn | grep -E "^steps=|collective busy|memcpy busy|device idle"
