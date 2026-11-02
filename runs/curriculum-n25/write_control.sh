#!/bin/bash
# Set every shard's in-flight limit live; shards poll control.json every 15 s and hold (and log ERROR)
# once it is older than 300 s, so a controller rewrites it as a heartbeat. Total load = N x running shards.
#   write_control.sh N [pilot|run]      {"concurrency_per_shard": N, "written_at": ..., "reason": "manual"}
#   write_control.sh pause [pilot|run]  no new request starts until a later write sets N
set -euo pipefail
source "$(dirname "$0")/env.sh"
VALUE=${1:?usage: write_control.sh N|pause [pilot|run]} SUB=${2:-run}
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
case "$VALUE" in
  pause) BODY=$(printf '{"pause": true, "written_at": "%s", "reason": "manual"}' "$NOW") ;;
  *[!0-9]* | "" | 0) echo "N must be a positive integer or pause, got $VALUE" >&2; exit 2 ;;
  *) BODY=$(printf '{"concurrency_per_shard": %d, "written_at": "%s", "reason": "manual"}' "$VALUE" "$NOW") ;;
esac
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
echo "$BODY" > "$tmp"
cd "$WT"
uv run --frozen fsutil cp "$tmp" "$ROOT/$SUB/control.json"
uv run --frozen fsutil cat "$ROOT/$SUB/control.json"
