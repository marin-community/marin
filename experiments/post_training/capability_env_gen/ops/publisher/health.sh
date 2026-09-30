#!/usr/bin/env bash
# Publisher health: heartbeat age/state, queue depth, and every non-published return.
#
#   ops/publisher/health.sh                 # human summary; exit 0 OK, 1 ALARM, 2 queue unreadable
#   ops/publisher/health.sh --json          # full per-packet summary
#   ops/publisher/health.sh --max-age 60    # stricter staleness alarm (default 120 s = 8 polls)
#
# Suitable for a cron/Monitor alarm: a dead, wedged or degraded publisher exits 1.
set -euo pipefail
set +x
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MARIN="${MARIN:-$HOME/openathena/marin-construct}"
QUEUE_URI="${QUEUE_URI:-s3://marin-us-east-02a/users/muchanem/capability-pipeline/publication}"
CW_KEY_ID="$(gcloud secrets versions access latest --secret=cw-object-storage-key-id --project=hai-gcp-models 2>/dev/null || true)"
CW_KEY_SECRET="$(gcloud secrets versions access latest --secret=cw-object-storage-key-secret --project=hai-gcp-models 2>/dev/null || true)"
[ -n "$CW_KEY_ID" ] && [ -n "$CW_KEY_SECRET" ] || { echo "health: object-store keys unavailable" >&2; exit 2; }
export CW_KEY_ID CW_KEY_SECRET
cd "$MARIN"
PYTHONPATH="$ROOT" exec uv run --frozen "$HERE/publisher_kit.py" health --queue "$QUEUE_URI" "$@"
