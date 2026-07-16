#!/usr/bin/env bash
# Local smoke for the patched Ballista image: scheduler + 2 executors run a scan, a
# GROUP BY (shuffle), and a join over a baked-in Parquet fixture. No cluster, no GCS.
#
#   ./smoke.sh            # build the base image if missing, then run
#   IMAGE=... ./smoke.sh  # use a prebuilt base image tag
set -euo pipefail
cd "$(dirname "$0")"

BASE="${IMAGE:-smallquery-ballista:local}"
SMOKE_IMAGE="smallquery-ballista-smoke:local"
COMPOSE="docker compose -f docker-compose.smoke.yml"

if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  echo "building base image $BASE ..."
  docker buildx build --load -t "$BASE" -f Dockerfile ..
fi

echo "generating fixture ..."
uv run --with pyarrow python smoke/gen_fixture.py

# Bake the fixture + queries onto the base image so no bind mounts are needed.
echo "building smoke overlay image ..."
docker buildx build --load --build-arg BASE="$BASE" -t "$SMOKE_IMAGE" -f smoke/Dockerfile smoke

$COMPOSE up -d
cleanup() { $COMPOSE down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Retry until the executors have registered with the scheduler (or give up).
out=""
for attempt in $(seq 1 20); do
  if out=$($COMPOSE run --rm --no-deps scheduler \
      ballista-cli --host scheduler --port 50050 -f /queries.sql 2>&1); then
    if grep -q "scan_count" <<<"$out"; then
      break
    fi
  fi
  echo "attempt $attempt: cluster not ready yet ..."
  sleep 3
done

echo "----- query output -----"
echo "$out"
echo "------------------------"

fail=0
grep -qE '\|[[:space:]]*100000[[:space:]]*\|' <<<"$out" || { echo "FAIL: expected count 100000 (scan/join)"; fail=1; }
grep -qE '\|[[:space:]]*100[[:space:]]*\|' <<<"$out" || { echo "FAIL: expected group count 100"; fail=1; }

if [ "$fail" -eq 0 ]; then
  echo "SMOKE OK"
else
  echo "SMOKE FAILED"
  exit 1
fi
