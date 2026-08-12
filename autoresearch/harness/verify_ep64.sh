#!/usr/bin/env bash
# autoresearch verify: one EP64 ladder-rung benchmark on cw-us-east-08a (1 GB200 rack).
# Prints exactly one number (tokens/s over the scored window) on the LAST line.
# Exits nonzero if: submit/poll fails, run dies, any scored loss is non-finite,
# or drop fraction over the last DROP_WINDOW steps exceeds DROP_BUDGET.
set -euo pipefail

REPO=/home/marin/projects/marin
cd "$REPO"

SIZE="${AR_SIZE:-d1024}"                 # 1-rack rungs only: d768|d1024|d1536 (d2048 needs 4 racks — excluded)
FLAVOR="${AR_FLAVOR:-ep}"                # hero EP arm: fixed_all_to_all, E192 top-4, latent d/2, CF from template
NUM_STEPS="${AR_NUM_STEPS:-1000}"
DROP_BUDGET="${AR_DROP_BUDGET:-0.02}"   # user constraint: trained-router-regime drop rate <= 2%
DROP_WINDOW="${AR_DROP_WINDOW:-50}"      # mirror the ladder's "Drop % (last 50)"
SCORE_START="${AR_SCORE_START:-100}"     # throughput window start (skip compile/warmup)
TIMEOUT_SECONDS="${AR_TIMEOUT:-21600}"   # 6h; a 5400s timeout has hidden a good result before
ITER="${AR_ITER:-x}"
SHA="$(git rev-parse --short HEAD)"
RUN_ID="ar8062-${ITER}-${SHA}"
# Unique JAX coordinator port per iteration (default 8476 is shared cluster-wide).
PORT=$((33000 + $(cksum <<<"$RUN_ID" | cut -d' ' -f1) % 999))
IRIS=(uv run iris --config lib/iris/config/marin.yaml)

# --- Rack-quota guard: never run a second verify rack concurrently (2-rack cap incl. guard node).
# Job paths are namespaced (/mwittmann/...) and states print lowercase.
running=$("${IRIS[@]}" job list --prefix "/mwittmann/ar8062-" 2>/dev/null | grep -vE 'guard' | grep -ciE 'running|pending' || true)
if [[ "$running" -gt 0 ]]; then
  echo "quota guard: $running ar8062 job(s) still live; refusing to submit" >&2
  exit 1
fi

echo "submitting ${RUN_ID} (${SIZE}/${FLAVOR}, ${NUM_STEPS} steps, port ${PORT})" >&2
"${IRIS[@]}" job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${RUN_ID}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT marin_moe \
  -e IRIS_USER mwittmann -e IRIS_PORT_JAX "$PORT" \
  -- python -m experiments.grug.moe_hero_ep.small_scale_abl_launch \
     --run-id "$RUN_ID" --size "$SIZE" --flavor "$FLAVOR" \
     --num-steps "$NUM_STEPS" --steps-per-eval 100000 \
     --version "$(date +%Y.%m.%d)" --run >&2

# --- Poll to terminal state, then score (single number on the last line).
exec "$(dirname "$0")/poll_score.sh" "$RUN_ID" "$NUM_STEPS"
