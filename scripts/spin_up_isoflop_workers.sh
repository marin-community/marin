#!/bin/bash
# Spin up TPU v4 workers for the completed-adamh-v2 isoflop sweep on big-run.
# Total: 2000 new cores (assumes 6 existing v4-8 nodes = 48 cores).
# Grand total with existing: 2048 cores.

set -euo pipefail

CONFIG="infra/marin-big-run.yaml"
CAPACITY="reserved"

# 15 x v4-64 (960 cores)
for i in $(seq 1 15); do
  echo "Adding v4-64 worker $i/15..."
  uv run scripts/ray/cluster.py --config "$CONFIG" add-worker v4-64 --capacity "$CAPACITY"
done

# 20 x v4-32 (640 cores)
for i in $(seq 1 20); do
  echo "Adding v4-32 worker $i/20..."
  uv run scripts/ray/cluster.py --config "$CONFIG" add-worker v4-32 --capacity "$CAPACITY"
done

# 12 x v4-16 (192 cores)
for i in $(seq 1 12); do
  echo "Adding v4-16 worker $i/12..."
  uv run scripts/ray/cluster.py --config "$CONFIG" add-worker v4-16 --capacity "$CAPACITY"
done

# 26 x v4-8 (208 cores) — 6 already exist, 32 needed total
for i in $(seq 1 26); do
  echo "Adding v4-8 worker $i/26..."
  uv run scripts/ray/cluster.py --config "$CONFIG" add-worker v4-8 --capacity "$CAPACITY"
done

echo "Done. 73 new workers added (2000 cores)."
echo "With 6 existing v4-8 nodes: 2048 total cores."
