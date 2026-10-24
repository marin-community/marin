#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source /Users/benfeuer/Documents/secrets.env >/dev/null
set +a
export AWS_ACCESS_KEY_ID="$CW_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$CW_KEY_SECRET"
TASK_S3_CONFIG=$(uv run --project /Users/benfeuer/Documents/marin python -c 'import json; from rigging.filesystem.s3_compat import fsspec_s3_conf,s3_endpoint; from rigging.filesystem.cluster_config import StoreType; print(json.dumps(fsspec_s3_conf(s3_endpoint(StoreType.COREWEAVE))))')
export FSSPEC_S3="$TASK_S3_CONFIG"
cd "$TASK_ROOT"
uv run --project marin --no-sync python scratch/collect-crossover.py --study crossover16k --state scratch/crossover16k-monitoring-state.json
uv run --project marin --no-sync python scratch/crossover-wandb-progress.py --study crossover16k
