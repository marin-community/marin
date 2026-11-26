#!/bin/bash
# The SAMPLE QC campaign for the taskspec re-pin's staged panels: fetch the landed
# marin revision, stage the frozen queue, and review the seed-zero panels with the
# pinned V4 reviewer. The reviewer serves as Iris child jobs at the campaign's
# priority; the review artifacts land under the output root.
#
# Args:
#   $1 = the landed marin revision
#   $2 = the frozen queue URI (object storage)
#   $3 = the queue sha256
#   $4 = the output root (object storage)
#   $5 = the review cache root (object storage)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "SAMPLE-QC FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
QUEUE_URI="${2:?frozen queue URI}"
QUEUE_SHA="${3:?queue sha256}"
OUTPUT_ROOT="${4:?output root}"
REVIEW_CACHE="${5:?review cache root}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/sample_qc.py" \
  || { echo "sample qc: the extracted tree is missing the campaign module" >&2; exit 1; }
cd "marin-$REVISION"

# The pod has no s3-capable curl; read the frozen queue through rigging's StoragePath.
uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python - <<PY
from rigging.filesystem.storage_path import StoragePath
from pathlib import Path
Path("/workspace/queue.json").write_bytes(StoragePath("$QUEUE_URI").read_bytes())
PY
test -s /workspace/queue.json

uv run --frozen --prerelease=allow --package marin-core --extra cpu \
  --with-editable ./lib/taskcompendium[pipeline] python -m \
  experiments.post_training.task_curation.sample_qc \
  --queue /workspace/queue.json \
  --queue-sha256 "$QUEUE_SHA" \
  --output-root "$OUTPUT_ROOT" \
  --review-cache "$REVIEW_CACHE" \
  --priority interactive \
  --run
echo "sample qc complete: $OUTPUT_ROOT"
