#!/bin/bash
# The pending-intake conversion for the repaired SWE-rebench V2 source: fetch the landed
# marin revision from GitHub, stage the license receipts from object storage, then run the
# driver's QUICK conversion (intake -> convert -> manifest with the within-source dedup
# census).
#
# Args:
#   $1 = the landed marin revision (fetched as a GitHub archive tarball)
#   $2 = the intake output root (object storage)
#   $3 = the license receipts object uri (object storage)
#   $4 = the source name (default: swe-rebench-v2-1-repaired)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail

REVISION="${1:?landed marin revision}"
OUTPUT_ROOT="${2:?intake output root}"
LICENSE_URI="${3:?license receipts object uri}"
SOURCE_NAME="${4:-swe-rebench-v2-1-repaired}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
curl -fsSL "https://github.com/marin-community/marin/archive/$REVISION.tar.gz" | tar xz
cd "marin-$REVISION"

STAGER=experiments.post_training.task_curation.datasets.swe_rebench_v2.stage_intake_inputs
uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python -m "$STAGER" \
  --uri "$LICENSE_URI" --output "$WORKDIR/license-receipts.json"

uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python -m \
  experiments.post_training.task_curation.driver \
  --source "$SOURCE_NAME" --mode quick \
  --license-receipts "$WORKDIR/license-receipts.json" \
  --artifact-root "$OUTPUT_ROOT/artifacts" \
  --output-root "$OUTPUT_ROOT" \
  --max-workers 16 --run
