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
#   TARBALL_URI = the source tarball uri (default: the GitHub archive for $REVISION; it must
#     be an http(s) url — the tree does not exist when it downloads)
set -euo pipefail
set -x
trap 'echo "INTAKE-CONVERSION FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
TARBALL_URI="${TARBALL_URI:-https://github.com/marin-community/marin/archive/$REVISION.tar.gz}"
OUTPUT_ROOT="${2:?intake output root}"
LICENSE_URI="${3:?license receipts object uri}"
SOURCE_NAME="${4:-swe-rebench-v2-1-repaired}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
# The repo tree does not exist yet, so the tarball download must not need the repo's own
# modules: curl handles http(s) uris (the GitHub archive default). The object-storage stager
# only runs after extraction (the license receipts below).
case "$TARBALL_URI" in
  https://* | http://*)
    curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL_URI"
    ;;
  *)
    echo "intake conversion: TARBALL_URI must be an http(s) url before the tree exists (got $TARBALL_URI)" >&2
    exit 1
    ;;
esac
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/datasets/swe_rebench_v2/run_intake_conversion.sh" \
  || { echo "intake conversion: the extracted tree is missing the intake script" >&2; exit 1; }
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
