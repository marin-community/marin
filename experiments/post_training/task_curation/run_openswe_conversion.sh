#!/bin/bash
# The OpenSWE keep-only conversion stage: fetch the landed marin revision, then
# run the dedicated profile's release module (pinned keep-only rows -> converted
# private TaskSpec release + per-task receipts + typed exclusions). The gate
# report must be staged: its divergent rows are the conversion's typed
# fidelity_gate_ exclusions.
#
# Args:
#   $1 = the landed marin revision (fetched as a GitHub archive tarball)
#   $2 = the staged condensed base bindings URI
#   $3 = the staged fidelity gate report prefix
#   $4 = the artifact version (the release repo suffix)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "OPENSWE-CONVERSION FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
CONDENSED_BINDINGS="${2:?condensed base bindings URI}"
GATE_ROOT="${3:?fidelity gate report prefix}"
VERSION="${4:?artifact version}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/datasets/openswe/release.py" \
  || { echo "openswe conversion: the extracted tree is missing the release module" >&2; exit 1; }
cd "marin-$REVISION"

OUTPUT_ROOT="$WORKDIR/openswe-conversion"
REPORT="$OUTPUT_ROOT/conversion-result.json"

uv run --frozen --prerelease=allow --with-editable ./lib/taskcompendium[pipeline] \
  --with-editable ./lib/shellbox[daytona] python -m \
  experiments.post_training.task_curation.datasets.openswe.release \
  --condensed-bindings "$CONDENSED_BINDINGS" \
  --gate-root "$GATE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --artifact-version "$VERSION" \
  --converter-revision "$REVISION" \
  --use train \
  --report "$REPORT" \
  --run
cat "$REPORT"
