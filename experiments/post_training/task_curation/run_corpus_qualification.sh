#!/bin/bash
# One corpus oracle-qualification shard: fetch the landed marin revision, then
# run the qualification driver over the shard's index-range slice of the
# repaired population. Every row runs the fail-closed no-op + oracle controls
# on its own Daytona EVAL sandbox; the shard's report lands in the gate's
# report shapes under the shard's prefix.
#
# Args:
#   $1 = the landed marin revision (fetched as a GitHub archive tarball)
#   $2 = the staged condensed base bindings URI
#   $3 = the gate root whose exclusions and sample records the corpus run carries
#   $4 = the shard index
#   $5 = the shard count
#   $6 = the report root (object storage; the shard stages to <root>/shard-<index>/)
#   $7 = the sandbox concurrency for this shard
# Env:
#   WORKDIR    = the scratch directory (default /workspace)
#   DAYTONA_*  = the eval account credentials
set -euo pipefail
set -x
trap 'echo "CORPUS-QUALIFICATION FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
CONDENSED_BINDINGS="${2:?condensed base bindings URI}"
GATE_ROOT="${3:?gate root}"
SHARD_INDEX="${4:?shard index}"
SHARD_COUNT="${5:?shard count}"
REPORT_ROOT="${6:?report root}"
CONCURRENCY="${7:?sandbox concurrency}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/datasets/openswe/corpus_qualification.py" \
  || { echo "corpus qualification: the extracted tree is missing the driver module" >&2; exit 1; }
cd "marin-$REVISION"

REPORT_DIR="$WORKDIR/qualification/shard-$SHARD_INDEX"
uv run --frozen --prerelease=allow --with-editable ./lib/taskcompendium[pipeline] \
  --with-editable ./lib/shellbox[daytona] python -m \
  experiments.post_training.task_curation.datasets.openswe.corpus_qualification \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --condensed-bindings "$CONDENSED_BINDINGS" \
  --gate-root "$GATE_ROOT" \
  --report-dir "$REPORT_DIR" \
  --concurrency "$CONCURRENCY"
uv run --no-sync fsutil cp -r "$REPORT_DIR/." "$REPORT_ROOT/shard-$SHARD_INDEX/"
echo "corpus qualification shard $SHARD_INDEX complete"
