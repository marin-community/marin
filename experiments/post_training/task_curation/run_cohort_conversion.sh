#!/bin/bash
# The TaskSpec-only conversion stage for one native Harbor cohort: fetch the landed
# marin revision from GitHub, then run the cohort conversion release module (license
# gates -> strict row conversion -> private TaskSpec release + per-task receipts).
#
# Args:
#   $1 = the landed marin revision (fetched as a GitHub archive tarball)
#   $2 = the cohort source id (e.g. TaskTrove:laion__exp_rpt_ghactions-v3)
#   $3 = the conversion output root (object storage)
#   $4 = the artifact version (the release repo suffix)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "COHORT-CONVERSION FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
SOURCE_ID="${2:?cohort source id}"
OUTPUT_ROOT="${3:?conversion output root}"
VERSION="${4:?artifact version}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/harbor_cohort_release.py" \
  || { echo "cohort conversion: the extracted tree is missing the release module" >&2; exit 1; }
cd "marin-$REVISION"

# The module resolves the cohort pin, its license receipt and the publication gates
# from the landed declarations; the conversion is strict and publishes only when the
# whole population converts.
uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python -m \
  experiments.post_training.task_curation.harbor_cohort_release \
  --source-id "$SOURCE_ID" \
  --output-root "$OUTPUT_ROOT" \
  --artifact-version "$VERSION" \
  --report "$OUTPUT_ROOT/conversion-result.json" \
  --run
