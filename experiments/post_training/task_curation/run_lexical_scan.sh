#!/bin/bash
# The lexical scan for the staged conversion-release populations: fetch the
# landed marin revision, then measure the staged populations against the pinned
# shared bloom with the DF masks rebuilt for the new cohort set. The report
# artifact lands under the lexical scan root; the job log carries the result.
#
# Args:
#   $1 = the landed marin revision
#   $2 = the staged populations manifest URI (object storage)
#   $3 = the staged source-ids URI (object storage)
#   $4 = the staged population version
#   $5 = the lexical artifact version
#   $6 = the max workers
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "LEXICAL-SCAN FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
POPULATIONS_URI="${2:?staged populations manifest}"
SOURCE_IDS_URI="${3:?staged source ids}"
POPULATION_VERSION="${4:?staged population version}"
ARTIFACT_VERSION="${5:?lexical artifact version}"
MAX_WORKERS="${6:-16}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/lexical_scan.py" \
  || { echo "lexical scan: the extracted tree is missing the scan module" >&2; exit 1; }
cd "marin-$REVISION"

curl -fsSL --retry 5 "$POPULATIONS_URI" -o /workspace/populations.json
curl -fsSL --retry 5 "$SOURCE_IDS_URI" -o /workspace/source-ids.json

# The DF masks rebuild for the new cohort set (no frozen drop sets: the new
# populations are not in the recorded artifact); the bloom is the pinned
# shared one and the population set is retained.
uv run --frozen --prerelease=allow --extra cpu --extra dedup \
  --with-editable ./lib/taskcompendium[pipeline] python -m \
  experiments.post_training.task_curation.lexical_scan \
  --populations /workspace/populations.json \
  --source-ids /workspace/source-ids.json \
  --bloom-path "s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_eca06d42" \
  --eval-corpus-version "aa-index-v4.1.1-v3" \
  --population-set retained \
  --artifact-version "$ARTIFACT_VERSION" \
  --artifact-root "s3://marin-us-east-02a/marin" \
  --max-workers "$MAX_WORKERS" \
  --output "/workspace/lexical/lexical-report-$POPULATION_VERSION.json" \
  --run
echo "lexical scan complete: /workspace/lexical/lexical-report-$POPULATION_VERSION.json"
