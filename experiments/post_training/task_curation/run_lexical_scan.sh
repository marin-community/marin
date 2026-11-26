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
#   $7 = the lexical report URI (object storage) the local report copies to
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
REPORT_URI="${7:?lexical report URI}"
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

# The pod has no s3-capable curl; read the manifests through rigging's StoragePath.
uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python - <<PY
from rigging.filesystem.storage_path import StoragePath
from pathlib import Path
Path("/workspace/populations.json").write_bytes(StoragePath("$POPULATIONS_URI").read_bytes())
Path("/workspace/source-ids.json").write_bytes(StoragePath("$SOURCE_IDS_URI").read_bytes())
PY
test -s /workspace/populations.json && test -s /workspace/source-ids.json

# The DF masks rebuild for the new cohort set (no frozen drop sets: the new
# populations are not in the recorded artifact); the bloom is the pinned
# shared one and the population set is retained. The cpu/dedup extras live on
# the marin-core package (lib/marin), not the root, so resolve them there.
# The scan writes its report with pathlib (no parent creation), so the parent
# must exist; the scan itself resumes the S3-materialized drop sets and marks.
REPORT_LOCAL="$WORKDIR/lexical/lexical-report-$POPULATION_VERSION.json"
mkdir -p "$WORKDIR/lexical"
uv run --frozen --prerelease=allow --package marin-core --extra cpu --extra dedup \
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
  --output "$REPORT_LOCAL" \
  --run
test -s "$REPORT_LOCAL"

# Publish the report to its object-storage URI (the pod's runtime config
# provides the S3 credentials rigging uses).
uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python - <<PY
from rigging.filesystem.storage_path import StoragePath
from pathlib import Path
StoragePath("$REPORT_URI").write_bytes(Path("$REPORT_LOCAL").read_bytes())
PY
test -s "$REPORT_LOCAL"
echo "lexical scan complete: $REPORT_LOCAL published to $REPORT_URI"
