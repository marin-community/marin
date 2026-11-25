#!/bin/bash
# The staged-population census job for one accepted conversion release: fetch
# the landed marin revision, measure the release's declared Daytona demands,
# verify the cohort's passing acceptance run live (fail closed), classify
# through the declarative path per the coordinator's ruling, and stage the
# census artifact and the ActorCensusReceipt shards to the regional bucket.
#
# Args:
#   $1 = the landed marin revision
#   $2 = the cohort source id
#   $3 = the published conversion release repo
#   $4 = the cohort's passing acceptance Iris job name
#   $5 = the census root (object storage)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "STAGED-CENSUS FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
SOURCE_ID="${2:?cohort source id}"
RELEASE_REPO="${3:?conversion release repo}"
ACCEPTANCE_JOB="${4:?acceptance job name}"
CENSUS_ROOT="${5:?census root}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/staged_census.py" \
  || { echo "staged census: the extracted tree is missing the census module" >&2; exit 1; }
cd "marin-$REVISION"

SLUG=$(printf '%s' "$SOURCE_ID" | sed -e 's/^[^:]*://' -e 's#[/@]#-#g' | tr '[:upper:]' '[:lower:]')
CENSUS_OUT="$WORKDIR/census/$SLUG"
RELEASE_REVISION=$(uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python - <<PY
from huggingface_hub import HfApi
print(HfApi().repo_info("$RELEASE_REPO", repo_type="dataset").sha)
PY
)

uv run --frozen --prerelease=allow --with-editable ./lib/taskcompendium[pipeline] \
  --with-editable "./lib/shellbox[daytona]" \
  --with-editable ./lib/rolloutengine python -m \
  experiments.post_training.task_curation.staged_census \
  --source-id "$SOURCE_ID" \
  --release-repo "$RELEASE_REPO" \
  --release-revision "$RELEASE_REVISION" \
  --converter-version "harbor-cohort-taskspec-v1" \
  --converter-revision "$REVISION" \
  --acceptance-job "$ACCEPTANCE_JOB" \
  --output "$CENSUS_OUT" \
  --receipts-output "$CENSUS_OUT-receipts" \
  --report "$CENSUS_OUT/census-result.json"

uv run --no-sync fsutil cp -r "$CENSUS_OUT" "$CENSUS_ROOT/$SLUG/"
uv run --no-sync fsutil cp -r "$CENSUS_OUT-receipts" "$CENSUS_ROOT/$SLUG-receipts/"
