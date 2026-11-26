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
#   $6 = the staged population record path (empty = measure the release's tasks.parquet)
#   $7 = the converter revision to bind (empty = this checkout)
#   $8 = the content-identity proof URI (optional; referenced in the census basis)
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
POPULATION_RECORD="${6:-}"
CONVERTER_REVISION="${7:-}"
CONTENT_IDENTITY_PROOF="${8:-}"
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

POPULATION_ARGS=()
if [ -n "$POPULATION_RECORD" ]; then
  POPULATION_ARGS+=(--population-record "$POPULATION_RECORD")
fi
if [ -n "$CONTENT_IDENTITY_PROOF" ]; then
  POPULATION_ARGS+=(--content-identity-proof "$CONTENT_IDENTITY_PROOF")
fi

uv run --frozen --prerelease=allow --with-editable ./lib/taskcompendium[pipeline] \
  --with-editable "./lib/shellbox[daytona]" \
  --with-editable ./lib/rolloutengine python -m \
  experiments.post_training.task_curation.staged_census \
  --source-id "$SOURCE_ID" \
  --release-repo "$RELEASE_REPO" \
  --release-revision "$RELEASE_REVISION" \
  --converter-version "harbor-cohort-taskspec-v1" \
  --converter-revision "${CONVERTER_REVISION:-$REVISION}" \
  --acceptance-job "$ACCEPTANCE_JOB" \
  "${POPULATION_ARGS[@]+"${POPULATION_ARGS[@]}"}" \
  --output "$CENSUS_OUT" \
  --receipts-output "$CENSUS_OUT-receipts" \
  --report "$CENSUS_OUT/census-result.json"

uv run --no-sync fsutil cp -r "$CENSUS_OUT" "$CENSUS_ROOT/$SLUG/"
uv run --no-sync fsutil cp -r "$CENSUS_OUT-receipts" "$CENSUS_ROOT/$SLUG-receipts/"
if [ -n "$CONTENT_IDENTITY_PROOF" ]; then
  # The staged population path doubles as the proof document's locator.
  uv run --no-sync fsutil cp "$POPULATION_RECORD/.artifact.json" "$CENSUS_ROOT/$SLUG-receipts/staged-population.record.json"
fi
