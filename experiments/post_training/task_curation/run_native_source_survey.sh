#!/bin/bash
# The native source-identity survey job: build the re-pinned native conversions'
# source fingerprints over their staged populations and run the deterministic
# content-NN neighbor survey in-region. Prints the SURVEY RESULT JSON that the
# FingerprintReference registry append consumes.
#
# Args:
#   $1 = the revision that carries this runbook and the driver (GitHub archive tarball)
#   $2 = the survey root (regional source-identity prefix)
#   $3 = the served identity survey artifact URI
#   $4 = the served identity survey artifact SHA-256
#   $5 = the staged population version (e.g. 2026.11.26.3)
#   $6 = comma-separated source assignments source_id=expected_tasks
#   $7 = allocated CPUs
#   $8 = workers (default 4)
# Env:
#   HF_TOKEN  = read access to the private atlas registry (the locator pointer)
#   WORKDIR   = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "NATIVE SOURCE SURVEY FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?revision}"
SURVEY_ROOT="${2:?survey root}"
SERVED_SURVEY_URI="${3:?served survey uri}"
SERVED_SURVEY_SHA256="${4:?served survey sha256}"
POPULATION_VERSION="${5:?population version}"
ASSIGNMENTS="${6:?comma-separated source_id=expected_tasks}"
ALLOCATED_CPUS="${7:?allocated cpus}"
WORKERS="${8:-4}"

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/native_source_survey.py" \
  || { echo "native source survey: the extracted tree is missing the driver module" >&2; exit 1; }
cd "marin-$REVISION"

SOURCE_ARGS=()
IFS=',' read -r -a assignments <<< "$ASSIGNMENTS"
for assignment in "${assignments[@]}"; do
  SOURCE_ARGS+=(--source "$assignment")
done

uv run --frozen --prerelease=allow --with-editable ./lib/taskcompendium[pipeline] \
  python -m experiments.post_training.task_curation.native_source_survey \
  --population-version "$POPULATION_VERSION" \
  --survey-root "$SURVEY_ROOT" \
  --served-survey-uri "$SERVED_SURVEY_URI" \
  --served-survey-sha256 "$SERVED_SURVEY_SHA256" \
  --allocated-cpus "$ALLOCATED_CPUS" \
  --workers "$WORKERS" \
  "${SOURCE_ARGS[@]}"
