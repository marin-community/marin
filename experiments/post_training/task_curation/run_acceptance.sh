#!/bin/bash
# The acceptance-control stage for one published cohort-conversion release: fetch
# the landed marin revision from GitHub, then replay the recorded negative
# controls against the converted graders on the Daytona RL org. The job exports
# the judge keys; the scratch stays on the pod and the job log carries the result.
#
# Args:
#   $1 = the landed marin revision (fetched as a GitHub archive tarball)
#   $2 = the cohort source id
#   $3 = the published conversion release repo
#   $4 = how many tasks to accept (default 12)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "ACCEPTANCE FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
SOURCE_ID="${2:?cohort source id}"
RELEASE_REPO="${3:?conversion release repo}"
SAMPLE="${4:-12}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/acceptance_controls.py" \
  || { echo "acceptance: the extracted tree is missing the acceptance module" >&2; exit 1; }
cd "marin-$REVISION"

SLUG=$(printf '%s' "$SOURCE_ID" | sed -e 's/^[^:]*://' -e 's#[/@]#-#g' | tr '[:upper:]' '[:lower:]')
REPORT="$WORKDIR/acceptance/$SLUG.json"

RELEASE_REVISION=$(uv run --frozen --with-editable ./lib/taskcompendium[pipeline] python - <<PY
from huggingface_hub import HfApi
print(HfApi().repo_info("$RELEASE_REPO", repo_type="dataset").sha)
PY
)

# The sandbox graders run on Daytona through the rollout engine's machine
# machinery, so the environment carries the shellbox Daytona extra and the
# rollout engine alongside the taskcompendium pipeline extra.
uv run --frozen --with-editable ./lib/taskcompendium[pipeline] \
  --with-editable "./lib/shellbox[daytona]" \
  --with-editable ./lib/rolloutengine python -m \
  experiments.post_training.task_curation.acceptance_controls \
  --source-id "$SOURCE_ID" \
  --release-repo "$RELEASE_REPO" \
  --release-revision "$RELEASE_REVISION" \
  --sample "$SAMPLE" \
  --report "$REPORT"
cat "$REPORT"
