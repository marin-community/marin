#!/bin/bash
# The OpenSWE fidelity gate job: fetch the landed marin revision, gate the
# sampled keep-only rows on Daytona (both arms run the oracle and the no-op
# control), and stage the per-row evidence and the result to the regional
# bucket. The gate runs on the Daytona eval account (declarative builds).
#
# Args:
#   $1 = the landed marin revision (fetched as a GitHub archive tarball)
#   $2 = the staged condensed base bindings URI
#   $3 = the staged upstream-fidelity base bindings URI
#   $4 = the regional bucket root for the gate report
#   $5 = rows sampled per base (default 20)
#   $6 = concurrent task machines (default 4)
#   $7 = optional comma-separated base filter (empty = every base)
#   $8 = optional sampling seed (empty = the module default)
# Env:
#   WORKDIR = the scratch directory (default /workspace)
set -euo pipefail
set -x
trap 'echo "FIDELITY GATE FAILED: line $LINENO: $BASH_COMMAND" >&2' ERR

REVISION="${1:?landed marin revision}"
CONDENSED_BINDINGS="${2:?condensed base bindings URI}"
UPSTREAM_BINDINGS="${3:?upstream-fidelity base bindings URI}"
REPORT_ROOT="${4:?regional report root}"
PER_BASE="${5:-20}"
CONCURRENCY="${6:-4}"
BASES="${7:-}"
SEED="${8:-}"

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
TARBALL="https://github.com/marin-community/marin/archive/$REVISION.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "marin-$REVISION.tar.gz" "$TARBALL"
tar -tzf "marin-$REVISION.tar.gz" > /dev/null
tar -xzf "marin-$REVISION.tar.gz" -C "$WORKDIR"
test -f "marin-$REVISION/experiments/post_training/task_curation/datasets/openswe/fidelity_gate.py" \
  || { echo "fidelity gate: the extracted tree is missing the gate module" >&2; exit 1; }
cd "marin-$REVISION"

REPORT="$WORKDIR/fidelity-gate"

BASE_ARGS=()
if [ -n "$BASES" ]; then
  BASE_ARGS+=(--bases "$BASES")
fi
SEED_ARGS=()
if [ -n "$SEED" ]; then
  SEED_ARGS+=(--seed "$SEED")
fi

uv run --frozen --prerelease=allow --with-editable ./lib/taskcompendium[pipeline] \
  --with-editable "./lib/shellbox[daytona]" \
  python -m experiments.post_training.task_curation.datasets.openswe.fidelity_gate \
  --condensed-bindings "$CONDENSED_BINDINGS" \
  --upstream-bindings "$UPSTREAM_BINDINGS" \
  --report-dir "$REPORT" \
  --per-base "$PER_BASE" \
  --concurrency "$CONCURRENCY" \
  "${BASE_ARGS[@]+"${BASE_ARGS[@]}"}" \
  "${SEED_ARGS[@]+"${SEED_ARGS[@]}"}"

uv run --no-sync fsutil cp -r "$REPORT" "$REPORT_ROOT/"
