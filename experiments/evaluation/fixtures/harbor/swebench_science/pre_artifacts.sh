#!/bin/sh
set -eu

# Pier runs this hook in the task workdir after the agent exits and before a
# separate verifier starts. Always create the artifact, including for a clean
# or timed-out agent run, so the verifier receives an explicit empty patch.
task_dir="${SCI_BENCH_TASK_DIR:-$PWD}"
artifacts_dir="${ENV_ARTIFACTS_PATH:-/logs/artifacts}"
patch="$artifacts_dir/model.patch"

mkdir -p "$artifacts_dir"
: > "$patch"

cd "$task_dir"
git config --global --add safe.directory "$task_dir"
roots="$(git rev-list --max-parents=0 --reverse HEAD)"
set -- $roots
if [ "$#" -ne 1 ]; then
    echo "task repository must have exactly one baseline root commit" >&2
    exit 1
fi
baseline="$1"
git add -A
git diff --cached --binary "$baseline" > "$patch"
