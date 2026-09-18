#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != fork-wandb && "$1" != launch ) ]]; then
  echo "Usage: $0 {fork-wandb|launch}" >&2
  exit 2
fi
mode=$1
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Fill these together after the single-rack A/B and d768 validation pass.
# Verify the parent W&B _step at the checkpoint boundary before choosing the fork point.
RUN_ID=hero-main-stepREPLACE_BEFORE_DEPLOYMENT
HANDOFF_CHECKPOINT=REPLACE_BEFORE_DEPLOYMENT
WANDB_FORK_FROM='hero-nopdl-step108k?_step=REPLACE_BEFORE_DEPLOYMENT'
WANDB_PROJECT=marin_moe
HERO_ISSUE=https://github.com/marin-community/marin/issues/8506

if [[ "$RUN_ID $HANDOFF_CHECKPOINT $WANDB_FORK_FROM" == *REPLACE_BEFORE_DEPLOYMENT* ]]; then
  echo "Handoff is not finalized: replace the run ID, checkpoint, and W&B fork point." >&2
  exit 1
fi
: "${WANDB_API_KEY:?Set WANDB_API_KEY before you start the hero.}"
tree_status=$(git status --porcelain --untracked-files=all)
if [[ -n "$tree_status" ]]; then
  echo "Launch and fork creation require a pristine worktree." >&2
  exit 1
fi

launch_commit=$(git rev-parse HEAD)
main_commit=$(git rev-parse origin/main)
if [[ "$mode" == fork-wandb && "$launch_commit" != "$main_commit" ]]; then
  echo "Fetch origin/main and use its exact commit before a cutover launch." >&2
  exit 1
fi

# Create the tracker lineage once, outside the coordinator and training retry loops.
# launch verifies the same child and lets training resume it without fork_from.
uv run python - "$mode" "$RUN_ID" "$WANDB_PROJECT" "$WANDB_FORK_FROM" "$HANDOFF_CHECKPOINT" "$launch_commit" <<'PYTHON'
import csv
import io
import subprocess
import sys

import wandb
from iris.cluster.types import TERMINAL_JOB_STATES

mode, run_id, project, fork_from, checkpoint, launch_commit = sys.argv[1:]
entity = "marin-community"
lineage = {
    "hero_handoff_checkpoint": checkpoint,
    "hero_wandb_fork_from": fork_from,
    "hero_launch_commit": launch_commit,
}
api = wandb.Api()
if mode == "fork-wandb":
    if list(api.runs(f"{entity}/{project}", filters={"name": run_id}, per_page=1)):
        raise ValueError(f"W&B child {run_id} already exists; inspect it before using launch")
    with wandb.init(
        entity=entity, project=project, id=run_id, name=run_id,
        fork_from=fork_from, config=lineage, mode="online",
    ):
        pass
else:
    result = subprocess.run(
        ["uv", "run", "iris", "--config", "lib/iris/config/marin.yaml", "query", "-f", "csv",
         "SELECT job_id, state FROM jobs WHERE depth = 1"],
        check=True, capture_output=True, text=True,
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    if reader.fieldnames != ["job_id", "state"]:
        raise ValueError("Unknown coordinator state: unexpected Iris query response")
    rows = list(reader)
    parent_id = fork_from.split("?_step=", 1)[0]
    parent_prefix = f"/marin/{parent_id}-coord-"
    child_prefix = f"/marin/{run_id}-coord-"
    if not any(row["job_id"].startswith(parent_prefix) for row in rows):
        raise ValueError("Unknown parent coordinator state; refusing launch")
    for row in rows:
        if row["job_id"].startswith((parent_prefix, child_prefix)) and int(row["state"]) not in TERMINAL_JOB_STATES:
            raise ValueError(f"Coordinator is not terminal: {row['job_id']}")
    child = api.run(f"{entity}/{project}/{run_id}")
    for key, expected in lineage.items():
        if child.config.get(key) != expected:
            raise ValueError(f"W&B child {run_id} has a different {key}")
PYTHON

if [[ "$mode" == fork-wandb ]]; then
  echo "W&B fork created. Complete the cutover preflight before running $0 launch."
  exit 0
fi

TARGET_CLUSTER=cw-us-east-08a
TARGET_DESCRIPTION='11 x NVL72'
short_uuid=$(uuidgen | tr '[:upper:]' '[:lower:]')
short_uuid=${short_uuid:0:8}
launch_tree_dirty=false
launch_job_name="${RUN_ID}-coord-${short_uuid}"
launch_record=$(printf '🤖 Hero launch requested.\n\n- Run ID: `%s`\n- Commit: `%s`\n- Tree dirty: `%s`\n- Coordinator job: `%s`\n- Target: `%s` (%s)' \
  "$RUN_ID" "$launch_commit" "$launch_tree_dirty" "$launch_job_name" "$TARGET_CLUSTER" "$TARGET_DESCRIPTION")

echo "Recording hero launch on ${HERO_ISSUE}"
launch_record_file=$(mktemp)
trap 'rm -f "$launch_record_file"' EXIT
printf '%s\n\nW&B fork: `%s`; handoff checkpoint: `%s`.\n' \
  "$launch_record" "$WANDB_FORK_FROM" "$HANDOFF_CHECKPOINT" > "$launch_record_file"
agent-gh issue comment "$HERO_ISSUE" --body-file "$launch_record_file"
echo "Launching hero from commit ${launch_commit}; tree_dirty=${launch_tree_dirty}"

IRIS_USER=marin uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster "$TARGET_CLUSTER" \
  --priority system \
  --system-reason "hero run" \
  --cpu 1 \
  --memory 4g \
  --max-retries 1000 \
  --job-name "$launch_job_name" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32614 \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.75 \
  -e XLA_FLAGS "--xla_gpu_memory_limit_slop_factor=85" \
  -- python -m experiments.grug.moe_hero_ep.launch_scaling_ladder \
    --run-id "$RUN_ID" \
    --initialize-from-checkpoint "$HANDOFF_CHECKPOINT" \
    --size d6144 \
    --version 2026.08.19.2 \
    --run
