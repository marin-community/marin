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

# Checkpoint 121638 resumes at global_step 121638; parent W&B _step 121637
# was verified to record global_step 121637 before the replayed window.
RUN_ID=$(uv run python -m experiments.grug.moe_hero_ep.current_run)
HANDOFF_CHECKPOINT=s3://hero-checkpoints/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/hero-nopdl-step108k/2026.08.19.2/checkpoints/step-121638
WANDB_FORK_FROM='hero-nopdl-step108k?_step=121637'
WANDB_PROJECT=marin_moe
IRIS_CONFIG=lib/iris/config/marin.yaml
HERO_ISSUE=https://github.com/marin-community/marin/issues/8506

if [[ "$RUN_ID $HANDOFF_CHECKPOINT $WANDB_FORK_FROM" == *REPLACE_BEFORE_DEPLOYMENT* ]]; then
  echo "Handoff is not finalized: replace the run ID, checkpoint, and W&B fork point." >&2
  exit 1
fi
: "${WANDB_API_KEY:?Set WANDB_API_KEY before you start the hero.}"
if ! command -v gh >/dev/null; then
  echo "GitHub CLI 'gh' is not installed; the launch record cannot be posted." >&2
  exit 1
fi
tree_status=$(git status --porcelain --untracked-files=all)
if [[ -n "$tree_status" ]]; then
  echo "Launch and fork creation require a pristine worktree." >&2
  exit 1
fi

launch_commit=$(git rev-parse HEAD)
if [[ "$mode" == fork-wandb ]]; then
  git fetch --quiet origin main
  main_commit=$(git rev-parse FETCH_HEAD)
  if [[ "$launch_commit" != "$main_commit" ]]; then
    echo "Check out the fetched origin/main commit ${main_commit} before creating the fork." >&2
    exit 1
  fi
fi

# Create the tracker lineage once, outside the coordinator and training retry loops.
# launch verifies the same child and lets training resume it without fork_from.
uv run python - "$mode" "$RUN_ID" "$WANDB_PROJECT" "$WANDB_FORK_FROM" "$HANDOFF_CHECKPOINT" "$launch_commit" "$IRIS_CONFIG" <<'PYTHON'
import csv
import io
import os
import re
import subprocess
import sys

import wandb
from iris.cluster.types import TERMINAL_JOB_STATES
from levanter.tracker.wandb import _WANDB_FORK_FROM_PATTERN

mode, run_id, project, fork_from, checkpoint, launch_commit, iris_config = sys.argv[1:]
entity = "marin-community"
lineage = {
    "hero_handoff_checkpoint": checkpoint,
    "hero_wandb_fork_from": fork_from,
    "hero_launch_commit": launch_commit,
}
fork = _WANDB_FORK_FROM_PATTERN.fullmatch(fork_from)
if fork is None:
    raise ValueError(f"WANDB_FORK_FROM must have the form '<parent-run-id>?_step=<step>': {fork_from}")
parent_id = fork["run_id"]
if parent_id == run_id:
    raise ValueError("WANDB_FORK_FROM must name a different run from RUN_ID")
checkpoint_step = re.fullmatch(r"step-(\d+)", checkpoint.rstrip("/").rsplit("/", 1)[-1])
if checkpoint_step is None:
    raise ValueError(f"HANDOFF_CHECKPOINT must end in step-<N>: {checkpoint}")
# The child resumes at checkpoint step N and logs training step N first; W&B starts a fork at
# _step + 1, so any fork point at or past N makes the child's rows fail step monotonicity.
if int(fork["step"]) >= int(checkpoint_step[1]):
    raise ValueError(f"W&B fork _step {fork['step']} must precede checkpoint step {checkpoint_step[1]}")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or not re.fullmatch(r"[A-Za-z0-9_.-]+", parent_id):
    raise ValueError("Run IDs may contain only letters, digits, '_', '.', and '-'")

if mode == "fork-wandb":
    api = wandb.Api()
    if list(api.runs(f"{entity}/{project}", filters={"name": run_id}, per_page=1)):
        raise ValueError(f"W&B child {run_id} already exists; inspect it before using launch")
    # W&B rejects fork_from alongside resume, and the operator's shell may export either.
    os.environ.pop("WANDB_RESUME", None)
    os.environ.pop("WANDB_RESUME_FROM", None)
    with wandb.init(
        entity=entity, project=project, id=run_id, name=run_id,
        fork_from=fork_from, config=lineage, mode="online",
    ):
        pass
else:
    # Root coordinators sit at depth 1 under the submitting user's namespace. Match every
    # namespace so a parent relaunched under another IRIS_USER still blocks the child.
    coordinator_sql = (
        "SELECT job_id, state FROM jobs WHERE depth = 1 AND "
        f"(job_id LIKE '%/{parent_id}-coord-%' OR job_id LIKE '%/{run_id}-coord-%')"
    )
    result = subprocess.run(
        ["uv", "run", "iris", "--config", iris_config, "query", "-f", "csv", coordinator_sql],
        check=True, capture_output=True, text=True,
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    if reader.fieldnames != ["job_id", "state"]:
        raise ValueError("Unknown coordinator state: unexpected Iris query response")
    # Terminal parent rows age out of the controller after its job retention window, so an
    # absent coordinator is the safe state; only a live one blocks the launch.
    for row in reader:
        if int(row["state"]) not in TERMINAL_JOB_STATES:
            raise ValueError(f"Coordinator is not terminal: {row['job_id']}")
    child = wandb.Api().run(f"{entity}/{project}/{run_id}")
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
launch_job_name="${RUN_ID}-coord-${short_uuid}"

echo "Recording hero launch on ${HERO_ISSUE}"
launch_record_file=$(mktemp)
trap 'rm -f "$launch_record_file"' EXIT
printf 'Hero launch requested.\n\n- Run ID: `%s`\n- Commit: `%s`\n- Coordinator job: `%s`\n- Target: `%s` (%s)\n\nW&B fork: `%s`; handoff checkpoint: `%s`.\n' \
  "$RUN_ID" "$launch_commit" "$launch_job_name" "$TARGET_CLUSTER" "$TARGET_DESCRIPTION" \
  "$WANDB_FORK_FROM" "$HANDOFF_CHECKPOINT" > "$launch_record_file"
gh issue comment "$HERO_ISSUE" --body-file "$launch_record_file"
echo "Launching hero from commit ${launch_commit}"

IRIS_USER=marin uv run iris --config "$IRIS_CONFIG" job run --no-wait --enable-extra-resources \
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
