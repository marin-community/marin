# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Stage 0 gate's tool-call parse rate on Terminus-2 trajectories.

Terminus-2 answers each turn with one JSON command object. When it cannot parse a reply it
performs none of its commands and tells the model so in the next observation
("Previous response had parsing errors"). The parse rate is the share of agent turns
without that error; warnings (e.g. extra text around the JSON) still parse.

    python -m experiments.post_training.stage0.agent_parse HARBOR_JOB_DIR
"""

import glob
import json
import sys

PARSE_ERROR_MARKER = "Previous response had parsing errors"
MIN_PARSE_SHARE = 0.95


def trajectory_parse_counts(trajectory: dict) -> tuple[int, int]:
    """(agent turns, agent turns whose reply failed to parse) in one trajectory."""
    turns = [step for step in trajectory["steps"] if step.get("source") == "agent"]
    failed = sum(1 for step in turns if PARSE_ERROR_MARKER in str(step.get("observation") or ""))
    return len(turns), failed


def parse_report(trajectories: list[dict]) -> dict:
    turns = failed = 0
    for trajectory in trajectories:
        trajectory_turns, trajectory_failed = trajectory_parse_counts(trajectory)
        turns += trajectory_turns
        failed += trajectory_failed
    share = (turns - failed) / turns if turns else None
    return {
        "trajectories": len(trajectories),
        "agent_turns": turns,
        "parse_failures": failed,
        "parse_share": share,
        "passed": share is not None and share >= MIN_PARSE_SHARE,
    }


def main() -> None:
    paths = sorted(glob.glob(f"{sys.argv[1]}/*/attempts/*/agent/trajectory.json"))
    trajectories = []
    for path in paths:
        with open(path) as handle:
            trajectories.append(json.load(handle))
    print(json.dumps(parse_report(trajectories)))


if __name__ == "__main__":
    main()
