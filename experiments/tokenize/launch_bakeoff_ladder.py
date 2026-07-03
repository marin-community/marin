# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the tokenizer bake-off isoFLOP ladder: one grug-moe run per (arm, compute point).

Each cell is an ``iris job run`` of :mod:`experiments.grug.moe.launch_tokenizer_bakeoff` with a
fixed proxy model shape and a varying ``SCALE_STEPS`` (the training-FLOP axis). Every arm is
trained on its own tokenization of the same corpus and scored on the same held-out bytes, so
the resulting BPB-vs-FLOPs points are compute-comparable across arms.

Building is opt-in: without ``--run`` it prints the plan (the exact iris commands) and submits
nothing, so ``python -m experiments.tokenize.launch_bakeoff_ladder`` never starts a large sweep
by accident.

    # print the plan
    uv run python -m experiments.tokenize.launch_bakeoff_ladder --arms marin-128k,gpt-oss-200k,qwen3-152k
    # actually submit
    uv run python -m experiments.tokenize.launch_bakeoff_ladder --arms marin-128k,gpt-oss-200k --run

Collect results afterwards with :mod:`experiments.tokenize.collect_metrics` and score with
:mod:`experiments.tokenize.bakeoff_analysis`.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

from experiments.tokenize.bakeoff_tokenizers import arm_by_name

# Proxy model shape held fixed across every arm and compute point (the bake-off measures the
# tokenizer, not the model). Only vocab_size varies, and it follows the arm inside the launcher.
PROXY_SHAPE = {
    "SCALE_HIDDEN_DIM": "1024",
    "SCALE_NUM_LAYERS": "16",
    "SCALE_NUM_EXPERTS": "32",
    "SCALE_TOP_K": "4",
    "SCALE_EXPERT_AXIS": "4",
    "SCALE_GPU_REPLICAS": "1",
    "SCALE_BATCH": "128",
    "SCALE_SEQ_LEN": "1024",
    "SCALE_STEPS_PER_EVAL": "500",
    "SCALE_TRACKER": "json_logger",
}

# The isoFLOP ladder: training steps per compute point. Same model + batch, so steps scale the
# token budget (and thus training FLOPs) linearly. >= 3 points lets bakeoff_analysis fit BPB(C).
DEFAULT_STEP_POINTS = (1500, 3500, 8000)

DEFAULT_ARMS = ("marin-128k", "gpt-oss-200k", "qwen3-152k", "gpt-neox-50k")


@dataclass(frozen=True)
class LadderCell:
    """One (arm, compute-point) run and the iris command that launches it."""

    arm: str
    steps: int
    run_id: str
    command: list[str]


def build_cell(arm_name: str, steps: int, cluster: str) -> LadderCell:
    arm = arm_by_name(arm_name)  # validates the arm exists and is registered
    run_id = f"bakeoff-{arm.name}-s{steps}"
    env = {**PROXY_SHAPE, "BAKEOFF_ARM": arm.name, "SCALE_STEPS": str(steps), "RUN_ID": run_id}
    cmd = [
        "iris",
        f"--cluster={cluster}",
        "job",
        "run",
        "--cpu",
        "2",
        "--memory",
        "3GB",
        "--extra",
        "cpu",
        "--job-name",
        f"grug-bakeoff-{arm.name}-s{steps}",
    ]
    for key, value in env.items():
        cmd += ["-e", key, value]
    cmd += ["--", "python", "-m", "experiments.grug.moe.launch_tokenizer_bakeoff"]
    return LadderCell(arm=arm.name, steps=steps, run_id=run_id, command=cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated arm names")
    ap.add_argument("--steps", default=",".join(map(str, DEFAULT_STEP_POINTS)), help="comma-separated step points")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--run", action="store_true", help="submit the jobs (default: print the plan only)")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    step_points = [int(s) for s in args.steps.split(",")]
    cells = [build_cell(arm, steps, args.cluster) for arm in arms for steps in step_points]

    print(f"ladder: {len(arms)} arms x {len(step_points)} points = {len(cells)} runs on {args.cluster}")
    for cell in cells:
        print(f"\n# {cell.run_id}")
        print(" ".join(cell.command))
        if args.run:
            subprocess.run(cell.command, check=True)

    if not args.run:
        print(f"\n(plan only — rerun with --run to submit all {len(cells)} jobs)")


if __name__ == "__main__":
    main()
