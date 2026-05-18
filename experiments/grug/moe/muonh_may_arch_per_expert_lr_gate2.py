# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-2 launcher for passing per-expert MuonH LR candidates.

This module is intentionally candidate-selective: pass only the gate-1
candidates that show effective speedup at both d512 and d768.

Submit one or more passing candidates:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_per_expert_lr_gate2 \\
        --candidate mid-ratio
"""

import argparse
import sys

from marin.execution.executor import ExecutorStep, executor_main

from experiments.grug.moe.muonh_may_arch_per_expert_lr_gate1 import (
    _CANDIDATES,
    _RUN_SUFFIX,
    _build_step,
)

_GATE2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_GROUP_NAME: str = "muonh-may-arch-per-expert-lr-gate2"
_GATE2_TPU_RAM: str = "256g"
_GATE2_TPU_REGIONS: tuple[str, ...] = ("us-east5",)


def _build_steps_for_candidates(candidates: tuple[str, ...], run_suffix: str = _RUN_SUFFIX) -> list[ExecutorStep]:
    unknown_candidates = sorted(set(candidates) - set(_CANDIDATES))
    if unknown_candidates:
        raise ValueError(f"unknown gate-2 candidates: {unknown_candidates}")
    if not candidates:
        raise ValueError("at least one gate-2 candidate is required")

    return [
        _build_step(
            candidate,
            hidden_dim=d,
            budget=c,
            run_suffix=run_suffix,
            stage="gate2",
            group_name=_GROUP_NAME,
            resource_ram=_GATE2_TPU_RAM,
            resource_regions=_GATE2_TPU_REGIONS,
        )
        for candidate in candidates
        for d, c in _GATE2_POINTS
    ]


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=_CANDIDATES,
        required=True,
        help="Gate-1 candidate to advance to gate 2. Pass multiple times for multiple candidates.",
    )
    parser.add_argument("--run-suffix", default=_RUN_SUFFIX)
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, executor_args = _parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    candidates = tuple(args.candidate)
    steps = _build_steps_for_candidates(candidates, run_suffix=args.run_suffix)
    executor_main(
        steps=steps,
        description=(
            "Gate-2 MoE may_arch per-expert MuonH LR scaling "
            f"(candidates={candidates!r}, run_suffix={args.run_suffix!r})."
        ),
    )
