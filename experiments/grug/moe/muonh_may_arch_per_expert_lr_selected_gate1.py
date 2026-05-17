# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Selective gate-1 launcher for recovering one per-expert LR run.

This is for babysitter recovery only. The primary gate-1 launchers submit the
planned matrix; this module resubmits exactly one candidate/size if a child job
fails independently.

Submit one replacement run:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_per_expert_lr_selected_gate1 \\
        --candidate shrink-expert \\
        --hidden-dim 512 \\
        --run-suffix retry1
"""

import argparse
import sys

from marin.execution.executor import ExecutorStep, executor_main

from experiments.grug.moe.muonh_may_arch_per_expert_lr_gate1 import (
    _CANDIDATES,
    _GATE1_POINTS,
    _RUN_SUFFIX,
    _build_step,
)

_GATE1_BUDGET_BY_DIM: dict[int, float] = {hidden_dim: budget for hidden_dim, budget in _GATE1_POINTS}


def _build_selected_step(candidate: str, *, hidden_dim: int, run_suffix: str = _RUN_SUFFIX) -> ExecutorStep:
    if candidate not in _CANDIDATES:
        raise ValueError(f"unknown gate-1 candidate: {candidate}")
    if hidden_dim not in _GATE1_BUDGET_BY_DIM:
        raise ValueError(f"unknown gate-1 hidden_dim: {hidden_dim}")
    return _build_step(candidate, hidden_dim=hidden_dim, budget=_GATE1_BUDGET_BY_DIM[hidden_dim], run_suffix=run_suffix)


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=_CANDIDATES, required=True)
    parser.add_argument("--hidden-dim", type=int, choices=tuple(_GATE1_BUDGET_BY_DIM), required=True)
    parser.add_argument("--run-suffix", default=_RUN_SUFFIX)
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, executor_args = _parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    step = _build_selected_step(args.candidate, hidden_dim=args.hidden_dim, run_suffix=args.run_suffix)
    executor_main(
        steps=[step],
        description=(
            "Selective gate-1 MoE may_arch per-expert MuonH LR recovery "
            f"(candidate={args.candidate!r}, hidden_dim={args.hidden_dim}, run_suffix={args.run_suffix!r})."
        ),
    )
