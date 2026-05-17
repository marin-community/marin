# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add-on gate-1 launcher for the per-expert LR mid-ratio candidate.

This submits only the geometric-middle candidate after the initial gate-1
launcher has already submitted ``shrink-expert`` and ``boost-nonexpert``.

Submit:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_per_expert_lr_mid_ratio_gate1
"""

from marin.execution.executor import executor_main

from experiments.grug.moe.muonh_may_arch_per_expert_lr_gate1 import (
    _GATE1_POINTS,
    _RUN_SUFFIX,
    _build_step,
)

_CANDIDATE: str = "mid-ratio"


if __name__ == "__main__":
    steps = [_build_step(_CANDIDATE, hidden_dim=d, budget=c, run_suffix=_RUN_SUFFIX) for d, c in _GATE1_POINTS]
    executor_main(
        steps=steps,
        description="Gate-1 MoE may_arch per-expert MuonH LR mid-ratio add-on.",
    )
