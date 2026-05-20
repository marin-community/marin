# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-2 sweep for context-norm-replaces-XSA attention variant.

Builds on the gate-1 PASS at d512 and d768 (issue #5854).

Variant change vs. baseline:
- Disable XSA (no projection-onto-v subtraction after attention).
- Per-head RMSNorm with learnable scale `[num_heads, head_dim]` on the
  attention context vector immediately after attention.
- Existing head-wise sigmoid gate after the norm.

Gate-2 points (per `experiments/grug/moe/agent.md`):
- d1024 / 9.00e18 FLOPs
- d1280 / 2.83e19 FLOPs

Submit (preemptible, per project convention):

    .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait \\
      --preemptible \\
      --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.context_norm_no_xsa_gate2
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_GATE2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "context-norm-no-xsa-gate2"
_RUN_SUFFIX: str = "v1"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str, stage: str = "gate2") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"context-norm-no-xsa-{stage}-{suffix}d{hidden_dim}-{budget_label}"


def _build_step(
    hidden_dim: int,
    budget: float,
    run_suffix: str = _RUN_SUFFIX,
    stage: str = "gate2",
    group_name: str = _GROUP_NAME,
) -> ExecutorStep:
    model, optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(model, use_context_norm=True, use_xsa=False)

    run_id = _format_run_id(hidden_dim, budget, run_suffix=run_suffix, stage=stage)
    step_name = f"grug/context_norm_no_xsa_{stage}/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["moe", "context_norm", "no_xsa", "gate2", f"d{hidden_dim}"],
                group=group_name,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(hidden_dim=d, budget=c) for d, c in _GATE2_POINTS]
    executor_main(
        steps=steps,
        description=(
            "Gate-2 MoE context-norm replaces XSA (no XSA + per-head RMSNorm + head-wise gate). "
            f"run_suffix={_RUN_SUFFIX!r}."
        ),
    )
