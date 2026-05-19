# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 sweep for context-norm attention variant.

Variant change vs. baseline (XSA + head-wise gate):
- After XSA, before the head-wise sigmoid gate, apply a per-head RMSNorm with
  learnable scale of shape `[num_heads, head_dim]` over the head_dim axis. The
  rest of the recipe (QB routing, GatedNorm, XSA, sigmoid combine) is unchanged.

Gate-1 points (per `experiments/grug/moe/agent.md`):
- d512 / 2.19e17 FLOPs
- d768 / 1.70e18 FLOPs

Submit (preemptible, per project convention):

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --preemptible \\
      --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.context_norm_gate1
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

_GATE1_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "context-norm-gate1"
_RUN_SUFFIX: str = "v1"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str, stage: str = "gate1") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"context-norm-{stage}-{suffix}d{hidden_dim}-{budget_label}"


def _build_step(
    hidden_dim: int,
    budget: float,
    run_suffix: str = _RUN_SUFFIX,
    stage: str = "gate1",
    group_name: str = _GROUP_NAME,
) -> ExecutorStep:
    model, optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(model, use_context_norm=True)

    run_id = _format_run_id(hidden_dim, budget, run_suffix=run_suffix, stage=stage)
    step_name = f"grug/context_norm_{stage}/{run_id}"

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
                tags=["moe", "context_norm", "gate1", f"d{hidden_dim}"],
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
    steps = [_build_step(hidden_dim=d, budget=c) for d, c in _GATE1_POINTS]
    executor_main(
        steps=steps,
        description=f"Gate-1 MoE context-norm (per-head RMSNorm post-XSA, pre-gate). run_suffix={_RUN_SUFFIX!r}.",
    )
