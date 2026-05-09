# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NorMuonH matrix-swap sweep across the compute-optimal MoE gate points.

This mirrors ``muonh_matrix_sweep.py`` but replaces the Muon direction inside
the hyperball update with NorMuon's output-axis adaptive direction. In Grug,
matrices use ``(fan_in, fan_out)`` layout, so NorMuon's neuron statistic is
tracked on the final axis.

Set ``NORMUONH_MATRIX_GATE`` to control which scales run:

    NORMUONH_MATRIX_GATE=1     # default: d512, d768 (gate 1 only)
    NORMUONH_MATRIX_GATE=2     # d1024, d1280 (gate 2 only)
    NORMUONH_MATRIX_GATE=both  # all four scales
"""

import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeNorMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_GATE_1_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)
_GATE_2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "normuonh-matrix-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _normuonh_optimizer_from_baseline(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeNorMuonHConfig:
    return GrugMoeNorMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=base_optimizer.warmup,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )


def _build_step(hidden_dim: int, budget: float) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    optimizer = _normuonh_optimizer_from_baseline(base_optimizer)

    budget_label = _format_budget(budget)
    run_id = f"normuonh-matrix-d{hidden_dim}-{budget_label}"
    step_name = f"grug/normuonh_matrix_sweep/{run_id}"

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
                project="marin_moe",
                tags=["moe", "normuonh_matrix_sweep", f"d{hidden_dim}"],
                group=_GROUP,
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


def _build_steps(gate: str) -> list[ExecutorStep]:
    if gate == "1":
        points = _GATE_1_POINTS
    elif gate == "2":
        points = _GATE_2_POINTS
    elif gate == "both":
        points = _GATE_1_POINTS + _GATE_2_POINTS
    else:
        raise ValueError(f"unknown gate: {gate!r} (expected '1', '2', or 'both')")

    return [_build_step(hidden_dim=hidden_dim, budget=budget) for hidden_dim, budget in points]


if __name__ == "__main__":
    gate = os.environ.get("NORMUONH_MATRIX_GATE", "1")
    steps = _build_steps(gate)
    executor_main(
        steps=steps,
        description=f"MoE NorMuonH matrix swap sweep (gate={gate}).",
    )
