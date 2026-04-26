# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch MoE AdamH global gradient-normalization ablations."""

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeAdamHGlobalGradientNormConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_TARGET_STEPS: int = 2**14
_GATE_SPECS: dict[str, tuple[tuple[str, float, int], ...]] = {
    "gate1": (
        ("d512-2p19e17", 2.19e17, 512),
        ("d768-1p70e18", 1.70e18, 768),
    ),
    "gate2": (
        ("d1024-9p00e18", 9.00e18, 1024),
        ("d1280-2p83e19", 2.83e19, 1280),
    ),
}


def _global_gradient_norm_optimizer(optimizer: GrugMoeAdamHConfig) -> GrugMoeAdamHGlobalGradientNormConfig:
    return GrugMoeAdamHGlobalGradientNormConfig(**dataclasses.asdict(optimizer))


def _resolve_run_id(label: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", f"moe-adamh-global-grad-norm-{label}")
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _make_step(label: str, budget: float, hidden_dim: int) -> ExecutorStep:
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_TARGET_STEPS,
    )
    return ExecutorStep(
        name=f"grug/moe-adamh-global-grad-norm/{label}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_resolve_run_id(label),
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "adamh-global-grad-norm"],
                group="moe-adamh-global-grad-norm",
                name=None,
            ),
            optimizer=versioned(_global_gradient_norm_optimizer(optimizer)),
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


def _selected_specs() -> tuple[tuple[str, float, int], ...]:
    gate = os.environ.get("GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE", "gate1")
    if gate == "all":
        return _GATE_SPECS["gate1"] + _GATE_SPECS["gate2"]
    if gate not in _GATE_SPECS:
        raise ValueError(f"Unknown GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE={gate!r}. Expected gate1, gate2, or all.")
    return _GATE_SPECS[gate]


if __name__ == "__main__":
    executor_main(
        steps=[_make_step(label, budget, hidden_dim) for label, budget, hidden_dim in _selected_specs()],
        description="Grug MoE AdamH with global gradient RMS normalization.",
    )
