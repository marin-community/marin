# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 MoE experiment: Muon with AOL Newton-Schulz coefficients."""

from __future__ import annotations

import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, run_grug_moe_trial
from experiments.grug.moe.optimizer import build_grug_moe_muon_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

WANDB_PROJECT = "marin_moe"
WANDB_GROUP = "moe-muon-aol-g1"
WANDB_TAGS = ["moe", "muon", "aol", "gate1"]
TPU_TYPE = "v5p-8"
RUN_PREFIX = "moe-muon-aol"
TARGET_STEPS = 2**14
COEFFICIENT_TYPE = "aol"

GATE_1_SCALES: tuple[tuple[str, float, int], ...] = (
    ("d512", 2.19e17, 512),
    ("d768", 1.70e18, 768),
)


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _build_step(scale_name: str, *, budget: float, hidden_dim: int) -> ExecutorStep:
    model_cfg, _baseline_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=TARGET_STEPS,
    )
    optimizer_cfg = build_grug_moe_muon_config(hidden_dim=hidden_dim, coefficient_type=COEFFICIENT_TYPE)
    run_id = _resolve_run_id(f"{RUN_PREFIX}-{scale_name}")
    return ExecutorStep(
        name=f"grug/{RUN_PREFIX}/{scale_name}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model_cfg),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(TPU_TYPE)),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=WANDB_TAGS,
                group=WANDB_GROUP,
                name=None,
            ),
            optimizer=versioned(optimizer_cfg),
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
    executor_main(
        steps=[
            _build_step(scale_name, budget=budget, hidden_dim=hidden_dim)
            for scale_name, budget, hidden_dim in GATE_1_SCALES
        ],
        description="Gate-1 Grug MoE sweep with Muon AOL coefficients against the AdamH baseline.",
    )
