# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 MoE experiment: swap AdamH matrices to MuonH and test a 2x batch axis."""

from __future__ import annotations

import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import SEQ_LEN, build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, run_grug_moe_trial
from experiments.grug.moe.optimizer import build_grug_moe_muonh_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

WANDB_PROJECT = "marin_moe"
WANDB_GROUP = "moe-muonh-batch-g1"
WANDB_TAGS = ["moe", "muonh", "gate1"]
TPU_TYPE = "v5p-8"
RUN_PREFIX = "moe-muonh"
TARGET_STEPS = 2**14
COEFFICIENT_TYPE = "quintic"

GATE_1_SCALES: tuple[tuple[str, float, int], ...] = (
    ("d512", 2.19e17, 512),
    ("d768", 1.70e18, 768),
)
MUONH_BATCH_MULTIPLIERS: tuple[int, ...] = (1, 2)


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _batch_variant_name(batch_multiplier: int) -> str:
    return "basebatch" if batch_multiplier == 1 else f"batch{batch_multiplier}x"


def _build_step(scale_name: str, *, budget: float, hidden_dim: int, batch_multiplier: int) -> ExecutorStep:
    model_cfg, baseline_optimizer_cfg, baseline_batch_size, baseline_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=TARGET_STEPS,
    )
    optimizer_cfg = build_grug_moe_muonh_config(
        baseline_optimizer_cfg,
        coefficient_type=COEFFICIENT_TYPE,
    )
    batch_size = baseline_batch_size * batch_multiplier
    baseline_tokens = baseline_steps * baseline_batch_size * SEQ_LEN
    num_steps = max(1, round(baseline_tokens / (batch_size * SEQ_LEN)))
    batch_variant = _batch_variant_name(batch_multiplier)
    run_id = _resolve_run_id(f"{RUN_PREFIX}-{batch_variant}-{scale_name}")
    return ExecutorStep(
        name=f"grug/{RUN_PREFIX}/{batch_variant}/{scale_name}",
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
                tags=[*WANDB_TAGS, batch_variant],
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
            _build_step(scale_name, budget=budget, hidden_dim=hidden_dim, batch_multiplier=batch_multiplier)
            for batch_multiplier in MUONH_BATCH_MULTIPLIERS
            for scale_name, budget, hidden_dim in GATE_1_SCALES
        ],
        description="Gate-1 Grug MoE sweep with MuonH, preserving AdamH LR schedules and testing a 2x batch axis.",
    )
