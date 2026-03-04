# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small Grug-MoE smoke run with router aux losses enabled."""

from __future__ import annotations

import dataclasses
import os

from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig


def main() -> None:
    run_id = os.environ.get("GRUG_MOE_SMOKE_RUN_ID", "grug-moe-smoke-v6e8-20260303-134754")
    output_path = os.environ.get("GRUG_MOE_SMOKE_OUTPUT_PATH", "/tmp/grug-moe-smoke")

    cfg = GrugMoeLaunchConfig(
        model=dataclasses.replace(
            GRUG_MOE_TRIAL_MODEL,
            load_balancing_loss_coef=0.01,
            router_z_loss_coef=0.001,
        ),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=output_path,
        run_id=run_id,
        steps=20,
        batch_size=32,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin",
            mode="disabled",
            tags=["grug", "moe", "smoke"],
            group="grug-moe-smoke-v6e8",
            name=run_id,
        ),
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=0.1,
            lr_schedule="cosine",
            decay=0.2,
            min_lr_ratio=0.1,
            warmup=10,
        ),
        grug_trainer=GrugTrainerConfig(
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
        ),
        eval=GrugEvalConfig(
            eval_batch_size=64,
            steps_per_eval=None,
            max_eval_batches=1,
            eval_current=False,
            eval_ema=False,
        ),
    )
    run_grug_moe_trial(cfg)


if __name__ == "__main__":
    main()
