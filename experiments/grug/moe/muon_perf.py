# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon throughput measurement on grug MoE.

Runs the same model architecture as the Adam baseline (launch.py) but swaps the
optimizer to GrugMuonConfig so we can compare wall-clock step time and MFU
between Adam and Muon on MoE.  This is the "perf" half of the Muon-on-MoE
evaluation; the "loss" half lives in a sibling issue (#4034).

Tracking issue: https://github.com/marin-community/marin/issues/4033
Parent sweep: https://github.com/marin-community/marin/issues/3469
"""

from levanter.optim.grugmuon import GrugMuonConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    GrugMoeLaunchConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    _resolve_run_id,
    run_grug_moe,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

from fray.cluster import ResourceConfig


MUON_MOE_RUN_ID = _resolve_run_id("grug-moe-muon-perf")

grug_moe_muon_perf = ExecutorStep(
    name="grug/moe-muon-perf",
    fn=run_grug_moe,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_TRIAL_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=MUON_MOE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(2_000),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "moe", "muon", "perf"],
            group="grug-moe-muon-perf",
            name=None,
        ),
        optimizer=versioned(
            GrugMuonConfig(
                lr=0.02,
                adam_lr=3e-3,
                momentum=0.95,
                nesterov=True,
                weight_decay=0.0,
                lr_schedule="cosine",
                min_lr_ratio=0.1,
                warmup=1000,
            )
        ),
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
        steps=[grug_moe_muon_perf],
        description="Muon throughput measurement on grug MoE (same arch as Adam baseline).",
    )
