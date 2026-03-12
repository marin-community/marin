# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry (MoE): Grug MoE daily pretraining canary on TPU.

Trains ~1B tokens on v5p-8 using the Grug MoE trial model (512 hidden,
8 experts pick-2, 6 layers), scaled down from the ~4.2B-token trial
template to fit a <1hr daily window.

Usage:
    python -m experiments.ferries.canary_ferry_moe
"""

import dataclasses
import datetime
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import AdamConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

RUN_ID = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

# ~1B tokens: batch=512 * seq=4096 -> 476 steps
BATCH_SIZE = 512
TARGET_TOKENS = 1_000_000_000
NUM_STEPS = TARGET_TOKENS // (BATCH_SIZE * GRUG_MOE_TRIAL_MODEL.max_seq_len)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_canary_moe(config: GrugMoeLaunchConfig) -> None:
    """Copy of run_grug_moe_trial with profiling enabled."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=True),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


canary_moe_step = ExecutorStep(
    name=f"canary-ferry-moe-{RUN_ID}",
    fn=run_canary_moe,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_TRIAL_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NUM_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["canary", "ferry", "grug", "moe"],
            group="canary-ferry-moe",
            name=None,
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=48,
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
                eval_batch_size=BATCH_SIZE,
                steps_per_eval=240,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


def main():
    executor_main(steps=[canary_moe_step])


if __name__ == "__main__":
    main()
