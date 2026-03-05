# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Iris entrypoint for the parallel-attn-mlp grug variant.

This runs training in-process on the job's allocated TPU instead of launching a
nested ExecutorStep.
"""

from __future__ import annotations

import os

from fray.cluster import ResourceConfig
from iris.marin_fs import marin_prefix
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig

from experiments.grug.parallel_attn_mlp.launch import (
    GRUG_130M_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugBaseLaunchConfig,
    _resolve_run_id,
    run_grug_base_trial,
)
from experiments.grug.parallel_attn_mlp.train import GrugEvalConfig, GrugTrainerConfig


def _resolve_output_path(run_id: str) -> str:
    prefix = os.environ.get("GRUG_OUTPUT_PREFIX") or marin_prefix()
    return f"{prefix}/grug/parallel-attn-mlp-direct/{run_id}"


def main() -> None:
    run_id = _resolve_run_id("grug-parallel-attn-mlp-trial")
    config = GrugBaseLaunchConfig(
        model=GRUG_130M_MODEL,
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=_resolve_output_path(run_id),
        run_id=run_id,
        resources=ResourceConfig.with_tpu("v5p-8"),
        steps=2_000,
        batch_size=512,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "parallel-attn-mlp"],
            group="grug-parallel-attn-mlp-trial",
            name=run_id,
        ),
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=0.1,
            lr_schedule="cosine",
            decay=0.2,
            min_lr_ratio=0.1,
            warmup=1000,
        ),
        grug_trainer=GrugTrainerConfig(
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
        ),
        eval=GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=200,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        ),
    )
    run_grug_base_trial(config)


if __name__ == "__main__":
    main()
