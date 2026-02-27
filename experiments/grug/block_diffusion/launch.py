# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug block diffusion trial run (learning experiment).

This follows the post-#3054 grug template pattern (see `experiments/grug/base`).
The *objective* is intentionally unfinished; fill in TODOs in:
- `experiments/grug/block_diffusion/objective.py`

Run (example):
  uv run python -m experiments.grug.block_diffusion.launch
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import GrugEvalConfig, GrugTrainerConfig
from experiments.grug.block_diffusion.objective import BlockDiffusionObjectiveConfig
from experiments.grug.block_diffusion.train import GrugBlockDiffusionRunConfig, run_grug_block_diffusion
from experiments.tootsie.exp1295_32b import nemotron_mix


@dataclass(frozen=True)
class GrugBlockDiffusionLaunchConfig:
    model: GrugModelConfig
    objective: BlockDiffusionObjectiveConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    steps: int
    batch_size: int
    seed: int
    mp: str
    wandb_project: str
    wandb_tags: tuple[str, ...]
    wandb_group: str | None
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


GRUG_130M_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=2048,
    head_dim=None,
)

BLOCK_DIFFUSION_OBJECTIVE = BlockDiffusionObjectiveConfig(
    block_size=128,
    num_denoise_steps=8,
    mask_token_id=0,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def run_grug_block_diffusion_trial(config: GrugBlockDiffusionLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(
            project=config.wandb_project,
            name=config.run_id,
            tags=list(config.wandb_tags),
            group=config.wandb_group,
        ),
        use_explicit_mesh_axes=True,
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

    run_config = GrugBlockDiffusionRunConfig(
        model=config.model,
        objective=config.objective,
        data=config.data,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug_block_diffusion(run_config)


RESOLVED_RUN_ID = _resolve_run_id("grug-block-diffusion-trial")


grug_block_diffusion_trial = ExecutorStep(
    name="grug/block-diffusion-trial",
    fn=run_grug_block_diffusion_trial,
    config=GrugBlockDiffusionLaunchConfig(
        model=versioned(GRUG_130M_MODEL),
        objective=versioned(BLOCK_DIFFUSION_OBJECTIVE),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=versioned(RESOLVED_RUN_ID),
        steps=versioned(2000),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        wandb_project=versioned("marin"),
        wandb_tags=versioned(("grug", "block_diffusion", "learning")),
        wandb_group=versioned("grug-block-diffusion-trial"),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
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
                steps_per_eval=200,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
    resources=ResourceConfig.with_tpu("v5p-8"),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_block_diffusion_trial],
        description="Template grug block diffusion trial run (learning experiment).",
    )
