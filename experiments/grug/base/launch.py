# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-base trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/base` so
variants can be copied and modified in-place (for example MoE forks).
"""

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
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import resolve_local_placeholders, this_output_path, versioned
from marin.execution.executor import compute_output_path
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import TrainingPlan, default_validation_sets, run_train
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _run_grug_local,
)
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@dataclass(frozen=True)
class GrugBaseLaunchConfig:
    """Last-mile run config for the base grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
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
    max_seq_len=4096,
    head_dim=None,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def _build_grug_run_config(
    launch: GrugBaseLaunchConfig,
    *,
    output_path: str,
) -> GrugRunConfig:
    """Map launch-knobs into the trainer's full ``GrugRunConfig``."""
    trainer = TrainerConfig(
        id=launch.run_id,
        seed=launch.seed,
        train_batch_size=launch.batch_size,
        num_train_steps=launch.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(launch.mp),
        tracker=_resolve_tracker(launch.tracker, launch.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(launch.grug_trainer, trainer=trainer)

    return GrugRunConfig(
        model=launch.model,
        data=launch.data,
        resources=launch.resources,
        optimizer=launch.optimizer,
        trainer=grug_trainer,
        eval=launch.eval,
    )


def prepare_grug_trial(
    name: str,
    launch: GrugBaseLaunchConfig,
    *,
    override_output_path: str | None = None,
    env_vars: dict[str, str] | None = None,
) -> TrainingPlan[GrugRunConfig]:
    """Build a ``TrainingPlan`` for the grug-base template.

    Resolves ``output_path`` via the executor's version-hashing pass so
    re-submissions of the same plan resume from the same checkpoint
    directory, then builds the full ``GrugRunConfig`` with that path baked in.
    """
    output_path = compute_output_path(name, launch, override_output_path=override_output_path)

    # Substitute OutputName placeholders in the launch config and unwrap
    # VersionedValue wrappers. Upstream InputName / ExecutorStep references
    # (data placeholders) are preserved for the worker's `materialize` call.
    launch = resolve_local_placeholders(launch, output_path)

    run_config = _build_grug_run_config(launch, output_path=output_path)

    return TrainingPlan(
        name=name,
        output_path=output_path,
        train_config=run_config,
        worker_fn=_run_grug_local,
        resources=launch.resources,
        env_vars=dict(env_vars or {}),
    )


RESOLVED_RUN_ID = _resolve_run_id("grug-base-trial")

# Shared between the launch config (where the trainer reads it to build the
# Levanter mesh) and the TrainingPlan dispatch field (where the worker is
# scheduled). Same object on both sides avoids drift.
_GRUG_BASE_RESOURCES = ResourceConfig.with_tpu("v5p-8")


grug_base_launch = GrugBaseLaunchConfig(
    model=versioned(GRUG_130M_MODEL),
    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    output_path=this_output_path(),
    # Keep run id out of versioning so changing job metadata doesn't create a new output path.
    run_id=RESOLVED_RUN_ID,
    resources=versioned(_GRUG_BASE_RESOURCES),
    steps=versioned(2_000),
    batch_size=versioned(512),
    seed=versioned(0),
    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
    tracker=WandbConfig(
        project="marin",
        tags=["grug", "template"],
        group="grug-base-trial",
        name=None,  # filled from run_id in _resolve_tracker
        replicate_path=this_output_path(),
    ),
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
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        )
    ),
)


grug_base_plan = prepare_grug_trial(name="grug/base-trial", launch=grug_base_launch)


if __name__ == "__main__":
    run_train(grug_base_plan)
