# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch-size-ramp resumes for the MI355X June trainer.

The AMD counterpart of ``experiments/june_tpu_67b_a2b/moe/launch_2x_bs.py``: resume from a source checkpoint with
a larger batch, a piecewise batch schedule and a piecewise learning rate (``GrugMoeMuonHResumeConfig``).

- ``initialize_from_path`` is loaded only when the run has no checkpoint of its own, so a restart after a crash
  resumes from this run's own checkpoints.
- With ``source_batch_size`` set, the batch schedule is ``source_batch_size`` before ``resume_step`` and
  ``batch_size`` from then on, so the data loader starts at the example the source run stopped at.
- ``per_device_parallelism`` stays unset: the AMD trainer splits a step only through
  ``GrugTrainerConfig.microbatches``, and no single per-device size fits both phases of the schedule.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.optim.config import OptimizerConfig
from levanter.schedule import ScheduleStep
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.amd.june_67b_a2b.train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig


@dataclass(frozen=True)
class AmdLaunchConfig2xBS:
    """A batch-size-ramp resume of the AMD June trainer.

    The fields match ``GrugMoeLaunchConfig2xBS`` except for the mesh and step options of the AMD trainer
    (``expert_parallel``, ``replica_parallel``, ``microbatches``, ``zero1``) and ``checkpoint_dir``, which places
    the checkpoints on a shared filesystem path instead of under a Marin output path.
    """

    model: GrugModelConfig
    data: LmDataConfig
    run_id: str
    checkpoint_dir: str
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig | tuple[TrackerConfig, ...]
    """A W&B tracker is renamed to ``run_id``."""
    optimizer: OptimizerConfig
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    expert_parallel: int = 1
    replica_parallel: int | None = None
    microbatches: int = 1
    zero1: bool = False
    diagnose_numerics: bool = False
    step_timeout: float | None = None
    checkpoint_keep: list[dict] | None = None
    save_interval: timedelta = timedelta(minutes=30)
    initialize_from_path: str | None = None
    """Source checkpoint, loaded only when ``checkpoint_dir`` has no checkpoint of this run."""
    source_batch_size: int | None = None
    """The source run's batch size. When it differs from ``batch_size``, steps before ``resume_step`` use it."""
    resume_step: int = 0
    log_dir: str = "logs"
    initialize_jax_distributed: bool = True
    """False when the caller has already started ``jax.distributed`` (a Slurm entry point)."""


def build_2x_bs_run_config(config: AmdLaunchConfig2xBS) -> GrugRunConfig:
    """The AMD trainer's run config for ``config``."""
    if config.source_batch_size is not None and config.source_batch_size != config.batch_size:
        train_batch_size: int | list[ScheduleStep] = [
            ScheduleStep(start=0, value=config.source_batch_size),
            ScheduleStep(start=config.resume_step, value=config.batch_size),
        ]
    else:
        train_batch_size = config.batch_size

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=train_batch_size,
        num_train_steps=config.steps,
        mp=jmp.get_policy(config.mp),
        tracker=(
            dataclasses.replace(config.tracker, name=config.run_id)
            if isinstance(config.tracker, WandbConfig)
            else config.tracker
        ),
        log_dir=config.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        distributed=DistributedConfig(initialize_jax_distributed=config.initialize_jax_distributed),
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.checkpoint_dir, "checkpoints"),
            temporary_base_path=os.path.join(config.checkpoint_dir, "checkpoints-temp"),
            append_run_id_to_base_path=False,
            save_interval=config.save_interval,
            keep=config.checkpoint_keep,
        ),
        # Search only this run's checkpoint directory; when it is empty, start from initialize_from.
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=config.initialize_from_path,
    )
    return GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=GrugTrainerConfig(
            trainer=trainer,
            expert_axis_size=config.expert_parallel,
            replica_axis_size=config.replica_parallel,
            microbatches=config.microbatches,
            zero1=config.zero1,
            diagnose_numerics=config.diagnose_numerics,
            step_timeout=config.step_timeout,
        ),
        eval=config.eval,
    )


def run_grug_moe_trial_2x_bs(config: AmdLaunchConfig2xBS) -> None:
    """Dispatch the resume through Fray, like ``run_grug_moe_trial_2x_bs`` for TPU."""
    run_grug(build_2x_bs_run_config(config))
