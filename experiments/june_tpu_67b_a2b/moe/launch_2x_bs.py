# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launcher helpers for batch-size-ramp resumes.

Provides a parallel ``run_grug_moe_trial_2x_bs`` entry point to ``launch.py``'s
``run_grug_moe_trial`` for the special case of "double the batch size and
resume training from a specific source checkpoint step":

- ``initialize_from_path`` semantics: the source checkpoint is loaded only
  on the first launch. Subsequent iris restarts (preemption, crash) auto-
  resume from this run's own output checkpoints. ``launch.py``'s
  ``load_checkpoint_path`` forces every restart to re-load the same source
  checkpoint, which is the wrong behaviour for a long-running resume.

- A piecewise LR schedule (``GrugMoeMuonHResumeConfig``) that pins the
  post-resume LR to the source run's current LR at the resume step, ramps
  linearly to the BS-doubled peak over a configurable number of steps,
  then decays linearly to the floor (= ``min_lr_ratio * peak``, which is
  ``sqrt(2) * source_floor`` because the whole MuonH schedule scales with
  ``sqrt(B*S)``). This avoids the LR discontinuity that would result from
  naively swapping in the fresh heuristic schedule for the new BS.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
import optax
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig
from levanter.schedule import ScheduleStep
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import temporary_checkpoint_base_path

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


@OptimizerConfig.register_subclass("grug_moe_muonh_resume")
@dataclass(frozen=True)
class GrugMoeMuonHResumeConfig(GrugMoeMuonHConfig):
    """MuonH config with a piecewise LR schedule for a BS-ramp resume.

    Schedule shape (with ``L = learning_rate`` as the BS-doubled reference
    peak — never actually reached, since the resume joins the schedule
    partway into its decay):

      - ``[0, resume_step)``: constant at ``lr_at_resume`` (scaled).
        Never executed during training; present for safety.
      - ``[resume_step, ramp_end_step)``: linear ramp from
        ``lr_at_resume`` up to ``lr_at_ramp_end``. The ramp_end value is
        where the doubled-BS schedule would put the LR at this point in
        training (e.g. ``L * (1 - decay_fraction * (1 - min_lr_ratio))``
        for a 5%-into-decay resume).
      - ``[ramp_end_step, end_step)``: linear decay from ``lr_at_ramp_end``
        down to ``L * min_lr_ratio``.
      - ``[end_step, +inf)``: constant at ``L * min_lr_ratio``.

    All three transition values (``lr_at_resume``, ``lr_at_ramp_end``,
    ``learning_rate * min_lr_ratio``) are scaled by the same
    ``(override_lr / learning_rate)`` ratio for the adam group so muonh
    and adam follow the same piecewise shape at their respective scales.
    """

    resume_step: int = 0
    ramp_end_step: int = 0
    end_step: int = 0
    lr_at_resume: float = 0.0
    """LR at the resume step, in the muonh_lr scale. Scaled by
    ``override_lr / learning_rate`` for the adam group."""
    lr_at_ramp_end: float = 0.0
    """LR at the end of the ramp, in the muonh_lr scale. Typically less
    than ``learning_rate`` when resuming partway into the decay phase --
    e.g. for a 5%-into-decay resume with min_lr_ratio=0.05, this is
    ``learning_rate * 0.9616``. Scaled by ``override_lr / learning_rate``
    for the adam group."""

    def lr_scheduler(self, num_train_steps, override_lr=None):
        peak = self.learning_rate if override_lr is None else override_lr
        # Keep muonh_lr and adam_lr aligned: both follow the same piecewise
        # shape with values differing by the (override_lr / muonh_lr) ratio.
        ratio = peak / self.learning_rate
        start = self.lr_at_resume * ratio
        ramp_end_val = self.lr_at_ramp_end * ratio
        end = peak * self.min_lr_ratio

        ramp_len = self.ramp_end_step - self.resume_step
        decay_len = self.end_step - self.ramp_end_step

        const_pre = optax.constant_schedule(start)
        ramp = optax.linear_schedule(start, ramp_end_val, ramp_len)
        decay = optax.linear_schedule(ramp_end_val, end, decay_len)
        const_post = optax.constant_schedule(end)

        return optax.join_schedules(
            [const_pre, ramp, decay, const_post],
            boundaries=[self.resume_step, self.ramp_end_step, self.end_step],
        )


@dataclass(frozen=True)
class GrugMoeLaunchConfig2xBS:
    """Launch config for a BS-ramp resume run.

    Mirrors ``GrugMoeLaunchConfig`` from ``launch.py``, but exposes
    ``initialize_from_path`` (used as the first-time init checkpoint with
    auto-resume from this run's own output thereafter) instead of
    ``load_checkpoint_path`` (which would force every restart to reload the
    source checkpoint at step 15k, losing all post-resume progress).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    expert_parallel: int = 1
    checkpointer: CheckpointerConfig | None = None
    checkpoint_keep: list[dict] | None = None
    save_interval_minutes: int = 10
    initialize_from_path: str | None = None
    """Source checkpoint path. Loaded only when ``output_path`` has no
    checkpoint of its own (first launch). On subsequent restarts the trainer
    auto-resumes from the latest temp/permanent checkpoint under
    ``output_path``, so iris preemption survives without re-loading the
    source step every time."""
    source_batch_size: int | None = None
    """Batch size the source run used (typically half of ``batch_size``).
    When set, ``run_grug_moe_trial_2x_bs`` builds a piecewise BS schedule
    -- ``source_batch_size`` for steps ``[0, resume_step)``, then
    ``batch_size`` for ``[resume_step, end_step)`` -- and feeds it to
    ``TrainerConfig.train_batch_size``. This makes the data loader's
    cumulative offset at ``resume_step`` exactly match where the source
    left off, instead of jumping to ``resume_step * batch_size`` samples
    (which would skip half the dataset when batch_size doubled). Pass
    ``None`` when this run was trained at a constant BS throughout (i.e.
    no source-BS handoff to reconcile)."""
    resume_step: int = 0
    """Step number of the source checkpoint being resumed. Used only when
    ``source_batch_size`` is set (defines the BS-schedule boundary)."""
    per_device_parallelism: int = -1
    """Per-chip micro-batch size. Required when ``source_batch_size`` is
    set, because ``TrainerConfig`` cannot auto-infer it from a non-int
    ``train_batch_size``. For mesh (replica, data, expert, model) with
    batch_shards = replica * data * expert, this should be
    ``batch_size // batch_shards``."""


def run_grug_moe_trial_2x_bs(config: GrugMoeLaunchConfig2xBS) -> None:
    # Build a piecewise BS schedule when resuming from a source run that
    # used a different batch size. This makes the data loader's cumulative
    # offset at config.resume_step exactly match where the source left off
    # (= resume_step * source_batch_size samples), instead of jumping ahead
    # to resume_step * batch_size and skipping the data the source already
    # trained on. iter_from_step uses the BatchSchedule to compute offsets,
    # so the schedule has to faithfully describe what the source consumed.
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
        per_device_parallelism=config.per_device_parallelism,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=config.save_interval_minutes),
            keep=config.checkpoint_keep,
        ),
        # load_checkpoint=None + no explicit load_checkpoint_path makes the
        # trainer search only ``output_path``'s checkpoint dir. When that
        # dir is empty (first launch) and ``initialize_from`` is set,
        # levanter falls back to initialize_from. Once this run starts
        # saving its own checkpoints, every subsequent restart auto-resumes
        # from those instead of from initialize_from.
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=config.initialize_from_path,
    )

    grug_trainer = dataclasses.replace(
        config.grug_trainer,
        trainer=trainer,
        expert_axis_size=config.expert_parallel,
    )

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)
