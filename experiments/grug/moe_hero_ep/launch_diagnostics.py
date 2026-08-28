# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded GB200 diagnostics for the production EP64 MoE hero recipe."""

import dataclasses
import os
from datetime import timedelta

import click
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointDebugConfig, CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    HARRIER_MIX_2026_08_18_TAG,
    harrier_mix_2026_08_18_data_config,
)
from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MASTER_PARAM_MODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_PROCESSES_PER_TASK,
    HERO_TENSORSTORE_CACHE_BYTES,
    HERO_WATCH_INTERVAL,
    HeroThroughputResult,
    hero_grug_trainer_config,
    hero_trainer_config,
    validation_datasets,
)
from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.train import (
    RAGGED_MOE_IMPLEMENTATION,
    GrugEvalConfig,
    GrugRunConfig,
    MasterParamMode,
    TrainingDataMode,
    WatchMode,
    _compute_flops,
    run_grug,
)

DEFAULT_HERO_STEPS = 25
HERO_CHECKPOINT_INTERVAL = timedelta(minutes=15)


def build_diagnostic_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None = None,
    seed: int = 0,
    batch_size: int = HERO_EP_BATCH_SIZE,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    latent_dim: int | None = None,
    moe_implementation: str | None = None,
    master_param_mode: MasterParamMode = HERO_MASTER_PARAM_MODE,
    processes_per_task: int = HERO_PROCESSES_PER_TASK,
    eval_every: int = 0,
    save_checkpoints: bool = False,
    checkpoint_interval: timedelta = HERO_CHECKPOINT_INTERVAL,
    checkpoint_path: str | None = None,
    checkpoint_debug: CheckpointDebugConfig | None = None,
    watch_interval: int = HERO_WATCH_INTERVAL,
    watch_mode: WatchMode = WatchMode.INLINE,
    profile_steps: int = 0,
    profile_start_step: int = 5,
    training_data_mode: TrainingDataMode = TrainingDataMode.MIXTURE,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build a bounded diagnostic run for the production EP64 hero recipe.

    The overrides sweep expert count, expert width, routed top-k, and routing capacity from the
    hero spec. They keep the hidden dimension, so the compute-scaled optimizer values stay
    comparable across a sweep. ``None`` keeps the hero value.

    ``batch_size`` is the global batch across all data-parallel racks. It does not scale with
    ``dp_racks``. ``batch_size`` and ``schedule_steps`` change the token budget for the heuristic.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if checkpoint_interval <= timedelta(0):
        raise ValueError(f"checkpoint_interval must be positive, got {checkpoint_interval}")
    if profile_steps < 0:
        raise ValueError(f"profile_steps must be non-negative, got {profile_steps}")
    if profile_start_step < 0:
        raise ValueError(f"profile_start_step must be non-negative, got {profile_start_step}")
    if profile_steps > 0 and profile_start_step >= num_steps:
        raise ValueError(f"profile_start_step must be less than num_steps={num_steps}, got {profile_start_step}")
    # `schedule_steps` sets the whole learning-rate schedule; `num_steps` is the absolute step the
    # run stops at (a restore resumes mid-schedule, so it must lie past the restored step).
    # Both matter, and they enter in different places. The optimizer heuristic scales learning rate,
    # adam_lr, and epsilon from a token budget (`num_train_steps * batch * seq`), which fixes the
    # peak. Warmup and decay are *fractions* of `TrainerConfig.num_train_steps`, so that field has to
    # carry the schedule length too -- passing `num_steps` there warms up in `0.01 * num_steps` and
    # decays to `min_lr_ratio` by the end of the short run, which is a whole miniature schedule
    # rather than the head of a long one. Default keeps the two equal, which is the previous behavior.
    if schedule_steps is not None and schedule_steps <= 0:
        raise ValueError(f"schedule_steps must be positive, got {schedule_steps}")
    if schedule_steps is not None and schedule_steps < num_steps:
        raise ValueError(f"schedule_steps={schedule_steps} must be at least num_steps={num_steps}")
    total_schedule_steps = schedule_steps if schedule_steps is not None else num_steps
    _, optimizer = build_hero_configs(
        num_train_steps=total_schedule_steps,
        batch_size=batch_size,
    )
    model = HERO_MODEL_CONFIG
    overrides = {
        name: value
        for name, value in (
            ("num_experts", num_experts),
            ("num_experts_per_token", num_experts_per_token),
            ("intermediate_dim", intermediate_dim),
            ("capacity_factor", capacity_factor),
            ("latent_dim", latent_dim),
            ("moe_implementation", moe_implementation),
        )
        if value is not None
    }
    if overrides:
        model = dataclasses.replace(model, **overrides)
    # A bank that is not divisible by the expert axis fails inside `moe_mlp`, which is after the rack
    # is already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % HERO_EP_EXPERT_AXIS_SIZE != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}")
    local_experts = model.num_experts // HERO_EP_EXPERT_AXIS_SIZE
    if local_experts % model.num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={model.num_expert_waves}"
        )
    # The ragged transport requires one GPU per process; fail fast if this is not satisfied.
    if model.moe_implementation == RAGGED_MOE_IMPLEMENTATION and processes_per_task != HERO_GPUS_PER_NODE:
        raise ValueError(
            f"{RAGGED_MOE_IMPLEMENTATION} needs one process per GPU: pass "
            f"processes_per_task={HERO_GPUS_PER_NODE}, got {processes_per_task}"
        )
    pooled = model.moe_implementation == "fixed_pooled_wave_all_to_all"
    if pooled and model.pooled_transport_capacity_factor is None:
        raise AssertionError("the pooled-wave hero requires a transport capacity factor")
    backend_tag = model.moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    # Only the pooled transport has a receiver capacity of its own to report.
    transport_capacity_tags = (f"transport-capacity-{model.pooled_transport_capacity_factor:g}",) if pooled else ()
    wave_tag = f"expert-waves-{model.num_expert_waves}"
    size_tag = f"e{model.num_experts}-i{model.intermediate_dim}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = hero_grug_trainer_config(
        replica_axis_size=dp_racks,
        training_data_mode=training_data_mode,
        watch_mode=watch_mode,
        save_checkpoints=save_checkpoints,
        master_param_mode=master_param_mode,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=HERO_NODE_CPU,
        ram=HERO_NODE_RAM,
        disk=HERO_NODE_DISK,
        replicas=HERO_EP_NODES * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = validation_datasets() if eval_every > 0 else []
    flops_per_example, _ = _compute_flops(model_config=model)
    experiment_flops = flops_per_example * batch_size * total_schedule_steps

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = hero_trainer_config(
            run_id=run_id,
            seed=seed,
            train_batch_size=batch_size,
            num_train_steps=total_schedule_steps,
            master_param_mode=master_param_mode,
            profiler=ProfilerConfig(
                enabled=profile_steps > 0,
                start_step=profile_start_step,
                num_steps=profile_steps,
                # One rank is enough for a step trace, and tracing all 64 multiplies the upload
                # without adding signal.
                process_index=0,
                # Host scopes and HLO metadata identify compiled model regions in XProf.
                profile_options=ProfileOptionsConfig(
                    host_tracer_level=1,
                    python_tracer_level=0,
                    enable_hlo_proto=True,
                ),
            ),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    backend_tag,
                    capacity_tag,
                    f"master-params-{master_param_mode.value.replace('_', '-')}",
                    *transport_capacity_tags,
                    wave_tag,
                    size_tag,
                    "gb200",
                    HARRIER_MIX_2026_08_18_TAG,
                    "MHEP",
                ],
                group="moe-hero-ep",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=watch_interval),
            # Levanter's default base path is pod-local, so a preempted run would have nothing to
            # resume from. `checkpoint_path` overrides this for runs targeting disposable storage.
            checkpointer=CheckpointerConfig(
                base_path=checkpoint_path or prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=checkpoint_interval,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
                debug=checkpoint_debug or CheckpointDebugConfig(),
            ),
        )
        data = harrier_mix_2026_08_18_data_config(
            ctx=ctx,
            total_steps=total_schedule_steps,
            batch_size=batch_size,
            max_seq_len=model.max_seq_len,
            experiment_flops=experiment_flops,
            validation=validation,
        )
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            # Off by default so a throughput run stays a throughput run. Turn it on to make a
            # run scoreable: comparing configs needs held-out loss, not train loss.
            eval=(
                GrugEvalConfig(
                    steps_per_eval=eval_every,
                    eval_batch_size=HERO_EP_EXPERT_AXIS_SIZE * dp_racks,
                    eval_ema=False,
                    compute_bpb=True,
                    dropless_eval=True,
                )
                if eval_every > 0
                else None
            ),
            stop_after_steps=num_steps,
            processes_per_task=processes_per_task,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(HARRIER_MIX_2026_08_18_STORE, *validation),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--dp-racks",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Data-parallel NVL72 rack count. --batch-size stays global across all racks.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_HERO_STEPS,
    show_default=True,
    help=(
        "Absolute step the run stops at, not a count. A run that restores a checkpoint must set "
        "this past the restored step (e.g. 30200 for 200 steps from a step-30000 restore), or the "
        "trainer sees the run as already complete and exits successfully after zero steps."
    ),
)
@click.option(
    "--schedule-steps",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Build the learning-rate schedule for this many steps instead of --num-steps. The optimizer "
        "heuristic scales its rates from the implied token budget, so this trains the head of a long "
        "run's schedule. Defaults to --num-steps."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Trainer seed. Vary it across otherwise identical runs to measure run-to-run variance.",
)
@click.option(
    "--num-experts",
    type=click.IntRange(min=1),
    default=HERO_MODEL_CONFIG.num_experts,
    show_default=True,
    help=(
        f"Override the routed expert count. The count must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}, "
        f"and the local expert count must support {HERO_MODEL_CONFIG.num_expert_waves} waves."
    ),
)
@click.option(
    "--num-experts-per-token",
    type=click.IntRange(min=1),
    default=HERO_MODEL_CONFIG.num_experts_per_token,
    show_default=True,
    help="Override the routed top-k. Scales both active parameters and the EP dispatch buffers.",
)
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=HERO_MODEL_CONFIG.intermediate_dim,
    show_default=True,
    help="Override the routed expert width.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=HERO_EP_BATCH_SIZE,
    show_default=True,
    help="Global sequences per step. This value does not scale with --dp-racks.",
)
@click.option(
    "--moe-implementation",
    default=None,
    help="Override the MoE backend, e.g. ragged_all_to_all. Defaults to the hero spec.",
)
@click.option(
    "--master-params",
    type=click.Choice([mode.value for mode in MasterParamMode]),
    default=HERO_MASTER_PARAM_MODE.value,
    show_default=True,
    help=("Whether to keep fp32 weights in host pinned memory. Disabling the master keeps them on device."),
)
@click.option(
    "--processes-per-task",
    type=click.IntRange(min=1),
    default=HERO_PROCESSES_PER_TASK,
    show_default=True,
    help="JAX processes per node.",
)
@click.option(
    "--latent-dim",
    type=click.IntRange(min=1),
    default=None,
    help="LatentMoE: run routed experts at this width. Divides all-to-all traffic by hidden/latent.",
)
@click.option(
    "--save-checkpoints/--no-save-checkpoints",
    default=False,
    show_default=True,
    help="Write periodic and final checkpoints, and resume from the newest complete checkpoint.",
)
@click.option(
    "--checkpoint-minutes",
    type=click.FloatRange(min=0, min_open=True),
    default=HERO_CHECKPOINT_INTERVAL.total_seconds() / 60,
    show_default=True,
    help="Wall-clock minutes between checkpoint writes.",
)
@click.option(
    "--checkpoint-path",
    default=None,
    help="Checkpoint output path, e.g. a marin_temp_bucket() path. Defaults to the step output path.",
)
@click.option(
    "--checkpoint-debug/--no-checkpoint-debug",
    default=False,
    show_default=True,
    help="Publish checkpoint phase and memory telemetry. Use with --save-checkpoints.",
)
@click.option(
    "--eval-every",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Run the paloma suite every N steps. 0 disables eval (throughput-only run).",
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=HERO_WATCH_INTERVAL,
    show_default=True,
    help="Steps between gradient and parameter norm logs. 0 disables norm logging.",
)
@click.option(
    "--watch-mode",
    type=click.Choice([mode.value for mode in WatchMode]),
    default=WatchMode.INLINE.value,
    show_default=True,
    help="Compute norms in the training step or in a separate forward and backward diagnostic step.",
)
@click.option(
    "--profile-steps",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Steps to trace with XProf on rank 0. 0 disables the profiler.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="First traced step. Keep it past compile and warmup.",
)
@click.option(
    "--training-data",
    type=click.Choice([mode.value for mode in TrainingDataMode]),
    default=TrainingDataMode.MIXTURE.value,
    show_default=True,
    help="Use the configured mixture or reuse a deterministic synthetic batch without opening TensorStore.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=HERO_MODEL_CONFIG.capacity_factor,
    show_default=True,
    help="Override the pooled receiver capacity factor.",
)
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None,
    seed: int,
    batch_size: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
    latent_dim: int | None,
    moe_implementation: str | None,
    master_params: str,
    processes_per_task: int,
    save_checkpoints: bool,
    checkpoint_minutes: float,
    checkpoint_path: str | None,
    checkpoint_debug: bool,
    eval_every: int,
    watch_interval: int,
    watch_mode: str,
    profile_steps: int,
    profile_start_step: int,
    training_data: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_diagnostic_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        schedule_steps=schedule_steps,
        seed=seed,
        batch_size=batch_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        latent_dim=latent_dim,
        moe_implementation=moe_implementation,
        master_param_mode=MasterParamMode(master_params),
        processes_per_task=processes_per_task,
        save_checkpoints=save_checkpoints,
        checkpoint_interval=timedelta(minutes=checkpoint_minutes),
        checkpoint_path=checkpoint_path,
        checkpoint_debug=(
            CheckpointDebugConfig(
                enabled=True,
                tracemalloc_frames=None,
                top_allocations=0,
                force_gc_before_serialize=False,
                flush_logs=False,
            )
            if checkpoint_debug
            else None
        ),
        eval_every=eval_every,
        watch_interval=watch_interval,
        watch_mode=WatchMode(watch_mode),
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
        training_data_mode=TrainingDataMode(training_data),
    )


if __name__ == "__main__":
    main()
