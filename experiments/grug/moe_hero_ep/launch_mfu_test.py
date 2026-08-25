# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GB200 launcher for the EP64 MoE hero configuration."""

import dataclasses
import os
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_17_1 import (
    HARRIER_MIX_2026_08_17_1_STORE,
    HARRIER_MIX_2026_08_17_1_TAG,
    harrier_mix_2026_08_17_1_data_config,
)
from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL, build_hero_configs
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    MasterParamMode,
    TrainingDataMode,
    WatchMode,
    run_grug,
)
from experiments.marin_tokenizer import marin_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
# The hero has never used tensor parallelism, and nothing here exposes a knob for it.
HERO_MODEL_AXIS_SIZE = 1
HERO_PROCESSES_PER_TASK = 1
HERO_MIXED_PRECISION = "params=bfloat16,compute=bfloat16,output=bfloat16"
# Weight storage that goes with each master-parameter mode. The pooled-wave hero needs the
# pinned-host master to fit at all. The ragged transport does fit with fp32 weights on device,
# and a paired hero measurement put the master 1.78 percent behind on that path -- it buys
# memory relief that transport is not short of, and charges a host round trip per step for it.
HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE = {
    MasterParamMode.FP32_PINNED_HOST: HERO_MIXED_PRECISION,
    MasterParamMode.DISABLED: "params=float32,compute=bfloat16,output=bfloat16",
}
# Keep MuonH state on pinned host memory to leave room for the pooled all-to-all buffers.
HERO_OFFLOAD_OPT_STATE = True
HERO_WATCH_INTERVAL = 0
HERO_CHECKPOINT_INTERVAL = timedelta(minutes=15)


# Held-out sets are added at weight 0 so they surface as tagged eval sets.
def _validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    return list(paloma_datasets(tokenizer=marin_tokenizer).values())


def _validate_mesh_axes(*, dp_racks: int, batch_size: int, context_axis_size: int, expert_axis_size: int) -> None:
    """Reject mesh shapes and batch sizes that would only fail once the rack is allocated.

    ``compact_grug_mesh`` gives ``data`` whatever the fixed axes leave free, so a bad
    ``--context-axis-size`` or ``--expert-axis-size`` surfaces as a mesh-construction failure on
    the allocated rack. Both checks run here instead: the axis sizes have to divide the rack's
    device count, and the global batch has to divide over the axes the data loader shards it on
    (``replica_dcn``, ``data``, ``expert``) or the loader refuses the batch after startup.
    """
    if context_axis_size <= 0:
        raise ValueError(f"context_axis_size must be positive, got {context_axis_size}")
    if expert_axis_size <= 0:
        raise ValueError(f"expert_axis_size must be positive, got {expert_axis_size}")
    devices = HERO_EP_NODES * HERO_GPUS_PER_NODE * dp_racks
    fixed = dp_racks * context_axis_size * expert_axis_size * HERO_MODEL_AXIS_SIZE
    if devices % fixed != 0:
        raise ValueError(
            f"device count ({devices}) must be divisible by replica ({dp_racks}) * "
            f"context ({context_axis_size}) * expert ({expert_axis_size}) * model "
            f"({HERO_MODEL_AXIS_SIZE})"
        )
    data_axis_size = devices // fixed
    batch_axes_product = dp_racks * data_axis_size * expert_axis_size
    if batch_size % batch_axes_product != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by the batch axes replica ({dp_racks}) * "
            f"data ({data_axis_size}) * expert ({expert_axis_size}) = {batch_axes_product}"
        )


class HeroThroughputResult(Artifact):
    """Result of the rack-scale throughput hero run.

    The run mirrors its tracker metrics to the output path. It writes no checkpoint by default.
    With checkpointing enabled, it resumes from the newest complete checkpoint.
    """


def build_hero_run(
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
    max_seq_len: int | None = None,
    qk_mult: float | None = None,
    latent_dim: int | None = None,
    context_axis_size: int = 1,
    expert_axis_size: int = HERO_EP_EXPERT_AXIS_SIZE,
    moe_implementation: str | None = None,
    master_param_mode: MasterParamMode = MasterParamMode.FP32_PINNED_HOST,
    restore_from: str | None = None,
    restore_master_param_mode: MasterParamMode | None = None,
    processes_per_task: int = HERO_PROCESSES_PER_TASK,
    eval_every: int = 0,
    save_checkpoints: bool = False,
    checkpoint_interval: timedelta = HERO_CHECKPOINT_INTERVAL,
    checkpoint_path: str | None = None,
    watch_interval: int = HERO_WATCH_INTERVAL,
    watch_mode: WatchMode = WatchMode.INLINE,
    profile_steps: int = 0,
    profile_start_step: int = 5,
    training_data_mode: TrainingDataMode = TrainingDataMode.MIXTURE,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the EP64 hero throughput run.

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
    # `schedule_steps` sets the whole learning-rate schedule; `num_steps` sets how far the run goes.
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
    model, optimizer = build_hero_configs(
        num_train_steps=total_schedule_steps,
        batch_size=batch_size,
    )
    overrides = {
        name: value
        for name, value in (
            ("num_experts", num_experts),
            ("num_experts_per_token", num_experts_per_token),
            ("intermediate_dim", intermediate_dim),
            ("capacity_factor", capacity_factor),
            ("max_seq_len", max_seq_len),
            ("qk_mult", qk_mult),
            ("latent_dim", latent_dim),
            ("moe_implementation", moe_implementation),
        )
        if value is not None
    }
    if overrides:
        model = dataclasses.replace(model, **overrides)
    _validate_mesh_axes(
        dp_racks=dp_racks,
        batch_size=batch_size,
        context_axis_size=context_axis_size,
        expert_axis_size=expert_axis_size,
    )
    # A bank that is not divisible by the expert axis fails inside `moe_mlp`, which is after the rack
    # is already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by {expert_axis_size}")
    local_experts = model.num_experts // expert_axis_size
    if local_experts % model.num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={model.num_expert_waves}"
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
    # Mesh and attention overrides are tagged only when they move off the hero shape, so a run at
    # the hero settings keeps the tag set every earlier hero run carries.
    shape_tags = (
        *((f"context-{context_axis_size}",) if context_axis_size != 1 else ()),
        *((f"expert-axis-{expert_axis_size}",) if expert_axis_size != HERO_EP_EXPERT_AXIS_SIZE else ()),
        *((f"seq-{model.max_seq_len}",) if max_seq_len is not None else ()),
        *((f"qk-mult-{model.qk_mult:g}",) if qk_mult is not None else ()),
    )
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=HERO_OFFLOAD_OPT_STATE,
        master_param_mode=master_param_mode,
        restore_master_param_mode=restore_master_param_mode,
        training_data_mode=training_data_mode,
        watch_mode=watch_mode,
        save_checkpoints=save_checkpoints,
        expert_axis_size=expert_axis_size,
        context_axis_size=context_axis_size,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_EP_NODES * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = _validation_datasets() if eval_every > 0 else []

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=seed,
            train_batch_size=batch_size,
            num_train_steps=total_schedule_steps,
            # Read another run's checkpoint without ever writing to it: this is a search path, and
            # the checkpointer below still points at this run's own output. `True` fails fast when
            # the path is gone, which matters when the source is a live run's rolling temporary.
            load_checkpoint_path=restore_from,
            load_checkpoint=True if restore_from else None,
            profiler=ProfilerConfig(
                enabled=profile_steps > 0,
                start_step=profile_start_step,
                num_steps=profile_steps,
                # One rank is enough for a step trace, and tracing all 64 multiplies the upload
                # without adding signal.
                process_index=0,
                profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
            ),
            mp=jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[master_param_mode]),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    backend_tag,
                    f"master-params-{master_param_mode.value.replace('_', '-')}",
                    capacity_tag,
                    *transport_capacity_tags,
                    wave_tag,
                    size_tag,
                    *shape_tags,
                    "gb200",
                    HARRIER_MIX_2026_08_17_1_TAG,
                    "MHEP",
                ],
                group="moe-hero-ep",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=watch_interval),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
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
            ),
        )
        data = harrier_mix_2026_08_17_1_data_config(
            ctx=ctx,
            total_steps=total_schedule_steps,
            batch_size=batch_size,
            max_seq_len=model.max_seq_len,
            validation=validation,
        )
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            # Off by default so a throughput run stays a throughput run. Turn it on to make a
            # run scoreable: comparing configs needs held-out loss, not train loss.
            eval=(
                GrugEvalConfig(steps_per_eval=eval_every, eval_ema=False, compute_bpb=True) if eval_every > 0 else None
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
        deps=(HARRIER_MIX_2026_08_17_1_STORE, *validation),
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
    help="Number of training steps.",
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
    default=HERO_MODEL.num_experts,
    show_default=True,
    help=(
        f"Override the routed expert count. The count must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}, "
        f"and the local expert count must support {HERO_MODEL.num_expert_waves} waves."
    ),
)
@click.option(
    "--num-experts-per-token",
    type=click.IntRange(min=1),
    default=HERO_MODEL.num_experts_per_token,
    show_default=True,
    help="Override the routed top-k. Scales both active parameters and the EP dispatch buffers.",
)
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=HERO_MODEL.intermediate_dim,
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
    "--seq-len",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Override the sequence length. Tokens per step is --batch-size times this, so halve the "
        "batch each time this doubles to hold the step's token count, and its memory, fixed."
    ),
)
@click.option(
    "--qk-mult",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help=(
        "Override the query scale applied before attention. Long-context runs raise it with the "
        "sequence, e.g. the YaRN mscale recipe 1.3 * (0.1 * ln(seq / 4096) + 1)."
    ),
)
@click.option(
    "--context-axis-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "Context-parallel shard count for the sequence dim. Attention shards Q over it and "
        "all-gathers K/V; `data` narrows by the same factor."
    ),
)
@click.option(
    "--expert-axis-size",
    type=click.IntRange(min=1),
    default=HERO_EP_EXPERT_AXIS_SIZE,
    show_default=True,
    help=(
        "Devices in the expert-parallel group. Lower it to free devices for --context-axis-size; "
        "the expert bank must stay divisible by it."
    ),
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
    default=HERO_MODEL.capacity_factor,
    show_default=True,
    help="Override the pooled receiver capacity factor.",
)
@click.option(
    "--moe-implementation",
    default=None,
    help="Override the MoE backend, e.g. ragged_all_to_all. Defaults to the hero spec.",
)
@click.option(
    "--restore-from",
    default=None,
    help=(
        "Initialize from another run's checkpoint directory, read-only. Use it to measure a "
        "trained router, where capacity clipping is real: a run from scratch drops nothing."
    ),
)
@click.option(
    "--restore-master-params",
    type=click.Choice([mode.value for mode in MasterParamMode]),
    default=None,
    help=(
        "Parameter storage the restored checkpoint was written under, when it differs from "
        "--master-params. The master is folded into fp32 device parameters after the load."
    ),
)
@click.option(
    "--master-params",
    type=click.Choice([mode.value for mode in MasterParamMode]),
    default=MasterParamMode.FP32_PINNED_HOST.value,
    show_default=True,
    help=(
        "Where the authoritative fp32 weights live. Disabling the master keeps them on device and "
        "measured 1.78 percent faster on the ragged transport, which does not need the memory relief."
    ),
)
@click.option(
    "--processes-per-task",
    type=click.IntRange(min=1),
    default=HERO_PROCESSES_PER_TASK,
    show_default=True,
    help="JAX processes per node. The ragged transport needs one process per GPU.",
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
    seq_len: int | None,
    qk_mult: float | None,
    context_axis_size: int,
    expert_axis_size: int,
    latent_dim: int | None,
    save_checkpoints: bool,
    checkpoint_minutes: float,
    checkpoint_path: str | None,
    eval_every: int,
    watch_interval: int,
    watch_mode: str,
    profile_steps: int,
    profile_start_step: int,
    training_data: str,
    moe_implementation: str | None,
    master_params: str,
    restore_from: str | None,
    restore_master_params: str | None,
    processes_per_task: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
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
        max_seq_len=seq_len,
        qk_mult=qk_mult,
        context_axis_size=context_axis_size,
        expert_axis_size=expert_axis_size,
        latent_dim=latent_dim,
        save_checkpoints=save_checkpoints,
        checkpoint_interval=timedelta(minutes=checkpoint_minutes),
        checkpoint_path=checkpoint_path,
        eval_every=eval_every,
        watch_interval=watch_interval,
        watch_mode=WatchMode(watch_mode),
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
        training_data_mode=TrainingDataMode(training_data),
        moe_implementation=moe_implementation,
        master_param_mode=MasterParamMode(master_params),
        restore_from=restore_from,
        restore_master_param_mode=MasterParamMode(restore_master_params) if restore_master_params else None,
        processes_per_task=processes_per_task,
    )


if __name__ == "__main__":
    main()
