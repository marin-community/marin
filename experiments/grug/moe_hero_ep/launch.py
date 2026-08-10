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
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem import prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, WatchMode, run_grug
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
# One JAX process per GPU. The old one-process-per-node layout has two reproducible
# failure modes on this stack, both from auto-PGLE (which only engages per-node):
# a profiling-session collision that kills the gang during early dispatch
# (ALREADY_EXISTS: Another profiling session active), and a silent wedge at the
# FDO-recompile clique re-initialization (#7344). Per-GPU also matches the FSDP
# hero (#8040) and is required by the ragged EP backend (#8081).
HERO_PROCESSES_PER_TASK = HERO_GPUS_PER_NODE
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
# The hero shape keeps its MuonH state on pinned host memory: 24.59 GiB of parameters and 27.78 GiB
# of optimizer state per device leave too little room for the fixed all-to-all buffers otherwise.
HERO_OFFLOAD_OPT_STATE = True
HERO_WATCH_INTERVAL = 0
HERO_CHECKPOINT_INTERVAL = timedelta(minutes=15)

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


FLAVORS: dict[str, str] = {
    "ep": "fixed_all_to_all",
    # The upstream transport. Its capacity factor scales one pooled receiver buffer per device
    # rather than a per-(sender, expert) cell, so at the same factor it buys the same bytes and
    # drops far less -- every expert on a device draws from one pool. `fixed_all_to_all` exists
    # because this path measured slow, which is what a same-shape run is for.
    "ep-ragged": "ragged_all_to_all",
}


# Held-out sets, added at weight 0 so they surface as tagged eval sets. The hero trains on
# llama3-tokenized SlimPajama, so these must carry the same tokenizer.
#
# Paloma only, deliberately. `paloma_dataset` and `uncheatable_dataset` both hardcode a `-llama3`
# suffix in the cache name while taking an arbitrary `tokenizer` argument, so callers asking for
# different tokenizers collide on one cache identity and whoever materializes first wins. The
# uncheatable caches under that name currently hold marin-tokenizer data, which fails the mixture's
# single-tokenizer check. Paloma is also the suite the scaling-law scoring uses, so dropping
# uncheatable costs nothing here.
def _validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    return list(paloma_datasets(tokenizer=llama3_tokenizer).values())


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


class HeroThroughputResult(Artifact):
    """Result of the rack-scale throughput hero run.

    The run mirrors its tracker metrics to the output path. It writes no checkpoint by default.
    Checkpoint restore has a known memory-kind mismatch with the offloaded optimizer state.
    """


def build_hero_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None = None,
    seed: int = 0,
    nodes: int = HERO_EP_NODES,
    processes_per_task: int = HERO_PROCESSES_PER_TASK,
    batch_size: int = HERO_EP_BATCH_SIZE,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    latent_dim: int | None = None,
    flavor: str = "ep",
    eval_every: int = 0,
    save_checkpoints: bool = False,
    checkpoint_interval: timedelta = HERO_CHECKPOINT_INTERVAL,
    checkpoint_path: str | None = None,
    watch_interval: int = HERO_WATCH_INTERVAL,
    watch_mode: WatchMode = WatchMode.INLINE,
    profile_steps: int = 0,
    profile_start_step: int = 5,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the EP64 hero throughput run.

    The overrides sweep expert count, expert width, routed top-k, and routing capacity from the
    hero spec. They keep the hidden dimension, so the compute-scaled optimizer values stay
    comparable across a sweep. ``None`` keeps the hero value.

    ``batch_size`` is the global batch across all data-parallel racks. It does not scale with
    ``dp_racks``. ``batch_size`` and ``schedule_steps`` change the token budget for the heuristic.

    ``nodes`` sizes one expert mesh: the expert axis spans every GPU of that many workers, so the
    default 16 nodes is the EP64 hero and 4 nodes is an EP16 slice of it. ``processes_per_task``
    selects the process topology on each worker: 1 keeps one four-device JAX process per node,
    ``HERO_GPUS_PER_NODE`` runs one single-device process per GPU.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if nodes <= 0:
        raise ValueError(f"nodes must be positive, got {nodes}")
    if processes_per_task not in (1, HERO_GPUS_PER_NODE):
        raise ValueError(f"processes_per_task must be 1 or {HERO_GPUS_PER_NODE}, got {processes_per_task}")
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
    if flavor not in FLAVORS:
        raise ValueError(f"flavor must be one of {sorted(FLAVORS)}, got {flavor!r}")
    moe_implementation = FLAVORS[flavor]
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
            ("latent_dim", latent_dim),
        )
        if value is not None
    }
    if overrides:
        model = dataclasses.replace(model, **overrides)
    model = dataclasses.replace(
        model,
        moe_implementation=moe_implementation,
        expert_chunks=1,
    )
    expert_axis_size = nodes * HERO_GPUS_PER_NODE
    # A bank that does not divide the expert axis fails inside `moe_mlp`, which is after the rack is
    # already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={model.num_experts} must divide the expert axis {expert_axis_size}")
    backend_tag = moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    size_tag = f"e{model.num_experts}-i{model.intermediate_dim}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=HERO_OFFLOAD_OPT_STATE,
        watch_mode=watch_mode,
        # The default offloaded optimizer state has a known memory-kind mismatch during restore.
        save_checkpoints=save_checkpoints,
        expert_axis_size=expert_axis_size,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=nodes * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()
    validation = _validation_datasets() if eval_every > 0 else []

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=seed,
            train_batch_size=batch_size,
            num_train_steps=total_schedule_steps,
            profiler=ProfilerConfig(
                enabled=profile_steps > 0,
                start_step=profile_start_step,
                num_steps=profile_steps,
                # One rank is enough for a step trace, and tracing all 64 multiplies the upload
                # without adding signal.
                process_index=0,
                profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
            ),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    backend_tag,
                    f"flavor-{flavor}",
                    capacity_tag,
                    size_tag,
                    "gb200",
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
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, validation=validation, shuffle=_SLIMPAJAMA_SHUFFLE),
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
        deps=(slim, *validation),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--dp-racks",
    type=click.IntRange(min=1),
    required=True,
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
    "--nodes",
    type=click.IntRange(min=1),
    default=HERO_EP_NODES,
    show_default=True,
    help="GB200 workers per expert mesh. The expert axis spans every GPU of these nodes "
    f"({HERO_GPUS_PER_NODE} per node), so 16 nodes is the EP64 hero and 4 nodes is EP16.",
)
@click.option(
    "--processes-per-task",
    type=click.Choice(["1", str(HERO_GPUS_PER_NODE)]),
    default=str(HERO_PROCESSES_PER_TASK),
    show_default=True,
    help="JAX processes per worker: 1 is one four-device process per node, "
    f"{HERO_GPUS_PER_NODE} is one single-device process per GPU (multi-controller).",
)
@click.option(
    "--num-experts",
    type=click.IntRange(min=1),
    default=None,
    help="Override the routed expert count. Must be divisible by the expert axis (4 x --nodes).",
)
@click.option(
    "--num-experts-per-token",
    type=click.IntRange(min=1),
    default=None,
    help="Override the routed top-k. Scales both active parameters and the EP dispatch buffers.",
)
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=None,
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
    "--latent-dim",
    type=click.IntRange(min=1),
    default=None,
    help="LatentMoE: run routed experts at this width. Divides all-to-all traffic by hidden/latent.",
)
@click.option(
    "--flavor",
    type=click.Choice(sorted(FLAVORS)),
    default="ep",
    show_default=True,
    help="Expert-parallel MoE implementation.",
)
@click.option(
    "--save-checkpoints/--no-save-checkpoints",
    default=False,
    show_default=True,
    help="Write checkpoints. Restore is not supported for the pinned-host optimizer state.",
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
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the fixed all-to-all capacity factor.",
)
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None,
    seed: int,
    nodes: int,
    processes_per_task: str,
    batch_size: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
    latent_dim: int | None,
    flavor: str,
    save_checkpoints: bool,
    checkpoint_minutes: float,
    checkpoint_path: str | None,
    eval_every: int,
    watch_interval: int,
    watch_mode: str,
    profile_steps: int,
    profile_start_step: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        schedule_steps=schedule_steps,
        seed=seed,
        nodes=nodes,
        processes_per_task=int(processes_per_task),
        batch_size=batch_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        latent_dim=latent_dim,
        flavor=flavor,
        save_checkpoints=save_checkpoints,
        checkpoint_interval=timedelta(minutes=checkpoint_minutes),
        checkpoint_path=checkpoint_path,
        eval_every=eval_every,
        watch_interval=watch_interval,
        watch_mode=WatchMode(watch_mode),
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
    )


if __name__ == "__main__":
    main()
