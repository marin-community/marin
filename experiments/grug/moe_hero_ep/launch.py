# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the EP64 MoE hero configuration."""

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
from experiments.grug.moe_hero_ep.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
HERO_PROCESSES_PER_TASK = 1
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
# The hero shape keeps its MuonH state on pinned host memory: 24.59 GiB of parameters and 27.78 GiB
# of optimizer state per device leave too little room for the fixed all-to-all buffers otherwise.
HERO_OFFLOAD_OPT_STATE = True
# Grad/param norm reductions run outside the scanned step and cost a visible slice of a short run's
# wall clock. Off by default; --watch-interval buys the diagnostic back every N steps.
HERO_WATCH_INTERVAL = 0
HERO_CHECKPOINT_INTERVAL = timedelta(minutes=15)


def _hero_watch_config(interval: int, full: bool = False) -> WatchConfig:
    """Global gradient norm only, which is what a stability check needs.

    `WatchConfig`'s defaults watch grads *and* params, take a norm per parameter tensor, and set
    `split_scan_layers`, which un-stacks the 48 scanned layers. At the hero shape that combination
    needs a ~117 GiB temporary against a 173 GiB device and reliably OOMs a run that fits without
    it -- measured twice, on shapes that had already completed 25 steps with the watch off.
    """
    if interval <= 0:
        return WatchConfig(watch_targets=[])
    if full:
        return WatchConfig(interval=interval)
    return WatchConfig(
        watch_targets=["grads"],
        include_per_parameter_norms=False,
        split_scan_layers=False,
        interval=interval,
    )


_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


@dataclasses.dataclass(frozen=True)
class Flavor:
    """How the MoE layer shards, and what that implies for routing capacity.

    ``ep`` spans the rack with expert parallelism and drops every assignment above a fixed
    (sender shard, expert) cell. ``fsdp-nodrop`` keeps one expert axis, so every device holds the
    whole bank and computes every assignment. That is the drop-free control the throughput
    comparison needs: the recorded FSDP reference uses ``sonic_cute`` with ``expert_chunks`` above
    one and still drops about 1.9 percent, so it cannot price the cost of dropping.
    """

    expert_axis_size: int
    moe_implementation: str


FLAVORS: dict[str, Flavor] = {
    "ep": Flavor(HERO_EP_EXPERT_AXIS_SIZE, "fixed_all_to_all"),
    # The upstream transport. Its capacity factor scales one pooled receiver buffer per device
    # rather than a per-(sender, expert) cell, so at the same factor it buys the same bytes and
    # drops far less -- every expert on a device draws from one pool. `fixed_all_to_all` exists
    # because this path measured slow, which is what a same-shape run is for.
    "ep-ragged": Flavor(HERO_EP_EXPERT_AXIS_SIZE, "ragged_all_to_all"),
    "fsdp-nodrop": Flavor(1, "scatter"),
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

    The run mirrors its tracker metrics to the output path. It writes no checkpoint by default, so
    this artifact is a plain path ref to those metrics rather than a promise of one; pass
    ``--save-checkpoints`` for a run long enough that restarting from step 0 is not acceptable.
    """


def build_hero_run(
    *,
    run_id: str,
    num_steps: int,
    schedule_steps: int | None = None,
    seed: int = 0,
    batch_size: int = HERO_EP_BATCH_SIZE,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    latent_dim: int | None = None,
    flavor: str = "ep",
    eval_every: int = 0,
    save_checkpoints: bool = False,
    watch_interval: int = HERO_WATCH_INTERVAL,
    full_watch: bool = False,
    profile_steps: int = 0,
    profile_start_step: int = 5,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the one-rack EP64 hero throughput run.

    The overrides sweep expert count, expert width, routed top-k, and routing capacity from the
    hero spec. They keep the hidden dimension, so the compute-scaled optimizer values stay
    comparable across a sweep. ``None`` keeps the hero value.

    Two arguments do move those optimizer values, deliberately: ``batch_size`` and
    ``schedule_steps`` both change the token budget the heuristic scales from. ``flavor`` also
    changes the expert axis, so ``fsdp-nodrop`` runs on axis 1 rather than EP64.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if flavor not in FLAVORS:
        raise ValueError(f"flavor must be one of {sorted(FLAVORS)}, got {flavor!r}")
    sharding = FLAVORS[flavor]
    # `scatter` computes every assignment, so a capacity factor would be silently inert. Reject it
    # rather than let a sweep think it swept something.
    if capacity_factor is not None and sharding.moe_implementation == "scatter":
        raise ValueError(f"flavor {flavor!r} never drops, so --capacity-factor has no effect")

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
        moe_implementation=sharding.moe_implementation,
        # Every flavor here is chunk-free; `expert_chunks` above one is the local dropping path
        # that only the separate FSDP hero uses.
        expert_chunks=1,
    )
    # A bank that does not divide the expert axis fails inside `moe_mlp`, which is after the rack is
    # already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % sharding.expert_axis_size != 0:
        raise ValueError(f"num_experts={model.num_experts} must divide the expert axis {sharding.expert_axis_size}")
    if model.moe_implementation is None:
        raise ValueError("the EP hero requires an explicit MoE implementation")
    backend_tag = model.moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    size_tag = f"e{model.num_experts}-i{model.intermediate_dim}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=HERO_OFFLOAD_OPT_STATE,
        # A 25-step throughput gate does not need checkpoints, and writing the offloaded optimizer
        # state costs more than the gate itself. A multi-thousand-step run on a contended rack does:
        # without this, every preemption restarts from step 0, so a run longer than the mean time
        # between evictions never finishes.
        save_checkpoints=save_checkpoints,
        expert_axis_size=sharding.expert_axis_size,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_EP_NODES,
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
                profile_options=ProfileOptionsConfig(enable_hlo_proto=profile_steps > 0),
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
            watch=_hero_watch_config(watch_interval, full_watch),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            # Write under the step's output path. Levanter's default base path is pod-local, so a
            # preempted run would find nothing to resume from and --save-checkpoints would buy
            # nothing.
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=HERO_CHECKPOINT_INTERVAL,
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
            processes_per_task=HERO_PROCESSES_PER_TASK,
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
    default=None,
    help=f"Override the routed expert count. Must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}.",
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
    help="Sequences per step. Scales the compute-scaled optimizer, so hold it fixed across a sweep.",
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
    help="MoE sharding: expert-parallel with routing capacity, or FSDP that never drops.",
)
@click.option(
    "--save-checkpoints/--no-save-checkpoints",
    default=False,
    show_default=True,
    help=(
        "Write resumable checkpoints. Off by default because a short throughput gate does not need "
        "them. Turn it on for long runs on a contended rack: without checkpoints a preemption "
        "restarts from step 0."
    ),
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=HERO_WATCH_INTERVAL,
    show_default=True,
    help="Steps between grad/param norm dumps. 0 disables them.",
)
@click.option(
    "--full-watch/--lean-watch",
    default=False,
    help=(
        "Use WatchConfig's upstream defaults (grads and params, per-parameter norms, split scan "
        "layers) instead of a single global gradient norm."
    ),
)
@click.option(
    "--eval-every",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Run the paloma suite every N steps. 0 disables eval (throughput-only run).",
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
    num_steps: int,
    schedule_steps: int | None,
    seed: int,
    batch_size: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
    latent_dim: int | None,
    flavor: str,
    save_checkpoints: bool,
    watch_interval: int,
    full_watch: bool,
    eval_every: int,
    profile_steps: int,
    profile_start_step: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        schedule_steps=schedule_steps,
        seed=seed,
        batch_size=batch_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        latent_dim=latent_dim,
        flavor=flavor,
        save_checkpoints=save_checkpoints,
        watch_interval=watch_interval,
        full_watch=full_watch,
        eval_every=eval_every,
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
    )


if __name__ == "__main__":
    main()
