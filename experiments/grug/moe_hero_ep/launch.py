# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the EP64 MoE hero configuration."""

import dataclasses
import os

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
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

from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    MoeComparisonMetadata,
    run_grug,
)
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
HERO_PROCESSES_PER_TASK = 1
HERO_MOK_PROCESSES_PER_TASK = HERO_GPUS_PER_NODE
HERO_PROFILE_NUM_STEPS = 3
HERO_PROFILE_PROCESS_INDEX = 0
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
HERO_COMPARISON_ENVIRONMENT = "torch-cu130-cublas13.2"
# The hero shape keeps its MuonH state on pinned host memory: 24.59 GiB of parameters and 27.78 GiB
# of optimizer state per device leave too little room for the fixed all-to-all buffers otherwise.
HERO_OFFLOAD_OPT_STATE = True

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


class HeroThroughputResult(Artifact):
    """Metrics-only result of the rack-scale throughput hero run.

    The run intentionally writes no checkpoint; it only mirrors its tracker metrics to the output
    path. This artifact is a plain path ref to those metrics, so the step does not promise a
    checkpoint it never produces.
    """


def _build_hero_run(
    *,
    run_id: str,
    num_steps: int,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    version: str | None = None,
    moe_implementation: str | None = None,
    mok_expert_placement: str | None = None,
    processes_per_task: int = HERO_PROCESSES_PER_TASK,
    runtime_pip_packages: tuple[str, ...] = (),
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the one-rack EP64 hero throughput run.

    The overrides sweep expert count, expert width, routed top-k, and routing capacity from the
    hero spec. They keep the hidden dimension, so the compute-scaled optimizer values stay
    comparable across a sweep. ``None`` keeps the hero value.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if profile_start_step is not None:
        if not 1 <= profile_start_step < num_steps:
            raise ValueError(
                f"profile_start_step must be in [1, num_steps), got {profile_start_step} for {num_steps} steps"
            )
        if profile_num_steps <= 0:
            raise ValueError(f"profile_num_steps must be positive, got {profile_num_steps}")

    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=HERO_EP_BATCH_SIZE)
    overrides = {
        name: value
        for name, value in (
            ("num_experts", num_experts),
            ("num_experts_per_token", num_experts_per_token),
            ("intermediate_dim", intermediate_dim),
            ("capacity_factor", capacity_factor),
            ("mok_expert_placement", mok_expert_placement),
        )
        if value is not None
    }
    if overrides:
        model = dataclasses.replace(model, **overrides)
    if moe_implementation is not None:
        model = dataclasses.replace(model, moe_implementation=moe_implementation)
    # A bank that does not divide the expert axis fails inside `moe_mlp`, which is after the rack is
    # already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % HERO_EP_EXPERT_AXIS_SIZE != 0:
        raise ValueError(f"num_experts={model.num_experts} must divide the expert axis {HERO_EP_EXPERT_AXIS_SIZE}")
    if model.moe_implementation is None:
        raise ValueError("the EP hero requires an explicit MoE implementation")
    backend_tag = model.moe_implementation.replace("_", "-")
    is_mok = model.moe_implementation == "mok"
    if is_mok and capacity_factor is not None:
        raise ValueError("capacity_factor does not apply to the dropless mok backend")
    routing_semantics = "dropless" if is_mok else f"capacity-{model.capacity_factor:g}"
    size_tag = f"e{model.num_experts}-i{model.intermediate_dim}"
    process_tag = f"processes-per-task-{processes_per_task}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=HERO_OFFLOAD_OPT_STATE,
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
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

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=HERO_EP_BATCH_SIZE,
            num_train_steps=num_steps,
            profiler=(
                ProfilerConfig(enabled=False)
                if profile_start_step is None
                else ProfilerConfig(
                    enabled=True,
                    start_step=profile_start_step,
                    num_steps=profile_num_steps,
                    process_index=HERO_PROFILE_PROCESS_INDEX,
                    profile_options=ProfileOptionsConfig(
                        host_tracer_level=1,
                        python_tracer_level=0,
                        enable_hlo_proto=True,
                    ),
                )
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
                    size_tag,
                    routing_semantics,
                    f"expert-placement-{model.mok_expert_placement}" if is_mok else "expert-placement-contiguous",
                    process_tag,
                    HERO_COMPARISON_ENVIRONMENT,
                    "gb200",
                    "MHEP",
                ],
                group="moe-hero-ep",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=0),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
        )
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=None,
            processes_per_task=processes_per_task,
            runtime_pip_packages=runtime_pip_packages,
            comparison=MoeComparisonMetadata(
                backend=model.moe_implementation,
                routing_semantics=routing_semantics,
                fused_shared_experts=model.num_shared_experts if is_mok else 0,
                software_environment=HERO_COMPARISON_ENVIRONMENT,
                processes_per_task=processes_per_task,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
    )


def build_hero_run(
    *,
    run_id: str,
    num_steps: int,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the existing capacity-limited fixed-all-to-all EP64 baseline."""
    return _build_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        version=version,
    )


def build_mok_hero_run(
    *,
    run_id: str,
    num_steps: int,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    mok_package: str,
    mok_expert_placement: str = "contiguous",
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the dropless MoK comparison arm with one JAX process per GB200."""
    return _build_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        version=version,
        moe_implementation="mok",
        mok_expert_placement=mok_expert_placement,
        processes_per_task=HERO_MOK_PROCESSES_PER_TASK,
        runtime_pip_packages=(mok_package,),
        profile_start_step=profile_start_step,
        profile_num_steps=profile_num_steps,
    )


def build_multiprocess_hero_run(
    *,
    run_id: str,
    num_steps: int,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build a four-process-per-node fixed-all-to-all control matched to MoK's topology."""
    return _build_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        version=version,
        processes_per_task=HERO_MOK_PROCESSES_PER_TASK,
        profile_start_step=profile_start_step,
        profile_num_steps=profile_num_steps,
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
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the fixed all-to-all capacity factor.",
)
@build_options
def main(
    run_id: str,
    num_steps: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
    )


if __name__ == "__main__":
    main()
