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
from levanter.grug.grug_moe import (
    MoeImplementation,
    MoonEPBucketSchedule,
    MoonEPConfig,
    MoonEPGroupedGemm,
    MoonEPMode,
    resolve_moe_implementation,
)
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
from experiments.grug.moe_hero_ep.jax_wheel_setup import MoonEPJaxWheelBuild
from experiments.grug.moe_hero_ep.model import RematMode
from experiments.grug.moe_hero_ep.quantile_balancing import QuantileBalancingMethod
from experiments.grug.moe_hero_ep.train import (
    FiniteDiagnostics,
    GrugRunConfig,
    GrugTrainerConfig,
    MoonEPTransport,
    run_grug,
)
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
MIN_HERO_PROFILE_START_STEP = 3
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
HERO_PROCESSES_PER_TASK = 1
HERO_WORKER_CPU = 32
HERO_WORKER_RAM_GB = 256
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"

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


def _hero_profiler_config(start_step: int | None, num_steps: int) -> ProfilerConfig:
    if start_step is None:
        return ProfilerConfig(enabled=False)
    return ProfilerConfig(
        enabled=True,
        start_step=start_step,
        num_steps=num_steps,
        process_index=0,
        barrier_timeout=600,
        profile_options=ProfileOptionsConfig(
            host_tracer_level=1,
            python_tracer_level=0,
            enable_hlo_proto=True,
            advanced_configuration={
                # A MoonEP step contains more than 100,000 events. Keep enough data for a full-step profile.
                "gpu_max_activity_api_events": 1_000_000,
                "gpu_max_callback_api_events": 1_000_000,
                "gpu_num_chips_to_profile_per_task": 1,
            },
        ),
    )


class HeroThroughputResult(Artifact):
    """Metrics-only result of the rack-scale throughput hero run.

    The run intentionally writes no checkpoint; it only mirrors its tracker metrics to the output
    path. This artifact is a plain path ref to those metrics, so the step does not promise a
    checkpoint it never produces.
    """


def build_hero_run(
    *,
    run_id: str,
    num_steps: int,
    moe_implementation: MoeImplementation = "fixed_all_to_all",
    moonep_token_padding: int = 128,
    moonep_token_buckets: int = 1,
    moonep_bucket_schedule: MoonEPBucketSchedule = MoonEPBucketSchedule.EAGER_DISPATCH,
    moonep_grouped_gemm: MoonEPGroupedGemm = MoonEPGroupedGemm.QUACK,
    moonep_mode: MoonEPMode = MoonEPMode.EXACT,
    moonep_fixed_capacity_factor: float = 1.1,
    qb_method: QuantileBalancingMethod = QuantileBalancingMethod.LOCAL_EXACT,
    qb_histogram_bins: int = 1000,
    moonep_jax_wheel_build: MoonEPJaxWheelBuild | None = None,
    moonep_transport: MoonEPTransport = MoonEPTransport.TWO_SLICE,
    remat_mode: RematMode = "recompute_all",
    processes_per_task: int = HERO_PROCESSES_PER_TASK,
    worker_cpu: int = HERO_WORKER_CPU,
    worker_ram_gb: int = HERO_WORKER_RAM_GB,
    finite_diagnostics: FiniteDiagnostics = FiniteDiagnostics.NONE,
    profile_start_step: int | None = None,
    profile_num_steps: int = 2,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the one-rack EP64 hero throughput run."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if profile_num_steps <= 0:
        raise ValueError(f"profile_num_steps must be positive, got {profile_num_steps}")
    if processes_per_task <= 0 or HERO_GPUS_PER_NODE % processes_per_task != 0:
        raise ValueError(f"processes_per_task={processes_per_task} must divide {HERO_GPUS_PER_NODE} GPUs per node")
    if worker_cpu <= 0:
        raise ValueError(f"worker_cpu must be positive, got {worker_cpu}")
    if worker_ram_gb <= 0:
        raise ValueError(f"worker_ram_gb must be positive, got {worker_ram_gb}")
    if profile_start_step is not None and profile_start_step < MIN_HERO_PROFILE_START_STEP:
        raise ValueError(f"profile_start_step must be at least {MIN_HERO_PROFILE_START_STEP}")
    if profile_start_step is not None and profile_start_step + profile_num_steps > num_steps:
        raise ValueError("the profile window must fit within the training run")
    if moonep_jax_wheel_build is not None and moe_implementation != "moonep_jax":
        raise ValueError("a MoonEP JAX wheel build requires the moonep_jax implementation")

    moonep_config = (
        MoonEPConfig(
            token_padding=moonep_token_padding,
            token_buckets=moonep_token_buckets,
            bucket_schedule=moonep_bucket_schedule,
            grouped_gemm=moonep_grouped_gemm,
            mode=moonep_mode,
            fixed_capacity_factor=moonep_fixed_capacity_factor,
        )
        if moe_implementation == "moonep_jax"
        else None
    )
    model, optimizer = build_hero_configs(
        num_train_steps=num_steps,
        batch_size=HERO_EP_BATCH_SIZE,
        moe_implementation=moe_implementation,
        moonep_config=moonep_config,
        remat_mode=remat_mode,
        qb_method=qb_method,
        qb_histogram_bins=qb_histogram_bins,
    )
    if model.moe_implementation is None:
        raise ValueError("the EP hero requires an explicit MoE implementation")
    backend_tag = model.moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    qb_tag = f"qb-{model.qb_method.value.replace('_', '-')}"
    transport_tag = f"transport-{moonep_transport.value.replace('_', '-')}"
    experiment_tag = "MNEP" if run_id.upper().startswith("MNEP") else "MHEP"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    if (
        moe_implementation == "moonep_jax"
        and moonep_transport == MoonEPTransport.DIRECT_DEVICE
        and finite_diagnostics == FiniteDiagnostics.NONE
    ):
        finite_diagnostics = FiniteDiagnostics.GRADS

    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,
        finite_diagnostics=finite_diagnostics,
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=worker_cpu,
        ram=f"{worker_ram_gb}g",
        disk="256g",
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
            profiler=_hero_profiler_config(profile_start_step, profile_num_steps),
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
                    capacity_tag,
                    qb_tag,
                    transport_tag,
                    "gb200",
                    experiment_tag,
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
            moonep_jax_wheel_build=moonep_jax_wheel_build,
            moonep_transport=moonep_transport,
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
    "--moe-implementation",
    type=click.Choice(["fixed_all_to_all", "moonep_jax"]),
    default="fixed_all_to_all",
    show_default=True,
    help="Expert-parallel transport backend.",
)
@click.option(
    "--moonep-token-padding",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="MoonEP compute-group padding.",
)
@click.option(
    "--moonep-grouped-gemm",
    type=click.Choice([choice.value for choice in MoonEPGroupedGemm]),
    default=MoonEPGroupedGemm.QUACK.value,
    show_default=True,
    help="MoonEP grouped GEMM implementation.",
)
@click.option(
    "--moonep-token-buckets",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Token exchange buckets for communication and compute overlap.",
)
@click.option(
    "--moonep-bucket-schedule",
    type=click.Choice([schedule.value for schedule in MoonEPBucketSchedule]),
    default=MoonEPBucketSchedule.EAGER_DISPATCH.value,
    show_default=True,
    help="Order for token dispatch and expert compute.",
)
@click.option(
    "--moonep-fixed-capacity-factor",
    type=click.FloatRange(min=1.0),
    default=1.1,
    show_default=True,
    help="No-drop fixed all-to-all capacity factor.",
)
@click.option(
    "--moonep-mode",
    type=click.Choice([mode.value for mode in MoonEPMode]),
    default=MoonEPMode.EXACT.value,
    show_default=True,
    help="Static MoonEP execution schedule.",
)
@click.option(
    "--qb-method",
    type=click.Choice([method.value for method in QuantileBalancingMethod]),
    default=QuantileBalancingMethod.LOCAL_EXACT.value,
    show_default=True,
    help="Quantile-balancing estimator.",
)
@click.option(
    "--qb-histogram-bins",
    type=click.IntRange(min=2),
    default=1000,
    show_default=True,
    help="Number of global histogram bins.",
)
@click.option(
    "--moonep-jax-wheel-build",
    type=click.Choice([build.value for build in MoonEPJaxWheelBuild]),
    default=None,
    help="Fixed JAX wheel build for MoonEP rack runs.",
)
@click.option(
    "--moonep-transport",
    type=click.Choice([transport.value for transport in MoonEPTransport]),
    default=MoonEPTransport.TWO_SLICE.value,
    show_default=True,
    help="XLA transport for MoonEP ragged collectives.",
)
@click.option(
    "--remat-mode",
    type=click.Choice(["recompute_all", "save_moe"]),
    default="recompute_all",
    show_default=True,
    help="Block values saved for the backward pass.",
)
@click.option(
    "--processes-per-task",
    type=click.IntRange(min=1, max=HERO_GPUS_PER_NODE),
    default=HERO_PROCESSES_PER_TASK,
    show_default=True,
    help="JAX processes per four-GPU rack worker.",
)
@click.option(
    "--worker-cpu",
    type=click.IntRange(min=1),
    default=HERO_WORKER_CPU,
    show_default=True,
    help="CPU count for each rack worker.",
)
@click.option(
    "--worker-ram-gb",
    type=click.IntRange(min=1),
    default=HERO_WORKER_RAM_GB,
    show_default=True,
    help="RAM in GiB for each rack worker.",
)
@click.option(
    "--finite-diagnostics",
    type=click.Choice([diagnostics.value for diagnostics in FiniteDiagnostics]),
    default=FiniteDiagnostics.NONE.value,
    show_default=True,
    help="Scan each training boundary for non-finite values.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=MIN_HERO_PROFILE_START_STEP),
    default=None,
    help="First training step in the one-process XPlane profile window.",
)
@click.option(
    "--profile-num-steps",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Number of training steps in the profile window.",
)
@build_options
def main(
    run_id: str,
    num_steps: int,
    moe_implementation: str,
    moonep_token_padding: int,
    moonep_token_buckets: int,
    moonep_bucket_schedule: str,
    moonep_grouped_gemm: str,
    moonep_mode: str,
    moonep_fixed_capacity_factor: float,
    qb_method: str,
    qb_histogram_bins: int,
    moonep_jax_wheel_build: str | None,
    moonep_transport: str,
    remat_mode: RematMode,
    processes_per_task: int,
    worker_cpu: int,
    worker_ram_gb: int,
    finite_diagnostics: str,
    profile_start_step: int | None,
    profile_num_steps: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        moe_implementation=resolve_moe_implementation(moe_implementation),
        moonep_token_padding=moonep_token_padding,
        moonep_token_buckets=moonep_token_buckets,
        moonep_bucket_schedule=MoonEPBucketSchedule(moonep_bucket_schedule),
        moonep_grouped_gemm=MoonEPGroupedGemm(moonep_grouped_gemm),
        moonep_mode=MoonEPMode(moonep_mode),
        moonep_fixed_capacity_factor=moonep_fixed_capacity_factor,
        qb_method=QuantileBalancingMethod(qb_method),
        qb_histogram_bins=qb_histogram_bins,
        moonep_jax_wheel_build=(
            MoonEPJaxWheelBuild(moonep_jax_wheel_build) if moonep_jax_wheel_build is not None else None
        ),
        moonep_transport=MoonEPTransport(moonep_transport),
        remat_mode=remat_mode,
        processes_per_task=processes_per_task,
        worker_cpu=worker_cpu,
        worker_ram_gb=worker_ram_gb,
        finite_diagnostics=FiniteDiagnostics(finite_diagnostics),
        profile_start_step=profile_start_step,
        profile_num_steps=profile_num_steps,
    )


if __name__ == "__main__":
    main()
