# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the Mixture-of-Kittens EP64 backend."""

import dataclasses
import os

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeForwardXStorage,
    MokLikeTopology,
)
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.launch import _SLIMPAJAMA_SHUFFLE, _slimpajama_6b_dataset
from experiments.grug.moe_hero_ep.launch_mok_like import (
    MOK_LIKE_BUILD_PACKAGES,
    MOK_LIKE_BUILD_ROOT,
    MOK_LIKE_SOURCE_ROOT,
    PROMOTED_MOK_LIKE_PINNED_HOST_MEMORY_LIMIT_GB,
)
from experiments.grug.moe_hero_ep.train import (
    GpuAllocator,
    GpuDefaultPoolPreallocation,
    GpuTempBufferPool,
    GrugRunConfig,
    GrugTrainerConfig,
    XlaAutotuneCacheMode,
    run_grug,
)

DEFAULT_STEPS = 25
DEFAULT_SCHEDULE_CAPACITY_FACTOR = 4.0
DEFAULT_MAX_LEARNING_RATE = 0.05
STRICT_DROPLESS_SCHEDULE_CAPACITY_FACTOR = 64.0
EP64_NODES = 16
GPUS_PER_NODE = 4
PROCESSES_PER_TASK = 4
EXPERT_AXIS_SIZE = EP64_NODES * GPUS_PER_NODE
GLOBAL_BATCH_SIZE = 1024
MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"


class MokEp64Result(Artifact):
    """Metrics-only result from one MoK EP64 validation run."""


def build_mok_ep64_run(
    *,
    run_id: str,
    num_steps: int,
    num_layers: int | None = None,
    schedule_capacity_factor: float = DEFAULT_SCHEDULE_CAPACITY_FACTOR,
    max_learning_rate: float = DEFAULT_MAX_LEARNING_RATE,
    version: str | None = None,
) -> ArtifactStep[MokEp64Result]:
    """Build the one-rack, one-JAX-process-per-GPU MoK EP64 arm."""

    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if schedule_capacity_factor < 1:
        raise ValueError("schedule_capacity_factor must be at least one")
    if max_learning_rate <= 0:
        raise ValueError("max_learning_rate must be positive")

    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=GLOBAL_BATCH_SIZE)
    optimizer = dataclasses.replace(
        optimizer,
        learning_rate=min(optimizer.learning_rate, max_learning_rate),
        adam_lr=min(optimizer.adam_lr, max_learning_rate),
    )
    model = dataclasses.replace(
        model,
        **({"num_layers": num_layers} if num_layers is not None else {}),
        moe_implementation="fixed_all_to_all",
        mok_like=MokLikeConfig(
            topology=MokLikeTopology.NVLINK_EP64,
            schedule_capacity_factor=schedule_capacity_factor,
            workspace_slots=1,
            forward_x_storage=MokLikeForwardXStorage.RUNTIME_STAGED,
            backward_peer_storage=MokLikeBackwardPeerStorage.RUNTIME_STAGED,
        ),
        expert_chunks=1,
        remat_mode="offload_moe",
    )
    if model.num_experts % EXPERT_AXIS_SIZE != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by EP64")

    capacity_policy = (
        "strict-dropless-ep64"
        if schedule_capacity_factor == STRICT_DROPLESS_SCHEDULE_CAPACITY_FACTOR
        else "capacity-limited"
    )
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=GPUS_PER_NODE,
        cpu=96,
        ram="900g",
        disk="1t",
        replicas=EP64_NODES,
    )
    grug_trainer = GrugTrainerConfig(
        data_seed=0,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        expert_axis_size=EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    name = f"grug/mok-ep64/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()
    wandb_project = os.environ.get("WANDB_PROJECT") or "marin_moe"

    def build_config(ctx: StepContext) -> GrugRunConfig:
        profiler_start_step = 80 if num_steps >= 100 else 5
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=GLOBAL_BATCH_SIZE,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(
                enabled=num_steps >= 10,
                start_step=profiler_start_step,
                num_steps=5,
                process_index=0,
                profile_options=ProfileOptionsConfig(
                    host_tracer_level=1,
                    python_tracer_level=0,
                    enable_hlo_proto=True,
                ),
            ),
            mp=jmp.get_policy(MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "mok-like",
                    "mok-ep64",
                    "ep-64",
                    "dp-1",
                    "jax-processes-64",
                    "processes-per-task-4",
                    "local-experts-2",
                    "symmetric-runtime-staging",
                    "mok-like-workspace-slots-1",
                    f"mok-like-schedule-capacity-{schedule_capacity_factor:g}",
                    f"max-learning-rate-{max_learning_rate:g}",
                    capacity_policy,
                    "allocator-cuda-async",
                    "device-memory-0.8",
                    f"mok-like-pinned-host-memory-{PROMOTED_MOK_LIKE_PINNED_HOST_MEMORY_LIMIT_GB}gb",
                    f"steps-{num_steps}",
                ],
                group="moe-hero-mok-ep64",
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
            mok_like_build=MokLikeBuildConfig(
                source_root=MOK_LIKE_SOURCE_ROOT,
                cache_root=MOK_LIKE_BUILD_ROOT,
                cuda_arch="sm_100a",
                clone_if_missing=True,
            ),
            mok_like_pinned_host_memory_limit_gb=PROMOTED_MOK_LIKE_PINNED_HOST_MEMORY_LIMIT_GB,
            gpu_allocator=GpuAllocator.CUDA_ASYNC,
            gpu_temp_buffer_pool=GpuTempBufferPool.SHARED,
            gpu_default_pool_preallocation=GpuDefaultPoolPreallocation.EAGER,
            gpu_default_pool_trim_interval_updates=None,
            xla_autotune_cache_mode=XlaAutotuneCacheMode.LOCAL_ONLY,
            gpu_device_memory_fraction=0.80,
            processes_per_task=PROCESSES_PER_TASK,
            pip_packages=MOK_LIKE_BUILD_PACKAGES,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=MokEp64Result,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": resources},
    )


@click.command()
@click.option("--run-id", required=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=DEFAULT_STEPS, show_default=True)
@click.option("--num-layers", type=click.IntRange(min=1), default=None)
@click.option(
    "--schedule-capacity-factor",
    type=click.FloatRange(min=1),
    default=DEFAULT_SCHEDULE_CAPACITY_FACTOR,
    show_default=True,
    help="Factor 64 is strict all-to-one dropless; smaller values are capacity-limited.",
)
@click.option(
    "--max-learning-rate",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_MAX_LEARNING_RATE,
    show_default=True,
    help="Clamp both MuonH and Adam peak learning rates.",
)
@build_options
def main(
    run_id: str,
    num_steps: int,
    num_layers: int | None,
    schedule_capacity_factor: float,
    max_learning_rate: float,
) -> ArtifactStep[MokEp64Result]:
    return build_mok_ep64_run(
        run_id=run_id,
        num_steps=num_steps,
        num_layers=num_layers,
        schedule_capacity_factor=schedule_capacity_factor,
        max_learning_rate=max_learning_rate,
    )


if __name__ == "__main__":
    main()
