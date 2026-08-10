# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the Mixture of Kittens experiment."""

import dataclasses
import os
from enum import StrEnum

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.kernels.mixture_of_kittens.forward_ffi import MoKForwardConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.grug.mixture_of_kittens.heuristic import build_mok_configs
from experiments.grug.mixture_of_kittens.train import (
    MOK_JAX_PACKAGES,
    GrugRunConfig,
    GrugTrainerConfig,
    RaggedAllToAllImplementation,
    run_grug,
)
from experiments.llama import llama3_tokenizer

DEFAULT_MOK_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
MOK_BATCH_SIZE_PER_GPU = 16
MOK_GPUS_PER_NODE = 4
MAX_MOK_NODES = 16
MOK_PROCESSES_PER_TASK = 1
MOK_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
MOK_FUSED_MACROBATCH_SIZE = 32768
# The model keeps its MuonH state on pinned host memory. This leaves room for both ragged
# all-to-all arms under the same allocation.
MOK_OFFLOAD_OPT_STATE = True

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


class MokExecution(StrEnum):
    """Forward execution boundary for the throughput gate."""

    XLA = "xla"
    FUSED = "fused"


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


class MokThroughputResult(Artifact):
    """Metrics-only result of the ragged all-to-all comparison.

    The run intentionally writes no checkpoint; it only mirrors its tracker metrics to the output
    path. This artifact is a plain path ref to those metrics, so the step does not promise a
    checkpoint it never produces.
    """


def build_mok_run(
    *,
    run_id: str,
    num_steps: int,
    execution: MokExecution,
    implementation: RaggedAllToAllImplementation,
    num_nodes: int,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    num_layers: int | None = None,
    capacity_factor: float | None = None,
    watch_interval: int = 0,
    version: str | None = None,
) -> ArtifactStep[MokThroughputResult]:
    """Build one arm of the XLA ragged all-to-all comparison.

    The overrides keep the hidden dimension fixed. One-shot and device arms
    must use identical overrides and step counts.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if watch_interval < 0:
        raise ValueError(f"watch_interval must be non-negative, got {watch_interval}")
    if not 1 <= num_nodes <= MAX_MOK_NODES:
        raise ValueError(f"num_nodes must be between 1 and {MAX_MOK_NODES}, got {num_nodes}")
    if execution is MokExecution.FUSED and num_nodes != 1:
        raise ValueError("Fused Mixture-of-Kittens execution requires one four-GPU worker")

    expert_axis_size = num_nodes * MOK_GPUS_PER_NODE
    train_batch_size = MOK_BATCH_SIZE_PER_GPU * expert_axis_size
    model, optimizer = build_mok_configs(num_train_steps=num_steps, batch_size=train_batch_size)
    model = dataclasses.replace(model, num_experts=2 * expert_axis_size)
    overrides = {
        name: value
        for name, value in (
            ("num_experts", num_experts),
            ("num_experts_per_token", num_experts_per_token),
            ("intermediate_dim", intermediate_dim),
            ("num_layers", num_layers),
            ("capacity_factor", capacity_factor),
        )
        if value is not None
    }
    if overrides:
        model = dataclasses.replace(model, **overrides)
    if execution is MokExecution.FUSED:
        model = dataclasses.replace(
            model,
            mixture_of_kittens=MoKForwardConfig(macrobatch_size=MOK_FUSED_MACROBATCH_SIZE),
            remat_mode="save_moe",
        )
    # A bank that does not divide the expert axis fails inside `moe_mlp`, which is after the rack is
    # already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % expert_axis_size != 0:
        raise ValueError(f"expert axis {expert_axis_size} must divide num_experts={model.num_experts}")
    if model.moe_implementation is None:
        raise ValueError("the experiment requires an explicit MoE implementation")
    backend_tag = model.moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    size_tag = f"e{model.num_experts}-i{model.intermediate_dim}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=MOK_OFFLOAD_OPT_STATE,
        ragged_all_to_all_implementation=implementation,
        expert_axis_size=expert_axis_size,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=MOK_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=num_nodes,
    )
    name = f"grug/mixture-of-kittens/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=train_batch_size,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(
                enabled=True,
                start_step=5,
                num_steps=5,
                profile_options=ProfileOptionsConfig(
                    host_tracer_level=1,
                    python_tracer_level=0,
                    enable_hlo_proto=True,
                ),
            ),
            mp=jmp.get_policy(MOK_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "ep",
                    backend_tag,
                    f"xla-{implementation.value}",
                    f"execution-{execution.value}",
                    capacity_tag,
                    size_tag,
                    f"batch-{train_batch_size}",
                    f"nodes-{num_nodes}",
                    "gb200",
                    "MOK-JAX",
                ],
                group="mixture-of-kittens",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(
                watch_targets=["grads", "updates", "params"],
                interval=watch_interval,
            ),
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
            processes_per_task=MOK_PROCESSES_PER_TASK,
            pip_packages=MOK_JAX_PACKAGES,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=MokThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--execution",
    type=click.Choice(MokExecution, case_sensitive=False),
    required=True,
    help="Use the XLA forward or the fused Mixture-of-Kittens forward.",
)
@click.option(
    "--implementation",
    type=click.Choice(RaggedAllToAllImplementation, case_sensitive=False),
    required=True,
    help="XLA implementation for both ragged all-to-all operations.",
)
@click.option(
    "--num-nodes",
    type=click.IntRange(min=1, max=MAX_MOK_NODES),
    required=True,
    help="Number of four-GPU GB200 workers.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_MOK_STEPS,
    show_default=True,
    help="Number of training steps.",
)
@click.option(
    "--num-experts",
    type=click.IntRange(min=1),
    default=None,
    help="Override the routed expert count. The worker GPU count must divide it.",
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
    "--num-layers",
    type=click.IntRange(min=1),
    default=None,
    help="Override the transformer layer count for a reduced-shape gate.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the ragged all-to-all receiver capacity factor.",
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Log per-parameter gradient, update, and parameter norms at this step interval. Zero disables logging.",
)
@build_options
def main(
    run_id: str,
    execution: MokExecution,
    implementation: RaggedAllToAllImplementation,
    num_nodes: int,
    num_steps: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    num_layers: int | None,
    capacity_factor: float | None,
    watch_interval: int,
) -> ArtifactStep[MokThroughputResult]:
    return build_mok_run(
        run_id=run_id,
        num_steps=num_steps,
        execution=execution,
        implementation=implementation,
        num_nodes=num_nodes,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        capacity_factor=capacity_factor,
        watch_interval=watch_interval,
    )


if __name__ == "__main__":
    main()
