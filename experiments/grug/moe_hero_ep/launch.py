# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the EP64 MoE hero configuration."""

import dataclasses
import os

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
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
from experiments.grug.moe_hero_ep.train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
HERO_PROCESSES_PER_TASK = 1
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


class HeroThroughputResult(Artifact):
    """Metrics-only result of the rack-scale throughput hero run.

    The run intentionally writes no checkpoint; it only mirrors its tracker metrics to the output
    path. This artifact is a plain path ref to those metrics, so the step does not promise a
    checkpoint it never produces.
    """


def build_hero_run(*, run_id: str, num_steps: int, version: str | None = None) -> ArtifactStep[HeroThroughputResult]:
    """Build the one-rack EP64 hero throughput run."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=HERO_EP_BATCH_SIZE)
    if model.moe_implementation is None:
        raise ValueError("the EP hero requires an explicit MoE implementation")
    backend_tag = model.moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=32,
        ram="256g",
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
            profiler=ProfilerConfig(enabled=False),
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
            processes_per_task=HERO_PROCESSES_PER_TASK,
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
@build_options
def main(run_id: str, num_steps: int) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(run_id=run_id, num_steps=num_steps)


if __name__ == "__main__":
    main()
