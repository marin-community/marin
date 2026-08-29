# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fast end-to-end soak for the EP hero's periodic hooks.

The hero host OOM appears after hours of training. Checkpoint, evaluation, watch, and data-loader
hooks run on multi-thousand-step cadences, so a rack run takes a day to produce a few samples.
This run puts the same hooks on a one-step cadence on up to two GB200 trays, so 100 steps exercise
100 checkpoints, tagged evaluations, and dropless evaluations.

The shape is downsized but the memory-relevant machinery is the hero's: MuonH optimizer state
offloaded to pinned host memory, and the ragged all-to-all transport. Those decide what the
checkpoint path has to move through host memory, which is what the soak measures.

Checkpoints go to a one-day temporary prefix, never to the hero's own checkpoint root.

    uv run experiments/grug/moe_hero_ep/memory_soak.py --run-id soak-1 --version dev --run
"""

import dataclasses
import os
from datetime import timedelta

import click
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_GPUS_PER_NODE,
    HERO_MASTER_PARAM_MODE,
    HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_QB_HIST_BINS,
    HeroThroughputResult,
    with_transport_remat_mode,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.small_scale_abl_launch import (
    _EP_CAPACITY_FACTOR,
    SMALL_SHAPES,
    SmallShape,
    _small_model,
)
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    TrainingDataMode,
    WatchMode,
    run_grug,
)

# A narrow shape makes the checkpointed state small enough to finish a soak in minutes.
#
# Replica splitting changes the write path. `plan_array_write` sends an array with replicas down
# its replica-split branch. Each process writes a slice of the shard it holds and stages through a
# transient slice. An array without replicas takes the single-writer branch and stages the training
# shard itself. The hero runs 11 racks, so every one of its arrays has 11 replicas and takes the first
# branch. A soak at expert 4 / replica 1 takes the second, and measures a path the hero never runs.
DEFAULT_EXPERT_AXIS = 1
DEFAULT_DP_REPLICAS = 1
DEFAULT_SIZE = "d384"
# Twelve routed experts divide across the hero's three receiver waves while keeping the model near
# 100M parameters once embeddings and shared layers are included.
DEFAULT_NUM_EXPERTS = 12
DEFAULT_BATCH_SIZE = 1
DEFAULT_STEPS = 100
DEFAULT_EVAL_BATCHES = 2
DEFAULT_SEQ_LEN = 512
MAX_SOAK_TRAYS = 2
# The schedule the hero's optimizer heuristic was tuned against. The soak trains 100 steps of its
# head, so the learning rates match the hero's early steps instead of a 100-step schedule's.
SOAK_SCHEDULE_STEPS = 390_251
CHECKPOINT_TTL_DAYS = 1
# Falsy `timedelta(0)` disables time-policy saves outright. A microsecond is below even a compiled
# no-op step, making every call a real temporary save without retaining 100 permanent checkpoints.
EVERY_STEP = timedelta(microseconds=1)
SOAK_SHAPES = {**SMALL_SHAPES, "d384": SmallShape(384, 4, 3, 1, 1)}


def _synthetic_data_config(*, seq_len: int, vocab_size: int, eval_examples: int) -> LmDataConfig:
    tokens = np.arange(seq_len, dtype=np.int32) % vocab_size
    loss_weight = np.ones(seq_len, dtype=np.float32)
    loss_weight[-1] = 0
    example = GrugLmExample(tokens=tokens, loss_weight=loss_weight)
    validation = ListAsyncDataset([example] * eval_examples)
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        shuffle=False,
        components={
            "synthetic": DirectDatasetComponent(
                datasets={"validation": validation},
                tags=["synthetic"],
            )
        },
        train_weights={},
    )


def build_memory_soak_run(
    *,
    run_id: str,
    size: str = DEFAULT_SIZE,
    num_experts: int = DEFAULT_NUM_EXPERTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    num_steps: int = DEFAULT_STEPS,
    eval_batches: int = DEFAULT_EVAL_BATCHES,
    expert_axis: int = DEFAULT_EXPERT_AXIS,
    dp_replicas: int = DEFAULT_DP_REPLICAS,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build a short synthetic run with checkpoints and evaluators on every step."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if size not in SOAK_SHAPES:
        raise ValueError(f"size must be one of {sorted(SOAK_SHAPES)}, got {size!r}")
    if num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={num_experts} must divide the expert axis {expert_axis}")
    devices = expert_axis * dp_replicas
    if devices > MAX_SOAK_TRAYS * HERO_GPUS_PER_NODE:
        raise ValueError(
            f"expert_axis x dp_replicas = {devices} exceeds {MAX_SOAK_TRAYS} " f"{HERO_GPUS_PER_NODE}-GPU trays"
        )
    if devices > HERO_GPUS_PER_NODE and devices % HERO_GPUS_PER_NODE != 0:
        raise ValueError(f"multi-tray soaks must use whole {HERO_GPUS_PER_NODE}-GPU trays, got {devices} devices")

    model = _small_model(
        shape=SOAK_SHAPES[size],
        capacity_factor=_EP_CAPACITY_FACTOR,
        # FA4's packed-segment helper gives equivalent size-one mesh axes distinct explicit
        # shardings. Reference attention avoids that single-device-only mismatch; attention
        # implementation does not participate in checkpoint host staging. The soak's shapes are
        # all small, where the hero's wide forward tile is unmeasured, so multi-device soaks
        # keep the plain FA4 name like the narrow ladder rungs.
        attention_implementation="reference" if devices == 1 else "gpu_fa4_cute",
        moe_implementation=HERO_MODEL_CONFIG.moe_implementation,
        expert_chunks=HERO_MODEL_CONFIG.expert_chunks,
        seq_len=seq_len,
        num_experts=num_experts,
        num_experts_per_token=HERO_MODEL_CONFIG.num_experts_per_token,
        intermediate_dim=None,
        latent_dim=None,
        pooled_transport_capacity_factor=HERO_MODEL_CONFIG.pooled_transport_capacity_factor,
        num_expert_waves=HERO_MODEL_CONFIG.num_expert_waves,
        qb_use_histogram=True,
        qb_hist_bins=HERO_QB_HIST_BINS,
    )
    model = with_transport_remat_mode(model)
    local_experts = num_experts // expert_axis
    if local_experts % model.num_expert_waves != 0:
        raise ValueError(f"local expert count={local_experts} must divide num_expert_waves={model.num_expert_waves}")
    if devices == 1:
        # MuonH's Newton--Schulz contraction expects a nontrivial model axis. Adam keeps the
        # single-device soak on the same training/checkpoint path without introducing a fake axis.
        optimizer = AdamConfig(learning_rate=1e-3, weight_decay=0.0)
    else:
        optimizer = MoeHeuristic().build_optimizer_config(
            num_train_steps=SOAK_SCHEDULE_STEPS,
            batch_size=batch_size,
            hidden_dim=model.hidden_dim,
            seq_len=seq_len,
        )
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        # Matches the hero: the MuonH state is what the checkpoint reads out of pinned host
        # memory, which is the path under suspicion.
        offload_opt_state=True,
        master_param_mode=HERO_MASTER_PARAM_MODE,
        training_data_mode=TrainingDataMode.SYNTHETIC,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=True,
        expert_axis_size=expert_axis,
        replica_axis_size=dp_replicas,
        sharding_dump_path=None,
    )
    gpus_per_task = min(devices, HERO_GPUS_PER_NODE)
    tasks = (devices + gpus_per_task - 1) // gpus_per_task
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=gpus_per_task,
        cpu=HERO_NODE_CPU if gpus_per_task == HERO_GPUS_PER_NODE else 32,
        ram=HERO_NODE_RAM if gpus_per_task == HERO_GPUS_PER_NODE else "192g",
        disk=HERO_NODE_DISK if gpus_per_task == HERO_GPUS_PER_NODE else "256g",
        replicas=tasks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=SOAK_SCHEDULE_STEPS,
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[HERO_MASTER_PARAM_MODE]),
            tracker=WandbConfig(
                entity="marin-community",
                project=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
                tags=["grug", "moe", "hero", "ep", "memory-soak", "gb200", f"shape-{size}"],
                group="moe-hero-ep-memory-soak",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=1),
            # Size-one axes can carry distinct explicit PartitionSpecs even though they map to
            # the same single device, which makes elementwise operations reject their operands.
            use_explicit_mesh_axes=devices > 1,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                # A disposable one-day prefix, kept away from the hero's checkpoint root. Only the
                # newest temporary checkpoint is retained, so 100 saves cost one checkpoint of space.
                base_path=prefix_join(
                    marin_temp_bucket(
                        ttl_days=CHECKPOINT_TTL_DAYS,
                        prefix=f"memory-soak/{run_id}",
                        source_prefix=ctx.output_path,
                    ),
                    "checkpoints",
                ),
                temporary_base_path=None,
                save_interval=EVERY_STEP,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        data = _synthetic_data_config(
            seq_len=seq_len,
            vocab_size=model.vocab_size,
            eval_examples=eval_batches * devices,
        )
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                eval_batch_size=devices,
                steps_per_eval=1,
                max_eval_batches=eval_batches,
                eval_current=True,
                eval_ema=False,
                dropless_eval=True,
            ),
            stop_after_steps=num_steps,
            processes_per_task=gpus_per_task,
            max_retries_failure=0,
            max_task_failures=0,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for the artifact, job, and W&B names.")
@click.option(
    "--size",
    type=click.Choice(sorted(SOAK_SHAPES)),
    default=DEFAULT_SIZE,
    show_default=True,
    help="Model width. Wider raises the per-process checkpoint share, which is what the save path moves.",
)
@click.option(
    "--seq-len",
    type=click.IntRange(min=128),
    default=DEFAULT_SEQ_LEN,
    show_default=True,
    help="Sequence length. Short sequences keep the soak quick.",
)
@click.option(
    "--num-experts",
    type=click.IntRange(min=1),
    default=DEFAULT_NUM_EXPERTS,
    show_default=True,
    help="Routed experts. Must divide the expert axis, with the per-GPU count divisible by the wave count.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Sequences per step. Small keeps the step short so 100 steps of hooks run quickly.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_STEPS,
    show_default=True,
    help="Steps to run. Each one saves a checkpoint and runs every evaluator.",
)
@click.option(
    "--eval-batches",
    type=click.IntRange(min=1),
    default=DEFAULT_EVAL_BATCHES,
    show_default=True,
    help="Batches per tagged eval set. These exercise evaluator memory.",
)
@click.option(
    "--expert-axis",
    type=click.IntRange(min=1),
    default=DEFAULT_EXPERT_AXIS,
    show_default=True,
    help="Devices in each expert-parallel group.",
)
@click.option(
    "--dp-replicas",
    type=click.IntRange(min=1),
    default=DEFAULT_DP_REPLICAS,
    show_default=True,
    help=(
        "Replica groups. Above 1 every array has replicas and a save takes the replica-split write "
        "path the hero runs; at 1 every array takes the single-writer path instead. "
        "--expert-axis x --dp-replicas may use at most two trays."
    ),
)
@build_options
def main(
    run_id: str,
    size: str,
    num_experts: int,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    eval_batches: int,
    expert_axis: int,
    dp_replicas: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_memory_soak_run(
        run_id=run_id,
        size=size,
        num_experts=num_experts,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
        eval_batches=eval_batches,
        expert_axis=expert_axis,
        dp_replicas=dp_replicas,
    )


if __name__ == "__main__":
    main()
