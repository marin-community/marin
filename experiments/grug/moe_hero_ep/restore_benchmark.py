# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time a hero-shaped checkpoint save and restore on one or more replica groups.

Builds a hero or hero-shaped small train state, writes it to a one-day temporary prefix, and
reads it back into the same exemplar a resume restores into. Offloaded optimizer state is
included, and the master-parameter mode follows the checkpoint. It trains nothing, so a run
measures the checkpoint paths alone.

The save timing is what a training step is blocked for. The first read is the only one that
can miss the node-local cache.

Dispatch needs an Iris task to submit from, so launch it as a coordinator job:

    iris --config lib/iris/config/marin.yaml job run --no-wait \\
        --enable-extra-resources --target-cluster cw-us-east-08a --priority production \\
        --cpu 2 --memory 8GB --disk 32GB --job-name restore-bench-coord \\
        -- python -m experiments.grug.moe_hero_ep.restore_benchmark --run-id restore-bench-1
"""

import gc
import logging
import time
from dataclasses import dataclass, field

import click
import equinox
import jax
import jax.experimental.array_serialization.serialization as array_ser
import jmp
from fray.cluster import ResourceConfig
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import OptimizerConfig
from levanter.tensorstore_serialization import (
    ReplicaRestoreMode,
    TensorStoreReadConfig,
    TensorStoreWriteConfig,
    tree_deserialize_leaves_tensorstore,
)
from levanter.tracker.telemetry import TelemetryConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.hero_recipe import (
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_QB_HIST_BINS,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic, build_hero_configs
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.small_scale_abl_launch import SMALL_SHAPES, _small_model
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    initial_state,
    restore_template_from,
)

logger = logging.getLogger(__name__)

# Matches the hero. The benchmark writes the checkpoint it reads, so nothing infers this.
BENCHMARK_MASTER_PARAM_MODE = MasterParamMode.DISABLED

# The hero's own schedule length, so this pytree matches the one a hero resume restores into.
HERO_SCHEDULE_STEPS = 390_251
CHECKPOINT_TTL_DAYS = 1
GIB = 1024**3
HERO_MODEL_SIZE = "hero"


@dataclass(frozen=True)
class RestoreBenchmarkConfig:
    """What to restore, into what shape, and how many times."""

    checkpoint_path: str
    model: GrugModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    read: TensorStoreReadConfig = field(default_factory=TensorStoreReadConfig)
    write: TensorStoreWriteConfig = field(default_factory=TensorStoreWriteConfig)
    repeats: int = 2
    expert_axis_size: int = HERO_EP_EXPERT_AXIS_SIZE
    replica_axis_size: int = 1


def _benchmark_state(config: RestoreBenchmarkConfig, mesh):
    """Build the configured train state with the hero's checkpoint offloading."""
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    @jax.jit
    def build(key):
        return initial_state(
            config.model,
            optimizer=optimizer,
            mp=config.trainer.mp,
            key=key,
            ema_beta=None,
            offload_opt_state=True,
            master_param_mode=BENCHMARK_MASTER_PARAM_MODE,
        )

    with set_mesh(mesh):
        return build(jax.random.PRNGKey(config.trainer.seed))


def _time_one_restore(config: RestoreBenchmarkConfig, template, mesh, attempt: int) -> None:
    """Restore the checkpoint once and log the fleet-wide elapsed seconds."""
    serializable, _ = equinox.partition(template, is_jax_array_like)
    multihost_utils.sync_global_devices(f"restore-start-{attempt}")
    started = time.time()
    restored = tree_deserialize_leaves_tensorstore(
        config.checkpoint_path, serializable, mesh=mesh, read_config=config.read
    )
    jax.block_until_ready(restored)
    multihost_utils.sync_global_devices(f"restore-done-{attempt}")
    elapsed = time.time() - started

    jax.tree.map(lambda leaf: leaf.delete() if isinstance(leaf, jax.Array) else None, restored)
    del restored
    gc.collect()
    logger.info(
        "RESULT attempt=%d seconds=%.1f budget=%.0fGiB requests=%d replica_mode=%s",
        attempt,
        elapsed,
        config.read.max_in_flight_bytes / GIB,
        config.read.request_concurrency,
        config.read.replica_mode,
    )


def _run_restore_benchmark_local(config: RestoreBenchmarkConfig) -> None:
    """Entry point that runs on every rank of the benchmark job."""
    config.trainer.initialize()
    mesh = compact_grug_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    state = _benchmark_state(config, mesh)
    with set_mesh(mesh):
        # A hero's rolling temporary checkpoint is pruned as soon as the next one commits, and
        # a read racing that deletion fails on a zero-length OCDBT shard, so own the write.
        # Handed no manager, `save_checkpoint` joins the commits itself; a training loop
        # returns once its data is copied out. Only `blocked` is time a training step loses.
        manager = array_ser.GlobalAsyncCheckpointManager()
        multihost_utils.sync_global_devices("save-start")
        started = time.time()
        save_checkpoint(
            state,
            step=0,
            checkpoint_path=config.checkpoint_path,
            manager=manager,
            is_temporary=True,
            write_config=config.write,
        )
        blocked = time.time() - started
        manager.wait_until_finished()
        committed = time.time() - started
        multihost_utils.sync_global_devices("save-done")
        logger.info(
            "RESULT save blocked=%.1fs committed=%.1fs stage_budget=%.0fGiB",
            blocked,
            committed,
            config.write.max_staged_host_bytes / GIB,
        )
        template = restore_template_from(state)
        for attempt in range(config.repeats):
            _time_one_restore(config, template, mesh, attempt)


@click.command()
@click.option("--run-id", required=True, help="Run identifier for the benchmark job name.")
@click.option(
    "--replica-groups",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Independent replica groups. The hero uses one 64-GPU rack per group.",
)
@click.option(
    "--model-size",
    type=click.Choice([HERO_MODEL_SIZE, *sorted(SMALL_SHAPES)]),
    default=HERO_MODEL_SIZE,
    show_default=True,
    help="Hero model or a hero-shaped small-scale ablation model.",
)
@click.option(
    "--expert-axis-size",
    type=click.IntRange(min=1),
    default=HERO_EP_EXPERT_AXIS_SIZE,
    show_default=True,
    help="Devices in each expert-parallel replica group.",
)
@click.option("--repeats", type=click.IntRange(min=1), default=2, show_default=True, help="Restores to time.")
@click.option(
    "--budget-gib",
    type=click.IntRange(min=1),
    default=TensorStoreReadConfig.max_in_flight_bytes // GIB,
    show_default=True,
    help="Transient staging memory per process. 48 GiB OOM-kills a GB200 node.",
)
@click.option(
    "--stage-gib",
    type=click.IntRange(min=1),
    default=TensorStoreWriteConfig.max_staged_host_bytes // GIB,
    show_default=True,
    help="Host memory a process may hold in staged snapshots. A save blocks training until its "
    "whole share has been admitted, so a share above this waits on commits.",
)
@click.option(
    "--requests",
    type=click.IntRange(min=1),
    default=TensorStoreReadConfig.request_concurrency,
    show_default=True,
    help="Concurrent object-store requests.",
)
@click.option(
    "--replica-mode",
    type=click.Choice([mode.value for mode in ReplicaRestoreMode]),
    default=ReplicaRestoreMode.ONE_REPLICA.value,
    show_default=True,
    help="Read each shard on every replica, or read it once and distribute it with a collective.",
)
def main(
    run_id: str,
    replica_groups: int,
    model_size: str,
    expert_axis_size: int,
    repeats: int,
    budget_gib: int,
    stage_gib: int,
    requests: int,
    replica_mode: str,
) -> None:
    batch_size = HERO_EP_BATCH_SIZE * replica_groups
    if model_size == HERO_MODEL_SIZE:
        model = HERO_MODEL_CONFIG
        _, optimizer = build_hero_configs(num_train_steps=HERO_SCHEDULE_STEPS, batch_size=batch_size)
    else:
        shape = SMALL_SHAPES[model_size]
        model = _small_model(
            shape=shape,
            capacity_factor=HERO_MODEL_CONFIG.capacity_factor,
            attention_implementation=HERO_MODEL_CONFIG.attention_implementation,
            moe_implementation=HERO_MODEL_CONFIG.moe_implementation,
            expert_chunks=HERO_MODEL_CONFIG.expert_chunks,
            seq_len=HERO_MODEL_CONFIG.max_seq_len,
            num_experts=HERO_MODEL_CONFIG.num_experts,
            num_experts_per_token=HERO_MODEL_CONFIG.num_experts_per_token,
            intermediate_dim=None,
            latent_dim=None,
            pooled_transport_capacity_factor=HERO_MODEL_CONFIG.pooled_transport_capacity_factor,
            num_expert_waves=HERO_MODEL_CONFIG.num_expert_waves,
            qb_use_histogram=True,
            qb_hist_bins=HERO_QB_HIST_BINS,
        )
        optimizer = MoeHeuristic().build_optimizer_config(
            num_train_steps=HERO_SCHEDULE_STEPS,
            batch_size=batch_size,
            hidden_dim=model.hidden_dim,
            seq_len=model.max_seq_len,
        )

    if model.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by expert_axis_size={expert_axis_size}")
    local_experts = model.num_experts // expert_axis_size
    if local_experts % model.num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={model.num_expert_waves}"
        )
    device_count = expert_axis_size * replica_groups
    if device_count % HERO_GPUS_PER_NODE != 0:
        raise ValueError(
            f"expert_axis_size * replica_groups must be divisible by {HERO_GPUS_PER_NODE} GPUs per node, "
            f"got {device_count}"
        )

    config = RestoreBenchmarkConfig(
        checkpoint_path=prefix_join(
            marin_temp_bucket(ttl_days=CHECKPOINT_TTL_DAYS, prefix=f"restore-benchmark/{run_id}"), "checkpoint"
        ),
        model=model,
        optimizer=optimizer,
        trainer=TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=HERO_SCHEDULE_STEPS,
            mp=jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[BENCHMARK_MASTER_PARAM_MODE]),
            tracker=TelemetryConfig(),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
        ),
        read=TensorStoreReadConfig(
            max_in_flight_bytes=budget_gib * GIB,
            request_concurrency=requests,
            replica_mode=ReplicaRestoreMode(replica_mode),
        ),
        write=TensorStoreWriteConfig(max_staged_host_bytes=stage_gib * GIB),
        repeats=repeats,
        expert_axis_size=expert_axis_size,
        replica_axis_size=replica_groups,
    )

    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation=model.moe_implementation,
        processes_per_task=HERO_GPUS_PER_NODE,
    )
    dispatch_grug_training_run(
        run_id=run_id,
        config=config,
        local_entrypoint=_run_restore_benchmark_local,
        resources=ResourceConfig.with_gpu(
            "GB200",
            count=HERO_GPUS_PER_NODE,
            cpu=HERO_NODE_CPU,
            ram=HERO_NODE_RAM,
            disk=HERO_NODE_DISK,
            replicas=device_count // HERO_GPUS_PER_NODE,
        ),
        processes_per_task=HERO_GPUS_PER_NODE,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
