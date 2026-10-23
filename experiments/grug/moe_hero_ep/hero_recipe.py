# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared model, fleet, and output definitions for EP hero launchers."""

import dataclasses

import jmp
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.grug.sharding import _compact_grug_mesh_shape
from levanter.tracker import TrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL
from experiments.grug.moe_hero_ep.model import OFFLOAD_CARRY_REMAT_MODE, GrugModelConfig, QbEstimator
from experiments.grug.moe_hero_ep.train import (
    RAGGED_MOE_IMPLEMENTATION,
    GrugTrainerConfig,
    MasterParamMode,
    TrainingDataMode,
    WatchMode,
    grug_trainer_mesh_config,
)
from experiments.marin_tokenizer import marin_tokenizer

DEFAULT_WANDB_PROJECT = "marin_moe"
# The hero started at 4K context with 1024 sequences per rack. Every later context keeps those tokens
# per rack per update.
HERO_INITIAL_SEQ_LEN = 4096
HERO_TOKENS_PER_RACK = 1024 * HERO_INITIAL_SEQ_LEN
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
# One rack's devices, which is the expert axis without context parallelism.
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
# The hero's context-parallel degree. Each rack splits into HERO_EP_EXPERT_AXIS_SIZE // HERO_CONTEXT_AXIS_SIZE
# expert shards of HERO_CONTEXT_AXIS_SIZE sequence shards each (see `hero_expert_axis_size`).
HERO_CONTEXT_AXIS_SIZE = 1
# Deal rows to batch shards by attention cost (`GrugTrainerConfig.balance_attention_rows`).
HERO_BALANCE_ATTENTION_ROWS = False
# The hero has no tensor parallelism.
HERO_MODEL_AXIS_SIZE = 1
HERO_PROCESSES_PER_TASK = HERO_GPUS_PER_NODE
HERO_NODE_CPU = 120
HERO_NODE_RAM = "890g"
HERO_NODE_DISK = "1t"
HERO_MIXED_PRECISION = "params=bfloat16,compute=bfloat16,output=bfloat16"
# The hero keeps fp32 weights on device; the diagnostics, soak, and benchmarks follow it.
HERO_MASTER_PARAM_MODE = MasterParamMode.DEVICE
# An fp32 master keeps the device copy in bf16; without one the device weights are the fp32 copy.
HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE = {
    MasterParamMode.FP32_PINNED_HOST: HERO_MIXED_PRECISION,
    MasterParamMode.DEVICE: "params=float32,compute=bfloat16,output=bfloat16",
}
HERO_QB_HIST_BINS = 10_000
# A two-tray loader benchmark found that 1 GB kept 18.6x throughput headroom. The process cache
# stayed near 0.923 GiB, while a 125 GB limit let native RSS increase until the cache was full.
HERO_TENSORSTORE_CACHE_BYTES = 1_000_000_000
HERO_WATCH_INTERVAL = 10
HERO_MODEL_CONFIG = dataclasses.replace(
    HERO_MODEL,
    qb_estimator=QbEstimator.HIST,
    qb_hist_bins=HERO_QB_HIST_BINS,
)


def hero_rack_batch_size(seq_len: int) -> int:
    """Sequences per rack that hold the hero's tokens per rack at ``seq_len``."""
    if HERO_TOKENS_PER_RACK % seq_len:
        raise ValueError(f"seq_len={seq_len} must divide the hero's {HERO_TOKENS_PER_RACK} tokens per rack")
    return HERO_TOKENS_PER_RACK // seq_len


def hero_expert_axis_size(context_axis_size: int) -> int:
    """The expert axis that fills one rack beside ``context_axis_size`` sequence shards."""
    if context_axis_size <= 0 or HERO_EP_EXPERT_AXIS_SIZE % context_axis_size:
        raise ValueError(
            f"context_axis_size={context_axis_size} must divide the {HERO_EP_EXPERT_AXIS_SIZE} devices of a rack"
        )
    return HERO_EP_EXPERT_AXIS_SIZE // context_axis_size


def validated_batch_axis_size(
    *, device_count: int, dp_racks: int, batch_size: int, context_axis_size: int, expert_axis_size: int
) -> int:
    """Validate mesh and batch divisibility; return the replica * data * expert product.

    Context and model axes do not partition the batch.
    """
    replica, data, _context, expert, _model = _compact_grug_mesh_shape(
        process_count=device_count,
        local_device_count=1,
        expert_axis_size=expert_axis_size,
        replica_axis_size=dp_racks,
        model_axis_size=HERO_MODEL_AXIS_SIZE,
        context_axis_size=context_axis_size,
    )
    batch_axes = replica * data * expert
    if batch_size % batch_axes != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by the batch axes replica ({replica}) * "
            f"data ({data}) * expert ({expert}) = {batch_axes}"
        )
    return batch_axes


def validate_model_split(model: GrugModelConfig, *, context_axis_size: int, expert_axis_size: int) -> None:
    """Reject a sequence or expert-bank split that ``model`` cannot take.

    Each would otherwise fail after the rack is allocated and the workspace is built: an indivisible
    sequence in the context-sharded forward, a bank not divisible by the expert axis in `moe_mlp`, and one
    not divisible by the (expert, context) storage split in parameter init.
    """
    if model.max_seq_len % context_axis_size:
        raise ValueError(f"context_axis_size={context_axis_size} must divide max_seq_len={model.max_seq_len}")
    bank_shards = expert_axis_size * context_axis_size
    if model.num_experts % bank_shards != 0:
        raise ValueError(
            f"num_experts={model.num_experts} must be divisible by expert ({expert_axis_size}) * "
            f"context ({context_axis_size}) = {bank_shards}, the expert bank's storage split"
        )
    local_experts = model.num_experts // expert_axis_size
    if local_experts % model.num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={model.num_expert_waves}"
        )


def with_transport_remat_mode(model: GrugModelConfig) -> GrugModelConfig:
    if model.moe_implementation != RAGGED_MOE_IMPLEMENTATION:
        return model
    return dataclasses.replace(model, remat_mode=OFFLOAD_CARRY_REMAT_MODE)


class HeroThroughputResult(Artifact):
    """Metrics artifact for an EP hero run."""


def hero_grug_trainer_config(
    *,
    replica_axis_size: int,
    training_data_mode: TrainingDataMode,
    watch_mode: WatchMode,
    save_checkpoints: bool,
    context_axis_size: int,
    expert_axis_size: int,
    balance_attention_rows: bool,
    master_param_mode: MasterParamMode = HERO_MASTER_PARAM_MODE,
    gc_interval: int | None = None,
) -> GrugTrainerConfig:
    """Set the Grug options that affect the compiled hero step."""
    return GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        gc_interval=gc_interval,
        ema_beta=None,
        z_loss_weight=1e-4,
        # Keep the MuonH state on pinned host memory so the transport buffers have sufficient HBM.
        offload_opt_state=True,
        master_param_mode=master_param_mode,
        training_data_mode=training_data_mode,
        watch_mode=watch_mode,
        save_checkpoints=save_checkpoints,
        expert_axis_size=expert_axis_size,
        replica_axis_size=replica_axis_size,
        context_axis_size=context_axis_size,
        balance_attention_rows=balance_attention_rows,
        sharding_dump_path=None,
    )


def hero_trainer_config(
    *,
    run_id: str,
    seed: int,
    train_batch_size: int,
    num_train_steps: int,
    profiler: ProfilerConfig,
    tracker: TrackerConfig,
    watch: WatchConfig,
    checkpointer: CheckpointerConfig,
    context_axis_size: int,
    progress_watchdog: ProgressWatchdogConfig = ProgressWatchdogConfig(),
    load_checkpoint_path: str | list[str] | None = None,
    load_checkpoint: bool | None = None,
    master_param_mode: MasterParamMode = HERO_MASTER_PARAM_MODE,
) -> TrainerConfig:
    """Set the Levanter options that affect the compiled hero step."""
    return TrainerConfig(
        id=run_id,
        seed=seed,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        profiler=profiler,
        mp=jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[master_param_mode]),
        tracker=tracker,
        watch=watch,
        progress_watchdog=progress_watchdog,
        # Context shards do not hold separate batch rows, so the batch count excludes them. Without context
        # parallelism the default mesh keeps the trainer exactly as it has always been.
        mesh=grug_trainer_mesh_config(context_axis_size) if context_axis_size > 1 else MeshConfig(),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        load_checkpoint_path=load_checkpoint_path,
        load_checkpoint=load_checkpoint,
        checkpointer=checkpointer,
    )


def validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    # Weight-zero datasets appear as tagged evaluation sets.
    return list(paloma_datasets(tokenizer=marin_tokenizer).values())
