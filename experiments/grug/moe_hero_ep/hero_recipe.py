# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared model, fleet, and output definitions for EP hero launchers."""

import dataclasses

import jmp
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker import TrackerConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL
from experiments.grug.moe_hero_ep.model import QbEstimator
from experiments.grug.moe_hero_ep.train import (
    GrugTrainerConfig,
    MasterParamMode,
    TrainingDataMode,
    WatchMode,
)
from experiments.marin_tokenizer import marin_tokenizer

DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
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


class HeroThroughputResult(Artifact):
    """Metrics artifact for an EP hero run."""


def hero_grug_trainer_config(
    *,
    replica_axis_size: int,
    training_data_mode: TrainingDataMode,
    watch_mode: WatchMode,
    save_checkpoints: bool,
    master_param_mode: MasterParamMode = HERO_MASTER_PARAM_MODE,
) -> GrugTrainerConfig:
    """Set the Grug options that affect the compiled hero step."""
    return GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        # Keep the MuonH state on pinned host memory so the transport buffers have sufficient HBM.
        offload_opt_state=True,
        master_param_mode=master_param_mode,
        training_data_mode=training_data_mode,
        watch_mode=watch_mode,
        save_checkpoints=save_checkpoints,
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=replica_axis_size,
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
    progress_watchdog: ProgressWatchdogConfig = ProgressWatchdogConfig(),
    load_checkpoint_path: str | list[str] | None = None,
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
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        load_checkpoint_path=load_checkpoint_path,
        checkpointer=checkpointer,
    )


def validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    # Weight-zero datasets appear as tagged evaluation sets.
    return list(paloma_datasets(tokenizer=marin_tokenizer).values())
