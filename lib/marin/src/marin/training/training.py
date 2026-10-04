# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import Any, TypeVar

from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.execution.artifact import Artifact

# The subdirectory Levanter writes rolling checkpoints into, relative to a run's output dir.
_CHECKPOINTS_SUBDIR = "checkpoints"
# The final-metrics file a run mirrors next to its output via the WandB ``replicate_path``.
_TRACKER_METRICS_FILE = "tracker_metrics.jsonl"


class TrainMetrics(BaseModel):
    """A finished run's final metrics, read on demand from its output.

    ``summary`` is the full mirrored WandB summary (keys like ``train/loss``, ``eval/loss``,
    ``eval/<name>/loss``); ``train_loss``/``eval_loss`` are the common scalars pulled out, each
    ``None`` when the run did not log it.
    """

    summary: dict[str, Any]
    train_loss: float | None = None
    eval_loss: float | None = None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


class LevanterCheckpoint(Artifact):
    """A Levanter training run's output: rolling checkpoints, config, and mirrored metrics.

    The realized artifact for every :func:`~marin.experiment.train.train_lm` handle. ``load`` is
    a path ref; the checkpoint structure and metrics are exposed as accessors so callers never
    hard-code the layout.
    """

    @property
    def checkpoint_dir(self) -> str:
        """The directory holding this run's rolling checkpoints."""
        return prefix_join(self.path, _CHECKPOINTS_SUBDIR)

    def training_metrics(self) -> TrainMetrics:
        """This run's final metrics, parsed from ``tracker_metrics.jsonl`` under its output.

        Raises :class:`FileNotFoundError` if the run wrote no metrics file.
        """
        path = prefix_join(self.path, _TRACKER_METRICS_FILE)
        if not url_to_fs(path, use_listings_cache=False)[0].exists(path):
            raise FileNotFoundError(f"no {_TRACKER_METRICS_FILE} for checkpoint at {self.path}")
        lines = [line for line in StoragePath(path).read_text().splitlines() if line.strip()]
        if not lines:
            raise FileNotFoundError(f"empty {_TRACKER_METRICS_FILE} at {path}")
        summary = json.loads(lines[-1]).get("summary", {})
        return TrainMetrics(
            summary=summary,
            train_loss=_as_float(summary.get("train/loss")),
            eval_loss=_as_float(summary.get("eval/loss")),
        )


@dataclass(frozen=True)
class TrainLmOnPodConfig:
    """Configuration for language model training on a pod."""

    train_config: object
    resources: ResourceConfig
    output_path: str | None = None
    """Base output directory for the run. The checkpointer, HF export, and run id derive from it."""
    env_vars: dict[str, str] | None = None
    """Environment variables to pass to the training task (e.g., WANDB_MODE, WANDB_API_KEY)."""
    auto_build_caches: bool = False
    """Whether to allow Levanter to build dataset caches on the fly.

    Defaults to False so Marin jobs fail fast when a cache is missing instead of
    spending time (and money) building it during training. Override to True if
    you explicitly want cache construction.
    """


@dataclass(frozen=True)
class TrainDpoOnPodConfig:
    """Configuration for DPO training on a pod."""

    train_config: object
    resources: ResourceConfig
    output_path: str | None = None
    """Base output directory for the run. The checkpointer, HF export, and run id derive from it."""
    env_vars: dict[str, str] | None = None
    """Environment variables to pass to the training task (e.g., WANDB_MODE, WANDB_API_KEY)."""
    auto_build_caches: bool = False
    """Whether to allow Levanter to build dataset caches on the fly.

    Defaults to False so Marin jobs fail fast when a cache is missing instead of
    spending time (and money) building it during training. Override to True if
    you explicitly want cache construction.
    """
    auto_num_epochs: float | None = None
    """When set, resolve num_train_steps from the concrete DPO train cache at launch time."""
    auto_validation_runs: int | None = None
    """When set, schedule this many validation passes including the initial and final evaluations."""


TrainConfigT = TypeVar("TrainConfigT")
TrainOnPodConfigT = TypeVar("TrainOnPodConfigT", TrainLmOnPodConfig, TrainDpoOnPodConfig)

DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_HF_CHECKPOINTS_PATH = "hf"
TEMPORARY_CHECKPOINT_TTL_DAYS = 14
TEMPORARY_CHECKPOINTS_PATH = "checkpoints-temp"


def _temporary_checkpoint_key(output_path: str) -> str:
    path = StoragePath(output_path)
    return str(StoragePath(path.bucket) / path.key) if path.bucket else path.key


def temporary_checkpoint_base_path(output_path: str, ttl_days: int = TEMPORARY_CHECKPOINT_TTL_DAYS) -> str:
    """Return the region-local temporary checkpoint base for an executor output path.

    Objects under the base expire ``ttl_days`` after they are written. A running job's newest temporary
    checkpoint is at most one save interval old, so the TTL mainly bounds how long checkpoints left
    behind by finished or replaced runs occupy storage.
    """
    temporary_root = marin_temp_bucket(
        ttl_days=ttl_days,
        source_prefix=output_path,
    )
    return str(
        StoragePath(temporary_root)
        / TEMPORARY_CHECKPOINTS_PATH
        / _temporary_checkpoint_key(output_path)
        / DEFAULT_CHECKPOINTS_PATH
    )


# XLA disables the NCCL collective watchdog by default (-1). A positive value
# makes XLA abort the communicators and raise TimeoutError once a collective is
# stuck that long, so a hung collective becomes a process crash Iris can retry
# instead of a silent mesh-wide hang.
GPU_NCCL_TERMINATION_TIMEOUT_FLAG = "xla_gpu_nccl_termination_timeout_seconds"
# Counts only time spent inside a collective, so it does not fire during compile
# or data loading. Set above the longest legitimate collective (cross-rank skew on
# a cold start, checkpoint/eval barriers). Override per run via XLA_FLAGS.
DEFAULT_GPU_NCCL_TERMINATION_TIMEOUT = 600
TENSORSTORE_CURL_LOW_SPEED_TIME_ENV = "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS"
TENSORSTORE_CURL_LOW_SPEED_LIMIT_ENV = "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES"
DEFAULT_TENSORSTORE_CURL_LOW_SPEED_TIME = "60"
# TensorStore passes this decimal bytes-per-second value directly to libcurl: 1 Mbit/s.
DEFAULT_TENSORSTORE_CURL_LOW_SPEED_LIMIT = "125000"
