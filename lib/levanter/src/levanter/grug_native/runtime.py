# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import jmp
from haliax import Axis
from jax.sharding import Mesh

from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.schedule import BatchSchedule


@dataclass(frozen=True)
class GrugRuntime:
    """Concrete runtime surface grug-native depends on."""

    _trainer_config: Any

    @property
    def id(self) -> str | None:
        return self._trainer_config.id

    @property
    def seed(self) -> int:
        return self._trainer_config.seed

    @property
    def num_train_steps(self) -> int:
        return self._trainer_config.num_train_steps

    @property
    def mp(self) -> jmp.Policy:
        return self._trainer_config.mp

    @property
    def watch(self) -> WatchConfig:
        return self._trainer_config.watch

    @property
    def device_mesh(self) -> Mesh:
        return self._trainer_config.device_mesh

    @property
    def batch_schedule(self) -> BatchSchedule:
        return self._trainer_config.batch_schedule

    @property
    def checkpointer(self):
        return self._trainer_config.checkpointer

    @property
    def load_checkpoint_path(self) -> str | None:
        return self._trainer_config.load_checkpoint_path

    @property
    def load_checkpoint(self) -> bool | None:
        return self._trainer_config.load_checkpoint

    @property
    def parameter_axis_mapping(self):
        return self._trainer_config.parameter_axis_mapping

    @property
    def allow_partial_checkpoint(self) -> bool:
        return self._trainer_config.allow_partial_checkpoint

    @property
    def profiler(self) -> ProfilerConfig:
        return self._trainer_config.profiler

    @property
    def log_dir(self) -> pathlib.Path:
        return self._trainer_config.log_dir

    @property
    def allow_nondivisible_batch_size(self) -> bool:
        return self._trainer_config.allow_nondivisible_batch_size

    @property
    def EvalBatch(self) -> Axis:
        return self._trainer_config.EvalBatch

    def initialize(self) -> None:
        self._trainer_config.initialize()

    def use_device_mesh(self) -> AbstractContextManager[None]:
        return self._trainer_config.use_device_mesh()


def as_grug_runtime(runtime: Any) -> GrugRuntime:
    """Normalize runtime-like inputs to a concrete GrugRuntime wrapper."""
    if isinstance(runtime, GrugRuntime):
        return runtime
    return GrugRuntime(runtime)


def default_grug_runtime() -> GrugRuntime:
    """Build the default runtime from TrainerConfig lazily."""
    from levanter.trainer import TrainerConfig

    return as_grug_runtime(TrainerConfig(use_explicit_mesh_axes=True))
