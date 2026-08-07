# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accelerator job entry points for June expert calibration and matching."""

from dataclasses import dataclass, field

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig

from experiments.grug.moe.expert_merge import SpectralProbeConfig
from experiments.june_tpu_67b_a2b.dispatch import dispatch_grug_training_run
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig


@dataclass(frozen=True)
class JuneSourceCheckpointConfig:
    model: GrugModelConfig
    checkpoint_dir: str
    source_commit: str | None = None


@dataclass(frozen=True)
class JuneCalibrationJobConfig:
    source: JuneSourceCheckpointConfig
    data: LmDataConfig
    output_path: str
    resources: ResourceConfig
    run_id: str
    layers: tuple[int, int] = (12, 13)
    calibration_tokens: int = 2_000_000
    batch_size: int = 128
    trace_sample_size: int = 131_072
    capacity_per_expert: int = 2_048
    trace_capacity: int = 8_192
    heldout_fraction: float = 0.2
    seed: int = 0
    expert_axis_size: int = 1
    replica_axis_size: int = 1
    model_axis_size: int = 1


@dataclass(frozen=True)
class JuneMatchingJobConfig:
    source: JuneSourceCheckpointConfig
    calibration_path: str
    output_path: str
    resources: ResourceConfig
    run_id: str
    representative_layer: int = 12
    source_layer: int = 13
    probe: SpectralProbeConfig = field(default_factory=SpectralProbeConfig)
    eta: float = 0.5
    expert_chunk_size: int = 16
    seed: int = 0
    expert_axis_size: int = 1
    replica_axis_size: int = 2
    model_axis_size: int = 16


def _run_calibration_local(config: JuneCalibrationJobConfig) -> None:
    from experiments.june_tpu_67b_a2b.moe.merge_job_runtime import run_calibration_local  # noqa: PLC0415

    run_calibration_local(config)


def _run_matching_local(config: JuneMatchingJobConfig) -> None:
    from experiments.june_tpu_67b_a2b.moe.merge_job_runtime import run_matching_local  # noqa: PLC0415

    run_matching_local(config)


def run_calibration(config: JuneCalibrationJobConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_calibration_local,
        resources=config.resources,
    )


def run_matching(config: JuneMatchingJobConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_matching_local,
        resources=config.resources,
    )


__all__ = [
    "JuneCalibrationJobConfig",
    "JuneMatchingJobConfig",
    "JuneSourceCheckpointConfig",
    "run_calibration",
    "run_matching",
]
