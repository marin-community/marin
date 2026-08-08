# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accelerator job entry points for the one-pair expert-merge pipeline."""

from dataclasses import dataclass, field

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.expert_merge import AssignmentMode, SpectralProbeConfig
from experiments.grug.moe.expert_prefit import PrefitConfig, PrefitObjective
from experiments.grug.moe.merge_recovery import (
    RecoveryCheckpointSelection,
    RecoveryInitialization,
    RecoveryStage,
    RecoveryTrainableScope,
)
from experiments.grug.moe.model import GrugModelConfig


@dataclass(frozen=True)
class SourceCheckpointConfig:
    model: GrugModelConfig
    optimizer: OptimizerConfig
    training_steps: int
    checkpoint_dir: str
    source_commit: str | None = None


@dataclass(frozen=True)
class CalibrationJobConfig:
    source: SourceCheckpointConfig
    data: LmDataConfig
    output_path: str
    resources: ResourceConfig
    run_id: str
    layers: tuple[int, int] = (2, 3)
    calibration_tokens: int = 2_000_000
    batch_size: int = 32
    capacity_per_expert: int = 2_048
    trace_capacity: int = 8_192
    heldout_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class MatchingJobConfig:
    source: SourceCheckpointConfig
    calibration_path: str
    output_path: str
    resources: ResourceConfig
    run_id: str
    representative_layer: int = 2
    source_layer: int = 3
    probe: SpectralProbeConfig = field(default_factory=SpectralProbeConfig)
    eta: float = 0.5
    expert_chunk_size: int = 16
    seed: int = 0


@dataclass(frozen=True)
class PrefitJobConfig:
    source: SourceCheckpointConfig
    calibration_path: str
    matching_path: str
    output_path: str
    resources: ResourceConfig
    run_id: str
    assignment_mode: AssignmentMode
    objective: PrefitObjective
    representative_layer: int = 2
    source_layer: int = 3
    probe: SpectralProbeConfig = field(default_factory=SpectralProbeConfig)
    config: PrefitConfig = field(default_factory=PrefitConfig)
    seed: int = 0
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


@dataclass(frozen=True)
class ConversionJobConfig:
    source: SourceCheckpointConfig
    calibration_path: str
    matching_path: str
    prefit_path: str | None
    output_path: str
    resources: ResourceConfig
    run_id: str
    assignment_mode: AssignmentMode
    representative_layer: int = 2
    source_layer: int = 3
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


@dataclass(frozen=True)
class CapacityOracleSplitJobConfig:
    source: SourceCheckpointConfig
    init_checkpoint_dir: str
    output_path: str
    resources: ResourceConfig
    run_id: str
    assignment_mode: AssignmentMode
    prefit_applied: bool
    affected_layers: tuple[int, int] = (2, 3)
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


@dataclass(frozen=True)
class LayerAdapterAugmentJobConfig:
    """Add a zero-output layer-conditioned adapter to a selected tied checkpoint."""

    source: SourceCheckpointConfig
    init_checkpoint_dir: str
    output_path: str
    resources: ResourceConfig
    run_id: str
    assignment_mode: AssignmentMode
    prefit_applied: bool
    adapter_rank: int
    affected_layers: tuple[int, int] = (2, 3)
    seed: int = 0
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


@dataclass(frozen=True)
class GradientConflictArtifactReference:
    """Exact immutable artifact record required by the diagnostic."""

    name: str
    version: str
    root: str
    fingerprint: str


@dataclass(frozen=True)
class GradientConflictCheckpointConfig:
    """One exact frozen checkpoint in the preregistered diagnostic trajectory."""

    label: str
    artifact: GradientConflictArtifactReference
    checkpoint_path: str
    expected_step: int
    continuation_tokens: int


@dataclass(frozen=True)
class GradientConflictJobConfig:
    """Read-only direct shared-bank gradient diagnostic over frozen checkpoints."""

    source: SourceCheckpointConfig
    teacher_artifact: GradientConflictArtifactReference
    data: LmDataConfig
    checkpoints: tuple[GradientConflictCheckpointConfig, ...]
    output_path: str
    resources: ResourceConfig
    run_id: str
    affected_layers: tuple[int, int] = (2, 3)
    batch_size: int = 32
    num_batches: int = 16
    loader_start_step: int = 382
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 8_032
    seed: int = 0
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


@dataclass(frozen=True)
class RecoveryJobConfig:
    source: SourceCheckpointConfig
    data: LmDataConfig
    matching_path: str
    init_checkpoint_dir: str
    output_path: str
    resources: ResourceConfig
    run_id: str
    stage: RecoveryStage
    trainable_scope: RecoveryTrainableScope
    initialization: RecoveryInitialization
    assignment_mode: AssignmentMode
    prefit_applied: bool
    training_tokens: int
    cross_entropy_weight: float
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    moe_loss_weight: float = 1.0
    logit_kl_weight: float = 0.0
    logit_kl_vocab_chunk_size: int = 1_024
    affected_layers: tuple[int, int] = (2, 3)
    seed: int = 0
    checkpoint_every: int = 100
    checkpoint_token_milestones: tuple[int, ...] = (25_000_000, 100_000_000, 200_000_000)
    recovery_loss_threshold_delta: float = 0.02
    select_best_validation_checkpoint: bool = False
    initial_checkpoint_selection: RecoveryCheckpointSelection = RecoveryCheckpointSelection.LATEST
    layer_adapter_rank: int | None = None
    layer_adapter_source_checkpoint_dir: str | None = None
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


def _run_calibration_local(config: CalibrationJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_job_runtime import run_calibration_local  # noqa: PLC0415

    run_calibration_local(config)


def _run_matching_local(config: MatchingJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_job_runtime import run_matching_local  # noqa: PLC0415

    run_matching_local(config)


def _run_prefit_local(config: PrefitJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_recovery_runtime import run_prefit_local  # noqa: PLC0415

    run_prefit_local(config)


def _run_conversion_local(config: ConversionJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_recovery_runtime import run_conversion_local  # noqa: PLC0415

    run_conversion_local(config)


def _run_capacity_oracle_split_local(config: CapacityOracleSplitJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_recovery_runtime import run_capacity_oracle_split_local  # noqa: PLC0415

    run_capacity_oracle_split_local(config)


def _run_layer_adapter_augment_local(config: LayerAdapterAugmentJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_recovery_runtime import run_layer_adapter_augment_local  # noqa: PLC0415

    run_layer_adapter_augment_local(config)


def _run_gradient_conflict_local(config: GradientConflictJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_recovery_runtime import run_gradient_conflict_local  # noqa: PLC0415

    run_gradient_conflict_local(config)


def _run_recovery_local(config: RecoveryJobConfig) -> None:
    # Break the config/runtime import cycle at the dispatched worker boundary.
    from experiments.grug.moe.merge_recovery_runtime import run_recovery_local  # noqa: PLC0415

    run_recovery_local(config)


def _dispatch(run_id: str, config, entrypoint, resources: ResourceConfig) -> None:
    dispatch_grug_training_run(
        run_id=run_id,
        config=config,
        local_entrypoint=entrypoint,
        resources=resources,
    )


def run_calibration(config: CalibrationJobConfig) -> None:
    _dispatch(config.run_id, config, _run_calibration_local, config.resources)


def run_matching(config: MatchingJobConfig) -> None:
    _dispatch(config.run_id, config, _run_matching_local, config.resources)


def run_prefit(config: PrefitJobConfig) -> None:
    _dispatch(config.run_id, config, _run_prefit_local, config.resources)


def run_conversion(config: ConversionJobConfig) -> None:
    _dispatch(config.run_id, config, _run_conversion_local, config.resources)


def run_capacity_oracle_split(config: CapacityOracleSplitJobConfig) -> None:
    _dispatch(config.run_id, config, _run_capacity_oracle_split_local, config.resources)


def run_layer_adapter_augment(config: LayerAdapterAugmentJobConfig) -> None:
    _dispatch(config.run_id, config, _run_layer_adapter_augment_local, config.resources)


def run_gradient_conflict(config: GradientConflictJobConfig) -> None:
    _dispatch(config.run_id, config, _run_gradient_conflict_local, config.resources)


def run_recovery(config: RecoveryJobConfig) -> None:
    _dispatch(config.run_id, config, _run_recovery_local, config.resources)
