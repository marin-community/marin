# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Next-generation domain/phase mixture optimization loop."""

from experiments.domain_phase_mix.nextgen.contracts import (
    Candidate,
    ImportSource,
    LoopConfig,
    LoopState,
    PlannedRun,
    PolicyArtifactRef,
    RunRecord,
    TrajectoryPoint,
    ValidationRecord,
)
from experiments.domain_phase_mix.nextgen.model_registry import (
    available_model_names,
    register_model_adapter,
    register_policy_artifact_model,
)
from experiments.domain_phase_mix.nextgen.validation import (
    register_validation_execution_adapter,
)

__all__ = [
    "Candidate",
    "ImportSource",
    "LoopConfig",
    "LoopState",
    "PlannedRun",
    "PolicyArtifactRef",
    "RunRecord",
    "TrajectoryPoint",
    "ValidationRecord",
    "available_model_names",
    "register_model_adapter",
    "register_policy_artifact_model",
    "register_validation_execution_adapter",
]
