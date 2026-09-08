# Copyright The Marin Authors
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


def available_model_names():
    """Lazily expose registered model names."""
    from experiments.domain_phase_mix.nextgen.model_registry import available_model_names as _available_model_names

    return _available_model_names()


def register_model_adapter(adapter, *, overwrite: bool = False) -> None:
    """Lazily register a model adapter."""
    from experiments.domain_phase_mix.nextgen.model_registry import register_model_adapter as _register_model_adapter

    _register_model_adapter(adapter, overwrite=overwrite)


def register_policy_artifact_model(
    *,
    model_name: str,
    policy_uri: str,
    policy_format: str = "json",
    predicted_objective: float | None = None,
    overwrite: bool = False,
) -> None:
    """Lazily register a policy artifact adapter."""
    from experiments.domain_phase_mix.nextgen.model_registry import (
        register_policy_artifact_model as _register_policy_artifact_model,
    )

    _register_policy_artifact_model(
        model_name=model_name,
        policy_uri=policy_uri,
        policy_format=policy_format,
        predicted_objective=predicted_objective,
        overwrite=overwrite,
    )


def register_validation_execution_adapter(adapter) -> None:
    """Lazily register the validation execution adapter."""
    from experiments.domain_phase_mix.nextgen.validation import (
        register_validation_execution_adapter as _register_validation_execution_adapter,
    )

    _register_validation_execution_adapter(adapter)


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
