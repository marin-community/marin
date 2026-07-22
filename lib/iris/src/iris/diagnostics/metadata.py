# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned job metadata for portable diagnostic tooling."""

import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from urllib.parse import urlparse

from iris.cluster.types import EnvironmentSpec

DIAGNOSTIC_METADATA_ENV = "IRIS_DIAGNOSTIC_METADATA"
DIAGNOSTIC_METADATA_VERSION = 1
_ARTIFACT_SCHEMES = frozenset(("gs", "s3"))


class DiagnosticWorkloadKind(StrEnum):
    """Workload families supported by the diagnostic clients."""

    DATAGEN = "datagen"
    EVAL = "eval"
    MIRROR = "mirror"
    RL = "rl"
    SERVE = "serve"
    OTHER = "other"


def validate_artifact_uri(value: str) -> str:
    """Validate and normalize an object-store artifact URI."""
    uri = value.strip().rstrip("/")
    parsed = urlparse(uri)
    if parsed.scheme not in _ARTIFACT_SCHEMES or not parsed.netloc:
        schemes = ", ".join(sorted(_ARTIFACT_SCHEMES))
        raise ValueError(f"Artifact URI must use one of {schemes}: {value!r}")
    return uri


def _optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string when provided")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty when provided")
    return normalized


@dataclass(frozen=True, slots=True)
class DiagnosticJobMetadata:
    """Stable metadata emitted with a newly submitted diagnosable job.

    The JSON representation is stored in the submitted job's explicit
    environment configuration. Iris persists that configuration with the job,
    and the task receives the same value without relying on a launch-host path
    or secret file.
    """

    workload_kind: DiagnosticWorkloadKind
    dataset: str | None = None
    artifact_uri: str | None = None
    total_trials: int | None = None
    serving_endpoint: str | None = None
    model_identifier: str | None = None
    schema_version: int = DIAGNOSTIC_METADATA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.workload_kind, DiagnosticWorkloadKind):
            raise ValueError("workload_kind must be a DiagnosticWorkloadKind")
        if self.schema_version != DIAGNOSTIC_METADATA_VERSION:
            raise ValueError(
                f"Unsupported diagnostic metadata schema version {self.schema_version}; "
                f"expected {DIAGNOSTIC_METADATA_VERSION}"
            )
        object.__setattr__(self, "dataset", _optional_text("dataset", self.dataset))
        object.__setattr__(self, "serving_endpoint", _optional_text("serving_endpoint", self.serving_endpoint))
        object.__setattr__(self, "model_identifier", _optional_text("model_identifier", self.model_identifier))
        if self.artifact_uri is not None:
            object.__setattr__(self, "artifact_uri", validate_artifact_uri(self.artifact_uri))
        if self.total_trials is not None and (
            isinstance(self.total_trials, bool) or not isinstance(self.total_trials, int) or self.total_trials < 0
        ):
            raise ValueError("total_trials must be a non-negative integer when provided")

    def to_json(self) -> str:
        """Encode the contract as canonical JSON for the submitted environment."""
        payload = asdict(self)
        payload["workload_kind"] = self.workload_kind.value
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "DiagnosticJobMetadata":
        """Decode one metadata value, rejecting unknown or malformed fields."""
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Diagnostic metadata is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Diagnostic metadata must be a JSON object")
        expected = {
            "artifact_uri",
            "dataset",
            "model_identifier",
            "schema_version",
            "serving_endpoint",
            "total_trials",
            "workload_kind",
        }
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"Diagnostic metadata has unknown fields: {', '.join(sorted(unknown))}")
        try:
            workload_kind = DiagnosticWorkloadKind(payload["workload_kind"])
        except KeyError as exc:
            raise ValueError("Diagnostic metadata is missing workload_kind") from exc
        except ValueError as exc:
            raise ValueError(f"Unknown diagnostic workload kind {payload.get('workload_kind')!r}") from exc
        total_trials = payload.get("total_trials")
        if total_trials is not None and (isinstance(total_trials, bool) or not isinstance(total_trials, int)):
            raise ValueError("total_trials must be an integer when provided")
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("schema_version must be an integer")
        return cls(
            workload_kind=workload_kind,
            dataset=payload.get("dataset"),
            artifact_uri=payload.get("artifact_uri"),
            total_trials=total_trials,
            serving_endpoint=payload.get("serving_endpoint"),
            model_identifier=payload.get("model_identifier"),
            schema_version=schema_version,
        )


def attach_diagnostic_metadata(
    environment: EnvironmentSpec | None,
    metadata: DiagnosticJobMetadata,
) -> EnvironmentSpec:
    """Return an environment that persists one diagnostic metadata contract."""
    env_vars = dict(environment.env_vars) if environment and environment.env_vars else {}
    encoded = metadata.to_json()
    existing = env_vars.get(DIAGNOSTIC_METADATA_ENV)
    if existing is not None and existing != encoded:
        raise ValueError(f"{DIAGNOSTIC_METADATA_ENV} is already set to different metadata")
    env_vars[DIAGNOSTIC_METADATA_ENV] = encoded
    if environment is None:
        return EnvironmentSpec(env_vars=env_vars)
    return replace(environment, env_vars=env_vars)


def metadata_from_environment(environment: dict[str, str] | None) -> DiagnosticJobMetadata | None:
    """Read typed diagnostic metadata from persisted explicit environment variables."""
    if environment is None:
        return None
    value = environment.get(DIAGNOSTIC_METADATA_ENV)
    if value is None:
        return None
    return DiagnosticJobMetadata.from_json(value)
