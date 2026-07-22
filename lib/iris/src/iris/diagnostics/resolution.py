# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve diagnostic storage and legacy artifact locations from cluster configuration."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from iris.cluster.config import IrisClusterConfig
from iris.cluster.types import JobName
from iris.diagnostics.metadata import DiagnosticJobMetadata, validate_artifact_uri


class DiagnosticArtifactSource(StrEnum):
    """How an artifact URI was determined."""

    EXPLICIT = "explicit"
    METADATA = "metadata"
    LEGACY = "legacy"


@dataclass(frozen=True, slots=True)
class DiagnosticCluster:
    """The diagnostic-relevant, configuration-derived cluster settings."""

    name: str
    evidence_root: str | None
    namespace: str | None
    kubeconfig_path: Path | None
    kube_context: str | None

    @classmethod
    def from_config(cls, config: IrisClusterConfig) -> "DiagnosticCluster":
        evidence_root = config.diagnostics.evidence_root or None
        if evidence_root is not None:
            evidence_root = validate_artifact_uri(evidence_root)
        coreweave = config.platform.coreweave
        if coreweave is None:
            return cls(config.name, evidence_root, None, None, None)
        kubeconfig_path = Path(coreweave.kubeconfig_path).expanduser() if coreweave.kubeconfig_path else None
        namespace = coreweave.namespace or None
        kube_context = coreweave.kube_context or None
        return cls(config.name, evidence_root, namespace, kubeconfig_path, kube_context)

    def evidence_uri(self, job_id: JobName) -> str | None:
        """Return this root job's configured remote evidence location, if any."""
        if self.evidence_root is None:
            return None
        if not job_id.is_root:
            raise ValueError(f"Diagnostic evidence is keyed by root jobs, got {job_id}")
        return f"{self.evidence_root}/jobs/{self.name}/{job_id.user}/{job_id.name}"


@dataclass(frozen=True, slots=True)
class ResolvedDiagnosticArtifact:
    """One resolved artifact URI and the authority that supplied it."""

    uri: str
    source: DiagnosticArtifactSource


_LEGACY_ARTIFACT_PATTERNS = (
    re.compile(r"(?:--harbor_extra_arg=)?--jobs-dir(?:=|\s+)(?P<uri>(?:s3|gs)://[^\s'\"\\]+)"),
    re.compile(r"(?:--trials-dir|--experiments_dir|--gcs-output-dir)(?:=|\s+)(?P<uri>(?:s3|gs)://[^\s'\"\\]+)"),
)


def infer_legacy_artifact_uri(entrypoint: str) -> str | None:
    """Infer an OpenThoughts-era artifact URI from a historical command string.

    This parser is intentionally isolated from normal resolution. New
    submissions must provide :class:`DiagnosticJobMetadata` instead.
    """
    for pattern in _LEGACY_ARTIFACT_PATTERNS:
        match = pattern.search(entrypoint)
        if match is not None:
            return validate_artifact_uri(match.group("uri"))
    return None


def resolve_diagnostic_artifact(
    metadata: DiagnosticJobMetadata | None,
    *,
    artifact_uri_override: str | None = None,
    legacy_entrypoint: str | None = None,
    allow_legacy: bool = False,
) -> ResolvedDiagnosticArtifact:
    """Resolve an artifact URI with explicit input, typed metadata, then opt-in legacy parsing.

    Legacy command parsing is unavailable when typed metadata exists, even if
    that metadata has no artifact URI. This prevents historical heuristics from
    overriding an intentional modern submission contract.
    """
    if artifact_uri_override is not None:
        return ResolvedDiagnosticArtifact(
            validate_artifact_uri(artifact_uri_override), DiagnosticArtifactSource.EXPLICIT
        )
    if metadata is not None:
        if metadata.artifact_uri is None:
            raise LookupError("Diagnostic metadata has no artifact_uri; pass an explicit artifact URI override")
        return ResolvedDiagnosticArtifact(metadata.artifact_uri, DiagnosticArtifactSource.METADATA)
    if allow_legacy and legacy_entrypoint is not None:
        uri = infer_legacy_artifact_uri(legacy_entrypoint)
        if uri is not None:
            return ResolvedDiagnosticArtifact(uri, DiagnosticArtifactSource.LEGACY)
    raise LookupError(
        "No diagnostic artifact URI is available. Submit typed metadata, pass an explicit artifact URI override, "
        "or opt into legacy entrypoint parsing for a historical OpenThoughts job."
    )
