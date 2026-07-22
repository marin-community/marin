# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the portable diagnostic metadata boundary."""

import pytest
from iris.cluster.config import CoreweavePlatformConfig, DiagnosticsConfig, IrisClusterConfig, PlatformConfig
from iris.cluster.types import EnvironmentSpec, JobName
from iris.diagnostics.metadata import (
    DiagnosticJobMetadata,
    DiagnosticWorkloadKind,
    attach_diagnostic_metadata,
    metadata_from_environment,
)
from iris.diagnostics.resolution import DiagnosticArtifactSource, DiagnosticCluster, resolve_diagnostic_artifact


def test_diagnostic_metadata_round_trips_through_explicit_environment():
    """Metadata uses the submit environment rather than local launch-host state."""
    metadata = DiagnosticJobMetadata(
        workload_kind=DiagnosticWorkloadKind.DATAGEN,
        dataset="DCAgent/code-contests-noblock",
        artifact_uri="s3://marin-us-east-02a/harbor/glm52-r10",
        total_trials=8728,
        serving_endpoint="/benjaminfeuer/glm52",
        model_identifier="vllm/laion/glm-5.2-awq",
    )

    environment = attach_diagnostic_metadata(EnvironmentSpec(env_vars={"UNCHANGED": "yes"}), metadata)

    assert environment.env_vars["UNCHANGED"] == "yes"
    assert metadata_from_environment(environment.env_vars) == metadata


def test_diagnostic_cluster_uses_configured_coreweave_context_not_kubeconfig_environment(monkeypatch):
    monkeypatch.setenv("KUBECONFIG", "/stale/kubeconfig")
    config = IrisClusterConfig(
        name="cw-rno2a",
        diagnostics=DiagnosticsConfig(evidence_root="s3://diagnostics/evidence"),
        platform=PlatformConfig(
            coreweave=CoreweavePlatformConfig(
                namespace="iris",
                kubeconfig_path="~/configured-kubeconfig",
                kube_context="marin-rn02a_RNO2A",
            )
        ),
    )

    cluster = DiagnosticCluster.from_config(config)

    assert cluster.namespace == "iris"
    assert cluster.kubeconfig_path is not None
    assert cluster.kubeconfig_path.name == "configured-kubeconfig"
    assert cluster.kube_context == "marin-rn02a_RNO2A"
    assert cluster.evidence_uri(JobName.root("alice", "run-1")) == "s3://diagnostics/evidence/jobs/cw-rno2a/alice/run-1"


def test_artifact_resolution_prefers_explicit_uri_and_gates_legacy_inference():
    legacy_entrypoint = "python run_tracegen.py --jobs-dir gs://legacy-bucket/jobs/run-1"
    metadata = DiagnosticJobMetadata(
        workload_kind=DiagnosticWorkloadKind.EVAL,
        artifact_uri="gs://typed-bucket/jobs/run-1",
    )

    explicit = resolve_diagnostic_artifact(
        metadata,
        artifact_uri_override="s3://override-bucket/results/run-1",
        legacy_entrypoint=legacy_entrypoint,
        allow_legacy=True,
    )
    typed = resolve_diagnostic_artifact(metadata, legacy_entrypoint=legacy_entrypoint, allow_legacy=True)
    legacy = resolve_diagnostic_artifact(None, legacy_entrypoint=legacy_entrypoint, allow_legacy=True)

    assert explicit.source == DiagnosticArtifactSource.EXPLICIT
    assert explicit.uri == "s3://override-bucket/results/run-1"
    assert typed.source == DiagnosticArtifactSource.METADATA
    assert typed.uri == "gs://typed-bucket/jobs/run-1"
    assert legacy.source == DiagnosticArtifactSource.LEGACY
    assert legacy.uri == "gs://legacy-bucket/jobs/run-1"


def test_artifact_resolution_refuses_legacy_layout_without_explicit_opt_in():
    with pytest.raises(LookupError, match="opt into legacy"):
        resolve_diagnostic_artifact(
            None,
            legacy_entrypoint="python run_tracegen.py --jobs-dir gs://legacy-bucket/jobs/run-1",
        )


def test_artifact_resolution_does_not_infer_legacy_for_typed_metadata_without_uri():
    with pytest.raises(LookupError, match="metadata has no artifact_uri"):
        resolve_diagnostic_artifact(
            DiagnosticJobMetadata(workload_kind=DiagnosticWorkloadKind.DATAGEN),
            legacy_entrypoint="python run_tracegen.py --jobs-dir gs://legacy-bucket/jobs/run-1",
            allow_legacy=True,
        )
