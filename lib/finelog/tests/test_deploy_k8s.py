# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for k8s manifest rendering in `finelog.deploy._k8s`."""

from __future__ import annotations

import pytest
from finelog.deploy._k8s import _K8S_MANIFEST_DIR, _MANIFESTS, _render_manifest
from finelog.deploy.config import Deployment, FinelogConfig, K8sDeployment


@pytest.fixture
def cfg() -> FinelogConfig:
    return FinelogConfig(
        name="finelog",
        port=20001,
        image="ghcr.io/example/finelog:dev",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(
            gcp=None,
            k8s=K8sDeployment(namespace="iris", storage_class="pd-ssd", storage_gb=42),
        ),
    )


@pytest.mark.parametrize("manifest_name", _MANIFESTS)
def test_render_manifest_does_not_raise(cfg: FinelogConfig, manifest_name: str) -> None:
    """Each manifest must render without `render_template` raising on unused vars."""
    rendered = _render_manifest(_K8S_MANIFEST_DIR / manifest_name, cfg)
    assert "{{" not in rendered, f"unsubstituted placeholder in {manifest_name}: {rendered}"


def test_render_pvc_includes_storage_settings(cfg: FinelogConfig) -> None:
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "01-pvc.yaml.tmpl", cfg)
    assert "finelog-cache" in rendered
    assert "namespace: iris" in rendered
    assert "storageClassName: pd-ssd" in rendered
    assert "storage: 42Gi" in rendered


def test_render_deployment_threads_port_to_env_and_probes(cfg: FinelogConfig) -> None:
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert "image: ghcr.io/example/finelog:dev" in rendered
    assert "containerPort: 20001" in rendered
    # Probes and env both reference the configured port — required for non-default ports.
    assert "port: 20001" in rendered
    assert "name: FINELOG_PORT" in rendered
    assert 'value: "20001"' in rendered
    assert 'value: "gs://bucket/logs"' in rendered


def test_render_service_uses_configured_port(cfg: FinelogConfig) -> None:
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "03-service.yaml.tmpl", cfg)
    assert "port: 20001" in rendered
    assert "targetPort: 20001" in rendered


def test_render_pvc_omits_storage_class_when_unset() -> None:
    cfg = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="",
        deployment=Deployment(
            gcp=None,
            k8s=K8sDeployment(namespace="default", storage_class=None, storage_gb=10),
        ),
    )
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "01-pvc.yaml.tmpl", cfg)
    assert "storageClassName" in rendered  # appears in the comment fallback
    assert "<cluster default>" in rendered


def test_manifest_dir_exists() -> None:
    """Guard against the parents[3] path math drifting if the package is moved."""
    assert _K8S_MANIFEST_DIR.is_dir(), _K8S_MANIFEST_DIR
    for name in _MANIFESTS:
        assert (_K8S_MANIFEST_DIR / name).is_file(), name
