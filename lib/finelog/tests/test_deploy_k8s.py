# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for k8s manifest rendering in `finelog.deploy._k8s`."""

import base64
import json

import click
import pytest
from finelog.deploy._k8s import (
    _K8S_MANIFEST_DIR,
    _MANIFESTS,
    _build_s3_secret_manifest,
    _render_manifest,
    _s3_secret_name,
)
from finelog.deploy.config import (
    CidrAuthLayer,
    Deployment,
    FinelogConfig,
    JwtAuthLayer,
    JwtKeyEntry,
    K8sDeployment,
)


def _s3_cfg(**k8s_overrides) -> FinelogConfig:
    k8s = {
        "namespace": "iris",
        "object_storage_endpoint": "https://acct.r2.cloudflarestorage.com",
    }
    k8s.update(k8s_overrides)
    return FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="img",
        remote_log_dir="s3://bucket/finelog/cw",
        deployment=Deployment(gcp=None, k8s=K8sDeployment(**k8s)),
    )


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


def test_render_deployment_inlines_cidr_auth_policy() -> None:
    cfg = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
        auth=(CidrAuthLayer(cidrs=("10.0.0.0/8",)),),
    )
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert "name: FINELOG_AUTH_POLICY" in rendered
    assert '"type":"cidr"' in rendered


def test_render_deployment_rejects_inline_jwt_secret() -> None:
    # jwt keys are secret material; the manifest is plaintext, so the deploy path must
    # refuse to inline them (they belong in the finelog-env Secret).
    cfg = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
        auth=(JwtAuthLayer(keys=(JwtKeyEntry(cluster="marin", secret="0123456789abcdef0123456789abcdef"),)),),
    )
    with pytest.raises(ValueError, match="jwt auth layers carry secret"):
        _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)


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


def test_s3_secret_minted_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SEKRIT")
    cfg = _s3_cfg()
    manifest = json.loads(_build_s3_secret_manifest(cfg))
    data = {k: base64.b64decode(v).decode() for k, v in manifest["data"].items()}
    # The R2->AWS name mapping + injected region are the actual logic the Rust
    # server's from_env() depends on; the rest of the manifest is boilerplate.
    assert data == {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SEKRIT",
        "AWS_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
        "AWS_REGION": "auto",
        "AWS_DEFAULT_REGION": "auto",
    }


def test_no_secret_for_non_s3_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SEKRIT")
    cfg = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(gcp=None, k8s=K8sDeployment(namespace="iris")),
    )
    assert _build_s3_secret_manifest(cfg) is None


def test_s3_secret_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SEKRIT")
    with pytest.raises(click.ClickException, match="object_storage_endpoint"):
        _build_s3_secret_manifest(_s3_cfg(object_storage_endpoint=None))


def test_s3_secret_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(click.ClickException, match="R2_ACCESS_KEY_ID"):
        _build_s3_secret_manifest(_s3_cfg())


def test_deployment_envfrom_matches_minted_secret_name() -> None:
    # The template's envFrom secret name is a hardcoded `{{ name }}-env` literal,
    # independent of the `_s3_secret_name` helper that names the minted Secret —
    # this guards the two from drifting (a mismatch silently breaks auth).
    cfg = _s3_cfg()
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert f"name: {_s3_secret_name(cfg)}" in rendered


def test_manifest_dir_exists() -> None:
    """Guard against the parents[3] path math drifting if the package is moved."""
    assert _K8S_MANIFEST_DIR.is_dir(), _K8S_MANIFEST_DIR
    for name in _MANIFESTS:
        assert (_K8S_MANIFEST_DIR / name).is_file(), name
