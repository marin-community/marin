# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for k8s manifest rendering in `finelog.deploy._k8s`."""

import base64
import json
import subprocess
from dataclasses import replace

import click
import pytest
from finelog.deploy._k8s import (
    _K8S_MANIFEST_DIR,
    _MANIFESTS,
    _build_env_secret_manifest,
    _env_secret_name,
    _render_manifest,
    k8s_down,
)
from finelog.deploy.config import (
    CidrAuthLayer,
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    JwtAuthLayer,
    JwtKeyEntry,
    K8sDeployment,
)
from rigging.secrets import SecretResolutionError


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


def test_render_deployment_omits_priority_class_by_default(cfg: FinelogConfig) -> None:
    """No priority_class_name configured -> the pod carries no priorityClassName."""
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert "priorityClassName" not in rendered


def test_render_deployment_stamps_priority_class_when_configured() -> None:
    """A configured priority_class_name lands in the pod spec so the control-plane
    finelog is not preemptible by user jobs on the shared node."""
    cfg = FinelogConfig(
        name="finelog",
        port=20001,
        image="ghcr.io/example/finelog:dev",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(
            k8s=K8sDeployment(namespace="iris", priority_class_name="iris-system", priority_class_value=10000)
        ),
    )
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert "priorityClassName: iris-system" in rendered


def test_k8s_deployment_rejects_priority_class_name_without_value() -> None:
    """Name and value are meaningless apart — deploy up needs the value to create
    the class, so half a config must fail at construction, not at apply time."""
    with pytest.raises(ValueError, match="must be set together"):
        K8sDeployment(namespace="iris", priority_class_name="iris-system")


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


# An Ed25519 public key in PEM (SubjectPublicKeyInfo). Public, so inline-safe.
_PUB_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MCowBQYDK2VwAyEAqwwvfFvyRQ+8Dhh0li8h2HtCT4yP40s0pzBwwSAkK5s=\n"
    "-----END PUBLIC KEY-----\n"
)


def test_render_deployment_inlines_jwt_public_key() -> None:
    # jwt keys are Ed25519 public keys, not symmetric secrets, so the deploy path
    # inlines them into the plaintext manifest like a cidr layer.
    cfg = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
        auth=(JwtAuthLayer(keys=(JwtKeyEntry(cluster="marin", public_keys=(_PUB_PEM,)),)),),
    )
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert "name: FINELOG_AUTH_POLICY" in rendered
    assert '"type":"jwt"' in rendered
    assert '"public_keys"' in rendered


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


def test_env_secret_minted_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SEKRIT")
    cfg = _s3_cfg()
    manifest = json.loads(_build_env_secret_manifest(cfg))
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
    assert _build_env_secret_manifest(cfg) is None


def test_env_secret_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SEKRIT")
    with pytest.raises(click.ClickException, match="object_storage_endpoint"):
        _build_env_secret_manifest(_s3_cfg(object_storage_endpoint=None))


def test_env_secret_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(click.ClickException, match="R2_ACCESS_KEY_ID"):
        _build_env_secret_manifest(_s3_cfg())


_FORWARDING = ForwardingConfig(
    target="https://finelog.oa.dev",
    cluster="cw-rno2a",
    signing_key=("env:TEST_FINELOG_SIGNING_KEY",),
)


def _forwarding_cfg() -> FinelogConfig:
    return FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
        forwarding=_FORWARDING,
    )


def test_render_deployment_inlines_forwarding_target_and_cluster() -> None:
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", _forwarding_cfg())
    assert "name: FINELOG_FORWARDING" in rendered
    assert '"target":"https://finelog.oa.dev"' in rendered
    assert '"cluster":"cw-rno2a"' in rendered


def test_forwarding_signing_key_never_leaves_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private key reaches the pod through the Secret and through nothing else.

    A rendered manifest is plaintext — `kubectl get deployment -o yaml` echoes it back to
    anyone with read access on the namespace — so a key that lands there is a key that
    leaks.
    """
    key_pem = "-----BEGIN PRIVATE KEY-----\nSEKRIT\n-----END PRIVATE KEY-----"
    monkeypatch.setenv("TEST_FINELOG_SIGNING_KEY", key_pem)
    cfg = _forwarding_cfg()

    manifest = json.loads(_build_env_secret_manifest(cfg))
    data = {k: base64.b64decode(v).decode() for k, v in manifest["data"].items()}
    assert data == {"FINELOG_SIGNING_KEY": key_pem}

    for manifest_name in _MANIFESTS:
        assert "SEKRIT" not in _render_manifest(_K8S_MANIFEST_DIR / manifest_name, cfg)


def test_env_secret_carries_both_s3_credentials_and_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forwarding server with an s3:// archive needs both, in the one Secret."""
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "R2SEKRIT")
    monkeypatch.setenv("TEST_FINELOG_SIGNING_KEY", "PRIVKEY")
    cfg = replace(_s3_cfg(), forwarding=_FORWARDING)
    manifest = json.loads(_build_env_secret_manifest(cfg))
    data = {k: base64.b64decode(v).decode() for k, v in manifest["data"].items()}
    assert data["AWS_ACCESS_KEY_ID"] == "AKID"
    assert data["FINELOG_SIGNING_KEY"] == "PRIVKEY"


def test_env_secret_fails_when_the_signing_key_source_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable key fails the deploy. Starting a server that can never
    authenticate to its hub looks exactly like a quiet cluster."""
    monkeypatch.delenv("TEST_FINELOG_SIGNING_KEY", raising=False)
    with pytest.raises(SecretResolutionError):
        _build_env_secret_manifest(_forwarding_cfg())


def test_deployment_envfrom_matches_minted_secret_name() -> None:
    # The template's envFrom secret name is a hardcoded `{{ name }}-env` literal,
    # independent of the `_env_secret_name` helper that names the minted Secret —
    # this guards the two from drifting (a mismatch silently breaks auth).
    cfg = _s3_cfg()
    rendered = _render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", cfg)
    assert f"name: {_env_secret_name(cfg)}" in rendered


def _kubectl_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture the argv of every kubectl invocation instead of running it."""
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _deleted_resources(calls: list[list[str]]) -> set[str]:
    """The `kind/name` resources named across every `kubectl delete` in `calls`."""
    return {arg for argv in calls if "delete" in argv for arg in argv if "/" in arg}


def test_teardown_deletes_the_secret_and_retains_only_the_cache_pvc(monkeypatch: pytest.MonkeyPatch) -> None:
    # The env Secret carries the forwarding private key. Leaving it behind after a
    # teardown strands key material in a namespace with nothing left to use it.
    calls = _kubectl_argv(monkeypatch)
    cfg = _forwarding_cfg()

    k8s_down(cfg, yes=False)

    assert _deleted_resources(calls) == {
        f"deployment/{cfg.name}",
        f"service/{cfg.name}",
        f"secret/{_env_secret_name(cfg)}",
    }


def test_manifest_dir_exists() -> None:
    """Guard against the parents[3] path math drifting if the package is moved."""
    assert _K8S_MANIFEST_DIR.is_dir(), _K8S_MANIFEST_DIR
    for name in _MANIFESTS:
        assert (_K8S_MANIFEST_DIR / name).is_file(), name
