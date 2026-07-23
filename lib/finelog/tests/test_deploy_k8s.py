# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for k8s manifest rendering in `finelog.deploy._k8s`."""

import base64
import json
import subprocess
from dataclasses import replace

import click
import pytest
import yaml
from finelog.deploy._k8s import (
    _K8S_MANIFEST_DIR,
    _MANIFESTS,
    _build_env_secret_manifest,
    _env_secret_name,
    _render_manifest,
    k8s_down,
)
from finelog.deploy.config import (
    Deployment,
    FinelogConfig,
    ForwardingConfig,
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


def test_k8s_deployment_rejects_priority_class_name_without_value() -> None:
    """Name and value are meaningless apart — deploy up needs the value to create
    the class, so half a config must fail at construction, not at apply time."""
    with pytest.raises(ValueError, match="must be set together"):
        K8sDeployment(namespace="iris", priority_class_name="iris-system")


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


def test_deployment_probes_the_http_health_endpoint() -> None:
    deployment = yaml.safe_load(_render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", _forwarding_cfg()))
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["readinessProbe"] == {
        "httpGet": {"path": "/health", "port": 10001},
        "initialDelaySeconds": 5,
        "periodSeconds": 10,
        "timeoutSeconds": 15,
        "failureThreshold": 3,
    }
    assert container["livenessProbe"] == {
        "httpGet": {"path": "/health", "port": 10001},
        "initialDelaySeconds": 15,
        "periodSeconds": 30,
        "timeoutSeconds": 15,
        "failureThreshold": 3,
    }
    assert container["startupProbe"] == {
        "httpGet": {"path": "/health", "port": 10001},
        "periodSeconds": 10,
        "timeoutSeconds": 15,
        "failureThreshold": 30,
    }


def test_k8s_deployment_reserves_burst_capacity_by_default() -> None:
    deployment = yaml.safe_load(_render_manifest(_K8S_MANIFEST_DIR / "02-deployment.yaml.tmpl", _forwarding_cfg()))
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["resources"] == {
        "requests": {"cpu": "2", "memory": "16Gi"},
        "limits": {"cpu": "8", "memory": "32Gi"},
    }


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
