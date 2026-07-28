# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the out-of-band Secret used by Pulumi-managed Finelog servers."""

import base64
import json
from dataclasses import replace

import click
import pytest
from click.testing import CliRunner
from finelog.deploy._k8s import _build_env_secret_manifest
from finelog.deploy.cli import cli
from finelog.deploy.config import (
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    K8sDeployment,
)
from rigging.secrets import SecretResolutionError


def _s3_config(**k8s_overrides) -> FinelogConfig:
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
        deployment=Deployment(k8s=K8sDeployment(**k8s)),
    )


def _secret_data(config: FinelogConfig) -> dict[str, str]:
    manifest = _build_env_secret_manifest(config)
    assert manifest is not None
    encoded = json.loads(manifest)["data"]
    return {key: base64.b64decode(value).decode() for key, value in encoded.items()}


def test_env_secret_minted_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "SEKRIT")

    assert _secret_data(_s3_config()) == {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SEKRIT",
        "AWS_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
        "AWS_REGION": "auto",
        "AWS_DEFAULT_REGION": "auto",
    }


def test_no_secret_for_non_s3_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "SEKRIT")
    config = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
    )

    assert _build_env_secret_manifest(config) is None


def test_env_secret_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "SEKRIT")

    with pytest.raises(click.ClickException, match="object_storage_endpoint"):
        _build_env_secret_manifest(_s3_config(object_storage_endpoint=None))


def test_env_secret_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R2_KEY_ID", raising=False)
    monkeypatch.delenv("R2_KEY_SECRET", raising=False)

    with pytest.raises(click.ClickException, match="R2_KEY_ID"):
        _build_env_secret_manifest(_s3_config())


FORWARDING = ForwardingConfig(
    target="https://finelog.oa.dev",
    cluster="cw-rno2a",
    signing_key=("env:TEST_FINELOG_SIGNING_KEY",),
)


def _forwarding_config() -> FinelogConfig:
    return FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
        forwarding=FORWARDING,
    )


def test_forwarding_signing_key_stays_in_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    key_pem = "-----BEGIN PRIVATE KEY-----\nSEKRIT\n-----END PRIVATE KEY-----"
    monkeypatch.setenv("TEST_FINELOG_SIGNING_KEY", key_pem)

    assert _secret_data(_forwarding_config()) == {"FINELOG_SIGNING_KEY": key_pem}


def test_env_secret_carries_s3_credentials_and_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "R2SEKRIT")
    monkeypatch.setenv("TEST_FINELOG_SIGNING_KEY", "PRIVKEY")
    config = replace(_s3_config(), forwarding=FORWARDING)

    data = _secret_data(config)

    assert data["AWS_ACCESS_KEY_ID"] == "AKID"
    assert data["FINELOG_SIGNING_KEY"] == "PRIVKEY"


def test_env_secret_fails_when_signing_key_source_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_FINELOG_SIGNING_KEY", raising=False)

    with pytest.raises(SecretResolutionError):
        _build_env_secret_manifest(_forwarding_config())


def test_k8s_mutation_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "finelog.yaml"
    config_path.write_text(
        """
name: finelog-cw
port: 10001
image: ghcr.io/marin-community/finelog:latest
deployment:
  k8s:
    namespace: iris
""".lstrip()
    )

    result = CliRunner().invoke(cli, ["deploy", "up", "--no-build", str(config_path)])

    assert result.exit_code == 1
    assert result.output.startswith("Error:")
