# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `finelog.deploy.config`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from finelog.deploy.config import (
    Deployment,
    FinelogConfig,
    GcpDeployment,
    K8sDeployment,
    derive_endpoint_uri,
    load_finelog_config,
)


def _write_config(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def test_load_config_from_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-test
        port: 10001
        image: ghcr.io/test/finelog:latest
        remote_log_dir: gs://bucket/test
        deployment:
          gcp:
            project: my-proj
            zone: us-central1-a
        """,
    )
    cfg = load_finelog_config(str(cfg_path))
    assert cfg.name == "finelog-test"
    assert cfg.port == 10001
    assert cfg.image == "ghcr.io/test/finelog:latest"
    assert cfg.remote_log_dir == "gs://bucket/test"
    assert cfg.deployment.gcp is not None
    assert cfg.deployment.gcp.project == "my-proj"
    assert cfg.deployment.gcp.zone == "us-central1-a"
    assert cfg.deployment.gcp.machine_type == "n2-standard-4"  # default
    assert cfg.deployment.k8s is None


def test_load_config_from_repo_marin() -> None:
    cfg = load_finelog_config("marin")
    assert cfg.name == "finelog-marin"
    assert cfg.port == 10001
    assert cfg.image == "ghcr.io/marin-community/finelog:latest"
    assert cfg.remote_log_dir == "gs://marin-us-central2/finelog/marin"
    assert cfg.deployment.gcp is not None
    assert cfg.deployment.gcp.project == "hai-gcp-models"
    assert cfg.deployment.gcp.zone == "us-central1-a"


def test_load_config_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point the user-config dir at an empty tmpdir to keep the searched-paths
    # output deterministic and rule out accidental hits.
    monkeypatch.setattr("finelog.deploy.config.USER_CONFIG_DIR", tmp_path / "no-such-dir")
    with pytest.raises(FileNotFoundError) as exc:
        load_finelog_config("definitely-not-a-real-config-name-xyz")
    msg = str(exc.value)
    assert "definitely-not-a-real-config-name-xyz" in msg
    assert "searched" in msg


def test_load_config_neither_deployment_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-bad
        port: 10001
        image: ghcr.io/test/finelog:latest
        deployment: {}
        """,
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_finelog_config(str(cfg_path))


def test_load_config_both_deployments_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "both.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-both
        port: 10001
        image: ghcr.io/test/finelog:latest
        deployment:
          gcp:
            project: p
            zone: z
          k8s:
            namespace: ns
        """,
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_finelog_config(str(cfg_path))


def test_derive_endpoint_uri_gcp() -> None:
    cfg = FinelogConfig(
        name="finelog-x",
        port=10001,
        image="ghcr.io/x/finelog:latest",
        remote_log_dir="",
        deployment=Deployment(gcp=GcpDeployment(project="proj", zone="us-central1-a")),
    )
    uri, metadata = derive_endpoint_uri(cfg)
    assert uri == "gcp://finelog-x"
    assert metadata == {"project": "proj", "zone": "us-central1-a", "port": "10001"}


def test_derive_endpoint_uri_k8s() -> None:
    cfg = FinelogConfig(
        name="finelog-x",
        port=10001,
        image="ghcr.io/x/finelog:latest",
        remote_log_dir="",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
    )
    uri, metadata = derive_endpoint_uri(cfg)
    assert uri == "k8s://finelog-x.iris"
    assert metadata == {"port": "10001"}
