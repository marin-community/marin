# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint resolution in the controller daemon entrypoint."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from iris.cluster.config import ClusterFinelogConfig, EndpointSpec, IrisClusterConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.main import LOG_SERVER_ENDPOINT_NAME, _local_db_epoch_ms, _resolve_cluster_endpoints


def test_resolve_returns_empty_for_empty_config():
    cfg = IrisClusterConfig()
    assert _resolve_cluster_endpoints(cfg) == {}


def test_resolve_dispatches_http_uri():
    cfg = IrisClusterConfig(endpoints={LOG_SERVER_ENDPOINT_NAME: EndpointSpec(uri="http://logs.example:10001")})
    resolved = _resolve_cluster_endpoints(cfg)
    assert resolved == {LOG_SERVER_ENDPOINT_NAME: "http://logs.example:10001"}


def test_resolve_does_not_require_log_server():
    """No /system/log-server in config is fine — Controller falls back to a
    bundled in-process MemStore log server (see iris.cluster.controller.controller
    for the fallback wiring)."""
    cfg = IrisClusterConfig(endpoints={"/system/something_else": EndpointSpec(uri="http://other.example:9000")})
    resolved = _resolve_cluster_endpoints(cfg)
    assert LOG_SERVER_ENDPOINT_NAME not in resolved
    assert "/system/something_else" in resolved


def _write_finelog_config(path: Path, body: dict) -> None:
    path.write_text(yaml.safe_dump(body))


def test_finelog_config_synthesizes_gcp_endpoint(tmp_path: Path):
    finelog_path = tmp_path / "test.yaml"
    _write_finelog_config(
        finelog_path,
        {
            "name": "finelog-test",
            "port": 10001,
            "image": "ghcr.io/marin-community/finelog:latest",
            "remote_log_dir": "",
            "deployment": {
                "gcp": {
                    "project": "test-project",
                    "zone": "us-central1-a",
                },
            },
        },
    )

    cfg = IrisClusterConfig(finelog=ClusterFinelogConfig(config=str(finelog_path)))

    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"networkInterfaces": [{"networkIP": "10.0.0.5"}]}),
        stderr="",
    )
    with patch("iris.cluster.endpoints.subprocess.run", return_value=fake) as mock_run:
        resolved = _resolve_cluster_endpoints(cfg)

    assert resolved[LOG_SERVER_ENDPOINT_NAME] == "http://10.0.0.5:10001"
    assert mock_run.call_count == 1


def test_finelog_config_synthesizes_k8s_endpoint(tmp_path: Path):
    finelog_path = tmp_path / "test.yaml"
    _write_finelog_config(
        finelog_path,
        {
            "name": "finelog-test",
            "port": 10001,
            "image": "ghcr.io/marin-community/finelog:latest",
            "remote_log_dir": "",
            "deployment": {
                "k8s": {
                    "namespace": "iris",
                },
            },
        },
    )

    cfg = IrisClusterConfig(finelog=ClusterFinelogConfig(config=str(finelog_path)))

    with patch("iris.cluster.endpoints.subprocess.run") as mock_run:
        resolved = _resolve_cluster_endpoints(cfg)

    assert resolved[LOG_SERVER_ENDPOINT_NAME] == "http://finelog-test.iris.svc.cluster.local:10001"
    assert mock_run.call_count == 0


def test_finelog_config_with_explicit_endpoint_raises(tmp_path: Path):
    finelog_path = tmp_path / "test.yaml"
    _write_finelog_config(
        finelog_path,
        {
            "name": "finelog-test",
            "port": 10001,
            "image": "ghcr.io/marin-community/finelog:latest",
            "remote_log_dir": "",
            "deployment": {"k8s": {"namespace": "iris"}},
        },
    )

    cfg = IrisClusterConfig(
        finelog=ClusterFinelogConfig(config=str(finelog_path)),
        endpoints={LOG_SERVER_ENDPOINT_NAME: EndpointSpec(uri="http://logs.example:10001")},
    )

    with pytest.raises(ValueError, match="cannot set both"):
        _resolve_cluster_endpoints(cfg)


def test_finelog_config_missing_file_raises():
    cfg = IrisClusterConfig(finelog=ClusterFinelogConfig(config="definitely-not-a-real-config-name-xyz"))

    with pytest.raises(FileNotFoundError):
        _resolve_cluster_endpoints(cfg)


def test_local_db_epoch_ms_none_when_absent(tmp_path: Path):
    assert _local_db_epoch_ms(tmp_path / "db") is None


def test_local_db_epoch_ms_none_when_auth_db_missing(tmp_path: Path):
    """A partial local dir (main DB only) must not be trusted as a freshness signal."""
    db_dir = tmp_path / "db"
    db = ControllerDB(db_dir=db_dir)
    db.close()
    (db_dir / ControllerDB.AUTH_DB_FILENAME).unlink()

    assert _local_db_epoch_ms(db_dir) is None


def test_local_db_epoch_ms_reflects_wal_mtime(tmp_path: Path):
    """A WAL-mode write lands in the -wal sibling first; its mtime must count."""
    db_dir = tmp_path / "db"
    db = ControllerDB(db_dir=db_dir)
    with db.transaction():
        pass  # commit an IMMEDIATE transaction so the WAL file is populated

    db_path = db_dir / ControllerDB.DB_FILENAME
    wal_path = db_dir / f"{ControllerDB.DB_FILENAME}-wal"
    assert wal_path.exists(), "WAL file must exist while the connection is open"

    older = db_path.stat().st_mtime - 100
    newer = older + 50
    # Backdate every file except the main WAL, so the newest mtime is unambiguous.
    for path in db_dir.iterdir():
        os.utime(path, (older, older))
    os.utime(wal_path, (newer, newer))

    assert _local_db_epoch_ms(db_dir) == int(newer * 1000)
    db.close()
