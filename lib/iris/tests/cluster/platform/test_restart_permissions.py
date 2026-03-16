# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from iris.cluster.platform import restart_permissions as rp
from iris.rpc import config_pb2


def _coreweave_config() -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    config.platform.coreweave.region = "US-WEST-04A"
    return config


def test_ensure_restart_permissions_skips_platforms_without_registered_checker():
    rp.ensure_restart_permissions(_coreweave_config(), scope=rp.RestartScope.CLUSTER)


def test_register_restart_permission_checker_enables_pluggable_platform(monkeypatch):
    called: dict[str, rp.RestartScope] = {}

    def checker(config: config_pb2.IrisClusterConfig, scope: rp.RestartScope) -> None:
        called["scope"] = scope

    monkeypatch.setitem(rp._RESTART_PERMISSION_CHECKERS, "coreweave", checker)

    rp.ensure_restart_permissions(_coreweave_config(), scope=rp.RestartScope.CONTROLLER)

    assert called["scope"] == rp.RestartScope.CONTROLLER


def test_ensure_restart_permissions_wraps_runtime_error(monkeypatch):
    config = config_pb2.IrisClusterConfig()
    config.platform.gcp.project_id = "test-project"

    monkeypatch.setitem(
        rp._RESTART_PERMISSION_CHECKERS,
        "gcp",
        lambda _config, _scope: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(rp.RestartPermissionError, match="boom"):
        rp.ensure_restart_permissions(config, scope=rp.RestartScope.CLUSTER)
