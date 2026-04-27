# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint resolution in the controller daemon entrypoint."""

from __future__ import annotations

from iris.cluster.controller.main import LOG_SERVER_ENDPOINT_NAME, _resolve_cluster_endpoints
from iris.rpc import config_pb2


def test_resolve_returns_empty_for_empty_config():
    cfg = config_pb2.IrisClusterConfig()
    assert _resolve_cluster_endpoints(cfg) == {}


def test_resolve_dispatches_http_uri():
    cfg = config_pb2.IrisClusterConfig()
    cfg.endpoints[LOG_SERVER_ENDPOINT_NAME].uri = "http://logs.example:10001"
    resolved = _resolve_cluster_endpoints(cfg)
    assert resolved == {LOG_SERVER_ENDPOINT_NAME: "http://logs.example:10001"}


def test_resolve_does_not_require_log_server():
    """No /system/log_server in config is fine — Controller falls back to a
    bundled in-process MemStore log server (see iris.cluster.controller.controller
    for the fallback wiring)."""
    cfg = config_pb2.IrisClusterConfig()
    cfg.endpoints["/system/something_else"].uri = "http://other.example:9000"
    resolved = _resolve_cluster_endpoints(cfg)
    assert LOG_SERVER_ENDPOINT_NAME not in resolved
    assert "/system/something_else" in resolved
