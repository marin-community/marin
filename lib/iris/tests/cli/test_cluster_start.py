# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``iris cluster start`` controller-reachability gating."""

import socket
from contextlib import closing

import pytest
from iris.cli import cluster as cluster_cli


def _closed_port_url() -> str:
    """A URL on a bound-then-released port, so connections are refused."""
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_cluster_start_waits_for_a_controller_that_is_not_yet_serving(
    monkeypatch: pytest.MonkeyPatch,
    kubernetes_cluster_config,
    run_cluster_start,
    stub_controller_health,
    stub_controller_provider,
) -> None:
    """The platform reports the controller up before it answers over a tunnel."""
    health = stub_controller_health(unhealthy_probes=2)
    provider = stub_controller_provider(health.url)
    monkeypatch.setattr(cluster_cli, "CONTROLLER_REACHABLE_TIMEOUT", 30.0)

    result = run_cluster_start(provider, kubernetes_cluster_config)

    assert result.exit_code == 0, result.output
    assert health.probes == 3
    assert provider.address in result.output


def test_cluster_start_fails_when_the_controller_never_answers(
    monkeypatch: pytest.MonkeyPatch, kubernetes_cluster_config, run_cluster_start, stub_controller_provider
) -> None:
    """An unreachable controller fails `cluster start` rather than the next command."""
    provider = stub_controller_provider(_closed_port_url())
    monkeypatch.setattr(cluster_cli, "CONTROLLER_REACHABLE_TIMEOUT", 1.0)

    result = run_cluster_start(provider, kubernetes_cluster_config)

    assert result.exit_code == 1
    assert provider.address in result.output
