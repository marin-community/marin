# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.connect — the click-free controller-endpoint resolution.

connect_controller() resolves a reachable controller URL and owns any tunnel it
opens without pushing a click.Context. The tunnel's lifetime is therefore not
tied to a thread, so a caller can resolve the endpoint on one thread and close
it on another (e.g. harbor's start()/stop() halves, which run in different
asyncio.to_thread workers).
"""

import threading
from pathlib import Path

import click
import pytest
from iris.cli import connect


class _RecordingTunnel:
    """A tunnel context manager that counts its enter/exit calls."""

    def __init__(self, url: str):
        self.url = url
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> str:
        self.entered += 1
        return self.url

    def __exit__(self, *exc_info) -> bool:
        self.exited += 1
        return False


class _FakeBundleController:
    def __init__(self, tunnel: _RecordingTunnel):
        self._tunnel = tunnel

    def tunnel(self, address: str, local_port: int | None = None) -> _RecordingTunnel:
        return self._tunnel


class _FakeBundle:
    def __init__(self, tunnel: _RecordingTunnel):
        self.controller = _FakeBundleController(tunnel)


class _FakeController:
    def controller_kind(self) -> str:
        return "manual"


class _FakeConfig:
    """Minimal stand-in exposing only what connect resolution reads."""

    name = "test-cluster"
    auth = None

    def __init__(self):
        self.controller = _FakeController()

    def controller_address(self) -> str:
        return "http://controller:9000"


def test_connect_controller_direct_url_needs_no_config():
    with connect.connect_controller(controller_url="http://direct:8080") as endpoint:
        assert endpoint.url == "http://direct:8080"
        assert endpoint.config is None
    # Closing again is a harmless no-op (empty resource stack).
    endpoint.close()


def test_connect_controller_rejects_url_and_config():
    with pytest.raises(click.UsageError):
        connect.connect_controller(controller_url="http://x:1", config_file=Path("cluster.yaml"))


def test_connect_controller_owns_tunnel_across_threads(monkeypatch):
    """The endpoint opens the tunnel on resolve and closes it on close() — from any thread."""
    tunnel = _RecordingTunnel("http://tunnel:1234")
    monkeypatch.setattr(connect, "load_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(connect, "client_credentials", lambda _config, _name: None)
    monkeypatch.setattr(connect, "provider_bundle", lambda _config: _FakeBundle(tunnel))

    # Resolve on a worker thread; the old click.Context workaround made this
    # unsafe because the context had to be popped by its pushing thread.
    resolved: dict[str, connect.ControllerEndpoint] = {}
    opener = threading.Thread(target=lambda: resolved.update(endpoint=connect.connect_controller(config_file=Path("c.yaml"))))
    opener.start()
    opener.join()
    endpoint = resolved["endpoint"]

    assert endpoint.url == "http://tunnel:1234"
    assert (tunnel.entered, tunnel.exited) == (1, 0)

    # Close on a *different* thread than the one that opened it.
    closer = threading.Thread(target=endpoint.close)
    closer.start()
    closer.join()
    assert (tunnel.entered, tunnel.exited) == (1, 1)


def test_connect_controller_closes_tunnel_when_resolution_fails(monkeypatch):
    """A tunnel that opens but whose resolution then fails must not leak."""
    tunnel = _RecordingTunnel("http://tunnel:1234")

    class _RaisingBundleController(_FakeBundleController):
        def tunnel(self, address: str, local_port: int | None = None):
            raise RuntimeError("tunnel refused")

    class _RaisingBundle:
        def __init__(self):
            self.controller = _RaisingBundleController(tunnel)

    monkeypatch.setattr(connect, "load_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(connect, "client_credentials", lambda _config, _name: None)
    monkeypatch.setattr(connect, "provider_bundle", lambda _config: _RaisingBundle())

    with pytest.raises(click.ClickException, match="Could not connect to controller"):
        connect.connect_controller(config_file=Path("c.yaml"))
    assert tunnel.exited == 0  # never entered, so never leaked
