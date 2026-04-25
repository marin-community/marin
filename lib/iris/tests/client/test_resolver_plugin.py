# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the iris:// resolver plugin."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

import iris.client  # noqa: F401  -- side-effect import: registers iris:// scheme
from iris.client import resolver_plugin
from iris.rpc import controller_pb2
from rigging.proxy import proxy_stack
from rigging.resolver import is_registered, resolve


class _FakeControllerClient:
    """Stubs ControllerServiceClientSync + its context manager."""

    def __init__(self, endpoints: dict[str, str]):
        self._endpoints = endpoints
        self.last_request: controller_pb2.Controller.ListEndpointsRequest | None = None
        self.address: str | None = None

    def __enter__(self) -> "_FakeControllerClient":
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        self.last_request = request
        matches = [
            controller_pb2.Controller.Endpoint(name=n, address=a)
            for n, a in self._endpoints.items()
            if n == request.prefix
        ]
        return controller_pb2.Controller.ListEndpointsResponse(endpoints=matches)


@dataclass
class _TunnelEvent:
    target: tuple[str, int]
    closed: bool = False


@dataclass
class _FakeControllerProvider:
    address: str
    tunnel_events: list[_TunnelEvent] = field(default_factory=list)
    _next_local: int = 50000

    def discover_controller(self, _controller_config) -> str:
        return self.address

    def tunnel_to(self, host: str, port: int, local_port: int | None = None):
        event = _TunnelEvent(target=(host, port))
        self.tunnel_events.append(event)
        local = self._next_local
        self._next_local += 1

        @contextmanager
        def _cm() -> Iterator[tuple[str, int]]:
            try:
                yield ("127.0.0.1", local)
            finally:
                event.closed = True

        return _cm()


@dataclass
class _FakeBundle:
    controller: _FakeControllerProvider


@dataclass
class _FakeIrisConfig:
    """Stubs the bits of ``IrisConfig`` the plugin reads."""

    controller_address: str
    _provider: _FakeControllerProvider | None = None

    @property
    def proto(self):
        # The plugin only reads .proto.controller and hands it to
        # discover_controller, which our fake provider ignores.
        class _P:
            controller = object()

        return _P()

    def provider_bundle(self) -> _FakeBundle:
        if self._provider is None:
            self._provider = _FakeControllerProvider(self.controller_address)
        return _FakeBundle(controller=self._provider)


@pytest.fixture
def patch_resolver(monkeypatch):
    """Replace load_cluster_config + ControllerServiceClientSync with stubs."""
    cluster_calls: list[str] = []
    client_addresses: list[str] = []
    configs: dict[str, _FakeIrisConfig] = {}

    def _install(
        *, controller_address: str = "controller.example.com:10000", endpoints: dict[str, str] | None = None
    ) -> _FakeControllerClient:
        endpoints = endpoints or {}
        fake = _FakeControllerClient(endpoints)

        def _fake_load(name: str) -> _FakeIrisConfig:
            cluster_calls.append(name)
            cfg = configs.get(name)
            if cfg is None:
                cfg = _FakeIrisConfig(controller_address=controller_address)
                configs[name] = cfg
            return cfg

        def _fake_client(address: str) -> _FakeControllerClient:
            client_addresses.append(address)
            fake.address = address
            return fake

        monkeypatch.setattr(resolver_plugin, "load_cluster_config", _fake_load)
        monkeypatch.setattr(resolver_plugin, "ControllerServiceClientSync", _fake_client)
        return fake

    _install.cluster_calls = cluster_calls  # type: ignore[attr-defined]
    _install.client_addresses = client_addresses  # type: ignore[attr-defined]
    _install.configs = configs  # type: ignore[attr-defined]
    return _install


def test_resolve_iris_returns_endpoint_address(patch_resolver):
    patch_resolver(
        controller_address="iris-controller-svc.iris.svc.cluster.local:10000",
        endpoints={"/system/x": "host.example.com:1234"},
    )
    assert resolve("iris://marin?endpoint=/system/x") == ("host.example.com", 1234)
    assert patch_resolver.cluster_calls == ["marin"]
    assert patch_resolver.client_addresses == ["http://iris-controller-svc.iris.svc.cluster.local:10000"]


def test_resolve_iris_uses_provider_address_for_gcp_form(patch_resolver):
    patch_resolver(
        controller_address="10.0.0.5:10000",
        endpoints={"/system/log-server": "10.0.0.42:10002"},
    )
    assert resolve("iris://marin?endpoint=/system/log-server") == ("10.0.0.42", 10002)
    assert patch_resolver.client_addresses == ["http://10.0.0.5:10000"]


def test_resolve_iris_not_found_raises(patch_resolver):
    patch_resolver(endpoints={})
    with pytest.raises(KeyError, match="iris endpoint not found"):
        resolve("iris://marin?endpoint=/system/missing")


def test_resolve_iris_requires_endpoint_query(patch_resolver):
    patch_resolver()
    with pytest.raises(ValueError, match="requires \\?endpoint="):
        resolve("iris://marin")


def test_resolve_iris_rejects_port(patch_resolver):
    patch_resolver()
    with pytest.raises(ValueError, match="cannot have a port"):
        resolve("iris://marin:9000?endpoint=/x")


def test_iris_scheme_registered_after_iris_client_import():
    assert is_registered("iris")


def test_resolve_iris_strips_http_scheme_from_endpoint(patch_resolver):
    """Controller stores system endpoints with scheme prefix
    (``controller.py:1201``); the resolver must strip it before returning
    a ``(host, port)`` tuple."""
    patch_resolver(
        controller_address="10.0.0.5:10000",
        endpoints={"/system/log-server": "http://10.0.0.42:10001"},
    )
    assert resolve("iris://marin?endpoint=/system/log-server") == ("10.0.0.42", 10001)


# ---------------------------------------------------------------------------
# proxy_stack integration: handler tunnels controller + endpoint when
# unreachable and skips the tunnel when directly reachable.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_reachability(monkeypatch):
    """Force is_reachable to a fixed answer (default: unreachable)."""
    state = {"reachable": False}

    def _set(reachable: bool) -> None:
        state["reachable"] = reachable

    monkeypatch.setattr(resolver_plugin, "is_reachable", lambda *a, **k: state["reachable"])
    return _set


def test_resolve_iris_tunnels_when_unreachable(patch_resolver, stub_reachability):
    stub_reachability(False)
    patch_resolver(
        controller_address="10.0.0.5:10000",
        endpoints={"/system/log-server": "10.0.0.42:10002"},
    )
    with proxy_stack():
        addr = resolve("iris://marin?endpoint=/system/log-server")

    assert addr[0] == "127.0.0.1"
    # gRPC dialed the tunneled controller (127.0.0.1:50000), not the raw 10.0.0.5.
    assert patch_resolver.client_addresses == ["http://127.0.0.1:50000"]
    provider = patch_resolver.configs["marin"]._provider
    assert [e.target for e in provider.tunnel_events] == [("10.0.0.5", 10000), ("10.0.0.42", 10002)]
    # Tunnels closed on scope exit.
    assert all(e.closed for e in provider.tunnel_events)


def test_resolve_iris_skips_tunnel_when_reachable(patch_resolver, stub_reachability):
    stub_reachability(True)
    patch_resolver(
        controller_address="10.0.0.5:10000",
        endpoints={"/system/log-server": "10.0.0.42:10002"},
    )
    with proxy_stack():
        assert resolve("iris://marin?endpoint=/system/log-server") == ("10.0.0.42", 10002)
    provider = patch_resolver.configs["marin"]._provider
    assert provider.tunnel_events == []


def test_resolve_iris_no_proxy_stack_returns_raw(patch_resolver, stub_reachability):
    """Outside a proxy_stack scope, the handler never tunnels even if the
    address would fail the reachability probe."""
    stub_reachability(False)
    patch_resolver(
        controller_address="10.0.0.5:10000",
        endpoints={"/system/log-server": "10.0.0.42:10002"},
    )
    assert resolve("iris://marin?endpoint=/system/log-server") == ("10.0.0.42", 10002)
    provider = patch_resolver.configs["marin"]._provider
    assert provider.tunnel_events == []


def test_resolve_iris_caches_controller_tunnel(patch_resolver, stub_reachability):
    """Two iris:// resolves in one scope reuse the controller tunnel."""
    stub_reachability(False)
    patch_resolver(
        controller_address="10.0.0.5:10000",
        endpoints={"/system/a": "10.0.0.42:10002", "/system/b": "10.0.0.43:10002"},
    )
    with proxy_stack():
        resolve("iris://marin?endpoint=/system/a")
        resolve("iris://marin?endpoint=/system/b")
    provider = patch_resolver.configs["marin"]._provider
    targets = [e.target for e in provider.tunnel_events]
    # Controller tunnel opened once; one tunnel each for the two endpoints.
    assert targets == [("10.0.0.5", 10000), ("10.0.0.42", 10002), ("10.0.0.43", 10002)]
