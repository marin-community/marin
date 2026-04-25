# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the iris:// resolver plugin."""

from dataclasses import dataclass

import pytest

import iris.client  # noqa: F401  -- side-effect import: registers iris:// scheme
from iris.client import resolver_plugin
from iris.rpc import controller_pb2
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
class _FakeControllerProvider:
    address: str

    def discover_controller(self, _controller_config) -> str:
        return self.address


@dataclass
class _FakeBundle:
    controller: _FakeControllerProvider


@dataclass
class _FakeIrisConfig:
    """Stubs the bits of ``IrisConfig`` the plugin reads."""

    controller_address: str

    @property
    def proto(self):
        # The plugin only reads .proto.controller and hands it to
        # discover_controller, which our fake provider ignores.
        class _P:
            controller = object()

        return _P()

    def provider_bundle(self) -> _FakeBundle:
        return _FakeBundle(controller=_FakeControllerProvider(self.controller_address))


@pytest.fixture
def patch_resolver(monkeypatch):
    """Replace load_cluster_config + ControllerServiceClientSync with stubs."""
    cluster_calls: list[str] = []
    client_addresses: list[str] = []

    def _install(
        *, controller_address: str = "controller.example.com:10000", endpoints: dict[str, str] | None = None
    ) -> _FakeControllerClient:
        endpoints = endpoints or {}
        fake = _FakeControllerClient(endpoints)

        def _fake_load(name: str) -> _FakeIrisConfig:
            cluster_calls.append(name)
            return _FakeIrisConfig(controller_address=controller_address)

        def _fake_client(address: str) -> _FakeControllerClient:
            client_addresses.append(address)
            fake.address = address
            return fake

        monkeypatch.setattr(resolver_plugin, "load_cluster_config", _fake_load)
        monkeypatch.setattr(resolver_plugin, "ControllerServiceClientSync", _fake_client)
        return fake

    _install.cluster_calls = cluster_calls  # type: ignore[attr-defined]
    _install.client_addresses = client_addresses  # type: ignore[attr-defined]
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
