# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the iris:// resolver plugin."""

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


@pytest.fixture
def patch_resolver(monkeypatch):
    """Replace gcp_vm_address + controller client with in-process stubs."""
    vm_calls: list[str] = []

    def _install(endpoints: dict[str, str]) -> _FakeControllerClient:
        fake = _FakeControllerClient(endpoints)

        def _fake_vm_address(name: str, *, port: int = 10002) -> tuple[str, int]:
            vm_calls.append(name)
            return ("127.0.0.1", 65000)

        monkeypatch.setattr(resolver_plugin, "gcp_vm_address", _fake_vm_address)
        monkeypatch.setattr(
            resolver_plugin,
            "ControllerServiceClientSync",
            lambda address: fake,
        )
        return fake

    _install.vm_calls = vm_calls  # type: ignore[attr-defined]
    return _install


def test_resolve_iris_returns_endpoint_address(patch_resolver):
    patch_resolver({"/system/x": "host.example.com:1234"})
    assert resolve("iris://marin?endpoint=/system/x") == ("host.example.com", 1234)
    assert patch_resolver.vm_calls == ["iris-controller-marin"]


def test_resolve_iris_not_found_raises(patch_resolver):
    patch_resolver({})
    with pytest.raises(KeyError, match="iris endpoint not found"):
        resolve("iris://marin?endpoint=/system/missing")


def test_resolve_iris_requires_endpoint_query(patch_resolver):
    patch_resolver({})
    with pytest.raises(ValueError, match="requires \\?endpoint="):
        resolve("iris://marin")


def test_resolve_iris_rejects_port(patch_resolver):
    patch_resolver({})
    with pytest.raises(ValueError, match="cannot have a port"):
        resolve("iris://marin:9000?endpoint=/x")


def test_iris_scheme_registered_after_iris_client_import():
    assert is_registered("iris")
