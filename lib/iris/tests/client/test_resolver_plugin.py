# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the iris resolver plugin."""

import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

import iris.client  # noqa: F401  -- side-effect import: registers iris:// scheme
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceSync, ControllerServiceWSGIApplication
from rigging import resolver as resolver_module
from rigging.resolver import resolve
from rigging.timing import Duration, ExponentialBackoff


class _StubControllerService(ControllerServiceSync):
    """Minimal ``ControllerServiceSync`` that only implements ``list_endpoints``.

    Inherits the Protocol base class so all other RPCs default to
    ``UNIMPLEMENTED`` errors, exactly what we want for an isolated test.
    """

    def __init__(self) -> None:
        self.endpoints: dict[str, str] = {}

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        results: list[controller_pb2.Controller.Endpoint] = []
        for name, address in self.endpoints.items():
            if request.exact:
                if name == request.prefix:
                    results.append(controller_pb2.Controller.Endpoint(name=name, address=address))
            else:
                if name.startswith(request.prefix):
                    results.append(controller_pb2.Controller.Endpoint(name=name, address=address))
        return controller_pb2.Controller.ListEndpointsResponse(endpoints=results)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app(service: _StubControllerService) -> Starlette:
    wsgi = ControllerServiceWSGIApplication(service=service)
    return Starlette(routes=[Mount(wsgi.path, app=WSGIMiddleware(wsgi))])


class _BackgroundServer:
    def __init__(self, app: Starlette, port: int) -> None:
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            log_config=None,
            timeout_keep_alive=5,
        )
        self.server = uvicorn.Server(config)
        self.port = port
        self._thread = threading.Thread(
            target=self.server.run,
            name=f"resolver-plugin-test-{port}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        started = ExponentialBackoff(initial=0.01, maximum=0.2).wait_until(
            lambda: self.server.started,
            timeout=Duration.from_seconds(5.0),
        )
        if not started:
            raise RuntimeError(f"uvicorn did not start within 5s on port {self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5.0)


@pytest.fixture
def stub_controller() -> Iterator[tuple[_StubControllerService, int]]:
    svc = _StubControllerService()
    port = _free_port()
    bg = _BackgroundServer(_build_app(svc), port)
    bg.start()
    try:
        yield svc, port
    finally:
        bg.stop()


def test_resolve_iris_round_trips(monkeypatch, stub_controller):
    svc, controller_port = stub_controller
    svc.endpoints["/system/x"] = "host.example.com:1234"

    captured: list[tuple] = []

    def _fake_vm_address(name: str, provider: str) -> tuple[str, int]:
        captured.append((name, provider))
        # Direct test traffic at the in-process stub rather than GCP.
        return ("127.0.0.1", controller_port)

    # The plugin binds vm_address as a module-global at import time; patch
    # it where it's looked up.
    from iris.client import resolver_plugin

    monkeypatch.setattr(resolver_plugin, "vm_address", _fake_vm_address)

    host, port = resolve("iris://marin?endpoint=/system/x")
    assert (host, port) == ("host.example.com", 1234)
    assert captured == [("iris-controller-marin", "gcp")]


def test_resolve_iris_not_found(monkeypatch, stub_controller):
    _svc, controller_port = stub_controller

    from iris.client import resolver_plugin

    monkeypatch.setattr(
        resolver_plugin,
        "vm_address",
        lambda name, provider: ("127.0.0.1", controller_port),
    )

    with pytest.raises(KeyError, match="iris endpoint not found"):
        resolve("iris://marin?endpoint=/system/missing")


def test_iris_scheme_registered_after_iris_client_import():
    # Sanity check: importing iris.client (done at the top of this module)
    # installs the iris:// handler in rigging.resolver's registry.
    assert "iris" in resolver_module._HANDLERS
