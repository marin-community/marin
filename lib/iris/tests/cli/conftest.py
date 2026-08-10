# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared stubs for the ``iris cluster start`` CLI tests."""

import json
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from click.testing import CliRunner, Result
from iris.cli import build as image_build
from iris.cli import cluster as cluster_cli
from iris.cluster.config import (
    ControllerVmConfig,
    DefaultsConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    WorkerConfig,
)
from rigging.provenance import Provenance

CONTROLLER_ADDRESS = "iris-controller-svc.iris.svc.cluster.local:10000"


class StubControllerHealth:
    """A real HTTP server standing in for a controller's ``/health`` route.

    ``unhealthy_probes`` answers 503 that many times before reporting healthy,
    reproducing the window where a just-rolled controller is not yet reachable
    over the tunnel clients use.
    """

    def __init__(self, unhealthy_probes: int = 0) -> None:
        self.unhealthy_probes = unhealthy_probes
        self.probes = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/health":
                    self.send_error(404)
                    return
                stub.probes += 1
                healthy = stub.probes > stub.unhealthy_probes
                body = json.dumps({"status": "ok" if healthy else "unhealthy"}).encode()
                self.send_response(200 if healthy else 503)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args) -> None:
                """Silence the stdlib handler's per-request stderr logging."""

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class StubControllerProvider:
    """Stands in for a platform's controller provider.

    Records the configs it was asked to start and tunnels to a fixed URL, so a
    test controls what ``cluster start`` finds when it probes for reachability.
    ``docker_commands`` collects the image builds ``run_cluster_start`` intercepts.
    """

    def __init__(self, tunnel_url: str, address: str = CONTROLLER_ADDRESS) -> None:
        self.tunnel_url = tunnel_url
        self.address = address
        self.started: list[IrisClusterConfig] = []
        self.docker_commands: list[list[str]] = []

    def start_controller(self, config: IrisClusterConfig, *, fresh: bool = False) -> str:
        self.started.append(config)
        return self.address

    def tunnel(self, address: str, local_port: int | None = None):
        return nullcontext(self.tunnel_url)


@pytest.fixture
def stub_controller_health():
    """Factory for a ``StubControllerHealth``; every stub is closed at teardown."""
    stubs: list[StubControllerHealth] = []

    def make(unhealthy_probes: int = 0) -> StubControllerHealth:
        stub = StubControllerHealth(unhealthy_probes)
        stubs.append(stub)
        return stub

    yield make
    for stub in stubs:
        stub.close()


@pytest.fixture
def stub_controller_provider():
    """Factory for a ``StubControllerProvider`` whose tunnel reaches *tunnel_url*."""

    def make(tunnel_url: str) -> StubControllerProvider:
        return StubControllerProvider(tunnel_url)

    return make


@pytest.fixture
def kubernetes_cluster_config() -> IrisClusterConfig:
    """A Kubernetes-runtime config whose images all carry the ``latest`` tag."""
    return IrisClusterConfig(
        controller=ControllerVmConfig(image="ghcr.io/marin-community/iris-controller:latest"),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                docker_image="ghcr.io/marin-community/iris-worker:latest",
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(),
    )


@pytest.fixture
def run_cluster_start(monkeypatch):
    """Factory invoking ``iris cluster start`` against a stub controller provider.

    Pins git provenance and records the image-build commands on the provider
    instead of shelling out to docker.
    """

    def run(
        provider: StubControllerProvider,
        config: IrisClusterConfig,
        *,
        dirty: bool = False,
        args: Sequence[str] = (),
    ) -> Result:
        def docker_run(command, **_kwargs):
            provider.docker_commands.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            cluster_cli,
            "get_git_provenance",
            lambda: Provenance(
                tree_hash="abc1234", base_commit="abc1234", dirty=dirty, branch="feature", built_by="tester"
            ),
        )
        monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=provider))
        monkeypatch.setattr(image_build.subprocess, "run", docker_run)
        return CliRunner().invoke(cluster_cli.cluster_start, list(args), obj={"config": config, "verbose": False})

    return run
