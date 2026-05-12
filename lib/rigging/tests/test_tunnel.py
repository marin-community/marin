# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for rigging.tunnel.

The watchdog and listener-wait paths take real subprocesses in production
(``gcloud``/``kubectl``), but the tunnel module accepts an injectable
``spawn`` callable. Tests substitute a fake that launches a local
``python -m http.server`` so we can exercise listener-readiness, normal
shutdown, and watchdog respawn against a real TCP listener without
depending on cloud CLIs.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest
from rigging.tunnel import (
    GcpSshForwardTarget,
    K8sPortForwardTarget,
    _build_argv,
    describe_target,
    open_tunnel,
)


def test_build_argv_gcp_ssh_forward():
    target = GcpSshForwardTarget(
        project="my-proj",
        zone="us-central1-a",
        instance="vm-1",
        port=10001,
    )
    argv = _build_argv(target, local_port=55555)
    assert argv == [
        "gcloud",
        "compute",
        "ssh",
        "vm-1",
        "--project=my-proj",
        "--zone=us-central1-a",
        "--ssh-flag=-L55555:localhost:10001",
        "--ssh-flag=-N",
        "--ssh-flag=-T",
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oExitOnForwardFailure=yes",
    ]


def test_build_argv_gcp_ssh_with_impersonation_and_iap():
    target = GcpSshForwardTarget(
        project="p",
        zone="z",
        instance="vm",
        port=1,
        impersonate_service_account="sa@p.iam.gserviceaccount.com",
        tunnel_through_iap=True,
    )
    argv = _build_argv(target, local_port=2)
    assert "--tunnel-through-iap" in argv
    assert argv[-1] == "--impersonate-service-account=sa@p.iam.gserviceaccount.com"


def test_build_argv_k8s_port_forward():
    target = K8sPortForwardTarget(namespace="iris", service="finelog", port=10001)
    argv = _build_argv(target, local_port=44444)
    assert argv == [
        "kubectl",
        "port-forward",
        "svc/finelog",
        "44444:10001",
        "-n",
        "iris",
    ]


def test_build_argv_k8s_with_kubeconfig_and_context():
    target = K8sPortForwardTarget(
        namespace="default",
        service="svc",
        port=80,
        kubeconfig="/tmp/kc",
        context="prod",
    )
    argv = _build_argv(target, local_port=8080)
    assert argv[:5] == ["kubectl", "--kubeconfig", "/tmp/kc", "--context", "prod"]


def test_describe_target_round_trip_label():
    assert describe_target(GcpSshForwardTarget(project="p", zone="z", instance="vm", port=1)) == "ssh://p/z/vm:1"
    assert describe_target(K8sPortForwardTarget(namespace="n", service="s", port=2)) == "k8s://n/s:2"


def _http_server_spawn(local_port: int) -> subprocess.Popen[str]:
    """Substitute spawn that runs a local HTTP listener on ``local_port``."""
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(local_port), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _make_spawn(local_port: int):
    def spawn(_argv: list[str]) -> subprocess.Popen[str]:
        return _http_server_spawn(local_port)

    return spawn


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_open_tunnel_yields_reachable_url():
    """open_tunnel yields only after the local port accepts connections."""
    target = K8sPortForwardTarget(namespace="n", service="s", port=1)
    port = _free_port()
    with open_tunnel(target, local_port=port, spawn=_make_spawn(port), watchdog=False, timeout=15) as url:
        assert url == f"http://127.0.0.1:{port}"
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass


def test_open_tunnel_terminates_child_on_exit():
    target = K8sPortForwardTarget(namespace="n", service="s", port=1)
    port = _free_port()
    procs: list[subprocess.Popen[str]] = []

    def spawn(_argv: list[str]) -> subprocess.Popen[str]:
        p = _http_server_spawn(port)
        procs.append(p)
        return p

    with open_tunnel(target, local_port=port, spawn=spawn, watchdog=False, timeout=15):
        pass

    assert procs, "spawn was never invoked"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if procs[-1].poll() is not None:
            break
        time.sleep(0.05)
    assert procs[-1].poll() is not None, "child subprocess outlived the context"


def test_watchdog_respawns_after_child_dies():
    """Killing the child mid-context triggers a watchdog respawn that re-binds."""
    target = K8sPortForwardTarget(namespace="n", service="s", port=1)
    port = _free_port()
    spawned: list[subprocess.Popen[str]] = []
    lock = threading.Lock()

    def spawn(_argv: list[str]) -> subprocess.Popen[str]:
        p = _http_server_spawn(port)
        with lock:
            spawned.append(p)
        return p

    with open_tunnel(target, local_port=port, spawn=spawn, watchdog=True, timeout=15) as url:
        # Kill the initial child; watchdog should respawn another listener
        # on the same port within a few seconds.
        with lock:
            first = spawned[0]
        first.kill()
        first.wait()

        # Wait for a second spawn AND for the new listener to bind.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with lock:
                count = len(spawned)
            if count >= 2:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    pass
            time.sleep(0.2)
        else:
            pytest.fail(f"watchdog did not respawn (spawned={len(spawned)})")

        # Sanity: yielded URL did not change.
        assert url == f"http://127.0.0.1:{port}"


def test_stderr_is_forwarded_through_logger(caplog):
    """The child's stderr lines land on rigging.tunnel's logger, not the terminal."""
    target = K8sPortForwardTarget(namespace="n", service="s", port=1)
    port = _free_port()

    def spawn(_argv: list[str]) -> subprocess.Popen[str]:
        # First write a recognizable line on stderr, *then* start a listener
        # so open_tunnel's readiness check passes after the line is pumped.
        script = (
            f"import sys, http.server, socketserver;"
            f"sys.stderr.write('TUNNEL-STDERR-MARKER\\n'); sys.stderr.flush();"
            f"socketserver.TCPServer.allow_reuse_address = True;"
            f"socketserver.TCPServer(('127.0.0.1', {port}), http.server.BaseHTTPRequestHandler).serve_forever()"
        )
        return subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    with caplog.at_level("INFO", logger="rigging.tunnel"):
        with open_tunnel(target, local_port=port, spawn=spawn, watchdog=False, timeout=15):
            # Give the pump thread a brief moment after readiness to drain.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not any("TUNNEL-STDERR-MARKER" in r.message for r in caplog.records):
                time.sleep(0.05)

    assert any("TUNNEL-STDERR-MARKER" in r.message for r in caplog.records), [r.message for r in caplog.records]


def test_open_tunnel_raises_when_listener_never_opens():
    """If the child exits immediately without binding, open_tunnel times out."""
    target = K8sPortForwardTarget(namespace="n", service="s", port=1)

    def spawn(_argv: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    with pytest.raises(RuntimeError, match="did not open local port"):
        with open_tunnel(target, spawn=spawn, watchdog=False, timeout=2):
            pass


@pytest.fixture(autouse=True)
def _bound_test_runtime() -> Iterator[None]:
    """Cap each test's wall-clock budget — tunnel tests can hang on bugs."""
    yield
