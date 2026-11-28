# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import socket
import sys
from unittest.mock import patch

import pytest

from cloud.iris import ray_ports
from cloud.iris.ray_ports import RayPort, allocated_ray_port, resolve_ray_ports


def _hold_placeholder(port, candidates, pipe):
    os.environ[f"IRIS_PORT_{port.value.upper()}"] = "0"
    with patch.object(ray_ports, "_service_port_candidates", return_value=candidates), resolve_ray_ports():
        pipe.send(allocated_ray_port(port))
        pipe.recv()


@pytest.mark.skipif(sys.platform != "linux", reason="Kubernetes host-network ports use Linux network namespaces")
def test_concurrent_launchers_lease_unbound_service_ports_until_exit(monkeypatch, iris_ray_ports):
    # Force two independent launchers to try the same candidate before either
    # Ray service binds. The kernel assigns the temporary test ports uniquely.
    with socket.socket() as first, socket.socket() as second:
        first.bind(("", 0))
        second.bind(("", 0))
        candidates = [first.getsockname()[1], second.getsockname()[1]]
    monkeypatch.setattr(ray_ports, "_service_port_candidates", lambda reserved: candidates)
    ctx = multiprocessing.get_context("spawn")
    children = []
    try:
        for role, expected in zip((RayPort.METRICS, RayPort.RUNTIME_ENV_AGENT), candidates, strict=True):
            receiver, sender = ctx.Pipe()
            child = ctx.Process(target=_hold_placeholder, args=(role, candidates, sender))
            child.start()
            sender.close()
            children.append((child, receiver))
            assert receiver.poll(10)
            assert receiver.recv() == expected

        owner, _ = children[0]
        owner.kill()
        owner.join(10)
        assert owner.exitcode is not None
        monkeypatch.setenv("IRIS_PORT_RAY_METRICS", "0")
        with resolve_ray_ports():
            assert allocated_ray_port(RayPort.METRICS) == candidates[0]
    finally:
        for child, receiver in children:
            if child.is_alive():
                child.kill()
            child.join(10)
            receiver.close()
