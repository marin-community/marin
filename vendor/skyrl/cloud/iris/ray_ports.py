# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray endpoints allocated by Iris for each task on a shared host."""

import errno
import os
import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from enum import StrEnum
from itertools import islice
from pathlib import Path
from random import SystemRandom


class RayPort(StrEnum):
    HEAD = "ray_head"
    DASHBOARD = "ray_dashboard"
    CLIENT = "ray_client"
    METRICS = "ray_metrics"
    RUNTIME_ENV_AGENT = "ray_runtime_env_agent"
    DASHBOARD_AGENT_GRPC = "ray_dashboard_agent_grpc"
    DASHBOARD_AGENT_LISTEN = "ray_dashboard_agent_listen"
    NODE_MANAGER = "ray_node_manager"
    OBJECT_MANAGER = "ray_object_manager"


# Ray 2.51 passes zero through to these listeners. Its remaining services either
# reject zero or select and close a temporary socket before starting the service.
_NATIVE_ZERO_PORTS = frozenset((RayPort.DASHBOARD, RayPort.NODE_MANAGER, RayPort.OBJECT_MANAGER))
_MAX_CANDIDATES_PER_PORT = 64


def _port_env(port: RayPort) -> str:
    return f"IRIS_PORT_{port.value.upper()}"


def allocated_ray_port(port: RayPort) -> int:
    """Return the task's required scheduler allocation for a Ray endpoint."""
    return int(os.environ[_port_env(port)])


def _service_port_candidates(reserved: set[int]) -> list[int]:
    # A native bind(0), including a Ray worker, must not take a leased number
    # during the handoff between our TCP probe and the actual server bind.
    network = Path("/proc/sys/net/ipv4")
    low, high = map(int, (network / "ip_local_port_range").read_text().split())
    unprivileged = max(1024, int((network / "ip_unprivileged_port_start").read_text()))
    unavailable = set(reserved)
    for item in (network / "ip_local_reserved_ports").read_text().strip().split(","):
        if item:
            start, _, end = item.partition("-")
            unavailable.update(range(int(start), int(end or start) + 1))
    available = [port for port in range(unprivileged, 65536) if not low <= port <= high and port not in unavailable]
    return SystemRandom().sample(available, min(len(available), len(RayPort) * _MAX_CANDIDATES_PER_PORT))


def _lease_service_port(port: RayPort, candidates: Iterator[int], leases: ExitStack) -> int:
    for value in islice(candidates, _MAX_CANDIDATES_PER_PORT):
        # Linux abstract Unix sockets share the TCP network namespace, including
        # across host-network pods with different mount/PID namespaces. The kernel
        # drops the claim if the launcher dies, even before Ray binds this port.
        lease = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            lease.bind(f"\0marin-skyrl-port-{value}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("", value))
        except OSError as error:
            lease.close()
            if error.errno in (errno.EADDRINUSE, errno.EACCES):
                continue
            raise
        leases.enter_context(lease)
        return value
    raise RuntimeError(
        f"Could not lease an unprivileged TCP port for {port.value} outside the kernel ephemeral range. "
        "Check host port availability and /proc/sys/net/ipv4/ip_local_port_range, or use explicit Iris allocations."
    )


@contextmanager
def resolve_ray_ports() -> Iterator[None]:
    """Hold leases around Ray bootstrap and execution for services requiring concrete ports."""
    ports = {port: allocated_ray_port(port) for port in RayPort}
    pending = [port for port, value in ports.items() if not value and port not in _NATIVE_ZERO_PORTS]
    with ExitStack() as leases:
        if pending:
            candidates = iter(_service_port_candidates(set(ports.values())))
            for port in pending:
                value = _lease_service_port(port, candidates, leases)
                key = _port_env(port)
                leases.callback(os.environ.__setitem__, key, "0")
                os.environ[key] = str(value)
        yield
