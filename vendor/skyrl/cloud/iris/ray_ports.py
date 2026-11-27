# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray endpoints allocated by Iris for each task on a shared host."""

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

from rigging.port_lease import configured_tcp_port, lease_tcp_ports


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


@contextmanager
def resolve_ray_ports() -> Iterator[dict[RayPort, int]]:
    """Yield the task's Ray endpoint ports and hold leases for unassigned endpoints."""
    ports = {port: configured_tcp_port(port) for port in RayPort}
    pending = [port for port, value in ports.items() if not value]
    if not pending:
        yield ports
        return
    with lease_tcp_ports(len(pending), excluded=set(ports.values())) as values:
        ports.update(zip(pending, values, strict=True))
        yield ports
