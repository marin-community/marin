# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray endpoints allocated by Iris for each task on a shared host."""

from collections.abc import Iterable
from enum import StrEnum

from rigging.port_lease import allocate_tcp_port, configured_tcp_port


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


RAY_WORKER_PORTS = tuple(port for port in RayPort if port not in (RayPort.HEAD, RayPort.DASHBOARD, RayPort.CLIENT))


def resolve_ray_ports(names: Iterable[RayPort] = RayPort) -> dict[RayPort, int]:
    """Resolve the requested Ray endpoints, claiming unassigned ports until exit."""
    ports = {port: configured_tcp_port(port) for port in names}
    assigned = {port for port in ports.values() if port}
    for port, value in ports.items():
        if not value:
            ports[port] = allocate_tcp_port(excluded=assigned)
    return ports
