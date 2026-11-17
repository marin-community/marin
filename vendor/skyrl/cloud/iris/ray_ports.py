# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray endpoints allocated by Iris for each task on a shared host."""

import os
from enum import StrEnum

from iris.cluster.platforms.types import find_free_port


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


def allocated_ray_port(port: RayPort) -> int:
    """Return the task's required scheduler allocation for a Ray endpoint."""
    return int(os.environ[f"IRIS_PORT_{port.value.upper()}"])


def resolve_ray_ports() -> None:
    """Resolve Iris's task-allocated port placeholders before Ray bootstrap."""
    ports = {port: allocated_ray_port(port) for port in RayPort}
    reserved = set(ports.values()) - {0}
    for port, value in ports.items():
        if value:
            continue
        # Kubernetes supplies zero placeholders; worker-backed tasks supply
        # concrete allocations. Use the same kernel selection as Iris's JAX launcher.
        value = find_free_port()
        while value in reserved:
            value = find_free_port()
        reserved.add(value)
        os.environ[f"IRIS_PORT_{port.value.upper()}"] = str(value)
