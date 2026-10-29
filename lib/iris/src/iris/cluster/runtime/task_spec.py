# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize Iris task requests into named containers independently of isolation policy."""

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass

from iris.cluster.runtime.env import WORKDIR_PATH, render_setup_steps
from iris.rpc import job_pb2

PRIMARY_CONTAINER = "task"
_CONTAINER_NAME = re.compile(r"[a-z](?:[a-z0-9-]*[a-z0-9])?")


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    image: str
    workdir: str
    resources: job_pb2.ResourceSpecProto
    env_vars: Mapping[str, str]
    entrypoint: tuple[str, ...] | None
    arguments: tuple[str, ...] | None


@dataclass(frozen=True)
class TaskSpec:
    containers: tuple[ContainerSpec, ...]
    primary: str

    @property
    def primary_container(self) -> ContainerSpec:
        return next(container for container in self.containers if container.name == self.primary)


def validate_container_names(environment: job_pb2.EnvironmentConfig) -> None:
    """Require unique DNS aliases, reserving the physical primary identity."""
    primary = environment.primary_container_name or PRIMARY_CONTAINER
    names = [primary, *(container.name for container in environment.sandbox_services)]
    if len(set(names)) != len(names):
        raise ValueError("Task container names must be unique")
    for name in names:
        if len(name) > 55 or _CONTAINER_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid task container name: {name!r}")
    if any(container.name == PRIMARY_CONTAINER for container in environment.sandbox_services):
        raise ValueError("The task container name is reserved for the primary")


def _primary_command(entrypoint: job_pb2.RuntimeEntrypoint) -> tuple[str, ...]:
    lines = ["set -e", "ulimit -c 0", "mkdir -p /app", "cd /app"]
    lines.extend(render_setup_steps(entrypoint.setup_commands))
    # Preserve the ordinary Iris workspace startup before executing its command.
    lines.append('[ -f "$IRIS_VENV/bin/activate" ] && source "$IRIS_VENV/bin/activate"')
    if entrypoint.run_command.argv:
        lines.append("exec " + shlex.join(entrypoint.run_command.argv))
    return ("bash", "-lc", "\n".join(lines))


def task_spec_from_request(request: job_pb2.RunTaskRequest, default_image: str) -> TaskSpec:
    """Normalize the current primary-plus-peers wire shape, including singleton tasks."""
    environment = request.environment
    validate_container_names(environment)
    primary = environment.primary_container_name or PRIMARY_CONTAINER
    resources = job_pb2.ResourceSpecProto()
    resources.CopyFrom(request.resources)
    for container in environment.sandbox_services:
        resources.cpu_millicores -= container.resources.cpu_millicores
        resources.memory_bytes -= container.resources.memory_bytes
        resources.disk_bytes -= container.resources.disk_bytes
    containers = [
        ContainerSpec(
            name=primary,
            image=request.task_image or default_image,
            workdir=WORKDIR_PATH,
            resources=resources,
            env_vars=dict(environment.env_vars),
            entrypoint=_primary_command(request.entrypoint),
            arguments=None,
        )
    ]
    for container in environment.sandbox_services:
        member_resources = job_pb2.ResourceSpecProto()
        member_resources.CopyFrom(container.resources)
        containers.append(
            ContainerSpec(
                name=container.name,
                image=container.image,
                workdir=container.workdir,
                resources=member_resources,
                env_vars=dict(container.env_vars),
                entrypoint=tuple(container.entrypoint.argv) if container.HasField("entrypoint") else None,
                arguments=tuple(container.arguments.argv) if container.HasField("arguments") else None,
            )
        )
    return TaskSpec(tuple(containers), primary)
