# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical named containers and the primary projection consumed by worker runtimes."""

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from iris.cluster.runtime.env import WORKDIR_PATH, render_setup_steps
from iris.cluster.runtime.sandbox import validate_sandbox_environment
from iris.rpc import controller_pb2, job_pb2

PRIMARY_CONTAINER = "task"
# Keep existing Pod container identities stable across specification renames.
COMPANION_CONTAINER_PREFIX = "service-"
TASK_CONTAINERS_CAPABILITY = "task-containers"
POD_NETWORK_CAPABILITY = "pod-network"
_CONTAINER_NAME = re.compile(r"[a-z](?:[a-z0-9-]*[a-z0-9])?")
_PINNED_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def required_task_capabilities(environment: job_pb2.EnvironmentConfig) -> frozenset[str]:
    """Execution features a destination must retain when accepting this task."""
    capabilities = {TASK_CONTAINERS_CAPABILITY} if len(environment.containers) > 1 else set()
    if environment.network_attachment == job_pb2.NETWORK_ATTACHMENT_POD or environment.HasField("driver_connection"):
        capabilities.add(POD_NETWORK_CAPABILITY)
    return frozenset(capabilities)


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


def primary_container(environment: job_pb2.EnvironmentConfig) -> job_pb2.ContainerSpec:
    for container in environment.containers:
        if container.name == environment.primary_container:
            return container
    raise ValueError("The primary container must belong to the task")


def aggregate_resources(containers: Sequence[job_pb2.ContainerSpec]) -> job_pb2.ResourceSpecProto:
    """Sum reservations without inventing a residual primary allocation."""
    total = job_pb2.ResourceSpecProto()
    for container in containers:
        resources = container.resources
        total.cpu_millicores += resources.cpu_millicores
        total.memory_bytes += resources.memory_bytes
        total.disk_bytes += resources.disk_bytes
        if resources.HasField("device") and resources.device.WhichOneof("device") not in (None, "cpu"):
            if total.device.WhichOneof("device") not in (None, "cpu"):
                raise ValueError("Multiple accelerator containers are not supported")
            total.device.CopyFrom(resources.device)
        elif resources.HasField("device") and not total.HasField("device"):
            total.device.CopyFrom(resources.device)
    return total


def validate_container_names(environment: job_pb2.EnvironmentConfig) -> None:
    names = [container.name for container in environment.containers]
    if not names or len(set(names)) != len(names):
        raise ValueError("Task containers must be nonempty and have unique names")
    if environment.primary_container not in names:
        raise ValueError("The primary container must belong to the task")
    for name in names:
        if len(name) > 55 or _CONTAINER_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid task container name: {name!r}")
        if name == PRIMARY_CONTAINER and name != environment.primary_container:
            raise ValueError("The task container name is reserved for the primary")


def normalize_task_request(request: controller_pb2.Controller.LaunchJobRequest | job_pb2.RunTaskRequest) -> None:
    """Promote legacy singleton inputs once, then derive flat worker fields."""
    environment = request.environment
    explicit = bool(environment.containers)
    if not environment.containers:
        container = environment.containers.add(
            name=PRIMARY_CONTAINER,
            image=request.task_image,
            workdir=WORKDIR_PATH,
            resources=request.resources,
            env_vars=environment.env_vars,
            secrets=environment.secrets,
            setup_scripts=environment.setup_scripts,
        )
        container.runtime_entrypoint.CopyFrom(request.entrypoint)
        environment.primary_container = PRIMARY_CONTAINER
    validate_container_names(environment)
    primary = primary_container(environment)
    resources = aggregate_resources(environment.containers)
    if explicit:
        projections = (
            (bool(environment.env_vars), dict(environment.env_vars) == dict(primary.env_vars), "environment variables"),
            (bool(environment.secrets), dict(environment.secrets) == dict(primary.secrets), "secrets"),
            (
                bool(environment.setup_scripts),
                list(environment.setup_scripts) == list(primary.setup_scripts),
                "setup scripts",
            ),
            (bool(request.task_image), request.task_image == primary.image, "image"),
            (bool(request.entrypoint.ByteSize()), request.entrypoint == primary.runtime_entrypoint, "entrypoint"),
            (
                bool(request.resources.ByteSize()),
                (
                    request.resources.cpu_millicores == resources.cpu_millicores
                    and request.resources.memory_bytes == resources.memory_bytes
                    and request.resources.disk_bytes == resources.disk_bytes
                    and request.resources.device == resources.device
                ),
                "aggregate resources",
            ),
        )
        for supplied, matches, name in projections:
            if supplied and not matches:
                raise ValueError(f"Named task containers conflict with legacy {name}")
    environment.env_vars.clear()
    environment.env_vars.update(primary.env_vars)
    environment.secrets.clear()
    environment.secrets.update(primary.secrets)
    environment.setup_scripts[:] = primary.setup_scripts
    request.entrypoint.Clear()
    if primary.HasField("runtime_entrypoint"):
        request.entrypoint.CopyFrom(primary.runtime_entrypoint)
    request.task_image = primary.image
    request.resources.CopyFrom(resources)


def validate_task_containers(profile: int, egress_policy: int, environment: job_pb2.EnvironmentConfig) -> None:
    """Validate generic containers, applying sandbox constraints only to SANDBOX."""
    validate_container_names(environment)
    if len(environment.containers) > 9:
        raise ValueError("Tasks support at most nine containers")
    sandbox = profile == job_pb2.CONTAINER_PROFILE_SANDBOX
    multiple = len(environment.containers) > 1
    if sandbox and multiple and egress_policy != job_pb2.EGRESS_POLICY_NONE:
        raise ValueError("Sandbox multi-container tasks require NONE egress")
    for container in environment.containers:
        primary = container.name == environment.primary_container
        if not primary and (
            container.secrets
            or container.setup_scripts
            or (
                container.HasField("runtime_entrypoint")
                and (
                    container.runtime_entrypoint.setup_commands
                    or container.runtime_entrypoint.workdir_files
                    or container.runtime_entrypoint.workdir_file_refs
                )
            )
        ):
            raise ValueError("Companion containers cannot receive secrets, setup scripts or workspace files")
        if container.WhichOneof("startup") is None:
            raise ValueError(f"Container {container.name} requires an explicit startup mode")
        if container.HasField("image_command"):
            for field in ("entrypoint", "arguments"):
                if container.image_command.HasField(field) and not getattr(container.image_command, field).argv:
                    raise ValueError("Empty image command overrides are unsupported")
        if multiple:
            if not PurePosixPath(container.workdir).is_absolute():
                raise ValueError(f"Container {container.name} requires an absolute workdir")
            if container.resources.cpu_millicores <= 0 or container.resources.memory_bytes <= 0:
                raise ValueError("Multi-container tasks require explicit positive CPU and memory reservations")
            if container.resources.disk_bytes < 0:
                raise ValueError("Container disk reservations must be nonnegative")
            if not primary and container.resources.device.WhichOneof("device") not in (None, "cpu"):
                raise ValueError("Companion accelerator containers are unsupported")
        if sandbox:
            validate_sandbox_environment(dict(container.env_vars))
            validate_sandbox_environment(dict.fromkeys(container.secrets, ""))
        if sandbox and multiple:
            if container.secrets or container.setup_scripts:
                raise ValueError("Sandbox multi-container tasks accept no secrets or environment setup scripts")
            if _PINNED_IMAGE.fullmatch(container.image) is None:
                raise ValueError(f"Sandbox container {container.name} requires a digest-pinned image")
            if container.resources.device.WhichOneof("device") not in (None, "cpu"):
                raise ValueError("Sandbox containers cannot request accelerator devices")


def _primary_command(entrypoint: job_pb2.RuntimeEntrypoint, workdir: str) -> tuple[str, ...]:
    quoted_workdir = shlex.quote(workdir)
    lines = ["set -e", "ulimit -c 0", f"mkdir -p {quoted_workdir}", f"cd {quoted_workdir}"]
    lines.extend(render_setup_steps(entrypoint.setup_commands))
    lines.append('[ -f "$IRIS_VENV/bin/activate" ] && source "$IRIS_VENV/bin/activate"')
    if entrypoint.run_command.argv:
        lines.append("exec " + shlex.join(entrypoint.run_command.argv))
    return ("bash", "-lc", "\n".join(lines))


def task_spec_from_request(request: job_pb2.RunTaskRequest, default_image: str) -> TaskSpec:
    """Render the canonical task; physical primary identity remains ``task``."""
    normalize_task_request(request)
    primary = request.environment.primary_container
    containers = []
    for container in sorted(request.environment.containers, key=lambda container: container.name != primary):
        if container.HasField("runtime_entrypoint"):
            entrypoint = _primary_command(container.runtime_entrypoint, container.workdir)
            arguments = None
        else:
            image_command = container.image_command
            entrypoint = tuple(image_command.entrypoint.argv) if image_command.HasField("entrypoint") else None
            arguments = tuple(image_command.arguments.argv) if image_command.HasField("arguments") else None
        containers.append(
            ContainerSpec(
                name=container.name,
                image=container.image or default_image,
                workdir=container.workdir,
                resources=container.resources,
                env_vars=dict(container.env_vars),
                entrypoint=entrypoint,
                arguments=arguments,
            )
        )
    return TaskSpec(tuple(containers), primary)
