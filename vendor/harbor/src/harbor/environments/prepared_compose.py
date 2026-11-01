# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate prepared Harbor Compose tasks into named Shellbox containers."""

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from harbor.models.task.config import EnvironmentConfig, TaskOS
from harbor.utils.env import resolve_env_vars, validate_task_environment
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy
from shellbox.task import (
    ContainerSpec,
    ReadinessProbe,
    ServiceEndpoint,
    TaskSpec,
)

COMPOSE_FILENAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)
PREPARED_IMAGES_FILE = "shellbox-images.json"
PINNED_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_SERVICE_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")
_DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)")
_LOG_VOLUME = "${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}"


class PreparedImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    workdir: str
    agent_logs: Literal["publish"] | None = None


class Healthcheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test: list[str] | str
    interval: str = "30s"
    timeout: str = "30s"
    retries: int = Field(default=3, gt=0)
    start_period: str = "0s"


class BuildRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: str
    dockerfile: str = "Dockerfile"
    args: dict[str, str] = Field(default_factory=dict)


class ComposeService(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str | None = None
    build: str | BuildRecipe | None = None
    working_dir: str | None = None
    environment: dict[str, str | int | float] | list[str] = Field(default_factory=dict)
    command: list[str] | str | None = None
    entrypoint: list[str] | str | None = None
    expose: list[str | int] = Field(default_factory=list)
    depends_on: dict[str, dict[str, str]] | list[str] = Field(default_factory=dict)
    healthcheck: Healthcheck | None = None
    volumes: list[str] = Field(default_factory=list)


class ComposeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    services: dict[str, ComposeService] = Field(default_factory=dict)


@dataclass(frozen=True)
class ImportedTask:
    task: TaskSpec
    service_log_paths: tuple[tuple[str, str], ...]


def _duration(value: str) -> float:
    match = _DURATION.fullmatch(value)
    if match is None:
        raise ValueError("Compose healthcheck duration must specify ms, s, m, or h")
    return float(match[1]) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[match[2]]


def _argv(value: list[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    arguments = tuple(shlex.split(value)) if isinstance(value, str) else tuple(value)
    if not arguments:
        raise ValueError("Empty Compose command or entrypoint overrides are unsupported")
    if any("${" in item for item in arguments):
        raise ValueError("Compose commands cannot interpolate harness variables")
    return tuple(item.replace("$$", "$") for item in arguments)


def _environment(value: dict[str, str | int | float] | list[str]) -> dict[str, str]:
    if isinstance(value, list):
        if any("=" not in item for item in value):
            raise ValueError("Compose environment entries must have explicit values")
        values = dict(item.split("=", 1) for item in value)
    else:
        values = {key: str(item) for key, item in value.items()}
    validate_task_environment(values)
    return resolve_env_vars(values)


def _prepared_images(environment_dir: Path) -> dict[str, PreparedImage]:
    path = environment_dir / PREPARED_IMAGES_FILE
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("shellbox-images.json must map service names to prepared image references and workdirs")
    return {name: PreparedImage.model_validate(value) for name, value in document.items()}


def _machine(
    name: str,
    service: ComposeService,
    prepared: Mapping[str, PreparedImage],
    task: EnvironmentConfig,
) -> MachineSpec:
    image = prepared.get(name)
    image_reference = image.image if image is not None else service.image
    workdir = service.working_dir or (image.workdir if image is not None else None)
    if name == "main":
        image_reference = image_reference or task.docker_image
        workdir = workdir or task.workdir
    if image_reference is None or PINNED_IMAGE.fullmatch(image_reference) is None:
        raise ValueError(f"Service {name!r} requires a prepared image pinned with @sha256 in shellbox-images.json")
    if not workdir or not PurePosixPath(workdir).is_absolute():
        raise ValueError(f"Service {name!r} requires its prepared image's absolute working directory")
    if service.build is not None and image is None:
        raise ValueError(f"Service {name!r} has a build recipe; prepare its image before task execution")
    if image is not None and service.image is not None and service.image != image.image:
        raise ValueError(f"Service {name!r} image conflicts with its prepared image reference")
    if name == "main" and task.docker_image is not None and task.docker_image != image_reference:
        raise ValueError("The task docker_image conflicts with the prepared main image")
    return MachineSpec(
        source=RegistryImage(image_reference),
        workdir=workdir,
        network=NetworkPolicy.DENY,
        env=(
            {**resolve_env_vars(task.env), **_environment(service.environment)}
            if name == "main"
            else _environment(service.environment)
        ),
        cpus=task.cpus,
        memory_mb=task.memory_mb,
        storage_mb=task.storage_mb,
    )


def _readiness(healthcheck: Healthcheck | None) -> ReadinessProbe | None:
    if healthcheck is None:
        return None
    test = healthcheck.test
    if isinstance(test, str):
        command = ("sh", "-c", test.replace("$$", "$"))
    elif len(test) > 1 and test[0] == "CMD":
        command = tuple(test[1:])
    elif len(test) == 2 and test[0] == "CMD-SHELL":
        command = ("sh", "-c", test[1].replace("$$", "$"))
    else:
        raise ValueError("Compose healthcheck must use CMD or CMD-SHELL")
    if any("${" in item for item in command):
        raise ValueError("Compose healthchecks cannot interpolate harness variables")
    interval = _duration(healthcheck.interval)
    timeout = _duration(healthcheck.timeout)
    start_period = _duration(healthcheck.start_period)
    if interval <= 0 or timeout <= 0:
        raise ValueError("Compose healthcheck interval and timeout must be positive")
    return ReadinessProbe(
        command=Command(command, timeout=timeout),
        timeout=start_period + healthcheck.retries * (timeout + interval),
        interval=interval,
        start_period=start_period,
        attempts=healthcheck.retries,
    )


def import_prepared_task(task: EnvironmentConfig, environment_dir: Path) -> ImportedTask:
    """Translate the supported Compose subset without building images or reading host variables."""
    if task.allow_internet:
        raise ValueError("Prepared service groups require allow_internet=false")
    if task.os is not TaskOS.LINUX:
        raise ValueError("Prepared service groups require Linux images")
    if task.gpus or task.gpu_types or task.tpu is not None:
        raise ValueError("Prepared task sandboxes do not support GPU or TPU allocations")
    validate_task_environment(task.env)
    compose_paths = [environment_dir / name for name in COMPOSE_FILENAMES if (environment_dir / name).exists()]
    if len(compose_paths) > 1:
        raise ValueError("Prepared tasks accept one Compose file per task")
    prepared = _prepared_images(environment_dir)
    content = yaml.safe_load(compose_paths[0].read_text()) if compose_paths else {}
    document = ComposeDocument.model_validate({} if content is None else content)
    if "main" not in document.services:
        document.services["main"] = ComposeService()
    if set(prepared) - document.services.keys():
        raise ValueError("shellbox-images.json contains images for undeclared services")
    for name in document.services:
        if _SERVICE_NAME.fullmatch(name) is None:
            raise ValueError("Compose service names must be lowercase DNS labels")
    main = document.services["main"]
    if main.command is not None or main.entrypoint is not None or main.healthcheck is not None or main.expose:
        raise ValueError("The actor main service cannot override its idle process or publish endpoints")
    dependencies = (
        {name: {"condition": "service_started"} for name in main.depends_on}
        if isinstance(main.depends_on, list)
        else main.depends_on
    )
    for name, dependency in dependencies.items():
        if name == "main" or name not in document.services:
            raise ValueError("Compose main depends_on references an undeclared service")
        if dependency.keys() != {"condition"} or dependency["condition"] not in {
            "service_started",
            "service_healthy",
        }:
            raise ValueError("Compose depends_on supports service_started or service_healthy only")
        if dependency["condition"] == "service_healthy" and document.services[name].healthcheck is None:
            raise ValueError("Compose service_healthy dependencies require a healthcheck")
    services = []
    log_paths = []
    for name, service in document.services.items():
        if service.volumes:
            if service.volumes != [_LOG_VOLUME]:
                raise ValueError(
                    "Prepared tasks support only Harbor's agent-log volume, collected through the control plane"
                )
            if name != "main":
                if name not in prepared or prepared[name].agent_logs != "publish":
                    raise ValueError(
                        "Prepared sidecar images must declare agent_logs='publish' to lower Harbor's log bind"
                    )
                log_paths.append((name, "/logs/agent"))
        if name == "main":
            continue
        if service.depends_on:
            raise ValueError("Inter-service Compose startup dependencies are not yet supported")
        ports = []
        for port in service.expose:
            text = str(port)
            if not text.isdecimal() or not 0 < int(text) < 65536:
                raise ValueError("Compose expose must declare a TCP port number")
            ports.append(ServiceEndpoint(f"tcp-{text}", int(text)))
        services.append(
            ContainerSpec(
                name=name,
                machine=_machine(name, service, prepared, task),
                endpoints=tuple(ports),
                readiness=_readiness(service.healthcheck),
                entrypoint=_argv(service.entrypoint),
                arguments=_argv(service.command),
            )
        )
    names_and_ports = {(service.name, endpoint.port) for service in services for endpoint in service.endpoints}
    for server in task.mcp_servers:
        endpoint = urlsplit(server.url or "")
        if server.transport not in {"sse", "streamable-http"} or endpoint.scheme != "http":
            raise ValueError("Prepared task MCP servers must use HTTP within the service group")
        if endpoint.username or endpoint.password or (endpoint.hostname, endpoint.port or 80) not in names_and_ports:
            raise ValueError("Prepared task MCP endpoint must name a declared service and exposed port")
    return ImportedTask(
        task=TaskSpec(
            containers=(ContainerSpec("main", _machine("main", main, prepared, task)), *services),
            primary="main",
        ),
        service_log_paths=tuple(log_paths),
    )


def validate_prepared_task(task_env_config: EnvironmentConfig, environment_dir: Path) -> None:
    """Validate prepared images and task services without acquiring a sandbox."""
    import_prepared_task(task_env_config, environment_dir)
