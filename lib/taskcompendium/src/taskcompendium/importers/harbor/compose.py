# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a package's Compose file to sidecar service requirements.

This is the Compose subset Harbor's Shellbox translation accepts: the ``main`` service is the
agent's container and keeps its idle process; sidecars publish TCP endpoints and readiness
probes on the task's shared network. Host variables are not read; whole-value ``${NAME}``
templates stay in the environment for the runtime to resolve.

A package may also override ``main`` as Harbor's Docker backends expect. They merge the task's file
after their own base files and fill its ``${...}`` placeholders, so ``main`` settings that restate
Harbor's own are accepted and dropped: the ``${CONTEXT_DIR}`` build of ``environment/`` and its
``${MAIN_IMAGE_NAME}`` tag, the idle command, the log-directory volumes, ``TEST_DIR=${TEST_DIR}``,
``deploy`` limits of ``${CPUS}`` and ``${MEMORY}``, which restate ``task.toml``'s resources, and
the default network. ``network_mode: none`` denies the agent network access.
"""

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from taskcompendium.importers.harbor.config import McpServer, McpTransport, UnsupportedHarborFeature
from taskcompendium.importers.harbor.sources import (
    DOCKERFILE,
    BuildSource,
    ContainerSource,
    ImageSource,
    Pin,
    lower_source,
)
from taskcompendium.models import (
    SERVICE_NAME_PATTERN,
    EndpointRequirement,
    ProviderRequirement,
    ReadinessRequirement,
    ServiceRequirement,
)

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
MAIN_SERVICE = "main"
HARBOR_LOG_VOLUMES = frozenset(
    f"${{HOST_{name}_PATH}}:${{ENV_{name}_PATH}}" for name in ("AGENT_LOGS", "VERIFIER_LOGS", "ARTIFACTS")
)
"""Harbor's own mounts of ``/logs/agent``, ``/logs/verifier`` and ``/logs/artifacts``."""
HARBOR_BUILD_CONTEXTS = (".", "${CONTEXT_DIR}")
"""Harbor sets ``CONTEXT_DIR`` to ``environment/``, which ``.`` also names."""
HARBOR_MAIN_IMAGE = "${MAIN_IMAGE_NAME}"
"""The tag Harbor gives the image it builds for ``main``."""
HARBOR_IDLE_COMMAND = ("sh", "-c", "sleep infinity")
"""The process Harbor's base Compose files run in ``main``; the runtime keeps ``main`` idle too."""
HARBOR_TEMPLATE_ENVIRONMENT = {"TEST_DIR": "${TEST_DIR}"}
"""Harbor's main-service template passes ``TEST_DIR``, which Harbor never sets."""
HARBOR_RESOURCE_LIMITS = {"cpus": "${CPUS}", "memory": "${MEMORY}"}
"""Harbor fills these from ``task.toml``'s ``cpus`` and ``memory_mb``, which the episode inputs carry."""
DEFAULT_NETWORK_MODES = ("bridge", "${NETWORK_MODE:-bridge}")
"""Docker's default network; Harbor never sets ``NETWORK_MODE``."""
NO_NETWORK = "none"
MCP_INTERFACE = "mcp:v1"
"""Tool-provider interface for an MCP server; its initial state holds ``transport`` and ``url``."""

_DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)")
_WHOLE_TEMPLATE = re.compile(r"\$\{[^}:]+(?::-.*)?\}")


class ComposeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ComposeHealthcheck(ComposeModel):
    test: list[str] | str
    interval: str = "30s"
    timeout: str = "30s"
    retries: int = Field(default=3, gt=0)
    start_period: str = "0s"


class BuildRecipe(ComposeModel):
    context: str
    dockerfile: str = DOCKERFILE
    args: dict[str, str] = Field(default_factory=dict)


class ResourceLimits(ComposeModel):
    cpus: str | float | None = None
    memory: str | None = None


class DeployResources(ComposeModel):
    limits: ResourceLimits = Field(default_factory=ResourceLimits)


class Deploy(ComposeModel):
    resources: DeployResources = Field(default_factory=DeployResources)


class ComposeService(ComposeModel):
    image: str | None = None
    build: str | BuildRecipe | None = None
    working_dir: str | None = None
    environment: dict[str, str | int | float] | list[str] = Field(default_factory=dict)
    command: list[str] | str | None = None
    entrypoint: list[str] | str | None = None
    expose: list[str | int] = Field(default_factory=list)
    depends_on: dict[str, dict[str, str]] | list[str] = Field(default_factory=dict)
    healthcheck: ComposeHealthcheck | None = None
    volumes: list[str] = Field(default_factory=list)
    network_mode: str | None = None
    deploy: Deploy | None = None


class ComposeDocument(ComposeModel):
    version: str | None = None
    """Obsolete and ignored by Compose."""
    services: dict[str, ComposeService] = Field(default_factory=dict)


@dataclass(frozen=True)
class ComposeGroup:
    """The agent's ``main`` service settings and the sidecars declared beside it."""

    main_environment: dict[str, str]
    main_working_directory: str | None
    main_network_disabled: bool
    sidecars: dict[str, ComposeService]
    environment_dir: Path


def _duration(value: str) -> float:
    match = _DURATION.fullmatch(value)
    if match is None:
        raise UnsupportedHarborFeature(f"Compose duration {value!r} must be one number with ms, s, m, or h")
    return float(match[1]) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[match[2]]


def _argv(value: list[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    arguments = tuple(shlex.split(value)) if isinstance(value, str) else tuple(value)
    if not arguments:
        raise UnsupportedHarborFeature("empty Compose command or entrypoint overrides")
    if any("${" in item for item in arguments):
        raise UnsupportedHarborFeature("Compose commands that interpolate host variables")
    return tuple(item.replace("$$", "$") for item in arguments)


def _environment(value: dict[str, str | int | float] | list[str]) -> dict[str, str]:
    if isinstance(value, list):
        if any("=" not in item for item in value):
            raise UnsupportedHarborFeature("Compose environment entries passed through from the host")
        values = dict(item.split("=", 1) for item in value)
    else:
        values = {key: str(item) for key, item in value.items()}
    resolved = {}
    for key, item in values.items():
        if _WHOLE_TEMPLATE.fullmatch(item):
            resolved[key] = item
        elif "$" in item.replace("$$", ""):
            raise UnsupportedHarborFeature(f"Compose environment {key} interpolates part of its value")
        else:
            resolved[key] = item.replace("$$", "$")
    return resolved


def _readiness(healthcheck: ComposeHealthcheck | None) -> ReadinessRequirement | None:
    if healthcheck is None:
        return None
    test = healthcheck.test
    if isinstance(test, str):
        argv = ("sh", "-c", test.replace("$$", "$"))
    elif len(test) > 1 and test[0] == "CMD":
        argv = tuple(test[1:])
    elif len(test) == 2 and test[0] == "CMD-SHELL":
        argv = ("sh", "-c", test[1].replace("$$", "$"))
    else:
        raise UnsupportedHarborFeature("Compose healthchecks must use CMD or CMD-SHELL")
    if any("${" in item for item in argv):
        raise UnsupportedHarborFeature("Compose healthchecks that interpolate host variables")
    interval, timeout = _duration(healthcheck.interval), _duration(healthcheck.timeout)
    start_period = _duration(healthcheck.start_period)
    return ReadinessRequirement(
        argv=argv,
        command_timeout=timeout,
        timeout=start_period + healthcheck.retries * (timeout + interval),
        interval=interval,
        start_period=start_period,
        attempts=healthcheck.retries,
    )


def _check_main(main: ComposeService, services: dict[str, ComposeService]) -> None:
    if (
        _argv(main.command) not in (None, HARBOR_IDLE_COMMAND)
        or main.entrypoint is not None
        or main.healthcheck is not None
        or main.expose
    ):
        raise UnsupportedHarborFeature("the main Compose service cannot override its process or publish endpoints")
    if main.image not in (None, HARBOR_MAIN_IMAGE):
        raise UnsupportedHarborFeature("the main Compose service image comes from task.toml or environment/Dockerfile")
    builds = (*HARBOR_BUILD_CONTEXTS, *(BuildRecipe(context=context) for context in HARBOR_BUILD_CONTEXTS))
    if main.build is not None and main.build not in builds:
        raise UnsupportedHarborFeature("the main Compose service builds only environment/Dockerfile")
    if set(main.volumes) - HARBOR_LOG_VOLUMES:
        raise UnsupportedHarborFeature("Compose volumes other than Harbor's log-directory mounts")
    if main.deploy is not None and any(
        HARBOR_RESOURCE_LIMITS[name] != value
        for name, value in main.deploy.resources.limits.model_dump(exclude_none=True).items()
    ):
        raise UnsupportedHarborFeature("Compose resource limits other than Harbor's ${CPUS} and ${MEMORY}")
    if main.network_mode not in (None, *DEFAULT_NETWORK_MODES, NO_NETWORK):
        raise UnsupportedHarborFeature(f"Compose network_mode {main.network_mode!r}")
    if main.network_mode == NO_NETWORK and services.keys() - {MAIN_SERVICE}:
        raise UnsupportedHarborFeature("a main Compose service without a network cannot reach its sidecars")
    dependencies = (
        {name: {"condition": "service_started"} for name in main.depends_on}
        if isinstance(main.depends_on, list)
        else main.depends_on
    )
    for name, dependency in dependencies.items():
        if name == MAIN_SERVICE or name not in services:
            raise UnsupportedHarborFeature(f"the main Compose service depends on undeclared service {name!r}")
        condition = dependency.get("condition")
        if dependency.keys() != {"condition"} or condition not in ("service_started", "service_healthy"):
            raise UnsupportedHarborFeature("Compose depends_on supports service_started or service_healthy only")
        if condition == "service_healthy" and services[name].healthcheck is None:
            raise UnsupportedHarborFeature(f"Compose service {name!r} is awaited as healthy but has no healthcheck")


def read_compose(environment_dir: Path) -> ComposeGroup | None:
    """The package's Compose group, or ``None`` when ``environment/`` has no Compose file."""
    paths = [environment_dir / name for name in COMPOSE_FILENAMES if (environment_dir / name).is_file()]
    if not paths:
        return None
    if len(paths) > 1:
        raise UnsupportedHarborFeature("more than one Compose file")
    content = yaml.safe_load(paths[0].read_text())
    try:
        document = ComposeDocument.model_validate({} if content is None else content)
    except ValidationError as error:
        raise UnsupportedHarborFeature(f"unsupported Compose configuration: {error}") from error
    for name in document.services:
        if re.fullmatch(SERVICE_NAME_PATTERN, name) is None:
            raise UnsupportedHarborFeature(f"Compose service name {name!r} must match {SERVICE_NAME_PATTERN}")
    main = document.services.get(MAIN_SERVICE, ComposeService())
    _check_main(main, document.services)
    sidecars = {name: service for name, service in document.services.items() if name != MAIN_SERVICE}
    for name, service in sidecars.items():
        if service.volumes:
            raise UnsupportedHarborFeature(f"Compose service {name!r} mounts volumes")
        if service.depends_on:
            raise UnsupportedHarborFeature("startup dependencies between sidecar services")
        if service.network_mode is not None or service.deploy is not None:
            raise UnsupportedHarborFeature(f"Compose service {name!r} sets its own network_mode or deploy resources")
    environment = {
        name: value
        for name, value in _environment(main.environment).items()
        if HARBOR_TEMPLATE_ENVIRONMENT.get(name) != value
    }
    return ComposeGroup(environment, main.working_dir, main.network_mode == NO_NETWORK, sidecars, environment_dir)


def sidecar_source(group: ComposeGroup, name: str) -> ContainerSource:
    service = group.sidecars[name]
    if service.build is None:
        if service.image is None:
            raise UnsupportedHarborFeature(f"Compose service {name!r} declares neither an image nor a build")
        return ImageSource(service.image)
    recipe = BuildRecipe(context=service.build) if isinstance(service.build, str) else service.build
    if recipe.dockerfile != DOCKERFILE or recipe.args:
        raise UnsupportedHarborFeature(f"Compose service {name!r} builds with a renamed Dockerfile or build args")
    root = group.environment_dir.resolve()
    context = (root / recipe.context).resolve()
    if not context.is_relative_to(root) or not (context / DOCKERFILE).is_file():
        raise UnsupportedHarborFeature(f"Compose service {name!r} builds outside environment/ or without a Dockerfile")
    return BuildSource(context)


def lower_sidecars(group: ComposeGroup, pin: Pin) -> tuple[ServiceRequirement, ...]:
    services = []
    for name, service in group.sidecars.items():
        ports = []
        for port in service.expose:
            text = str(port)
            if not text.isdecimal() or not 0 < int(text) < 65536:
                raise UnsupportedHarborFeature(f"Compose service {name!r} must expose TCP port numbers")
            ports.append(EndpointRequirement(name=f"tcp-{text}", port=int(text)))
        lowered = lower_source(sidecar_source(group, name), pin)
        services.append(
            ServiceRequirement(
                name=name,
                docker_image=lowered.docker_image,
                docker_build=lowered.docker_build,
                working_directory=service.working_dir or lowered.workdir,
                environment_variables=_environment(service.environment),
                entrypoint=_argv(service.entrypoint),
                command=_argv(service.command),
                endpoints=tuple(ports),
                readiness=_readiness(service.healthcheck),
            )
        )
    return tuple(services)


def mcp_providers(
    servers: tuple[McpServer, ...], services: tuple[ServiceRequirement, ...]
) -> dict[str, ProviderRequirement]:
    """Each MCP server as a tool provider. Every server must be served over HTTP by a sidecar endpoint."""
    reachable = {(service.name, endpoint.port) for service in services for endpoint in service.endpoints}
    providers = {}
    for server in servers:
        endpoint = urlsplit(server.url or "")
        if (
            server.transport == McpTransport.STDIO
            or endpoint.scheme != "http"
            or endpoint.username
            or endpoint.password
            or (endpoint.hostname, endpoint.port or 80) not in reachable
        ):
            raise UnsupportedHarborFeature(f"MCP server {server.name!r} is not served by a Compose sidecar endpoint")
        providers[server.name] = ProviderRequirement(
            action_interface=MCP_INTERFACE, initial_state={"transport": server.transport.value, "url": server.url}
        )
    return providers
