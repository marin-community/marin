# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment entry point that submits validated RL artifact graphs to Iris."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

import click
from fray.iris_backend import FrayIrisClient
from fray.types import CpuConfig, Entrypoint, GpuConfig, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from rigging.config_discovery import find_project_root
from rigging.secrets import get_secret, list_secret_names, looks_like_credential, require_secret, resolve_secret_spec
from rigging.timing import Duration

from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import sanitize_job_name
from marin.experiment.cli import BuildResult, build_options_with_runner, graph_handles
from marin.rl.skyrl import IrisSkyRLExecution, skyrl_execution

_COORDINATOR_CPU = 4
_COORDINATOR_MEMORY = "16GB"
_COORDINATOR_DISK = "64GB"
_COORDINATOR_GRAPH_ENV = "MARIN_RL_COORDINATOR_GRAPH"
_COORDINATOR_ENV_VARS = (
    "GIT_COMMIT",
    "DAYTONA_API_KEY",
    "DAYTONA_RL_API_KEY",
    "DAYTONA_EVAL_API_KEY",
    "TOGETHER_API_KEY",
)


@dataclass(frozen=True)
class _IrisSubmissionRoute:
    cluster_config: str
    target_cluster: str | None
    priority: str
    timeout_hours: int
    container_profile: job_pb2.ContainerProfile | None
    coordinator_name: str | None
    coordinator_gpu: GpuConfig | None


class _CoordinatorRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    required_env_vars: tuple[str, ...]
    forward_env: tuple[str, ...]


_COORDINATOR_GRAPH = TypeAdapter(tuple[_CoordinatorRun, ...])


def _coordinator_graph(handles: list[ArtifactStep]) -> tuple[_CoordinatorRun, ...]:
    return tuple(
        _CoordinatorRun(
            name=step.name,
            version=step.version,
            forward_env=(
                tuple(sorted(set(step.run.keywords.get("forward_env", ())))) if isinstance(step.run, partial) else ()
            ),
            required_env_vars=tuple(sorted(set(execution.required_env_vars))),
        )
        for step in graph_handles(handles)
        if (execution := skyrl_execution(step)) is not None
    )


def validate_coordinator_graph(handles: list[ArtifactStep]) -> None:
    """Refuse a re-entered graph whose run identities or forwarded environment names differ."""
    payload = os.environ.get(_COORDINATOR_GRAPH_ENV)
    if payload is None:
        raise click.UsageError("submitted RL graph identity is missing from the coordinator")
    try:
        submitted = _COORDINATOR_GRAPH.validate_json(payload, strict=True)
    except ValidationError as error:
        raise click.UsageError("submitted RL graph identity is malformed") from error
    rebuilt = _coordinator_graph(handles)
    if rebuilt != submitted:
        raise click.UsageError(
            "coordinator rebuilt a different RL graph: "
            f"submitted={payload}, rebuilt={_COORDINATOR_GRAPH.dump_json(rebuilt).decode()}"
        )


def _module_name(module: ModuleType) -> str:
    spec = module.__spec__
    if spec is not None and spec.name is not None:
        return spec.name
    if module.__name__ != "__main__":
        return module.__name__
    raise ValueError("RL experiment main must be defined in an importable Python module")


def _skyrl_executions(handles: list[ArtifactStep]) -> tuple[IrisSkyRLExecution, ...]:
    executions = {execution for handle in graph_handles(handles) if (execution := skyrl_execution(handle)) is not None}
    if not executions:
        raise ValueError("RL experiment did not construct a SkyRL artifact step")
    return tuple(executions)


def _submission_route(executions: tuple[IrisSkyRLExecution, ...]) -> _IrisSubmissionRoute:
    routes = {
        (
            execution.parent_cluster_config or execution.cluster_config,
            execution.target_cluster,
            execution.container_profile,
        )
        for execution in executions
    }
    if len(routes) != 1:
        raise ValueError(f"one RL main cannot submit through multiple Iris routes: {sorted(routes, key=repr)!r}")
    priorities = {execution.priority for execution in executions}
    if len(priorities) != 1:
        raise ValueError(f"one RL main cannot use multiple coordinator priorities: {sorted(priorities)!r}")
    names = {execution.coordinator_name for execution in executions}
    if len(names) != 1:
        raise ValueError("one RL main must use one coordinator name")
    devices = {execution.coordinator_gpu for execution in executions}
    if len(devices) != 1:
        raise ValueError("one RL main must use one coordinator device allocation")
    cluster_config, target_cluster, container_profile = routes.pop()
    return _IrisSubmissionRoute(
        cluster_config=cluster_config,
        target_cluster=target_cluster,
        priority=priorities.pop(),
        coordinator_name=names.pop(),
        coordinator_gpu=devices.pop(),
        timeout_hours=max(execution.coordinator_timeout_hours for execution in executions),
        container_profile=container_profile,
    )


def _coordinator_request(
    handles: list[ArtifactStep],
    module_name: str,
    argv: tuple[str, ...],
    workspace: Path,
    env_vars: Mapping[str, str],
) -> tuple[str, JobRequest]:
    executions = _skyrl_executions(handles)
    required_names = {name for execution in executions for name in execution.required_env_vars}
    secret_sources = {}
    for execution in executions:
        for name, source in execution.coordinator_secret_sources:
            if name in secret_sources and secret_sources[name] != source:
                raise ValueError(f"RL coordinator has conflicting sources for secret {name!r}")
            secret_sources[name] = source
    secrets = {name: resolve_secret_spec(source).value for name, source in secret_sources.items()}
    empty_secrets = sorted(name for name, value in secrets.items() if not value.strip())
    if empty_secrets:
        raise ValueError("Missing or empty required coordinator secrets: " + ", ".join(empty_secrets))
    forwarded_names = required_names | set(_COORDINATOR_ENV_VARS)
    secret_names = set(list_secret_names()) | set(secret_sources)
    secret_names.update(name for name in forwarded_names if looks_like_credential(name))
    inherited_secrets = {name: get_secret(name) for name in forwarded_names & secret_names if name not in secret_sources}
    secrets.update(
        {
            name: require_secret(name) if value is not None else env_vars[name]
            for name, value in inherited_secrets.items()
            if value is not None or name in env_vars
        }
    )
    missing_names = sorted(
        name for name in required_names if not (secrets.get(name) if name in secret_names else env_vars.get(name))
    )
    if missing_names:
        raise ValueError("Missing required environment variables for RL coordinator: " + ", ".join(missing_names))
    selected_env = {name: env_vars[name] for name in forwarded_names - secret_names if name in env_vars}
    selected_env[_COORDINATOR_GRAPH_ENV] = _COORDINATOR_GRAPH.dump_json(_coordinator_graph(handles)).decode()
    route = _submission_route(executions)
    terminal = graph_handles(handles)[-1]
    job_name = sanitize_job_name(
        route.coordinator_name or f"{terminal.name}-{terminal.version}-coordinator-{uuid.uuid4().hex[:8]}"
    )
    request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_binary(
            "python",
            ["-m", module_name, *argv],
        ),
        resources=ResourceConfig(
            device=route.coordinator_gpu or CpuConfig(),
            cpu=_COORDINATOR_CPU,
            ram=_COORDINATOR_MEMORY,
            disk=_COORDINATOR_DISK,
            target_cluster=route.target_cluster,
            container_profile=(
                job_pb2.ContainerProfile.Name(route.container_profile) if route.container_profile is not None else None
            ),
        ),
        environment=create_environment(workspace=str(workspace), env_vars=selected_env, secrets=secrets, extras=["cpu"]),
        priority=priority_band_value(route.priority),
        timeout=Duration.from_hours(route.timeout_hours),
    )
    return route.cluster_config, request


def submit_or_run(
    module_name: str,
    handles: list[ArtifactStep],
    max_concurrent: int,
    coordinator_env: Mapping[str, str],
    *,
    argv: tuple[str, ...],
) -> JobName | None:
    """Run the graph in-job, or submit it and return the coordinator's ID."""
    if get_job_info() is not None:
        if _COORDINATOR_GRAPH_ENV in os.environ:
            validate_coordinator_graph(handles)
        named = _submission_route(_skyrl_executions(handles)).coordinator_name is not None
        run(*handles, max_concurrent=max_concurrent, force_run_failed=not named)
        return None

    workspace = find_project_root()
    cluster_config, request = _coordinator_request(
        handles,
        module_name,
        argv,
        workspace,
        coordinator_env,
    )
    with open_iris_client(config_file=Path(cluster_config), workspace=workspace) as iris_client:
        named = _submission_route(_skyrl_executions(handles)).coordinator_name is not None
        handle = FrayIrisClient.from_iris_client(iris_client).submit(request, adopt_existing=named)
    click.echo(f"submitted RL coordinator {handle.job_id} ({request.name}) via {cluster_config}")
    return JobName.from_wire(handle.job_id)


def rl_build_options(
    fn: Callable[..., BuildResult],
    *,
    before_submit: Callable[[list[ArtifactStep]], None] | None = None,
) -> Callable[..., None]:
    """Build and validate an RL graph, then run it in or submit it to Iris."""

    def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
        # Spawn workers import the main module before registering its module spec.
        module_name = _module_name(sys.modules[fn.__module__])
        if before_submit is not None and (get_job_info() is None or _COORDINATOR_GRAPH_ENV not in os.environ):
            before_submit(handles)
        submit_or_run(module_name, handles, max_concurrent, os.environ, argv=tuple(sys.argv[1:]))

    return build_options_with_runner(fn, runner)
