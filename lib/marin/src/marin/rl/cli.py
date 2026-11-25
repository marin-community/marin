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
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from rigging.config_discovery import find_project_root
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


class _CoordinatorRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    required_env_vars: tuple[str, ...]
    forward_env: tuple[str, ...]
    fingerprint: str | None = None


_COORDINATOR_GRAPH = TypeAdapter(tuple[_CoordinatorRun, ...])


def _coordinator_graph(
    handles: list[ArtifactStep], *, execution: IrisSkyRLExecution | None = None
) -> tuple[_CoordinatorRun, ...]:
    if execution is not None:
        return tuple(
            _CoordinatorRun(
                name=step.name,
                version=step.version,
                fingerprint=step.fingerprint(),
                required_env_vars=tuple(sorted(set(execution.required_env_vars))),
                forward_env=(),
            )
            for step in handles
        )
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


def validate_coordinator_graph(handles: list[ArtifactStep], *, execution: IrisSkyRLExecution | None = None) -> None:
    """Refuse a re-entered graph whose run identities or forwarded environment names differ."""
    payload = os.environ.get(_COORDINATOR_GRAPH_ENV)
    if payload is None:
        raise click.UsageError("submitted RL graph identity is missing from the coordinator")
    try:
        submitted = _COORDINATOR_GRAPH.validate_json(payload, strict=True)
    except ValidationError as error:
        raise click.UsageError("submitted RL graph identity is malformed") from error
    rebuilt = _coordinator_graph(handles, execution=execution)
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
    cluster_config, target_cluster, container_profile = routes.pop()
    return _IrisSubmissionRoute(
        cluster_config=cluster_config,
        target_cluster=target_cluster,
        priority=priorities.pop(),
        timeout_hours=(
            0
            if any(execution.coordinator_timeout_hours == 0 for execution in executions)
            else max(execution.coordinator_timeout_hours for execution in executions)
        ),
        container_profile=container_profile,
    )


def _coordinator_request(
    handles: list[ArtifactStep],
    module_name: str,
    argv: tuple[str, ...],
    workspace: Path,
    env_vars: Mapping[str, str],
    *,
    execution: IrisSkyRLExecution | None = None,
) -> tuple[str, JobRequest]:
    executions = (execution,) if execution is not None else _skyrl_executions(handles)
    selected_env = {name: env_vars[name] for name in _COORDINATOR_ENV_VARS if name in env_vars}
    for configured_execution in executions:
        for name in configured_execution.required_env_vars:
            selected_env[name] = env_vars[name]
    selected_env[_COORDINATOR_GRAPH_ENV] = _COORDINATOR_GRAPH.dump_json(
        _coordinator_graph(handles, execution=execution)
    ).decode()
    route = _submission_route(executions)
    terminal = graph_handles(handles)[-1]
    job_name = sanitize_job_name(f"{terminal.name}-{terminal.version}-coordinator-{uuid.uuid4().hex[:8]}")
    request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_binary(
            "python",
            ["-m", module_name, *argv],
        ),
        resources=ResourceConfig.with_cpu(
            cpu=_COORDINATOR_CPU,
            ram=_COORDINATOR_MEMORY,
            disk=_COORDINATOR_DISK,
            target_cluster=route.target_cluster,
            container_profile=(
                job_pb2.ContainerProfile.Name(route.container_profile) if route.container_profile is not None else None
            ),
        ),
        environment=create_environment(workspace=str(workspace), env_vars=selected_env, extras=["cpu"]),
        priority=priority_band_value(route.priority),
        timeout=Duration.from_hours(route.timeout_hours) if route.timeout_hours else None,
    )
    return route.cluster_config, request


def submit_or_run(
    module_name: str,
    handles: list[ArtifactStep],
    max_concurrent: int,
    coordinator_env: Mapping[str, str],
    *,
    argv: tuple[str, ...],
    execution: IrisSkyRLExecution | None = None,
) -> JobName | None:
    """Run the graph in-job, or submit it and return the coordinator's ID."""
    if get_job_info() is not None:
        if _COORDINATOR_GRAPH_ENV in os.environ:
            validate_coordinator_graph(handles, execution=execution)
        run(*handles, max_concurrent=max_concurrent)
        return None

    workspace = find_project_root()
    cluster_config, request = _coordinator_request(
        handles,
        module_name,
        argv,
        workspace,
        coordinator_env,
        execution=execution,
    )
    with open_iris_client(config_file=Path(cluster_config), workspace=workspace) as iris_client:
        handle = FrayIrisClient.from_iris_client(iris_client).submit(request, adopt_existing=False)
    click.echo(f"submitted RL coordinator {handle.job_id} ({request.name}) via {cluster_config}")
    return JobName.from_wire(handle.job_id)


def rl_build_options(fn: Callable[..., BuildResult]) -> Callable[..., None]:
    """Build and validate an RL graph, then run it in or submit it to Iris."""

    def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
        # Spawn workers import the main module before registering its module spec.
        module_name = _module_name(sys.modules[fn.__module__])
        submit_or_run(module_name, handles, max_concurrent, os.environ, argv=tuple(sys.argv[1:]))

    return build_options_with_runner(fn, runner)
