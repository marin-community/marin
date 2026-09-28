# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment entry point that submits validated RL artifact graphs to Iris."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import click
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.rpc.proto_display import priority_band_value
from rigging.config_discovery import find_project_root
from rigging.timing import Duration

from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import sanitize_job_name
from marin.experiment.cli import BuildResult, build_options_with_runner, graph_handles
from marin.rl.skyrl import IrisSkyRLExecution

_COORDINATOR_CPU = 4
_COORDINATOR_MEMORY = "16GB"
_COORDINATOR_DISK = "64GB"
_COORDINATOR_ENV_VARS = ("DAYTONA_API_KEY", "HF_TOKEN", "WANDB_API_KEY")


@dataclass(frozen=True)
class _IrisSubmissionRoute:
    cluster_config: str
    target_cluster: str | None
    priority: str
    timeout_hours: int


def _module_name(module: ModuleType) -> str:
    spec = module.__spec__
    if spec is not None and spec.name is not None:
        return spec.name
    if module.__name__ != "__main__":
        return module.__name__
    raise ValueError("RL experiment main must be defined in an importable Python module")


def _skyrl_executions(handles: list[ArtifactStep]) -> tuple[IrisSkyRLExecution, ...]:
    executions = {
        execution
        for handle in graph_handles(handles)
        for execution in handle.runtime_args.values()
        if isinstance(execution, IrisSkyRLExecution)
    }
    if not executions:
        raise ValueError("RL experiment did not construct a SkyRL artifact step")
    return tuple(executions)


def _submission_route(executions: tuple[IrisSkyRLExecution, ...]) -> _IrisSubmissionRoute:
    routes = {
        (execution.parent_cluster_config or execution.cluster_config, execution.target_cluster)
        for execution in executions
    }
    if len(routes) != 1:
        raise ValueError(f"one RL main cannot submit through multiple Iris routes: {sorted(routes, key=repr)!r}")
    priorities = {execution.priority for execution in executions}
    if len(priorities) != 1:
        raise ValueError(f"one RL main cannot use multiple coordinator priorities: {sorted(priorities)!r}")
    cluster_config, target_cluster = routes.pop()
    return _IrisSubmissionRoute(
        cluster_config=cluster_config,
        target_cluster=target_cluster,
        priority=priorities.pop(),
        timeout_hours=max(execution.coordinator_timeout_hours for execution in executions),
    )


def _coordinator_request(
    handles: list[ArtifactStep],
    module_name: str,
    argv: tuple[str, ...],
    max_concurrent: int,
    workspace: Path,
    env_vars: Mapping[str, str],
) -> tuple[str, JobRequest]:
    executions = _skyrl_executions(handles)
    route = _submission_route(executions)
    terminal = graph_handles(handles)[-1]
    job_name = sanitize_job_name(f"{terminal.name}-{terminal.version}-coordinator-{uuid.uuid4().hex[:8]}")
    request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_binary(
            "python",
            ["-m", module_name, *argv, "--max-concurrent", str(max_concurrent)],
        ),
        resources=ResourceConfig.with_cpu(
            cpu=_COORDINATOR_CPU,
            ram=_COORDINATOR_MEMORY,
            disk=_COORDINATOR_DISK,
            target_cluster=route.target_cluster,
        ),
        environment=create_environment(workspace=str(workspace), env_vars=dict(env_vars), extras=["cpu"]),
        priority=priority_band_value(route.priority),
        timeout=Duration.from_hours(route.timeout_hours),
    )
    return route.cluster_config, request


def _submit_or_run(
    module_name: str, handles: list[ArtifactStep], max_concurrent: int, coordinator_env: Mapping[str, str]
) -> None:
    if get_job_info() is not None:
        run(*handles, max_concurrent=max_concurrent)
        return

    workspace = find_project_root()
    cluster_config, request = _coordinator_request(
        handles,
        module_name,
        tuple(sys.argv[1:]),
        max_concurrent,
        workspace,
        coordinator_env,
    )
    with open_iris_client(config_file=Path(cluster_config), workspace=workspace) as iris_client:
        handle = FrayIrisClient.from_iris_client(iris_client).submit(request, adopt_existing=False)
    click.echo(f"submitted RL coordinator {handle.job_id} ({request.name}) via {cluster_config}")


def rl_build_options(fn: Callable[..., BuildResult]) -> Callable[..., None]:
    """Build and validate an RL graph, then run it in or submit it to Iris."""
    module = sys.modules[fn.__module__]
    module_name = _module_name(module)

    def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
        coordinator_env = {name: os.environ[name] for name in _COORDINATOR_ENV_VARS if name in os.environ}
        _submit_or_run(module_name, handles, max_concurrent, coordinator_env)

    return build_options_with_runner(fn, runner)
