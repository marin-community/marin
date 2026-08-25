# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure direct process telemetry with Iris job identity."""

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rigging import telemetry

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, TELEMETRY_ENDPOINT_PATH
from iris.cluster.runtime.env import IRIS_NODE_NAME_ENV
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger(__name__)

_RESERVED_RESOURCE_ATTRIBUTES = frozenset(
    {
        "execution_uid",
        "job_id",
        "task_id",
        "attempt",
        "worker",
        "process_index",
        "node_name",
        "node_uid",
        "serving_job_id",
        "run",
        "run_id",
    }
)


@dataclass(frozen=True)
class FinelogRuntime:
    """Resolved Finelog endpoint and canonical identity for an Iris process."""

    endpoint: str
    resolver: Callable[[str], str]
    attributes: dict[str, str]


def _identity(job_info: JobInfo, process_index: int | None) -> dict[str, str]:
    identity = {
        "job_id": str(job_info.job_id),
        "task_id": str(job_info.task_id),
        "attempt": str(job_info.attempt_id),
    }
    if job_info.worker_id:
        identity["worker"] = job_info.worker_id
    env_process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if process_index is not None:
        if isinstance(process_index, bool) or process_index < 0:
            raise ValueError("process_index must be a nonnegative integer")
        resolved_process_index = str(process_index)
        if env_process_index is not None and env_process_index != resolved_process_index:
            raise ValueError(
                f"process_index {resolved_process_index} conflicts with {IRIS_MULTIGPU_PROCESS_INDEX_ENV}="
                f"{env_process_index}"
            )
        identity["process_index"] = resolved_process_index
    elif env_process_index is not None:
        identity["process_index"] = env_process_index
    if node_name := os.environ.get(IRIS_NODE_NAME_ENV):
        identity["node_name"] = node_name
    return identity


def _execution_uid(job_info: JobInfo) -> str:
    if job_info.attempt_uid:
        return f"iris:{job_info.attempt_uid}"
    return f"iris:{job_info.task_id}:attempt:{job_info.attempt_id}"


def configure(
    service: str,
    *,
    run_id: str | None = None,
    execution_uid: str | None = None,
    process_index: int | None = None,
    attributes: Mapping[str, str] | None = None,
) -> None:
    """Configure telemetry once for an owning application running under Iris."""
    try:
        runtime = resolve(
            run_id=run_id,
            execution_uid=execution_uid,
            process_index=process_index,
            attributes=attributes,
        )
        if runtime is None:
            logger.debug("no in-cluster Iris context; leaving %s telemetry inert", service)
            return
        resource = dict(runtime.attributes)
        if service == "vllm":
            resource["serving_job_id"] = resource["job_id"]
        endpoint = runtime.resolver(runtime.endpoint).rstrip("/") + TELEMETRY_ENDPOINT_PATH
        telemetry.configure(endpoint=endpoint, service=service, attributes=resource)
    except Exception:
        try:
            logger.warning("could not configure Finelog for %s telemetry", service, exc_info=True)
        except Exception:
            pass
        return


def resolve(
    *,
    run_id: str | None = None,
    execution_uid: str | None = None,
    process_index: int | None = None,
    attributes: Mapping[str, str] | None = None,
) -> FinelogRuntime | None:
    """Resolve the direct Finelog route and Iris-owned process identity."""
    job_info = get_job_info()
    ctx = get_iris_ctx()
    if job_info is None or ctx is None or ctx.client is None:
        return None
    extra = dict(attributes or {})
    if conflicts := _RESERVED_RESOURCE_ATTRIBUTES.intersection(extra):
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"Iris owns canonical telemetry attributes: {names}")
    resource = _identity(job_info, process_index)
    resource.update(extra)
    resource["run_id"] = run_id or str(job_info.job_id)
    resource["execution_uid"] = execution_uid or _execution_uid(job_info)
    return FinelogRuntime(
        endpoint=LOG_SERVER_ENDPOINT_NAME,
        resolver=ctx.client.resolve_endpoint,
        attributes=resource,
    )
