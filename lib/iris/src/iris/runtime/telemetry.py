# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure direct process telemetry with Iris job identity."""

import logging
import os
from collections.abc import Mapping

from rigging import telemetry

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import Namespace
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger(__name__)


def _identity(job_info: JobInfo) -> dict[str, str]:
    identity = {
        "job_id": str(Namespace.from_job_id(job_info.task_id)),
        "task_id": str(job_info.task_id),
        "task_index": str(job_info.task_index),
        "attempt": str(job_info.attempt_id),
    }
    if job_info.worker_id:
        identity["worker"] = job_info.worker_id
    if job_info.worker_region:
        identity["region"] = job_info.worker_region
    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if process_index is not None:
        identity["process_index"] = process_index
    return identity


def configure(service: str, *, attributes: Mapping[str, str] | None = None) -> None:
    """Configure telemetry once for an owning application running under Iris."""
    try:
        job_info = get_job_info()
        ctx = get_iris_ctx()
        if job_info is None or ctx is None or ctx.client is None:
            logger.debug("no in-cluster Iris context; leaving %s telemetry inert", service)
            return
        endpoint = ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME).rstrip("/") + "/v1/telemetry"
        resource = _identity(job_info)
        resource.update(attributes or {})
        telemetry.configure(endpoint=endpoint, service=service, attributes=resource)
    except Exception:
        try:
            logger.warning("could not configure Finelog for %s telemetry", service, exc_info=True)
        except Exception:
            pass
        return
