# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up an HTTP server for the ``rigging.telltale`` pages and register it."""

import atexit
import logging
import os

import uvicorn
from finelog.telltale import FinelogMetricSink
from rigging import telltale
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette

from iris.client.client import IrisContext, get_iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.platforms.types import find_free_port
from iris.cluster.types import Namespace
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.managed_thread import get_thread_container

logger = logging.getLogger(__name__)

ENDPOINT_PREFIX = "telltale"

_started = False


def _identity(job_info: JobInfo) -> telltale.MetricIdentity:
    """The Iris job coordinates stamped onto every metric row this process forwards."""
    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    # The job root (``/user/job``), not JobInfo.job_id — the latter is the task's
    # immediate parent, which for a nested ``.../worker/3`` task is ``.../worker``.
    return telltale.MetricIdentity(
        job_id=str(Namespace.from_job_id(job_info.task_id)),
        task_index=job_info.task_index,
        attempt=job_info.attempt_id,
        worker=job_info.worker_id,
        region=job_info.worker_region,
        process_index=int(process_index) if process_index is not None else None,
    )


def _start_forwarding(job_info: JobInfo, ctx: IrisContext) -> None:
    """Persist this process's telltale registry to finelog. Best-effort.

    Resolves the cluster's finelog endpoint, connects a sink, and hands it to the
    forwarder. A resolve/connect failure is logged and the job is unaffected.
    """
    try:
        endpoint = ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
    except Exception:
        logger.warning("telltale: could not resolve the finelog endpoint", exc_info=True)
        return
    try:
        sink = FinelogMetricSink(endpoint)
    except Exception:
        logger.warning("telltale: could not connect the finelog sink at %s", endpoint, exc_info=True)
        return
    telltale.start_forwarding(sink, identity=_identity(job_info))


def _endpoint_name() -> str:
    """Name identifying this exact process under the job's namespace."""
    job_info = get_job_info()
    assert job_info is not None, "no Iris job context"

    task = job_info.task_id
    # Namespace.from_job_id already carries its leading slash ("/alice/train").
    suffix = str(task).removeprefix(f"{Namespace.from_job_id(task)}/")
    name = f"{ENDPOINT_PREFIX}/{suffix}"

    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    return f"{name}/{process_index}" if process_index is not None else name


def start() -> str | None:
    """Serve telltale on an ephemeral port and register it for discovery.

    Idempotent, so callers on a shared boot path need not coordinate.

    Returns:
        The registered address, or None if nothing was started — either this call
        was a repeat, or the process is outside an in-cluster Iris job, where
        there is nothing to register with and no proxy to reach it through.
    """
    global _started
    if _started:
        return None

    job_info = get_job_info()
    ctx = get_iris_ctx()
    if job_info is None:
        logger.debug("no in-cluster Iris job context; skipping telltale")
        return None
    if ctx is None or ctx.client is None:
        logger.warning("telltale: no Iris controller client for task %s; skipping server startup", job_info.task_id)
        return None

    # Ephemeral rather than a named port: a named port is allocated per task, so
    # the processes sharing a multi-process host would collide on it, and a fixed
    # port gets taken by whatever co-tenant grabs it first.
    port = find_free_port()
    address = f"http://{job_info.advertise_host}:{port}"
    logger.info("telltale: starting HTTP server at %s", address)

    app = Starlette(routes=telltale.routes())
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error", log_config=None))
    # Daemon: this runs on the process-wide default container, which the callable
    # runner never stop()s. As a non-daemon thread it wedged threading._shutdown()
    # after the callable returned, so the task never reached a terminal state.
    get_thread_container().spawn_server(server, name=f"telltale-{port}", daemon=True)
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until_or_raise(
        lambda: server.started,
        timeout=Duration.from_seconds(5.0),
        error_message=f"telltale server did not start on port {port}",
    )

    name = _endpoint_name()
    endpoint_id = ctx.registry.register(name, address, {"job_id": ctx.job_id.to_wire()})
    # Match the registrations in jax_init: drop the endpoint on a clean exit so a
    # dead address is not served. A crash leaves it to the controller's cascade
    # delete on task cleanup.
    atexit.register(ctx.registry.unregister, endpoint_id)
    _started = True
    logger.info("telltale serving at %s, registered as %s", address, name)
    # Persist the served metrics to finelog too, so the series outlive the job.
    _start_forwarding(job_info, ctx)
    return address
