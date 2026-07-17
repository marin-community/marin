# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve ``rigging.telltale`` from a process that runs no server of its own.

Actor processes need none of this: ``ActorServer`` already owns a port and mounts
the telltale routes into its own app. This module is for the other side — a
training process, which boots the JAX network and then blocks in XLA without ever
serving anything. It supplies the two things ``rigging`` structurally cannot: a
server (``rigging`` has no ``uvicorn``, being the dependency leaf ``finelog``
builds on) and Iris identity.

Each process serves its own registry. On a multi-process host every child has its
own Python interpreter and its own metrics, so each registers a distinct endpoint
including its process index. A page here is rank-local by construction; merging
across ranks is a query over shipped rows, not something this can fake.
"""

import logging
import os
import socket

import uvicorn
from rigging import telltale
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Namespace
from iris.managed_thread import get_thread_container
from iris.runtime.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger(__name__)

ENDPOINT_PREFIX = "telltale"

_started = False


def _endpoint_name() -> str:
    """Name identifying this exact process under the job's namespace.

    Every task in a job hierarchy shares one namespace, and endpoint resolution
    picks arbitrarily among live endpoints with the same name — so the task path
    and process index have to be in the name, or a lookup lands on a random peer.
    """
    job_info = get_job_info()
    assert job_info is not None, "no Iris job context"

    task = job_info.task_id
    suffix = str(task).removeprefix(f"/{Namespace.from_job_id(task)}/")
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
    if job_info is None or ctx is None or ctx.client is None:
        logger.debug("no in-cluster Iris job context; skipping telltale")
        return None

    # Bind ephemeral rather than taking a named port: named ports are allocated
    # per task, so multi-process hosts would collide on one, and a fixed port
    # gets taken by whatever co-tenant grabs it first.
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]

    app = Starlette(routes=telltale.routes())
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error", log_config=None))
    get_thread_container().spawn_server(server, name=f"telltale-{port}")
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(lambda: server.started, timeout=Duration.from_seconds(5.0))

    address = f"http://{job_info.advertise_host}:{port}"
    name = _endpoint_name()
    ctx.registry.register(name, address, {"job_id": ctx.job_id.to_wire()})
    _started = True
    logger.info("telltale serving at %s, registered as %s", address, name)
    return address
