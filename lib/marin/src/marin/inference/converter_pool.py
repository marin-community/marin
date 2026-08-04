# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A brokered pool of persistent CPU converter processes.

This reuses the brokered-inference machinery -- the :class:`~marin.inference.broker.InferenceBroker`
actor as the work queue and :func:`~marin.inference.proxy.serve_inference_proxy` as the client-facing
front door -- but replaces the GPU engine behind it with a fleet of single-CPU worker processes that
each hold a request handler built once and reused for the life of the process. The intended workload
is document conversion: requests are whole documents, a handler is expensive to build (it loads
models) and cheap to keep, and throughput comes from hundreds of independent single-core converters
rather than from batching.

The topology differs from brokered vLLM in one way: there is no per-pod upstream HTTP server and no
forwarding worker. Each converter process leases requests *directly* from the broker actor -- the
actor handle pickles down to an endpoint name and re-resolves from the Iris job context, which
children inherit -- and executes the handler in-process. The broker's lease queue is therefore the
load balancer: a converter chewing a long document simply does not lease another one, and a lease
whose converter died is re-queued for the rest of the fleet.

Crash isolation is the supervisor's job. A handler is expected to turn a bad *document* into an
error payload (that is data, not a failure), but native code can still kill the process outright --
PyMuPDF segfaults on adversarial input. The pod's parent process respawns dead converters, and the
in-flight lease either gets an explicit error response or expires and is re-delivered. The proxy's
request timeout is the backstop that keeps a document which repeatedly kills converters from cycling
through the fleet forever.
"""

import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import requests
from fray.client import JobHandle
from fray.current_client import current_client
from fray.types import ActorConfig, Entrypoint, EnvironmentConfig, JobRequest, JobStatus, ResourceConfig
from iris.cluster.client.job_info import get_job_info
from rigging.log_setup import configure_logging
from rigging.timing import ExponentialBackoff

from marin.inference.broker import InferenceBroker
from marin.inference.config import BrokerConfig
from marin.inference.proxy import serve_inference_proxy
from marin.inference.types import (
    BrokerStatsProvider,
    InferenceRequest,
    InferenceRequestProvider,
    InferenceResponse,
    InferenceResponseProvider,
    InferenceWorkerMetadata,
    LeasedInferenceResponse,
    OpenAIEndpoint,
)
from marin.inference.worker import inference_error_response

logger = logging.getLogger(__name__)

Handler = Callable[[InferenceRequest], InferenceResponse]

_BACKEND_NAME = "converter-pool"
_MODELS_PATH = "/v1/models"

# Children inherit these, and they must be set before a child imports numpy or torch. Handlers pin
# their own inference threads (docling sets torch to one thread), but OpenMP and BLAS pools default
# to the machine core count, so N converters on an N-core pod would each spin machine-wide pools.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

_SUPERVISE_POLL_SECONDS = 5.0
# A converter that dies before serving anything is usually a handler that cannot build at all; the
# delay keeps that from becoming a tight crash loop competing with its healthy siblings for CPU.
_RESPAWN_DELAY_SECONDS = 10.0
# The idle-poll ceiling trades broker load against pickup latency. fetch_requests does not
# long-poll, so every idle converter polls at up to 1/maximum Hz: a 256-process fleet at 2 seconds
# is at worst ~128 trivial RPCs per second against the broker actor, and a converter busy on a
# document does not poll at all.
_IDLE_POLL = ExponentialBackoff(initial=0.05, maximum=2.0, factor=1.5)


@dataclass(frozen=True)
class ConverterPoolConfig:
    """The pool's shape: what the converters run, and where.

    ``handler_factory`` is called once per converter process, after thread-environment pinning and
    before the first lease, so it is the place to do expensive model loading; it must defer heavy
    imports to call time so that pickling it around does not drag them into supervisor processes.
    It crosses a process boundary, so it must be picklable with the standard pickle module -- a
    module-level function or a ``partial`` over one, not a closure.
    """

    handler_factory: Callable[[], Handler]
    model_id: str
    instances: int
    processes_per_instance: int
    worker_resources: ResourceConfig
    worker_environment: EnvironmentConfig
    broker: BrokerConfig
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.instances <= 0:
            raise ValueError("instances must be positive")
        if self.processes_per_instance <= 0:
            raise ValueError("processes_per_instance must be positive")
        try:
            pickle.dumps(self.handler_factory)
        except Exception as exc:
            raise ValueError(
                "handler_factory must be picklable with the standard pickle module (it crosses a "
                "process boundary into converter subprocesses); use a module-level function or a "
                "functools.partial over one"
            ) from exc


@dataclass(frozen=True)
class ConverterPoolSession:
    """A running pool: the proxy endpoint senders talk to, and the jobs behind it.

    ``broker`` is the queue's stats surface -- queued/leased depth, registered pods, and the
    completed-response total -- for callers that monitor a long run.
    """

    endpoint: OpenAIEndpoint
    jobs: tuple[JobHandle, ...]
    broker: BrokerStatsProvider

    def check_alive(self) -> None:
        """Raise when any pool job has reached a terminal state."""
        for job in self.jobs:
            status = job.status()
            if JobStatus.finished(status):
                raise RuntimeError(f"Converter pool job {job.job_id} finished unexpectedly with status {status}")


@contextlib.contextmanager
def remote_converter_pool(config: ConverterPoolConfig) -> Iterator[ConverterPoolSession]:
    """Start the pool on Iris, wait for the first converter to answer, and yield the endpoint.

    The proxy serves from the calling process, so the caller holds the fleet for exactly the
    lifetime of this context. Readiness is a real ``GET /v1/models`` brokered to a converter, which
    means it measures a fully built handler, not merely a scheduled job.
    """
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("remote_converter_pool must run inside an Iris job")
    client = current_client()
    run_id = uuid.uuid4().hex
    broker = config.broker
    broker_group = None
    jobs: list[JobHandle] = []
    try:
        broker_group = client.create_actor_group(
            InferenceBroker,
            name=f"converter-broker-{run_id}",
            count=1,
            request_lease_timeout_seconds=broker.request_lease_timeout_seconds,
            resources=broker.broker_resources,
            actor_config=ActorConfig(max_task_retries=0, priority=config.priority),
        )
        broker_handle = broker_group.wait_ready(count=1, timeout=broker.broker_ready_timeout_seconds)[0]
        request_provider = cast(InferenceRequestProvider, broker_handle)
        response_provider = cast(InferenceResponseProvider, broker_handle)
        for index in range(config.instances):
            worker_id = f"converter-pool-{run_id}-{index}"
            jobs.append(
                client.submit(
                    JobRequest(
                        name=worker_id,
                        entrypoint=Entrypoint.from_callable(
                            _run_pool_worker, args=(worker_id, config, request_provider)
                        ),
                        resources=config.worker_resources,
                        environment=config.worker_environment,
                        max_retries_failure=broker.max_retries_failure,
                        max_retries_preemption=broker.max_retries_preemption,
                        priority=config.priority,
                    )
                )
            )
        proxy = broker.proxy
        with serve_inference_proxy(
            broker=response_provider,
            model=config.model_id,
            host=job_info.advertise_host,
            port=proxy.port,
            request_timeout_seconds=proxy.request_timeout_seconds,
            readiness_timeout_seconds=proxy.readiness_timeout_seconds,
            max_pending_requests=proxy.max_pending_requests,
            response_fetch_batch_size=proxy.response_fetch_batch_size,
            server_start_timeout_seconds=proxy.server_start_timeout_seconds,
            ignored_request_fields=proxy.ignored_request_fields,
        ) as running_model:
            readiness = requests.get(running_model.endpoint.url("models"), timeout=proxy.readiness_timeout_seconds)
            readiness.raise_for_status()
            yield ConverterPoolSession(
                endpoint=running_model.endpoint,
                jobs=tuple(jobs),
                broker=cast(BrokerStatsProvider, broker_handle),
            )
    finally:
        for job in jobs:
            try:
                job.terminate()
            except Exception:
                logger.warning("Failed to terminate converter pool job job_id=%s", job.job_id, exc_info=True)
        if broker_group is not None:
            try:
                broker_group.shutdown()
            except Exception:
                logger.warning("Failed to shut down converter pool broker actor", exc_info=True)


def _run_pool_worker(worker_id: str, config: ConverterPoolConfig, broker: InferenceRequestProvider) -> None:
    """One pod: spawn the converter processes and keep them alive.

    Children are plain subprocesses running ``python -m marin.inference.converter_pool``, the same
    pattern as zephyr's SubprocessRunner, and deliberately not ``multiprocessing``: an Iris
    entrypoint runner executes the job function while its own ``__main__`` is still being imported
    (itself inside a multiprocessing bootstrap), so every multiprocessing start method either trips
    ``_check_not_importing_main`` or would re-import that runner and re-run this function in the
    child. A fresh exec sidesteps all of it, and the child inherits the environment -- including
    the Iris job context the broker handle re-resolves from -- through the ordinary process
    environment.
    """
    configure_logging()
    broker.register_worker(
        worker_id,
        InferenceWorkerMetadata(tensor_parallel_size=1, backend_name=_BACKEND_NAME),
    )
    os.environ.update(_SINGLE_THREAD_ENV)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as payload_file:
        pickle.dump((worker_id, config.handler_factory, config.model_id, broker), payload_file)
        payload_path = payload_file.name

    def spawn(slot: int) -> subprocess.Popen:
        # ``-u`` keeps the child's output unbuffered so a crash traceback reaches the pod log.
        return subprocess.Popen(
            [sys.executable, "-u", "-m", "marin.inference.converter_pool", payload_path, str(slot)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    children = {slot: spawn(slot) for slot in range(config.processes_per_instance)}
    logger.info("Pool worker %s started %d converter processes", worker_id, len(children))
    while True:
        time.sleep(_SUPERVISE_POLL_SECONDS)
        respawned = _respawn_dead(children, spawn)
        if respawned:
            logger.warning("Pool worker %s respawned converter slots %s", worker_id, sorted(respawned))


def _respawn_dead(children: dict[int, subprocess.Popen], spawn: Callable[[int], subprocess.Popen]) -> list[int]:
    """Replace dead converter processes, returning the slots that were respawned.

    The in-flight lease of a dead converter is not recovered here: it either got an explicit error
    response before the crash, or it expires on the broker and is re-delivered to another converter.
    """
    respawned = []
    for slot, process in children.items():
        if process.poll() is None:
            continue
        logger.warning("Converter process slot=%d died with exit code %s", slot, process.returncode)
        time.sleep(_RESPAWN_DELAY_SECONDS)
        children[slot] = spawn(slot)
        respawned.append(slot)
    return respawned


def _run_converter_process(payload_path: str, slot: str) -> None:
    """One converter: build the handler, then lease and answer one request at a time."""
    configure_logging()
    with open(payload_path, "rb") as payload_file:
        worker_id, handler_factory, model_id, broker = pickle.load(payload_file)
    handler = handler_factory()
    logger.info("Converter %s slot %s ready", worker_id, slot)
    serve_leases(broker, handler, model_id)


def serve_leases(
    broker: InferenceRequestProvider,
    handler: Handler,
    model_id: str,
    *,
    stop_event: threading.Event | None = None,
    backoff: ExponentialBackoff | None = None,
) -> None:
    """Lease one request at a time from the broker and answer it.

    Single-slot by design: a converter is a single core running one document, so leasing more than
    one at a time would only hold work hostage that an idle converter could have taken. The lease id
    is echoed exactly and every lease gets exactly one response -- the broker drops mismatches.
    """
    stop_event = threading.Event() if stop_event is None else stop_event
    backoff = (_IDLE_POLL if backoff is None else backoff).copy()
    while not stop_event.is_set():
        leased = broker.fetch_requests(max_items=1)
        if not leased:
            stop_event.wait(backoff.next_interval())
            continue
        backoff.reset()
        [lease] = leased
        response = _respond(handler, model_id, lease.request)
        broker.submit_responses([LeasedInferenceResponse(lease_id=lease.lease_id, response=response)])


def _respond(handler: Handler, model_id: str, request: InferenceRequest) -> InferenceResponse:
    if request.path == _MODELS_PATH:
        # The proxy's readiness probe. Answering it here, after handler_factory has returned, is
        # what makes readiness mean "a converter finished loading its models".
        return _models_response(request, model_id)
    try:
        return handler(request)
    except Exception as exc:
        # A handler is expected to turn a bad document into an error payload; anything that escapes
        # is a handler bug, reported in the same error envelope the forwarding worker uses.
        return inference_error_response(request, 502, "converter handler raised", detail=repr(exc), exc_info=True)


def _models_response(request: InferenceRequest, model_id: str) -> InferenceResponse:
    payload = {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "marin"}]}
    return InferenceResponse(
        request_id=request.request_id,
        status_code=200,
        payload=json.dumps(payload).encode(),
        headers=(("content-type", "application/json"),),
    )


if __name__ == "__main__":
    _run_converter_process(sys.argv[1], sys.argv[2])
