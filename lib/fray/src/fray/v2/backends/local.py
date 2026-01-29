# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local backend for Fray v2.

Provides in-process execution for testing and development:
- LocalCluster: Thread-based job execution
- LocalJob: Job handle for local jobs
- LocalResolver: In-memory actor registry
- LocalActorPool: In-process actor calls with serialization
- LocalWorkerPool: ThreadPoolExecutor-based task dispatch
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cloudpickle

from fray.v2.types import (
    Entrypoint,
    EnvironmentSpec,
    JobId,
    JobStatus,
    Namespace,
    ResourceSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class LocalEndpoint:
    """Endpoint for a local actor."""

    name: str
    actor: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


class LocalJob:
    """Job handle for local execution."""

    def __init__(
        self,
        job_id: JobId,
        entrypoint: Entrypoint,
        name: str,
        cluster: LocalCluster,
    ):
        self._job_id = job_id
        self._entrypoint = entrypoint
        self._name = name
        self._cluster = cluster
        self._status = JobStatus.PENDING
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._output = io.StringIO()
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def status(self) -> JobStatus:
        with self._lock:
            return self._status

    def _set_status(self, status: JobStatus) -> None:
        with self._lock:
            self._status = status

    def start(self) -> None:
        """Start the job in a background thread."""
        self._set_status(JobStatus.RUNNING)
        self._thread = threading.Thread(target=self._run, name=f"job-{self._job_id}")
        self._thread.daemon = True
        self._thread.start()

    def _run(self) -> None:
        """Execute the entrypoint."""
        try:
            # Serialize and deserialize to catch pickling issues
            serialized = cloudpickle.dumps((self._entrypoint.callable, self._entrypoint.args, self._entrypoint.kwargs))
            fn, args, kwargs = cloudpickle.loads(serialized)

            # Capture stdout/stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = self._output
                sys.stderr = self._output
                fn(*args, **kwargs)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            self._set_status(JobStatus.SUCCEEDED)
        except Exception as e:
            self._error = e
            logger.error(f"Job {self._job_id} failed: {e}\n{traceback.format_exc()}")
            self._set_status(JobStatus.FAILED)

    def wait(
        self,
        timeout: float = 300.0,
        *,
        stream_logs: bool = False,
        raise_on_failure: bool = True,
    ) -> JobStatus:
        """Wait for job to complete."""
        if self._thread is None:
            raise RuntimeError("Job not started")

        start_time = time.monotonic()
        log_position = 0

        while True:
            # Check if done
            status = self.status()
            if JobStatus.is_finished(status):
                # Final log flush
                if stream_logs:
                    output = self._output.getvalue()
                    if output[log_position:]:
                        for line in output[log_position:].splitlines():
                            logger.info(f"[{self._job_id}] {line}")

                if raise_on_failure and status == JobStatus.FAILED:
                    error_msg = str(self._error) if self._error else "Unknown error"
                    raise RuntimeError(f"Job {self._job_id} failed: {error_msg}")
                return status

            # Stream logs
            if stream_logs:
                output = self._output.getvalue()
                new_output = output[log_position:]
                if new_output:
                    for line in new_output.splitlines():
                        logger.info(f"[{self._job_id}] {line}")
                    log_position = len(output)

            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Job {self._job_id} did not complete in {timeout}s")

            time.sleep(0.1)

    def terminate(self) -> None:
        """Terminate the job (best effort)."""
        # Python threads can't be forcefully killed, but we mark it as killed
        self._set_status(JobStatus.KILLED)


class LocalActorCallProxy:
    """Proxy that routes method calls to an actor with serialization."""

    def __init__(self, endpoint: LocalEndpoint):
        self._endpoint = endpoint

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        def method(*args: Any, **kwargs: Any) -> Any:
            # Serialize args/kwargs (catches pickling issues)
            serialized_args = cloudpickle.dumps((args, kwargs))
            args, kwargs = cloudpickle.loads(serialized_args)

            # Call the method under lock for thread safety
            with self._endpoint.lock:
                actor_method = getattr(self._endpoint.actor, method_name)
                result = actor_method(*args, **kwargs)

            # Serialize result
            return cloudpickle.loads(cloudpickle.dumps(result))

        return method


class LocalBroadcastResult:
    """Result of broadcasting to local actors."""

    def __init__(self, results: list[Any], errors: list[Exception | None]):
        self._results = results
        self._errors = errors

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Return all results (may include exceptions)."""
        return [e if e is not None else r for r, e in zip(self._results, self._errors, strict=True)]

    def wait_any(self, timeout: float | None = None) -> Any:
        """Return first successful result."""
        for result, error in zip(self._results, self._errors, strict=True):
            if error is None:
                return result
        if self._errors:
            raise self._errors[0]  # type: ignore[misc]
        raise RuntimeError("No results")


class LocalBroadcastProxy:
    """Proxy that broadcasts method calls to all actors."""

    def __init__(self, endpoints: list[LocalEndpoint]):
        self._endpoints = endpoints

    def __getattr__(self, method_name: str) -> Callable[..., LocalBroadcastResult]:
        def method(*args: Any, **kwargs: Any) -> LocalBroadcastResult:
            results = []
            errors: list[Exception | None] = []

            for endpoint in self._endpoints:
                try:
                    proxy = LocalActorCallProxy(endpoint)
                    result = getattr(proxy, method_name)(*args, **kwargs)
                    results.append(result)
                    errors.append(None)
                except Exception as e:
                    results.append(None)
                    errors.append(e)

            return LocalBroadcastResult(results, errors)

        return method


class LocalActorPool:
    """Actor pool for local execution."""

    def __init__(self, endpoints: list[LocalEndpoint]):
        self._endpoints = list(endpoints)
        self._call_index = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._endpoints)

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Wait until pool has at least min_size actors."""
        start = time.monotonic()
        while self.size < min_size:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Pool did not reach size {min_size} in {timeout}s (current: {self.size})")
            time.sleep(0.1)

    def call(self) -> LocalActorCallProxy:
        """Get proxy for round-robin calls."""
        with self._lock:
            if not self._endpoints:
                raise RuntimeError("No actors in pool")
            endpoint = self._endpoints[self._call_index % len(self._endpoints)]
            self._call_index += 1
        return LocalActorCallProxy(endpoint)

    def broadcast(self) -> LocalBroadcastProxy:
        """Get proxy for broadcasting to all actors."""
        return LocalBroadcastProxy(list(self._endpoints))

    def _add_endpoint(self, endpoint: LocalEndpoint) -> None:
        """Add an endpoint to the pool."""
        self._endpoints.append(endpoint)


class LocalResolver:
    """In-memory actor resolver for local execution."""

    def __init__(self, registry: dict[str, list[LocalEndpoint]], namespace: Namespace):
        self._registry = registry
        self._namespace = namespace

    def lookup(self, name: str) -> LocalActorPool:
        """Look up actors by name."""
        prefixed_name = f"{self._namespace}/{name}"
        endpoints = self._registry.get(prefixed_name, [])
        return LocalActorPool(endpoints)

    def _register(self, name: str, actor: Any) -> LocalEndpoint:
        """Register an actor (internal use)."""
        prefixed_name = f"{self._namespace}/{name}"
        endpoint = LocalEndpoint(name=prefixed_name, actor=actor)
        if prefixed_name not in self._registry:
            self._registry[prefixed_name] = []
        self._registry[prefixed_name].append(endpoint)
        return endpoint

    def _unregister(self, endpoint: LocalEndpoint) -> None:
        """Unregister an actor (internal use)."""
        if endpoint.name in self._registry:
            self._registry[endpoint.name] = [ep for ep in self._registry[endpoint.name] if ep is not endpoint]


class LocalWorkerPool:
    """Thread-based worker pool for local execution."""

    def __init__(self, num_workers: int, cluster: LocalCluster):
        self._num_workers = num_workers
        self._cluster = cluster
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._shutdown = False

    @property
    def size(self) -> int:
        return self._num_workers

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a task for execution."""
        if self._shutdown:
            raise RuntimeError("WorkerPool is shutdown")

        # Serialize and deserialize to catch pickling issues
        def wrapped() -> Any:
            serialized = cloudpickle.dumps((fn, args, kwargs))
            fn_, args_, kwargs_ = cloudpickle.loads(serialized)
            result = fn_(*args_, **kwargs_)
            return cloudpickle.loads(cloudpickle.dumps(result))

        return self._executor.submit(wrapped)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> LocalWorkerPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


class LocalActorServer:
    """Actor server for local execution."""

    def __init__(self, cluster: LocalCluster, host: str = "0.0.0.0", port: int = 0):
        self._cluster = cluster
        self._host = host
        self._port = port
        self._actors: dict[str, LocalEndpoint] = {}
        self._serving = False
        self._shutdown_event = threading.Event()

    def register(self, name: str, actor: Any) -> None:
        """Register an actor instance."""
        resolver = self._cluster._get_resolver_for_registration()
        endpoint = resolver._register(name, actor)
        self._actors[name] = endpoint

    def serve(self) -> None:
        """Start serving (blocks until shutdown)."""
        self._serving = True
        self._shutdown_event.wait()

    def serve_background(self) -> int:
        """Start serving in background, return port."""
        self._serving = True
        thread = threading.Thread(target=self._shutdown_event.wait, daemon=True)
        thread.start()
        return self._port

    def shutdown(self) -> None:
        """Shutdown the server."""
        self._shutdown_event.set()
        # Unregister actors
        resolver = self._cluster._get_resolver_for_registration()
        for endpoint in self._actors.values():
            resolver._unregister(endpoint)
        self._actors.clear()


class LocalCluster:
    """Local in-process cluster for testing and development.

    Jobs run as threads, actors are called in-process with serialization,
    and worker pools use ThreadPoolExecutor.
    """

    def __init__(self, **_kwargs: Any):
        self._jobs: dict[JobId, LocalJob] = {}
        self._registry: dict[str, list[LocalEndpoint]] = {}
        self._namespace = Namespace(f"local-{uuid.uuid4().hex[:8]}")
        self._lock = threading.Lock()

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> LocalJob:
        """Submit a job to the cluster."""
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        job_id = JobId(f"{self._namespace}/{name}-{uuid.uuid4().hex[:8]}")
        job = LocalJob(job_id, entrypoint, name, self)

        with self._lock:
            self._jobs[job_id] = job

        job.start()
        return job

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].terminate()

    def list_jobs(self) -> list[LocalJob]:  # type: ignore[override]
        """List all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def resolver(self) -> LocalResolver:
        """Get resolver for actor discovery."""
        return LocalResolver(self._registry, self._namespace)

    def _get_resolver_for_registration(self) -> LocalResolver:
        """Get resolver for actor registration (internal use)."""
        return self.resolver()

    def worker_pool(
        self,
        num_workers: int,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> LocalWorkerPool:
        """Create a worker pool."""
        return LocalWorkerPool(num_workers, self)

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    def create_actor_server(self, host: str = "0.0.0.0", port: int = 0) -> LocalActorServer:
        """Create an actor server (internal use by ActorServer)."""
        return LocalActorServer(self, host, port)
