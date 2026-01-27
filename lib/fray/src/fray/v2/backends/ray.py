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

"""Ray backend for Fray v2.

Provides Ray-based implementations:
- RayCluster: Adapts Ray Jobs/remote functions to v2 API
- RayJob: Job handle wrapping ObjectRef or submission_id
- RayResolver: Uses ray.get_actor() for actor discovery
- RayActorPool: Ray-based actor RPC
- RayWorkerPool: Ray actors as task workers
- RayActorServer: Ray named actors for RPC

Ray's object store is used internally for efficient data transfer,
but is not exposed in the public API.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import cloudpickle
import ray

from fray.v2.types import (
    Entrypoint,
    EnvironmentSpec,
    JobId,
    JobStatus,
    Namespace,
    ResourceSpec,
)

logger = logging.getLogger(__name__)


def _ray_to_v2_status(ready: bool, failed: bool = False) -> JobStatus:
    """Convert Ray task state to v2 JobStatus."""
    if failed:
        return JobStatus.FAILED
    if ready:
        return JobStatus.SUCCEEDED
    return JobStatus.RUNNING


@dataclass
class _RayJobState:
    """Internal state for a Ray job."""

    ref: ray.ObjectRef | None = None
    name: str = ""
    error: Exception | None = None
    killed: bool = False


class RayJob:
    """Job handle for Ray execution."""

    def __init__(
        self,
        job_id: JobId,
        entrypoint: Entrypoint,
        name: str,
        cluster: RayCluster,
        state: _RayJobState,
    ):
        self._job_id = job_id
        self._entrypoint = entrypoint
        self._name = name
        self._cluster = cluster
        self._state = state
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def status(self) -> JobStatus:
        """Get current job status."""
        with self._lock:
            if self._state.killed:
                return JobStatus.KILLED
            if self._state.error is not None:
                return JobStatus.FAILED
            if self._state.ref is None:
                return JobStatus.PENDING

        # Check if the ref is ready
        ready, _ = ray.wait([self._state.ref], timeout=0)
        if not ready:
            return JobStatus.RUNNING

        # Check if it succeeded or failed
        try:
            ray.get(self._state.ref, timeout=0)
            return JobStatus.SUCCEEDED
        except ray.exceptions.RayTaskError:
            return JobStatus.FAILED
        except Exception:
            return JobStatus.FAILED

    def wait(
        self,
        timeout: float = 300.0,
        *,
        stream_logs: bool = False,
        raise_on_failure: bool = True,
    ) -> JobStatus:
        """Wait for job to complete."""
        if self._state.ref is None:
            raise RuntimeError("Job not started")

        start_time = time.monotonic()
        while True:
            status = self.status()
            if JobStatus.is_finished(status):
                if raise_on_failure and status == JobStatus.FAILED:
                    # Get the actual error
                    try:
                        ray.get(self._state.ref, timeout=0)
                    except Exception as e:
                        raise RuntimeError(f"Job {self._job_id} failed: {e}") from e
                    raise RuntimeError(f"Job {self._job_id} failed: Unknown error")
                return status

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Job {self._job_id} did not complete in {timeout}s")

            time.sleep(0.1)

    def terminate(self) -> None:
        """Terminate the job."""
        with self._lock:
            self._state.killed = True
            if self._state.ref is not None:
                try:
                    ray.cancel(self._state.ref)
                except Exception as e:
                    logger.warning(f"Failed to cancel job {self._job_id}: {e}")


class RayActorCallProxy:
    """Proxy that routes method calls to a Ray actor."""

    def __init__(self, actor: ray.actor.ActorHandle):
        self._actor = actor

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        def method(*args: Any, **kwargs: Any) -> Any:
            # Serialize args/kwargs via cloudpickle for consistency with local backend
            serialized_args = cloudpickle.dumps((args, kwargs))
            args, kwargs = cloudpickle.loads(serialized_args)

            # Call the remote method and get the result
            actor_method = getattr(self._actor, method_name)
            ref = actor_method.remote(*args, **kwargs)
            result = ray.get(ref)

            # Serialize result for consistency
            return cloudpickle.loads(cloudpickle.dumps(result))

        return method


class RayBroadcastResult:
    """Result of broadcasting to Ray actors."""

    def __init__(self, refs: list[ray.ObjectRef], actors: list[ray.actor.ActorHandle]):
        self._refs = refs
        self._actors = actors

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Return all results."""
        results = []
        for ref in self._refs:
            try:
                result = ray.get(ref, timeout=timeout)
                results.append(cloudpickle.loads(cloudpickle.dumps(result)))
            except Exception as e:
                results.append(e)
        return results

    def wait_any(self, timeout: float | None = None) -> Any:
        """Return first successful result."""
        ready, _ = ray.wait(self._refs, num_returns=1, timeout=timeout)
        if ready:
            result = ray.get(ready[0])
            return cloudpickle.loads(cloudpickle.dumps(result))
        raise TimeoutError("No results within timeout")


class RayBroadcastProxy:
    """Proxy that broadcasts method calls to all Ray actors."""

    def __init__(self, actors: list[ray.actor.ActorHandle]):
        self._actors = actors

    def __getattr__(self, method_name: str) -> Callable[..., RayBroadcastResult]:
        def method(*args: Any, **kwargs: Any) -> RayBroadcastResult:
            # Serialize args/kwargs
            serialized_args = cloudpickle.dumps((args, kwargs))
            args, kwargs = cloudpickle.loads(serialized_args)

            refs = []
            for actor in self._actors:
                actor_method = getattr(actor, method_name)
                refs.append(actor_method.remote(*args, **kwargs))

            return RayBroadcastResult(refs, self._actors)

        return method


class RayActorPool:
    """Actor pool for Ray execution."""

    def __init__(self, name: str, namespace: str):
        self._name = name
        self._namespace = namespace
        self._actors: list[ray.actor.ActorHandle] = []
        self._call_index = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        self._refresh_actors()
        return len(self._actors)

    def _refresh_actors(self) -> None:
        """Refresh actor list from Ray."""
        # List all actors with matching name pattern in namespace
        try:
            actors = ray.util.list_named_actors(all_namespaces=False)
            prefix = f"{self._name}/"
            self._actors = []
            for actor_name in actors:
                if actor_name.startswith(prefix) or actor_name == self._name:
                    try:
                        actor = ray.get_actor(actor_name, namespace=self._namespace)
                        self._actors.append(actor)
                    except ValueError:
                        pass  # Actor not found
        except Exception as e:
            logger.warning(f"Failed to refresh actors: {e}")

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Wait until pool has at least min_size actors."""
        start = time.monotonic()
        while self.size < min_size:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Pool did not reach size {min_size} in {timeout}s (current: {self.size})")
            time.sleep(0.1)

    def call(self) -> RayActorCallProxy:
        """Get proxy for round-robin calls."""
        with self._lock:
            self._refresh_actors()
            if not self._actors:
                raise RuntimeError(f"No actors in pool for {self._name}")
            actor = self._actors[self._call_index % len(self._actors)]
            self._call_index += 1
        return RayActorCallProxy(actor)

    def broadcast(self) -> RayBroadcastProxy:
        """Get proxy for broadcasting to all actors."""
        self._refresh_actors()
        return RayBroadcastProxy(list(self._actors))


class RayResolver:
    """Ray-based actor resolver using named actors."""

    def __init__(self, namespace: str):
        self._namespace = namespace

    def lookup(self, name: str) -> RayActorPool:
        """Look up actors by name."""
        return RayActorPool(name, self._namespace)


@ray.remote
class _RayWorkerActor:
    """Ray actor that executes tasks."""

    def execute(self, serialized_task: bytes) -> bytes:
        """Execute a serialized task and return serialized result."""
        fn, args, kwargs = cloudpickle.loads(serialized_task)
        result = fn(*args, **kwargs)
        return cloudpickle.dumps(result)


class _RayFuture:
    """Future-compatible wrapper for Ray ObjectRef."""

    def __init__(self, ref: ray.ObjectRef):
        self._ref = ref
        self._result: Any = None
        self._exception: Exception | None = None
        self._done = False
        self._lock = threading.Lock()

    def result(self, timeout: float | None = None) -> Any:
        """Get the result, blocking if necessary."""
        with self._lock:
            if self._done:
                if self._exception:
                    raise self._exception
                return self._result

        try:
            serialized_result = ray.get(self._ref, timeout=timeout)
            result = cloudpickle.loads(serialized_result)
            with self._lock:
                self._result = result
                self._done = True
            return result
        except Exception as e:
            with self._lock:
                self._exception = e
                self._done = True
            raise

    def done(self) -> bool:
        """Check if the future is done."""
        with self._lock:
            if self._done:
                return True
        ready, _ = ray.wait([self._ref], timeout=0)
        return len(ready) > 0

    def exception(self, timeout: float | None = None) -> Exception | None:
        """Get the exception if any."""
        try:
            self.result(timeout=timeout)
            return None
        except Exception as e:
            return e


class RayWorkerPool:
    """Ray-based worker pool using Ray actors."""

    def __init__(
        self,
        num_workers: int,
        cluster: RayCluster,
        resources: ResourceSpec,
    ):
        self._num_workers = num_workers
        self._cluster = cluster
        self._resources = resources
        self._shutdown = False
        self._workers: list[ray.actor.ActorHandle] = []
        self._worker_index = 0
        self._lock = threading.Lock()

        # Create worker actors
        self._create_workers()

    def _create_workers(self) -> None:
        """Create Ray worker actors."""
        # Determine resource requirements
        num_cpus = self._resources.cpu if self._resources.cpu > 0 else 1
        num_gpus = self._resources.device_count if self._resources.device_type == "gpu" else 0

        for _i in range(self._num_workers):
            worker = _RayWorkerActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                namespace=self._cluster._namespace,
            ).remote()
            self._workers.append(worker)

    @property
    def size(self) -> int:
        return self._num_workers

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a task for execution."""
        if self._shutdown:
            raise RuntimeError("WorkerPool is shutdown")

        # Serialize the task
        serialized_task = cloudpickle.dumps((fn, args, kwargs))

        # Round-robin dispatch to workers
        with self._lock:
            worker = self._workers[self._worker_index % len(self._workers)]
            self._worker_index += 1

        # Submit to worker and wrap in Future
        ref = worker.execute.remote(serialized_task)
        return _RayFuture(ref)  # type: ignore[return-value]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool."""
        self._shutdown = True
        if wait:
            # Kill all worker actors
            for worker in self._workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
        self._workers = []

    def __enter__(self) -> RayWorkerPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


@ray.remote
class _RayActorWrapper:
    """Ray actor that wraps user actors for RPC."""

    def __init__(self, serialized_actor: bytes):
        self._actor = cloudpickle.loads(serialized_actor)

    def call_method(self, method_name: str, serialized_args: bytes) -> bytes:
        """Call a method on the wrapped actor."""
        args, kwargs = cloudpickle.loads(serialized_args)
        method = getattr(self._actor, method_name)
        result = method(*args, **kwargs)
        return cloudpickle.dumps(result)


class RayActorServer:
    """Actor server for Ray execution using named actors."""

    def __init__(self, cluster: RayCluster, host: str = "0.0.0.0", port: int = 0):
        self._cluster = cluster
        self._host = host
        self._port = port
        self._actors: dict[str, ray.actor.ActorHandle] = {}
        self._serving = False
        self._shutdown_event = threading.Event()

    def register(self, name: str, actor: Any) -> None:
        """Register an actor instance as a Ray named actor."""
        # Serialize the actor
        serialized_actor = cloudpickle.dumps(actor)

        # Create a unique name for this actor instance
        actor_name = f"{name}/{uuid.uuid4().hex[:8]}"

        # Create the Ray actor
        ray_actor = _RayActorWrapper.options(
            name=actor_name,
            namespace=self._cluster._namespace,
            lifetime="detached",
        ).remote(serialized_actor)

        self._actors[name] = ray_actor
        logger.info(f"Registered actor {actor_name} in namespace {self._cluster._namespace}")

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
        """Shutdown the server and kill actors."""
        self._shutdown_event.set()
        for actor in self._actors.values():
            try:
                ray.kill(actor)
            except Exception:
                pass
        self._actors.clear()


# Proxy class for RayActorWrapper that uses call_method
class _RayWrappedActorCallProxy:
    """Proxy for calling methods on _RayActorWrapper."""

    def __init__(self, actor: ray.actor.ActorHandle):
        self._actor = actor

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        def method(*args: Any, **kwargs: Any) -> Any:
            serialized_args = cloudpickle.dumps((args, kwargs))
            ref = self._actor.call_method.remote(method_name, serialized_args)
            serialized_result = ray.get(ref)
            return cloudpickle.loads(serialized_result)

        return method


class RayCluster:
    """Ray-based cluster implementation for Fray v2.

    Adapts Ray to the v2 API:
    - Jobs run as ray.remote tasks
    - Actors use Ray named actors for discovery
    - Worker pools use Ray actors for task dispatch
    """

    def __init__(
        self,
        namespace: str | None = None,
        address: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Ray cluster connection.

        Args:
            namespace: Ray namespace for actor isolation
            address: Ray cluster address (default: connects to existing cluster)
            **kwargs: Additional options (ignored)
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            if address:
                ray.init(address=address, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)

        # Use provided namespace or create a unique one
        if namespace is None:
            self._namespace = f"fray_v2_{uuid.uuid4().hex[:8]}"
        else:
            self._namespace = namespace

        self._jobs: dict[JobId, RayJob] = {}
        self._lock = threading.Lock()

        logger.info(f"RayCluster initialized with namespace: {self._namespace}")

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> RayJob:
        """Submit a job to the cluster."""
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        job_id = JobId(f"{self._namespace}/{name}-{uuid.uuid4().hex[:8]}")
        state = _RayJobState(name=name)

        job = RayJob(job_id, entrypoint, name, self, state)

        # Determine resource requirements
        num_cpus = resources.cpu if resources.cpu > 0 else 1
        num_gpus = resources.device_count if resources.device_type == "gpu" else 0

        # Use Ray's native serialization by making the callable a remote function directly
        fn = entrypoint.callable
        args = entrypoint.args
        kwargs = entrypoint.kwargs

        # Create and run the remote function
        remote_fn = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(fn)
        ref = remote_fn.remote(*args, **kwargs)

        state.ref = ref

        with self._lock:
            self._jobs[job_id] = job

        return job

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].terminate()

    def list_jobs(self) -> list[RayJob]:  # type: ignore[override]
        """List all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def resolver(self) -> RayResolver:
        """Get resolver for actor discovery."""
        return RayResolver(self._namespace)

    def worker_pool(
        self,
        num_workers: int,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> RayWorkerPool:
        """Create a worker pool."""
        return RayWorkerPool(num_workers, self, resources)

    @property
    def namespace(self) -> Namespace:
        return Namespace(self._namespace)

    def create_actor_server(self, host: str = "0.0.0.0", port: int = 0) -> RayActorServer:
        """Create an actor server (internal use by ActorServer)."""
        return RayActorServer(self, host, port)
