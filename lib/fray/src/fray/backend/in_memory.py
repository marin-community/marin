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

"""In-memory backend implementation for local testing."""

import concurrent.futures
import contextvars
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from fray.cluster import ClusterContext
from fray.job import JobContext
from fray.types import EntryPoint, JobInfo, RuntimeEnv


@dataclass
class LocalJobInfo(JobInfo):
    """Extended JobInfo with subprocess handle for local backend."""

    process: subprocess.Popen | None = None


class LocalObjectRef:
    def __init__(self, future: Future):
        self.future = future

    def __repr__(self):
        return f"LocalObjectRef({id(self)})"


class _MethodWrapper:
    """
    Wrapper for actor methods that executes them on the actor's executor.

    All method calls are scheduled on the actor's single-threaded executor to
    ensure serialization of all operations on the actor."""

    def __init__(self, method: Callable, executor: ThreadPoolExecutor):
        self._method = method
        self._executor = executor

    def __call__(self, *args, **kwargs):
        """Execute method on actor's executor and return a future, preserving context."""
        # Capture the current context (including ContextVars)
        ctx = contextvars.copy_context()

        # Submit a wrapper that runs the method in the captured context
        def run_in_context():
            return ctx.run(self._method, *args, **kwargs)

        future = self._executor.submit(run_in_context)
        return LocalObjectRef(future)


class LocalActorRef:
    """
    In-memory actor reference.

    Actors are executed on a dedicated single-threaded executor to ensure
    that all method calls are serialized (no concurrent access to state).

    Method calls on an actor return futures (LocalObjectRef) that can be
    retrieved with ctx.get().
    """

    def __init__(self, instance: Any, executor: ThreadPoolExecutor):
        self._instance = instance
        self._executor = executor

    def __getattr__(self, name: str):
        """
        Forward attribute access to wrapped instance.

        Returns a _MethodWrapper for methods, which can be called directly
        to schedule execution on the actor's executor.
        """
        attr = getattr(self._instance, name)
        if callable(attr):
            return _MethodWrapper(attr, self._executor)
        return attr


class LocalJobContext(JobContext):
    """Local in-memory job context using thread pools."""

    def __init__(self, max_workers: int = 10):
        """
        Initialize local job context.

        Args:
            max_workers: Maximum number of concurrent worker threads
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._named_actors: dict[str, LocalActorRef] = {}

    def create_task(self, fn: Callable, args: tuple = (), kwargs: dict | None = None, options: Any | None = None):
        """Executes function in thread pool."""
        if kwargs is None:
            kwargs = {}

        ctx = contextvars.copy_context()

        def run_in_context():
            return ctx.run(fn, *args, **kwargs)

        future = self._executor.submit(run_in_context)
        return LocalObjectRef(future)

    def get(self, ref):
        """Block and get result from reference(s)."""
        if isinstance(ref, LocalObjectRef):
            return ref.future.result()
        if isinstance(ref, list):
            return [self.get(r) for r in ref]
        # If it's not a ref, just return it as-is
        return ref

    def wait(self, refs, num_returns=1, timeout=None):
        """Wait for num_returns refs to complete."""

        if not refs:
            return [], []

        futures = [r.future for r in refs]

        # Determine when to return
        if num_returns >= len(futures):
            return_when = concurrent.futures.ALL_COMPLETED
        else:
            return_when = concurrent.futures.FIRST_COMPLETED

        done, not_done = concurrent.futures.wait(futures, timeout=timeout, return_when=return_when)

        # For FIRST_COMPLETED, we need to wait for exactly num_returns
        while len(done) < num_returns and len(not_done) > 0:
            if timeout is not None:
                # Timeout already passed, return what we have
                break

            # Wait for one more
            newly_done, not_done = concurrent.futures.wait(
                not_done, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
            )
            done.update(newly_done)

        done_refs = [LocalObjectRef(f) for f in done]
        not_done_refs = [LocalObjectRef(f) for f in not_done]
        return done_refs, not_done_refs

    def put(self, obj):
        """Store object in local memory."""
        # Create a future that's already resolved
        future = Future()
        # Copy the object to simulate distributed semantics
        import copy

        future.set_result(copy.deepcopy(obj))
        return LocalObjectRef(future)

    def create_actor(self, klass, args=(), kwargs=None, options=None):
        """
        Create actor instance with dedicated single-threaded executor.

        Args:
            klass: Actor class to instantiate
            args: Positional arguments for klass.__init__
            kwargs: Keyword arguments for klass.__init__
            options: ActorOptions (name, get_if_exists, resources, scheduling_strategy)

        Returns:
            LocalActorRef: Handle to the actor
        """
        if kwargs is None:
            kwargs = {}

        # Handle named actors with get_if_exists
        if options is not None and options.name:
            if options.get_if_exists:
                with self._lock:
                    existing = self._named_actors.get(options.name)
                    if existing is not None:
                        return existing

            # Create new actor and register it
            instance = klass(*args, **kwargs)
            actor_executor = ThreadPoolExecutor(max_workers=1)
            actor_ref = LocalActorRef(instance, actor_executor)

            with self._lock:
                self._named_actors[options.name] = actor_ref

            return actor_ref

        # Unnamed actor - just create it
        instance = klass(*args, **kwargs)
        # Each actor gets its own single-threaded executor to ensure
        # method calls are serialized
        actor_executor = ThreadPoolExecutor(max_workers=1)
        return LocalActorRef(instance, actor_executor)


class LocalClusterContext(ClusterContext):
    """
    Local cluster context that runs jobs as subprocesses.

    Provides a complete testing environment for cluster job submission
    by executing shell commands in subprocesses, similar to how Ray
    JobSubmissionClient works.
    """

    def __init__(self):
        """Initialize local cluster context."""
        self._jobs: dict[str, LocalJobInfo] = {}
        self._job_counter = 0
        self._lock = threading.Lock()

    def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
        """
        Run entrypoint as subprocess for complete testing environment.

        Args:
            entrypoint: Shell command to execute (e.g., "python script.py")
            env: Runtime environment specification

        Returns:
            job_id: Unique identifier for the job
        """
        with self._lock:
            job_id = f"local-job-{self._job_counter}"
            self._job_counter += 1

        # Build environment dict by merging current environment with RuntimeEnv.env
        subprocess_env = os.environ.copy()
        if env.env:
            subprocess_env.update(env.env)

        # Spawn subprocess
        process = subprocess.Popen(
            entrypoint,
            shell=True,
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Create job info with process handle
        submission_time = time.time()
        job_info = LocalJobInfo(
            id=job_id,
            status="RUNNING",
            submission_time=submission_time,
            start_time=submission_time,
            end_time=None,
            process=process,
        )

        # Store job info
        with self._lock:
            self._jobs[job_id] = job_info

        # Start background thread to monitor process completion
        def monitor_process():
            exit_code = process.wait()
            with self._lock:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    # Update job status and end_time
                    job.status = "SUCCEEDED" if exit_code == 0 else "FAILED"
                    job.end_time = time.time()

        thread = threading.Thread(target=monitor_process, daemon=True)
        thread.start()

        return job_id

    def list_jobs(self) -> list[JobInfo]:
        """List all jobs with their status."""
        with self._lock:
            # Return base JobInfo objects (without process handle)
            return [
                JobInfo(
                    id=job.id,
                    status=job.status,
                    submission_time=job.submission_time,
                    start_time=job.start_time,
                    end_time=job.end_time,
                )
                for job in self._jobs.values()
            ]

    def delete_job(self, job_id: str) -> None:
        """
        Terminate subprocess and remove from tracking.

        If the job is still running, terminates the process.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            if job.process and job.process.poll() is None:
                # Process still running - terminate it
                job.process.terminate()
                try:
                    # Wait up to 5 seconds for graceful termination
                    job.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    job.process.kill()
                    job.process.wait()

                job.status = "STOPPED"
                job.end_time = time.time()

            # Remove from tracking
            self._jobs.pop(job_id, None)

    def run_on_tpu(self, fn, config, runtime_env=None):
        """
        Mock TPU execution by running function multiple times locally in parallel.

        Simulates the behavior of running on multiple TPU hosts by executing
        the function once per VM in the TPU configuration, all in parallel.

        For v4-32 with 2 slices: runs function 8 times (4 VMs/slice * 2 slices)
        """
        from fray.backend.ray.ray_tpu import get_tpu_config

        # Get TPU configuration to determine number of VMs
        try:
            tpu_config = get_tpu_config(config.tpu_type)
            num_hosts = tpu_config.vm_count * config.num_slices
        except ValueError:
            # Unknown TPU type - use default of 4 VMs per slice
            import logging

            logging.warning(f"Unknown TPU type {config.tpu_type}, assuming 4 VMs per slice")
            num_hosts = 4 * config.num_slices

        # Run function once per host in parallel to simulate multi-host execution
        def run_on_host():
            # Each host gets its own JobContext
            job_ctx = LocalJobContext()
            return fn(job_ctx)

        with ThreadPoolExecutor(max_workers=num_hosts) as executor:
            futures = [executor.submit(run_on_host) for _ in range(num_hosts)]
            results = [future.result() for future in futures]

        return results
