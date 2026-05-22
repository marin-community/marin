# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-facing execution context and the coordinator-as-job entrypoint.

``ZephyrContext`` is the public entry point — ``execute(dataset)`` submits a
coordinator job that hosts a ``ZephyrCoordinator`` actor in-process, spawns a
worker actor group, runs the pipeline to completion, and writes the result to
disk for the caller to read back.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
from fray import ActorConfig, Client, ResourceConfig
from fray.client import JobHandle
from fray.current_client import current_client, set_current_client
from fray.local_backend import LocalClient
from fray.types import Entrypoint, JobRequest
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem import marin_temp_bucket, open_url
from rigging.timing import ExponentialBackoff

from zephyr.dataset import Dataset
from zephyr.execution.coordinator import ZephyrCoordinator, _get_stage_description
from zephyr.execution.internals import (
    _NON_RETRYABLE_ERRORS,
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    StageRunner,
    _cleanup_execution,
    _generate_execution_id,
    _shared_data_path,
)
from zephyr.execution.worker import ZephyrWorker
from zephyr.plan import PhysicalPlan, compute_plan
from zephyr.runners import InlineRunner, SubprocessRunner
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZephyrExecutionResult:
    """Result of running a Zephyr pipeline.

    This is also the wire format pickled by ``_run_coordinator_job`` into the
    result file, so callers of ``ZephyrContext.execute`` receive it as-is.

    Attributes:
        results: Flat list of items produced by the terminal stage of the
            pipeline (e.g. output file paths for write stages).
        counters: Aggregated counter values from the run, including built-in
            zephyr counters (e.g. ``zephyr/records_in``) and any user counters
            recorded via ``zephyr.counters.increment``.
    """

    results: list
    counters: dict[str, int]


@dataclass(frozen=True)
class _CoordinatorJobConfig:
    """Serializable config for the coordinator job entrypoint."""

    plan: PhysicalPlan
    execution_id: str
    chunk_storage_prefix: str
    no_workers_timeout: float
    max_workers: int
    worker_resources: ResourceConfig
    name: str
    pipeline_id: int
    map_workers_per_actor: int = 1
    reduce_workers_per_actor: int = 1
    # None → workers fall back to InlineRunner (default). The factory is
    # cloudpickled and re-invoked per worker slot, so per-runner mutable
    # state is per-slot.
    stage_runner_factory: Callable[[int], StageRunner] | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES


def _default_stage_runner_factory_for(client: Client) -> Callable[[int], StageRunner]:
    """Pick the default ``stage_runner_factory`` based on the client type.

    ``LocalClient`` is the dev/test backend — workers are threads in a
    single process, so per-shard subprocess isolation adds latency without
    delivering meaningful isolation. Distributed clients run each worker
    actor as its own VM where subprocess-per-shard gives real protection
    against native crashes and per-shard memory growth. Callers that want
    the other behavior pass ``stage_runner_factory=...`` explicitly.
    """
    if isinstance(client, LocalClient):
        return lambda n: InlineRunner(num_workers=n)
    return lambda n: SubprocessRunner(num_workers=n)


def _run_coordinator_job(config_path: str, result_path: str) -> None:
    """Entrypoint for the coordinator job.

    Hosts the coordinator actor in-process via host_actor(), creates
    worker actors as child jobs, runs the pipeline, and writes results
    to disk. The coordinator monitors worker job health directly in its
    maintenance loop (no separate watchdog thread).
    """
    logger.info("Loading coordinator config from %s", config_path)
    with open_url(config_path, "rb") as f:
        config: _CoordinatorJobConfig = cloudpickle.loads(f.read())

    job_info = get_job_info()
    attempt_id = job_info.attempt_id if job_info else 0

    logger.info(
        "Coordinator job starting: name=%s, execution_id=%s, pipeline=%d, attempt=%d",
        config.name,
        config.execution_id,
        config.pipeline_id,
        attempt_id,
    )

    client = current_client()

    # Host coordinator actor in this process (no child job needed)
    coord_name = f"zephyr-{config.name}-p{config.pipeline_id}-coord"
    hosted = client.host_actor(
        ZephyrCoordinator,
        name=coord_name,
        actor_config=ActorConfig(max_concurrency=100),
    )
    coordinator = hosted.handle
    worker_group = None
    # host_actor starts a non-daemon uvicorn thread; the finally below must
    # run on every exit path or the process will stay alive after the main
    # body raises and the Iris task will be stuck RUNNING.
    try:
        coordinator.initialize.remote(
            config.chunk_storage_prefix,
            coordinator,
            config.no_workers_timeout,
            config.heartbeat_timeout,
            config.max_shard_failures,
            config.max_shard_infra_failures,
        ).result()

        # Create workers (child jobs)
        num_shards = config.plan.num_shards
        actual_workers = min(config.max_workers, num_shards) if num_shards > 0 else 0

        if actual_workers > 0:
            # Worker name includes attempt ID so that if a stale coordinator
            # process from a previous attempt is still running, its shutdown
            # targets the old name and cannot kill this attempt's workers.
            worker_name = f"zephyr-{config.name}-p{config.pipeline_id}-workers-a{attempt_id}"
            logger.info("Starting %d workers (max=%d, shards=%d)", actual_workers, config.max_workers, num_shards)
            worker_group = client.create_actor_group(
                ZephyrWorker,
                coordinator,
                config.stage_runner_factory,
                config.map_workers_per_actor,
                config.reduce_workers_per_actor,
                name=worker_name,
                count=actual_workers,
                resources=config.worker_resources,
                actor_config=ActorConfig(max_task_retries=10),
            )
            ready_wait_s = float(os.environ.get("ZEPHYR_WORKERS_READY_WAIT") or 12 * 60 * 60)
            worker_group.wait_ready(count=1, timeout=ready_wait_s)

            # Let the coordinator poll worker job health in its maintenance loop
            coordinator.set_worker_group.remote(worker_group).result()

        try:
            results = coordinator.run_pipeline.submit(config.plan, config.execution_id).result()
            counters = coordinator.get_counters.remote().result(timeout=10.0) or {}
            payload = ZephyrExecutionResult(results=results, counters=counters)

            ensure_parent_dir(result_path)
            with open_url(result_path, "wb") as f:
                f.write(cloudpickle.dumps(payload))
        except Exception as e:
            # Persist the exception so the caller can recover the original type
            # (important for non-retryable error detection).
            with suppress(Exception):
                ensure_parent_dir(result_path)
                with open_url(result_path, "wb") as f:
                    f.write(cloudpickle.dumps(e))
            raise
    finally:
        # Signal coordinator shutdown first so workers receive SHUTDOWN from
        # pull_task and self-terminate via shutdown_event → exit_actor(). Then
        # give the worker job a brief window to land in a terminal state on
        # its own so its Iris tasks record SUCCEEDED instead of KILLED
        # (#5484); fall back to forcibly terminating if they don't.
        with suppress(Exception):
            coordinator.shutdown.remote().result(timeout=10.0)
        if worker_group is not None:
            with suppress(Exception):
                # LocalActorGroup has no Iris task state to wait on — its
                # synthetic job handles are marked succeeded at registration
                # and is_done() is permanently False — so the graceful-exit
                # wait would always exhaust its full 5s budget without
                # observing any change. Skip it for LocalClient.
                if isinstance(client, LocalClient):
                    worker_group.shutdown()
                else:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        if worker_group.is_done():
                            break
                        time.sleep(0.5)
                    else:
                        logger.warning("Workers did not exit naturally, terminating")
                        worker_group.shutdown()
        with suppress(Exception):
            hosted.shutdown()


def _read_coordinator_result(result_path: str) -> Any:
    """Read the coordinator job's result file. Returns the deserialized object."""
    with open_url(result_path, "rb") as f:
        return cloudpickle.loads(f.read())


def _try_read_coordinator_result(result_path: str) -> Any:
    """Best-effort read of the result file. Returns None if unreadable.

    Used only in the retry error-recovery path where the coordinator job
    may have crashed before writing the file.
    """
    try:
        return _read_coordinator_result(result_path)
    except Exception:
        return None


@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    Each execute() call submits a coordinator *job* that internally creates
    coordinator and worker actors as child jobs. The coordinator job owns the
    full lifecycle: it boots workers, runs the pipeline, writes results to
    disk, and tears everything down. Iris cascading termination ensures that
    if the coordinator job dies, its children are cleaned up automatically.

    Args:
        client: The fray client to use. If None, auto-detects using current_client().
        max_workers: Upper bound on worker count. The actual count is
            min(max_workers, num_shards), computed at first execute(). If None,
            defaults to os.cpu_count() for LocalClient, or 128 for distributed clients.
        resources: Resource config per worker.
        coordinator_resources: Resource config for the coordinator job. Defaults to 2 GB.
        chunk_storage_prefix: Storage prefix for intermediate chunks. If None, defaults
            to MARIN_PREFIX/tmp/zephyr or /tmp/zephyr.
        name: Descriptive name for this context, used in actor group names for debugging.
            Defaults to a random 8-character hex string.
        no_workers_timeout: Seconds to wait for at least one worker before failing a stage.
            Defaults to 600s.
        max_execution_retries: Maximum number of times to retry a pipeline execution after
            an infrastructure failure (e.g., coordinator VM preemption). Application errors
            (ZephyrWorkerError) are never retried. Defaults to 100.
        stage_runner_factory: Callable ``(num_workers: int) -> StageRunner``.
            Defaults to ``InlineRunner`` for ``LocalClient`` and ``SubprocessRunner``
            for distributed clients.
        map_workers_per_actor: Number of concurrent subprocess workers per actor
            for map-type stages (Map, Write). Defaults to 1.
        reduce_workers_per_actor: Number of concurrent subprocess workers per actor
            for reduce-type stages (Scatter, Reduce, Fold, Join). Defaults to 1.
        heartbeat_timeout: Seconds without a worker heartbeat before the coordinator
            marks the worker FAILED and requeues its in-flight shard. Defaults to 120.
            Long-running stages (e.g. vLLM inference with cold XLA compile) may need
            to raise this; the JAX/XLA tracer can starve the worker's heartbeat thread
            during compile.
        max_shard_failures: Maximum explicit task-error retries per shard before the
            pipeline aborts. Defaults to ``MAX_SHARD_FAILURES``.
        max_shard_infra_failures: Maximum infra failures (preemption / heartbeat timeout)
            observed while the same shard was in flight before treating the shard payload
            as a deterministic crasher and aborting. Defaults to ``MAX_SHARD_INFRA_FAILURES``.
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="1g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=0.1, ram="1g", preemptible=False)
    )
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    # NOTE: 100 is fairly aggressive but it fits the preemptible env better
    max_execution_retries: int = 100
    stage_runner_factory: Callable[[int], StageRunner] | None = None
    map_workers_per_actor: int = 1
    reduce_workers_per_actor: int = 1
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES

    # Shared data staged by put(), uploaded to disk at the start of execute()
    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    # Handle to the coordinator job (for termination on retry/shutdown)
    _coordinator_job: JobHandle | None = field(default=None, repr=False)
    # NOTE: execute calls increment this at the very beginning
    _pipeline_id: int = field(default=-1, repr=False)

    def __post_init__(self):
        if self.client is None:
            self.client = current_client()

        if self.max_workers is None:
            if isinstance(self.client, LocalClient):
                self.max_workers = os.cpu_count() or 1
            else:
                # Default to 128 for distributed, but allow override
                env_val = os.environ.get("ZEPHYR_MAX_WORKERS")
                self.max_workers = int(env_val) if env_val else 128

        if self.no_workers_timeout is None:
            self.no_workers_timeout = 6 * 60 * 60  # 6 hours

        if self.chunk_storage_prefix is None:
            # TODO: consider increasing TTL for long-running pipelines (e.g. multi-day fuzzy dedup)
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")

        if self.stage_runner_factory is None:
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)

        # make sure each context is unique
        self.name = f"{self.name}-{uuid.uuid4().hex[:8]}"

    def put(self, name: str, obj: Any) -> None:
        """Stage shared data for workers to load on demand.

        Must be called before execute(). The object must be picklable.
        Workers access it via zephyr_worker_ctx().get_shared(name), which
        loads from disk on first access and caches locally.

        The actual serialization to disk happens at the start of execute(),
        once the execution_id is known, so each execution is isolated.
        """
        self._shared_data[name] = obj

    def _upload_shared_data(self, execution_id: str) -> None:
        """Serialize all staged shared data to disk under the execution directory."""
        for name, obj in self._shared_data.items():
            path = _shared_data_path(self.chunk_storage_prefix, execution_id, name)
            ensure_parent_dir(path)
            t0 = time.monotonic()
            data = cloudpickle.dumps(obj)
            elapsed = time.monotonic() - t0
            with open_url(path, "wb") as f:
                f.write(data)
            logger.info(
                "Shared data '%s' written to %s (serialized %d bytes in %.2fs)",
                name,
                path,
                len(data),
                elapsed,
            )

    def execute(
        self,
        dataset: Dataset,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> ZephyrExecutionResult:
        """Execute a dataset pipeline.

        Submits a coordinator *job* that creates coordinator and worker
        actors as child jobs, runs the pipeline, and writes results to
        disk. If the coordinator job dies (e.g., VM preemption), the
        pipeline is retried up to ``max_execution_retries`` times.
        Application errors (``ZephyrWorkerError``) are never retried.

        Returns:
            A ``ZephyrExecutionResult`` containing the flat list of results
            produced by the terminal stage and the aggregated counters from
            the run. Callers that only care about the results should access
            ``.results``; counters are exposed for callers that want to
            persist or surface them.
        """
        plan = compute_plan(dataset)
        if verbose or dry_run:
            _print_plan(dataset.operations, plan)
        if dry_run:
            return ZephyrExecutionResult(results=[], counters={})

        # NOTE: pipeline ID incremented on clean completion only
        self._pipeline_id += 1
        last_exception: Exception | None = None
        # Backoff between retries to avoid hammering an overloaded controller.
        # Starts at 2s, caps at 60s. Resets on successful pipeline startup.
        backoff = ExponentialBackoff(initial=2.0, maximum=60.0, factor=2.0, jitter=0.1)
        for attempt in range(self.max_execution_retries + 1):
            execution_id = _generate_execution_id()
            logger.info(
                "Starting zephyr pipeline: %s (pipeline %d, attempt %d)", execution_id, self._pipeline_id, attempt
            )

            config_path = f"{self.chunk_storage_prefix}/{execution_id}/job-config.pkl"
            result_path = f"{self.chunk_storage_prefix}/{execution_id}/results.pkl"

            try:
                self._upload_shared_data(execution_id)

                config = _CoordinatorJobConfig(
                    plan=plan,
                    execution_id=execution_id,
                    chunk_storage_prefix=self.chunk_storage_prefix,
                    no_workers_timeout=self.no_workers_timeout,
                    max_workers=self.max_workers,
                    worker_resources=self.resources,
                    name=self.name,
                    pipeline_id=self._pipeline_id,
                    map_workers_per_actor=self.map_workers_per_actor,
                    reduce_workers_per_actor=self.reduce_workers_per_actor,
                    stage_runner_factory=self.stage_runner_factory,
                    heartbeat_timeout=self.heartbeat_timeout,
                    max_shard_failures=self.max_shard_failures,
                    max_shard_infra_failures=self.max_shard_infra_failures,
                )
                ensure_parent_dir(config_path)
                with open_url(config_path, "wb") as f:
                    f.write(cloudpickle.dumps(config))

                job_name = f"zephyr-{self.name}-p{self._pipeline_id}-a{attempt}"
                # The wrapper job just blocks on child actors; real
                # resources are requested by the coordinator/worker children.
                # Set the context var so the coordinator job inherits self.client
                # instead of auto-detecting (which may pick a different backend).
                with set_current_client(self.client):
                    self._coordinator_job = self.client.submit(
                        JobRequest(
                            name=job_name,
                            entrypoint=Entrypoint.from_callable(
                                _run_coordinator_job,
                                args=(config_path, result_path),
                            ),
                            resources=self.coordinator_resources,
                        )
                    )

                backoff.reset()
                logger.info("Coordinator job submitted: %s (job_id=%s)", job_name, self._coordinator_job.job_id)

                self._coordinator_job.wait(timeout=None, raise_on_failure=True)

                # Read results written by the coordinator job.
                # This must succeed — the job completed successfully.
                payload = _read_coordinator_result(result_path)
                if isinstance(payload, Exception):
                    raise payload
                return payload

            except _NON_RETRYABLE_ERRORS:
                raise

            except Exception as e:
                # The coordinator job may have persisted the original
                # exception before failing. Recover it so non-retryable
                # errors are detected correctly.
                result = _try_read_coordinator_result(result_path)
                if isinstance(result, _NON_RETRYABLE_ERRORS):
                    raise result from None

                last_exception = e
                if attempt >= self.max_execution_retries:
                    raise

                delay = backoff.next_interval()
                logger.warning(
                    "Pipeline attempt %d failed (%d retries left), retrying in %.1fs: %s",
                    attempt,
                    self.max_execution_retries - attempt,
                    delay,
                    e,
                )
                time.sleep(delay)

            finally:
                # Kill coordinator job (cascade kills child actors)
                self._terminate_coordinator_job()
                _cleanup_execution(self.chunk_storage_prefix, execution_id)

        # Should be unreachable, but just in case
        raise last_exception  # type: ignore[misc]

    def _terminate_coordinator_job(self) -> None:
        if self._coordinator_job is not None:
            with suppress(Exception):
                self._coordinator_job.terminate()
            self._coordinator_job = None

    def shutdown(self) -> None:
        """Shutdown the coordinator job and all child actors."""
        self._terminate_coordinator_job()


def _print_plan(original_ops: list, plan: PhysicalPlan) -> None:
    """Print the physical plan showing shard count and operation fusion."""
    total_physical_ops = sum(len(stage.operations) for stage in plan.stages)

    logger.info("\n=== Physical Execution Plan ===\n")
    logger.info(f"Shards: {plan.num_shards}")
    logger.info(f"Original operations: {len(original_ops)}")
    logger.info(f"Stages: {len(plan.stages)}")
    logger.info(f"Physical ops: {total_physical_ops}\n")

    logger.info("Original pipeline:")
    for i, op in enumerate(original_ops, 1):
        logger.info(f"  {i}. {op}")

    logger.info("\nPhysical stages:")
    for i, stage in enumerate(plan.stages, 1):
        stage_desc = _get_stage_description(stage)
        logger.info(f"  {i}. {stage_desc}")
