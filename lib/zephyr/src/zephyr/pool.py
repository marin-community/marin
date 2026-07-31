# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator + worker pool, hosted as one fray job.

``ZephyrPool`` submits the job that hosts a ``ZephyrCoordinator`` and its worker
actor group, publishes the coordinator's endpoint, and tears the whole thing
down on ``shutdown()``. A standing pool serves many pipelines;
``ZephyrContext.execute()`` builds a one-shot pool for a single pipeline.
"""

import logging
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
from fray.client import Client, JobHandle
from fray.current_client import current_client, set_current_client
from fray.local_backend import LocalClient
from fray.types import ActorConfig, Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, create_environment
from fray.types import JobStatus as FrayJobStatus
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem import StoragePath, marin_temp_bucket
from rigging.timing import ExponentialBackoff

from zephyr.coordinator import (
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    ZephyrCoordinator,
)
from zephyr.runners import InlineRunner, SubprocessRunner
from zephyr.stage_io import (
    StageRunner,
    ZephyrTaskResources,
)
from zephyr.worker import ZephyrWorker
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


# Worker task-retry budget for pool workers — effectively "replenish forever".
# Preemptions on contended clusters can be misclassified as task failures
# (marin#7121: dirty k8s preemptions land as FAILED), and a pool must survive
# them rather than die on a small failure budget.
POOL_MAX_TASK_RETRIES = 1000

# How long ZephyrPool.start() waits for the pool job to publish its endpoint
# before giving up.
POOL_START_TIMEOUT = 600.0

# How often the serve loop asks the coordinator whether it has shut down. This
# is teardown latency paid by every pipeline: ZephyrContext.execute() shuts its
# one-shot pool down and then waits for this loop to notice, so a slow cadence
# taxes exactly the many-small-pipelines case the pool exists to speed up.
_SERVE_POLL_INTERVAL = 0.25

# Worker-job liveness is far more expensive to check and far less urgent, so it
# rides a slower cadence than the shutdown poll.
_WORKER_GROUP_POLL_INTERVAL = 5.0


def _default_stage_runner_factory_for(client: Client) -> Callable[[], StageRunner]:
    """Pick the default ``stage_runner_factory`` based on the client type.

    ``LocalClient`` is the dev/test backend — workers are threads in a
    single process, so per-shard subprocess isolation adds latency without
    delivering meaningful isolation. Distributed clients run each worker
    actor as its own VM where subprocess-per-shard gives real protection
    against native crashes and per-shard memory growth. Callers that want
    the other behavior pass ``stage_runner_factory=...`` explicitly.
    """
    if isinstance(client, LocalClient):
        return lambda: InlineRunner()

    return lambda: SubprocessRunner()


def _stop_worker_group(client: Client, worker_group: Any) -> None:
    """Give workers a brief window to exit on their own, then terminate.

    Workers receive SHUTDOWN from pull_task after coordinator.shutdown() and
    self-terminate; waiting lets their Iris tasks record SUCCEEDED instead of
    KILLED (#5484). LocalActorGroup has no Iris task state to wait on — its
    synthetic job handles are marked succeeded at registration and is_done()
    is permanently False — so the graceful-exit wait would always exhaust its
    full budget without observing any change. Skip it for LocalClient.
    """
    with suppress(Exception):
        if isinstance(client, LocalClient):
            worker_group.shutdown()
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if worker_group.is_done():
                return
            time.sleep(0.5)
        logger.warning("Workers did not exit naturally, terminating")
        worker_group.shutdown()


def _job_environment(
    pip_dependency_groups: list[str] | None,
    job_env_vars: dict[str, str] | None,
) -> EnvironmentConfig | None:
    """Environment for a coordinator + worker pool job: uv dependency extras + env vars.

    A pool's generic workers run every connecting pipeline's stage code,
    so the pool must be launched with the union of the stages' extras and any
    env vars they need. Returns None when neither is set, so the job inherits
    the parent environment exactly as before — existing callers are unaffected.
    """
    if not pip_dependency_groups and not job_env_vars:
        return None
    return create_environment(extras=pip_dependency_groups, env_vars=job_env_vars)


@dataclass(frozen=True)
class _PoolConfig:
    """Serializable config for the coordinator + worker pool job entrypoint."""

    name: str
    chunk_storage_prefix: str
    max_workers: int
    worker_resources: ResourceConfig
    # Cloudpickled and re-invoked per worker slot, so per-runner mutable
    # state is per-slot.
    stage_runner_factory: Callable[[], StageRunner]
    no_workers_timeout: float
    heartbeat_timeout: float
    max_shard_failures: int
    max_shard_infra_failures: int
    drain_idle_workers: bool


def _run_pool_job(config_path: str) -> None:
    """Entrypoint for the coordinator + worker pool job.

    Hosts the coordinator actor, boots ``max_workers`` workers, and serves
    until the owning ``ZephyrPool`` shuts it down. The coordinator's address is
    derivable from this job, so nothing is published. Drivers submit through
    ``run_pipeline`` and the coordinator runs them concurrently. A one-shot
    pipeline is just a pool that runs one pipeline and is then torn down. The
    job fails (visibly) if the worker pool terminates permanently.
    """
    logger.info("Loading pool config from %s", config_path)
    config: _PoolConfig = cloudpickle.loads(StoragePath(config_path).read_bytes())

    job_info = get_job_info()
    attempt_id = job_info.attempt_id if job_info else 0
    logger.info("Pool job starting: name=%s, attempt=%d", config.name, attempt_id)

    client = current_client()
    total = ZephyrTaskResources.from_resource_config(config.worker_resources)

    hosted = client.host_actor(
        ZephyrCoordinator,
        config.chunk_storage_prefix,
        total,
        config.no_workers_timeout,
        config.heartbeat_timeout,
        config.max_shard_failures,
        config.max_shard_infra_failures,
        config.drain_idle_workers,
        name=coordinator_actor_name(config.name),
        actor_config=ActorConfig(max_concurrency=100),
    )
    coordinator = hosted.handle
    worker_group = None
    try:
        # Worker name includes attempt ID so that if a stale coordinator
        # process from a previous attempt is still running, its shutdown
        # targets the old name and cannot kill this attempt's workers.
        worker_name = f"zephyr-{config.name}-workers-a{attempt_id}"
        logger.info("Starting %d pool workers", config.max_workers)
        worker_group = client.create_actor_group(
            ZephyrWorker,
            coordinator,
            config.stage_runner_factory,
            total,
            name=worker_name,
            count=config.max_workers,
            resources=config.worker_resources,
            actor_config=ActorConfig(max_task_retries=POOL_MAX_TASK_RETRIES),
        )
        coordinator.set_worker_group.remote(worker_group).result()

        logger.info("Pool coordinator serving")

        last_group_check = time.monotonic()
        while True:
            if coordinator.is_shutdown.remote().result(timeout=30.0):
                logger.info("Coordinator shut down; exiting serve loop")
                return
            now = time.monotonic()
            if now - last_group_check >= _WORKER_GROUP_POLL_INTERVAL:
                last_group_check = now
                if worker_group.is_done():
                    raise RuntimeError(
                        "Zephyr worker pool terminated permanently (all retries exhausted); " "shutting the pool down."
                    )
            time.sleep(_SERVE_POLL_INTERVAL)
    finally:
        with suppress(Exception):
            coordinator.shutdown.remote().result(timeout=10.0)
        if worker_group is not None:
            _stop_worker_group(client, worker_group)
        with suppress(Exception):
            hosted.shutdown()


# 6 hours: long enough to wait out cluster contention for at least one worker.
_DEFAULT_NO_WORKERS_TIMEOUT = 6 * 60 * 60


def pool_job_name(pool: str) -> str:
    """Job name a pool named ``pool`` runs under.

    A sibling job resolves the pool by this name, so it must depend on nothing
    but the pool's logical name — no uuid, no job id.
    """
    return f"zephyr-{pool}-pool"


def coordinator_actor_name(pool: str) -> str:
    """Actor name the pool's coordinator hosts itself under."""
    return f"zephyr-{pool}-coord"


def _default_max_workers(client: Client) -> int:
    """Default worker count: cpu_count for LocalClient, else ``ZEPHYR_MAX_WORKERS`` or 128."""
    if isinstance(client, LocalClient):
        return os.cpu_count() or 1
    env_val = os.environ.get("ZEPHYR_MAX_WORKERS")
    return int(env_val) if env_val else 128


@dataclass
class ZephyrPool:
    """A Zephyr coordinator + worker pool, hosted as one fray job.

    Start one pool and run many pipelines against it concurrently — from this
    process and from other Iris jobs. Open the pool as a context manager to get
    the coordinator endpoint; a driver runs pipelines on it by pointing a
    ``ZephyrContext`` at that endpoint::

        with ZephyrPool(name="ingest", max_workers=200, resources=ResourceConfig(cpu=2, ram="8g")) as endpoint:
            ZephyrContext(coordinator_endpoint=endpoint, resources=ResourceConfig(cpu=1, ram="2g")).execute(pipeline)

    Drivers connect either by passing ``coordinator_endpoint=endpoint`` or by
    exporting the endpoint as ``ZEPHYR_COORDINATOR_ENDPOINT`` on their jobs (a
    plain ``ZephyrContext()`` then picks it up). Tasks from all active pipelines
    are packed onto whichever workers have free capacity; a failing pipeline
    only fails its own ``execute()``, and preempted workers are replenished by
    Iris. The pool is torn down when the ``with`` block exits, or on
    ``shutdown()``.

    ``resources`` x ``max_workers`` sizes the worker pool. Connecting pipelines
    declare their own per-task cost via the driver's
    ``map/reduce_task_resources``, independent of the pool's worker size.

    Args:
        client: The fray client to use. If None, auto-detects via current_client().
        max_workers: Number of workers in the pool. If None, defaults to
            os.cpu_count() for LocalClient, or ``ZEPHYR_MAX_WORKERS`` / 128 for
            distributed clients.
        resources: Resource config per worker.
        coordinator_resources: Resource config for the (non-preemptible) coordinator job.
        chunk_storage_prefix: Storage prefix for intermediate chunks. Defaults to a
            temp bucket under MARIN_PREFIX.
        name: Descriptive name, used in the coordinator/worker job names.
        no_workers_timeout: Seconds a pipeline waits for at least one live worker
            before failing a stage.
        stage_runner_factory: Callable ``() -> StageRunner``. Defaults to
            ``InlineRunner`` for LocalClient and ``SubprocessRunner`` otherwise.
        heartbeat_timeout: Seconds without a worker heartbeat before the coordinator
            marks the worker FAILED and requeues its in-flight shard.
        max_shard_failures: Max explicit task-error retries per shard before a
            pipeline aborts (the pipeline, not the pool).
        max_shard_infra_failures: Max infra failures observed while the same shard is
            in flight before treating the shard payload as a deterministic crasher.
        pip_dependency_groups: Extra uv dependency groups the coordinator + worker
            pool job is launched with. Because the pool's generic workers run
            every connecting pipeline's stage functions, a pool must carry
            the union of the stages' extras (e.g. ``["datakit"]`` so embed/assign
            workers have luxical/faiss/sklearn/scipy).
        job_env_vars: Env vars set on the coordinator + worker pool job (e.g.
            ``{"JAX_PLATFORMS": "cpu"}`` so jax stages don't probe CUDA when the
            pool spills onto GPU nodes).
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
    stage_runner_factory: Callable[[], StageRunner] | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
    pip_dependency_groups: list[str] | None = None
    job_env_vars: dict[str, str] | None = None
    # Release workers as they go idle during the final stage instead of holding
    # the pool at full size until shutdown. Only safe when no further pipeline
    # will be submitted, so ``ZephyrContext.execute()`` sets it for the one-shot
    # pool it owns and a standing pool leaves it off.
    drain_idle_workers: bool = False

    endpoint: str | None = field(default=None, repr=False)
    _serve_job: JobHandle | None = field(default=None, repr=False)
    # Storage directory holding the running attempt's config + endpoint files.
    _job_dir: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.client is None:
            self.client = current_client()
        if self.max_workers is None:
            self.max_workers = _default_max_workers(self.client)
        if self.no_workers_timeout is None:
            self.no_workers_timeout = _DEFAULT_NO_WORKERS_TIMEOUT
        if self.chunk_storage_prefix is None:
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")
        if self.stage_runner_factory is None:
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)
        if self.max_workers < 1:
            # A pool with no workers still publishes an endpoint, so every
            # pipeline submitted to it would block until no_workers_timeout.
            raise ValueError(f"ZephyrPool needs at least one worker, got max_workers={self.max_workers}")
        if not self.name:
            raise ValueError("ZephyrPool needs a name: it is how sibling jobs address the pool")

    def start(self, timeout: float = POOL_START_TIMEOUT) -> str:
        """Submit the coordinator + worker pool job; block until its endpoint is published.

        Returns the coordinator endpoint. Pass it to drivers as
        ``ZephyrContext(coordinator_endpoint=...)`` or export it as
        ``ZEPHYR_COORDINATOR_ENDPOINT``. Call ``shutdown()`` (or use the pool as
        a ``with`` block) to tear it down.
        """
        if self._serve_job is not None:
            raise RuntimeError("Pool is already started")

        config = _PoolConfig(
            name=self.name,
            chunk_storage_prefix=self.chunk_storage_prefix,
            max_workers=self.max_workers,
            worker_resources=self.resources,
            stage_runner_factory=self.stage_runner_factory,
            no_workers_timeout=self.no_workers_timeout,
            heartbeat_timeout=self.heartbeat_timeout,
            max_shard_failures=self.max_shard_failures,
            max_shard_infra_failures=self.max_shard_infra_failures,
            drain_idle_workers=self.drain_idle_workers,
        )
        base = f"{self.chunk_storage_prefix}/{pool_job_name(self.name)}"
        self._job_dir = base
        config_path = f"{base}/config.pkl"
        ensure_parent_dir(config_path)
        StoragePath(config_path).write_bytes(cloudpickle.dumps(config))

        # Set the context var so the coordinator job inherits self.client
        # instead of auto-detecting (which may pick a different backend).
        with set_current_client(self.client):
            self._serve_job = self.client.submit(
                JobRequest(
                    name=pool_job_name(self.name),
                    entrypoint=Entrypoint.from_callable(_run_pool_job, args=(config_path,)),
                    resources=self.coordinator_resources,
                    environment=_job_environment(self.pip_dependency_groups, self.job_env_vars),
                )
            )
        logger.info("Shared coordinator job submitted: %s", self._serve_job.job_id)

        # The coordinator's address is derivable from the job we just created,
        # so there is nothing to publish and nothing to read back. If the pool
        # never becomes ready (crash, or slow scheduling), terminate it before
        # re-raising so we never leak a running coordinator + full worker pool.
        endpoint = self.client.actor_endpoint(self._serve_job, coordinator_actor_name(self.name))
        try:
            coordinator = self.client.get_actor(endpoint)
            backoff = ExponentialBackoff(initial=0.2, maximum=5.0)
            deadline = time.monotonic() + timeout
            while True:
                with suppress(Exception):
                    if coordinator.is_ready.remote().result(timeout=10.0):
                        break
                status = self._serve_job.status()
                if status in (FrayJobStatus.FAILED, FrayJobStatus.STOPPED, FrayJobStatus.SUCCEEDED):
                    raise RuntimeError(f"Pool job exited ({status}) before its coordinator was ready")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Pool coordinator was not ready within {timeout}s")
                time.sleep(backoff.next_interval())
        except BaseException:
            with suppress(Exception):
                self._serve_job.terminate()
            self._serve_job = None
            raise

        self.endpoint = endpoint
        logger.info("Zephyr pool ready: %s", self.endpoint)
        return self.endpoint

    def shutdown(self) -> None:
        """Gracefully stop the coordinator (workers drain and exit) and terminate the pool job.

        Raises if the job is still running and could not be terminated — the
        caller must not believe the pool is gone while it still holds workers.
        """
        serve_job = self._serve_job
        if serve_job is None:
            return
        # Best-effort: a coordinator that already died cannot drain its workers,
        # and terminating the job below cascades to them anyway.
        if self.endpoint is not None:
            with suppress(Exception):
                self.client.get_actor(self.endpoint).shutdown.remote().result(timeout=30.0)
        with suppress(Exception):
            status = serve_job.wait(timeout=60.0, raise_on_failure=False)
            if status in (FrayJobStatus.SUCCEEDED, FrayJobStatus.FAILED, FrayJobStatus.STOPPED):
                # Already finished: terminating now would only relabel a clean
                # exit as STOPPED.
                self._serve_job = None
                self.endpoint = None
                self._remove_job_dir()
                return
        try:
            serve_job.terminate()
        finally:
            self._serve_job = None
            self.endpoint = None
            self._remove_job_dir()

    def _remove_job_dir(self) -> None:
        """Delete this attempt's config + endpoint files."""
        job_dir, self._job_dir = self._job_dir, None
        if job_dir is None:
            return
        with suppress(Exception):
            path = StoragePath(job_dir)
            if path.exists():
                path.rmtree()

    def __enter__(self) -> str:
        """Start the pool and return its coordinator endpoint; ``__exit__`` tears it down."""
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
