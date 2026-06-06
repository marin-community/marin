# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import asyncio
import atexit
import enum
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from finelog.client import LogClient, RemoteLogHandler
from finelog.client.proxy import LogServiceProxy, StatsServiceProxy
from finelog.server import LogServiceImpl
from finelog.server.asgi import build_log_server_asgi
from finelog.server.stats_service import StatsServiceImpl
from finelog.store.duckdb_store import EMBEDDED_DUCKDB_MEMORY_LIMIT, EMBEDDED_DUCKDB_THREADS, DuckDBLogStore
from rigging.timing import Duration, ExponentialBackoff, RateLimiter, Timestamp, TokenBucket
from sqlalchemy import bindparam, select

from iris.cluster.backends.types import find_free_port, resolve_external_host
from iris.cluster.bundle import BundleStore
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.persistence import persist_autoscaler_state
from iris.cluster.controller.backend import (
    BackendReconcileInput,
    CapacityInput,
    ClusterCapacity,
    PlacementOwner,
    ScheduleInput,
    SchedulingEvent,
    TaskBackend,
)
from iris.cluster.controller.checkpoint import (
    CheckpointResult,
    backup_databases,
    upload_checkpoint,
    write_checkpoint,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.ops.task import (
    Assignment,
    apply_dispatch_updates,
    finalize,
)
from iris.cluster.controller.ops.worker import (
    apply_reconcile,
)
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.pruner import prune_old_data
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.dispatch import (
    DISPATCH_PROMOTION_RATE,
)
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
    build_reconcile_plans,
)
from iris.cluster.controller.run_template import RunTemplateCache, new_run_template_cache
from iris.cluster.controller.scheduling.policy import (
    build_scheduling_context,
    compute_demand_entries,
    get_running_tasks_with_band_and_value,
    read_reservation_claims,
    refresh_reservation_claims,
)
from iris.cluster.controller.scheduling.scheduler import (
    Scheduler,
    SchedulingContext,
)
from iris.cluster.controller.schema import (
    ReservationClaim,
    hint_rare_state,
    job_config_table,
    task_attempts_table,
    tasks_table,
    workers_table,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.log_keys import CONTROLLER_LOG_KEY
from iris.cluster.runtime.profile import PROFILE_NAMESPACE, IrisProfile
from iris.cluster.types import (
    JobName,
    UserBudgetDefaults,
    WorkerId,
    WorkerStatus,
    WorkerStatusMap,
)
from iris.cluster.worker.stats import TASK_STATS_NAMESPACE, IrisTaskStat
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import AuthTokenInjector, NullAuthInterceptor, StaticTokenProvider, TokenVerifier

logger = logging.getLogger(__name__)

# Sync Connect RPC handlers are dispatched via ``asyncio.to_thread``, which
# uses the running loop's default executor. asyncio's default executor sizes
# at ``min(32, os.cpu_count() + 4)`` — only 8 threads on a 4-vCPU controller
# VM. A handful of slow handlers (e.g. ``launch_job`` blocking up to 120s in
# ``_wait_until_job_drained``) saturates that pool and head-of-line blocks
# every other RPC, including the worker heartbeats that would unblock the
# drain. Install a wider, named pool so a burst of slow handlers cannot
# starve the rest.
_RPC_HANDLER_THREADS = 1024


def _install_rpc_executor(server: uvicorn.Server, *, max_workers: int) -> None:
    """Replace ``server.run`` with a variant that pins a sized default executor."""

    def run_with_executor() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_default_executor(ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rpc-handler"))
        try:
            loop.run_until_complete(server.serve())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    server.run = run_with_executor


class SchedulingOutcome(enum.Enum):
    """Result of a scheduling cycle, used to drive adaptive backoff."""

    NO_PENDING_TASKS = "no_pending_tasks"
    NO_ASSIGNMENTS = "no_assignments"
    ASSIGNMENTS_MADE = "assignments_made"


# Log a detailed per-phase scheduling trace every this many rounds.
_SCHEDULING_TRACE_INTERVAL = 50


@dataclass
class ControllerConfig:
    """Controller configuration."""

    host: str = "127.0.0.1"
    """Host to bind the HTTP server to."""

    port: int = 0
    """Port to bind the HTTP server to. Use 0 for auto-assign."""

    remote_state_dir: str = ""
    """Remote URI for controller checkpoints and worker profiles (e.g. gs://bucket/iris/state)."""

    scheduler_min_interval: Duration = field(default_factory=lambda: Duration.from_seconds(10.0))
    """Minimum scheduling loop interval (used when cluster is active)."""

    scheduler_max_interval: Duration = field(default_factory=lambda: Duration.from_seconds(10.0))
    """Maximum scheduling loop interval (reached via exponential backoff when idle)."""

    heartbeat_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    """How often to send heartbeats to workers."""

    autoscaler_evaluation_interval: Duration = field(default_factory=lambda: Duration.from_seconds(10.0))
    """How often the controller runs an autoscale cycle (backend ``manage_capacity``).
    Only used when the backend does not manage its own capacity."""

    poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(1.0))
    """Polling reconcile cadence. The polling thread wakes every ``poll_interval``
    (or sooner if ``_polling_wake`` is set) and runs ``_reconcile_tick``
    against every healthy worker. The Reconcile RPC is the sole channel that
    dispatches new ASSIGNED rows and observes worker-side state changes."""

    max_tasks_per_job_per_cycle: int = 4
    """Maximum tasks from a single non-coscheduled job to consider per scheduling
    cycle. Bounds CPU time in the scheduler when many tasks are pending, preventing
    GIL starvation of the heartbeat thread. Coscheduled jobs are exempt (they need
    all tasks for atomic assignment). Set to 0 for unlimited."""

    checkpoint_interval: Duration | None = None
    """If set, take a periodic best-effort snapshot this often.
    Runs in the autoscaler loop thread; does not pause scheduling."""

    prune_interval: Duration = field(default_factory=lambda: Duration.from_seconds(3600))
    """How often to run the data pruning sweep (default: 1 hour)."""

    job_retention: Duration = field(default_factory=lambda: Duration.from_seconds(7 * 86400))
    """Delete terminal jobs older than this (default: 7 days)."""

    worker_retention: Duration = field(default_factory=lambda: Duration.from_seconds(86400))
    """Delete inactive/unhealthy workers whose last heartbeat exceeds this (default: 24 hours)."""

    local_state_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="iris_controller_state_")))
    """Local directory for controller DB, logs, bundle cache."""

    auth_verifier: TokenVerifier | None = None
    """When set, all RPC calls require a valid bearer token verified by this verifier."""

    auth_provider: str | None = None
    """Name of the auth provider (e.g. "gcp", "static") for the dashboard UI."""

    auth: ControllerAuth | None = None
    """Full auth config passed to the service layer for login and API key management."""

    dry_run: bool = False
    """Start in dry-run mode: compute scheduling but suppress all side effects."""

    user_budget_defaults: UserBudgetDefaults = field(default_factory=UserBudgetDefaults)
    """Default budget settings applied when a new user is first seen."""

    log_service_address: str | None = None
    """Address of an externally-hosted log server (e.g. http://localhost:10001).
    When set, the controller connects to the existing server. When None,
    the Controller starts an in-process LogServiceImpl on a free port (used by
    tests and local-mode runs). In production this address is sourced from
    `endpoints["/system/log-server"]` and passed in here by the daemon entrypoint."""

    endpoints: dict[str, str] = field(default_factory=dict)
    """Resolved cluster endpoints: logical name -> concrete URL. Built from
    cluster_config.endpoints by the daemon entrypoint. Registered into the
    controller service's _system_endpoints during start()."""


def _log_client_interceptors(config: "ControllerConfig") -> tuple:
    """Return Connect interceptors for controller-originated LogService RPCs.

    When auth is configured, attach the worker JWT as a bearer token so the
    log server accepts PushLogs/FetchLogs. The worker token is signed with
    the same key the log server verifies against; no separate admin token
    is required for controller-initiated pushes.
    """
    token = config.auth.worker_token if config.auth and config.auth.worker_token else None
    if not token:
        return ()
    return (AuthTokenInjector(StaticTokenProvider(token)),)


class Controller:
    """Unified controller managing all components and lifecycle.

    Runs three background loops:
    - Scheduling loop: finds task assignments, checks worker timeouts
    - Provider loop: syncs task state with the execution backend via TaskBackend
    - Autoscaler loop: evaluates scaling decisions, manages slice lifecycle

    Each loop runs on its own thread so blocking operations in one don't
    stall the others.

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(
            config=config,
            provider=RpcTaskBackend(stub_factory=RpcWorkerStubFactory()),
        )
        controller.start()
        try:
            job_id = controller.launch_job(request)
            status = controller.get_job_status(job_id)
        finally:
            controller.stop()
        ```

    Args:
        config: Controller configuration
        provider: TaskBackend for communicating with the execution backend. When
            it does not manage its own capacity (``manages_capacity`` is False),
            the controller drives its ``manage_capacity`` in a background loop.
    """

    def __init__(
        self,
        config: ControllerConfig,
        provider: TaskBackend,
        threads: ThreadContainer | None = None,
        db: ControllerDB | None = None,
    ):
        if not config.remote_state_dir:
            raise ValueError(
                "remote_state_dir is required. Set via ControllerConfig.remote_state_dir. "
                "Example: remote_state_dir='gs://my-bucket/iris/state'"
            )

        self._config = config
        self._stopped = False
        self._task_backend: TaskBackend = provider
        self._provider_scheduling_events: list[SchedulingEvent] = []
        self._provider_capacity: ClusterCapacity | None = None
        self._promotion_bucket = TokenBucket(
            capacity=DISPATCH_PROMOTION_RATE,
            refill_period=Duration.from_minutes(1),
        )

        config.local_state_dir.mkdir(parents=True, exist_ok=True)
        if db is not None:
            self._db = db
        else:
            self._db = ControllerDB(db_dir=config.local_state_dir / "db")
        self._health = WorkerHealthTracker()
        self._endpoints = EndpointsProjection(self._db)
        self._worker_attrs = WorkerAttrsProjection(self._db)
        writes.validate()
        self._seed_liveness_from_workers()
        self._db.register_reopen_hook(self._seed_liveness_from_workers)

        # ThreadContainer must be initialized before the log service setup
        # because _start_local_log_server spawns a uvicorn thread.
        self._threads = threads if threads is not None else get_thread_container()

        # --- Log service setup ---
        # The log server is always accessed via RPC. In production the
        # controller's main() starts a subprocess; in tests/local mode
        # the Controller spins up an in-process uvicorn thread. After the
        # server is running, all access goes through RPC clients — no
        # branching on hosting mode.
        self._log_service: LogServiceImpl | None = None
        self._log_server: uvicorn.Server | None = None

        if config.log_service_address:
            self._log_service_address = config.log_service_address
        else:
            self._log_service_address = self._start_local_log_server()

        log_client_interceptors = _log_client_interceptors(config)
        self._remote_log_service = LogServiceProxy(self._log_service_address, interceptors=log_client_interceptors)
        self._remote_stats_service = StatsServiceProxy(self._log_service_address, interceptors=log_client_interceptors)

        # Providers that collect logs outside the worker process push directly
        # to the log server via RPC. K8s pods have no worker daemon, so the
        # provider also writes per-pod resource samples to iris.task itself —
        # mirroring what the worker daemon does on the GCE/TPU path.
        if self._task_backend.placement is PlacementOwner.TASK_BACKEND:
            k8s_log_client = LogClient.connect(self._log_service_address, interceptors=log_client_interceptors)
            self._task_backend.set_log_sink(
                k8s_log_client,
                k8s_log_client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat),
                k8s_log_client.get_table(PROFILE_NAMESPACE, IrisProfile),
            )

        # Controller process logs ship to the log server via RemoteLogHandler.
        self._log_client = LogClient.connect(self._log_service_address, interceptors=log_client_interceptors)
        self._log_handler = RemoteLogHandler(self._log_client, key=CONTROLLER_LOG_KEY)

        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger("iris").addHandler(self._log_handler)

        self._run_template_cache: RunTemplateCache = new_run_template_cache()
        self._scheduler = Scheduler()

        self._bundle_store = BundleStore(storage_dir=f"{config.remote_state_dir.rstrip('/')}/bundles")

        self._service = ControllerServiceImpl(
            controller=self,
            bundle_store=self._bundle_store,
            log_client=self._log_client,
            db=self._db,
            health=self._health,
            endpoints=self._endpoints,
            worker_attrs=self._worker_attrs,
            auth=config.auth,
            system_endpoints={},
            user_budget_defaults=config.user_budget_defaults,
        )
        self._dashboard = ControllerDashboard(
            self._service,
            log_service=self._remote_log_service,
            host=config.host,
            port=config.port,
            auth_verifier=config.auth_verifier,
            auth_provider=config.auth_provider,
            auth_optional=config.auth.optional if config.auth else False,
            finelog_stats_service=self._remote_stats_service,
        )

        # Background loop state. Two wake events drive two threads:
        #   * ``_scheduling_wake`` — producers that may free capacity (terminal
        #     heartbeats, attempt finalization). Scheduling tick re-evaluates
        #     pending tasks.
        #   * ``_polling_wake`` — producers that change ``tasks.state`` such
        #     that the per-worker reconcile snapshot differs (new ASSIGNED,
        #     bulk cancel). Polling tick re-fans-out the next worker batch.
        #
        # Most write paths set both: over-waking is cheaper than missing a
        # capacity-return or a fresh ASSIGNED row.
        self._scheduling_wake = threading.Event()
        self._polling_wake = threading.Event()
        self._server: uvicorn.Server | None = None
        self._scheduling_thread: ManagedThread | None = None
        self._polling_thread: ManagedThread | None = None
        self._dispatch_thread: ManagedThread | None = None
        self._autoscaler_thread: ManagedThread | None = None
        self._prune_thread: ManagedThread | None = None
        self._ping_thread: ManagedThread | None = None
        self._checkpoint_thread: ManagedThread | None = None

        # Throttles the execution-timeout deadline scan in _reconcile_tick.
        # The reconcile tick runs frequently (poll cadence); the timeout query
        # only needs minute-granularity, so we gate it behind a 60s limiter.
        self._timeout_rate_limiter: RateLimiter = RateLimiter(interval_seconds=60.0)

        # Cached scheduling diagnostics: populated each scheduling cycle for
        # pending jobs that could not be assigned.  Keyed by job wire ID.
        # RPC handlers read this dict instead of recomputing diagnostics,
        # avoiding expensive scheduler work on every CLI poll.
        self._scheduling_diagnostics: dict[str, str] = {}
        self._scheduling_round: int = 0

        # Last completed scheduling context — None until the first tick runs.
        # The dashboard diagnostics path reads this instead of rebuilding from
        # the DB. This is the only ``| None`` attribute on Controller: it is
        # genuinely None before the first scheduling tick has run.
        self._last_scheduling_context: SchedulingContext | None = None

        # Set to True once start() is called. Used to gate operations that
        # are only valid before the controller loops begin (e.g. LoadCheckpoint).
        self._started = False

        self._atexit_registered = False

        # Rate-limits periodic (best-effort) checkpoint writes.
        # None when checkpoint_interval is not configured.
        # mark_run() seeds the last-run time so the first checkpoint fires
        # one interval after boot rather than immediately — avoids a
        # checkpoint storm right when the controller comes up.
        self._periodic_checkpoint_limiter: RateLimiter | None = (
            RateLimiter(interval_seconds=config.checkpoint_interval.to_seconds())
            if config.checkpoint_interval is not None
            else None
        )
        if self._periodic_checkpoint_limiter is not None:
            self._periodic_checkpoint_limiter.mark_run()

    def wake(self) -> None:
        """Signal both the scheduling and polling loops to run immediately.

        Called on new job submission. The scheduling tick picks up the new
        pending tasks, and the polling tick re-fans-out the next worker batch
        so any ASSIGNED rows the scheduler writes land on the worker without
        waiting for a full polling rotation.
        """
        self._scheduling_wake.set()
        self._polling_wake.set()

    def _seed_liveness_from_workers(self) -> None:
        """Mark every persisted worker healthy so the scheduler sees them before they ping back.

        Workers that fail to ping within the heartbeat window are timed out
        by the ping loop. ``find_prunable`` relies on this seed to maintain
        the invariant that every ``workers`` row has a tracker entry.
        """
        now_ms = Timestamp.now().epoch_ms()
        with self._db.read_snapshot() as q:
            rows = q.execute(select(workers_table.c.worker_id)).all()
        worker_ids = [WorkerId(str(row.worker_id)) for row in rows]
        if worker_ids:
            self._health.heartbeat(worker_ids, now_ms)

    @property
    def started(self) -> bool:
        """Whether the controller loops have been started."""
        return self._started

    def _start_local_log_server(self) -> str:
        """Start a bundled in-process log + stats server and return its address.

        Used as a fallback when ``cluster_config.endpoints`` does not declare
        ``/system/log-server`` (and in tests). Backed by an in-memory
        ``DuckDBLogStore`` — no segmentation, no flush thread, no compaction,
        and logs are lost on controller restart. The stats RPC surface
        (RegisterTable / WriteRows / Query / DropTable) is still available.
        For any deployment that needs persistence or scale, run
        ``finelog-server`` out-of-band (point it at a local dir with
        ``remote_log_dir=""`` if you just want disk persistence without
        remote sync) and point ``endpoints["/system/log-server"]`` at it.
        """
        log_server_port = find_free_port()
        log_store = DuckDBLogStore(
            log_dir=None,
            duckdb_memory_limit=EMBEDDED_DUCKDB_MEMORY_LIMIT,
            duckdb_threads=EMBEDDED_DUCKDB_THREADS,
        )
        self._log_service = LogServiceImpl(log_store=log_store)
        stats_service = StatsServiceImpl(log_store=log_store)

        interceptors = (NullAuthInterceptor(verifier=self._config.auth_verifier),)
        app = build_log_server_asgi(
            self._log_service,
            interceptors=interceptors,
            stats_service=stats_service,
        )
        log_server_config = uvicorn.Config(
            app,
            host=self._config.host,
            port=log_server_port,
            log_level="warning",
            log_config=None,
            timeout_keep_alive=120,
        )
        self._log_server = uvicorn.Server(log_server_config)
        self._threads.spawn_server(self._log_server, name="log-server")
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._log_server is not None and self._log_server.started,
            timeout=Duration.from_seconds(5.0),
        )

        address = f"http://{self.external_host}:{log_server_port}"
        logger.info("Local log server ready at %s", address)
        return address

    def start(self) -> None:
        """Start main controller loop, dashboard server, and optionally autoscaler."""
        self._started = True
        if self._config.dry_run:
            logger.info("[DRY-RUN] Controller started in dry-run mode — all side effects suppressed")

        if self._task_backend.placement is PlacementOwner.TASK_BACKEND:
            self._dispatch_thread = self._threads.spawn(self._run_dispatch_loop, name="dispatch-loop")
        else:
            self._scheduling_thread = self._threads.spawn(self._run_scheduling_loop, name="scheduling-loop")
            self._polling_thread = self._threads.spawn(self._run_polling_loop, name="polling-loop")
            self._ping_thread = self._threads.spawn(self._run_ping_loop, name="ping-loop")
            if not self._config.dry_run:
                self._prune_thread = self._threads.spawn(self._run_prune_loop, name="prune-loop")

        # Create and start uvicorn server via spawn_server, which bridges the
        # ManagedThread stop_event to server.should_exit automatically.
        # timeout_keep_alive: uvicorn defaults to 5s, which races with client polling
        # intervals of the same length, causing TCP resets on idle connections. Use 120s
        # to safely cover long polling gaps during job waits.
        # proxy_headers / forwarded_allow_ips: production traffic arrives via
        # GCP IAP + an HTTPS load balancer. Without trusting their forwarded
        # headers, ``scope["server"]`` is the controller's bind address, so
        # any absolute URL built by Starlette (notably the trailing-slash
        # redirect on routes like ``/proxy/<name>``) leaks the internal IP
        # back to the browser as ``http://10.x.x.x:10000/...`` — unreachable
        # outside the VPC. Trusting all upstream IPs is safe because the
        # controller's only ingress is the LB.
        server_config = uvicorn.Config(
            self._dashboard.app,
            host=self._config.host,
            port=self._config.port,
            log_level="warning",
            log_config=None,
            timeout_keep_alive=120,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        self._server = uvicorn.Server(server_config)
        _install_rpc_executor(self._server, max_workers=_RPC_HANDLER_THREADS)
        self._threads.spawn_server(self._server, name="controller-server")

        # Register cluster endpoints BEFORE spawning the autoscaler. Otherwise the
        # autoscaler's first tick can create buffer slices whose workers query the
        # controller for /system/log-server before this dict is populated, returning
        # an empty result. The slice creation fails, the group enters backoff, and
        # any task constrained to that group hangs until the backoff expires.
        for name, url in self._config.endpoints.items():
            self._service._system_endpoints[name] = url
            logger.info("Registered system endpoint %s -> %s", name, url)
        self._service._system_endpoints["/system/log-server"] = self._log_service_address

        if not self._task_backend.manages_capacity and not self._config.dry_run:
            self._autoscaler_thread = self._threads.spawn(self._run_autoscaler_loop, name="autoscaler-loop")

        if self._periodic_checkpoint_limiter is not None and not self._config.dry_run:
            self._checkpoint_thread = self._threads.spawn(self._run_checkpoint_loop, name="checkpoint-loop")

        # Register atexit hook to capture final state for post-mortem analysis.
        # Unregistered in stop() so it doesn't fire against a closed DB.
        self._atexit_registered = True
        atexit.register(self._atexit_checkpoint)

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=Duration.from_seconds(5.0),
        )

    def stop(self) -> None:
        """Stop all background components gracefully. Idempotent.

        Shutdown ordering:
        1. Unregister atexit hook so it doesn't fire against a closed DB.
        2. Stop scheduling/provider/autoscaler loops so no new work is triggered.
        3. Shut down the autoscaler (stops monitors, terminates VMs, stops platform).
        4. Stop remaining threads (server) and executors.
        """
        if self._stopped:
            return
        self._stopped = True
        # Unregister atexit hook before closing DB connections.
        if self._atexit_registered:
            atexit.unregister(self._atexit_checkpoint)
            self._atexit_registered = False
        self._scheduling_wake.set()
        self._polling_wake.set()
        join_timeout = Duration.from_seconds(5.0)
        if self._scheduling_thread:
            self._scheduling_thread.stop()
            self._scheduling_thread.join(timeout=join_timeout)
        if self._polling_thread:
            self._polling_thread.stop()
            self._polling_thread.join(timeout=join_timeout)
        if self._dispatch_thread:
            self._dispatch_thread.stop()
            self._dispatch_thread.join(timeout=join_timeout)
        if self._ping_thread:
            self._ping_thread.stop()
            self._ping_thread.join(timeout=join_timeout)
        if self._prune_thread:
            self._prune_thread.stop()
            self._prune_thread.join(timeout=join_timeout)
        if self._autoscaler_thread:
            self._autoscaler_thread.stop()
            self._autoscaler_thread.join(timeout=join_timeout)
        if self._checkpoint_thread:
            self._checkpoint_thread.stop()
            self._checkpoint_thread.join(timeout=join_timeout)

        self._threads.stop()
        # The backend owns the autoscaler now; close() shuts it down (terminates
        # VMs, stops the platform) and releases the backend's own resources.
        self._task_backend.close()

        # Remove log handler before closing log resources to avoid errors
        # from late log records hitting a closed store or connection.
        logging.getLogger("iris").removeHandler(self._log_handler)
        self._log_handler.close()
        self._log_client.close()
        self._remote_log_service.close()
        self._remote_stats_service.close()
        if self._log_service:
            self._log_service.close()
        self._db.close()
        self._bundle_store.close()

    def _atexit_checkpoint(self) -> None:
        """Best-effort checkpoint at interpreter shutdown for post-mortem analysis."""
        if self._config.dry_run:
            return
        try:
            path, _result = write_checkpoint(self._db, self._config.remote_state_dir)
            logger.info("atexit checkpoint written: %s", path)
        except Exception:
            logger.exception("atexit checkpoint failed")

    def _run_scheduling_loop(self, stop_event: threading.Event) -> None:
        """Scheduling loop with adaptive backoff.

        Backs off from min to max interval when idle (no pending tasks or no
        assignments possible). Resets to min interval when woken by a producer
        that may free capacity or by a new job submission.

        Reconciliation runs on the separate polling thread
        (``_run_polling_loop``) on its own cadence via the Reconcile RPC.
        Sharing the
        same write transaction is no longer required: producers write
        ``tasks.state = ASSIGNED`` and the polling thread reads that state via
        a snapshot query, so dispatch never crosses a transaction boundary.
        """
        backoff = ExponentialBackoff(
            initial=self._config.scheduler_min_interval.to_seconds(),
            maximum=self._config.scheduler_max_interval.to_seconds(),
            factor=2.0,
            jitter=0.1,
        )
        while not stop_event.is_set():
            interval = backoff.next_interval()
            woken = self._scheduling_wake.wait(timeout=interval)
            self._scheduling_wake.clear()

            if stop_event.is_set():
                break

            if woken:
                backoff.reset()

            outcome = self._run_scheduling()
            if outcome == SchedulingOutcome.ASSIGNMENTS_MADE:
                backoff.reset()

    def _run_polling_loop(self, stop_event: threading.Event) -> None:
        """Per-worker reconcile loop on the configured ``poll_interval`` cadence.

        Each tick:
          1. Snapshot every healthy active worker.
          2. Fan out the Reconcile RPC carrying the desired-attempt set per
             worker (ASSIGNED rows to start, BUILDING/RUNNING rows to keep
             alive, and stops for everything else).
          3. Apply observation-driven results in one write txn.

        The polling thread is the sole path that pushes work to a worker.
        Worker auto-kill is implicit: any attempt absent from the desired
        set on the next reconcile is killed locally by the worker.
        """
        tick_seconds = self._config.poll_interval.to_seconds()
        while not stop_event.is_set():
            self._polling_wake.wait(timeout=tick_seconds)
            self._polling_wake.clear()
            if stop_event.is_set():
                break
            try:
                self._reconcile_tick()
            except Exception:
                logger.exception("Polling reconcile tick failed")

    def _run_prune_loop(self, stop_event: threading.Event) -> None:
        """Background maintenance: WAL checkpoint every 10 min, full data prune on the configured interval."""
        wal_checkpoint_interval = 600.0
        last_full_prune = 0.0
        full_prune_interval = self._config.prune_interval.to_seconds()

        while not stop_event.is_set():
            stop_event.wait(timeout=wal_checkpoint_interval)
            if stop_event.is_set():
                break

            try:
                busy, log_frames, checkpointed = self._db.wal_checkpoint()
                logger.info(
                    "wal_checkpoint(TRUNCATE): busy=%d log_frames=%d checkpointed=%d",
                    busy,
                    log_frames,
                    checkpointed,
                )
            except Exception:
                logger.exception("WAL checkpoint failed")

            now = time.monotonic()
            if now - last_full_prune >= full_prune_interval:
                last_full_prune = now
                try:
                    prune_old_data(
                        self._db,
                        self._health,
                        self._endpoints,
                        self._worker_attrs,
                        job_retention=self._config.job_retention,
                        worker_retention=self._config.worker_retention,
                        stop_event=stop_event,
                    )
                except Exception:
                    logger.exception("Data pruning failed")

    def _run_autoscaler_loop(self, stop_event: threading.Event) -> None:
        """Autoscaler loop: runs on its own thread so blocking cloud API calls
        don't stall scheduling or heartbeats."""
        limiter = RateLimiter(interval_seconds=self._config.autoscaler_evaluation_interval.to_seconds())
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            try:
                self._run_autoscaler_once()
            except Exception:
                logger.exception("Autoscaler loop iteration failed")

    def _run_checkpoint_loop(self, stop_event: threading.Event) -> None:
        """Periodic checkpoint loop: runs on its own thread so the multi-second
        backup+upload doesn't stall the autoscaler cadence."""
        limiter = self._periodic_checkpoint_limiter
        assert limiter is not None, "checkpoint loop spawned without configured limiter"
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            try:
                write_checkpoint(self._db, self._config.remote_state_dir)
            except Exception:
                logger.exception("Periodic checkpoint failed")

    def _run_dispatch_loop(self, stop_event: threading.Event) -> None:
        """Reconcile loop for BACKEND-placement backends: no scheduling, no workers."""
        limiter = RateLimiter(interval_seconds=self._config.heartbeat_interval.to_seconds())
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            try:
                self._sync_dispatch()
            except Exception:
                logger.exception("Direct provider sync round failed, will retry next interval")

    def _sync_dispatch(self) -> None:
        if self._config.dry_run:
            return
        max_promotions = self._promotion_bucket.available
        with self._db.transaction() as cur:
            batch = dispatch.drain_for_dispatch(
                cur,
                cache=self._run_template_cache,
                max_promotions=max_promotions,
            )
        if batch.tasks_to_run:
            self._promotion_bucket.try_acquire(len(batch.tasks_to_run))
        result = self._task_backend.reconcile(batch)
        with self._db.transaction() as cur:
            apply_dispatch_updates(
                cur,
                result.updates,
                health=self._health,
                endpoints=self._endpoints,
                now=Timestamp.now(),
            )
        self._provider_scheduling_events = list(result.scheduling_events) if result.scheduling_events else []
        self._provider_capacity = result.capacity
        # Worker-side kills are surfaced through the next K8s pod-diff sync;
        # no immediate RPC fan-out here.

    def _run_scheduling(self) -> SchedulingOutcome:
        """Run one scheduling cycle.

        The controller owns only the I/O: it refreshes reservation claims and
        reads a DB-less snapshot (scheduling context + running-task band/value
        for preemption), hands the snapshot to ``backend.schedule`` for the pure
        placement decision, then commits the returned assignments, preemptions,
        and unschedulable marks. The backend (IRIS placement) runs the full
        gates → order → taints → preference → find_assignments → preemption
        pipeline; BACKEND placement returns an empty result (Kueue schedules).

        No lock is needed since only one scheduling thread exists. Every DB
        access is serialized by ControllerDB._lock with multi-statement
        mutations wrapped in BEGIN IMMEDIATE transactions.
        """
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0

        claims = self._refresh_reservation_claims()
        ctx = build_scheduling_context(
            self._db,
            self._health,
            self._worker_attrs,
            self._config.user_budget_defaults,
        )
        # Lifted DB read: band/value of running tasks the preemption pass may
        # evict. Read here (against the claimed-worker set) so ``schedule`` stays
        # DB-less; the same workers the old conditional read covered.
        running_for_preemption = get_running_tasks_with_band_and_value(self._db, set(claims.keys()))

        if trace:
            logger.info(
                "[TRACE round=%d] Phase 0: %d pending tasks, %d workers, %d reservation claims",
                self._scheduling_round,
                len(ctx.pending_task_rows),
                len(ctx.workers),
                len(claims),
            )

        if not ctx.pending_task_rows:
            self._scheduling_diagnostics = {}
            self._last_scheduling_context = ctx
            return SchedulingOutcome.NO_PENDING_TASKS

        result = self._task_backend.schedule(
            ScheduleInput(
                context=ctx,
                claims=claims,
                running_for_preemption=running_for_preemption,
                max_tasks_per_job_per_cycle=self._config.max_tasks_per_job_per_cycle,
                trace=trace,
            )
        )

        # Commit the decisions. Expired/deadline tasks are marked UNSCHEDULABLE;
        # assignments stamp ASSIGNED; preemption finalizes victims.
        if result.unschedulable:
            self._mark_tasks_unschedulable(result.unschedulable)
        if result.assignments:
            self._commit_assignments(result.assignments)
        self._apply_preemptions(result.preemptions)

        self._scheduling_diagnostics = result.diagnostics
        self._last_scheduling_context = result.post_taint_context

        if result.assignments or result.preemptions:
            log_event(
                "scheduling_pass_completed",
                "scheduler",
                assignments=len(result.assignments),
                preempted=len(result.preemptions),
                pending=len(ctx.pending_task_rows),
                workers=len(ctx.workers),
            )
            return SchedulingOutcome.ASSIGNMENTS_MADE
        return SchedulingOutcome.NO_ASSIGNMENTS

    def _refresh_reservation_claims(self) -> dict[WorkerId, ReservationClaim]:
        """Read, clean up, and refresh reservation claims. Returns updated claims."""
        return refresh_reservation_claims(
            self._db,
            self._health,
            self._worker_attrs,
            persist=not self._config.dry_run,
        )

    def _commit_assignments(self, assignments: list[Assignment]) -> None:
        """Persist scheduler decisions to ``tasks.state = ASSIGNED`` rows.

        Each :class:`Assignment` carries the effective priority band the backend
        computed against the snapshot's user spend, so ``assign_task`` stamps it
        onto ``tasks.priority_band``. The preemption pass then trusts that
        stamped value instead of recomputing from current spend every tick.

        The polling reconcile thread reads ASSIGNED rows on its next tick
        (woken via ``_polling_wake``) and fans out the Reconcile RPCs.
        """
        if self._config.dry_run:
            for assignment in assignments:
                logger.info("[DRY-RUN] Would assign task %s to worker %s", assignment.task_id, assignment.worker_id)
            return
        with self._db.transaction() as cur:
            ops.task.assign(cur, assignments, health=self._health)
        # Wake the polling thread; every tick reconciles every healthy worker,
        # so the new ASSIGNED rows turn into Reconcile RPCs on the next tick.
        self._polling_wake.set()

    def _apply_preemptions(self, preemptions: list[TerminalDecision]) -> None:
        """Finalize the backend's PREEMPT decisions.

        Slice evictions for a coscheduled preemptor's N siblings are
        all-or-nothing. Victims stop on the next reconcile tick: the planner
        drops them from the worker's desired set.
        """
        if not preemptions:
            return
        with self._db.transaction() as cur:
            finalize(
                cur,
                preemptions,
                health=self._health,
                endpoints=self._endpoints,
                now=Timestamp.now(),
            )
        logger.info("Preemption pass: %d tasks preempted", len(preemptions))

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None:
        """Return cached scheduling diagnostic for a job, or None if unavailable."""
        return self._scheduling_diagnostics.get(job_wire_id)

    def _scan_execution_timeouts(self, snap: Any, now_ms: int) -> list[TerminalDecision]:
        """Find executing tasks past their deadline within an existing read snapshot.

        Issued inline with the reconcile snapshot so the timeout sweep adds one
        query (not a fresh snapshot open) per reconcile tick where the limiter
        fires. Returned decisions are applied by the caller inside the same
        write transaction as the reconcile results.
        """
        rows = snap.execute(
            select(
                tasks_table.c.task_id,
                task_attempts_table.c.started_at_ms,
                job_config_table.c.timeout_ms,
            )
            .select_from(
                tasks_table.join(job_config_table, job_config_table.c.job_id == tasks_table.c.job_id).join(
                    task_attempts_table,
                    (task_attempts_table.c.task_id == tasks_table.c.task_id)
                    & (task_attempts_table.c.attempt_id == tasks_table.c.current_attempt_id),
                )
            )
            .where(
                hint_rare_state(tasks_table.c.state.in_(bindparam("executing_states", expanding=True))),
                job_config_table.c.timeout_ms.is_not(None),
                job_config_table.c.timeout_ms > 0,
                task_attempts_table.c.started_at_ms.is_not(None),
            ),
            {"executing_states": [int(job_pb2.TASK_STATE_BUILDING), int(job_pb2.TASK_STATE_RUNNING)]},
        ).all()
        decisions: list[TerminalDecision] = []
        for row in rows:
            if row.started_at_ms.epoch_ms() + int(row.timeout_ms) > now_ms:
                continue
            logger.warning("Task %s exceeded execution timeout, killing", row.task_id)
            decisions.append(
                TerminalDecision(
                    kind=TerminalKind.TIMEOUT,
                    task_id=row.task_id,
                    reason="Execution timeout exceeded",
                )
            )
        return decisions

    def _mark_tasks_unschedulable(self, tasks: list[Any]) -> None:
        """Mark a batch of tasks as unschedulable due to scheduling timeout.

        Each entry must be a row from ``_pending_tasks_with_jobs``; it carries
        ``scheduling_timeout_ms`` so no secondary DB fetch is needed.
        """
        if not tasks:
            return
        if self._config.dry_run:
            for task in tasks:
                logger.info("[DRY-RUN] Would mark task %s as unschedulable", task.task_id)
            return
        decisions: list[TerminalDecision] = []
        for task in tasks:
            timeout_ms = task.scheduling_timeout_ms
            timeout = Duration.from_ms(timeout_ms) if timeout_ms is not None else None
            logger.warning(f"Task {task.task_id} exceeded scheduling timeout ({timeout}), marking as UNSCHEDULABLE")
            decisions.append(
                TerminalDecision(
                    kind=TerminalKind.UNSCHEDULABLE,
                    task_id=task.task_id,
                    reason=f"Scheduling timeout exceeded ({timeout})",
                )
            )
        with self._db.transaction() as cur:
            finalize(
                cur,
                decisions,
                health=self._health,
                endpoints=self._endpoints,
                now=Timestamp.now(),
            )

    @property
    def last_scheduling_context(self) -> "SchedulingContext | None":
        """Return the most recent finalized scheduling context.

        ``None`` before the first scheduling tick has run; otherwise the
        post-taint context from the last completed ``_run_scheduling`` pass.
        Consumed by dashboard diagnostics that need a snapshot of capacities
        and pending tasks without rebuilding from the DB.
        """
        return self._last_scheduling_context

    # =========================================================================
    # Worker lifecycle RPC dispatch (Reconcile / Ping)
    # =========================================================================

    def _snapshot_reconcile_inputs(
        self, scan_timeouts: bool, now_ms: int
    ) -> tuple[ReconcileInputs, dict[WorkerId, str], list[TerminalDecision]]:
        """Snapshot the DB and assemble the reconcile inputs for one tick.

        When ``scan_timeouts`` is True, also issues the execution-timeout
        deadline scan inside the same read snapshot so the sweep adds one
        query (not a fresh snapshot open) per limiter tick.
        """
        with self._db.read_snapshot() as snap:
            addresses = reads.list_active_healthy_workers(snap, self._health)
            if not addresses:
                timeout_decisions = self._scan_execution_timeouts(snap, now_ms) if scan_timeouts else []
                return ReconcileInputs(job_specs={}, worker_ids=[], rows_by_worker={}), {}, timeout_decisions
            worker_ids = list(addresses)
            # Snapshot current attempts for ``worker_ids``. Workers not in
            # ``worker_ids`` are filtered in Python so the partial index
            # ``idx_task_attempts_live_workerbound`` remains active rather
            # than falling back to a scan on a long IN list. We deliberately
            # do NOT filter on task state: active rows (ASSIGNED/BUILDING/
            # RUNNING) drive normal reconciliation; rows whose task has
            # already moved to a terminal state but whose attempt is still
            # worker-bound (worker_id set, finished_at_ms NULL) are stranded
            # attempts whose terminal Reconcile observation was lost.
            # Including them in the desired set gives the worker a second
            # chance to report -- either with the real terminal status or
            # via the MISSING synthesis in ``handle_reconcile`` -- so the
            # reconcile-observation path can stamp finished_at_ms. Without this, a single
            # lost RPC strands the attempt forever, since no other code path
            # polls about it.
            target_ids: set[WorkerId] = set(worker_ids)
            raw_rows = snap.execute(
                select(
                    task_attempts_table.c.worker_id,
                    tasks_table.c.task_id,
                    task_attempts_table.c.attempt_id,
                    tasks_table.c.state.label("task_state"),
                    task_attempts_table.c.state.label("attempt_state"),
                    tasks_table.c.job_id,
                    task_attempts_table.c.attempt_uid,
                )
                .select_from(
                    task_attempts_table.join(
                        tasks_table,
                        (tasks_table.c.task_id == task_attempts_table.c.task_id)
                        & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                    )
                )
                .where(
                    task_attempts_table.c.worker_id.is_not(None),
                    task_attempts_table.c.finished_at_ms.is_(None),
                ),
            ).all()
            rows = [row for row in raw_rows if row.worker_id in target_ids]
            templates_by_job: dict[JobName, job_pb2.RunTaskRequest | None] = {}
            for row in rows:
                if row.task_state != job_pb2.TASK_STATE_ASSIGNED:
                    continue
                if row.job_id not in templates_by_job:
                    templates_by_job[row.job_id] = dispatch.run_request_template(
                        self._run_template_cache, snap, row.job_id
                    )
            timeout_decisions = self._scan_execution_timeouts(snap, now_ms) if scan_timeouts else []

        rows_by_worker: dict[WorkerId, list[ReconcileRow]] = {wid: [] for wid in worker_ids}
        for row in rows:
            rows_by_worker[row.worker_id].append(row)

        # ``templates_by_job`` can carry ``None`` for jobs the scheduler hasn't
        # cached yet; reconcile_worker checks the dict membership so feeding it
        # the raw map is harmless. Filter Nones to keep the type tight.
        job_specs = {jid: spec for jid, spec in templates_by_job.items() if spec is not None}
        inputs = ReconcileInputs(job_specs=job_specs, worker_ids=worker_ids, rows_by_worker=rows_by_worker)
        return inputs, addresses, timeout_decisions

    def _reconcile_tick(self) -> None:
        """One polling-tick reconcile pass: snapshot, fan out, apply.

        The execution-timeout deadline scan is folded into this tick (gated by
        ``_timeout_rate_limiter`` so it fires at most once per minute). When it
        fires, timeout-driven terminal decisions ride the same write txn as
        the reconcile results.
        """
        if self._config.dry_run:
            return

        now = Timestamp.now()
        scan_timeouts = self._timeout_rate_limiter.should_run()
        inputs, addresses, timeout_decisions = self._snapshot_reconcile_inputs(scan_timeouts, now.epoch_ms())
        if not inputs.worker_ids and not timeout_decisions:
            return

        plans = build_reconcile_plans(inputs) if inputs.worker_ids else []
        if plans:
            reconcile_input = BackendReconcileInput(plans=plans, worker_addresses=addresses)
            results = self._task_backend.reconcile(reconcile_input).worker_results
        else:
            results = []

        plan_by_worker: dict[WorkerId, WorkerReconcilePlan] = {p.worker_id: p for p in plans}
        for result in results:
            if result.error is not None:
                logger.debug("Reconcile failed for worker %s: %s", result.worker_id, result.error)
        with self._db.transaction() as cur:
            if plans:
                apply_reconcile(
                    cur,
                    plan_by_worker,
                    results,
                    health=self._health,
                    endpoints=self._endpoints,
                    now=now,
                )
            if timeout_decisions:
                finalize(
                    cur,
                    timeout_decisions,
                    health=self._health,
                    endpoints=self._endpoints,
                    now=now,
                )

    def _get_active_worker_addresses(self) -> list[tuple[WorkerId, str | None]]:
        """Get healthy active workers as (worker_id, address) tuples for ping."""
        with self._db.read_snapshot() as tx:
            workers = reads.healthy_active_workers_with_attributes(tx, self._health, self._worker_attrs)
        return [(w.worker_id, w.address) for w in workers]

    def _run_ping_loop(self, stop_event: threading.Event) -> None:
        """Fast ping loop for liveness detection and prompt worker termination.

        Sends Ping RPCs to all healthy workers every heartbeat_interval,
        bumps the WorkerHealthTracker on failures, and immediately terminates
        workers that cross the ping threshold.
        """
        ping_interval_s = self._config.heartbeat_interval.to_seconds()
        limiter = RateLimiter(interval_seconds=ping_interval_s)

        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            try:
                workers = self._get_active_worker_addresses()
                results = self._task_backend.ping_workers(workers)

                live_worker_ids: list[WorkerId] = []
                for result in results:
                    if result.error is not None:
                        self._health.ping(result.worker_id, healthy=False)
                    else:
                        self._health.ping(result.worker_id, healthy=True)
                        live_worker_ids.append(result.worker_id)

                if live_worker_ids:
                    self._health.bump_heartbeat(live_worker_ids, Timestamp.now().epoch_ms())

                unhealthy = self._health.workers_over_threshold()
                if unhealthy:
                    logger.warning(
                        "Ping loop: failing %d workers over ping threshold: %s",
                        len(unhealthy),
                        [str(wid) for wid in unhealthy[:10]],
                    )
                    removed = self._terminate_workers(
                        [str(wid) for wid in unhealthy],
                        reason="worker ping threshold exceeded",
                        sibling_reason="unhealthy worker failed, slice terminated",
                    )
                    self._health.forget_many(removed)

            except Exception:
                logger.exception("Ping loop iteration failed")

    def _terminate_workers(self, worker_ids: list[str], reason: str, sibling_reason: str) -> list[WorkerId]:
        """Fail the given workers, terminate their slice siblings, and kill running tasks.

        Returns the set of worker_ids that were actually removed (primary + siblings),
        so callers can drop them from in-memory state like the health tracker.
        """
        for wid in worker_ids:
            log_event("worker_failing", wid, trigger=reason)
        failure_result = ops.worker.fail(
            self._db,
            worker_ids=worker_ids,
            reason=reason,
            health=self._health,
            endpoints=self._endpoints,
            worker_attrs=self._worker_attrs,
        )
        removed: list[WorkerId] = []
        for wid, addr in failure_result.removed_workers:
            self._task_backend.on_worker_failed(wid, addr)
            removed.append(wid)
        failed_result = self._task_backend.on_workers_failed([wid for wid, _ in failure_result.removed_workers])
        persist_autoscaler_state(self._db, failed_result.state)
        if failed_result.sibling_worker_ids:
            for wid in failed_result.sibling_worker_ids:
                log_event("worker_failing", str(wid), trigger=sibling_reason)
            sibling_failures = ops.worker.fail(
                self._db,
                worker_ids=failed_result.sibling_worker_ids,
                reason=sibling_reason,
                health=self._health,
                endpoints=self._endpoints,
                worker_attrs=self._worker_attrs,
            )
            for wid, addr in sibling_failures.removed_workers:
                self._task_backend.on_worker_failed(wid, addr)
                removed.append(wid)
        # Surviving-slice siblings stop on the next reconcile tick: the
        # planner drops them from the worker's desired set (or marks them
        # 'stop'); the failed workers themselves are already gone from the
        # worker table.
        return removed

    def _run_autoscaler_once(self) -> None:
        """Run one autoscale cycle: read the snapshot (DB), drive the backend's
        ``manage_capacity``, then persist the returned state.

        Called from the autoscaler loop thread. The controller owns every DB
        read and write; the backend never touches the database.
        """
        if self._config.dry_run:
            logger.info("[DRY-RUN] Skipping autoscaler cycle (manage_capacity)")
            return

        worker_status_map = self._build_worker_status_map()
        with self._db.read_snapshot() as tx:
            workers = reads.healthy_active_workers_with_attributes(tx, self._health, self._worker_attrs)
        demand_entries = compute_demand_entries(
            self._db,
            self._scheduler,
            workers,
            reservation_claims=read_reservation_claims(self._db),
        )
        result = self._task_backend.manage_capacity(
            CapacityInput(worker_status_map=worker_status_map, demand_entries=demand_entries)
        )
        persist_autoscaler_state(self._db, result.state)

    def _build_worker_status_map(self) -> WorkerStatusMap:
        """Build a map of worker_id to worker status for autoscaler idle tracking."""
        result: WorkerStatusMap = {}
        worker_ids = {wid for wid, l in self._health.all().items() if l.active}
        with self._db.read_snapshot() as tx:
            running_by_worker = reads.running_tasks_by_worker(tx, worker_ids)
        for wid in worker_ids:
            result[wid] = WorkerStatus(
                worker_id=wid,
                running_task_ids=frozenset(tid.to_wire() for tid in running_by_worker.get(wid, set())),
            )
        return result

    def begin_checkpoint(self) -> tuple[str, CheckpointResult]:
        """Write a consistent SQLite checkpoint copy.

        The backup runs through a dedicated read-only source connection
        (see ``ControllerDB.backup_to``), so writers proceed concurrently
        under WAL semantics. Heartbeat rounds apply their updates as
        atomic batches, so each SQLite snapshot already captures a
        consistent state without needing the heartbeat lock.
        """
        if self._config.dry_run:
            logger.info("[DRY-RUN] Skipping checkpoint write")
            return ("dry-run", CheckpointResult(created_at=Timestamp.now(), job_count=0, task_count=0, worker_count=0))
        backup = backup_databases(self._db)
        try:
            path, result = upload_checkpoint(self._db, backup, self._config.remote_state_dir)
        finally:
            backup.cleanup()
        log_event(
            "checkpoint_written",
            "controller",
            path=path,
            jobs=result.job_count,
            tasks=result.task_count,
            workers=result.worker_count,
        )
        return path, result

    def launch_job(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Submit a job to the controller."""
        return self._service.launch_job(request, None)

    def get_job_status(
        self,
        job_id: str,
    ) -> controller_pb2.Controller.GetJobStatusResponse:
        """Get the status of a job."""
        request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def terminate_job(
        self,
        job_id: str,
    ) -> job_pb2.Empty:
        """Terminate a running job."""
        request = controller_pb2.Controller.TerminateJobRequest(job_id=job_id)
        return self._service.terminate_job(request, None)

    # Properties

    @property
    def provider(self) -> TaskBackend:
        return self._task_backend

    @property
    def has_direct_provider(self) -> bool:
        return self._task_backend.placement is PlacementOwner.TASK_BACKEND

    @property
    def run_template_cache(self) -> RunTemplateCache:
        """Per-job RunTaskRequest template cache, shared with the dispatch path."""
        return self._run_template_cache

    @property
    def provider_scheduling_events(self) -> list[SchedulingEvent]:
        return self._provider_scheduling_events

    @property
    def provider_capacity(self) -> ClusterCapacity | None:
        return self._provider_capacity

    @property
    def port(self) -> int:
        """Actual bound port (may differ from config if port=0 was specified)."""
        if self._server and self._server.started:
            if self._server.servers and self._server.servers[0].sockets:
                return self._server.servers[0].sockets[0].getsockname()[1]
        return self._config.port

    @property
    def external_host(self) -> str:
        """Externally-reachable host address.

        When bound to 0.0.0.0, probes for the real network IP via
        ``probe_outbound_ip``.
        """
        return resolve_external_host(self._config.host)

    @property
    def url(self) -> str:
        return f"http://{self.external_host}:{self.port}"

    @property
    def reservation_claims(self) -> dict[WorkerId, ReservationClaim]:
        """Current reservation claims, keyed by worker ID."""
        return read_reservation_claims(self._db)

    @property
    def autoscaler(self) -> Autoscaler | None:
        """The Iris autoscaler driving capacity for this backend, if any.

        Read-only handle for dashboard/status RPCs (VM info, feasibility,
        pending hints). Capacity is driven through ``backend.manage_capacity``,
        not this handle.
        """
        return self._task_backend.autoscaler
