# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import asyncio
import atexit
import enum
import logging
import sys
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
from rigging.timing import Duration, ExponentialBackoff, RateLimiter, Timer, Timestamp, TokenBucket
from sqlalchemy import bindparam, select

from iris.cluster.bundle import BundleStore
from iris.cluster.controller import direct_provider, ops, reads, writes
from iris.cluster.controller.audit import log_event
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.checkpoint import (
    CheckpointResult,
    backup_databases,
    upload_checkpoint,
    write_checkpoint,
)
from iris.cluster.controller.codec import (
    reservation_entries_from_json,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.direct_provider import (
    DIRECT_PROVIDER_PROMOTION_RATE,
    ClusterCapacity,
    RunTemplateCache,
    SchedulingEvent,
)
from iris.cluster.controller.ops.task import (
    Assignment,
    apply_terminal_decisions,
)
from iris.cluster.controller.ops.task import (
    apply_provider_updates as apply_direct_provider_updates,
)
from iris.cluster.controller.ops.worker import (
    apply_reconcile_observations as apply_reconcile,
)
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.provider import TaskProvider
from iris.cluster.controller.pruner import prune_old_data
from iris.cluster.controller.reads import ReservationClaim
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
    reconcile_workers,
)
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingContext,
)
from iris.cluster.controller.scheduling_policy import (
    GatedCandidates,
    PreemptionCandidate,
    SchedulingOrder,
    _get_running_tasks_with_band_and_value,
    _inject_reservation_taints,
    _inject_taint_constraints,
    _job_state_by_id,
    _jobs_with_reservations,
    _preference_pass,
    _read_reservation_claims,
    _run_preemption_pass,
    _worker_matches_reservation_entry,
    apply_scheduling_gates,
    build_scheduling_context,
    compute_demand_entries,
    compute_scheduling_order,
)
from iris.cluster.controller.schema import (
    job_config_table,
    task_attempts_table,
    tasks_table,
    workers_table,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.task_state import hint_rare_state
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.log_keys import CONTROLLER_LOG_KEY
from iris.cluster.providers.k8s.tasks import K8sTaskProvider
from iris.cluster.providers.types import find_free_port, resolve_external_host
from iris.cluster.runtime.profile import PROFILE_NAMESPACE, IrisProfile
from iris.cluster.types import (
    JobName,
    UserBudgetDefaults,
    WorkerId,
    WorkerStatus,
    WorkerStatusMap,
    is_job_finished,
)
from iris.cluster.worker.stats import TASK_STATS_NAMESPACE, IrisTaskStat
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import AuthTokenInjector, NullAuthInterceptor, StaticTokenProvider, TokenVerifier

logger = logging.getLogger(__name__)

# Sentinel for dry-run scheduling with per-worker limits disabled.
_UNLIMITED = sys.maxsize

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

    poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(1.0))
    """Polling reconcile cadence. The polling thread wakes every ``poll_interval``
    (or sooner if ``_polling_wake`` is set) and runs ``_reconcile_worker_batch``
    against every healthy worker. The Reconcile RPC is the sole channel that
    dispatches new ASSIGNED rows and observes worker-side state changes."""

    max_dispatch_parallelism: int = 32
    """Maximum number of concurrent RPC dispatch operations."""

    max_tasks_per_job_per_cycle: int = 4
    """Maximum tasks from a single non-coscheduled job to consider per scheduling
    cycle. Bounds CPU time in the scheduler when many tasks are pending, preventing
    GIL starvation of the heartbeat thread. Coscheduled jobs are exempt (they need
    all tasks for atomic assignment). Set to 0 for unlimited."""

    autoscaler_enabled: bool = False
    worker_access_address: str = ""

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
    - Provider loop: syncs task state with the execution backend via TaskProvider
    - Autoscaler loop: evaluates scaling decisions, manages slice lifecycle

    Each loop runs on its own thread so blocking operations in one don't
    stall the others.

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(
            config=config,
            provider=WorkerProvider(stub_factory=RpcWorkerStubFactory()),
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
        provider: TaskProvider for communicating with the execution backend
        autoscaler: Optional Autoscaler for managing VM slices. If provided,
                   the controller will run it in a background thread.
    """

    def __init__(
        self,
        config: ControllerConfig,
        provider: TaskProvider | K8sTaskProvider,
        autoscaler: "Autoscaler | None" = None,
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
        self._provider: TaskProvider | K8sTaskProvider = provider
        self._provider_scheduling_events: list[SchedulingEvent] = []
        self._provider_capacity: ClusterCapacity | None = None
        self._promotion_bucket = TokenBucket(
            capacity=DIRECT_PROVIDER_PROMOTION_RATE,
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
        if isinstance(self._provider, K8sTaskProvider):
            k8s_log_client = LogClient.connect(self._log_service_address, interceptors=log_client_interceptors)
            self._provider.log_client = k8s_log_client
            self._provider.task_stats_table = k8s_log_client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat)
            self._provider.profile_table = k8s_log_client.get_table(PROFILE_NAMESPACE, IrisProfile)

        # Controller process logs ship to the log server via RemoteLogHandler.
        self._log_client = LogClient.connect(self._log_service_address, interceptors=log_client_interceptors)
        self._log_handler = RemoteLogHandler(self._log_client, key=CONTROLLER_LOG_KEY)

        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger("iris").addHandler(self._log_handler)

        self._run_template_cache: RunTemplateCache = RunTemplateCache()
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
        self._direct_provider_thread: ManagedThread | None = None
        self._autoscaler_thread: ManagedThread | None = None
        self._prune_thread: ManagedThread | None = None
        self._ping_thread: ManagedThread | None = None

        self._autoscaler: Autoscaler | None = autoscaler

        # Throttles the execution-timeout deadline scan in _reconcile_worker_batch.
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

        if isinstance(self._provider, K8sTaskProvider):
            self._direct_provider_thread = self._threads.spawn(self._run_direct_provider_loop, name="provider-loop")
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

        if self._autoscaler:
            logger.info("Autoscaler configured with %d scale groups", len(self._autoscaler.groups))
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
        if self._direct_provider_thread:
            self._direct_provider_thread.stop()
            self._direct_provider_thread.join(timeout=join_timeout)
        if self._ping_thread:
            self._ping_thread.stop()
            self._ping_thread.join(timeout=join_timeout)
        if self._prune_thread:
            self._prune_thread.stop()
            self._prune_thread.join(timeout=join_timeout)
        if self._autoscaler_thread:
            self._autoscaler_thread.stop()
            self._autoscaler_thread.join(timeout=join_timeout)

        if self._autoscaler:
            self._autoscaler.shutdown()

        self._threads.stop()
        self._provider.close()

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
                self._reconcile_worker_batch()
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
        limiter = RateLimiter(interval_seconds=self._autoscaler.evaluation_interval.to_seconds())
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

    def _run_direct_provider_loop(self, stop_event: threading.Event) -> None:
        """Provider sync loop for K8sTaskProvider: no scheduling, no workers."""
        limiter = RateLimiter(interval_seconds=self._config.heartbeat_interval.to_seconds())
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            try:
                self._sync_direct_provider()
            except Exception:
                logger.exception("Direct provider sync round failed, will retry next interval")

    def _sync_direct_provider(self) -> None:
        if self._config.dry_run:
            return
        assert isinstance(self._provider, K8sTaskProvider)
        provider = self._provider
        max_promotions = self._promotion_bucket.available
        with self._db.transaction() as cur:
            batch = direct_provider.drain_for_direct_provider(
                cur,
                cache=self._run_template_cache,
                max_promotions=max_promotions,
            )
        if batch.tasks_to_run:
            self._promotion_bucket.try_acquire(len(batch.tasks_to_run))
        result = provider.sync(batch)
        with self._db.transaction() as cur:
            apply_direct_provider_updates(
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

    def _cleanup_stale_claims(self, claims: dict[WorkerId, ReservationClaim] | None = None) -> bool:
        """Remove claims for workers that disappeared or jobs that finished."""
        persisted = False
        if claims is None:
            claims = _read_reservation_claims(self._db)
            persisted = True
        active_worker_ids = {wid for wid, l in self._health.all().items() if l.active}
        claimed_job_ids = {JobName.from_wire(claim.job_id) for claim in claims.values()}
        # Only job.state is needed here; use the thin 2-column query.
        job_states = _job_state_by_id(self._db, claimed_job_ids)
        stale: list[WorkerId] = []
        for worker_id, claim in claims.items():
            if worker_id not in active_worker_ids:
                stale.append(worker_id)
                continue
            job_state = job_states.get(JobName.from_wire(claim.job_id))
            if job_state is None or is_job_finished(job_state):
                stale.append(worker_id)
        for wid in stale:
            del claims[wid]
        if stale and persisted:
            with self._db.transaction() as cur:
                writes.replace_reservation_claims(cur, claims)
            log_event("reservation_claims_cleaned", "controller", count=len(stale))
        return bool(stale)

    def _claim_workers_for_reservations(self, claims: dict[WorkerId, ReservationClaim] | None = None) -> bool:
        """Assign unclaimed workers to unsatisfied reservation entries.

        Scans all non-finished jobs with reservations. For each unfulfilled
        entry, finds an eligible unclaimed worker and records the claim.
        """
        persisted = False
        if claims is None:
            claims = _read_reservation_claims(self._db)
            persisted = True
        claimed_entries: set[tuple[str, int]] = {(c.job_id, c.entry_idx) for c in claims.values()}
        claimed_worker_ids: set[WorkerId] = set(claims.keys())
        with self._db.read_snapshot() as tx:
            all_workers = reads.healthy_active_workers_with_attributes(tx, self._health, self._worker_attrs)
        changed = False

        reservable_states = (
            job_pb2.JOB_STATE_PENDING,
            job_pb2.JOB_STATE_BUILDING,
            job_pb2.JOB_STATE_RUNNING,
        )
        reservation_jobs = _jobs_with_reservations(self._db, reservable_states)
        for job in reservation_jobs:
            job_wire = job.job_id.to_wire()
            for idx, res_entry in enumerate(reservation_entries_from_json(job.reservation_json)):
                if (job_wire, idx) in claimed_entries:
                    continue

                for worker in all_workers:
                    if worker.worker_id in claimed_worker_ids:
                        continue
                    if not _worker_matches_reservation_entry(worker, res_entry):
                        continue

                    claims[worker.worker_id] = ReservationClaim(
                        job_id=job_wire,
                        entry_idx=idx,
                    )
                    claimed_worker_ids.add(worker.worker_id)
                    claimed_entries.add((job_wire, idx))
                    changed = True
                    break
        if changed and persisted:
            with self._db.transaction() as cur:
                writes.replace_reservation_claims(cur, claims)
            log_event("reservation_claims_updated", "controller", total_claims=len(claims))
        return changed

    def _run_scheduling(self) -> SchedulingOutcome:
        """Run one scheduling cycle.

        Six-phase scheduling:
        1. Reservation claims: clean up stale claims and claim workers for
           reservation jobs.
        2. State reads: fetch pending tasks and workers, filter by deadlines,
           reservation gates, and per-job cap.
        3. Budget/band interleaving: compute user spend, map tasks to effective
           priority bands (down-weighting over-budget users), round-robin users
           within each band.
        4. Preference pass: steer reservation tasks toward their claimed workers
           (skips coscheduled jobs which need atomic assignment).
        5. Normal scheduling: run find_assignments for all remaining tasks.
        6. Preemption pass: evict lower-priority running tasks to free capacity
           for higher-priority unscheduled work.

        Phases 4-6 share a single SchedulingContext so capacity deductions
        are visible across passes.

        No lock is needed since only one scheduling thread exists. Every DB
        access is serialized by ControllerDB._lock with multi-statement
        mutations wrapped in BEGIN IMMEDIATE transactions.
        """
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0

        claims = self._refresh_reservation_claims()

        timer = Timer()
        ctx = build_scheduling_context(
            self._db,
            self._health,
            self._worker_attrs,
            self._config.user_budget_defaults,
        )

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

        gated = apply_scheduling_gates(
            ctx,
            claims,
            max_tasks_per_job_per_cycle=self._config.max_tasks_per_job_per_cycle,
            trace=trace,
        )
        # Mark deadline-expired tasks UNSCHEDULABLE — kept out of the pure
        # gate evaluation so the gate stays free of DB writes.
        if gated.expired_tasks:
            self._mark_tasks_unschedulable(list(gated.expired_tasks))

        if not gated.schedulable_task_ids:
            self._scheduling_diagnostics = {}
            self._last_scheduling_context = ctx
            return SchedulingOutcome.NO_PENDING_TASKS

        order = compute_scheduling_order(ctx, gated, trace=trace)

        all_assignments, context, tainted_jobs = self._run_scheduler_pass(order, gated, ctx, claims, timer, trace=trace)

        preemptions = self._apply_preemptions(order, tainted_jobs, all_assignments, claims, context)

        self._cache_scheduling_diagnostics(context, tainted_jobs, all_assignments, order.ordered_task_ids)
        # Post-taint context (or the un-tainted ctx when no claims were active)
        # — exposed via ``last_scheduling_context`` for dashboard diagnostics.
        self._last_scheduling_context = context

        if all_assignments or preemptions:
            log_event(
                "scheduling_pass_completed",
                "scheduler",
                assignments=len(all_assignments),
                preempted=len(preemptions),
                pending=len(ctx.pending_task_rows),
                workers=len(ctx.workers),
            )
            return SchedulingOutcome.ASSIGNMENTS_MADE
        return SchedulingOutcome.NO_ASSIGNMENTS

    def _refresh_reservation_claims(self) -> dict[WorkerId, ReservationClaim]:
        """Read, clean up, and refresh reservation claims. Returns updated claims."""
        # Claims are read outside the scheduling transaction. This creates a
        # narrow race window where a worker could be removed between claim reads
        # and scheduling, but it's benign: queue_assignments() re-validates all
        # assignments transactionally, and stale claims are cleaned up next cycle.
        claims = _read_reservation_claims(self._db)
        claims_changed = self._cleanup_stale_claims(claims)
        claims_changed = self._claim_workers_for_reservations(claims) or claims_changed
        if claims_changed:
            if self._config.dry_run:
                logger.info("[DRY-RUN] Would update %d reservation claims", len(claims))
            else:
                with self._db.transaction() as cur:
                    writes.replace_reservation_claims(cur, claims)
        return claims

    def _run_scheduler_pass(
        self,
        order: SchedulingOrder,
        gated: GatedCandidates,
        ctx: SchedulingContext,
        claims: dict[WorkerId, ReservationClaim],
        timer: Timer,
        trace: bool = False,
    ) -> tuple[list[tuple[JobName, WorkerId]], SchedulingContext, dict[JobName, JobRequirements]]:
        """Run preference + normal assignment passes.

        Reservation taints are injected here so gates/order/diagnostics see
        un-tainted workers. When there are no claims we reuse ``ctx`` directly
        to avoid an index rebuild.
        """
        modified_jobs = _inject_taint_constraints(gated.jobs, gated.has_reservation, gated.has_direct_reservation)

        if claims:
            modified_workers = _inject_reservation_taints(list(ctx.workers), claims)
            building_counts = {wid: cap.building_task_count for wid, cap in ctx.capacities.items()}
            ctx.pending_tasks = list(order.ordered_task_ids)
            context = ctx.evolve_with_workers(
                workers=modified_workers,
                jobs=modified_jobs,
                building_counts=building_counts,
                max_building_tasks=self._scheduler.max_building_tasks_per_worker,
            )
        else:
            ctx.pending_tasks = list(order.ordered_task_ids)
            ctx.jobs = modified_jobs
            context = ctx

        if trace:
            logger.info(
                "[TRACE] Phase 4 context: %d workers, %d pending tasks, %d jobs",
                len(context.capacities),
                len(context.pending_tasks),
                len(context.jobs),
            )

        # Soft preference — steer reservation tasks toward claimed workers.
        # Skips coscheduled jobs (they need atomic all-or-nothing via find_assignments).
        preference_assignments = _preference_pass(context, gated.has_reservation, claims)

        result = self._scheduler.find_assignments(context)

        all_assignments = preference_assignments + result.assignments
        if trace:
            logger.info(
                "[TRACE] Phase 5 assignments: %d total (%d preferred, %d normal)",
                len(all_assignments),
                len(preference_assignments),
                len(result.assignments),
            )
        if all_assignments:
            self._commit_assignments(all_assignments, order.task_band_map)
            logger.debug(
                "Scheduling cycle: %d assignments (%d preferred, %d normal), %dms",
                len(all_assignments),
                len(preference_assignments),
                len(result.assignments),
                timer.elapsed_ms(),
            )
        return all_assignments, context, modified_jobs

    def _commit_assignments(
        self,
        assignments: list[tuple[JobName, WorkerId]],
        task_band_map: dict[JobName, int],
    ) -> None:
        """Persist scheduler decisions to ``tasks.state = ASSIGNED`` rows.

        Each assignment carries the effective priority band from
        ``task_band_map`` (computed against the snapshot's user spend) so
        ``assign_task`` can stamp it onto ``tasks.priority_band``. The
        preemption pass then trusts that stamped value instead of
        recomputing from current spend on every tick.

        The polling reconcile thread reads ASSIGNED rows on its next tick
        (woken via ``_polling_wake``) and fans out the Reconcile RPCs.
        """
        if self._config.dry_run:
            for task_id, worker_id in assignments:
                logger.info("[DRY-RUN] Would assign task %s to worker %s", task_id, worker_id)
            return
        command = [
            Assignment(
                task_id=task_id,
                worker_id=worker_id,
                priority_band=task_band_map.get(task_id),
            )
            for task_id, worker_id in assignments
        ]
        with self._db.transaction() as cur:
            ops.task.queue_assignments(cur, command, health=self._health)
        # Wake the polling thread; every tick reconciles every healthy worker,
        # so the new ASSIGNED rows turn into Reconcile RPCs on the next tick.
        self._polling_wake.set()

    def _apply_preemptions(
        self,
        order: SchedulingOrder,
        jobs: dict[JobName, JobRequirements],
        all_assignments: list[tuple[JobName, WorkerId]],
        claims: dict[WorkerId, ReservationClaim],
        context: SchedulingContext,
    ) -> list[tuple[JobName, JobName]]:
        """Evict lower-priority running tasks for higher-priority unscheduled work."""
        assigned_ids = {task_id for task_id, _ in all_assignments}
        unscheduled = [
            PreemptionCandidate(
                job_name=tid,
                requirements=jobs[tid.parent],
                band=order.task_band_map.get(tid, job_pb2.PRIORITY_BAND_INTERACTIVE),
            )
            for tid in order.ordered_task_ids
            if tid not in assigned_ids and tid.parent is not None and tid.parent in jobs
        ]
        preemptions: list[tuple[JobName, JobName]] = []
        if unscheduled:
            claimed_workers = set(claims.keys())
            running_info = _get_running_tasks_with_band_and_value(self._db, claimed_workers)
            preemptions = _run_preemption_pass(unscheduled, running_info, context)
            # Apply all preemptions in one transaction so slice evictions
            # (N siblings of a coscheduled preemptor) are all-or-nothing.
            if preemptions:
                with self._db.transaction() as cur:
                    now = Timestamp.now()
                    apply_terminal_decisions(
                        cur,
                        [
                            TerminalDecision(
                                kind=TerminalKind.PREEMPT,
                                task_id=victim_id,
                                reason=f"Preempted by {preemptor_name}",
                            )
                            for preemptor_name, victim_id in preemptions
                        ],
                        health=self._health,
                        endpoints=self._endpoints,
                        now=now,
                    )
                # Killed-task RPCs land on the next polling tick via the
                # worker's expected_tasks diff.
                logger.info("Preemption pass: %d tasks preempted", len(preemptions))
        return preemptions

    def _cache_scheduling_diagnostics(
        self,
        context: SchedulingContext,
        jobs: dict[JobName, JobRequirements],
        assignments: list[tuple[JobName, WorkerId]],
        schedulable_task_ids: list[JobName],
    ) -> None:
        """Compute and cache scheduling diagnostics for unassigned jobs."""
        assigned_task_ids = {task_id for task_id, _ in assignments}

        # Find unassigned jobs with a representative task
        unscheduled: dict[JobName, tuple[JobName, int]] = {}
        for task_id in schedulable_task_ids:
            if task_id in assigned_task_ids or task_id.parent is None:
                continue
            job_id = task_id.parent
            if job_id in unscheduled:
                _, count = unscheduled[job_id]
                unscheduled[job_id] = (unscheduled[job_id][0], count + 1)
            else:
                unscheduled[job_id] = (task_id, 1)

        diagnostics: dict[str, str] = {}
        for job_id, (representative_task, num_tasks) in unscheduled.items():
            req = jobs.get(job_id)
            if req is None:
                continue
            reason = self._scheduler.get_job_scheduling_diagnostics(
                req,
                context,
                representative_task,
                num_tasks=num_tasks,
            )
            diagnostics[job_id.to_wire()] = reason

        # Atomic replacement — safe for concurrent reads under the GIL.
        self._scheduling_diagnostics = diagnostics

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
            apply_terminal_decisions(
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
            # heartbeat path can stamp finished_at_ms. Without this, a single
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
                    templates_by_job[row.job_id] = direct_provider.run_request_template(
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

    def _reconcile_worker_batch(self) -> None:
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

        plans = reconcile_workers(inputs) if inputs.worker_ids else []
        results = self._provider.reconcile_workers(plans, addresses) if plans else []

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
                apply_terminal_decisions(
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
                results = self._provider.ping_workers(workers)

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
            self._provider.on_worker_failed(wid, addr)
            removed.append(wid)
        if self._autoscaler:
            sibling_worker_ids = self._autoscaler.terminate_slices_for_workers(
                [str(wid) for wid, _ in failure_result.removed_workers]
            )
            for wid in sibling_worker_ids:
                log_event("worker_failing", str(wid), trigger=sibling_reason)
            sibling_failures = ops.worker.fail(
                self._db,
                worker_ids=sibling_worker_ids,
                reason=sibling_reason,
                health=self._health,
                endpoints=self._endpoints,
                worker_attrs=self._worker_attrs,
            )
            for wid, addr in sibling_failures.removed_workers:
                self._provider.on_worker_failed(wid, addr)
                removed.append(wid)
        # Surviving-slice siblings get killed on the next polling tick via
        # the worker's expected_tasks diff; the failed workers themselves
        # are already gone from the worker table.
        return removed

    def _run_autoscaler_once(self) -> None:
        """Run one autoscaler cycle: refresh (I/O) then update (CPU).

        Called from the autoscaler loop thread.
        """
        if not self._autoscaler:
            return

        if self._config.dry_run:
            logger.info("[DRY-RUN] Skipping autoscaler cycle (refresh + update)")
            return

        worker_status_map = self._build_worker_status_map()
        self._autoscaler.refresh(worker_status_map)
        self._autoscaler.probe_health()
        with self._db.read_snapshot() as tx:
            workers = reads.healthy_active_workers_with_attributes(tx, self._health, self._worker_attrs)
        demand_entries = compute_demand_entries(
            self._db,
            self._scheduler,
            workers,
            reservation_claims=_read_reservation_claims(self._db),
        )
        self._autoscaler.update(demand_entries)

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
    def provider(self) -> TaskProvider | K8sTaskProvider:
        return self._provider

    @property
    def has_direct_provider(self) -> bool:
        return isinstance(self._provider, K8sTaskProvider)

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
        return _read_reservation_claims(self._db)

    @property
    def autoscaler(self) -> "Autoscaler | None":
        """The autoscaler instance, if autoscaling is enabled."""
        return self._autoscaler
