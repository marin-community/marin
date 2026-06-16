# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import asyncio
import atexit
import dataclasses
import enum
import logging
import socket
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from finelog.client import LogClient, RemoteLogHandler
from finelog.client.proxy import LogServiceProxy, StatsServiceProxy
from finelog.embedded import require_embedded_server
from rigging.timing import Duration, ExponentialBackoff, RateLimiter, Timestamp, TokenBucket
from sqlalchemy import Row

from iris.cluster.backends.types import resolve_external_host
from iris.cluster.bundle import BundleStore
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.persistence import persist_autoscaler_state
from iris.cluster.controller.backend import (
    AutoscaleResult,
    BackendCapability,
    ReconcileResult,
    ScheduleInput,
    ScheduleResult,
    TaskBackend,
)
from iris.cluster.controller.checkpoint import (
    CheckpointResult,
    backup_databases,
    upload_checkpoint,
    write_checkpoint,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB, Tx
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
from iris.cluster.controller.reconcile import ControllerEffects, dispatch
from iris.cluster.controller.reconcile.dispatch import (
    DISPATCH_PROMOTION_RATE,
)
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import ReconcileRow
from iris.cluster.controller.run_template import RunTemplateCache, new_run_template_cache
from iris.cluster.controller.scheduling.policy import (
    build_scheduling_context,
    read_reservation_claims,
    refresh_reservation_claims,
    refresh_reservation_claims_in_tx,
)
from iris.cluster.controller.scheduling.scheduler import (
    SchedulingContext,
)
from iris.cluster.controller.schema import ReservationClaim
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind, WorkerHealthTracker
from iris.cluster.log_keys import CONTROLLER_LOG_KEY
from iris.cluster.runtime.profile import PROFILE_NAMESPACE, IrisProfile
from iris.cluster.types import (
    JobName,
    PendingTask,
    UserBudgetDefaults,
    WorkerId,
    WorkerStatus,
    WorkerStatusMap,
    WorkerUsability,
)
from iris.cluster.worker.stats import TASK_STATS_NAMESPACE, IrisTaskStat
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider, TokenVerifier

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

    def run_with_executor(sockets: list[socket.socket] | None = None) -> None:
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
class _TickInputs:
    """DB-less per-tick inputs the control driver assembles in one read txn.

    Only the sections for the phases due this tick are populated; the rest stay
    at their empty defaults. ``ctx``/``claims`` feed the schedule phase,
    ``control`` the reconcile phase, ``worker_status_map`` the autoscale phase.
    """

    ctx: SchedulingContext | None = None
    claims: dict[WorkerId, ReservationClaim] = field(default_factory=dict)
    claims_changed: bool = False
    control: "reads.ControlSnapshot" = field(
        default_factory=lambda: reads.ControlSnapshot(worker_addresses={}, reconcile_rows=[], timeout_rows=[])
    )
    worker_status_map: WorkerStatusMap = field(default_factory=dict)


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
    """Schedule-phase cadence: the control tick runs its schedule phase at most
    this often (a submit wake still forces an immediate schedule-only mini-tick)."""

    autoscaler_evaluation_interval: Duration = field(default_factory=lambda: Duration.from_seconds(10.0))
    """How often the controller runs an autoscale provisioning cycle
    (``backend.autoscale``). A capacity-managing backend (k8s) no-ops."""

    poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(1.0))
    """Reconcile cadence — the sole reconcile + liveness channel. The control
    tick runs its reconcile phase every ``poll_interval`` (or sooner when a fresh
    assignment forces one) against every active worker. The reconcile RPC outcome
    is the only liveness signal; ``worker_unreachable_grace`` sets how long a
    worker may stay unreachable before teardown. The Reconcile RPC is also the
    sole channel that dispatches new ASSIGNED rows and observes worker state."""

    worker_unreachable_grace: Duration = field(default_factory=lambda: Duration.from_seconds(50.0))
    """How long a worker may be continuously unreachable (or self-report
    unhealthy) before the controller fails and tears it down. Realized as a count
    of consecutive failed reconcile passes — ``round(grace / poll_interval)`` —
    so detection latency stays fixed regardless of the reconcile cadence. ~50s
    tolerates brief network blips without reaping a multi-VM slice; tests shorten
    it for fast deterministic teardown."""

    max_tasks_per_job_per_cycle: int = 4
    """Maximum tasks from a single non-coscheduled job to consider per scheduling
    cycle. Bounds CPU time in the scheduler when many tasks are pending, preventing
    GIL starvation of the heartbeat thread. Coscheduled jobs are exempt (they need
    all tasks for atomic assignment). Set to 0 for unlimited."""

    checkpoint_interval: Duration | None = None
    """If set, take a periodic best-effort snapshot this often.
    Runs on its own checkpoint thread; does not pause the control tick."""

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
    When set, the controller connects to the existing server. When None, the
    Controller starts an in-process native ``finelog_server`` server on a free
    port (used by tests and local-mode runs). In production this address is
    sourced from `endpoints["/system/log-server"]` and passed in here by the
    daemon entrypoint."""

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

    One driver thread runs the control tick — schedule -> reconcile -> autoscale
    as phases over a single read snapshot, committed through one end-of-tick write
    transaction — alongside the prune and checkpoint housekeeping threads.

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
        provider: TaskBackend for communicating with the execution backend. The
            controller drives its ``autoscale`` from the control tick's autoscale
            phase and persists the returned state.
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
        # A cluster backend that owns placement (no Iris scheduler) needs the
        # reconcile tick to drain pending dispatch (promote PENDING→ASSIGNED) and
        # ride it on the snapshot; a worker-daemon backend reconciles the
        # already-scheduled worker-bound rows. Resolved once from capability.
        self._backend_drains_dispatch = BackendCapability.CLUSTER_VIEW in provider.capabilities
        self._promotion_bucket = TokenBucket(
            capacity=DISPATCH_PROMOTION_RATE,
            refill_period=Duration.from_minutes(1),
        )

        config.local_state_dir.mkdir(parents=True, exist_ok=True)
        if db is not None:
            self._db = db
        else:
            self._db = ControllerDB(db_dir=config.local_state_dir / "db")
        # Detection latency is fixed in wall-clock by worker_unreachable_grace and
        # converted to a consecutive-failure count for the reconcile cadence, so it
        # is unaffected by poll_interval.
        reconcile_failure_threshold = max(
            1, round(config.worker_unreachable_grace.to_seconds() / config.poll_interval.to_seconds())
        )
        self._health = WorkerHealthTracker(reconcile_failure_threshold=reconcile_failure_threshold)
        self._endpoints = EndpointsProjection(self._db)
        self._worker_attrs = WorkerAttrsProjection(self._db)
        writes.validate()
        self._seed_liveness_from_workers()
        self._db.register_reopen_hook(self._seed_liveness_from_workers)

        self._threads = threads if threads is not None else get_thread_container()

        # --- Log service setup ---
        # The log server is always accessed via RPC. In production the
        # controller's main() starts a subprocess; in tests/local mode the
        # Controller spins up an in-process native finelog server
        # (finelog_server). After the server is running, all access goes
        # through RPC clients — no branching on hosting mode.
        self._log_server: Any = None  # finelog_server.EmbeddedServer when started locally

        if config.log_service_address:
            self._log_service_address = config.log_service_address
        else:
            self._log_service_address = self._start_local_log_server()

        log_client_interceptors = _log_client_interceptors(config)
        self._remote_log_service = LogServiceProxy(self._log_service_address, interceptors=log_client_interceptors)
        self._remote_stats_service = StatsServiceProxy(self._log_service_address, interceptors=log_client_interceptors)

        # A single log client serves both the controller's own logs and any backend
        # that collects logs out-of-process.
        self._log_client = LogClient.connect(self._log_service_address, interceptors=log_client_interceptors)

        # Backends without a worker daemon push per-task resource/profile samples to the
        # log server directly; daemon-backed backends (RPC) ignore the sink.
        self._task_backend.set_log_sink(
            self._log_client,
            self._log_client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat),
            self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile),
        )

        self._log_handler = RemoteLogHandler(self._log_client, key=CONTROLLER_LOG_KEY)

        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger("iris").addHandler(self._log_handler)

        self._run_template_cache: RunTemplateCache = new_run_template_cache()

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

        # Wakes the control-tick driver. A submit triggers a schedule-only
        # mini-tick so submit->assign latency is the schedule time, not gated on
        # the next reconcile cadence.
        self._tick_wake = threading.Event()
        # Set after a tick commits new ASSIGNED rows so the next tick reconciles
        # immediately (dispatching them) instead of waiting a full poll interval.
        self._force_reconcile = False
        self._server: uvicorn.Server | None = None
        self._control_thread: ManagedThread | None = None
        self._prune_thread: ManagedThread | None = None
        self._checkpoint_thread: ManagedThread | None = None

        # Throttles the execution-timeout deadline scan in the reconcile phase.
        # The reconcile phase runs frequently (poll cadence); the timeout query
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
        """Wake the control tick to run a schedule-only mini-tick immediately.

        Called on new job submission so the next tick picks up the new pending
        tasks (and a fresh assignment then forces the following reconcile) instead
        of waiting a full poll interval.
        """
        self._tick_wake.set()

    def _seed_liveness_from_workers(self) -> None:
        """Seed every persisted worker as healthy so the scheduler sees them at startup.

        Liveness is in-memory and reseeded from the worker rows on restart;
        workers that then go unreachable accrue failures through the reconcile
        health-event fold and are torn down once over threshold. ``find_prunable``
        relies on this seed to maintain the invariant that every ``workers`` row
        has a tracker entry.
        """
        now_ms = Timestamp.now().epoch_ms()
        with self._db.read_snapshot() as q:
            worker_ids = reads.all_worker_ids(q)
        if worker_ids:
            self._health.heartbeat(worker_ids, now_ms)

    @property
    def started(self) -> bool:
        """Whether the controller loops have been started."""
        return self._started

    def _start_local_log_server(self) -> str:
        """Start a bundled in-process log + stats server and return its address.

        Used as a fallback when ``cluster_config.endpoints`` does not declare
        ``/system/log-server`` (and in tests). Backed by the native
        ``finelog_server`` module (the same engine the ``finelog-server`` binary
        runs), storing segments under ``local_state_dir/log-server`` so written
        logs are queryable: the engine's in-memory mode spawns no maintenance
        task, so its RAM buffer never flushes to a readable segment — only a
        disk-backed store serves reads. The server is bound and ready when the
        constructor returns. For any deployment that needs scale or durability
        beyond the controller's local disk, run ``finelog-server`` out-of-band
        and point ``endpoints["/system/log-server"]`` at it.
        """
        log_dir = self._config.local_state_dir / "log-server"
        embedded_server_cls = require_embedded_server()
        self._log_server = embedded_server_cls(log_dir=str(log_dir), host=self._config.host)

        address = f"http://{self.external_host}:{self._log_server.port}"
        logger.info("Local log server ready at %s (log_dir=%s)", address, log_dir)
        return address

    def start(self) -> None:
        """Start the dashboard server and the control + housekeeping threads.

        Every backend gets the same threads; each phase no-ops where it does not
        apply. The unified control tick drives schedule -> reconcile -> autoscale;
        the reconcile phase is the sole reconcile + liveness channel — it
        reconciles every active worker (worker-daemon backends) or drains + syncs
        pods (cluster backends), folds the backend's observed health events, and
        tears down workers that cross the failure threshold.
        """
        self._started = True
        if self._config.dry_run:
            logger.info("[DRY-RUN] Controller started in dry-run mode — all side effects suppressed")

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

        # Register cluster endpoints BEFORE spawning the control loop. Otherwise
        # the autoscale phase's first tick can create buffer slices whose workers
        # query the controller for /system/log-server before this dict is
        # populated, returning an empty result. The slice creation fails, the
        # group enters backoff, and any task constrained to that group hangs until
        # the backoff expires.
        for name, url in self._config.endpoints.items():
            self._service._system_endpoints[name] = url
            logger.info("Registered system endpoint %s -> %s", name, url)
        self._service._system_endpoints["/system/log-server"] = self._log_service_address

        # One driver runs schedule -> reconcile -> autoscale as phases of a single
        # tick (one read snapshot + one end-of-tick commit). Spawned after endpoint
        # registration because its first autoscale phase may provision buffer slices
        # whose workers query /system/log-server. In dry-run it runs the schedule
        # phase only.
        self._control_thread = self._threads.spawn(self._run_control_loop, name="control-loop")

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
        2. Stop the control loop so no new work is triggered.
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
        self._tick_wake.set()
        join_timeout = Duration.from_seconds(5.0)
        if self._control_thread:
            self._control_thread.stop()
            self._control_thread.join(timeout=join_timeout)
        if self._prune_thread:
            self._prune_thread.stop()
            self._prune_thread.join(timeout=join_timeout)
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
        if self._log_server is not None:
            self._log_server.stop()
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

    def _run_checkpoint_loop(self, stop_event: threading.Event) -> None:
        """Periodic checkpoint loop: runs on its own thread so the multi-second
        backup+upload doesn't stall the control tick cadence."""
        limiter = self._periodic_checkpoint_limiter
        assert limiter is not None, "checkpoint loop spawned without configured limiter"
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            try:
                write_checkpoint(self._db, self._config.remote_state_dir)
            except Exception:
                logger.exception("Periodic checkpoint failed")

    # =========================================================================
    # Unified control tick
    # =========================================================================

    def _run_control_loop(self, stop_event: threading.Event) -> None:
        """Single driver: schedule -> reconcile -> autoscale as phases of one tick.

        Each iteration builds one read snapshot, runs the phases that are due (or,
        on a wake, a schedule-only mini-tick), folds backend-observed health, and
        commits through a single end-of-tick write transaction. Wakes every
        ``poll_interval`` (the reconcile cadence) or sooner on a submit/wake, so
        the per-phase cadences match the legacy three-loop structure.
        """
        base_interval = self._config.poll_interval.to_seconds()
        schedule_limiter = RateLimiter(interval_seconds=self._config.scheduler_min_interval.to_seconds())
        reconcile_limiter = RateLimiter(interval_seconds=self._config.poll_interval.to_seconds())
        autoscale_limiter = RateLimiter(interval_seconds=self._config.autoscaler_evaluation_interval.to_seconds())
        while not stop_event.is_set():
            woken = self._tick_wake.wait(timeout=base_interval)
            self._tick_wake.clear()
            if stop_event.is_set():
                break
            try:
                self._control_tick(
                    woken=woken,
                    schedule_limiter=schedule_limiter,
                    reconcile_limiter=reconcile_limiter,
                    autoscale_limiter=autoscale_limiter,
                )
            except Exception:
                logger.exception("Control tick failed")

    def _control_tick(
        self,
        *,
        woken: bool,
        schedule_limiter: RateLimiter,
        reconcile_limiter: RateLimiter,
        autoscale_limiter: RateLimiter,
    ) -> None:
        """Run one control tick: one read snapshot, due phases, one write txn.

        Phase order is schedule -> reconcile -> autoscale; the schedule decision is
        pure, reconcile and autoscale do bounded I/O (no DB), and every DB write is
        deferred to a single end-of-tick transaction (the worker-daemon path; a
        placement-owning backend also drains dispatch in its own pre-reconcile
        write). A wake runs a schedule-only mini-tick so submit->assign latency is
        the schedule time. Autoscale always pairs with a fresh schedule so it
        provisions against this tick's residual demand — no cross-tick handoff.
        Health-driven teardown of workers that crossed the failure threshold rides
        the post-commit fold (its own writes), exactly as on the legacy path.
        """
        now = Timestamp.now()

        # Dry-run: the schedule phase computes and logs intended assignments but
        # writes nothing; reconcile and autoscale are suppressed entirely.
        if self._config.dry_run:
            self._run_scheduling()
            return

        run_autoscale = autoscale_limiter.should_run()
        run_schedule = woken or run_autoscale or schedule_limiter.should_run()
        run_reconcile = self._force_reconcile or reconcile_limiter.should_run()
        self._force_reconcile = False
        scan_timeouts = run_reconcile and self._timeout_rate_limiter.should_run()

        inputs = self._build_tick_inputs(
            run_schedule=run_schedule,
            run_reconcile=run_reconcile,
            run_autoscale=run_autoscale,
            scan_timeouts=scan_timeouts,
        )

        sched_result = self._schedule_phase(inputs) if run_schedule else None

        recon_result: ReconcileResult | None = None
        timeout_decisions: list[TerminalDecision] = []
        if run_reconcile:
            timeout_decisions = self._timeout_decisions(inputs.control.timeout_rows, now.epoch_ms())
            # A cluster-view backend always reconciles (cluster-wide GC). A
            # worker-daemon backend skips a tick with no workers/tasks/timeouts.
            has_work = bool(
                inputs.control.worker_addresses
                or inputs.control.tasks_to_run
                or inputs.control.running_tasks
                or timeout_decisions
            )
            if self._backend_drains_dispatch or has_work:
                recon_result = self._task_backend.reconcile(inputs.control)

        auto_result: AutoscaleResult | None = None
        if run_autoscale:
            residual_demand = sched_result.residual_demand if sched_result is not None else []
            autoscale_snap = reads.ControlSnapshot(
                worker_addresses={},
                reconcile_rows=[],
                timeout_rows=[],
                worker_status_map=inputs.worker_status_map,
            )
            auto_result = self._task_backend.autoscale(autoscale_snap, residual_demand, dead_workers=[])

        reconcile_effects = self._commit_tick(
            inputs=inputs,
            sched_result=sched_result,
            recon_result=recon_result,
            timeout_decisions=timeout_decisions,
            auto_result=auto_result,
            now=now,
        )

        # Post-commit, in-memory: cache scheduling diagnostics, request a prompt
        # dispatch follow-up for fresh assignments, fold health.
        if sched_result is not None:
            self._scheduling_diagnostics = sched_result.diagnostics
            self._last_scheduling_context = sched_result.post_taint_context
            if sched_result.assignments:
                self._force_reconcile = True
                self._tick_wake.set()

        if recon_result is not None:
            self._fold_health(recon_result.health_events, reconcile_effects, inputs.control, now)

    def _build_tick_inputs(
        self,
        *,
        run_schedule: bool,
        run_reconcile: bool,
        run_autoscale: bool,
        scan_timeouts: bool,
    ) -> _TickInputs:
        """Assemble the due phases' inputs from a single read transaction.

        A placement-owning (``CLUSTER_VIEW``) backend's reconcile snapshot comes
        from the dispatch drain (a write), so it is built first, outside the read
        txn; the worker-daemon reconcile snapshot, the scheduling context +
        reservation claims, and the autoscale worker-status map all share the one
        read snapshot.
        """
        drained_control: reads.ControlSnapshot | None = None
        if run_reconcile and self._backend_drains_dispatch:
            drained_control = self._drain_dispatch_snapshot()

        inputs = _TickInputs()
        # Dedicated control pool: the tick's snapshot must not queue behind a slow
        # dashboard read for a connection.
        with self._db.control_read_snapshot() as snap:
            if run_schedule:
                claims, changed = refresh_reservation_claims_in_tx(snap, self._health, self._worker_attrs)
                inputs.claims = claims
                inputs.claims_changed = changed
                inputs.ctx = build_scheduling_context(
                    snap,
                    self._health,
                    self._worker_attrs,
                    self._config.user_budget_defaults,
                    claims,
                )
            if run_reconcile and not self._backend_drains_dispatch:
                control = reads.load_control_snapshot(snap, self._health, scan_timeouts=scan_timeouts)
                job_specs = self._build_run_templates(snap, control.reconcile_rows)
                inputs.control = dataclasses.replace(control, job_specs=job_specs)
            if run_autoscale:
                inputs.worker_status_map = self._worker_status_map_from_tx(snap)
        if drained_control is not None:
            inputs.control = drained_control
        return inputs

    def _schedule_phase(self, inputs: _TickInputs) -> ScheduleResult | None:
        """Run the pure scheduling decision over this tick's snapshot.

        The decision does no DB writes — the returned ``ScheduleResult`` is
        applied in the end-of-tick commit. Returns None when the schedule phase
        did not run (no context built this tick).
        """
        ctx = inputs.ctx
        if ctx is None:
            return None
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0
        if not ctx.pending_task_rows:
            # No pending work: empty decision, but keep the context as the
            # dashboard diagnostics snapshot for this tick.
            return ScheduleResult(post_taint_context=ctx)
        return self._task_backend.schedule(
            ScheduleInput(
                context=ctx,
                claims=inputs.claims,
                max_tasks_per_job_per_cycle=self._config.max_tasks_per_job_per_cycle,
                trace=trace,
            )
        )

    def _commit_tick(
        self,
        *,
        inputs: _TickInputs,
        sched_result: ScheduleResult | None,
        recon_result: ReconcileResult | None,
        timeout_decisions: list[TerminalDecision],
        auto_result: AutoscaleResult | None,
        now: Timestamp,
    ) -> ControllerEffects | None:
        """Apply this tick's decisions and observations in one write transaction.

        Order within the txn: reservation claims, schedule decisions, reconcile
        observations, execution-timeout finalizations, autoscaler state. Returns
        the reconcile kernel effects (consumed by the health fold) or None when the
        backend reported no worker results. A no-op tick opens no transaction.
        """
        has_claims = inputs.claims_changed
        has_sched = sched_result is not None and bool(
            sched_result.unschedulable or sched_result.assignments or sched_result.preemptions
        )
        has_recon = recon_result is not None and bool(recon_result.worker_results or recon_result.updates)
        has_state = auto_result is not None and auto_result.autoscaler_state is not None
        if not (has_claims or has_sched or has_recon or timeout_decisions or has_state):
            return None

        reconcile_effects: ControllerEffects | None = None
        with self._db.transaction() as cur:
            if has_claims:
                writes.replace_reservation_claims(cur, inputs.claims)
            if sched_result is not None:
                self._commit_schedule_decisions(cur, sched_result, now)
            if recon_result is not None:
                if recon_result.worker_results:
                    reconcile_effects = apply_reconcile(
                        cur, recon_result.worker_results, endpoints=self._endpoints, now=now
                    )
                if recon_result.updates:
                    apply_dispatch_updates(cur, recon_result.updates, endpoints=self._endpoints, now=now)
            if timeout_decisions:
                finalize(cur, timeout_decisions, endpoints=self._endpoints, now=now)
            if has_state:
                assert auto_result is not None and auto_result.autoscaler_state is not None
                persist_autoscaler_state(cur, auto_result.autoscaler_state)
        return reconcile_effects

    def _commit_schedule_decisions(self, cur: Tx, result: ScheduleResult, now: Timestamp) -> None:
        """Persist a ``ScheduleResult`` within the caller's write transaction.

        Expired/deadline tasks finalize UNSCHEDULABLE; assignments stamp ASSIGNED
        (carrying the backend-computed priority band); preemption victims finalize
        PREEMPT.
        """
        if result.unschedulable:
            finalize(cur, self._unschedulable_decisions(result.unschedulable), endpoints=self._endpoints, now=now)
        if result.assignments:
            ops.task.assign(cur, result.assignments, health=self._health)
        if result.preemptions:
            finalize(cur, result.preemptions, endpoints=self._endpoints, now=now)
            logger.info("Preemption pass: %d tasks preempted", len(result.preemptions))

    def _run_scheduling(self) -> SchedulingOutcome:
        """Run one self-contained scheduling cycle (its own snapshot + commits).

        This is the dry-run scheduling path; the live control tick computes its
        schedule via ``_schedule_phase`` and commits it in the shared end-of-tick
        transaction instead.

        The controller owns only the I/O: it refreshes reservation claims and
        builds the scheduling context in a single DB snapshot (which folds in the
        running-task band/value the preemption pass may evict), hands the
        resulting DB-less snapshot to ``backend.schedule`` for the pure placement
        decision, then commits the returned assignments, preemptions, and
        unschedulable marks. A worker-daemon backend runs the full
        gates → order → taints → preference → find_assignments → preemption
        pipeline; a cluster backend returns an empty result (Kueue schedules).

        No lock is needed since the control driver is single-threaded. Every DB
        access is serialized by ControllerDB._lock with multi-statement
        mutations wrapped in BEGIN IMMEDIATE transactions.
        """
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0

        claims = self._refresh_reservation_claims()
        with self._db.control_read_snapshot() as snap:
            ctx = build_scheduling_context(
                snap,
                self._health,
                self._worker_attrs,
                self._config.user_budget_defaults,
                claims,
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

        result = self._task_backend.schedule(
            ScheduleInput(
                context=ctx,
                claims=claims,
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

        The next control tick's reconcile phase reads the ASSIGNED rows and fans
        out the Reconcile RPCs.
        """
        if self._config.dry_run:
            for assignment in assignments:
                logger.info("[DRY-RUN] Would assign task %s to worker %s", assignment.task_id, assignment.worker_id)
            return
        with self._db.transaction() as cur:
            ops.task.assign(cur, assignments, health=self._health)

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
                endpoints=self._endpoints,
                now=Timestamp.now(),
            )
        logger.info("Preemption pass: %d tasks preempted", len(preemptions))

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None:
        """Return cached scheduling diagnostic for a job, or None if unavailable."""
        return self._scheduling_diagnostics.get(job_wire_id)

    def _timeout_decisions(self, timeout_rows: Sequence[Row], now_ms: int) -> list[TerminalDecision]:
        """Turn execution-timeout rows from the snapshot into TIMEOUT decisions.

        A row becomes a decision only once its attempt's
        ``started_at_ms + timeout_ms`` is already in the past.
        """
        decisions: list[TerminalDecision] = []
        for row in timeout_rows:
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

    def _mark_tasks_unschedulable(self, tasks: list[PendingTask]) -> None:
        """Mark a batch of tasks as unschedulable due to scheduling timeout.

        Each entry must be a row from ``reads.pending_tasks_with_jobs``; it carries
        ``scheduling_timeout_ms`` so no secondary DB fetch is needed.
        """
        if not tasks:
            return
        if self._config.dry_run:
            for task in tasks:
                logger.info("[DRY-RUN] Would mark task %s as unschedulable", task.task_id)
            return
        with self._db.transaction() as cur:
            finalize(
                cur,
                self._unschedulable_decisions(tasks),
                endpoints=self._endpoints,
                now=Timestamp.now(),
            )

    def _unschedulable_decisions(self, tasks: list[PendingTask]) -> list[TerminalDecision]:
        """Build UNSCHEDULABLE terminal decisions for scheduling-timeout tasks.

        Each entry is a row from ``reads.pending_tasks_with_jobs`` carrying
        ``scheduling_timeout_ms``. Logs one warning per task.
        """
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
        return decisions

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
    # Worker reconcile pass (snapshot → backend.reconcile → apply + health)
    # =========================================================================

    def _build_run_templates(
        self, snap: Tx, reconcile_rows: list[ReconcileRow]
    ) -> dict[JobName, job_pb2.RunTaskRequest]:
        """Build per-job ``RunTaskRequest`` templates for the ASSIGNED reconcile rows.

        ``run_request_template`` can return ``None`` for jobs the scheduler hasn't
        cached yet (e.g. reservation holders); those are dropped from the map.
        """
        templates_by_job: dict[JobName, job_pb2.RunTaskRequest | None] = {}
        for row in reconcile_rows:
            if row.task_state != job_pb2.TASK_STATE_ASSIGNED:
                continue
            if row.job_id not in templates_by_job:
                templates_by_job[row.job_id] = dispatch.run_request_template(self._run_template_cache, snap, row.job_id)
        return {jid: spec for jid, spec in templates_by_job.items() if spec is not None}

    def _drain_dispatch_snapshot(self) -> reads.ControlSnapshot:
        """Promote PENDING->ASSIGNED for a placement-owning backend and ride the drain.

        The dispatch drain is the single DB write a ``CLUSTER_VIEW`` backend needs
        before reconcile (the controller owns the write; the backend places tasks
        itself). It runs in its own write transaction, so a cluster backend's tick
        commits twice (drain + end-of-tick) rather than once.
        """
        max_promotions = self._promotion_bucket.available
        with self._db.transaction() as cur:
            batch = dispatch.drain_for_dispatch(cur, cache=self._run_template_cache, max_promotions=max_promotions)
        if batch.tasks_to_run:
            self._promotion_bucket.try_acquire(len(batch.tasks_to_run))
        return reads.ControlSnapshot(
            worker_addresses={},
            reconcile_rows=[],
            timeout_rows=[],
            tasks_to_run=batch.tasks_to_run,
            running_tasks=batch.running_tasks,
        )

    def _fold_health(
        self,
        observed: list[WorkerHealthEvent],
        reconcile_effects: ControllerEffects | None,
        snapshot: reads.ControlSnapshot,
        now: Timestamp,
    ) -> None:
        """Fold backend-observed + kernel-derived health through the one apply site.

        The backend reports the per-worker liveness it observed
        (REACHED/UNREACHABLE); the reconcile kernel derives build failures
        (BUILDING/ASSIGNED→FAILED on the worker path). Both feed the single
        ``WorkerHealthTracker.apply``, which returns the workers over a
        termination threshold for ``_fail_and_teardown``.
        """
        events = list(observed)
        if reconcile_effects is not None:
            events.extend(
                WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED)
                for wid in reconcile_effects.health.build_failed
            )
        if not events:
            return
        dead_workers = self._health.apply(events, now_ms=now.epoch_ms())
        if dead_workers:
            self._fail_and_teardown(dead_workers, snapshot)

    def _fail_and_teardown(self, dead_workers: list[WorkerId], snapshot: reads.ControlSnapshot) -> None:
        """Serialize worker failure, tear down slices + siblings, forget the lot.

        Fail the dead workers (``ops.worker.fail``), hand them to
        ``backend.autoscale`` which terminates their slices AND healthy siblings
        and returns the full ``removed_workers`` set, fail those siblings, persist
        the autoscaler state, and forget every removed worker from the tracker.
        The only health-driven write is removal.
        """
        reason = "worker reconcile failure threshold exceeded"
        sibling_reason = "unhealthy worker failed, slice terminated"
        for wid in dead_workers:
            log_event("worker_failing", str(wid), trigger=reason)
        failure_result = ops.worker.fail(
            self._db,
            worker_ids=[str(wid) for wid in dead_workers],
            reason=reason,
            health=self._health,
            endpoints=self._endpoints,
            worker_attrs=self._worker_attrs,
        )
        removed_ids = [wid for wid, _ in failure_result.removed_workers]
        if not removed_ids:
            # A concurrent reaper already failed every candidate (or they had no
            # address). Nothing was removed, so skip backend.autoscale entirely:
            # calling it here with no dead workers would run a full provisioning
            # cycle on the polling thread (probe_health + update_slice_activity)
            # racing the autoscaler thread.
            return
        auto = self._task_backend.autoscale(snapshot, [], dead_workers=removed_ids)
        if auto.autoscaler_state is not None:
            with self._db.transaction() as cur:
                persist_autoscaler_state(cur, auto.autoscaler_state)

        siblings = [wid for wid in auto.removed_workers if wid not in set(removed_ids)]
        if siblings:
            for wid in siblings:
                log_event("worker_failing", str(wid), trigger=sibling_reason)
            ops.worker.fail(
                self._db,
                worker_ids=[str(wid) for wid in siblings],
                reason=sibling_reason,
                health=self._health,
                endpoints=self._endpoints,
                worker_attrs=self._worker_attrs,
            )
        self._health.forget_many(set(removed_ids) | set(auto.removed_workers))

    def _worker_status_map_from_tx(self, tx: Tx) -> WorkerStatusMap:
        """Per-worker idle/running status for the autoscale phase, read from ``tx``.

        Shares the control tick's snapshot so the autoscale phase adds no extra
        read transaction.
        """
        result: WorkerStatusMap = {}
        liveness = self._health.all()
        usability = {wid: l.usability for wid, l in liveness.items()}
        worker_ids = {wid for wid, u in usability.items() if u is not WorkerUsability.DEAD}
        running_by_worker = reads.running_tasks_by_worker(tx, worker_ids)
        for wid in worker_ids:
            result[wid] = WorkerStatus(
                worker_id=wid,
                running_task_ids=frozenset(tid.to_wire() for tid in running_by_worker.get(wid, set())),
                usability=usability[wid],
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
    def capabilities(self) -> frozenset[BackendCapability]:
        return self._task_backend.capabilities

    @property
    def run_template_cache(self) -> RunTemplateCache:
        """Per-job RunTaskRequest template cache, shared with the dispatch path."""
        return self._run_template_cache

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
        pending hints). Capacity is driven through ``backend.autoscale``, not
        this handle.
        """
        return self._task_backend.autoscaler
