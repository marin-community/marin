# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import asyncio
import atexit
import enum
import json
import logging
import secrets
import socket
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import uvicorn
from finelog.client import RemoteLogHandler
from rigging import telemetry
from rigging.filesystem.storage_path import prefix_join
from rigging.server_auth import IAP_ISSUER, IAP_PUBLIC_KEYS_URL, TokenVerifier
from rigging.timing import Duration, ExponentialBackoff, RateLimiter, Timestamp, TokenBucket
from sqlalchemy import Row

from iris.cluster.bundle import BundleStore
from iris.cluster.config import PeerConfig
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.auth import (
    CONTROL_PLANE_AUDIENCE,
    DEFAULT_USER_ROLE,
    ENDPOINT_TOKEN_SCOPE,
    FEDERATION_AUDIENCE,
    NATIVE_PROXY_JWT_CACHE_CAPACITY,
    NATIVE_PROXY_JWT_CACHE_TTL_SECONDS,
    NATIVE_PROXY_JWT_LEEWAY_SECONDS,
    PROXY_PLANE_AUDIENCE,
    ControllerAuth,
    FederationTokenProvider,
    NativeProxyAuthConfig,
    NativeProxyAuthMode,
    native_proxy_auth_policy,
    request_auth_policy,
)
from iris.cluster.controller.autoscaler.persistence import persist_autoscaler_state
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendKind,
    BackendRuntime,
    ReconcileRequest,
    ScheduleRequest,
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
from iris.cluster.controller.endpoint_service import EndpointServiceImpl, ProxyMappingDelta, ProxyRegistryReset
from iris.cluster.controller.federation_proxy import FederatedEndpointHandoff
from iris.cluster.controller.federation_store import ControllerFederationStore, build_queued_candidates
from iris.cluster.controller.log_stack import LogStack
from iris.cluster.controller.native_proxy import NativeProxy, NativeProxyStats
from iris.cluster.controller.native_proxy_metrics import (
    NativeProxyTelemetry,
    install_native_proxy_metrics,
    uninstall_native_proxy_metrics,
)
from iris.cluster.controller.ops.reconcile import apply_observation
from iris.cluster.controller.ops.task import Assignment, finalize
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.pruner import prune_old_data
from iris.cluster.controller.reconcile import ControllerEffects, dispatch
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.dispatch import (
    DISPATCH_PROMOTION_RATE,
)
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.scheduling.policy import (
    build_scheduling_context,
)
from iris.cluster.controller.scheduling.scheduler import (
    SchedulingContext,
)
from iris.cluster.controller.service import CapabilityUrlConfig, ControllerServiceImpl, PendingKick
from iris.cluster.controller.task_state_stats import TaskStateCollector
from iris.cluster.controller.transition_reader import DbTransitionReader
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.endpoints import TELEMETRY_ENDPOINT_PATH
from iris.cluster.federation.availability import Promotion, QueuedCandidate
from iris.cluster.federation.manager import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MAX_HANDOFFS_PER_CYCLE,
    FederationManager,
)
from iris.cluster.federation.peer import FederationPeer, build_peers
from iris.cluster.log_keys import CONTROLLER_LOG_KEY
from iris.cluster.platforms.types import resolve_external_host
from iris.cluster.types import (
    JobName,
    PendingTask,
    UserBudgetDefaults,
    WorkerId,
)
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import SESSION_COOKIE

logger = logging.getLogger(__name__)

# Sync Connect RPC handlers are dispatched via ``asyncio.to_thread``, which
# uses the running loop's default executor. asyncio's default executor sizes
# at ``min(32, os.cpu_count() + 4)`` — only 8 threads on a 4-vCPU controller
# VM. A handful of slow handlers (e.g. ``launch_job`` blocking up to 120s in
# ``_wait_until_job_drained``) saturates that pool and head-of-line blocks
# every other RPC, including the worker heartbeats that would unblock the
# drain. Install a wider, named pool so a burst of slow handlers cannot
# starve the rest.
_RPC_HANDLER_THREADS = 64
_CONTROLLER_KEEPALIVE = 120
_PRIVATE_CONTROLLER_HOST = "127.0.0.1"
_SYNCHRONOUS_PHASE_INTERVAL = 0.0
_WORKER_RECONCILE_TEARDOWN_REASON = "worker reconcile failure threshold exceeded"


def _install_rpc_executor(server: uvicorn.Server, *, max_workers: int) -> None:
    """Replace ``server.run`` with a variant that pins a sized default executor."""

    def run_with_executor(sockets: list[socket.socket] | None = None) -> None:
        # Preserve Uvicorn's configured loop factory. Constructing an asyncio
        # loop directly bypasses ``loop=auto`` and silently disables uvloop.
        with asyncio.Runner(loop_factory=server.config.get_loop_factory()) as runner:
            runner.get_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rpc-handler")
            )
            runner.run(server.serve(sockets=sockets))

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
    """Per-tick inputs the control driver assembles for the due phases.

    The controller reads its task and worker state once. ``scheduling_context``
    is the backend's complete scheduling workspace, ``reconcile_request`` carries
    the Kubernetes dispatch drain when applicable, and ``timeout_rows`` is the
    execution-timeout sweep.
    """

    scheduling_context: SchedulingContext | None = None
    reconcile_request: ReconcileRequest = field(default_factory=ReconcileRequest)
    timeout_rows: Sequence[Row] = ()
    # Federated jobs queued on this parent awaiting a peer with free capacity, in
    # priority-then-age order. The tick's federation pass assigns them to peers.
    queued_federation: list[QueuedCandidate] = field(default_factory=list)
    # Queued federated jobs whose scheduling deadline has elapsed while waiting for a
    # peer; the tick fails them UNSCHEDULABLE (they own no task rows, so the task-level
    # timeout scan never sees them).
    expired_queued_federation: list[JobName] = field(default_factory=list)


@dataclass(frozen=True)
class SchedulePhaseResult:
    """One schedule phase's outputs, before any DB write."""

    result: ScheduleResult
    pins: list[tuple[JobName, str]]


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
    unhealthy) before it is failed and torn down. Threaded into each worker-daemon
    backend at construction to size the ``WorkerHealthTracker`` it owns. Realized as
    wall-clock elapsed since the worker's last successful reconcile, so detection
    latency is ~grace regardless of the reconcile cadence or how long a failing pass
    takes. ~50s tolerates brief network blips without reaping a multi-VM slice;
    tests shorten it for fast deterministic teardown."""

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

    slice_retention: Duration = field(default_factory=lambda: Duration.from_seconds(3600))
    """Delete orphaned slices (no backing worker row) older than this (default: 1 hour).

    Must comfortably exceed worst-case slice boot + worker-registration lag, so a
    freshly-created slice whose VMs are still booting is never reaped before its
    workers register."""

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

    endpoints: dict[str, str] = field(default_factory=dict)
    """Resolved cluster endpoints: logical name -> concrete URL. Built from
    cluster_config.endpoints by the daemon entrypoint. Registered as system
    endpoints on the EndpointService during start()."""

    cluster_id: str = ""
    """This cluster's real federation identity (from the cluster config ``name``).

    Sent as the ``requester_id`` on each ``FederationSync``. Required once this cluster
    hands jobs off; unused otherwise. Also the tag a minted capability URL carries so a
    federation parent can relay it back here."""

    dashboard_url: str = ""
    """This cluster's public origin (cluster config ``dashboard_url``); the local origin
    a minted capability URL uses when no public parent is configured."""

    federation_public_parent: str = ""
    """Public origin of the federation parent that fronts this cluster (cluster config
    ``federation_public_parent``). Set on a child whose own origin is not world-visible:
    a minted capability URL is then tagged with ``cluster_id`` and points at the parent,
    which relays it back here."""

    peers: dict[str, PeerConfig] = field(default_factory=dict)
    """Federation peers (peer id -> declaration). Empty leaves federation inert:
    no peer connections, no heartbeat, an empty ListPeers view."""

    federation_heartbeat_interval: Duration = field(default_factory=lambda: DEFAULT_HEARTBEAT_INTERVAL)
    """How often the federation capability heartbeat probes each peer."""

    max_federation_handoffs_per_cycle: int = DEFAULT_MAX_HANDOFFS_PER_CYCLE
    """Cap on federation queue promotions to any one peer per control tick. Bounds a
    burst of over-assignment against a single (possibly stale) availability
    observation, on top of the reservation ledger."""


class Controller:
    """Unified controller managing all components and lifecycle.

    One driver thread runs the control tick — schedule -> reconcile -> autoscale
    as phases over a single read snapshot, committed through one end-of-tick write
    transaction — alongside the prune and checkpoint housekeeping threads.

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(config=config, log_stack=log_stack)
        controller.register_backend(RpcTaskBackend(descriptor=descriptor, stub_factory=stub_factory))
        controller.start()
        try:
            job_id = controller.launch_job(request)
            status = controller.get_job_status(job_id)
        finally:
            controller.stop()
        ```

    Args:
        config: Controller configuration
        federation_peers: Optional prebuilt peer connections for an embedding
            that owns transport composition. Production builds peers from config.
    """

    def __init__(
        self,
        config: ControllerConfig,
        log_stack: LogStack,
        threads: ThreadContainer | None = None,
        db: ControllerDB | None = None,
        federation_peers: Sequence[FederationPeer] | None = None,
    ):
        if not config.remote_state_dir:
            raise ValueError(
                "remote_state_dir is required. Set via ControllerConfig.remote_state_dir. "
                "Example: remote_state_dir='gs://my-bucket/iris/state'"
            )
        self._config = config
        self._stopped = False
        self._started = False
        self._backend: TaskBackend | None = None

        self._promotion_bucket = TokenBucket(
            capacity=DISPATCH_PROMOTION_RATE,
            refill_period=Duration.from_minutes(1),
        )

        config.local_state_dir.mkdir(parents=True, exist_ok=True)
        if db is not None:
            self._db = db
        else:
            self._db = ControllerDB(db_dir=config.local_state_dir / "db")
        # Projections self-register into ``self._db.caches`` on construction; every
        # cursor the DB mints reaches them as ``tx.caches[Projection]`` without any
        # threaded references.
        EndpointsProjection(self._db)
        AttemptCountsProjection(self._db)
        WorkerAttrsProjection(self._db)
        RunTemplatesProjection(self._db)

        writes.validate(self._db.caches)

        self._threads = threads if threads is not None else get_thread_container()

        # Federation: remote clusters this controller may delegate whole jobs to.
        # Inert with no peers configured (build_peers returns nothing, the loops
        # never start), so a single-cluster deployment is unchanged. The store
        # gives the manager durable access to this controller's tables. Each peer
        # connection presents this cluster's federation token (minted from the auth
        # signing key) so an enforcing peer admits the handoff as a trusted requester.
        federation_token_provider = (
            FederationTokenProvider(config.cluster_id, config.auth.jwt_manager)
            if config.peers and config.auth and config.auth.jwt_manager
            else None
        )
        self._bundle_store = BundleStore(storage_dir=prefix_join(config.remote_state_dir, "bundles"))
        peers = (
            list(federation_peers)
            if federation_peers is not None
            else build_peers(config.peers, federation_token_provider=federation_token_provider)
        )
        self._federation = FederationManager(
            peers,
            threads=self._threads,
            store=ControllerFederationStore(
                self._db,
            ),
            bundles=self._bundle_store,
            cluster_id=config.cluster_id,
            heartbeat_interval=config.federation_heartbeat_interval,
            max_handoffs_per_cycle=config.max_federation_handoffs_per_cycle,
        )

        # The log client and its tables are built before the backend and autoscaler
        # (their finelog handles are constructor args), so the controller only holds
        # the stack for its own logging and shuts it down at stop().
        self._log_stack = log_stack
        self._log_client = log_stack.client
        self._log_service_address = log_stack.address
        self._log_handler = RemoteLogHandler(self._log_client, key=CONTROLLER_LOG_KEY)

        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger("iris").addHandler(self._log_handler)

        # Periodic iris.task_state emitter: per-root-job task counts + wait ages
        # aggregated from the controller DB. Only cluster-view (k8s) controllers
        # emit it — their rows must ride finelog federation, while a GCP
        # controller's DB is directly queryable via ExecuteRawQuery. Construction
        # starts the emitter thread, so it is built in start(), closed in stop().
        self._task_state_collector: TaskStateCollector | None = None

        self._db.register_reopen_hook(self._seed_backend_liveness)

        self._endpoint_service = EndpointServiceImpl(
            db=self._db,
            system_endpoints={},
        )
        self._service = ControllerServiceImpl(
            controller=self,
            bundle_store=self._bundle_store,
            log_client=self._log_client,
            db=self._db,
            endpoint_service=self._endpoint_service,
            auth=config.auth,
            user_budget_defaults=config.user_budget_defaults,
            capability_url_config=CapabilityUrlConfig(
                cluster_name=config.cluster_id,
                local_origin=config.dashboard_url,
                parent_origin=config.federation_public_parent,
            ),
        )
        # Forwards a /proxy request for an endpoint that lives on a federated child
        # to that peer's controller, presenting this cluster's federation bearer.
        # Present only when this controller has peers and a signing key to mint with.
        federated_handoff = (
            FederatedEndpointHandoff(self._federation.peer_controller_address, federation_token_provider.get_token)
            if federation_token_provider is not None
            else None
        )

        def _federation_owner_check(root_job: JobName, peer_id: str) -> bool:
            with self._db.read_snapshot() as q:
                return reads.has_received_job_from_peer(q, peer_id, root_job)

        external_auth_policy = request_auth_policy(config.auth)
        proxy_decision_secret = secrets.token_urlsafe(32)
        self._auth_policy = native_proxy_auth_policy(external_auth_policy)
        self._external_auth_allows_anonymous = external_auth_policy.allows_anonymous
        self._dashboard = ControllerDashboard(
            self._service,
            endpoint_service=self._endpoint_service,
            auth_provider=config.auth_provider,
            auth_policy=self._auth_policy,
            reported_auth_policy=external_auth_policy,
            jwt_manager=config.auth.jwt_manager if config.auth else None,
            federated_handoff=federated_handoff,
            federation_owner_check=_federation_owner_check,
            proxy_decision_secret=proxy_decision_secret,
        )

        # Wakes the control-tick driver. A submit triggers a schedule-only
        # mini-tick so submit->assign latency is the schedule time, not gated on
        # the next reconcile cadence.
        self._tick_wake = threading.Event()
        # Set after a tick commits new ASSIGNED rows so the next tick reconciles
        # immediately (dispatching them) instead of waiting a full poll interval.
        self._force_reconcile = False
        # Workers queued off the control loop for teardown on the next tick; see
        # request_worker_eviction / _drain_pending_evictions.
        self._pending_evictions: set[WorkerId] = set()
        self._pending_evictions_lock = threading.Lock()
        # Task terminal-state overrides queued off the control loop for the next
        # tick; see request_task_kicks / _drain_pending_kicks.
        self._pending_kicks: list[PendingKick] = []
        self._pending_kicks_lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._native_proxy = None
        self._native_proxy_metrics: NativeProxyTelemetry | None = None
        self._endpoint_service.subscribe_proxy_updates(self._publish_native_proxy_update)
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

    def register_backend(self, backend: TaskBackend) -> None:
        """Bind this controller's sole execution backend before startup."""
        if self._started:
            raise RuntimeError("the backend must be registered before Controller.start()")
        if self._backend is not None:
            raise ValueError(
                f"controller already has backend {self._backend.descriptor.backend_id!r}; "
                "use federation to compose multiple clusters"
            )

        descriptor = backend.descriptor
        backend_id = descriptor.backend_id
        if not backend_id or backend_id != backend_id.strip():
            raise ValueError("backend_id must be a non-empty canonical string")
        if descriptor.kind is BackendKind.WORKER:
            if backend.health is None:
                raise ValueError(f"worker backend {backend_id!r} must provide worker liveness")
            backend.bind_runtime(BackendRuntime(db=self._db))
        elif backend.health is not None or backend.autoscaler is not None:
            raise ValueError(f"Kubernetes backend {backend_id!r} cannot expose worker liveness or an Iris autoscaler")

        self._backend = backend

    def wake(self) -> None:
        """Wake the control tick to run a schedule-only mini-tick immediately.

        Called on new job submission so the next tick picks up the new pending
        tasks (and a fresh assignment then forces the following reconcile) instead
        of waiting a full poll interval.
        """
        self._tick_wake.set()

    def request_worker_eviction(self, worker_ids: Sequence[WorkerId]) -> None:
        """Queue workers for fail-and-teardown on the next control tick.

        Called off the control-loop thread (the Register RPC, when a worker claims
        an address still held by a stale row — a recycled internal IP). The
        teardown reaps the worker's slice through the autoscaler, which is only
        safe on the control-loop thread, so the work is deferred to the tick drain.
        """
        if not worker_ids:
            return
        with self._pending_evictions_lock:
            self._pending_evictions.update(worker_ids)
        self.wake()

    def request_task_kicks(self, kicks: Sequence[PendingKick]) -> None:
        """Queue task terminal-state overrides to apply on the next control tick.

        Called off the control-loop thread by the KickTasks RPC. Queuing keeps the
        kicks inside the tick's single write transaction so they cannot race the
        scheduler's view of task state.
        """
        if not kicks:
            return
        with self._pending_kicks_lock:
            self._pending_kicks.extend(kicks)
        self.wake()

    def _seed_backend_liveness(self) -> None:
        """Seed persisted worker liveness after startup or checkpoint restore."""
        if self._backend is not None and self._backend.descriptor.kind is BackendKind.WORKER:
            self._backend.seed_liveness()

    def all_liveness(self) -> dict[WorkerId, WorkerLiveness]:
        """Return the worker backend's liveness map, or empty for Kubernetes."""
        health = self.backend.health
        return health.all() if health is not None else {}

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness:
        """Return one worker's liveness, or a default for an unknown worker."""
        return self.all_liveness().get(worker_id, WorkerLiveness())

    @property
    def started(self) -> bool:
        """Whether the controller loops have been started."""
        return self._started

    def begin_shutdown(self) -> None:
        """Reject new control-plane requests without stopping workers."""
        self._dashboard.begin_shutdown()

    def start(self) -> None:
        """Start the dashboard server and the control + housekeeping threads.

        The unified control tick drives schedule -> reconcile -> autoscale;
        the reconcile phase is the sole reconcile + liveness channel — it
        reconciles every active worker (worker-daemon backends) or drains + syncs
        pods (cluster backends), applies the backend's observed health events, and
        tears down workers that cross the failure threshold.
        """
        if self._backend is None:
            raise ValueError("Controller requires a registered backend")
        if self._started:
            raise RuntimeError("Controller has already started")
        self._seed_backend_liveness()
        self._started = True
        if self._config.dry_run:
            logger.info("[DRY-RUN] Controller started in dry-run mode — all side effects suppressed")

        if not self._config.dry_run:
            self._prune_thread = self._threads.spawn(self._run_prune_loop, name="prune-loop")
            if self.backend.descriptor.kind is BackendKind.KUBERNETES:
                self._task_state_collector = TaskStateCollector(self._db, self._log_stack.task_state_table)

        # Create and start uvicorn server via spawn_server, which bridges the
        # ManagedThread stop_event to server.should_exit automatically.
        # timeout_keep_alive: uvicorn defaults to 5s, which races with client polling
        # intervals of the same length, causing TCP resets on idle connections. Use 120s
        # to safely cover long polling gaps during job waits.
        # The native listener is Uvicorn's only ingress and preserves the load
        # balancer's forwarded headers. Trust its loopback connection so
        # Starlette builds externally reachable absolute URLs.
        server_config = uvicorn.Config(
            self._dashboard.app,
            host=_PRIVATE_CONTROLLER_HOST,
            port=0,
            log_level="warning",
            log_config=None,
            timeout_keep_alive=_CONTROLLER_KEEPALIVE,
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
            self._endpoint_service.register_system_endpoint(name, url)
            logger.info("Registered system endpoint %s -> %s", name, url)
        self._endpoint_service.register_system_endpoint("/system/log-server", self._log_service_address)

        # One driver runs schedule -> reconcile -> autoscale as phases of a single
        # tick (one read snapshot + one end-of-tick commit). Spawned after endpoint
        # registration because its first autoscale phase may provision buffer slices
        # whose workers query /system/log-server. In dry-run it runs the schedule
        # phase only.
        self._control_thread = self._threads.spawn(self._run_control_loop, name="control-loop")

        if self._periodic_checkpoint_limiter is not None and not self._config.dry_run:
            self._checkpoint_thread = self._threads.spawn(self._run_checkpoint_loop, name="checkpoint-loop")

        # Start the federation capability heartbeat (a no-op with no peers).
        self._federation.start()

        # Register atexit hook to capture final state for post-mortem analysis.
        # Unregistered in stop() so it doesn't fire against a closed DB.
        self._atexit_registered = True
        atexit.register(self._atexit_checkpoint)

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=Duration.from_seconds(5.0),
        )
        assert self._server is not None
        assert self._server.servers
        private_port = self._server.servers[0].sockets[0].getsockname()[1]
        self._native_proxy = NativeProxy(
            self._config.host,
            self._config.port,
            f"http://{_PRIVATE_CONTROLLER_HOST}:{private_port}",
            self._dashboard.proxy_decision_secret,
            json.dumps(asdict(self._native_proxy_auth_config())),
        )
        telemetry.configure(
            endpoint=self._log_service_address.rstrip("/") + TELEMETRY_ENDPOINT_PATH,
            service="iris-controller",
            attributes={"role": "controller"},
        )
        self._native_proxy_metrics = install_native_proxy_metrics(self._native_proxy)
        self._replace_native_proxy_registry()

    def _publish_native_proxy_update(self, update: ProxyMappingDelta | ProxyRegistryReset) -> None:
        if self._native_proxy is None:
            return
        if isinstance(update, ProxyRegistryReset):
            self._recover_native_proxy_registry()
            return
        payload = json.dumps(asdict(update))
        try:
            self._native_proxy.update_mappings(payload)
        except ValueError:
            logger.exception(
                "Native proxy rejected endpoint mapping generation %d -> %d; replacing registry",
                update.base_generation,
                update.next_generation,
            )
            self._recover_native_proxy_registry()

    def _recover_native_proxy_registry(self) -> None:
        assert self._native_proxy is not None
        try:
            self._replace_native_proxy_registry()
        except ValueError:
            logger.exception("Native proxy registry replacement failed; pausing native routing")
            self._native_proxy.pause_registry()

    def _replace_native_proxy_registry(self) -> None:
        if self._native_proxy is None:
            return
        self._native_proxy.pause_registry()
        snapshot = self._endpoint_service.proxy_registry_snapshot()
        self._native_proxy.replace_registry(json.dumps(asdict(snapshot)))

    def _native_proxy_auth_config(self) -> NativeProxyAuthConfig:
        auth = self._config.auth
        if auth is None or auth.provider is None:
            mode = NativeProxyAuthMode.PERMISSIVE
        elif self._external_auth_allows_anonymous:
            mode = NativeProxyAuthMode.OPTIONAL
        else:
            mode = NativeProxyAuthMode.ENFORCING
        if auth is not None and auth.jwt_manager is not None:
            issuers, jwks = auth.jwt_manager.native_proxy_verification_material()
        else:
            issuers, jwks = (), {"keys": []}
        return NativeProxyAuthConfig(
            mode=mode,
            issuers=issuers,
            jwks=jwks,
            leeway_seconds=NATIVE_PROXY_JWT_LEEWAY_SECONDS,
            cache_capacity=NATIVE_PROXY_JWT_CACHE_CAPACITY,
            cache_ttl_seconds=NATIVE_PROXY_JWT_CACHE_TTL_SECONDS,
            trusted_cidrs=auth.trusted_cidrs if auth is not None else (),
            control_audience=CONTROL_PLANE_AUDIENCE,
            proxy_audience=PROXY_PLANE_AUDIENCE,
            proxy_scope=ENDPOINT_TOKEN_SCOPE,
            federation_audience=FEDERATION_AUDIENCE,
            session_cookie=SESSION_COOKIE,
            iap_public_keys_url=IAP_PUBLIC_KEYS_URL,
            iap_issuer=IAP_ISSUER,
            iap_audience=auth.iap_audience if auth is not None else None,
            federation_keys=auth.federation_keys if auth is not None else {},
            admin_users=tuple(sorted(auth.role_policy.admins)) if auth is not None and auth.role_policy else (),
            default_user_role=(
                auth.role_policy.default_role if auth is not None and auth.role_policy else DEFAULT_USER_ROLE
            ),
        )

    def stop(self) -> None:
        """Stop all background components gracefully. Idempotent.

        Shutdown ordering:
        1. Reject new control-plane requests.
        2. Unregister atexit hook so it doesn't fire against a closed DB.
        3. Stop the control loop so no new work is triggered.
        4. Shut down the autoscaler (stops monitors, terminates VMs, stops platform).
        5. Stop remaining threads (server) and executors.
        """
        if self._stopped:
            return
        self.begin_shutdown()
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
        if self._task_state_collector is not None:
            self._task_state_collector.close()
        self._federation.stop()

        if self._native_proxy_metrics is not None and self._native_proxy is not None:
            uninstall_native_proxy_metrics(self._native_proxy)
            self._native_proxy_metrics = None
        if self._native_proxy is not None:
            self._native_proxy.stop()
        self._threads.stop()
        # The backend owns its autoscaler; close() shuts it down and releases its
        # provider resources.
        if self._backend is not None:
            self._backend.close()

        # Remove log handler before closing log resources to avoid errors
        # from late log records hitting a closed store or connection.
        logging.getLogger("iris").removeHandler(self._log_handler)
        self._log_handler.close()
        self._log_stack.close()
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
                        self.backend,
                        job_retention=self._config.job_retention,
                        worker_retention=self._config.worker_retention,
                        slice_retention=self._config.slice_retention,
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
        on a wake, a schedule-only mini-tick), applies backend-observed health, and
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
        force_timeout_scan: bool = False,
    ) -> None:
        """Run one control tick: one read snapshot and one write transaction.

        Phase order is schedule -> reconcile -> autoscale. A wake runs a
        schedule-only mini-tick; autoscale always pairs with a fresh schedule so
        it provisions against this tick's residual demand.
        """
        now = Timestamp.now()

        # Dry-run: the schedule phase computes and logs intended assignments but
        # writes nothing; reconcile and autoscale are suppressed entirely.
        if self._config.dry_run:
            self._run_scheduling()
            return

        self._drain_pending_evictions()
        pending_kicks = self._drain_pending_kicks()

        run_autoscale = autoscale_limiter.should_run()
        run_schedule = woken or run_autoscale or schedule_limiter.should_run()
        run_reconcile = self._force_reconcile or reconcile_limiter.should_run()
        self._force_reconcile = False
        scan_timeouts = run_reconcile and (force_timeout_scan or self._timeout_rate_limiter.should_run())

        inputs = self._build_tick_inputs(
            now=now,
            run_schedule=run_schedule,
            run_reconcile=run_reconcile,
            scan_timeouts=scan_timeouts,
        )

        sched_result: ScheduleResult | None = None
        backend_pins: list[tuple[JobName, str]] = []
        if run_schedule:
            sched = self._schedule_phase(inputs)
            sched_result, backend_pins = sched.result, sched.pins

        # Federation pass: assign queued federated jobs to peers that have room. A pure
        # decision over the tick's snapshot + the manager's reservation ledger; the
        # promotions commit (conditionally) in the same end-of-tick transaction. Runs in
        # the single scheduling thread right after local scheduling, so every scheduling
        # decision — local placement and peer selection — flows through one place.
        federation_promotions: list[Promotion] = []
        if run_schedule and inputs.queued_federation:
            federation_promotions = self._federation.plan_federation(inputs.queued_federation)

        recon_effects: ControllerEffects | None = None
        reaped_workers: list[WorkerId] = []
        timeout_decisions: list[TerminalDecision] = []
        if run_reconcile:
            timeout_decisions = self._timeout_decisions(inputs.timeout_rows, now.epoch_ms())
            observation = self.backend.reconcile(inputs.reconcile_request)
            application = apply_observation(
                DbTransitionReader(self._db),
                observation.task_updates,
                observation.worker_health_events,
                worker_health=self.backend.health,
                now=Timestamp.now(),
            )
            recon_effects = application.effects
            reaped_workers = application.reaped_workers

        auto_result: AutoscaleResult | None = None
        if run_autoscale:
            residual_demand = sched_result.residual_demand if sched_result is not None else []
            auto_result = self.backend.autoscale(AutoscaleRequest(residual_demand=residual_demand))

        confirmed_promotions = self._commit_tick(
            sched_result=sched_result,
            backend_pins=backend_pins,
            recon_effects=recon_effects,
            timeout_decisions=timeout_decisions,
            pending_kicks=pending_kicks,
            auto_result=auto_result,
            federation_promotions=federation_promotions,
            expired_queued_federation=inputs.expired_queued_federation,
            now=now,
        )

        # Charge the reservation ledger only for promotions whose CAS committed, so a
        # promotion raced by a cancel does not hold phantom peer capacity. The sync
        # loop delivers each newly-PENDING handle on its next pass.
        if confirmed_promotions:
            self._federation.confirm_promotions(confirmed_promotions)
            logger.info(
                "Federation: promoted %d queued job(s): %s",
                len(confirmed_promotions),
                ", ".join(f"{p.job_id.to_wire()}->{p.peer_id}" for p in confirmed_promotions),
            )

        # Force the next reconcile so workers are told to stop the kicked attempts
        # promptly instead of waiting a full reconcile interval.
        if pending_kicks:
            self._force_reconcile = True
            self._tick_wake.set()

        # Post-commit, in-memory: cache scheduling diagnostics, request a prompt
        # dispatch follow-up for fresh assignments.
        if sched_result is not None:
            self._scheduling_diagnostics = sched_result.diagnostics
            self._last_scheduling_context = sched_result.scheduling_context
            if sched_result.assignments:
                self._force_reconcile = True
                self._tick_wake.set()

        # Reaped workers are torn down only after their task transitions commit,
        # so teardown's fresh snapshot sees finalized attempts and skips them.
        if reaped_workers:
            self.backend.teardown(
                reaped_workers,
                reason=_WORKER_RECONCILE_TEARDOWN_REASON,
            )

    def _build_tick_inputs(
        self,
        *,
        now: Timestamp,
        run_schedule: bool,
        run_reconcile: bool,
        scan_timeouts: bool,
    ) -> _TickInputs:
        """Assemble the due phases' controller-owned inputs.

        A Kubernetes backend's reconcile request comes from the dispatch drain,
        built first. The controller then reads its state in one snapshot: a
        complete scheduling workspace and the execution-timeout rows.
        """
        inputs = _TickInputs()

        if run_reconcile and self.backend.descriptor.kind is BackendKind.KUBERNETES:
            drain = self._drain_dispatch_snapshot()
            inputs.reconcile_request = ReconcileRequest(
                tasks_to_run=drain.tasks_to_run,
                running_tasks=drain.running_tasks,
            )

        # Dedicated control pool: the tick's snapshot must not queue behind a slow
        # dashboard read for a connection.
        with self._db.control_read_snapshot() as snap:
            if run_schedule:
                health = self.backend.health if self.backend.descriptor.kind is BackendKind.WORKER else None
                context = build_scheduling_context(
                    snap,
                    health,
                    snap.caches[WorkerAttrsProjection],
                    self._config.user_budget_defaults,
                )
                if context.pending_task_rows:
                    inputs.scheduling_context = context
                if self._config.peers:
                    inputs.queued_federation = build_queued_candidates(snap)
                    inputs.expired_queued_federation = reads.expired_queued_handoffs(snap, now.epoch_ms())
            # Execution-timeout finalization is global across worker-daemon and
            # K8s backends. K8s gangs rely on it because they omit
            # activeDeadlineSeconds.
            if run_reconcile and scan_timeouts:
                inputs.timeout_rows = reads.scan_execution_timeout_rows(snap)
        return inputs

    def _schedule_phase(self, inputs: _TickInputs) -> SchedulePhaseResult:
        """Run the backend scheduler and identify newly local jobs to stamp."""
        context = inputs.scheduling_context
        if context is None:
            return SchedulePhaseResult(ScheduleResult(), [])
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0
        if trace:
            logger.info(
                "[TRACE round=%d] Phase 0: %d pending tasks",
                self._scheduling_round,
                len(context.pending_task_rows),
            )
        pins = {
            task.job_id: self.backend.descriptor.backend_id for task in context.pending_task_rows if not task.backend_id
        }
        result = self.backend.schedule(
            ScheduleRequest(
                context=context,
                max_tasks_per_job_per_cycle=self._config.max_tasks_per_job_per_cycle,
                trace=trace,
            )
        )
        return SchedulePhaseResult(result, list(pins.items()))

    def _commit_tick(
        self,
        *,
        sched_result: ScheduleResult | None,
        backend_pins: list[tuple[JobName, str]],
        recon_effects: ControllerEffects | None,
        timeout_decisions: list[TerminalDecision],
        pending_kicks: list[PendingKick],
        auto_result: AutoscaleResult | None,
        federation_promotions: list[Promotion],
        expired_queued_federation: list[JobName],
        now: Timestamp,
    ) -> list[Promotion]:
        """Apply this tick's decisions and authored effects in one write transaction.

        Order within the transaction: schedule decisions, queued-handoff timeout
        failures, federation promotions, reconcile effects, execution-timeout
        finalizations, administrative kicks, and autoscaler state.
        A no-op tick opens no transaction.

        Returns the federation promotions whose conditional CAS actually committed (a
        concurrent cancel/terminalize between the tick's read and this write drops the
        rest); the caller charges the reservation ledger for those.
        """
        autoscaler_state = auto_result.autoscaler_state if auto_result is not None else None

        has_sched = sched_result is not None and bool(
            sched_result.unschedulable or sched_result.assignments or sched_result.preemptions or backend_pins
        )
        has_recon = recon_effects is not None and not recon_effects.is_empty
        if not (
            has_sched
            or has_recon
            or timeout_decisions
            or pending_kicks
            or autoscaler_state is not None
            or federation_promotions
            or expired_queued_federation
        ):
            return []

        confirmed: list[Promotion] = []
        with self._db.transaction() as cur:
            if sched_result is not None:
                self._commit_schedule_decisions(cur, sched_result, now, backend_pins)
            # Fail queued handoffs past their scheduling deadline before promoting, so a
            # just-expired job's promotion CAS (guarded on job-nonterminal) rejects it.
            for job_id in expired_queued_federation:
                writes.mark_federated_job_unschedulable(
                    cur,
                    job_id,
                    now_ms=now.epoch_ms(),
                    error="Scheduling timeout exceeded while queued for a federation peer",
                )
            for promotion in federation_promotions:
                if writes.promote_queued_handoff(cur, promotion.job_id, promotion.peer_id):
                    confirmed.append(promotion)
            if has_recon and recon_effects is not None:
                commit_effects(cur, recon_effects)
            if timeout_decisions:
                finalize(cur, timeout_decisions, now=now)
            if pending_kicks:
                # Resolve after the schedule/reconcile writes so the attempt
                # re-check sees this tick's reassignments.
                kick_decisions = self._resolve_pending_kicks(cur, pending_kicks)
                if kick_decisions:
                    finalize(cur, kick_decisions, now=now)
                    logger.info("Admin kick: finalized %d task attempt(s)", len(kick_decisions))
            if autoscaler_state is not None:
                persist_autoscaler_state(cur, autoscaler_state)
        return confirmed

    def _commit_schedule_decisions(
        self,
        cur: Tx,
        result: ScheduleResult,
        now: Timestamp,
        backend_pins: list[tuple[JobName, str]],
    ) -> None:
        """Persist a ``ScheduleResult`` within the caller's write transaction.

        Backend pins stamp ``backend_id`` on newly local jobs and tasks;
        expired/deadline tasks finalize UNSCHEDULABLE; assignments stamp ASSIGNED;
        preemption victims finalize PREEMPT.
        """
        if backend_pins:
            writes.stamp_backend(cur, backend_pins)
        if result.unschedulable:
            finalize(cur, self._unschedulable_decisions(result.unschedulable), now=now)
        if result.assignments:
            health = self.backend.health
            assert health is not None, "a backend produced Iris assignments without worker liveness"
            ops.task.assign(cur, result.assignments, health=health)
        if result.preemptions:
            finalize(cur, result.preemptions, now=now)
            logger.info("Preemption pass: %d tasks preempted", len(result.preemptions))

    def _run_scheduling(self) -> SchedulingOutcome:
        """Run one self-contained scheduling cycle (its own snapshot + commits).

        This is the dry-run scheduling path; the live control tick computes its
        schedule via ``_schedule_phase`` and commits it in the shared end-of-tick
        transaction instead.

        The controller reads pending tasks, budgets, and worker state in one
        snapshot, asks the backend for a pure placement decision, then commits the
        assignments, preemptions, and unschedulable marks. A worker-daemon backend
        runs the full gates → order → find_assignments → preemption pipeline; a
        cluster backend returns an empty result (Kueue schedules).

        No lock is needed since the control driver is single-threaded. Every DB
        access is serialized by ControllerDB._lock with multi-statement
        mutations wrapped in BEGIN IMMEDIATE transactions.
        """
        inputs = self._build_tick_inputs(
            now=Timestamp.now(),
            run_schedule=True,
            run_reconcile=False,
            scan_timeouts=False,
        )
        context = inputs.scheduling_context
        if context is None:
            self._scheduling_diagnostics = {}
            self._last_scheduling_context = None
            return SchedulingOutcome.NO_PENDING_TASKS
        result = self._schedule_phase(inputs).result

        # Commit the decisions. Expired/deadline tasks are marked UNSCHEDULABLE;
        # assignments stamp ASSIGNED; preemption finalizes victims.
        if result.unschedulable:
            self._mark_tasks_unschedulable(result.unschedulable)
        if result.assignments:
            self._commit_assignments(result.assignments)
        self._apply_preemptions(result.preemptions)

        self._scheduling_diagnostics = result.diagnostics
        self._last_scheduling_context = result.scheduling_context

        if result.assignments or result.preemptions:
            log_event(
                "scheduling_pass_completed",
                "scheduler",
                assignments=len(result.assignments),
                preempted=len(result.preemptions),
                pending=len(context.pending_task_rows),
                workers=len(result.scheduling_context.workers) if result.scheduling_context else 0,
            )
            return SchedulingOutcome.ASSIGNMENTS_MADE
        return SchedulingOutcome.NO_ASSIGNMENTS

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
        health = self.backend.health
        assert health is not None, "scheduling assignments produced by a backend with no liveness tracker"
        with self._db.transaction() as cur:
            ops.task.assign(cur, assignments, health=health)

    def _apply_preemptions(self, preemptions: list[TerminalDecision]) -> None:
        """Finalize the backend's PREEMPT decisions.

        Slice evictions for a coscheduled preemptor's N siblings are
        all-or-nothing. Victims stop on the next reconcile tick: the planner
        drops them from the worker's desired set.
        """
        if not preemptions:
            return
        if self._config.dry_run:
            for decision in preemptions:
                logger.info("[DRY-RUN] Would preempt task %s", decision.task_id)
            return
        with self._db.transaction() as cur:
            finalize(
                cur,
                preemptions,
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

    def _drain_dispatch_snapshot(self) -> reads.ControlSnapshot:
        """Promote PENDING->ASSIGNED for Kubernetes and ride the drain.

        The dispatch drain is the single DB write a Kubernetes backend needs
        before reconcile (the controller owns the write; the backend places tasks
        itself). It runs in its own write transaction, so that tick commits twice
        (drain + end-of-tick).
        """
        max_promotions = self._promotion_bucket.available
        with self._db.transaction() as cur:
            batch = dispatch.drain_for_dispatch(
                cur,
                max_promotions=max_promotions,
                defaults=self._config.user_budget_defaults,
            )
        if batch.tasks_to_run:
            self._promotion_bucket.try_acquire(len(batch.tasks_to_run))
        return reads.ControlSnapshot(
            worker_addresses={},
            reconcile_rows=[],
            timeout_rows=[],
            tasks_to_run=batch.tasks_to_run,
            running_tasks=batch.running_tasks,
        )

    def _drain_pending_evictions(self) -> None:
        """Tear down workers queued by :meth:`request_worker_eviction`."""
        with self._pending_evictions_lock:
            if not self._pending_evictions:
                return
            drained = sorted(self._pending_evictions)
            self._pending_evictions.clear()
        reason = "address reused by newly-registered worker (recycled IP)"
        self.backend.teardown(drained, reason=reason)

    def _drain_pending_kicks(self) -> list[PendingKick]:
        """Take the queued administrative kicks for this tick's commit."""
        with self._pending_kicks_lock:
            if not self._pending_kicks:
                return []
            drained = self._pending_kicks
            self._pending_kicks = []
        return drained

    def _resolve_pending_kicks(self, cur: Tx, pending_kicks: list[PendingKick]) -> list[TerminalDecision]:
        """Turn queued kicks into terminal decisions, dropping superseded attempts.

        A kick targeting a specific attempt is dropped if that attempt is no longer
        current (the task retried in the meantime); a kick with no attempt id takes
        whatever attempt is current. Reads ``cur`` to see this tick's earlier writes.
        """
        decisions: list[TerminalDecision] = []
        for kick in pending_kicks:
            if kick.attempt_id is not None:
                detail = reads.get_task_detail(cur, kick.task_id)
                if detail is None or detail.current_attempt_id != kick.attempt_id:
                    logger.info(
                        "Dropping kick for %s: attempt %d is no longer current",
                        kick.task_id.to_wire(),
                        kick.attempt_id,
                    )
                    continue
            decisions.append(TerminalDecision(kind=kick.kind, task_id=kick.task_id, reason=kick.reason))
        return decisions

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

    def run_control_tick(self) -> None:
        """Run one complete control cycle synchronously before :meth:`start`.

        This is the deterministic embedding boundary for callers that own the
        controller lifecycle themselves. It drives scheduling, reconciliation,
        and autoscaling once without starting background threads. A running
        controller already owns its control loop, so mixing the two modes is an
        error.
        """
        if self.started:
            raise RuntimeError("run_control_tick cannot be used after Controller.start")
        self._control_tick(
            woken=True,
            schedule_limiter=RateLimiter(interval_seconds=_SYNCHRONOUS_PHASE_INTERVAL),
            reconcile_limiter=RateLimiter(interval_seconds=_SYNCHRONOUS_PHASE_INTERVAL),
            autoscale_limiter=RateLimiter(interval_seconds=_SYNCHRONOUS_PHASE_INTERVAL),
            force_timeout_scan=True,
        )

    def get_job_status(
        self,
        job_id: str,
    ) -> controller_pb2.Controller.GetJobStatusResponse:
        """Get the status of a job."""
        request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def list_jobs(
        self,
        request: controller_pb2.Controller.ListJobsRequest | None = None,
    ) -> controller_pb2.Controller.ListJobsResponse:
        """Return Jobs matching the request query."""
        return self._service.list_jobs(request or controller_pb2.Controller.ListJobsRequest(), None)

    def list_tasks(self, job_id: str) -> controller_pb2.Controller.ListTasksResponse:
        """Return current public Task rows for a Job."""
        request = controller_pb2.Controller.ListTasksRequest(job_id=job_id)
        return self._service.list_tasks(request, None)

    def get_task_status(self, task_id: str) -> controller_pb2.Controller.GetTaskStatusResponse:
        """Get one Task and its Attempt history."""
        request = controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id)
        return self._service.get_task_status(request, None)

    def terminate_job(
        self,
        job_id: str,
    ) -> job_pb2.Empty:
        """Terminate a running job."""
        request = controller_pb2.Controller.TerminateJobRequest(job_id=job_id)
        return self._service.terminate_job(request, None)

    def complete_job(
        self,
        job_id: str,
    ) -> job_pb2.Empty:
        """Complete a running job successfully."""
        request = controller_pb2.Controller.CompleteJobRequest(job_id=job_id)
        return self._service.complete_job(request, None)

    def kick_tasks(
        self,
        request: controller_pb2.Controller.KickTasksRequest,
    ) -> controller_pb2.Controller.KickTasksResponse:
        """Queue validated administrative state overrides for Tasks."""
        return self._service.kick_tasks(request, None)

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        """Register or renew a Task endpoint."""
        return self._service.endpoint_service.register_endpoint(request, None)

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest | None = None,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        """Return Task endpoints matching the optional query."""
        return self._service.endpoint_service.list_endpoints(
            request or controller_pb2.Controller.ListEndpointsRequest(),
            None,
        )

    def unregister_endpoint(self, endpoint_id: str) -> job_pb2.Empty:
        """Remove a Task endpoint by ID."""
        request = controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=endpoint_id)
        return self._service.endpoint_service.unregister_endpoint(request, None)

    def set_user_budget(
        self,
        request: controller_pb2.Controller.SetUserBudgetRequest,
    ) -> controller_pb2.Controller.SetUserBudgetResponse:
        """Set the budget limit and maximum priority band for one user."""
        return self._service.set_user_budget(request, None)

    def get_user_budget(self, user_id: str) -> controller_pb2.Controller.GetUserBudgetResponse:
        """Return one user's budget configuration and current spend."""
        request = controller_pb2.Controller.GetUserBudgetRequest(user_id=user_id)
        return self._service.get_user_budget(request, None)

    def register_worker(
        self,
        request: controller_pb2.Controller.RegisterRequest,
    ) -> controller_pb2.Controller.RegisterResponse:
        """Register or renew a worker identity and capacity."""
        return self._service.register(request, None)

    def list_workers(
        self,
        request: controller_pb2.Controller.ListWorkersRequest | None = None,
    ) -> controller_pb2.Controller.ListWorkersResponse:
        """Return workers matching the optional request filters."""
        return self._service.list_workers(request or controller_pb2.Controller.ListWorkersRequest(), None)

    def get_worker_status(self, worker_id: str) -> controller_pb2.Controller.GetWorkerStatusResponse:
        """Return current health and metadata for one worker."""
        request = controller_pb2.Controller.GetWorkerStatusRequest(id=worker_id)
        return self._service.get_worker_status(request, None)

    def federation_sync(
        self,
        request: controller_pb2.Controller.FederationSyncRequest,
    ) -> controller_pb2.Controller.FederationSyncResponse:
        """Apply an authenticated federation delta from a peer."""
        return self._service.federation_sync(request, None)

    def list_peers(self) -> controller_pb2.Controller.ListPeersResponse:
        """Return configured federation peers and their current status."""
        return self._service.list_peers(controller_pb2.Controller.ListPeersRequest(), None)

    # Properties

    @property
    def backend(self) -> TaskBackend:
        """The controller's registered execution backend."""
        if self._backend is None:
            raise RuntimeError("Controller backend has not been registered")
        return self._backend

    @property
    def federation(self) -> FederationManager:
        """The federation manager: peer registry, heartbeat, and submit-time router."""
        return self._federation

    @property
    def port(self) -> int:
        """Actual bound port (may differ from config if port=0 was specified)."""
        return self._native_proxy.port if self._native_proxy is not None else self._config.port

    @property
    def native_proxy_stats(self) -> NativeProxyStats | None:
        """Return native registry and JWT-cache counters, or ``None`` before startup."""
        if self._native_proxy is None:
            return None
        return NativeProxyStats.from_json(self._native_proxy.stats_json)

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
