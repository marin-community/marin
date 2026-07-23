# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import asyncio
import atexit
import enum
import logging
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from finelog.client import RemoteLogHandler
from rigging.filesystem import prefix_join
from rigging.server_auth import TokenVerifier
from rigging.timing import Duration, ExponentialBackoff, RateLimiter, Timestamp, TokenBucket
from sqlalchemy import Row

from iris.cluster.bundle import BundleStore
from iris.cluster.config import BackendConfig, PeerConfig
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.auth import ControllerAuth, FederationTokenProvider, request_auth_policy
from iris.cluster.controller.autoscaler.persistence import persist_autoscaler_state
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    ReconcileRequest,
    ReconcileResult,
    ScheduleRequest,
    ScheduleResult,
    TaskBackend,
)
from iris.cluster.controller.budget import resource_value
from iris.cluster.controller.checkpoint import (
    CheckpointResult,
    backup_databases,
    upload_checkpoint,
    write_checkpoint,
)
from iris.cluster.controller.codec import constraints_from_json, device_counts_from_json
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.endpoint_proxy import FederatedEndpointProxy
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_store import ControllerFederationStore, build_queued_candidates
from iris.cluster.controller.log_stack import LogStack
from iris.cluster.controller.ops.task import (
    Assignment,
    finalize,
)
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.pruner import prune_old_data
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.dispatch import (
    DISPATCH_PROMOTION_RATE,
)
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.scheduling.meta_scheduler import (
    BackendRouting,
    RoutableJob,
    build_backend_index,
    route_jobs_to_backends,
)
from iris.cluster.controller.scheduling.policy import (
    RoutingInputs,
    build_routing_inputs,
)
from iris.cluster.controller.scheduling.scheduler import (
    SchedulingContext,
)
from iris.cluster.controller.service import ControllerServiceImpl, PendingKick
from iris.cluster.controller.task_state_stats import TaskStateCollector
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.federation.availability import Promotion, QueuedCandidate
from iris.cluster.federation.manager import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MAX_HANDOFFS_PER_CYCLE,
    FederationManager,
)
from iris.cluster.federation.peer import build_peers
from iris.cluster.log_keys import CONTROLLER_LOG_KEY
from iris.cluster.platforms.types import resolve_external_host
from iris.cluster.types import (
    DEFAULT_BACKEND_ID,
    JobName,
    PendingTask,
    UserBudgetDefaults,
    WorkerId,
)
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import controller_pb2, job_pb2

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
    """Per-tick inputs the control driver assembles for the due phases.

    The controller reads only its own task-lifecycle state: ``routing`` carries
    the pending tasks + budgets the meta-scheduler and per-user budget thread off of;
    ``reconcile_requests`` carries each ``CLUSTER_VIEW`` backend's dispatch drain
    (worker-daemon backends source their own reconcile snapshot, so they have no
    entry); ``timeout_rows`` is the global execution-timeout sweep. Workers are
    read by each backend, never here.
    """

    routing: RoutingInputs | None = None
    reconcile_requests: dict[str, ReconcileRequest] = field(default_factory=dict)
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
    """One schedule phase's outputs, before any DB write.

    ``results`` is the per-backend placement decision; ``pins`` are the
    ``(job_id, backend_id)`` routings the meta-scheduler chose this tick; and
    ``unschedulable`` are the ``(task, reason)`` pairs no backend could take.
    """

    results: dict[str, ScheduleResult]
    pins: list[tuple[JobName, str]]
    unschedulable: list[tuple[PendingTask, str]]


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
    hands jobs off; unused otherwise."""

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
        controller = Controller(
            config=config,
            backends={DEFAULT_BACKEND_ID: RpcTaskBackend(stub_factory=RpcWorkerStubFactory())},
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
        backends: The ``{backend_id: TaskBackend}`` collection the controller
            drives. Each control-tick phase runs per backend over a snapshot
            filtered to that backend's workers/tasks; a single-backend mapping
            keyed :data:`DEFAULT_BACKEND_ID` reproduces the single-backend behavior
            exactly.
        backend_configs: Per-backend ``BackendConfig`` map used to route tasks to
            backends (the meta-scheduler index + allow policies) and to map each
            worker's scale group to its owning backend. Defaults to one implicit
            worker-daemon backend per entry in ``backends``.
    """

    def __init__(
        self,
        config: ControllerConfig,
        backends: dict[str, TaskBackend],
        log_stack: LogStack,
        threads: ThreadContainer | None = None,
        db: ControllerDB | None = None,
        backend_configs: dict[str, BackendConfig] | None = None,
    ):
        if not config.remote_state_dir:
            raise ValueError(
                "remote_state_dir is required. Set via ControllerConfig.remote_state_dir. "
                "Example: remote_state_dir='gs://my-bucket/iris/state'"
            )
        if not backends:
            raise ValueError("Controller requires at least one backend")

        self._config = config
        self._stopped = False
        self._backends: dict[str, TaskBackend] = dict(backends)
        # Stable processing order: the per-tick loop and the in-tick user-budget
        # thread walk backends in this order so the decision is deterministic.
        self._backend_ids: list[str] = sorted(self._backends)
        # A cluster backend that owns placement (no Iris scheduler) needs the
        # reconcile tick to drain pending dispatch (promote PENDING→ASSIGNED) and
        # ride it on the snapshot; a worker-daemon backend reconciles the
        # already-scheduled worker-bound rows. Resolved per backend from capability.
        self._dispatch_backends: set[str] = {
            bid for bid, backend in self._backends.items() if BackendCapability.CLUSTER_VIEW in backend.capabilities
        }

        # Routing/partition config. Absent (tests with a bare backend) synthesizes
        # one attribute-less worker-daemon backend per id: every job routes to it
        # and every scale group maps to it, so routing/partition is the identity.
        if backend_configs is None:
            backend_configs = {bid: BackendConfig(kind="worker_daemon") for bid in self._backends}
        # The meta-scheduler routes against what each backend advertises, not the
        # config. Attributes are immutable, so the routing index is built once.
        self._backend_routing = {
            bid: BackendRouting(advertised=backend.advertised_attributes()) for bid, backend in self._backends.items()
        }
        self._backend_index = build_backend_index(self._backend_routing)
        # Worker→backend ownership by scale group, used to wire each backend's
        # worker source and to route a failed worker's teardown to its owner.
        self._scale_group_to_backend: dict[str, str] = {
            scale_group: bid for bid, cfg in backend_configs.items() for scale_group in cfg.scale_groups
        }
        self._last_unroutable_jobs: dict[str, str] = {}

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
        self._federation = FederationManager(
            build_peers(config.peers, federation_token_provider=federation_token_provider),
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

        # Give each worker-daemon backend its own scale-group-scoped view of the DB
        # so it sources its own workers (the controller never partitions a worker
        # snapshot). Each such backend constructs and owns its liveness tracker, then
        # builds its worker source from the runtime it is bound here; the controller
        # reaches it through the backend, routed by scale group. A placement-owning
        # backend (k8s) has no workers and reads its dispatch effects through the
        # transition reader it received at construction.
        for backend_id, backend in self._backends.items():
            if BackendCapability.WORKER_DAEMON in backend.capabilities:
                backend.bind_runtime(self._build_runtime(backend_id))

        # Seed each backend's liveness from its persisted workers so the scheduler
        # sees them at startup, and reseed after a DB reopen (checkpoint restore).
        # ``find_prunable`` relies on this to keep every ``workers`` row tracked.
        self._seed_backend_liveness()
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
        )
        # Forwards a /proxy request for an endpoint that lives on a federated child
        # to that peer's controller, presenting this cluster's federation bearer.
        # Present only when this controller has peers and a signing key to mint with.
        federated_proxy = (
            FederatedEndpointProxy(self._federation.peer_controller_address, federation_token_provider.get_token)
            if federation_token_provider is not None
            else None
        )

        def _federation_owner_check(root_job: JobName, peer_id: str) -> bool:
            with self._db.read_snapshot() as q:
                return reads.has_received_job_from_peer(q, peer_id, root_job)

        self._dashboard = ControllerDashboard(
            self._service,
            endpoint_service=self._endpoint_service,
            host=config.host,
            port=config.port,
            auth_provider=config.auth_provider,
            auth_policy=request_auth_policy(config.auth),
            jwt_manager=config.auth.jwt_manager if config.auth else None,
            federated_proxy=federated_proxy,
            federation_owner_check=_federation_owner_check,
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
        """Seed each worker-daemon backend's tracker from its own persisted workers.

        Each backend reads its scale-group-scoped workers and heartbeats them into
        the tracker it owns, so every worker comes up ACTIVE at startup (and again
        after a DB reopen). Workers that then go unreachable accrue failures through
        the reconcile fold and are torn down once over threshold.
        """
        for backend in self._backends.values():
            if BackendCapability.WORKER_DAEMON in backend.capabilities:
                backend.seed_liveness()

    def all_liveness(self) -> dict[WorkerId, WorkerLiveness]:
        """Union of every worker-daemon backend's liveness (the trackers are disjoint).

        The controller's Fleet/exec/capacity readers reach worker liveness through
        this instead of a single shared tracker; a backend that tracks no Iris
        workers (k8s) contributes nothing.
        """
        merged: dict[WorkerId, WorkerLiveness] = {}
        for backend in self._backends.values():
            if backend.health is not None:
                merged.update(backend.health.all())
        return merged

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness:
        """One worker's liveness from its owning backend's tracker, or a default.

        The backends' trackers are disjoint by scale group, so the union resolves
        the worker to the single tracker that holds it; an untracked worker yields a
        default (not-healthy) snapshot, matching a direct tracker miss.
        """
        return self.all_liveness().get(worker_id, WorkerLiveness())

    @property
    def started(self) -> bool:
        """Whether the controller loops have been started."""
        return self._started

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
            if any(BackendCapability.CLUSTER_VIEW in b.capabilities for b in self._backends.values()):
                self._task_state_collector = TaskStateCollector(self._db, self._log_stack.task_state_table)

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
        if self._task_state_collector is not None:
            self._task_state_collector.close()
        self._federation.stop()

        self._threads.stop()
        # Each backend owns its autoscaler; close() shuts it down (terminates VMs,
        # stops the platform) and releases the backend's own resources.
        for backend in self._backends.values():
            backend.close()

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
                        self._backends.values(),
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
        """Run one control tick: one read snapshot, due phases per backend, one write txn.

        Phase order is schedule -> reconcile -> autoscale. The controller routes
        pending tasks to each backend (by ``backend_id``) and threads the per-user
        budget; each backend sources its own workers and runs its
        ``schedule``/``reconcile``/``autoscale`` over them, and the per-backend
        results merge into one end-of-tick write transaction. With a single backend
        every job routes to it, so behavior matches the single-backend path exactly.
        A wake runs a schedule-only mini-tick; autoscale always pairs with
        a fresh schedule so it provisions against this tick's residual demand.
        Execution-timeout finalization and health-driven teardown stay global.
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
        scan_timeouts = run_reconcile and self._timeout_rate_limiter.should_run()

        inputs = self._build_tick_inputs(
            now=now,
            run_schedule=run_schedule,
            run_reconcile=run_reconcile,
            run_autoscale=run_autoscale,
            scan_timeouts=scan_timeouts,
        )

        sched_results: dict[str, ScheduleResult] = {}
        backend_pins: list[tuple[JobName, str]] = []
        routing_unschedulable: list[tuple[PendingTask, str]] = []
        if run_schedule:
            sched = self._schedule_phase(inputs)
            sched_results, backend_pins, routing_unschedulable = sched.results, sched.pins, sched.unschedulable

        # Federation pass: assign queued federated jobs to peers that have room. A pure
        # decision over the tick's snapshot + the manager's reservation ledger; the
        # promotions commit (conditionally) in the same end-of-tick transaction. Runs in
        # the single scheduling thread right after local scheduling, so every scheduling
        # decision — local placement and peer selection — flows through one place.
        federation_promotions: list[Promotion] = []
        if run_schedule and inputs.queued_federation:
            federation_promotions = self._federation.plan_federation(inputs.queued_federation)

        recon_results: dict[str, ReconcileResult] = {}
        timeout_decisions: list[TerminalDecision] = []
        if run_reconcile:
            timeout_decisions = self._timeout_decisions(inputs.timeout_rows, now.epoch_ms())
            for backend_id in self._backend_ids:
                # Worker-daemon backends source their own placement (empty request);
                # a cluster-view backend gets its dispatch drain.
                request = inputs.reconcile_requests.get(backend_id, ReconcileRequest())
                recon_results[backend_id] = self._backends[backend_id].reconcile(request)

        auto_results: dict[str, AutoscaleResult] = {}
        if run_autoscale:
            for backend_id in self._backend_ids:
                residual_demand = sched_results[backend_id].residual_demand if backend_id in sched_results else []
                auto_results[backend_id] = self._backends[backend_id].autoscale(
                    AutoscaleRequest(residual_demand=residual_demand)
                )

        merged_sched = self._merge_schedule_results(sched_results) if run_schedule else None

        confirmed_promotions = self._commit_tick(
            sched_result=merged_sched,
            sched_results=sched_results,
            backend_pins=backend_pins,
            routing_unschedulable=routing_unschedulable,
            recon_results=recon_results,
            timeout_decisions=timeout_decisions,
            pending_kicks=pending_kicks,
            auto_results=auto_results,
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
        if merged_sched is not None:
            self._scheduling_diagnostics = merged_sched.diagnostics
            self._last_scheduling_context = merged_sched.scheduling_context
            if merged_sched.assignments:
                self._force_reconcile = True
                self._tick_wake.set()

        # Drain each backend's reaped-worker stash (folded during reconcile) AFTER
        # the reconcile effects are committed, so its teardown reads a fresh snapshot
        # where the just-finalized terminal attempts are already terminal and skipped.
        # No worker identity passes through the controller.
        if run_reconcile:
            for backend_id in self._backend_ids:
                self._backends[backend_id].run_teardown()

    def _build_tick_inputs(
        self,
        *,
        now: Timestamp,
        run_schedule: bool,
        run_reconcile: bool,
        run_autoscale: bool,
        scan_timeouts: bool,
    ) -> _TickInputs:
        """Assemble the due phases' controller-owned inputs.

        A placement-owning (``CLUSTER_VIEW``) backend's reconcile request comes
        from its own dispatch drain (a write), built first. The controller then
        reads only its own state in one read snapshot — the routing inputs
        (pending tasks + budgets) for scheduling and the execution-timeout rows;
        each backend reads its own workers, so nothing here is partitioned.
        """
        inputs = _TickInputs()

        # Placement-owning backends each drain their own pending dispatch first.
        if run_reconcile:
            for backend_id in self._dispatch_backends:
                drain = self._drain_dispatch_snapshot(backend_id)
                inputs.reconcile_requests[backend_id] = ReconcileRequest(
                    tasks_to_run=drain.tasks_to_run,
                    running_tasks=drain.running_tasks,
                )

        # Dedicated control pool: the tick's snapshot must not queue behind a slow
        # dashboard read for a connection.
        with self._db.control_read_snapshot() as snap:
            if run_schedule:
                inputs.routing = build_routing_inputs(snap, self._config.user_budget_defaults)
                if self._config.peers:
                    inputs.queued_federation = build_queued_candidates(snap)
                    inputs.expired_queued_federation = reads.expired_queued_handoffs(snap, now.epoch_ms())
            # Execution-timeout finalization is global across worker-daemon and
            # K8s backends. K8s gangs rely on it because they omit
            # activeDeadlineSeconds.
            if run_reconcile and scan_timeouts:
                inputs.timeout_rows = reads.scan_execution_timeout_rows(snap)
        return inputs

    def _build_runtime(self, backend_id: str) -> BackendRuntime:
        """Assemble the controller-owned deps a backend builds its worker source from.

        The backend joins this with its own liveness tracker to build a worker source
        scoped to its scale groups, covering exactly the workers that backend owns.
        The default backend also owns workers whose scale group is unmapped, matching
        the scale-group→backend resolution used everywhere else.
        """

        def owns_scale_group(scale_group: str) -> bool:
            return self.backend_id_for_scale_group(scale_group) == backend_id

        return BackendRuntime(
            backend_id=backend_id,
            db=self._db,
            owns_scale_group=owns_scale_group,
            budget_defaults=self._config.user_budget_defaults,
        )

    def _worker_to_backend_map(self, snap: Tx) -> dict[WorkerId, str]:
        """Map each persisted worker to its owning backend via its scale group.

        Used only to route a failed worker's slice teardown to its owning backend;
        scheduling and reconcile do not partition workers in the controller.
        """
        return {
            wid: self._scale_group_to_backend.get(scale_group, DEFAULT_BACKEND_ID)
            for wid, scale_group in reads.worker_scale_groups(snap).items()
        }

    def _schedule_phase(self, inputs: _TickInputs) -> SchedulePhaseResult:
        """Route unpinned jobs, then run each backend's scheduler over its tasks.

        Returns the per-backend ``ScheduleResult``s, the ``(job_id, backend_id)``
        pins to stamp, and the ``(task, reason)`` pairs the meta-scheduler could
        not route (finalized UNSCHEDULABLE in the commit). The decisions do no DB
        writes. The controller groups *pending tasks* by their routed backend and
        hands each backend its slice; the backend sources its own workers. The
        global user budget is threaded across backends in ``self._backend_ids``
        order so two backends cannot double-spend one user's budget in a tick.
        """
        routing = inputs.routing
        if routing is None:
            return SchedulePhaseResult({}, [], [])
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0

        pins, routing_unschedulable = self._route_pending(routing)
        unschedulable_jobs = {task.job_id for task, _ in routing_unschedulable}
        backend_of_job = self._make_backend_of_job(pins, unschedulable_jobs, routing)

        tasks_by_backend: dict[str, list[PendingTask]] = {backend_id: [] for backend_id in self._backend_ids}
        for task in routing.pending_task_rows:
            backend_id = backend_of_job(task.job_id)
            if backend_id in tasks_by_backend:
                tasks_by_backend[backend_id].append(task)

        results: dict[str, ScheduleResult] = {}
        user_spend = dict(routing.user_spend)
        for backend_id in self._backend_ids:
            pending = tasks_by_backend[backend_id]
            if not pending:
                results[backend_id] = ScheduleResult()
                continue
            kept_jobs = {task.job_id for task in pending}
            result = self._backends[backend_id].schedule(
                ScheduleRequest(
                    pending_task_rows=pending,
                    requested_bands={jid: band for jid, band in routing.requested_bands.items() if jid in kept_jobs},
                    user_spend=user_spend,
                    user_budget_limits=routing.user_budget_limits,
                    user_budget_defaults=routing.user_budget_defaults,
                    max_tasks_per_job_per_cycle=self._config.max_tasks_per_job_per_cycle,
                    trace=trace,
                )
            )
            results[backend_id] = result
            self._accumulate_user_spend(user_spend, result.assignments, routing)

        return SchedulePhaseResult(results, list(pins.items()), routing_unschedulable)

    def _route_pending(self, routing: RoutingInputs) -> tuple[dict[JobName, str], list[tuple[PendingTask, str]]]:
        """Run the task->backend meta-scheduler over this tick's unpinned jobs.

        Unpinned jobs (``backend_id == ""``) are routed once; pinned jobs keep
        their backend. Returns the new pins and the per-task UNSCHEDULABLE list for
        jobs that match no backend.
        """
        unpinned: dict[JobName, RoutableJob] = {}
        rows_by_job: dict[JobName, list[PendingTask]] = {}
        for task in routing.pending_task_rows:
            rows_by_job.setdefault(task.job_id, []).append(task)
            if task.backend_id == "" and task.job_id not in unpinned:
                unpinned[task.job_id] = RoutableJob(
                    job_id=task.job_id,
                    constraints=constraints_from_json(task.constraints_json),
                )
        if not unpinned:
            self._last_unroutable_jobs = {}
            return {}, []
        result = route_jobs_to_backends(list(unpinned.values()), self._backend_routing, self._backend_index)
        routing_unschedulable = [
            (task, reason) for job_id, reason in result.unschedulable.items() for task in rows_by_job.get(job_id, [])
        ]
        self._last_unroutable_jobs = {job_id.to_wire(): reason for job_id, reason in result.unschedulable.items()}
        return result.pins, routing_unschedulable

    def _make_backend_of_job(
        self, pins: dict[JobName, str], unschedulable_jobs: set[JobName], routing: RoutingInputs
    ) -> "Callable[[JobName], str]":
        """Build the job->backend resolver for grouping this tick's pending tasks.

        A job routed UNSCHEDULABLE this tick maps to ``""`` so no backend adopts it
        (it is finalized in the commit instead); a job pinned this tick maps to its
        pin; an already-pinned job to its stored backend_id; anything else to the
        default backend.
        """
        db_backend = {task.job_id: task.backend_id for task in routing.pending_task_rows}

        def resolve(job_id: JobName) -> str:
            if job_id in unschedulable_jobs:
                return ""
            if job_id in pins:
                return pins[job_id]
            backend_id = db_backend.get(job_id, "")
            return backend_id if backend_id else DEFAULT_BACKEND_ID

        return resolve

    def _accumulate_user_spend(
        self, user_spend: dict[str, int], assignments: list[Assignment], routing: RoutingInputs
    ) -> None:
        """Add each non-BATCH assignment's resource value to the in-tick spend tally.

        Excludes BATCH (matching ``compute_user_spend``) so a later backend in this
        tick sees the budget the earlier ones already committed.
        """
        if not assignments:
            return
        rows_by_task = {task.task_id: task for task in routing.pending_task_rows}
        for assignment in assignments:
            if assignment.priority_band == job_pb2.PRIORITY_BAND_BATCH:
                continue
            row = rows_by_task.get(assignment.task_id)
            if row is None:
                continue
            counts = device_counts_from_json(row.res_device_json)
            value = resource_value(row.res_cpu_millicores, row.res_memory_bytes, counts.gpu + counts.tpu)
            user_spend[assignment.task_id.user] = user_spend.get(assignment.task_id.user, 0) + value

    def _merge_schedule_results(self, results: dict[str, ScheduleResult]) -> ScheduleResult:
        """Concatenate the per-backend schedule results into one for the commit.

        List fields concatenate; diagnostics merge. The cached
        ``scheduling_context`` (dashboard diagnostics) is the representative
        backend's post-placement context. With one backend this is exactly that
        backend's result.
        """
        assignments: list[Assignment] = []
        preemptions: list[TerminalDecision] = []
        unschedulable: list[PendingTask] = []
        residual_demand = []
        diagnostics: dict[str, str] = {}
        scheduling_context: SchedulingContext | None = None
        for backend_id in self._backend_ids:
            result = results.get(backend_id)
            if result is None:
                continue
            assignments.extend(result.assignments)
            preemptions.extend(result.preemptions)
            unschedulable.extend(result.unschedulable)
            residual_demand.extend(result.residual_demand)
            diagnostics.update(result.diagnostics)
        representative = results.get(DEFAULT_BACKEND_ID) or next(
            (results[bid] for bid in self._backend_ids if bid in results), None
        )
        if representative is not None and representative.scheduling_context is not None:
            scheduling_context = representative.scheduling_context
        return ScheduleResult(
            assignments=assignments,
            preemptions=preemptions,
            unschedulable=unschedulable,
            residual_demand=residual_demand,
            diagnostics=diagnostics,
            scheduling_context=scheduling_context,
        )

    def _commit_tick(
        self,
        *,
        sched_result: ScheduleResult | None,
        sched_results: dict[str, ScheduleResult],
        backend_pins: list[tuple[JobName, str]],
        routing_unschedulable: list[tuple[PendingTask, str]],
        recon_results: dict[str, ReconcileResult],
        timeout_decisions: list[TerminalDecision],
        pending_kicks: list[PendingKick],
        auto_results: dict[str, AutoscaleResult],
        federation_promotions: list[Promotion],
        expired_queued_federation: list[JobName],
        now: Timestamp,
    ) -> list[Promotion]:
        """Apply this tick's merged decisions and authored effects in one write transaction.

        Order within the txn: schedule decisions (incl. backend pins + routing
        UNSCHEDULABLE), queued-handoff scheduling-timeout failures, federation queue
        promotions, each backend's reconcile effects, execution-timeout finalizations,
        administrative kicks, per-backend autoscaler state. Each backend already authored
        its own ``effects`` during reconcile; the controller just commits them uniformly.
        A no-op tick opens no transaction.

        Returns the federation promotions whose conditional CAS actually committed (a
        concurrent cancel/terminalize between the tick's read and this write drops the
        rest); the caller charges the reservation ledger for those.
        """
        states = [result.autoscaler_state for result in auto_results.values() if result.autoscaler_state is not None]

        has_sched = sched_result is not None and bool(
            sched_result.unschedulable
            or sched_result.assignments
            or sched_result.preemptions
            or backend_pins
            or routing_unschedulable
        )
        has_recon = any(not result.effects.is_empty for result in recon_results.values())
        if not (
            has_sched
            or has_recon
            or timeout_decisions
            or pending_kicks
            or states
            or federation_promotions
            or expired_queued_federation
        ):
            return []

        confirmed: list[Promotion] = []
        with self._db.transaction() as cur:
            if sched_result is not None:
                self._commit_schedule_decisions(
                    cur, sched_result, sched_results, now, backend_pins, routing_unschedulable
                )
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
            for backend_id in self._backend_ids:
                result = recon_results.get(backend_id)
                if result is not None and not result.effects.is_empty:
                    commit_effects(cur, result.effects)
            if timeout_decisions:
                finalize(cur, timeout_decisions, now=now)
            if pending_kicks:
                # Resolve after the schedule/reconcile writes so the attempt
                # re-check sees this tick's reassignments.
                kick_decisions = self._resolve_pending_kicks(cur, pending_kicks)
                if kick_decisions:
                    finalize(cur, kick_decisions, now=now)
                    logger.info("Admin kick: finalized %d task attempt(s)", len(kick_decisions))
            for state in states:
                persist_autoscaler_state(cur, state)
        return confirmed

    def _commit_schedule_decisions(
        self,
        cur: Tx,
        result: ScheduleResult,
        results: dict[str, ScheduleResult],
        now: Timestamp,
        backend_pins: list[tuple[JobName, str]],
        routing_unschedulable: list[tuple[PendingTask, str]],
    ) -> None:
        """Persist a ``ScheduleResult`` within the caller's write transaction.

        ``result`` is the merged decision; ``results`` is the per-backend split,
        used to re-check each backend's assignments against the tracker it owns.
        Backend pins stamp ``backend_id`` on routed jobs+tasks; jobs that match no
        backend finalize UNSCHEDULABLE; expired/deadline tasks finalize
        UNSCHEDULABLE; assignments stamp ASSIGNED (carrying the backend-computed
        priority band); preemption victims finalize PREEMPT.
        """
        if backend_pins:
            writes.stamp_backend(cur, backend_pins)
        if routing_unschedulable:
            finalize(
                cur,
                [
                    TerminalDecision(kind=TerminalKind.UNSCHEDULABLE, task_id=task.task_id, reason=reason)
                    for task, reason in routing_unschedulable
                ],
                now=now,
            )
        if result.unschedulable:
            finalize(cur, self._unschedulable_decisions(result.unschedulable), now=now)
        # Each backend's assignments are re-checked against the liveness tracker that
        # backend owns. Walking backends in order reproduces the merged ordering.
        for backend_id in self._backend_ids:
            backend_result = results.get(backend_id)
            if backend_result is None or not backend_result.assignments:
                continue
            health = self._backends[backend_id].health
            assert health is not None, f"backend {backend_id!r} produced assignments without a liveness tracker"
            ops.task.assign(cur, backend_result.assignments, health=health)
        if result.preemptions:
            finalize(cur, result.preemptions, now=now)
            logger.info("Preemption pass: %d tasks preempted", len(result.preemptions))

    def _run_scheduling(self) -> SchedulingOutcome:
        """Run one self-contained scheduling cycle (its own snapshot + commits).

        This is the dry-run scheduling path; the live control tick computes its
        schedule via ``_schedule_phase`` and commits it in the shared end-of-tick
        transaction instead.

        The controller reads its own routing inputs (pending tasks + budgets) and
        hands them to the representative backend, which sources its own workers and
        returns the pure placement decision; the controller then commits the
        assignments, preemptions, and unschedulable marks. A worker-daemon backend
        runs the full gates → order → find_assignments → preemption pipeline; a
        cluster backend returns an empty result (Kueue schedules).

        No lock is needed since the control driver is single-threaded. Every DB
        access is serialized by ControllerDB._lock with multi-statement
        mutations wrapped in BEGIN IMMEDIATE transactions.
        """
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0

        with self._db.control_read_snapshot() as snap:
            routing = build_routing_inputs(snap, self._config.user_budget_defaults)

        if trace:
            logger.info(
                "[TRACE round=%d] Phase 0: %d pending tasks",
                self._scheduling_round,
                len(routing.pending_task_rows),
            )

        if not routing.pending_task_rows:
            self._scheduling_diagnostics = {}
            self._last_scheduling_context = None
            return SchedulingOutcome.NO_PENDING_TASKS

        result = self._representative_backend.schedule(
            ScheduleRequest(
                pending_task_rows=routing.pending_task_rows,
                requested_bands=routing.requested_bands,
                user_spend=routing.user_spend,
                user_budget_limits=routing.user_budget_limits,
                user_budget_defaults=routing.user_budget_defaults,
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
        self._last_scheduling_context = result.scheduling_context

        if result.assignments or result.preemptions:
            log_event(
                "scheduling_pass_completed",
                "scheduler",
                assignments=len(result.assignments),
                preempted=len(result.preemptions),
                pending=len(routing.pending_task_rows),
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
        # The dry-run scheduling path routes through the representative backend, so
        # these assignments are all its workers — re-checked against its tracker.
        health = self._representative_backend.health
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

    def _drain_dispatch_snapshot(self, backend_id: str) -> reads.ControlSnapshot:
        """Promote PENDING->ASSIGNED for a placement-owning backend and ride the drain.

        The dispatch drain is the single DB write a ``CLUSTER_VIEW`` backend needs
        before reconcile (the controller owns the write; the backend places tasks
        itself). It runs in its own write transaction, so a cluster backend's tick
        commits twice (drain + end-of-tick) rather than once. With multiple
        backends the drain is scoped to ``backend_id``'s tasks; a lone backend
        drains every pending task.
        """
        max_promotions = self._promotion_bucket.available
        backend_filter = None if len(self._backends) == 1 else backend_id
        with self._db.transaction() as cur:
            batch = dispatch.drain_for_dispatch(cur, max_promotions=max_promotions, backend_id=backend_filter)
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
        """Tear down the workers queued by :meth:`request_worker_eviction`.

        Resolves each evicted worker to its owning backend (by scale group, while
        the rows still carry it) and hands that backend its share to tear down --
        the same fail → slice-and-sibling teardown → forget the reconcile fold
        runs, off the liveness path. The recycled-IP eviction reason is recorded on
        the failure.
        """
        with self._pending_evictions_lock:
            if not self._pending_evictions:
                return
            drained = sorted(self._pending_evictions)
            self._pending_evictions.clear()
        with self._db.read_snapshot() as snap:
            worker_to_backend = self._worker_to_backend_map(snap)
        reason = "address reused by newly-registered worker (recycled IP)"
        for backend_id in self._backend_ids:
            group = [wid for wid in drained if worker_to_backend.get(wid, DEFAULT_BACKEND_ID) == backend_id]
            if group:
                self._backends[backend_id].teardown(group, reason=reason)

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
    def _representative_backend(self) -> TaskBackend:
        """The backend the dry-run path and on-demand service RPCs route through.

        Prefers :data:`DEFAULT_BACKEND_ID`; otherwise the first backend in
        processing order. With a single backend this is that backend.
        """
        return self._backends.get(DEFAULT_BACKEND_ID) or self._backends[self._backend_ids[0]]

    @property
    def backends(self) -> dict[str, TaskBackend]:
        """The controller's ``{backend_id: TaskBackend}`` collection."""
        return self._backends

    @property
    def federation(self) -> FederationManager:
        """The federation manager: peer registry, heartbeat, and submit-time router."""
        return self._federation

    def backend_id_for_scale_group(self, scale_group: str) -> str:
        """Return the backend id owning ``scale_group``, or DEFAULT_BACKEND_ID."""
        return self._scale_group_to_backend.get(scale_group, DEFAULT_BACKEND_ID)

    @property
    def scale_group_to_backend(self) -> dict[str, str]:
        """The ``{scale_group: backend_id}`` routing map."""
        return self._scale_group_to_backend

    @property
    def last_unroutable_jobs(self) -> dict[str, str]:
        """Job wire ids -> reason from the last scheduling tick's routing pass."""
        return self._last_unroutable_jobs

    @property
    def provider(self) -> TaskBackend:
        return self._representative_backend

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        """Union of every backend's capabilities (which dashboard tabs/RPCs apply)."""
        return frozenset(cap for backend in self._backends.values() for cap in backend.capabilities)

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
