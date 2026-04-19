# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process harness for reproducing the 2026-04-18 autoscaler spike.

Boots an ``Autoscaler`` (optionally a full ``Controller``) against a *copy* of
a captured controller sqlite snapshot with ``InMemoryGcpService`` as the GCP
backend. Scenarios layer stimulus and measurement on top.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from iris.cluster.config import create_autoscaler
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.controller.worker_provider import RpcWorkerStubFactory, WorkerProvider
from iris.cluster.providers.gcp import workers as gcp_workers_module
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.types import find_free_port
from iris.cluster.service_mode import ServiceMode
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration

from iris.loadtest.configs import DEFAULT_MARIN_YAML, load_scale_group_protos, zones_for
from iris.loadtest.fakes import LoadtestGcpService
from iris.loadtest.preload import (
    DEFAULT_FD_TARGET,
    ensure_fd_limit,
    group_distribution_from_snapshot,
    preload_workers as _preload_workers_impl,
    scale_allocations,
)
from iris.loadtest.synthetic_worker import (
    HEARTBEAT_INTERVAL_SECONDS,
    LifecycleDelays,
    SyntheticWorkerPool,
)

logger = logging.getLogger(__name__)


def _like_match(name: str, pattern: str) -> bool:
    """Approximate SQL LIKE in-memory: ``%`` matches anything."""
    import fnmatch

    return fnmatch.fnmatchcase(name, pattern.replace("%", "*"))


LABEL_PREFIX = "iris-loadtest"
PROJECT_ID = "loadtest-project"
DEFAULT_EVALUATION_INTERVAL = Duration.from_seconds(1.0)


@dataclass
class HarnessConfig:
    """Tunables for the load-test harness."""

    evaluation_interval: Duration = DEFAULT_EVALUATION_INTERVAL
    # Map of group_name_glob -> failure spec for ``InMemoryGcpService.inject_failure``.
    inject_failures: dict | None = None
    tpu_create_timeout_seconds: float | None = None
    max_inflight_scale_ups: int | None = None
    enable_timeout_backoff: bool = True
    # Per-transition delays for the synthetic-worker task lifecycle. Prod
    # cadence (~60 s each) is the default; fast tests compress to <1 s.
    synthetic_worker_building_seconds: float = 60.0
    synthetic_worker_running_seconds: float = 60.0
    # Controller ping cadence — passed through to ``ControllerConfig.heartbeat_interval``.
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS
    # Raise RLIMIT_NOFILE at startup so pre-loaded fleets don't blow the
    # default 256-FD soft cap. Set to 0 to skip.
    fd_limit_target: int = DEFAULT_FD_TARGET
    # 0 picks an ephemeral port (preferred for parallel tests).
    controller_port: int = 0
    controller_checkpoint_interval: Duration | None = None
    # When True, controller uses direct StartTasks/StopTasks RPCs + task-updater
    # thread. When False, falls back to the legacy monolithic Heartbeat path.
    use_split_heartbeat: bool = True


@dataclass
class HarnessMetrics:
    """Snapshot of harness-level counters."""

    active_scale_up_threads: int = 0
    db_writer_lock_hold_ms_p95: float | None = None
    dashboard_query_ms_p95: float | None = None


def _known_migration_names() -> list[str]:
    """Return every migration file shipped with the current tree.

    We stamp each name into ``schema_migrations`` before opening the copy so
    `ControllerDB.apply_migrations()` finds nothing pending — we never run an
    untrusted migration against the 1.5 GB captured snapshot.
    """
    import iris.cluster.controller.db as db_module

    migrations_dir = Path(db_module.__file__).with_name("migrations")
    return sorted(p.name for p in migrations_dir.glob("*.py") if not p.name.startswith("__"))


def _purge_stale_workers(db_path: Path) -> int:
    """Delete every ``workers`` row from the snapshot copy.

    The load-test harness must NEVER attempt to connect to real workers —
    it is only intended to exercise synthetic workers that the harness itself
    spawns. Any ``workers`` row present in the snapshot at harness boot has
    a stale, unreachable prod IP; leaving those rows in place would cause
    the Controller's ping / poll / heartbeat loops to burn every cycle
    timing out on ~1,000 dead endpoints before the harness even gets a
    probe off.

    Called unconditionally from :meth:`LoadtestHarness.start` BEFORE any
    controller thread is constructed — otherwise the ping-loop could race
    the purge and dispatch RPCs to real TPUs.

    Returns the number of rows deleted. Must be called before ``ControllerDB``
    is opened against the file; the sqlite3 direct connection bypasses the DB
    object's write lock. Scaling groups + job/task rows are left untouched.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        # Foreign-key cascades wipe worker_attributes / worker_resource_history
        # / dispatch_queue / worker_task_history automatically. ``tasks`` has a
        # non-cascading FK on ``current_worker_id`` so null it first.
        conn.execute("PRAGMA foreign_keys = ON")
        count = conn.execute("SELECT COUNT(*) FROM workers").fetchone()[0]
        conn.execute("UPDATE tasks SET current_worker_id = NULL, current_worker_address = NULL")
        conn.execute("DELETE FROM workers")
        conn.commit()
        return int(count)
    finally:
        conn.close()


def _pin_migrations(db_path: Path) -> list[str]:
    """Stamp every known migration as already-applied on the copy.

    Returns the list of migration names newly stamped (for logging/assertions).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at_ms INTEGER NOT NULL
            )
            """
        )
        existing = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        existing_stems = {Path(name).stem for name in existing}

        now_ms = int(time.time() * 1000)
        newly_stamped: list[str] = []
        for name in _known_migration_names():
            if Path(name).stem in existing_stems:
                continue
            conn.execute(
                "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                (name, now_ms),
            )
            newly_stamped.append(name)
        conn.commit()
        return newly_stamped
    finally:
        conn.close()


def _placeholder_config(name: str) -> config_pb2.ScaleGroupConfig:
    """Build a minimal CPU on-demand ``ScaleGroupConfig`` for *name*.

    Used as fallback for scale-groups present in the DB snapshot but absent
    from the cluster YAML (historical config drift). The autoscaler needs a
    well-formed proto to instantiate a ``ScalingGroup``; we never exercise
    scale-up on these, but we do need them present so the snapshot's
    ``scaling_groups`` rows line up.
    """
    cfg = config_pb2.ScaleGroupConfig(name=name, num_vms=1, max_slices=0)
    cfg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    cfg.resources.cpu_millicores = 1000
    cfg.resources.memory_bytes = 1024**3
    cfg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
    return cfg


def _db_scale_group_names(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT name FROM scaling_groups ORDER BY name").fetchall()
    finally:
        conn.close()
    return [name for (name,) in rows]


def _load_scale_group_configs(
    db_path: Path,
    cluster_config_path: Path | None,
) -> dict[str, config_pb2.ScaleGroupConfig]:
    """Return the set of ScaleGroupConfig protos to boot the autoscaler with.

    When *cluster_config_path* is provided, real per-(size, zone) protos from
    the YAML take precedence. Any scaling-group row in the DB that the YAML
    doesn't cover falls back to a CPU placeholder — groups from prior config
    generations still exist in the snapshot and the autoscaler needs a proto
    for each persisted row.
    """
    db_names = _db_scale_group_names(db_path)

    real_configs: dict[str, config_pb2.ScaleGroupConfig] = {}
    if cluster_config_path is not None:
        real_configs = load_scale_group_protos(cluster_config_path)

    merged: dict[str, config_pb2.ScaleGroupConfig] = dict(real_configs)
    for name in db_names:
        if name not in merged:
            merged[name] = _placeholder_config(name)
    return merged


def _build_scale_group_name_lookup(
    scale_group_configs: dict[str, config_pb2.ScaleGroupConfig],
) -> list[tuple[str, str, str]]:
    """Return a list of ``(mangled_prefix, group_name, device_variant)`` triples.

    Sorted by decreasing mangled-prefix length so the longest match wins.
    The GCP provider truncates slice names to GCE's 63-char limit; matching
    on the longest shared prefix handles truncation gracefully.
    """
    triples: list[tuple[str, str, str]] = []
    for name, config in scale_group_configs.items():
        mangled = name.replace("_", "-")
        variant = config.resources.device_variant if config.HasField("resources") else ""
        triples.append((mangled, name, variant))
    triples.sort(key=lambda t: len(t[0]), reverse=True)
    return triples


def _resolve_scale_group(
    lookup_table: list[tuple[str, str, str]],
    slice_name: str,
    accelerator_type: str,
) -> tuple[str, str]:
    """Return ``(group_name, device_variant)`` for a slice name.

    Falls back to ``(unknown, accelerator_type)`` so a registration still
    proceeds even if a slice name doesn't match any known group (e.g.
    placeholder CPU groups).
    """
    for mangled_prefix, group_name, variant in lookup_table:
        if mangled_prefix in slice_name:
            return group_name, variant or accelerator_type
    return ("unknown", accelerator_type)


def _make_autoscaler_config(evaluation_interval: Duration) -> config_pb2.AutoscalerConfig:
    cfg = config_pb2.AutoscalerConfig()
    cfg.evaluation_interval.CopyFrom(duration_to_proto(evaluation_interval))
    cfg.scale_up_delay.CopyFrom(duration_to_proto(Duration.from_seconds(60)))
    cfg.scale_down_delay.CopyFrom(duration_to_proto(Duration.from_minutes(10)))
    return cfg


class LoadtestHarness:
    """Boot + shutdown controller autoscaler in-process, against a DB copy.

    Usage::

        with LoadtestHarness(snapshot_copy_dir) as h:
            time.sleep(5)
            print(h.metrics())

    `snapshot_copy_dir` must already contain ``controller.sqlite3``. Use the
    conftest `snapshot_copy_dir` fixture to get one.
    """

    def __init__(
        self,
        db_dir: Path,
        *,
        config: HarnessConfig | None = None,
        cluster_config_path: Path | None = DEFAULT_MARIN_YAML,
    ) -> None:
        """Construct a harness bound to *db_dir*.

        Args:
            db_dir: directory containing the per-test ``controller.sqlite3``.
            config: tunables (evaluation interval, failure injection plans).
            cluster_config_path: YAML used to expand scale-group protos.
                Defaults to ``lib/iris/examples/marin.yaml`` (the production
                cluster config). Pass ``None`` to fall back entirely to DB-row
                CPU placeholders (useful for micro-tests).
        """
        self._db_dir = db_dir
        self._config = config or HarnessConfig()
        self._cluster_config_path = cluster_config_path
        self._autoscaler: Autoscaler | None = None
        self._provider: GcpWorkerProvider | None = None
        self._gcp_service: LoadtestGcpService | None = None
        self._db: ControllerDB | None = None
        self._transitions: ControllerTransitions | None = None
        self._worker_pool: SyntheticWorkerPool | None = None
        self._scale_group_configs: dict[str, config_pb2.ScaleGroupConfig] = {}
        self._threads = ThreadContainer(name="loadtest")
        self._controller: Controller | None = None
        self._controller_url: str | None = None
        self._controller_temp_dir: Any | None = None
        self._controller_threads: ThreadContainer | None = None
        self._task_provider: WorkerProvider | None = None
        # `patch.object(...)` returns a `_patch` helper — there is no public
        # stable type to annotate against, so this stays as `Any`.
        self._bootstrap_patch: Any | None = None
        self._original_mtime: float | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._autoscaler is not None:
            raise RuntimeError("Harness already started")

        if self._config.fd_limit_target > 0:
            soft, hard = ensure_fd_limit(self._config.fd_limit_target)
            logger.info("RLIMIT_NOFILE soft=%s hard=%s (target=%d)", soft, hard, self._config.fd_limit_target)

        db_path = self._db_dir / ControllerDB.DB_FILENAME
        if not db_path.exists():
            raise FileNotFoundError(f"No controller.sqlite3 in {self._db_dir}")
        self._original_mtime = db_path.stat().st_mtime

        pinned = _pin_migrations(db_path)
        if pinned:
            logger.info("Pinned %d migration(s) on copy: %s", len(pinned), pinned)

        # Mandatory: the load-test harness connects only to synthetic workers
        # it spawns itself. Any captured prod ``workers`` row points to an
        # unreachable address and MUST be removed before any controller
        # thread (ping-loop, poll-loop, task-updater) could RPC it.
        purged = _purge_stale_workers(db_path)
        logger.info(
            "Purged %d stale worker rows from snapshot copy before controller boot",
            purged,
        )

        # Replace `_spawn_bootstrap_thread` with a no-op for the lifetime of
        # the harness. Without this, any slice the harness ever creates via
        # `create_slice` would launch a daemon thread polling synthetic
        # `10.0.x.y` health endpoints until bootstrap timeout.
        self._bootstrap_patch = patch.object(
            gcp_workers_module,
            "_spawn_bootstrap_thread",
            lambda handle, bootstrap_fn: None,
        )
        self._bootstrap_patch.start()

        inner_gcp = InMemoryGcpService(
            mode=ServiceMode.DRY_RUN,
            project_id=PROJECT_ID,
            label_prefix=LABEL_PREFIX,
        )
        self._gcp_service = LoadtestGcpService(
            inner_gcp,
            tpu_create_timeout_seconds=self._config.tpu_create_timeout_seconds,
        )
        zones = zones_for(self._cluster_config_path) if self._cluster_config_path else ["us-central1-a"]
        gcp_platform_cfg = config_pb2.GcpPlatformConfig(project_id=PROJECT_ID, zones=zones)
        self._provider = GcpWorkerProvider(
            gcp_config=gcp_platform_cfg,
            label_prefix=LABEL_PREFIX,
            gcp_service=self._gcp_service,
        )

        self._db = ControllerDB(db_dir=self._db_dir)
        # Sanity check — with the pin above, no migrations should have run.
        # If any did, we'd have mutated the copy's schema against an untested
        # delta. Raise loudly rather than silently continue.
        assert not pinned or all(
            (self._db_dir / ControllerDB.DB_FILENAME).exists() for _ in [0]
        ), "DB vanished after migration pin"

        scale_group_configs = _load_scale_group_configs(db_path, self._cluster_config_path)
        self._scale_group_configs = scale_group_configs
        logger.info(
            "Loaded %d scale-group configs (cluster_yaml=%s)",
            len(scale_group_configs),
            self._cluster_config_path,
        )

        self._transitions = ControllerTransitions(db=self._db)
        delays = LifecycleDelays(
            building_seconds=self._config.synthetic_worker_building_seconds,
            running_seconds=self._config.synthetic_worker_running_seconds,
        )
        self._worker_pool = SyntheticWorkerPool(transitions=self._transitions, db=self._db, delays=delays)
        # The GCP name builder translates underscores to hyphens and truncates
        # at the GCE 63-char limit; match by longest mangled-prefix substring.
        lookup_table = _build_scale_group_name_lookup(scale_group_configs)
        self._gcp_service.attach_worker_pool(
            self._worker_pool,
            scale_group_lookup=lambda name, accel_type: _resolve_scale_group(lookup_table, name, accel_type),
        )

        self._autoscaler = create_autoscaler(
            platform=self._provider,
            autoscaler_config=_make_autoscaler_config(self._config.evaluation_interval),
            scale_groups=scale_group_configs,
            label_prefix=LABEL_PREFIX,
            base_worker_config=None,  # disables bootstrap metadata injection
            threads=self._threads,
            db=self._db,
        )
        self._start_full_controller()

    def _start_full_controller(self) -> None:
        """Construct and start a real :class:`Controller` in-process.

        Runs every background thread the production controller runs
        (scheduling, ping, task-updater, poll, profile, prune, autoscaler,
        checkpoint) and binds a real Connect/RPC uvicorn server on
        ``controller_port``. Probes hit the real RPC surface via
        :class:`ControllerServiceClientSync`.
        """
        assert self._db is not None
        import tempfile as _tempfile  # local import: only used here

        port = self._config.controller_port or find_free_port()
        # Checkpoints are disabled in the harness: remote_state_dir must be
        # set (Controller rejects empty) so we point at a throwaway file://.
        self._controller_temp_dir = _tempfile.TemporaryDirectory(prefix="loadtest-controller-state-")
        state_dir = Path(self._controller_temp_dir.name)
        remote_state_dir = f"file://{state_dir}"

        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=port,
            remote_state_dir=remote_state_dir,
            heartbeat_interval=Duration.from_seconds(self._config.heartbeat_interval_seconds),
            local_state_dir=self._db_dir,
            checkpoint_interval=self._config.controller_checkpoint_interval,
            # Null-auth mode: no verifier, no provider — the dashboard's
            # NullAuthInterceptor lets anonymous calls through, matching the
            # default local-cluster test setup.
            auth_verifier=None,
            auth_provider=None,
            use_split_heartbeat=self._config.use_split_heartbeat,
        )

        # TaskProvider used by scheduler / ping / poll loops. Hits the
        # synthetic workers' real Connect/RPC endpoints over sockets.
        task_provider = WorkerProvider(stub_factory=RpcWorkerStubFactory())
        self._task_provider = task_provider

        # Give the Controller its own ThreadContainer so its loops are tracked
        # separately from the autoscaler's scale-up threads (metrics inspects
        # ``_autoscaler._threads`` for active-scale-up counts and would
        # otherwise see controller loops mixed in).
        controller_threads = ThreadContainer(name="loadtest-controller")
        self._controller_threads = controller_threads
        self._controller = Controller(
            config=controller_config,
            provider=task_provider,
            autoscaler=self._autoscaler,
            threads=controller_threads,
            db=self._db,
        )
        self._controller.start()
        self._controller_url = self._controller.url
        logger.info("Full controller started at %s", self._controller_url)

    def stop(self) -> None:
        # Order matters: stop the controller (which drains the autoscaler) BEFORE
        # tearing down the worker pool. The autoscaler's scale-up threads call
        # LoadtestGcpService.tpu_create, which synchronously spawns subprocess
        # workers and registers them against the DB — if we stopped the worker
        # pool first, in-flight tpu_create calls would race with db.close() and
        # fail with "Cannot operate on a closed database."
        if self._controller is not None:
            # Controller.stop() drains its own threads and calls
            # autoscaler.shutdown() internally.
            try:
                self._controller.stop()
            except Exception:
                logger.exception("Controller shutdown raised; continuing")
            self._controller = None
            self._controller_url = None
            self._autoscaler = None

        if self._worker_pool is not None:
            self._worker_pool.stop_all(timeout=5.0)
            self._worker_pool = None

        if self._db is not None:
            self._db.close()
            self._db = None

        if self._controller_temp_dir is not None:
            try:
                self._controller_temp_dir.cleanup()
            except Exception:
                logger.exception("Failed to clean controller temp dir")
            self._controller_temp_dir = None

        if self._bootstrap_patch is not None:
            self._bootstrap_patch.stop()
            self._bootstrap_patch = None

        self._provider = None
        self._gcp_service = None

    def __enter__(self) -> LoadtestHarness:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- accessors -----------------------------------------------------------

    @property
    def autoscaler(self) -> Autoscaler:
        if self._autoscaler is None:
            raise RuntimeError("Harness not started")
        return self._autoscaler

    @property
    def db(self) -> ControllerDB:
        if self._db is None:
            raise RuntimeError("Harness not started")
        return self._db

    @property
    def gcp_service(self) -> LoadtestGcpService:
        if self._gcp_service is None:
            raise RuntimeError("Harness not started")
        return self._gcp_service

    @property
    def transitions(self) -> ControllerTransitions:
        if self._transitions is None:
            raise RuntimeError("Harness not started")
        return self._transitions

    @property
    def worker_pool(self) -> SyntheticWorkerPool | None:
        return self._worker_pool

    @property
    def controller_url(self) -> str | None:
        """Return the Connect/RPC URL of the in-process Controller."""
        return self._controller_url

    @property
    def controller(self):
        """The in-process Controller instance."""
        return self._controller

    def reset_scale_group_state(self, name_pattern: str | None = None) -> None:
        """Zero backoff/failure/quota fields for matching scaling_groups rows.

        The snapshot records real prod backoff (e.g. europe-west4-a has 64
        consecutive failures and an active backoff_until); tests that want to
        observe fresh scale-up behaviour on those groups must first clear
        those fields. ``name_pattern`` is a SQL LIKE pattern; ``None`` clears
        everything.

        This only touches the DB copy and must be called after ``start()``.
        The in-memory ScalingGroup state is *not* reset here; call this
        before exercising stimuli that spin through new scale-ups.
        """
        if self._db is None:
            raise RuntimeError("Harness not started")
        with self._db.transaction() as cur:
            if name_pattern is None:
                cur.execute(
                    "UPDATE scaling_groups SET consecutive_failures=0, backoff_until_ms=0, "
                    "quota_exceeded_until_ms=0, quota_reason=''"
                )
            else:
                cur.execute(
                    "UPDATE scaling_groups SET consecutive_failures=0, backoff_until_ms=0, "
                    "quota_exceeded_until_ms=0, quota_reason='' WHERE name LIKE ?",
                    (name_pattern,),
                )
        # Also reset the in-memory ScalingGroup state so the next tick is
        # unencumbered by the pre-load values constructed from the DB row.
        for name, group in self._autoscaler._groups.items():
            if name_pattern and not _like_match(name, name_pattern):
                continue
            group._consecutive_failures = 0
            group._backoff_until = None
            group._quota_exceeded_until = None
            group._quota_reason = ""

    def preload_workers(
        self,
        *,
        count: int,
        snapshot_db: Path,
        scale_group_pattern: str | None = "tpu_%",
        slice_prefix: str = "preload",
    ) -> int:
        """Pre-populate the controller with ``count`` real-RPC synthetic workers.

        Distribution mirrors the snapshot's ``workers`` table (largest-remainder
        scaled to ``count``). Each worker binds a real localhost port, registers
        via ``ControllerTransitions.register_worker``, and responds to Ping /
        PollTasks / HealthCheck over real sockets. Lifecycle delays inherit
        from ``HarnessConfig.synthetic_worker_*_seconds`` — so fast tests can
        pass 60-worker / 0.5-s configs and prod-magnitude runs use the defaults.

        Args:
            count: Target number of live synthetic workers.
            snapshot_db: Path to the read-only snapshot to sample distribution
                from. We deliberately do not use the harness's copy — the copy
                may have been rewritten by prior test stimuli.
            scale_group_pattern: SQL LIKE filter over ``workers.scale_group``.
                Default ``tpu_%`` excludes CPU-only groups which don't exercise
                the TPU-focused autoscaler paths.
            slice_prefix: Slice-ID prefix. Preemption stimuli target pre-loaded
                workers by the mangled group name embedded in the slice-ID.

        Returns:
            The number of workers actually spawned (may differ from ``count``
            if the snapshot distribution is coarser than ``count`` and a group
            rounds to zero).
        """
        if self._worker_pool is None or self._transitions is None:
            raise RuntimeError("Harness not started")

        allocations = group_distribution_from_snapshot(
            snapshot_db,
            scale_group_configs=self._scale_group_configs,
            scale_group_pattern=scale_group_pattern,
        )
        if not allocations:
            raise RuntimeError(
                f"No snapshot allocations matched pattern={scale_group_pattern!r} "
                f"against {len(self._scale_group_configs)} known groups."
            )

        scaled = scale_allocations(allocations, total=count)
        delays = LifecycleDelays(
            building_seconds=self._config.synthetic_worker_building_seconds,
            running_seconds=self._config.synthetic_worker_running_seconds,
        )
        workers = _preload_workers_impl(
            pool=self._worker_pool,
            transitions=self._transitions,
            db=self._db,
            allocations=scaled,
            delays=delays,
            slice_prefix=slice_prefix,
        )
        return len(workers)

    # -- inspection ----------------------------------------------------------

    def metrics(self) -> HarnessMetrics:
        active = 0
        if self._autoscaler is not None:
            with self._autoscaler._threads._lock:
                active = len(self._autoscaler._threads._threads)
        return HarnessMetrics(active_scale_up_threads=active)

    def controller_thread_names(self) -> list[str]:
        """Return every live background thread the in-process Controller is running."""
        if self._controller is None or self._controller_threads is None:
            return []
        with self._controller_threads._lock:
            return sorted(t.name for t in self._controller_threads._threads if t.name is not None)

    def snapshot_unchanged(self) -> bool:
        """Return True if the DB file's mtime has not been touched since start.

        The harness opens the copy read-write (ControllerDB does not have a
        read-only mode), so small mtime drift from WAL checkpoints is expected
        on any run that writes.
        """
        if self._original_mtime is None:
            return False
        db_path = self._db_dir / ControllerDB.DB_FILENAME
        return db_path.stat().st_mtime == self._original_mtime
