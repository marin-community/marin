# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EvalDash: a benchmark panel and browsable run log over every Marin eval run.

A Marina Python app. The kernel mounts :func:`create_api` at ``/evaldash/api/`` behind its own
authentication and serves ``web/``'s build from ``dist/`` under ``/evaldash/``; this module is
therefore only the JSON API and the background record ingest behind it.

Eval runs write one canonical ``record.json`` per run to object storage. That remains the producer
and recovery format; the app's own Postgres schema is the serving catalog, and a background
reconciler scans the record roots after the API is serving and commits changes as new catalog
generations. The ``local`` store keeps the direct object scan used for development and journeys,
with no database at all.

``/status`` reports each prefix's last-probe health, the active store, and the ingest cadence;
``POST /refresh`` runs one ingest pass immediately, serialised with the loop.

Per-run drill-in endpoints read beyond the record: ``/runs/{id}/jobs`` and ``.../logs`` fetch live
iris job/attempt status and finelog log lines over Direct VPC egress, ``.../samples`` pages the
per-question parquet exports, and ``.../group`` plus ``/history`` serve a run's group siblings and a
model-by-task score-over-time series.

The caller is the kernel's: handlers read it with ``rigging.server_auth.get_verified_identity``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from marin.evaluation.eval_stats import DEFAULT_MIN_COVERAGE, Completeness, MissingPolicy, SelectionRequest
from marin.evaluation.records import (
    DEFAULT_SCAN_PREFIXES,
    EvalRunRecord,
    RecordParseFailure,
    list_record_paths,
    scan_records,
)
from marina.apps import Services
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.server_auth import get_verified_identity
from sqlalchemy.engine import Engine
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from . import review, samples
from .metrics import (
    RUN_FACETS,
    build_comparison,
    build_meta,
    build_model_detail,
    build_panel,
    panel_request,
    record_headline,
)
from .record_reconciliation import VerificationSchedule, inspect_record_paths
from .results_db import (
    PrefixStatus,
    RecordObservation,
    SourceState,
    catalog_generation,
    configure_prefixes,
    fetch_archived_models,
    fetch_snapshot,
    mark_prefix_failed,
    migrate_schema,
    prefix_statuses,
    prune_untracked_records,
    reconcile_prefix,
    set_model_archived,
    source_states,
    verify_schema,
)

logger = logging.getLogger(__name__)

CATALOG_POLL_SECONDS = 10
DEFAULT_RUNS_LIMIT = 200
MAX_RUNS_LIMIT = 1000
DEFAULT_LOG_TAIL = 200
MAX_LOG_TAIL = 5000
DEFAULT_SAMPLE_LIMIT = 50
MAX_SAMPLE_LIMIT = 500
DEFAULT_REVIEW_SAMPLES = 20
MAX_REVIEW_SAMPLES = 40
# Most models one request may compare head-to-head; mirrors the SPA's picker cap.
MAX_COMPARE_MODELS = 4
REVIEW_FILTERS = ("all", "correct", "incorrect", "ungraded")

DEFAULT_INGEST_INTERVAL = 600.0
DEFAULT_REVALIDATE_AFTER = 86400.0
# The review model is cheap and fast by design; override with EVALDASH_REVIEW_MODEL.
DEFAULT_REVIEW_MODEL = "claude-haiku-4-5-20251001"

PREFIXES_ENV = "RECORDS_PREFIXES"
STORE_ENV = "EVALDASH_STORE"
INGEST_INTERVAL_ENV = "EVALDASH_INGEST_INTERVAL"
REVALIDATE_AFTER_ENV = "EVALDASH_REVALIDATE_AFTER"
REVIEW_MODEL_ENV = "EVALDASH_REVIEW_MODEL"


class StoreMode(StrEnum):
    """Which store backs reads."""

    # The deployed service: the kernel's Postgres schema is the serving catalog.
    POSTGRES = "postgres"
    # Development and journeys: everything is served from the record snapshot, no database.
    LOCAL = "local"


@dataclass(frozen=True)
class EvaldashConfig:
    """Everything the environment decides, resolved once when the kernel mounts the app."""

    prefixes: tuple[str, ...]
    store: StoreMode
    ingest_interval: float
    revalidate_after: float
    review_model: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> EvaldashConfig:
        """Resolve the configuration, failing on an unknown store rather than guessing one."""
        raw_prefixes = environ.get(PREFIXES_ENV) or ",".join(DEFAULT_SCAN_PREFIXES)
        store = environ.get(STORE_ENV, StoreMode.POSTGRES).strip().lower()
        if store not in tuple(StoreMode):
            raise ValueError(f"unknown {STORE_ENV}={store!r}; expected one of {[mode.value for mode in StoreMode]}")
        return cls(
            prefixes=tuple(part.strip() for part in raw_prefixes.split(",") if part.strip()),
            store=StoreMode(store),
            ingest_interval=float(environ.get(INGEST_INTERVAL_ENV, DEFAULT_INGEST_INTERVAL)),
            revalidate_after=float(environ.get(REVALIDATE_AFTER_ENV, DEFAULT_REVALIDATE_AFTER)),
            review_model=environ.get(REVIEW_MODEL_ENV, DEFAULT_REVIEW_MODEL),
        )


# --------------------------------------------------------------------------------------
# Record stores
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreInfo:
    """Which store serves reads and, for Postgres, the instance/database behind it."""

    backend: str
    instance: str | None
    database: str | None
    record_count: int
    catalog_generation: int | None
    snapshot_updated_at: str | None
    catalog_error: str | None


def _deduplicate_records(records: list[EvalRunRecord]) -> list[EvalRunRecord]:
    """Keep the first record for each run ID so prefix order defines migration precedence."""
    by_id: dict[str, EvalRunRecord] = {}
    for record in records:
        by_id.setdefault(record.run_id, record)
    return list(by_id.values())


def record_to_row(record: EvalRunRecord) -> dict:
    """Flatten one record to the canonical API run-row shape (ISO ``created_at``, task list, jobs map).

    The single definition of that shape, so the run list is identical whichever store produced it.
    """
    return {
        "run_id": record.run_id,
        "group_id": record.group_id,
        "created_at": record.created_at,
        "version": record.version,
        "user_name": record.user,
        "model_name": record.model.name,
        "model_location": record.model.location,
        "eval_name": record.evaluation.name,
        "mechanism": record.evaluation.mechanism,
        "backend": record.model.backend,
        "platform": record.hardware.platform,
        "accelerator": record.hardware.accelerator,
        "region": record.hardware.region_or_cluster,
        "status": record.status.value,
        "results_path": record.results_path,
        "git_sha": record.provenance.git_sha,
        "image_digest": record.provenance.eval_runtime,
        "error": record.error,
        "tasks": [task.name for task in record.evaluation.tasks],
        "jobs": dict(record.jobs),
    }


def _group_sibling_row(record: EvalRunRecord) -> dict:
    """One sibling run in a group, for the run-detail group panel."""
    return {
        "run_id": record.run_id,
        "eval_name": record.evaluation.name,
        "model_name": record.model.name,
        "status": record.status.value,
        "created_at": record.created_at,
    }


class RecordStore:
    """Expose dashboard query views over one consistent in-memory record snapshot."""

    backend = "memory"

    def __init__(self) -> None:
        self._records: list[EvalRunRecord] = []
        self._by_id: dict[str, EvalRunRecord] = {}
        self._archived: set[str] = set()
        self._lock = threading.Lock()

    def _set_snapshot(self, records: list[EvalRunRecord]) -> None:
        by_id = {record.run_id: record for record in records}
        with self._lock:
            self._records = records
            self._by_id = by_id

    def _snapshot(self) -> tuple[list[EvalRunRecord], dict[str, EvalRunRecord]]:
        with self._lock:
            return self._records, self._by_id

    def store_info(self) -> StoreInfo:
        records, _by_id = self._snapshot()
        return StoreInfo(
            backend=self.backend,
            instance=None,
            database=None,
            record_count=len(records),
            catalog_generation=None,
            snapshot_updated_at=None,
            catalog_error=None,
        )

    def refresh(self, records: list[EvalRunRecord]) -> None:
        """Replace the direct-scan snapshot used by the local store."""
        records = _deduplicate_records(records)
        self._set_snapshot(records)
        logger.info("memory store refreshed: %d records", len(records))

    def archived_models(self) -> set[str]:
        """Model names hidden from the headline panel. In-memory in the base; a table in Postgres."""
        with self._lock:
            return set(self._archived)

    def set_model_archived(self, model_name: str, archived: bool, updated_by: str | None) -> None:
        with self._lock:
            if archived:
                self._archived.add(model_name)
            else:
                self._archived.discard(model_name)

    def get_record(self, run_id: str) -> dict | None:
        _records, by_id = self._snapshot()
        record = by_id.get(run_id)
        return record.model_dump(mode="json", by_alias=True) if record is not None else None

    def fetch_runs(
        self,
        *,
        model: str | None = None,
        eval_name: str | None = None,
        user: str | None = None,
        status: str | None = None,
        group: str | None = None,
        limit: int = DEFAULT_RUNS_LIMIT,
    ) -> list[dict]:
        records, _by_id = self._snapshot()
        rows = [record_to_row(record) for record in records]
        rows = [
            row
            for row in rows
            if (model is None or row["model_name"] == model)
            and (eval_name is None or row["eval_name"] == eval_name)
            and (user is None or row["user_name"] == user)
            and (status is None or row["status"] == status)
            and (group is None or row["group_id"] == group)
        ]
        rows.sort(key=lambda row: row["created_at"] or "", reverse=True)
        return rows[:limit]

    def panel(self, request: SelectionRequest, aggregate: MissingPolicy | None, include_archived: bool) -> dict:
        """The model x benchmark panel the request selects, over the snapshot.

        Archived models are dropped unless requested; when included, their rows carry
        ``archived: true`` so the UI can style them apart.
        """
        records, _by_id = self._snapshot()
        archived = self.archived_models()
        if not include_archived:
            records = [record for record in records if record.model.name not in archived]
        return build_panel(records, request, frozenset(archived), aggregate)

    def comparison(self, request: SelectionRequest, models: tuple[str, ...]) -> dict:
        """Head-to-head difference intervals between named models, over the snapshot.

        Archived models are always in scope: naming a model is an explicit request for it.
        """
        records, _by_id = self._snapshot()
        return build_comparison(records, request, models)

    def meta(self) -> dict:
        records, _by_id = self._snapshot()
        return build_meta(records, frozenset(self.archived_models()))

    def groups(
        self, *, model: str | None = None, user: str | None = None, limit: int = DEFAULT_RUNS_LIMIT
    ) -> list[dict]:
        """Runs collapsed into launches (one per ``group_id``), newest first.

        Each launch carries its model, version label, description, and a per-eval member list (with
        each member's headline score) so the runs view can show one row per launch and expand it to
        the individual evals it ran.
        """
        records, _by_id = self._snapshot()
        by_group: dict[str, list[EvalRunRecord]] = {}
        for record in records:
            if (model and record.model.name != model) or (user and record.user != user):
                continue
            by_group.setdefault(record.group_id, []).append(record)
        groups: list[dict] = []
        for group_id, members in by_group.items():
            ordered = sorted(members, key=lambda record: record.created_at or "")
            newest = ordered[-1]
            statuses = {record.status.value for record in members}
            groups.append(
                {
                    "group_id": group_id,
                    "model_name": newest.model.name,
                    "version": newest.version,
                    "description": newest.description,
                    "user_name": newest.user,
                    "accelerator": newest.hardware.accelerator,
                    "created_at": newest.created_at,
                    "status": _status_rollup(statuses),
                    "n_evals": len(members),
                    "n_succeeded": sum(1 for record in members if record.status.value == "succeeded"),
                    "evals": [_group_member(record) for record in ordered],
                }
            )
        groups.sort(key=lambda group: group["created_at"] or "", reverse=True)
        return groups[:limit]

    def history(self, model: str, task: str) -> list[dict]:
        """Every run's headline score for one ``(model, eval)`` over time, oldest first.

        ``task`` is a panel column, i.e. a registry eval name. One point per run that produced a
        primary metric, each carrying its interval, coverage, and provenance for the tooltip.
        """
        records, _by_id = self._snapshot()
        points = []
        for record in records:
            if record.model.name != model or record.evaluation.name != task:
                continue
            headline = record_headline(record)
            if headline is None:
                continue
            points.append({**headline, "status": record.status.value})
        points.sort(key=lambda point: point["created_at"] or "")
        return points

    def model_detail(self, model: str) -> dict | None:
        """One model's aggregated detail view: cohorts, per-eval history, and every run.

        ``None`` when the model has no records, so the route can answer 404.
        """
        records, _by_id = self._snapshot()
        return build_model_detail(records, model)

    def group_siblings(self, group_id: str, exclude_run_id: str) -> list[dict]:
        records, _by_id = self._snapshot()
        siblings = [
            _group_sibling_row(record)
            for record in records
            if record.group_id == group_id and record.run_id != exclude_run_id
        ]
        siblings.sort(key=lambda sibling: sibling["created_at"] or "", reverse=True)
        return siblings


class MemoryRecordStore(RecordStore):
    """Serves every view from the object-store record snapshot, with no database.

    Used for local development and offline runs (``EVALDASH_STORE=local``): records listed from
    ``RECORDS_PREFIXES`` fill the snapshot, archive state lives in memory, and there is no Postgres
    index. The base class already implements every read from the snapshot, so this is the base
    behaviour under an explicit, intentional name.
    """

    backend = "memory"


class PgRecordStore(RecordStore):
    """Boots and serves from a committed PostgreSQL catalog generation."""

    backend = "postgres"

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self._engine = engine
        # The kernel owns the connection, so all this store can name is where the engine points:
        # a host and database for a URL engine, nothing for one built on the Cloud SQL connector.
        self._instance = engine.url.host
        self._database = engine.url.database
        snapshot = fetch_snapshot(engine)
        self._catalog_generation = snapshot.generation
        self._snapshot_updated_at = snapshot.updated_at
        self._catalog_error: str | None = None
        self._set_snapshot(snapshot.records)

    def store_info(self) -> StoreInfo:
        with self._lock:
            return StoreInfo(
                backend=self.backend,
                instance=self._instance,
                database=self._database,
                record_count=len(self._records),
                catalog_generation=self._catalog_generation,
                snapshot_updated_at=self._snapshot_updated_at.isoformat(),
                catalog_error=self._catalog_error,
            )

    def reload_if_changed(self) -> bool:
        """Load a newer committed generation, returning whether the snapshot advanced."""
        generation = catalog_generation(self._engine)
        with self._lock:
            current_generation = self._catalog_generation
        if generation == current_generation:
            return False
        snapshot = fetch_snapshot(self._engine)
        with self._lock:
            self._records = snapshot.records
            self._by_id = {record.run_id: record for record in snapshot.records}
            self._catalog_generation = snapshot.generation
            self._snapshot_updated_at = snapshot.updated_at
        logger.info("postgres store loaded generation %d with %d records", snapshot.generation, len(snapshot.records))
        return True

    def set_catalog_error(self, error: str | None) -> None:
        with self._lock:
            self._catalog_error = error

    def configure_prefixes(self, prefixes: tuple[str, ...]) -> None:
        configure_prefixes(self._engine, prefixes)
        self.reload_if_changed()

    def source_states(self, prefix: str) -> dict[str, SourceState]:
        return source_states(self._engine, prefix)

    def reconcile_prefix(
        self,
        prefix: str,
        paths: list[str],
        observations: list[RecordObservation],
        probe_at: datetime,
        confirm_missing_after: float,
    ) -> None:
        reconcile_prefix(
            self._engine,
            prefix,
            paths,
            observations,
            probe_at,
            confirm_missing_after,
        )

    def mark_prefix_failed(self, prefix: str, probe_at: datetime, error: str) -> None:
        mark_prefix_failed(self._engine, prefix, probe_at, error)

    def finish_reconciliation(self, prefixes: tuple[str, ...]) -> None:
        prune_untracked_records(self._engine, prefixes)
        self.reload_if_changed()

    def prefix_statuses(self) -> list[PrefixStatus]:
        return prefix_statuses(self._engine)

    def archived_models(self) -> set[str]:
        return fetch_archived_models(self._engine)

    def set_model_archived(self, model_name: str, archived: bool, updated_by: str | None) -> None:
        set_model_archived(self._engine, model_name, archived, updated_by)


def create_store(services: Services, config: EvaldashConfig) -> RecordStore:
    """The store this process serves reads from.

    ``postgres`` reads the committed catalog generation in the app's own schema, which must
    already be migrated. ``local`` serves entirely from the object-store record snapshot with
    no database, for development and journeys against a ``RECORDS_PREFIXES`` directory.
    """
    if config.store is StoreMode.LOCAL:
        logger.info("%s=local: serving from the record snapshot, no database", STORE_ENV)
        return MemoryRecordStore()
    engine = services.engine()
    verify_schema(engine)
    store = PgRecordStore(engine)
    logger.info(
        "loaded the eval catalog: generation %s with %d records",
        store.store_info().catalog_generation,
        store.store_info().record_count,
    )
    return store


# --------------------------------------------------------------------------------------
# Background ingest
# --------------------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _run_periodically(
    operation: Callable[[], Awaitable[None]],
    interval: float,
    label: str,
    set_error: Callable[[str | None], None],
) -> None:
    while True:
        try:
            await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            set_error(error)
            logger.exception("%s failed; retrying in %ss", label, interval)
        else:
            set_error(None)
        await asyncio.sleep(interval)


@dataclass
class PrefixProbe:
    """Health of the most recent listing of one records prefix.

    ``error`` is None exactly when the last probe succeeded; ``last_success_time`` and
    ``record_count`` retain their last good values across a subsequent failing probe.
    """

    prefix: str
    last_probe_time: str | None = None
    last_success_time: str | None = None
    record_count: int | None = None
    error: str | None = None
    parse_failures: list[RecordParseFailure] = field(default_factory=list)
    """Records under this prefix that were found but failed to parse on the last successful listing --
    dropped from the snapshot and surfaced here rather than only logged. Empty when all parsed."""


class Ingestor:
    """Runs the periodic ingest and tracks per-prefix probe health.

    Each pass probes every prefix, then refreshes the store from the union of what was found. A
    prefix whose listing fails this pass contributes its last successfully-listed records instead
    of nothing, so a transient outage on one prefix (missing CW keys, a GCS blip) cannot make runs
    from that prefix disappear from the store's in-memory snapshot -- only a failure on the very
    first pass, before any prefix has ever listed successfully, leaves it empty. ``run_once`` holds
    ``_lock`` for the whole pass, so the background loop and a manual ``/api/refresh`` never ingest
    concurrently — whichever arrives second waits for the first to finish, then runs its own pass.
    """

    def __init__(self, store: RecordStore, prefixes: tuple[str, ...], interval: float) -> None:
        """Create an ingestor whose prefixes are ordered from highest to lowest precedence."""
        self._store = store
        self._prefixes = prefixes
        self.interval = interval
        self._lock = asyncio.Lock()
        self._probes = {prefix: PrefixProbe(prefix=prefix) for prefix in prefixes}
        self._last_good: dict[str, list[EvalRunRecord]] = {prefix: [] for prefix in prefixes}
        self._record_cache: dict[str, dict[str, EvalRunRecord]] = {prefix: {} for prefix in prefixes}
        self.last_pass_time: str | None = None
        self.cycle_error: str | None = None

    async def run_once(self) -> None:
        """Run one full ingest pass, serialised against any other pass via ``_lock``."""
        if not self._prefixes:
            # No roots to scan: leave the (externally populated) store untouched rather than
            # refreshing it to empty.
            return
        async with self._lock:
            records: list[EvalRunRecord] = []
            for prefix in self._prefixes:
                probe = self._probes[prefix]
                probe.last_probe_time = _utcnow_iso()
                try:
                    scan = await asyncio.to_thread(scan_records, prefix, self._record_cache[prefix])
                    found = list(scan.records)
                    failures = list(scan.failures)
                except Exception as exc:
                    # One unreachable store (missing CW keys, transient outage) must not hide the
                    # rest, and must not drop this prefix's previously-ingested runs from the
                    # snapshot -- carry its last-good listing forward instead.
                    probe.error = f"{type(exc).__name__}: {exc}"
                    logger.exception("ingest: listing %s failed; keeping last-good records this pass", prefix)
                    records.extend(self._last_good[prefix])
                    continue
                probe.last_success_time = probe.last_probe_time
                probe.record_count = len(found)
                probe.parse_failures = failures
                probe.error = None
                logger.info("ingest: %d records (%d unparseable) from %s", len(found), len(failures), prefix)
                self._last_good[prefix] = found
                self._record_cache[prefix] = scan.records_by_path
                records.extend(found)
            await asyncio.to_thread(self._store.refresh, records)
            self.last_pass_time = _utcnow_iso()

    async def run_loop(self) -> None:
        if not self._prefixes:
            return  # ingestion disabled; nothing to poll
        await _run_periodically(self.run_once, self.interval, "ingest cycle", self._set_cycle_error)

    def _set_cycle_error(self, error: str | None) -> None:
        self.cycle_error = error

    def status(self) -> dict:
        """Serialisable ingest health: cadence, last full pass, and each prefix's probe."""
        return {
            "interval_seconds": self.interval,
            "revalidate_after_seconds": None,
            "last_pass_time": self.last_pass_time,
            "cycle_error": self.cycle_error,
            "prefixes": [asdict(self._probes[prefix]) for prefix in self._prefixes],
        }


class PostgresIngestor:
    """Reconcile object membership and versions into PostgreSQL after serving has started."""

    def __init__(
        self,
        store: PgRecordStore,
        prefixes: tuple[str, ...],
        interval: float,
        revalidate_after: float,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._prefixes = prefixes
        self.interval = interval
        self.revalidate_after = revalidate_after
        self._now = now
        self._lock = asyncio.Lock()
        store.configure_prefixes(prefixes)
        self._probes = {prefix: PrefixProbe(prefix=prefix) for prefix in prefixes}
        for row in store.prefix_statuses():
            probe = self._probes.get(row.prefix)
            if probe is None:
                continue
            probe.last_probe_time = row.last_probe_at.isoformat() if row.last_probe_at else None
            probe.last_success_time = row.last_success_at.isoformat() if row.last_success_at else None
            probe.record_count = row.record_count
            probe.error = row.error
        for prefix, probe in self._probes.items():
            probe.parse_failures = [
                RecordParseFailure(path=path, error=state.error)
                for path, state in sorted(store.source_states(prefix).items())
                if state.error is not None
            ]
        self.last_pass_time: str | None = None
        self.cycle_error: str | None = None

    async def run_once(self) -> None:
        if not self._prefixes:
            return
        async with self._lock:
            for prefix in self._prefixes:
                probe = self._probes[prefix]
                probe_at = self._now()
                probe.last_probe_time = probe_at.isoformat()
                try:
                    paths = await asyncio.to_thread(list_record_paths, prefix)
                    states = await asyncio.to_thread(self._store.source_states, prefix)
                    observations = await asyncio.to_thread(
                        inspect_record_paths,
                        paths,
                        states,
                        VerificationSchedule(
                            checked_at=probe_at,
                            retry_after=self.interval,
                            revalidate_after=self.revalidate_after,
                        ),
                    )
                    failures = {
                        path: state.error for path, state in states.items() if path in paths and state.error is not None
                    }
                    for observation in observations:
                        if observation.error is None:
                            failures.pop(observation.path, None)
                        else:
                            failures[observation.path] = observation.error
                    await asyncio.to_thread(
                        self._store.reconcile_prefix,
                        prefix,
                        paths,
                        observations,
                        probe_at,
                        self.interval,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    probe.error = error
                    logger.exception("reconcile: %s failed; keeping its committed catalog rows", prefix)
                    await asyncio.to_thread(self._store.mark_prefix_failed, prefix, probe_at, error)
                    continue
                probe.last_success_time = probe.last_probe_time
                probe.record_count = len(paths)
                probe.parse_failures = [
                    RecordParseFailure(path=path, error=error) for path, error in sorted(failures.items())
                ]
                probe.error = None
                logger.info(
                    "reconcile: %d candidates, %d checked, %d invalid from %s",
                    len(paths),
                    len(observations),
                    len(probe.parse_failures),
                    prefix,
                )
            await asyncio.to_thread(self._store.finish_reconciliation, self._prefixes)
            self.last_pass_time = self._now().isoformat()

    async def run_loop(self) -> None:
        if not self._prefixes:
            return
        await _run_periodically(self.run_once, self.interval, "reconcile cycle", self._set_cycle_error)

    def _set_cycle_error(self, error: str | None) -> None:
        self.cycle_error = error

    def status(self) -> dict:
        return {
            "interval_seconds": self.interval,
            "revalidate_after_seconds": self.revalidate_after,
            "last_pass_time": self.last_pass_time,
            "cycle_error": self.cycle_error,
            "prefixes": [asdict(self._probes[prefix]) for prefix in self._prefixes],
        }


async def _reload_catalog_loop(store: PgRecordStore) -> None:
    """Poll for committed generations and expose any refresh failure through store status."""

    async def reload_once() -> None:
        await asyncio.to_thread(store.reload_if_changed)

    await _run_periodically(
        reload_once,
        CATALOG_POLL_SECONDS,
        "catalog generation poll",
        store.set_catalog_error,
    )


class ApiWithBackgroundLoops:
    """The API, with its background loops started on the first request it serves.

    Starlette delivers lifespan events only to the top-level application, and the kernel mounts this
    API under ``/evaldash/api/``; the first request is therefore the earliest moment inside the
    running event loop at which the ingest and catalog-reload loops can be started. They start once
    and then run for the life of the process, which ends with it -- there is no mounted-app shutdown
    event to cancel them on either.

    More than one instance may run these loops at once. Each reconciliation commits a prefix's
    source changes and their serving projection in one transaction that advances the catalog
    generation, so concurrent passes repeat work rather than corrupt state, and every instance picks
    up whichever generation committed last. What is not shared is per-process and therefore
    per-instance: the probe health ``/status`` reports, and the memory store's snapshot. Two
    instances can report different last-probe times for the same prefix.
    """

    def __init__(self, app: ASGIApp, loops: tuple[Callable[[], Awaitable[None]], ...]) -> None:
        self._app = app
        self._loops = loops
        self._tasks: list[asyncio.Task] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not self._tasks:
            self._tasks = [asyncio.create_task(loop()) for loop in self._loops]
        await self._app(scope, receive, send)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def _current_user() -> str | None:
    """The caller the kernel authenticated: an email address, ``anonymous`` on loopback, or None
    when the API is exercised outside the kernel's mount."""
    identity = get_verified_identity()
    return identity.user_id if identity is not None else None


class BadRequest(ValueError):
    """A query parameter the server will not guess at, surfaced to the caller as a 400."""


def _parse_limit(raw: str | None) -> int:
    return _parse_int(raw, default=DEFAULT_RUNS_LIMIT, low=1, high=MAX_RUNS_LIMIT)


def _parse_int(raw: str | None, *, default: int, low: int, high: int) -> int:
    """Parse a query-param int, clamped to ``[low, high]``; ``default`` on absent/unparseable."""
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(value, high))


def _parse_flag(raw: str | None) -> bool:
    return raw in ("1", "true")


def _parse_names(raw: str | None) -> tuple[str, ...] | None:
    """A comma-separated benchmark selection, or None for "every benchmark present"."""
    if not raw:
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return names or None


def _parse_coverage(raw: str | None) -> float:
    """The coverage floor a result must clear to be displayed."""
    if not raw:
        return DEFAULT_MIN_COVERAGE
    try:
        value = float(raw)
    except ValueError as exc:
        raise BadRequest(f"min_coverage must be a number in [0, 1], got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise BadRequest(f"min_coverage must be in [0, 1], got {value}")
    return value


def _parse_aggregate(raw: str | None) -> MissingPolicy | None:
    """The cross-benchmark aggregation policy, or None for no aggregate at all.

    Absent by default: a mean across benchmarks has no interpretation without a declared panel and
    missing-data policy, so a caller has to ask for one and say which policy it wants. An unrecognized
    policy is an error rather than "no aggregate": the two answer different questions, and silently
    substituting one for the other hides the typo.
    """
    if not raw:
        return None
    try:
        return MissingPolicy(raw)
    except ValueError as exc:
        policies = ", ".join(policy.value for policy in MissingPolicy)
        raise BadRequest(f"unknown aggregate policy {raw!r}; expected one of: {policies}") from exc


def _parse_review_n(raw: object) -> int:
    """Clamp the review sample count to ``[1, MAX_REVIEW_SAMPLES]``; the default on absent/unparseable."""
    if raw is None:
        return DEFAULT_REVIEW_SAMPLES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_REVIEW_SAMPLES
    return max(1, min(value, MAX_REVIEW_SAMPLES))


def _collect_job_status(gateway: ClusterGatewayLike, jobs: dict[str, str]) -> list[dict]:
    """Live iris job status for each pipeline role in a record's ``jobs`` map, order preserved."""
    return [{"role": role, "job_path": path, **gateway.job_status(path)} for role, path in jobs.items()]


class IngestorLike(Protocol):
    interval: float

    async def run_once(self) -> None: ...

    async def run_loop(self) -> None: ...

    def status(self) -> dict: ...


def _status_payload(store: RecordStore, ingestor: IngestorLike) -> dict:
    """The ``/api/status`` body: which store serves reads plus ingest/probe health."""
    return {"store": asdict(store.store_info()), "ingest": ingestor.status()}


def _status_rollup(statuses: set[str]) -> str:
    """Collapse a launch's per-eval statuses without inventing an evaluator failure."""
    if statuses == {"succeeded"}:
        return "succeeded"
    if "succeeded" not in statuses:
        if len(statuses) == 1:
            return next(iter(statuses))
        if "failed" in statuses:
            return "failed"
    return "mixed"


def _run_headline(record: dict) -> dict | None:
    """The run's overall grade for the detail header: its rolled-up primary metric with the interval
    and coverage behind it, or None when nothing scored (an infra or eval failure that never produced
    metrics)."""
    return record_headline(EvalRunRecord.model_validate(record))


def _group_member(record: EvalRunRecord) -> dict:
    """One eval within a launch: its identity, status, and headline score for the expanded group row."""
    return {
        "run_id": record.run_id,
        "eval_name": record.evaluation.name,
        "status": record.status.value,
        "created_at": record.created_at,
        "headline": record_headline(record),
    }


class ClusterGatewayLike(Protocol):
    """The live-status surface the run-detail endpoints call: real Iris/finelog, or a local no-op."""

    def job_status(self, job_path: str) -> dict: ...

    def fetch_logs(self, job_path: str, *, max_lines: int, substring: str | None) -> dict: ...


class NullClusterGateway:
    """Local-mode gateway: every live query degrades to unreachable, exactly as the real gateway does
    off-VPC, without any GCE discovery or RPC. Keeps run-detail working with no cluster access."""

    def job_status(self, job_path: str) -> dict:
        return {"reachable": False, "error": "local mode: cluster unavailable", "job": None, "tasks": []}

    def fetch_logs(self, job_path: str, *, max_lines: int, substring: str | None) -> dict:
        return {"reachable": False, "error": "local mode: cluster unavailable", "source": "", "entries": []}


def build_api(store: RecordStore, gateway: ClusterGatewayLike, config: EvaldashConfig) -> ApiWithBackgroundLoops:
    """Build the JSON API over a store, the cluster gateway, and the resolved configuration.

    ``config.prefixes`` are the record roots the background ingest scans; an empty tuple disables
    ingestion entirely (for a store populated out of band, as the tests do), which keeps the app from
    ever reaching the remote defaults.
    """
    ingestor: IngestorLike
    if isinstance(store, PgRecordStore):
        ingestor = PostgresIngestor(store, config.prefixes, config.ingest_interval, config.revalidate_after)
    else:
        ingestor = Ingestor(store, config.prefixes, config.ingest_interval)

    async def api_runs(request: Request) -> JSONResponse:
        params = request.query_params
        rows = await asyncio.to_thread(
            store.fetch_runs,
            model=params.get("model") or None,
            eval_name=params.get("eval") or None,
            user=params.get("user") or None,
            status=params.get("status") or None,
            group=params.get("group") or None,
            limit=_parse_limit(params.get("limit")),
        )
        return JSONResponse(rows)

    async def api_run_detail(request: Request) -> JSONResponse:
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        return JSONResponse({**record, "headline": _run_headline(record)})

    async def api_run_jobs(request: Request) -> JSONResponse:
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        roles = await asyncio.to_thread(_collect_job_status, gateway, record.get("jobs") or {})
        return JSONResponse({"roles": roles})

    async def api_run_logs(request: Request) -> JSONResponse:
        params = request.query_params
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        role = params.get("role")
        jobs = record.get("jobs") or {}
        if role not in jobs:
            return JSONResponse({"error": f"run has no {role!r} job"}, status_code=404)
        tail = _parse_int(params.get("tail"), default=DEFAULT_LOG_TAIL, low=1, high=MAX_LOG_TAIL)
        payload = await asyncio.to_thread(
            gateway.fetch_logs, jobs[role], max_lines=tail, substring=params.get("substring") or None
        )
        payload["role"] = role
        return JSONResponse(payload)

    async def api_run_samples_tasks(request: Request) -> JSONResponse:
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        payload = await asyncio.to_thread(samples.list_sample_tasks, record.get("results_path"))
        return JSONResponse(payload.model_dump(mode="json"))

    async def api_run_samples(request: Request) -> JSONResponse:
        params = request.query_params
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        task = params.get("task")
        if not task:
            return JSONResponse({"error": "task is required"}, status_code=400)
        payload = await asyncio.to_thread(
            samples.fetch_samples,
            record.get("results_path"),
            task,
            offset=_parse_int(params.get("offset"), default=0, low=0, high=10_000_000),
            limit=_parse_int(params.get("limit"), default=DEFAULT_SAMPLE_LIMIT, low=1, high=MAX_SAMPLE_LIMIT),
            correct=params.get("correct") or "all",
            extraction_filter=params.get("extraction_filter") or None,
        )
        return JSONResponse(payload.model_dump(mode="json"))

    async def api_run_samples_artifact(request: Request) -> JSONResponse:
        params = request.query_params
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        uri = params.get("uri")
        if not uri:
            return JSONResponse({"error": "uri is required"}, status_code=400)
        payload = await asyncio.to_thread(samples.fetch_artifact, record.get("results_path"), uri)
        return JSONResponse(payload.model_dump(mode="json"))

    async def api_run_samples_review(request: Request) -> JSONResponse:
        record = await asyncio.to_thread(store.get_record, request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        body = await request.json()
        task = body.get("task")
        if not task:
            return JSONResponse({"error": "task is required"}, status_code=400)
        sample_filter = body.get("filter", "all")
        if sample_filter not in REVIEW_FILTERS:
            return JSONResponse({"error": f"filter must be one of {REVIEW_FILTERS}"}, status_code=400)
        payload = await asyncio.to_thread(
            review.review_run_samples,
            record.get("results_path"),
            (record.get("model") or {}).get("name"),
            task,
            sample_filter,
            _parse_review_n(body.get("n")),
            config.review_model,
        )
        return JSONResponse(payload.model_dump(mode="json"))

    async def api_run_group(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        record = await asyncio.to_thread(store.get_record, run_id)
        if record is None:
            return JSONResponse({"error": "unknown run_id"}, status_code=404)
        group_id = record.get("group_id")
        siblings = await asyncio.to_thread(store.group_siblings, group_id, run_id) if group_id else []
        return JSONResponse({"group_id": group_id, "siblings": siblings})

    async def api_history(request: Request) -> JSONResponse:
        params = request.query_params
        model = params.get("model")
        task = params.get("task")
        if not model or not task:
            return JSONResponse({"error": "model and task are required"}, status_code=400)
        points = await asyncio.to_thread(store.history, model, task)
        return JSONResponse({"model": model, "task": task, "points": points})

    async def api_model_detail(request: Request) -> JSONResponse:
        detail = await asyncio.to_thread(store.model_detail, request.path_params["model_name"])
        if detail is None:
            return JSONResponse({"error": "unknown model"}, status_code=404)
        return JSONResponse(detail)

    def _selection(params: Mapping[str, str]) -> SelectionRequest:
        """The panel selection a query string asks for, shared by the panel and compare endpoints."""
        return panel_request(
            benchmarks=_parse_names(params.get("benchmarks")),
            cohort_version=params.get("cohort") or None,
            completeness=Completeness.COMPLETE_PANEL if _parse_flag(params.get("complete")) else Completeness.ANY,
            min_coverage=_parse_coverage(params.get("min_coverage")),
            filters={facet: value for facet in RUN_FACETS if (value := params.get(facet))},
            model_query=params.get("model") or None,
            include_flagged=_parse_flag(params.get("include_flagged")),
        )

    async def api_panel(request: Request) -> JSONResponse:
        params = request.query_params
        try:
            selection = _selection(params)
            aggregate = _parse_aggregate(params.get("aggregate"))
        except BadRequest as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        payload = await asyncio.to_thread(store.panel, selection, aggregate, _parse_flag(params.get("include_archived")))
        return JSONResponse(payload)

    async def api_compare(request: Request) -> JSONResponse:
        params = request.query_params
        models = _parse_names(params.get("models"))
        if models is None or len(models) < 2:
            return JSONResponse({"error": "compare needs at least two models"}, status_code=400)
        if len(models) > MAX_COMPARE_MODELS:
            return JSONResponse({"error": f"compare takes at most {MAX_COMPARE_MODELS} models"}, status_code=400)
        try:
            selection = _selection(params)
        except BadRequest as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        payload = await asyncio.to_thread(store.comparison, selection, models)
        return JSONResponse(payload)

    async def api_groups(request: Request) -> JSONResponse:
        params = request.query_params
        groups = await asyncio.to_thread(
            store.groups,
            model=params.get("model") or None,
            user=params.get("user") or None,
            limit=_parse_limit(params.get("limit")),
        )
        return JSONResponse(groups)

    async def api_model_archive(request: Request) -> JSONResponse:
        model_name = request.path_params["model_name"]
        body = await request.json()
        archived = bool(body.get("archived", True))
        await asyncio.to_thread(store.set_model_archived, model_name, archived, _current_user())
        return JSONResponse({"model_name": model_name, "archived": archived})

    async def api_meta(_request: Request) -> JSONResponse:
        meta = store.meta()
        meta["current_user"] = _current_user()
        meta["store"] = store.backend
        return JSONResponse(meta)

    async def api_status(_request: Request) -> JSONResponse:
        return JSONResponse(_status_payload(store, ingestor))

    async def api_refresh(_request: Request) -> JSONResponse:
        await ingestor.run_once()
        return JSONResponse(_status_payload(store, ingestor))

    routes = [
        Route("/runs", api_runs),
        Route("/groups", api_groups),
        Route("/models/{model_name:str}/archive", api_model_archive, methods=["POST"]),
        Route("/models/{model_name:str}", api_model_detail),
        Route("/runs/{run_id:str}/jobs", api_run_jobs),
        Route("/runs/{run_id:str}/logs", api_run_logs),
        Route("/runs/{run_id:str}/samples/tasks", api_run_samples_tasks),
        Route("/runs/{run_id:str}/samples/artifact", api_run_samples_artifact),
        Route("/runs/{run_id:str}/samples/review", api_run_samples_review, methods=["POST"]),
        Route("/runs/{run_id:str}/samples", api_run_samples),
        Route("/runs/{run_id:str}/group", api_run_group),
        Route("/runs/{run_id:str}", api_run_detail),
        Route("/panel", api_panel),
        Route("/compare", api_compare),
        Route("/history", api_history),
        Route("/meta", api_meta),
        Route("/status", api_status),
        Route("/refresh", api_refresh, methods=["POST"]),
    ]
    loops: tuple[Callable[[], Awaitable[None]], ...] = (ingestor.run_loop,)
    if isinstance(store, PgRecordStore):
        loops += (functools.partial(_reload_catalog_loop, store),)
    return ApiWithBackgroundLoops(Starlette(routes=routes), loops)


def create_api(services: Services) -> ASGIApp:
    """The kernel's entry point: the JSON API mounted at ``/evaldash/api/``.

    Configuration is resolved from the environment once, here. Nothing in this path writes to the
    database: ``marina migrate`` has already applied the schema when the deploy reaches this point.
    """
    config = EvaldashConfig.from_env(os.environ)
    gateway: ClusterGatewayLike
    if config.store is StoreMode.LOCAL:
        # Local mode reads records straight from RECORDS_PREFIXES and never reaches the cluster or
        # the CoreWeave object store, so skip the CW S3 credential setup and the live gateway.
        gateway = NullClusterGateway()
    else:
        # Production only: the live gateway pulls in the iris/finelog connect clients, which local
        # mode neither has nor needs. Import it lazily so local dev runs without those deps.
        from .cluster import ClusterGateway  # noqa: PLC0415

        configure_coreweave_s3()
        gateway = ClusterGateway()
    return build_api(create_store(services, config), gateway, config)


def migrate(engine: Engine) -> None:
    """Bring the app's schema up to this build, under an advisory lock. Run by ``marina migrate``."""
    migrate_schema(engine)
