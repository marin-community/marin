# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-results dashboard server (Starlette + uvicorn).

Serves a bundled Vue SPA plus a small JSON API over eval run records under the GCS or CoreWeave
``evals`` output root. It also scans the former flat ``eval-metadata/runs`` roots while older CLI
checkouts can still write there.

A background task ingests the records on startup and every ``EVALDASH_INGEST_INTERVAL`` seconds
(default 300). Reads are served through a ``RecordStore`` selected by ``EVALDASH_STORE``: the
production ``postgres`` store upserts each record into Cloud SQL and fails fast if no DB is
configured, while the ``local`` store serves entirely from the object-store record snapshot with
no database (for development against a ``RECORDS_PREFIXES`` directory). Both keep an in-memory
snapshot the panel/meta/groups/history views read from, since ``results_db`` exposes no aggregate
query for them; a prefix whose listing fails keeps its last successfully-listed records in that
snapshot rather than dropping out of it.

``/api/status`` reports each prefix's last-probe health, the active store, and the ingest
cadence; ``POST /api/refresh`` runs one ingest pass immediately, serialised with the loop.

Per-run drill-in endpoints read beyond the record: ``/api/runs/{id}/jobs`` and ``.../logs`` fetch
live iris job/attempt status and finelog log lines over Direct VPC egress, ``.../samples`` pages the
per-question parquet exports, and ``.../group`` plus ``/api/history`` serve a run's group siblings and
a model-by-task score-over-time series.

Cloud Run sits behind IAP, which is the only gate; there is no application auth. IAP stamps
the caller into ``X-Goog-Authenticated-User-Email`` (``accounts.google.com:<email>``), which
``/api/meta`` echoes as ``current_user``.

``records`` and ``samples`` are copied from ``lib/marin/src/marin/evaluation/``. Generated Iris
and finelog RPC packages are copied as directories; ``results_db`` lives beside this server under
``infra/evaldash/src``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import review
import samples
import sqlalchemy
import uvicorn
from marin.evaluation.eval_stats import DEFAULT_MIN_COVERAGE, Completeness, MissingPolicy, SelectionRequest
from marin.evaluation.records import (
    DEFAULT_SCAN_PREFIXES,
    EvalRunRecord,
    RecordParseFailure,
    scan_records,
)
from metrics import (
    RUN_FACETS,
    build_comparison,
    build_meta,
    build_model_detail,
    build_panel,
    panel_request,
    record_headline,
)
from results_db import (
    connect_engine,
    ensure_schema,
    eval_runs,
    fetch_archived_models,
    fetch_runs,
    resolve_db_config,
    set_model_archived,
    upsert_record,
)
from rigging.filesystem.s3_compat import configure_coreweave_s3
from sqlalchemy.engine import Engine
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

RECORDS_PREFIXES = tuple(
    part.strip()
    for part in os.environ.get(
        "RECORDS_PREFIXES",
        ",".join(DEFAULT_SCAN_PREFIXES),
    ).split(",")
    if part.strip()
)
INGEST_INTERVAL_SECONDS = int(os.environ.get("EVALDASH_INGEST_INTERVAL", "300"))
# Which record store backs reads: "postgres" (production, requires the eval DB) or "local"
# (development, serves entirely from the RECORDS_PREFIXES record snapshot with no database).
STORE_MODE = os.environ.get("EVALDASH_STORE", "postgres").strip().lower()
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

IAP_USER_HEADER = "x-goog-authenticated-user-email"
IAP_USER_PREFIX = "accounts.google.com:"


# --------------------------------------------------------------------------------------
# Record stores
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreInfo:
    """Which store serves reads and, for Postgres, the instance/database behind it."""

    backend: str
    instance: str | None
    database: str | None


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
    """In-memory snapshot of eval records plus the read views the API serves over it.

    The base serves every read from the snapshot the ingest loop swaps wholesale each cycle (run
    list, run detail, group siblings, panel, meta, groups, history) and holds the archived-model
    set in memory. :class:`MemoryRecordStore` uses these directly for local, offline runs;
    :class:`PgRecordStore` overrides the run list, run detail, group siblings, refresh, and archive
    state to read the durable Postgres index instead. The lock guards the snapshot swap against the
    ingest worker thread.
    """

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
        return StoreInfo(backend=self.backend, instance=None, database=None)

    def refresh(self, records: list[EvalRunRecord]) -> None:
        """Absorb a fresh record listing. The base only swaps the snapshot; Postgres also upserts."""
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
    """Serves the run list and run details from the indexed Postgres tables; upserts on refresh.

    ``get_record`` reads the durable ``record`` jsonb from Postgres -- the same table the run list
    is served from -- so a run indexed there but absent from the latest ingest snapshot (its source
    prefix failed to list this cycle) still resolves. ``panel``, ``meta``, ``groups``, and
    ``history`` inherit the base's snapshot reads, since ``results_db`` exposes no aggregate query.
    """

    backend = "postgres"

    def __init__(self, engine: Engine, instance: str, database: str) -> None:
        super().__init__()
        self._engine = engine
        self._instance = instance
        self._database = database

    def store_info(self) -> StoreInfo:
        return StoreInfo(backend=self.backend, instance=self._instance, database=self._database)

    def refresh(self, records: list[EvalRunRecord]) -> None:
        records = _deduplicate_records(records)
        self._set_snapshot(records)
        for record in records:
            upsert_record(self._engine, record)
        logger.info("postgres store upserted %d records", len(records))

    def archived_models(self) -> set[str]:
        return fetch_archived_models(self._engine)

    def set_model_archived(self, model_name: str, archived: bool, updated_by: str | None) -> None:
        set_model_archived(self._engine, model_name, archived, updated_by)

    def get_record(self, run_id: str) -> dict | None:
        stmt = sqlalchemy.select(eval_runs.c.record).where(eval_runs.c.run_id == run_id)
        with self._engine.begin() as conn:
            row = conn.execute(stmt).first()
        return row[0] if row is not None else None

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
        rows = fetch_runs(
            self._engine, model=model, eval_name=eval_name, user=user, status=status, group=group, limit=limit
        )
        # The task list and jobs map live in the record jsonb, so enrich each row from the cache.
        _records, by_id = self._snapshot()
        for row in rows:
            record = by_id.get(row.get("run_id"))
            row["tasks"] = [task.name for task in record.evaluation.tasks] if record else []
            row["jobs"] = dict(record.jobs) if record else {}
            # version lives only in the record jsonb, not an eval_runs column, so fill it from the cache.
            row["version"] = record.version if record else None
        return rows

    def group_siblings(self, group_id: str, exclude_run_id: str) -> list[dict]:
        stmt = (
            sqlalchemy.select(
                eval_runs.c.run_id,
                eval_runs.c.eval_name,
                eval_runs.c.model_name,
                eval_runs.c.status,
                eval_runs.c.created_at,
            )
            .where(eval_runs.c.group_id == group_id, eval_runs.c.run_id != exclude_run_id)
            .order_by(eval_runs.c.created_at.desc())
        )
        with self._engine.begin() as conn:
            rows = [dict(row) for row in conn.execute(stmt).mappings().all()]
        for row in rows:
            row["created_at"] = row["created_at"].isoformat()
        return rows


def create_store() -> RecordStore:
    """Pick the store at boot from ``EVALDASH_STORE`` (default ``postgres``).

    ``postgres`` requires a reachable eval DB and fails fast without one -- the production service
    has no reads without its index. ``local`` serves entirely from the object-store record snapshot
    with no database, for development against a ``RECORDS_PREFIXES`` directory.
    """
    if STORE_MODE == "local":
        logger.info("EVALDASH_STORE=local: serving from the record snapshot, no database")
        return MemoryRecordStore()
    if STORE_MODE != "postgres":
        raise RuntimeError(f"unknown EVALDASH_STORE={STORE_MODE!r}; expected 'postgres' or 'local'")
    config = resolve_db_config()
    if config is None:
        raise RuntimeError(
            "eval DB unavailable: set EVAL_DB_PASSWORD, grant access to EVAL_DB_PASSWORD_SECRET, "
            "or run with EVALDASH_STORE=local"
        )
    engine = connect_engine(config.instance, config.db, config.user, config.password)
    ensure_schema(engine)
    logger.info("connected to eval DB %s/%s", config.instance, config.db)
    return PgRecordStore(engine, instance=config.instance, database=config.db)


# --------------------------------------------------------------------------------------
# Background ingest
# --------------------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


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
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ingest cycle failed; retrying in %ss", self.interval)
            await asyncio.sleep(self.interval)

    def status(self) -> dict:
        """Serialisable ingest health: cadence, last full pass, and each prefix's probe."""
        return {
            "interval_seconds": self.interval,
            "last_pass_time": self.last_pass_time,
            "prefixes": [asdict(self._probes[prefix]) for prefix in self._prefixes],
        }


# --------------------------------------------------------------------------------------
# SPA serving
# --------------------------------------------------------------------------------------

_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>Marin Evals</title>"
    "<body style='font-family:system-ui;margin:3rem'><h1>Marin Evals</h1>"
    "<p>Dashboard not built — run "
    "<code>npm --prefix infra/evaldash/dashboard install &amp;&amp; "
    "npm --prefix infra/evaldash/dashboard run build</code>.</p>"
)


def _dashboard_dist() -> Path:
    """Locate the built SPA: env override, the image layout (beside this file), or the repo
    layout (``../dashboard/dist``)."""
    override = os.environ.get("EVALDASH_DASHBOARD_DIST")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    candidates = [here.parent / "dist", here.parents[1] / "dashboard" / "dist"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _index_html(dist: Path, forwarded_prefix: str) -> HTMLResponse:
    """Serve ``dist/index.html``, rewriting ``<base href="/">`` to any reverse-proxy prefix.

    The controller/proxy sets ``X-Forwarded-Prefix``; rewriting the base makes the SPA's
    relative asset and API URLs resolve under it. An empty prefix leaves the base at ``/``.
    """
    index_path = dist / "index.html"
    if not index_path.is_file():
        return HTMLResponse(_NOT_BUILT_HTML, status_code=503)
    html = index_path.read_text(encoding="utf-8")
    prefix = forwarded_prefix.rstrip("/")
    if prefix:
        html = html.replace('<base href="/"', f'<base href="{prefix}/"', 1)
    return HTMLResponse(html)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def _current_user(request: Request) -> str | None:
    """The IAP-stamped caller email, prefix stripped, or None outside IAP."""
    raw = request.headers.get(IAP_USER_HEADER)
    if not raw:
        return None
    return raw.removeprefix(IAP_USER_PREFIX)


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


def _status_payload(store: RecordStore, ingestor: Ingestor) -> dict:
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


def create_app(
    store: RecordStore,
    dist: Path,
    gateway: ClusterGatewayLike,
    prefixes: tuple[str, ...] = RECORDS_PREFIXES,
) -> Starlette:
    """Build the Starlette app over a store, the built SPA directory, and the cluster gateway.

    ``prefixes`` are the record roots the background ingestor scans; pass an empty tuple to disable
    ingestion entirely (for a store populated out of band, e.g. tests or a one-shot screenshot run),
    which keeps the app from ever reaching the remote defaults.
    """
    ingestor = Ingestor(store, prefixes, INGEST_INTERVAL_SECONDS)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        task = asyncio.create_task(ingestor.run_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "store": store.backend})

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
        await asyncio.to_thread(store.set_model_archived, model_name, archived, _current_user(request))
        return JSONResponse({"model_name": model_name, "archived": archived})

    async def api_meta(request: Request) -> JSONResponse:
        meta = store.meta()
        meta["current_user"] = _current_user(request)
        meta["store"] = store.backend
        return JSONResponse(meta)

    async def api_status(_request: Request) -> JSONResponse:
        return JSONResponse(_status_payload(store, ingestor))

    async def api_refresh(_request: Request) -> JSONResponse:
        await ingestor.run_once()
        return JSONResponse(_status_payload(store, ingestor))

    async def index(request: Request) -> HTMLResponse:
        return _index_html(dist, request.headers.get("x-forwarded-prefix", ""))

    routes = [
        Route("/healthz", healthz),
        Route("/api/runs", api_runs),
        Route("/api/groups", api_groups),
        Route("/api/models/{model_name:str}/archive", api_model_archive, methods=["POST"]),
        Route("/api/models/{model_name:str}", api_model_detail),
        Route("/api/runs/{run_id:str}/jobs", api_run_jobs),
        Route("/api/runs/{run_id:str}/logs", api_run_logs),
        Route("/api/runs/{run_id:str}/samples/tasks", api_run_samples_tasks),
        Route("/api/runs/{run_id:str}/samples/artifact", api_run_samples_artifact),
        Route("/api/runs/{run_id:str}/samples/review", api_run_samples_review, methods=["POST"]),
        Route("/api/runs/{run_id:str}/samples", api_run_samples),
        Route("/api/runs/{run_id:str}/group", api_run_group),
        Route("/api/runs/{run_id:str}", api_run_detail),
        Route("/api/panel", api_panel),
        Route("/api/compare", api_compare),
        Route("/api/history", api_history),
        Route("/api/meta", api_meta),
        Route("/api/status", api_status),
        Route("/api/refresh", api_refresh, methods=["POST"]),
        Mount("/static", StaticFiles(directory=dist / "static", check_dir=False), name="static"),
        # SPA catch-all: any other path serves index.html so client-side routing works on
        # deep links and refreshes. Registered last so it never shadows the API or /static.
        Route("/{full_path:path}", index),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    store = create_store()
    gateway: ClusterGatewayLike
    if STORE_MODE == "local":
        # Local mode reads records straight from RECORDS_PREFIXES and never reaches the cluster or
        # the CoreWeave object store, so skip the CW S3 credential setup and the live gateway.
        gateway = NullClusterGateway()
    else:
        # Production only: the live gateway pulls in the iris/finelog connect clients, which local
        # mode neither has nor needs. Import it lazily so local dev runs without those deps.
        from cluster import ClusterGateway  # noqa: PLC0415

        configure_coreweave_s3()
        gateway = ClusterGateway()
    app = create_app(store, _dashboard_dist(), gateway)
    port = int(os.environ.get("PORT", "8080"))
    # Cloud Run needs the container to listen on all interfaces; local dev binds loopback only.
    host = os.environ.get("EVALDASH_HOST", "127.0.0.1" if STORE_MODE == "local" else "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
