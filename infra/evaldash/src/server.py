# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-results dashboard server (Starlette + uvicorn).

Serves a bundled Vue SPA plus a small JSON API over the eval run records. Records are
the canonical per-run JSON written to ``gs://marin-eval-metadata/runs/<run_id>/record.json``
and indexed into CloudSQL Postgres.

A background task ingests the records on startup and every ``EVALDASH_INGEST_INTERVAL`` seconds
(default 300). Reads are served through a ``RecordStore`` selected by ``EVALDASH_STORE``: the
production ``postgres`` store upserts each record into Cloud SQL and fails fast if no DB is
configured, while the ``local`` store serves entirely from the object-store record snapshot with
no database (for development against a ``RECORDS_PREFIXES`` directory). Both keep an in-memory
snapshot the matrix/meta/groups/history views read from, since ``results_db`` exposes no aggregate
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
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import samples
import sqlalchemy
import uvicorn
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    EvalRunRecord,
    list_records,
)
from metrics import build_matrix, build_meta, record_score
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
    for part in os.environ.get("RECORDS_PREFIXES", f"{DEFAULT_RECORDS_PREFIX},{CW_RECORDS_PREFIX}").split(",")
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


def record_to_row(record: EvalRunRecord) -> dict:
    """Flatten one record to the API run-row shape (ISO ``created_at``, task list, jobs map).

    Shared by both stores so the memory and Postgres backends return the same run rows.
    """
    return {
        "run_id": record.run_id,
        "group_id": record.group_id,
        "created_at": record.created_at,
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
    list, run detail, group siblings, matrix, meta, groups, history) and holds the archived-model
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
        self._set_snapshot(records)
        logger.info("memory store refreshed: %d records", len(records))

    def archived_models(self) -> set[str]:
        """Model names hidden from the headline matrix. In-memory in the base; a table in Postgres."""
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

    def matrix(self, include_archived: bool = False) -> dict:
        """The model x eval matrix over the snapshot. Archived models are dropped unless requested;
        when included, their rows carry ``archived: true`` so the UI can style them apart."""
        records, _by_id = self._snapshot()
        archived = self.archived_models()
        if not include_archived:
            records = [record for record in records if record.model.name not in archived]
        return build_matrix(records, frozenset(archived))

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

        ``task`` is a matrix column, i.e. a registry eval name. One point per run that produced a
        primary metric -- with its stderr, status, and provenance for the score-over-time tooltip.
        """
        records, _by_id = self._snapshot()
        points = []
        for record in records:
            if record.model.name != model or record.evaluation.name != task:
                continue
            score = record_score(record)
            if score is None:
                continue
            points.append(
                {
                    "run_id": record.run_id,
                    "created_at": record.created_at,
                    "value": score.value,
                    "stderr": score.stderr,
                    "metric": score.metric,
                    "status": record.status.value,
                    "git_sha": record.provenance.git_sha,
                }
            )
        points.sort(key=lambda point: point["created_at"] or "")
        return points

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
    prefix failed to list this cycle) still resolves. ``matrix``, ``meta``, ``groups``, and
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
        self._store = store
        self._prefixes = prefixes
        self.interval = interval
        self._lock = asyncio.Lock()
        self._probes = {prefix: PrefixProbe(prefix=prefix) for prefix in prefixes}
        self._last_good: dict[str, list[EvalRunRecord]] = {prefix: [] for prefix in prefixes}
        self.last_pass_time: str | None = None

    async def run_once(self) -> None:
        """Run one full ingest pass, serialised against any other pass via ``_lock``."""
        async with self._lock:
            records: list[EvalRunRecord] = []
            for prefix in self._prefixes:
                probe = self._probes[prefix]
                probe.last_probe_time = _utcnow_iso()
                try:
                    found = await asyncio.to_thread(list_records, prefix)
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
                probe.error = None
                logger.info("ingest: %d records from %s", len(found), prefix)
                self._last_good[prefix] = found
                records.extend(found)
            await asyncio.to_thread(self._store.refresh, records)
            self.last_pass_time = _utcnow_iso()

    async def run_loop(self) -> None:
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


def _collect_job_status(gateway: ClusterGatewayLike, jobs: dict[str, str]) -> list[dict]:
    """Live iris job status for each pipeline role in a record's ``jobs`` map, order preserved."""
    return [{"role": role, "job_path": path, **gateway.job_status(path)} for role, path in jobs.items()]


def _status_payload(store: RecordStore, ingestor: Ingestor) -> dict:
    """The ``/api/status`` body: which store serves reads plus ingest/probe health."""
    return {"store": asdict(store.store_info()), "ingest": ingestor.status()}


def _status_rollup(statuses: set[str]) -> str:
    """Collapse a launch's per-eval statuses into one: all-succeeded, a single shared failure, or mixed."""
    if statuses == {"succeeded"}:
        return "succeeded"
    if "succeeded" not in statuses:
        return next(iter(statuses)) if len(statuses) == 1 else "failed"
    return "mixed"


def _group_member(record: EvalRunRecord) -> dict:
    """One eval within a launch: its identity, status, and headline score for the expanded group row."""
    score = record_score(record)
    return {
        "run_id": record.run_id,
        "eval_name": record.evaluation.name,
        "status": record.status.value,
        "created_at": record.created_at,
        "value": score.value if score else None,
        "metric": score.metric if score else None,
        "stderr": score.stderr if score else None,
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


def create_app(store: RecordStore, dist: Path, gateway: ClusterGatewayLike) -> Starlette:
    """Build the Starlette app over a store, the built SPA directory, and the cluster gateway."""
    ingestor = Ingestor(store, RECORDS_PREFIXES, INGEST_INTERVAL_SECONDS)

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
        return JSONResponse(record)

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

    async def api_matrix(request: Request) -> JSONResponse:
        include_archived = request.query_params.get("include_archived") in ("1", "true")
        return JSONResponse(await asyncio.to_thread(store.matrix, include_archived))

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
        Route("/api/runs/{run_id:str}/jobs", api_run_jobs),
        Route("/api/runs/{run_id:str}/logs", api_run_logs),
        Route("/api/runs/{run_id:str}/samples/tasks", api_run_samples_tasks),
        Route("/api/runs/{run_id:str}/samples/artifact", api_run_samples_artifact),
        Route("/api/runs/{run_id:str}/samples", api_run_samples),
        Route("/api/runs/{run_id:str}/group", api_run_group),
        Route("/api/runs/{run_id:str}", api_run_detail),
        Route("/api/matrix", api_matrix),
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
