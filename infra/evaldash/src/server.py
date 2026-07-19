# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-results dashboard server (Starlette + uvicorn).

Serves a bundled Vue SPA plus a small JSON API over the eval run records. Records are
the canonical per-run JSON written to ``gs://marin-eval-metadata/runs/<run_id>/record.json``
and indexed into CloudSQL Postgres.

A background task ingests the GCS records on startup and every ``EVALDASH_INGEST_INTERVAL``
seconds (default 300), upserting each into Postgres. Reads are served through a
``RecordStore``, backed by Postgres: the service fails to start if no DB is configured. Each
ingest pass also refreshes an in-memory snapshot the matrix/meta/history views read from,
since ``results_db`` exposes no aggregate query for them; a prefix whose listing fails keeps
its last successfully-listed records in that snapshot rather than dropping out of it.

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

import asyncio
import contextlib
import logging
import os
import threading
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import samples
import sqlalchemy
import uvicorn
from cluster import ClusterGateway
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
    fetch_runs,
    resolve_db_config,
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
    """The store backing reads: always Postgres, plus its instance/database."""

    backend: str
    instance: str
    database: str


class RecordStore:
    """Serves the run list and run details from the indexed Postgres tables; upserts on refresh.

    ``get_record`` reads the durable ``record`` jsonb from Postgres -- the same table the run list
    is served from -- so a run indexed there but absent from the latest ingest snapshot (its source
    prefix failed to list this cycle) still resolves. ``matrix``, ``meta``, and ``history`` read an
    in-memory snapshot instead, since ``results_db`` exposes no aggregate query for them; the lock
    guards its swap against the ingest worker thread.
    """

    backend = "postgres"

    def __init__(self, engine: Engine, instance: str, database: str) -> None:
        self._engine = engine
        self._instance = instance
        self._database = database
        self._records: list[EvalRunRecord] = []
        self._by_id: dict[str, EvalRunRecord] = {}
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
        return StoreInfo(backend=self.backend, instance=self._instance, database=self._database)

    def get_record(self, run_id: str) -> dict | None:
        stmt = sqlalchemy.select(eval_runs.c.record).where(eval_runs.c.run_id == run_id)
        with self._engine.begin() as conn:
            row = conn.execute(stmt).first()
        return row[0] if row is not None else None

    def refresh(self, records: list[EvalRunRecord]) -> None:
        self._set_snapshot(records)
        for record in records:
            upsert_record(self._engine, record)
        logger.info("postgres store upserted %d records", len(records))

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

    def matrix(self) -> dict:
        records, _by_id = self._snapshot()
        return build_matrix(records)

    def meta(self) -> dict:
        records, _by_id = self._snapshot()
        return build_meta(records)

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
    """Connect to the configured eval DB and build its store; the DB is required to start.

    Raises if ``EVAL_DB_*`` resolves no password or the instance is unreachable -- the dashboard
    has no reads without Postgres, so it must fail fast at boot rather than serve degraded.
    """
    config = resolve_db_config()
    if config is None:
        raise RuntimeError("eval DB unavailable: set EVAL_DB_PASSWORD or grant access to EVAL_DB_PASSWORD_SECRET")
    engine = connect_engine(config.instance, config.db, config.user, config.password)
    ensure_schema(engine)
    logger.info("connected to eval DB %s/%s", config.instance, config.db)
    return RecordStore(engine, instance=config.instance, database=config.db)


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


def _collect_job_status(gateway: ClusterGateway, jobs: dict[str, str]) -> list[dict]:
    """Live iris job status for each pipeline role in a record's ``jobs`` map, order preserved."""
    return [{"role": role, "job_path": path, **gateway.job_status(path)} for role, path in jobs.items()]


def _status_payload(store: RecordStore, ingestor: Ingestor) -> dict:
    """The ``/api/status`` body: which store serves reads plus ingest/probe health."""
    return {"store": asdict(store.store_info()), "ingest": ingestor.status()}


def create_app(store: RecordStore, dist: Path, gateway: ClusterGateway) -> Starlette:
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

    async def api_matrix(_request: Request) -> JSONResponse:
        return JSONResponse(store.matrix())

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
        Route("/api/runs/{run_id:str}/jobs", api_run_jobs),
        Route("/api/runs/{run_id:str}/logs", api_run_logs),
        Route("/api/runs/{run_id:str}/samples/tasks", api_run_samples_tasks),
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
    configure_coreweave_s3()
    store = create_store()
    app = create_app(store, _dashboard_dist(), ClusterGateway())
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
