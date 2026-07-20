# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ducky's Starlette dashboard: paste SQL, run it, see a capped result table.

Queries run **asynchronously**: ``POST /query`` returns a ``query_id`` immediately
and the SQL runs in a background thread pool (up to ``max_concurrent_queries``, each on
its own DuckDB cursor); the page polls ``GET /result/{query_id}`` until it is done. This
decouples a long query from the Iris controller proxy's 30 s request timeout
(``endpoint_proxy.PROXY_TIMEOUT_SECONDS``) — each HTTP call returns in well under
30 s while the query itself may run for minutes.

The page talks plain JSON over relative URLs (so it works behind the controller's
``/proxy/ducky/`` prefix). ``main()`` wires the app to an Iris named port and
registers it with the endpoint registry so the controller can route to it.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import multiprocessing
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import uvicorn
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.dashboard_common import on_shutdown
from rigging.server_auth import public, requires_auth
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ducky.catalog import Catalog, build_catalog
from ducky.config import ENDPOINT_NAME, HEALTH_PATH, PORT_NAME, DuckyConfig
from ducky.query_log import QueryLog, QueryLogRow, now_utc
from ducky.runner import DuckyError, QueryResult, QueryRunner

logger = logging.getLogger(__name__)


def _log_sql(sql: str, limit: int = 300) -> str:
    """Collapse SQL to a single truncated line for logging."""
    one_line = " ".join(sql.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


class QueryStatus(enum.StrEnum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class QueryState:
    status: QueryStatus
    result: QueryResult | None = None
    error: str | None = None
    cached: bool = False


class QueryManager:
    """Runs queries in a background thread pool and tracks their state.

    Up to ``max_workers`` queries run concurrently; ``submit`` returns immediately so
    the HTTP request never blocks on the query. Identical SQL is served from an
    in-memory result cache keyed on the exact query text — a cache hit reuses the
    prior spilled parquet and returns instantly with ``cached=True``. State and cache
    are process-local; ducky is stateless and restartable, so a restart drops both.
    """

    def __init__(
        self,
        runner: QueryRunner,
        executor: Executor | None = None,
        max_workers: int = 8,
        max_retained_states: int = 1024,
        query_log: QueryLog | None = None,
    ) -> None:
        self._runner = runner
        # Optional finelog sink: every submitted query (done, error, or cache hit) is
        # recorded to the `ducky.query` namespace for later analysis. None disables it
        # (tests, and any deploy without an in-cluster finelog endpoint).
        self._query_log = query_log
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ducky-query")
        # Bounded LRU of recent query states: an always-on service would otherwise grow the heap
        # unbounded, since each result retains up to preview_row_cap rows. Oldest entries are
        # evicted; an evicted query_id just 404s on /result (results also live on GCS). Result
        # *reuse* across identical SQL is served by the runner's scratch-bucket cache, not here.
        self._states: OrderedDict[str, QueryState] = OrderedDict()
        self._max_retained_states = max_retained_states
        self._lock = threading.Lock()

    def _set_state(self, query_id: str, state: QueryState) -> None:
        """Record a query's state (most-recent-last), evicting the oldest past the cap. Under the lock."""
        self._states[query_id] = state
        self._states.move_to_end(query_id)
        while len(self._states) > self._max_retained_states:
            self._states.popitem(last=False)

    def submit(self, sql: str, use_cache: bool = True) -> str:
        """Submit ``sql`` and return a query_id. With ``use_cache`` (default), a result cached in
        the scratch bucket by an earlier identical query is reused instead of re-running; pass
        ``use_cache=False`` to force a fresh run (e.g. when the underlying data changed) — it still
        refreshes the cache. The cache lookup runs on the worker, so submit always returns quickly."""
        query_id = uuid.uuid4().hex
        with self._lock:
            self._set_state(query_id, QueryState(QueryStatus.RUNNING))
        logger.info("query %s submitted: %s", query_id, _log_sql(sql))
        self._executor.submit(self._run, sql, query_id, use_cache)
        return query_id

    def get(self, query_id: str) -> QueryState | None:
        with self._lock:
            return self._states.get(query_id)

    def _run(self, sql: str, query_id: str, use_cache: bool = True) -> None:
        try:
            # Reuse a prior identical query's result from the scratch-bucket cache (which survives
            # the restarts an in-process cache wouldn't) instead of re-scanning the source. A hit
            # here is still a "cached" reply; a miss falls through to a fresh run.
            result = self._runner.lookup_persistent(sql) if use_cache else None
            from_cache = result is not None
            if result is None:
                result = self._runner.run_query(sql, query_id)
        except DuckyError as e:
            logger.warning("query %s failed: %s", query_id, str(e).splitlines()[0])
            error_state = QueryState(QueryStatus.ERROR, error=str(e))
            with self._lock:
                self._set_state(query_id, error_state)
            self._record(query_id, sql, error_state)
            return
        except Exception as e:  # background task: record instead of hanging in RUNNING forever
            logger.exception("query %s crashed", query_id)
            error_state = QueryState(QueryStatus.ERROR, error=f"internal error: {e}")
            with self._lock:
                self._set_state(query_id, error_state)
            self._record(query_id, sql, error_state)
            return
        logger.info(
            "query %s %s: %d rows, %s, %d ms",
            query_id,
            "persistent-cache hit" if from_cache else "done",
            result.total_rows,
            _human_bytes(result.result_bytes),
            result.elapsed_ms,
        )
        done_state = QueryState(QueryStatus.DONE, result=result, cached=from_cache)
        with self._lock:
            self._set_state(query_id, done_state)
        self._record(query_id, sql, done_state)

    def _record(self, query_id: str, sql: str, state: QueryState) -> None:
        """Persist one terminal query state to the finelog query log (best-effort no-op if disabled)."""
        if self._query_log is None:
            return
        result = state.result
        self._query_log.record(
            QueryLogRow(
                ts=now_utc(),
                query_id=query_id,
                sql=sql,
                status=state.status.value,
                cached=state.cached,
                elapsed_ms=result.elapsed_ms if result is not None else None,
                total_rows=result.total_rows if result is not None else None,
                result_bytes=result.result_bytes if result is not None else None,
                result_path=result.result_path if result is not None else None,
                error=state.error,
            )
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


def _result_payload(state: QueryState) -> dict:
    if state.status is QueryStatus.RUNNING:
        return {"status": QueryStatus.RUNNING.value}
    if state.status is QueryStatus.ERROR:
        return {"status": QueryStatus.ERROR.value, "error": state.error}
    result = state.result
    assert result is not None  # DONE always carries a result
    return {
        "status": QueryStatus.DONE.value,
        "columns": result.columns,
        "rows": result.preview_rows,
        "total_rows": result.total_rows,
        "truncated": result.truncated,
        "result_path": result.result_path,
        "cached": state.cached,
        "elapsed_ms": result.elapsed_ms,
        "result_bytes": result.result_bytes,
    }


# The dashboard is a bundled Vue SPA built into dashboard/dist by `npm run build`
# (gitignored; shipped in the Iris bundle via GENERATED_ARTIFACT_GLOBS). Resolve its
# dist dir: env override → the in-repo build output next to this package.
def _dashboard_dist() -> Path:
    override = os.environ.get("DUCKY_DASHBOARD_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "dashboard" / "dist"


_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>ducky</title>"
    "<body style='font-family:system-ui;margin:3rem'><h1>🦆 ducky</h1>"
    "<p>Dashboard not built — run "
    "<code>npm --prefix lib/ducky/dashboard install &amp;&amp; npm --prefix lib/ducky/dashboard run build</code>.</p>"
)


def _index_html(dist: Path, forwarded_prefix: str) -> HTMLResponse:
    """Serve dist/index.html, rewriting ``<base href="/">`` to the proxy sub-path.

    The controller proxy sets ``X-Forwarded-Prefix`` (e.g. ``/proxy/ducky``) in
    path-style mode; rewriting the base makes the SPA's relative asset and API URLs
    resolve under it. Empty prefix (subdomain/direct) leaves the base at ``/``.
    """
    index_path = dist / "index.html"
    if not index_path.is_file():
        return HTMLResponse(_NOT_BUILT_HTML, status_code=503)
    html = index_path.read_text(encoding="utf-8")
    prefix = forwarded_prefix.rstrip("/")
    if prefix:
        html = html.replace('<base href="/"', f'<base href="{prefix}/"', 1)
    return HTMLResponse(html)


def _catalog_payload(catalog: Catalog) -> dict:
    """Serialize the pre-baked catalog for the dashboard's sources/examples panel.

    ``insert_sql`` is the ready-to-run starter a click drops into the editor: a capped
    SELECT over the view. The dashboard treats views and examples uniformly as SQL snippets.
    """
    return {
        "views": [
            {
                "schema": view.schema,
                "name": view.name,
                "qualified_name": view.qualified_name,
                "description": view.description,
                "insert_sql": f"SELECT * FROM {view.qualified_name} LIMIT 100",
            }
            for view in catalog.views
        ],
        "examples": [
            {"title": example.title, "description": example.description, "sql": example.sql}
            for example in catalog.examples
        ],
    }


def create_app(
    runner: QueryRunner,
    config: DuckyConfig,
    executor: Executor | None = None,
    query_log: QueryLog | None = None,
) -> Starlette:
    """Build the ducky Starlette app over a query runner. No Iris context required.

    ``executor`` overrides the query executor (tests inject a synchronous one).
    ``query_log``, when given, records every submitted query to finelog.
    """
    dist = _dashboard_dist()
    # Advertise only the views the runner actually created — an absent dataset or a root
    # outside the allowlist is skipped, and the dashboard shouldn't offer a view that errors.
    catalog = build_catalog(config, available=set(runner.created_view_names))
    manager = QueryManager(
        runner,
        executor=executor,
        max_workers=config.max_concurrent_queries,
        query_log=query_log,
    )

    @requires_auth
    async def index(request: Request) -> HTMLResponse:
        return _index_html(dist, request.headers.get("x-forwarded-prefix", ""))

    @requires_auth
    async def query(request: Request) -> JSONResponse:
        body = await request.json()
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return JSONResponse({"error": "missing 'sql'"}, status_code=400)
        use_cache = body.get("use_cache", True)
        return JSONResponse({"query_id": manager.submit(sql, use_cache=bool(use_cache))}, status_code=202)

    @requires_auth
    async def result(request: Request) -> JSONResponse:
        state = manager.get(request.path_params["query_id"])
        if state is None:
            return JSONResponse({"error": "unknown query_id"}, status_code=404)
        return JSONResponse(_result_payload(state))

    @requires_auth
    async def api_config(_request: Request) -> JSONResponse:
        return JSONResponse({"result_ttl_days": config.result_ttl_days})

    @requires_auth
    async def api_catalog(_request: Request) -> JSONResponse:
        return JSONResponse(_catalog_payload(catalog))

    @public
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    routes = [
        Route("/", index),
        Route("/query", query, methods=["POST"]),
        Route("/result/{query_id:str}", result),
        Route("/api/config", api_config),
        Route("/api/catalog", api_catalog),
        Route(HEALTH_PATH, health),
    ]
    # check_dir=False: the app still boots (index shows a "not built" page) when the
    # SPA hasn't been built yet, instead of raising at startup.
    routes.append(Mount("/static", StaticFiles(directory=dist / "static", check_dir=False), name="static"))
    app = Starlette(routes=routes)
    app.state.query_manager = manager
    return app


class _QuietPolls(logging.Filter):
    """Drop the high-frequency dashboard-poll access lines so query lifecycle logs stand out."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return '"GET /result/' not in message and '"GET /health ' not in message


_RESTART_DELAY = 3  # base seconds between supervised server restarts
_RESTART_DELAY_MAX = 300  # cap on the backoff for a crash-looping server
_HEALTHY_RUNTIME = 60  # a child that ran at least this long is "healthy" → reset the backoff


def _serve() -> None:
    """Serve ducky in this process. Runs in a supervised child (see `main`)."""
    logging.getLogger("uvicorn.access").addFilter(_QuietPolls())
    config = DuckyConfig.from_environment()
    runner = QueryRunner(config)

    ctx = iris_ctx()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("No Iris job info available — ducky must run inside an Iris job")

    # Persist submitted queries to finelog (best-effort). Needs an in-cluster iris client to
    # resolve the log server; if absent (e.g. a standalone smoke), skip the sink rather than fail.
    query_log = QueryLog.connect(ctx) if ctx.client is not None else None
    if query_log is None:
        logger.warning("no in-cluster iris client — query logging to finelog disabled")
    app = create_app(runner, config, query_log=query_log)

    port = ctx.get_port(PORT_NAME)
    address = f"http://{job_info.advertise_host}:{port}"

    endpoint_id = ctx.registry.register(ENDPOINT_NAME, address, {"job_id": ctx.job_id.to_wire()})
    logger.info("ducky registered as %s at %s", ENDPOINT_NAME, address)

    async def _on_shutdown() -> None:
        ctx.registry.unregister(endpoint_id)
        app.state.query_manager.shutdown()
        if query_log is not None:
            query_log.close()
        logger.info("ducky endpoint unregistered")

    app.router.lifespan_context = on_shutdown(_on_shutdown)
    uvicorn.run(app, host="0.0.0.0", port=port)


def main() -> None:
    """Supervise the server: run it in a child process and restart it if it exits.

    A cgroup OOM-kill reaps the largest process — the server child — while this tiny
    supervisor survives, so ducky restarts in-process without consuming an Iris job
    retry. If the whole cgroup is killed the Iris task retry is the backstop.

    Restarts back off exponentially (capped) so a server that crash-loops on a permanent
    fault (e.g. bad config) doesn't hot-loop; the backoff resets once a child stays up.
    """
    logging.basicConfig(level=logging.INFO)
    delay = _RESTART_DELAY
    while True:
        started = time.monotonic()
        server = multiprocessing.Process(target=_serve, name="ducky-server")
        server.start()
        server.join()
        delay = _RESTART_DELAY if time.monotonic() - started >= _HEALTHY_RUNTIME else min(delay * 2, _RESTART_DELAY_MAX)
        logger.error("ducky server exited (exitcode=%s) — restarting in %ds", server.exitcode, delay)
        time.sleep(delay)


if __name__ == "__main__":
    main()
