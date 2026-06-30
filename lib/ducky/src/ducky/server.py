# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ducky's Starlette dashboard: paste SQL, run it, see a capped result table.

Queries run **asynchronously**: ``POST /query`` returns a ``query_id`` immediately
and the SQL runs in a background single-worker executor (one DuckDB query at a
time); the page polls ``GET /result/{query_id}`` until it is done. This decouples a
long query from the Iris controller proxy's 30 s request timeout
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
import threading
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor

import uvicorn
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.dashboard_common import on_shutdown, public, requires_auth
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from ducky.config import DuckyConfig
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
    """Runs queries one at a time in a background thread and tracks their state.

    A single-worker executor serializes execution (one DuckDB query at a time, per
    design); ``submit`` returns immediately so the HTTP request never blocks on the
    query. Identical SQL is served from an in-memory result cache keyed on the exact
    query text — a cache hit reuses the prior spilled parquet and returns instantly
    with ``cached=True``. State and cache are process-local; ducky is stateless and
    restartable, so a restart drops both.
    """

    def __init__(self, runner: QueryRunner, executor: Executor | None = None) -> None:
        self._runner = runner
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="ducky-query")
        self._states: dict[str, QueryState] = {}
        self._cache: dict[str, QueryResult] = {}
        self._lock = threading.Lock()

    def submit(self, sql: str) -> str:
        query_id = uuid.uuid4().hex
        with self._lock:
            cached = self._cache.get(sql)
            if cached is not None:
                self._states[query_id] = QueryState(QueryStatus.DONE, result=cached, cached=True)
                logger.info(
                    "query %s cache hit (%d rows, %s): %s",
                    query_id,
                    cached.total_rows,
                    _human_bytes(cached.result_bytes),
                    _log_sql(sql),
                )
                return query_id
            self._states[query_id] = QueryState(QueryStatus.RUNNING)
        logger.info("query %s submitted: %s", query_id, _log_sql(sql))
        self._executor.submit(self._run, sql, query_id)
        return query_id

    def get(self, query_id: str) -> QueryState | None:
        with self._lock:
            return self._states.get(query_id)

    def _run(self, sql: str, query_id: str) -> None:
        try:
            result = self._runner.run_query(sql, query_id)
        except DuckyError as e:
            logger.warning("query %s failed: %s", query_id, str(e).splitlines()[0])
            with self._lock:
                self._states[query_id] = QueryState(QueryStatus.ERROR, error=str(e))
            return
        except Exception as e:  # background task: record instead of hanging in RUNNING forever
            logger.exception("query %s crashed", query_id)
            with self._lock:
                self._states[query_id] = QueryState(QueryStatus.ERROR, error=f"internal error: {e}")
            return
        logger.info(
            "query %s done: %d rows, %s, %d ms",
            query_id,
            result.total_rows,
            _human_bytes(result.result_bytes),
            result.elapsed_ms,
        )
        with self._lock:
            self._cache[sql] = result
            self._states[query_id] = QueryState(QueryStatus.DONE, result=result, cached=False)

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


# CodeMirror 5 (SQL mode) gives DuckDB-flavored SQL highlighting from a CDN — loaded
# by the browser, so no server-side egress. __TTL_DAYS__ is substituted per config.
_INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ducky</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/sql/sql.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/addon/display/placeholder.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; }
  .CodeMirror { border: 1px solid #ccc; height: 12rem; font-size: 13px; }
  .CodeMirror-placeholder { color: #aaa !important; font-style: italic; }
  button { margin-top: .5rem; padding: .4rem 1rem; font-size: 14px; }
  #status { margin: .6rem 0; color: #555; }
  #status .cached { color: #2a7; font-weight: 600; }
  #status .computed { color: #888; }
  #status .loc { font-family: monospace; }
  table { border-collapse: collapse; margin-top: 1rem; font-size: 13px; }
  th, td { border: 1px solid #ccc; padding: .2rem .5rem; text-align: left; }
  th { background: #f0f0f0; }
  .error { color: #b00; white-space: pre-wrap; font-family: monospace; }
</style>
</head>
<body>
<h1>🦆 ducky</h1>
<textarea id="sql" placeholder="SELECT * FROM read_parquet('gs://bucket/path/*.parquet') LIMIT 100"></textarea>
<div><button id="run">Run</button> <span style="color:#888;font-size:12px">(⌘/Ctrl-Enter)</span></div>
<div id="status"></div>
<div id="result"></div>
<script>
const TTL_DAYS = __TTL_DAYS__;
const POLL_MS = 1000;
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

const editor = CodeMirror.fromTextArea(document.getElementById("sql"), {
  mode: "text/x-sql",
  lineNumbers: true,
  lineWrapping: true,
  placeholder: "-- write DuckDB SQL, then \\u2318/Ctrl-Enter to run\\n"
    + "SELECT *\\nFROM read_parquet('gs://marin-us-east5/<path>/*.parquet')\\nLIMIT 100",
  extraKeys: { "Cmd-Enter": run, "Ctrl-Enter": run },
});

function showError(msg) {
  statusEl.textContent = "";
  resultEl.innerHTML = '<div class="error"></div>';
  resultEl.firstChild.textContent = msg;
}

async function run() {
  const sql = editor.getValue().trim();
  if (!sql) return;
  runBtn.disabled = true;
  statusEl.textContent = "Submitting…";
  resultEl.innerHTML = "";
  try {
    const resp = await fetch("query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql }),
    });
    const data = await resp.json();
    if (!resp.ok) { showError(data.error || ("HTTP " + resp.status)); runBtn.disabled = false; return; }
    poll(data.query_id);
  } catch (e) {
    showError(String(e));
    runBtn.disabled = false;
  }
}

async function poll(queryId) {
  statusEl.textContent = "Running…";
  try {
    const resp = await fetch("result/" + queryId);
    const data = await resp.json();
    if (!resp.ok) { showError(data.error || ("HTTP " + resp.status)); runBtn.disabled = false; return; }
    if (data.status === "running") { setTimeout(() => poll(queryId), POLL_MS); return; }
    if (data.status === "error") { showError(data.error); runBtn.disabled = false; return; }
    renderResult(data);
  } catch (e) {
    showError(String(e));
    runBtn.disabled = false;
  }
}

function fmtBytes(n) {
  if (n == null) return "?";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n : n.toFixed(1)) + " " + u[i];
}
function fmtDuration(ms) {
  if (ms == null) return "?";
  return ms < 1000 ? ms + " ms" : (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + " s";
}

function renderResult(data) {
  const shown = data.rows.length;
  const rowsLabel = data.truncated
    ? "showing " + shown + " of " + data.total_rows + " rows"
    : shown + " row" + (shown === 1 ? "" : "s");
  const cacheBadge = data.cached
    ? '<span class="cached">cached ✓</span>'
    : '<span class="computed">computed</span>';
  statusEl.innerHTML = rowsLabel
    + " · " + fmtDuration(data.elapsed_ms) + " · " + fmtBytes(data.result_bytes) + " result"
    + " · " + cacheBadge
    + ' · output: <span class="loc"></span> <span style="color:#888">(expires in ' + TTL_DAYS + "d)</span>";
  statusEl.querySelector(".loc").textContent = data.result_path;

  const table = document.createElement("table");
  table.innerHTML = "<thead><tr>" + data.columns.map(() => "<th></th>").join("") + "</tr></thead><tbody></tbody>";
  table.querySelectorAll("th").forEach((th, i) => th.textContent = data.columns[i]);
  const tbody = table.querySelector("tbody");
  for (const row of data.rows) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell === null ? "NULL" : String(cell);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  resultEl.innerHTML = "";
  resultEl.appendChild(table);
  runBtn.disabled = false;
}

runBtn.addEventListener("click", run);
</script>
</body>
</html>
"""


def _index_html(ttl_days: int) -> str:
    return _INDEX_HTML.replace("__TTL_DAYS__", str(ttl_days))


def create_app(runner: QueryRunner, config: DuckyConfig, executor: Executor | None = None) -> Starlette:
    """Build the ducky Starlette app over a query runner. No Iris context required.

    ``executor`` overrides the query executor (tests inject a synchronous one).
    """
    index_html = _index_html(config.result_ttl_days)
    manager = QueryManager(runner, executor=executor)

    @requires_auth
    async def index(_request: Request) -> HTMLResponse:
        return HTMLResponse(index_html)

    @requires_auth
    async def query(request: Request) -> JSONResponse:
        body = await request.json()
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return JSONResponse({"error": "missing 'sql'"}, status_code=400)
        return JSONResponse({"query_id": manager.submit(sql)}, status_code=202)

    @requires_auth
    async def result(request: Request) -> JSONResponse:
        state = manager.get(request.path_params["query_id"])
        if state is None:
            return JSONResponse({"error": "unknown query_id"}, status_code=404)
        return JSONResponse(_result_payload(state))

    @public
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/query", query, methods=["POST"]),
            Route("/result/{query_id:str}", result),
            Route("/health", health),
        ]
    )
    app.state.query_manager = manager
    return app


class _QuietPolls(logging.Filter):
    """Drop the high-frequency dashboard-poll access lines so query lifecycle logs stand out."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return '"GET /result/' not in message and '"GET /health ' not in message


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("uvicorn.access").addFilter(_QuietPolls())
    config = DuckyConfig.from_environment()
    runner = QueryRunner(config)
    app = create_app(runner, config)

    ctx = iris_ctx()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("No Iris job info available — ducky must run inside an Iris job")
    port = ctx.get_port(config.port_name)
    address = f"http://{job_info.advertise_host}:{port}"

    endpoint_id = ctx.registry.register(config.endpoint_name, address, {"job_id": ctx.job_id.to_wire()})
    logger.info("ducky registered as %s at %s", config.endpoint_name, address)

    async def _on_shutdown() -> None:
        ctx.registry.unregister(endpoint_id)
        app.state.query_manager.shutdown()
        logger.info("ducky endpoint unregistered")

    app.router.lifespan_context = on_shutdown(_on_shutdown)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
