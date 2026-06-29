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
from concurrent.futures import ThreadPoolExecutor

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


class QueryStatus(enum.StrEnum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class QueryState:
    status: QueryStatus
    result: QueryResult | None = None
    error: str | None = None


class QueryManager:
    """Runs queries one at a time in a background thread and tracks their state.

    A single-worker executor serializes execution (one DuckDB query at a time, per
    design); ``submit`` returns immediately so the HTTP request never blocks on the
    query. State is kept in memory for the process lifetime — ducky is a stateless,
    restartable service, so a restart simply drops in-flight/finished query state.
    """

    def __init__(self, runner: QueryRunner) -> None:
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ducky-query")
        self._states: dict[str, QueryState] = {}
        self._lock = threading.Lock()

    def submit(self, sql: str) -> str:
        query_id = uuid.uuid4().hex
        with self._lock:
            self._states[query_id] = QueryState(QueryStatus.RUNNING)
        self._executor.submit(self._run, sql, query_id)
        return query_id

    def get(self, query_id: str) -> QueryState | None:
        with self._lock:
            return self._states.get(query_id)

    def _run(self, sql: str, query_id: str) -> None:
        try:
            state = QueryState(QueryStatus.DONE, result=self._runner.run_query(sql, query_id))
        except DuckyError as e:
            state = QueryState(QueryStatus.ERROR, error=str(e))
        except Exception as e:  # background task: record instead of hanging in RUNNING forever
            logger.exception("Unexpected error running query %s", query_id)
            state = QueryState(QueryStatus.ERROR, error=f"internal error: {e}")
        with self._lock:
            self._states[query_id] = state

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
    }


def _index_html(ttl_days: int) -> str:
    return f"""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ducky</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  textarea {{ width: 100%; height: 12rem; font-family: monospace; font-size: 13px; }}
  button {{ margin-top: .5rem; padding: .4rem 1rem; font-size: 14px; }}
  #status {{ margin: .5rem 0; color: #555; }}
  table {{ border-collapse: collapse; margin-top: 1rem; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: .2rem .5rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .error {{ color: #b00; white-space: pre-wrap; font-family: monospace; }}
</style>
</head>
<body>
<h1>🦆 ducky</h1>
<textarea id="sql" placeholder="SELECT * FROM read_parquet('gs://bucket/path/*.parquet') LIMIT 100"></textarea>
<div><button id="run">Run</button></div>
<div id="status"></div>
<div id="result"></div>
<script>
const TTL_DAYS = {ttl_days};
const POLL_MS = 1000;
const runBtn = document.getElementById("run");
const sqlEl = document.getElementById("sql");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

function showError(msg) {{
  statusEl.textContent = "";
  resultEl.innerHTML = '<div class="error"></div>';
  resultEl.firstChild.textContent = msg;
}}

async function run() {{
  const sql = sqlEl.value.trim();
  if (!sql) return;
  runBtn.disabled = true;
  statusEl.textContent = "Submitting…";
  resultEl.innerHTML = "";
  try {{
    const resp = await fetch("query", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ sql }}),
    }});
    const data = await resp.json();
    if (!resp.ok) {{ showError(data.error || ("HTTP " + resp.status)); return; }}
    poll(data.query_id);
  }} catch (e) {{
    showError(String(e));
    runBtn.disabled = false;
  }}
}}

async function poll(queryId) {{
  statusEl.textContent = "Running…";
  try {{
    const resp = await fetch("result/" + queryId);
    const data = await resp.json();
    if (!resp.ok) {{ showError(data.error || ("HTTP " + resp.status)); runBtn.disabled = false; return; }}
    if (data.status === "running") {{ setTimeout(() => poll(queryId), POLL_MS); return; }}
    if (data.status === "error") {{ showError(data.error); runBtn.disabled = false; return; }}
    renderTable(data);
  }} catch (e) {{
    showError(String(e));
  }} finally {{
    if (statusEl.textContent !== "Running…") runBtn.disabled = false;
  }}
}}

function renderTable(data) {{
  const shown = data.rows.length;
  let msg = shown + " row" + (shown === 1 ? "" : "s");
  if (data.truncated) {{
    msg = "showing " + shown + " of " + data.total_rows
        + " rows — full result at " + data.result_path + " (expires in " + TTL_DAYS + "d)";
  }}
  statusEl.textContent = msg;
  const thead = "<tr>" + data.columns.map(c => "<th></th>").join("") + "</tr>";
  const table = document.createElement("table");
  table.innerHTML = "<thead>" + thead + "</thead><tbody></tbody>";
  table.querySelectorAll("th").forEach((th, i) => th.textContent = data.columns[i]);
  const tbody = table.querySelector("tbody");
  for (const row of data.rows) {{
    const tr = document.createElement("tr");
    for (const cell of row) {{
      const td = document.createElement("td");
      td.textContent = cell === null ? "NULL" : String(cell);
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }}
  resultEl.innerHTML = "";
  resultEl.appendChild(table);
  runBtn.disabled = false;
}}

runBtn.addEventListener("click", run);
sqlEl.addEventListener("keydown", e => {{
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
}});
</script>
</body>
</html>
"""


def create_app(runner: QueryRunner, config: DuckyConfig) -> Starlette:
    """Build the ducky Starlette app over a query runner. No Iris context required."""
    index_html = _index_html(config.result_ttl_days)
    manager = QueryManager(runner)

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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = DuckyConfig.from_environment()
    runner = QueryRunner(config)
    app = create_app(runner, config)

    ctx = iris_ctx()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("No Iris job info available — ducky must run inside an Iris job")
    port = ctx.get_port(config.port_name)
    address = f"http://{job_info.advertise_host}:{port}"

    endpoint_id = ctx.registry.register(config.port_name, address, {"job_id": ctx.job_id.to_wire()})
    logger.info("ducky registered as %s at %s", config.port_name, address)

    async def _on_shutdown() -> None:
        ctx.registry.unregister(endpoint_id)
        app.state.query_manager.shutdown()
        logger.info("ducky endpoint unregistered")

    app.router.lifespan_context = on_shutdown(_on_shutdown)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
