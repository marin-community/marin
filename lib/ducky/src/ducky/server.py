# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ducky's Starlette dashboard: paste SQL, run it, see a capped result table.

The page talks plain JSON to ``POST query`` (relative URL, so it works behind the
controller's ``/proxy/ducky/`` prefix). ``main()`` wires the app to an Iris named
port and registers it with the endpoint registry so the controller can route to it.
"""

from __future__ import annotations

import logging
import uuid

import uvicorn
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.dashboard_common import on_shutdown, public, requires_auth
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from ducky.config import DuckyConfig
from ducky.runner import DuckyError, QueryRunner

logger = logging.getLogger(__name__)


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
const runBtn = document.getElementById("run");
const sqlEl = document.getElementById("sql");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

async function run() {{
  const sql = sqlEl.value.trim();
  if (!sql) return;
  runBtn.disabled = true;
  statusEl.textContent = "Running…";
  resultEl.innerHTML = "";
  try {{
    const resp = await fetch("query", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ sql }}),
    }});
    const data = await resp.json();
    if (!resp.ok) {{
      statusEl.textContent = "";
      resultEl.innerHTML = '<div class="error"></div>';
      resultEl.firstChild.textContent = data.error || ("HTTP " + resp.status);
      return;
    }}
    renderTable(data);
  }} catch (e) {{
    statusEl.textContent = "";
    resultEl.innerHTML = '<div class="error"></div>';
    resultEl.firstChild.textContent = String(e);
  }} finally {{
    runBtn.disabled = false;
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

    @requires_auth
    async def index(_request: Request) -> HTMLResponse:
        return HTMLResponse(index_html)

    @requires_auth
    async def query(request: Request) -> JSONResponse:
        body = await request.json()
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return JSONResponse({"error": "missing 'sql'"}, status_code=400)
        query_id = uuid.uuid4().hex
        try:
            result = await run_in_threadpool(runner.run_query, sql, query_id)
        except DuckyError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(
            {
                "columns": result.columns,
                "rows": result.preview_rows,
                "total_rows": result.total_rows,
                "truncated": result.truncated,
                "result_path": result.result_path,
            }
        )

    @public
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/query", query, methods=["POST"]),
            Route("/health", health),
        ]
    )


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

    async def _unregister() -> None:
        ctx.registry.unregister(endpoint_id)
        logger.info("ducky endpoint unregistered")

    app.router.lifespan_context = on_shutdown(_unregister)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
