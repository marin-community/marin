# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared dashboard components for controller and worker dashboards."""

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from iris.logging import LogBuffer

LOGS_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Iris Logs</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      max-width: 1400px;
      margin: 40px auto;
      padding: 0 20px;
      color: #1f2328;
      background: #f6f8fa;
      font-size: 14px;
    }
    h1 {
      color: #1f2328;
      border-bottom: 2px solid #d1d9e0;
      padding-bottom: 10px;
      font-size: 24px;
      font-weight: 600;
    }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .controls {
      margin: 15px 0;
      display: flex;
      gap: 12px;
      align-items: center;
      background: white;
      padding: 12px 16px;
      border-radius: 6px;
      border: 1px solid #d1d9e0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .controls label { font-size: 13px; color: #57606a; font-weight: 500; }
    input, select {
      padding: 5px 10px;
      background: white;
      color: #1f2328;
      border: 1px solid #d1d9e0;
      border-radius: 6px;
      font-size: 13px;
    }
    input { width: 300px; }
    input:focus, select:focus { outline: none; border-color: #0969da; box-shadow: 0 0 0 3px rgba(9,105,218,0.15); }
    #log-container {
      max-height: 80vh;
      overflow-y: auto;
      background: white;
      border-radius: 6px;
      border: 1px solid #d1d9e0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      padding: 12px;
    }
    .log-line {
      padding: 2px 4px;
      white-space: pre-wrap;
      word-break: break-all;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      border-radius: 3px;
    }
    .log-line:hover { background: #f6f8fa; }
    .log-line.WARNING { color: #9a6700; }
    .log-line.ERROR { color: #cf222e; }
    .log-line.DEBUG { color: #57606a; }
    .log-line.INFO { color: #1f2328; }
    .empty-state {
      text-align: center;
      padding: 40px;
      color: #57606a;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <a href="/">&larr; Dashboard</a>
  <h1>Process Logs</h1>
  <div class="controls">
    <label>Prefix: <input id="prefix" type="text" placeholder="e.g. iris.cluster.controller"></label>
    <label>Limit: <select id="limit">
      <option value="100">100</option>
      <option value="200" selected>200</option>
      <option value="500">500</option>
      <option value="1000">1000</option>
    </select></label>
  </div>
  <div id="log-container"><div class="empty-state">Loading logs...</div></div>
  <script>
    function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
    async function refresh() {
      const prefix = document.getElementById('prefix').value;
      const limit = document.getElementById('limit').value;
      const params = new URLSearchParams({limit});
      if (prefix) params.set('prefix', prefix);
      const logs = await fetch('/api/logs?' + params).then(r => r.json());
      const container = document.getElementById('log-container');
      if (logs.length === 0) {
        container.innerHTML = '<div class="empty-state">No logs found</div>';
      } else {
        container.innerHTML = logs.map(
          l => `<div class="log-line ${escapeHtml(l.level)}">${escapeHtml(l.message)}</div>`
        ).join('');
      }
    }
    refresh();
    setInterval(refresh, 3000);
    document.getElementById('prefix').addEventListener('input', refresh);
    document.getElementById('limit').addEventListener('change', refresh);
  </script>
</body>
</html>
"""


def logs_page_response(_request: Request) -> HTMLResponse:
    """Starlette route handler returning the logs HTML page."""
    return HTMLResponse(LOGS_HTML)


def logs_api_response(request: Request, buffer: LogBuffer | None) -> JSONResponse:
    """Starlette route handler returning log records as JSON."""
    if not buffer:
        return JSONResponse([])
    prefix = request.query_params.get("prefix") or None
    try:
        limit = int(request.query_params.get("limit", "200"))
    except (ValueError, TypeError):
        limit = 200
    records = buffer.query(prefix=prefix, limit=limit)
    return JSONResponse(
        [
            {"timestamp": r.timestamp, "level": r.level, "logger_name": r.logger_name, "message": r.message}
            for r in records
        ]
    )
