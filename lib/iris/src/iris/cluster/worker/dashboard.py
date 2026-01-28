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

"""HTTP dashboard with Connect RPC and web UI for worker monitoring."""

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.dashboard_common import logs_api_response, logs_page_response
from iris.logging import LogBuffer
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceWSGIApplication
from iris.rpc.proto_utils import task_state_name


class FakeRequestContext:
    """Minimal stub RequestContext for internal REST-to-RPC bridging.

    RPC methods require a RequestContext parameter but never access it.
    """

    pass


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Iris Worker</title>
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
    h1 { color: #1f2328; border-bottom: 2px solid #d1d9e0; padding-bottom: 10px; font-size: 24px; font-weight: 600; }
    h2 { color: #1f2328; margin-top: 30px; font-size: 20px; font-weight: 600; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #d1d9e0;
    }
    th {
      background-color: #f6f8fa;
      color: #1f2328;
      padding: 10px 12px;
      text-align: left;
      font-weight: 600;
      font-size: 13px;
      border-bottom: 1px solid #d1d9e0;
    }
    td {
      padding: 8px 12px;
      border-bottom: 1px solid #d1d9e0;
      font-size: 13px;
    }
    tr:hover { background-color: #f6f8fa; }
    .status-running { color: #0969da; }
    .status-building { color: #8250df; }
    .status-succeeded { color: #1a7f37; }
    .status-failed { color: #cf222e; }
    .status-killed { color: #57606a; }
    .status-pending { color: #9a6700; }
    .task-link { color: #0969da; text-decoration: none; font-weight: 500; }
    .task-link:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Iris Worker Dashboard</h1>
  <div id="stats"></div>
  <h2>Tasks</h2>
  <table id="tasks">
    <tr>
      <th>Task ID</th><th>Job ID</th><th>Index</th><th>Status</th><th>Exit</th>
      <th>Memory</th><th>CPU</th><th>Started</th><th>Finished</th><th>Error</th>
    </tr>
  </table>
  <script>
    async function refresh() {
      const stats = await fetch('/api/stats').then(r => r.json());
      document.getElementById('stats').innerHTML =
        `<b>Running:</b> ${stats.running} | <b>Pending:</b> ${stats.pending} | ` +
        `<b>Building:</b> ${stats.building} | <b>Completed:</b> ${stats.completed}`;

      const tasks = await fetch('/api/tasks').then(r => r.json());
      const tbody = tasks.map(t => {
        const started = t.started_at ? new Date(t.started_at).toLocaleString() : '-';
        const finished = t.finished_at ? new Date(t.finished_at).toLocaleString() : '-';
        const exitCode = t.exit_code !== null && t.exit_code !== undefined ? t.exit_code : '-';
        const taskDisplay = t.attempt_id > 0
          ? `${t.task_id.slice(0, 12)}... (attempt ${t.attempt_id})`
          : `${t.task_id.slice(0, 12)}...`;
        return `<tr>
          <td><a href="/task/${encodeURIComponent(t.task_id)}" class="task-link" target="_blank">${taskDisplay}</a></td>
          <td>${t.job_id.slice(0, 8)}...</td>
          <td>${t.task_index}/${t.num_tasks || '?'}</td>
          <td class="status-${t.status}">${t.status}</td>
          <td>${exitCode}</td>
          <td>${t.memory_mb || 0}/${t.memory_peak_mb || 0} MB</td>
          <td>${t.cpu_percent || 0}%</td>
          <td>${started}</td>
          <td>${finished}</td>
          <td>${t.error || '-'}</td>
        </tr>`;
      }).join('');
      document.getElementById('tasks').innerHTML =
        '<tr><th>Task ID</th><th>Job ID</th><th>Index</th><th>Status</th><th>Exit</th><th>Memory</th><th>CPU</th>' +
        '<th>Started</th><th>Finished</th><th>Error</th></tr>' + tbody;
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


TASK_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Task {{task_id}} - Iris Worker</title>
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
    h1 { color: #1f2328; border-bottom: 2px solid #d1d9e0; padding-bottom: 10px; font-size: 24px; font-weight: 600; }
    h2 { color: #1f2328; margin-top: 30px; font-size: 20px; font-weight: 600; }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .section {
      margin: 20px 0;
      padding: 15px 20px;
      background: white;
      border: 1px solid #d1d9e0;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .status-running { color: #0969da; font-weight: bold; }
    .status-building { color: #8250df; font-weight: bold; }
    .status-succeeded { color: #1a7f37; font-weight: bold; }
    .status-failed { color: #cf222e; font-weight: bold; }
    .status-killed { color: #57606a; font-weight: bold; }
    .status-pending { color: #9a6700; font-weight: bold; }
    .tabs { display: flex; border-bottom: 2px solid #d1d9e0; }
    .tab {
      padding: 10px 20px;
      cursor: pointer;
      background: transparent;
      margin-right: 2px;
      border: 1px solid #d1d9e0;
      border-bottom: none;
      border-radius: 6px 6px 0 0;
      font-size: 13px;
      font-weight: 500;
      color: #57606a;
    }
    .tab.active { background: #0969da; color: white; border-color: #0969da; }
    .tab-content {
      display: none;
      padding: 15px;
      border: 1px solid #d1d9e0;
      max-height: 500px;
      overflow-y: auto;
      background: white;
      font-family: monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .tab-content.active { display: block; }
    .metrics { display: flex; gap: 30px; flex-wrap: wrap; }
    .metric { text-align: center; padding: 10px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #0969da; }
    .metric-label { font-size: 12px; color: #57606a; }
  </style>
</head>
<body>
  <h1>Task: <code>{{task_id}}</code></h1>
  <a href="/">← Back to Dashboard</a>

  <div class="section">
    <h2>Status: <span id="status"></span></h2>
    <div id="details"></div>
  </div>

  <div class="section">
    <h2>Resources</h2>
    <div class="metrics" id="resources"></div>
  </div>

  <div class="section">
    <h2>Build</h2>
    <div id="build"></div>
  </div>

  <div class="section">
    <h2>Logs</h2>
    <div class="tabs">
      <div class="tab active" data-tab="all">ALL</div>
      <div class="tab" data-tab="stdout">STDOUT</div>
      <div class="tab" data-tab="stderr">STDERR</div>
      <div class="tab" data-tab="build">BUILD</div>
    </div>
    <div id="log-all" class="tab-content active"></div>
    <div id="log-stdout" class="tab-content"></div>
    <div id="log-stderr" class="tab-content"></div>
    <div id="log-build" class="tab-content"></div>
  </div>

  <script>
    const taskId = "{{task_id}}";

    async function refresh() {
      const task = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`).then(r => r.json());
      if (task.error === "Not found") {
        document.getElementById('status').textContent = "Not Found";
        return;
      }

      document.getElementById('status').innerHTML = `<span class="status-${task.status}">${task.status}</span>`;
      document.getElementById('details').innerHTML = `
        <p><b>Job ID:</b> ${task.job_id}</p>
        <p><b>Task Index:</b> ${task.task_index}</p>
        <p><b>Attempt:</b> ${task.attempt_id}</p>
        <p><b>Started:</b> ${task.started_at ? new Date(task.started_at).toLocaleString() : '-'}</p>
        <p><b>Finished:</b> ${task.finished_at ? new Date(task.finished_at).toLocaleString() : '-'}</p>
        <p><b>Exit Code:</b> ${task.exit_code !== null ? task.exit_code : '-'}</p>
        <p><b>Error:</b> ${task.error || '-'}</p>
        <p><b>Ports:</b> ${JSON.stringify(task.ports)}</p>
      `;

      document.getElementById('resources').innerHTML = `
        <div class="metric"><div class="metric-value">${task.resources.memory_mb}</div>
          <div class="metric-label">Memory (MB)</div></div>
        <div class="metric"><div class="metric-value">${task.resources.memory_peak_mb}</div>
          <div class="metric-label">Peak Memory (MB)</div></div>
        <div class="metric"><div class="metric-value">${task.resources.cpu_percent}%</div>
          <div class="metric-label">CPU</div></div>
        <div class="metric"><div class="metric-value">${task.resources.process_count}</div>
          <div class="metric-label">Processes</div></div>
        <div class="metric"><div class="metric-value">${task.resources.disk_mb}</div>
          <div class="metric-label">Disk (MB)</div></div>
      `;

      const buildDuration = task.build.duration_ms > 0 ? (task.build.duration_ms / 1000).toFixed(2) + 's' : '-';
      document.getElementById('build').innerHTML = `
        <p><b>Image:</b> <code>${task.build.image_tag || '-'}</code></p>
        <p><b>Build Time:</b> ${buildDuration}</p>
        <p><b>From Cache:</b> ${task.build.from_cache ? 'Yes' : 'No'}</p>
      `;

      const logs = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/logs`).then(r => r.json());
      const format = (logs) => logs.map(l =>
        `[${new Date(l.timestamp).toLocaleTimeString()}] ${l.data}`
      ).join('\\n') || 'No logs';
      document.getElementById('log-all').textContent = format(logs);
      document.getElementById('log-stdout').textContent = format(logs.filter(l => l.source === 'stdout'));
      document.getElementById('log-stderr').textContent = format(logs.filter(l => l.source === 'stderr'));
      document.getElementById('log-build').textContent = format(logs.filter(l => l.source === 'build'));
    }

    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('log-' + tab.dataset.tab).classList.add('active');
      });
    });

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class WorkerDashboard:
    """HTTP dashboard with Connect RPC and web UI."""

    def __init__(
        self,
        service: WorkerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
        log_buffer: LogBuffer | None = None,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._log_buffer = log_buffer
        self._app = self._create_app()
        self._server: uvicorn.Server | None = None

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        rpc_wsgi_app = WorkerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            # Health check (for bootstrap and load balancers)
            Route("/health", self._health),
            # Web dashboard
            Route("/", self._dashboard),
            Route("/task/{task_id:path}", self._task_detail_page),
            Route("/logs", self._logs_page),
            # REST API (for dashboard)
            # Note: logs route must come before get_task to avoid {task_id:path} matching "task-id/logs"
            Route("/api/stats", self._stats),
            Route("/api/tasks", self._list_tasks),
            Route("/api/logs", self._api_logs),
            Route("/api/tasks/{task_id:path}/logs", self._get_logs),
            Route("/api/tasks/{task_id:path}", self._get_task),
            # Connect RPC - mount WSGI app wrapped for ASGI
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _logs_page(self, request: Request) -> HTMLResponse:
        return logs_page_response(request)

    def _api_logs(self, request: Request):
        return logs_api_response(request, self._log_buffer)

    def _health(self, _request: Request) -> JSONResponse:
        """Simple health check endpoint for bootstrap and load balancers."""
        return JSONResponse({"status": "healthy"})

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    def _task_detail_page(self, request: Request) -> HTMLResponse:
        task_id = request.path_params["task_id"]
        return HTMLResponse(TASK_DETAIL_HTML.replace("{{task_id}}", task_id))

    def _stats(self, _request: Request) -> JSONResponse:
        ctx = FakeRequestContext()
        response = self._service.list_tasks(cluster_pb2.Worker.ListTasksRequest(), ctx)
        tasks = response.tasks

        return JSONResponse(
            {
                "running": sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_RUNNING),
                "pending": sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_PENDING),
                "building": sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_BUILDING),
                "completed": sum(
                    1
                    for t in tasks
                    if t.state
                    in (
                        cluster_pb2.TASK_STATE_SUCCEEDED,
                        cluster_pb2.TASK_STATE_FAILED,
                        cluster_pb2.TASK_STATE_KILLED,
                    )
                ),
            }
        )

    def _list_tasks(self, _request: Request) -> JSONResponse:
        ctx = FakeRequestContext()
        response = self._service.list_tasks(cluster_pb2.Worker.ListTasksRequest(), ctx)
        tasks = response.tasks

        return JSONResponse(
            [
                {
                    "task_id": t.task_id,
                    "job_id": t.job_id,
                    "task_index": t.task_index,
                    "attempt_id": t.current_attempt_id,
                    "status": task_state_name(t.state),
                    "started_at": t.started_at_ms,
                    "finished_at": t.finished_at_ms,
                    "exit_code": t.exit_code,
                    "error": t.error,
                    # Add resource metrics
                    "memory_mb": t.resource_usage.memory_mb,
                    "memory_peak_mb": t.resource_usage.memory_peak_mb,
                    "cpu_percent": t.resource_usage.cpu_percent,
                    "process_count": t.resource_usage.process_count,
                    "disk_mb": t.resource_usage.disk_mb,
                    # Add build metrics
                    "build_from_cache": t.build_metrics.from_cache,
                    "image_tag": t.build_metrics.image_tag,
                }
                for t in tasks
            ]
        )

    def _get_task(self, request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        ctx = FakeRequestContext()
        try:
            task = self._service.get_task_status(cluster_pb2.Worker.GetTaskStatusRequest(task_id=task_id), ctx)
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing tasks
            return JSONResponse({"error": "Not found"}, status_code=404)

        return JSONResponse(
            {
                "task_id": task.task_id,
                "job_id": task.job_id,
                "task_index": task.task_index,
                "attempt_id": task.current_attempt_id,
                "status": task_state_name(task.state),
                "started_at": task.started_at_ms,
                "finished_at": task.finished_at_ms,
                "exit_code": task.exit_code,
                "error": task.error,
                "ports": dict(task.ports),
                "resources": {
                    "memory_mb": task.resource_usage.memory_mb,
                    "memory_peak_mb": task.resource_usage.memory_peak_mb,
                    "cpu_percent": task.resource_usage.cpu_percent,
                    "disk_mb": task.resource_usage.disk_mb,
                    "process_count": task.resource_usage.process_count,
                },
                "build": {
                    "started_ms": task.build_metrics.build_started_ms,
                    "finished_ms": task.build_metrics.build_finished_ms,
                    "duration_ms": (
                        (task.build_metrics.build_finished_ms - task.build_metrics.build_started_ms)
                        if task.build_metrics.build_started_ms
                        else 0
                    ),
                    "from_cache": task.build_metrics.from_cache,
                    "image_tag": task.build_metrics.image_tag,
                },
            }
        )

    def _get_logs(self, request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        tail = request.query_params.get("tail")
        start_line = -int(tail) if tail else 0
        source = request.query_params.get("source")
        ctx = FakeRequestContext()
        log_filter = cluster_pb2.Worker.FetchLogsFilter(start_line=start_line)
        try:
            response = self._service.fetch_task_logs(
                cluster_pb2.Worker.FetchTaskLogsRequest(task_id=task_id, filter=log_filter), ctx
            )
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing tasks
            return JSONResponse({"error": "Not found"}, status_code=404)

        logs = [
            {
                "timestamp": entry.timestamp_ms,
                "source": entry.source,
                "data": entry.data,
            }
            for entry in response.logs
        ]

        # Apply source filter if specified
        if source:
            logs = [log for log in logs if log["source"] == source]

        return JSONResponse(logs)

    def run(self) -> None:
        import uvicorn

        uvicorn.run(self._app, host=self._host, port=self._port)

    async def run_async(self) -> None:
        import uvicorn

        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        if self._server:
            self._server.should_exit = True
