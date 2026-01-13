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

"""HTTP dashboard with Connect RPC and web UI for worker monitoring.

The WorkerDashboard provides:
- Connect RPC at /fluster.cluster.WorkerService
- Web dashboard at / with live job statistics
- REST API at /api/* for dashboard consumption

REST endpoints are implemented by calling the canonical RPC methods and
converting proto responses to JSON for browser consumption.
"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from fluster import cluster_pb2
from starlette.middleware.wsgi import WSGIMiddleware

from fluster.cluster_connect import WorkerServiceWSGIApplication
from fluster.cluster.worker.service import WorkerServiceImpl


class FakeRequestContext:
    """Minimal stub RequestContext for internal REST-to-RPC bridging.

    The WorkerDashboard translates REST API calls to RPC method calls,
    which require a RequestContext parameter. Since the RPC methods never
    actually access the context, this minimal stub satisfies the type signature.
    """

    pass


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Fluster Worker</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .status-running { color: blue; }
    .status-succeeded { color: green; }
    .status-failed { color: red; }
    .job-link { color: #2196F3; text-decoration: none; font-weight: bold; }
    .job-link:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Fluster Worker Dashboard</h1>
  <div id="stats"></div>
  <h2>Jobs</h2>
  <table id="jobs">
    <tr><th>ID</th><th>Status</th><th>Exit</th><th>Memory</th><th>CPU</th><th>Started</th><th>Finished</th><th>Error</th></tr>
  </table>
  <script>
    async function refresh() {
      const stats = await fetch('/api/stats').then(r => r.json());
      document.getElementById('stats').innerHTML =
        `<b>Running:</b> ${stats.running} | <b>Pending:</b> ${stats.pending} | ` +
        `<b>Building:</b> ${stats.building} | <b>Completed:</b> ${stats.completed}`;

      const jobs = await fetch('/api/jobs').then(r => r.json());
      const tbody = jobs.map(j => {
        const started = j.started_at ? new Date(j.started_at).toLocaleString() : '-';
        const finished = j.finished_at ? new Date(j.finished_at).toLocaleString() : '-';
        const exitCode = j.exit_code !== null && j.exit_code !== undefined ? j.exit_code : '-';
        return `<tr>
          <td><a href="/job/${j.job_id}" class="job-link" target="_blank">${j.job_id.slice(0, 8)}...</a></td>
          <td class="status-${j.status}">${j.status}</td>
          <td>${exitCode}</td>
          <td>${j.memory_mb || 0}/${j.memory_peak_mb || 0} MB</td>
          <td>${j.cpu_percent || 0}%</td>
          <td>${started}</td>
          <td>${finished}</td>
          <td>${j.error || '-'}</td>
        </tr>`;
      }).join('');
      document.getElementById('jobs').innerHTML =
        '<tr><th>ID</th><th>Status</th><th>Exit</th><th>Memory</th><th>CPU</th><th>Started</th><th>Finished</th><th>Error</th></tr>' + tbody;
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


JOB_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Job {{job_id}} - Fluster Worker</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    .status-running { color: blue; font-weight: bold; }
    .status-succeeded { color: green; font-weight: bold; }
    .status-failed { color: red; font-weight: bold; }
    .tabs { display: flex; border-bottom: 2px solid #4CAF50; }
    .tab { padding: 10px 20px; cursor: pointer; background: #f0f0f0; margin-right: 2px; border: 1px solid #ddd; border-bottom: none; border-radius: 5px 5px 0 0; }
    .tab.active { background: #4CAF50; color: white; }
    .tab-content { display: none; padding: 15px; border: 1px solid #ddd; max-height: 500px; overflow-y: auto; background: #f9f9f9; font-family: monospace; font-size: 12px; white-space: pre-wrap; }
    .tab-content.active { display: block; }
    .metrics { display: flex; gap: 30px; flex-wrap: wrap; }
    .metric { text-align: center; padding: 10px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
    .metric-label { font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <h1>Job: <code>{{job_id}}</code></h1>
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
    const jobId = "{{job_id}}";

    async function refresh() {
      const job = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
      if (job.error === "Not found") {
        document.getElementById('status').textContent = "Not Found";
        return;
      }

      document.getElementById('status').innerHTML = `<span class="status-${job.status}">${job.status}</span>`;
      document.getElementById('details').innerHTML = `
        <p><b>Started:</b> ${job.started_at ? new Date(job.started_at).toLocaleString() : '-'}</p>
        <p><b>Finished:</b> ${job.finished_at ? new Date(job.finished_at).toLocaleString() : '-'}</p>
        <p><b>Exit Code:</b> ${job.exit_code !== null ? job.exit_code : '-'}</p>
        <p><b>Error:</b> ${job.error || '-'}</p>
        <p><b>Ports:</b> ${JSON.stringify(job.ports)}</p>
      `;

      document.getElementById('resources').innerHTML = `
        <div class="metric"><div class="metric-value">${job.resources.memory_mb}</div><div class="metric-label">Memory (MB)</div></div>
        <div class="metric"><div class="metric-value">${job.resources.memory_peak_mb}</div><div class="metric-label">Peak Memory (MB)</div></div>
        <div class="metric"><div class="metric-value">${job.resources.cpu_percent}%</div><div class="metric-label">CPU</div></div>
        <div class="metric"><div class="metric-value">${job.resources.process_count}</div><div class="metric-label">Processes</div></div>
        <div class="metric"><div class="metric-value">${job.resources.disk_mb}</div><div class="metric-label">Disk (MB)</div></div>
      `;

      const buildDuration = job.build.duration_ms > 0 ? (job.build.duration_ms / 1000).toFixed(2) + 's' : '-';
      document.getElementById('build').innerHTML = `
        <p><b>Image:</b> <code>${job.build.image_tag || '-'}</code></p>
        <p><b>Build Time:</b> ${buildDuration}</p>
        <p><b>From Cache:</b> ${job.build.from_cache ? 'Yes' : 'No'}</p>
      `;

      const logs = await fetch(`/api/jobs/${jobId}/logs`).then(r => r.json());
      const format = (logs) => logs.map(l => `[${new Date(l.timestamp).toLocaleTimeString()}] ${l.data}`).join('\\n') || 'No logs';
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
    """HTTP dashboard with Connect RPC and web UI.

    Connect RPC is mounted at /fluster.cluster.WorkerService
    Web dashboard at /
    REST API for dashboard at /api/*
    """

    def __init__(
        self,
        service: WorkerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        """Create Starlette application with all routes."""
        # Use WSGI application for sync RPC handlers, wrapped for ASGI compatibility
        rpc_wsgi_app = WorkerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            # Web dashboard
            Route("/", self._dashboard),
            Route("/job/{job_id}", self._job_detail_page),
            # REST API (for dashboard)
            Route("/api/stats", self._stats),
            Route("/api/jobs", self._list_jobs),
            Route("/api/jobs/{job_id}", self._get_job),
            Route("/api/jobs/{job_id}/logs", self._get_logs),
            # Connect RPC - mount WSGI app wrapped for ASGI
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        """Serve web dashboard HTML."""
        return HTMLResponse(DASHBOARD_HTML)

    def _job_detail_page(self, request: Request) -> HTMLResponse:
        """Serve detailed job view page."""
        job_id = request.path_params["job_id"]
        return HTMLResponse(JOB_DETAIL_HTML.replace("{{job_id}}", job_id))

    def _stats(self, _request: Request) -> JSONResponse:
        """Return job statistics by status."""
        # Call canonical RPC method
        ctx = FakeRequestContext()
        response = self._service.list_jobs(cluster_pb2.ListJobsRequest(), ctx)
        jobs = response.jobs

        return JSONResponse(
            {
                "running": sum(1 for j in jobs if j.state == cluster_pb2.JOB_STATE_RUNNING),
                "pending": sum(1 for j in jobs if j.state == cluster_pb2.JOB_STATE_PENDING),
                "building": sum(1 for j in jobs if j.state == cluster_pb2.JOB_STATE_BUILDING),
                "completed": sum(
                    1
                    for j in jobs
                    if j.state
                    in (
                        cluster_pb2.JOB_STATE_SUCCEEDED,
                        cluster_pb2.JOB_STATE_FAILED,
                        cluster_pb2.JOB_STATE_KILLED,
                    )
                ),
            }
        )

    def _list_jobs(self, _request: Request) -> JSONResponse:
        """List all jobs as JSON."""
        # Call canonical RPC method
        ctx = FakeRequestContext()
        response = self._service.list_jobs(cluster_pb2.ListJobsRequest(), ctx)
        jobs = response.jobs

        return JSONResponse(
            [
                {
                    "job_id": j.job_id,
                    "status": self._status_name(j.state),
                    "started_at": j.started_at_ms,
                    "finished_at": j.finished_at_ms,
                    "exit_code": j.exit_code,
                    "error": j.error,
                    # Add resource metrics
                    "memory_mb": j.resource_usage.memory_mb,
                    "memory_peak_mb": j.resource_usage.memory_peak_mb,
                    "cpu_percent": j.resource_usage.cpu_percent,
                    "process_count": j.resource_usage.process_count,
                    "disk_mb": j.resource_usage.disk_mb,
                    # Add build metrics
                    "build_from_cache": j.build_metrics.from_cache,
                    "image_tag": j.build_metrics.image_tag,
                }
                for j in jobs
            ]
        )

    def _get_job(self, request: Request) -> JSONResponse:
        """Get single job by ID."""
        job_id = request.path_params["job_id"]

        # Call canonical RPC method
        ctx = FakeRequestContext()
        try:
            job = self._service.get_job_status(cluster_pb2.GetStatusRequest(job_id=job_id), ctx)
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing jobs
            return JSONResponse({"error": "Not found"}, status_code=404)

        return JSONResponse(
            {
                "job_id": job.job_id,
                "status": self._status_name(job.state),
                "started_at": job.started_at_ms,
                "finished_at": job.finished_at_ms,
                "exit_code": job.exit_code,
                "error": job.error,
                "ports": dict(job.ports),
                "resources": {
                    "memory_mb": job.resource_usage.memory_mb,
                    "memory_peak_mb": job.resource_usage.memory_peak_mb,
                    "cpu_percent": job.resource_usage.cpu_percent,
                    "disk_mb": job.resource_usage.disk_mb,
                    "process_count": job.resource_usage.process_count,
                },
                "build": {
                    "started_ms": job.build_metrics.build_started_ms,
                    "finished_ms": job.build_metrics.build_finished_ms,
                    "duration_ms": (
                        (job.build_metrics.build_finished_ms - job.build_metrics.build_started_ms)
                        if job.build_metrics.build_started_ms
                        else 0
                    ),
                    "from_cache": job.build_metrics.from_cache,
                    "image_tag": job.build_metrics.image_tag,
                },
            }
        )

    def _get_logs(self, request: Request) -> JSONResponse:
        """Get logs with optional tail and source parameters."""
        job_id = request.path_params["job_id"]

        # Support ?tail=N for last N lines
        tail = request.query_params.get("tail")
        start_line = -int(tail) if tail else 0

        # Support ?source=stdout|stderr|build for filtering
        source = request.query_params.get("source")

        # Call canonical RPC method
        ctx = FakeRequestContext()
        log_filter = cluster_pb2.FetchLogsFilter(start_line=start_line)
        try:
            response = self._service.fetch_logs(cluster_pb2.FetchLogsRequest(job_id=job_id, filter=log_filter), ctx)
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing jobs
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

    def _status_name(self, status: cluster_pb2.JobState) -> str:
        """Convert status enum to string name."""
        status_map = {
            cluster_pb2.JOB_STATE_PENDING: "pending",
            cluster_pb2.JOB_STATE_BUILDING: "building",
            cluster_pb2.JOB_STATE_RUNNING: "running",
            cluster_pb2.JOB_STATE_SUCCEEDED: "succeeded",
            cluster_pb2.JOB_STATE_FAILED: "failed",
            cluster_pb2.JOB_STATE_KILLED: "killed",
        }
        return status_map.get(status, "unknown")

    def run(self) -> None:
        """Run server (blocking)."""
        import uvicorn

        uvicorn.run(self._app, host=self._host, port=self._port)

    async def run_async(self) -> None:
        """Run server asynchronously (for use with asyncio.create_task)."""
        import uvicorn

        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        """Shutdown the async server gracefully."""
        if hasattr(self, "_server") and self._server:
            self._server.should_exit = True
