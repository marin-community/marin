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

"""HTTP dashboard for controller visibility with Connect RPC mounting.

Provides:
- Web dashboard at / with auto-refresh
- REST API at /api/* for dashboard consumption
- Health endpoint at /health
- Connect RPC at /fluster.cluster.ControllerService/*
"""

from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from fluster import cluster_pb2
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster_connect import ControllerServiceWSGIApplication


class FakeRequestContext:
    """Minimal stub RequestContext for internal REST-to-RPC bridging.

    The ControllerDashboard translates REST API calls to RPC method calls,
    which require a RequestContext parameter. Since the RPC methods never
    actually access the context, this minimal stub satisfies the type signature.
    """

    pass


def _job_state_name(state: int) -> str:
    """Convert job state integer to human-readable name."""
    state_map: dict[int, str] = {
        cluster_pb2.JOB_STATE_PENDING: "pending",
        cluster_pb2.JOB_STATE_RUNNING: "running",
        cluster_pb2.JOB_STATE_SUCCEEDED: "succeeded",
        cluster_pb2.JOB_STATE_FAILED: "failed",
        cluster_pb2.JOB_STATE_KILLED: "killed",
        cluster_pb2.JOB_STATE_WORKER_FAILED: "worker_failed",
    }
    return state_map.get(state, f"unknown({state})")


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Fluster Controller</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      max-width: 1400px;
      margin: 40px auto;
      padding: 0 20px;
      color: #333;
      background: #f5f5f5;
    }
    h1 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      color: #34495e;
      margin-top: 30px;
    }
    .stats {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    .stat-card {
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      min-width: 150px;
      text-align: center;
    }
    .stat-value {
      font-size: 32px;
      font-weight: bold;
      color: #3498db;
    }
    .stat-label {
      font-size: 14px;
      color: #7f8c8d;
      margin-top: 5px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      border-radius: 8px;
      overflow: hidden;
    }
    th {
      background-color: #3498db;
      color: white;
      padding: 12px;
      text-align: left;
      font-weight: 600;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #ecf0f1;
    }
    tr:hover {
      background-color: #f8f9fa;
    }
    .status-pending { color: #f39c12; }
    .status-running { color: #3498db; }
    .status-succeeded { color: #27ae60; }
    .status-failed { color: #e74c3c; }
    .status-killed { color: #95a5a6; }
    .status-worker_failed { color: #9b59b6; }
    .healthy { color: #27ae60; }
    .unhealthy { color: #e74c3c; }
    .actions-log {
      background: white;
      padding: 15px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      max-height: 300px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 13px;
    }
    .action-entry {
      padding: 5px 0;
      border-bottom: 1px solid #ecf0f1;
    }
    .action-time {
      color: #7f8c8d;
      margin-right: 10px;
    }
    .future-feature {
      color: #95a5a6;
      font-style: italic;
      padding: 20px;
      background: white;
      border-radius: 8px;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1>Fluster Controller Dashboard</h1>

  <div class="stats" id="stats">
    <div class="stat-card">
      <div class="stat-value" id="jobs-pending">-</div>
      <div class="stat-label">Jobs Pending</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="jobs-running">-</div>
      <div class="stat-label">Jobs Running</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="jobs-completed">-</div>
      <div class="stat-label">Jobs Completed</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="workers-healthy">-</div>
      <div class="stat-label">Workers Healthy</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="workers-total">-</div>
      <div class="stat-label">Workers Total</div>
    </div>
  </div>

  <h2>Recent Actions</h2>
  <div class="actions-log" id="actions"></div>

  <h2>Workers</h2>
  <table id="workers-table">
    <tr><th>ID</th><th>Healthy</th><th>Running Jobs</th><th>Last Heartbeat</th></tr>
  </table>

  <h2>Job Queue</h2>
  <table id="jobs-table">
    <tr><th>ID</th><th>Name</th><th>State</th><th>Worker</th><th>Error</th></tr>
  </table>

  <h2>Users</h2>
  <div class="future-feature">Coming in future release</div>

  <h2>Reservations</h2>
  <div class="future-feature">Coming in future release</div>

  <script>
    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }

    async function refresh() {
      try {
        const [stats, actions, workers, jobs] = await Promise.all([
          fetch('/api/stats').then(r => r.json()),
          fetch('/api/actions').then(r => r.json()),
          fetch('/api/workers').then(r => r.json()),
          fetch('/api/jobs').then(r => r.json())
        ]);

        // Update stats
        document.getElementById('jobs-pending').textContent = stats.jobs_pending;
        document.getElementById('jobs-running').textContent = stats.jobs_running;
        document.getElementById('jobs-completed').textContent = stats.jobs_completed;
        document.getElementById('workers-healthy').textContent = stats.workers_healthy;
        document.getElementById('workers-total').textContent = stats.workers_total;

        // Update actions log
        const actionsHtml = actions.map(a => {
          const time = new Date(a.timestamp_ms).toLocaleTimeString();
          const jobInfo = a.job_id ? ` [job: ${a.job_id.slice(0,8)}...]` : '';
          const workerInfo = a.worker_id ? ` [worker: ${escapeHtml(a.worker_id)}]` : '';
          const details = a.details ? ` - ${escapeHtml(a.details)}` : '';
          return `<div class="action-entry"><span class="action-time">${time}</span>` +
            `${escapeHtml(a.action)}${jobInfo}${workerInfo}${details}</div>`;
        }).reverse().join('');
        document.getElementById('actions').innerHTML = actionsHtml || '<div class="action-entry">No actions yet</div>';

        // Update workers table
        const workersHtml = workers.map(w => {
          const lastHb = w.last_heartbeat_ms ? new Date(w.last_heartbeat_ms).toLocaleString() : '-';
          const healthClass = w.healthy ? 'healthy' : 'unhealthy';
          return `<tr>
            <td>${escapeHtml(w.worker_id)}</td>
            <td class="${healthClass}">${w.healthy ? 'Yes' : 'No'}</td>
            <td>${w.running_jobs}</td>
            <td>${lastHb}</td>
          </tr>`;
        }).join('');
        document.getElementById('workers-table').innerHTML =
          '<tr><th>ID</th><th>Healthy</th><th>Running Jobs</th><th>Last Heartbeat</th></tr>' + workersHtml;

        // Update jobs table
        const jobsHtml = jobs.map(j => {
          return `<tr>
            <td>${escapeHtml(j.job_id.slice(0,8))}...</td>
            <td class="status-${j.state}">${escapeHtml(j.state)}</td>
            <td>${escapeHtml(j.name)}</td>
            <td>${escapeHtml(j.worker_id) || '-'}</td>
            <td>${escapeHtml(j.error) || '-'}</td>
          </tr>`;
        }).join('');
        document.getElementById('jobs-table').innerHTML =
          '<tr><th>ID</th><th>State</th><th>Name</th><th>Worker</th><th>Error</th></tr>' + jobsHtml;
      } catch (e) {
        console.error('Failed to refresh:', e);
      }
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class ControllerDashboard:
    """HTTP dashboard with Connect RPC and web UI.

    Connect RPC is mounted at /fluster.cluster.ControllerService
    Web dashboard at /
    REST API for dashboard at /api/*
    """

    def __init__(
        self,
        service: ControllerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._service = service
        self._state = service._state
        self._host = host
        self._port = port
        self._app = self._create_app()
        self._server = None

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        """Create Starlette application with all routes."""
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            # Web dashboard
            Route("/", self._dashboard),
            # REST API (for dashboard)
            Route("/api/stats", self._api_stats),
            Route("/api/actions", self._api_actions),
            Route("/api/workers", self._api_workers),
            Route("/api/jobs", self._api_jobs),
            Route("/health", self._health),
            # Connect RPC - mount WSGI app wrapped for ASGI
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        """Serve web dashboard HTML."""
        return HTMLResponse(DASHBOARD_HTML)

    def _api_stats(self, _request: Request) -> JSONResponse:
        """Return aggregated statistics for the dashboard."""
        ctx = FakeRequestContext()
        jobs_response = self._service.list_jobs(cluster_pb2.ListJobsRequest(), ctx)
        workers = self._state.list_all_workers()

        jobs_pending = sum(1 for j in jobs_response.jobs if j.state == cluster_pb2.JOB_STATE_PENDING)
        jobs_running = sum(1 for j in jobs_response.jobs if j.state == cluster_pb2.JOB_STATE_RUNNING)
        jobs_completed = sum(
            1
            for j in jobs_response.jobs
            if j.state
            in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_WORKER_FAILED,
            )
        )
        workers_healthy = sum(1 for w in workers if w.healthy)

        return JSONResponse(
            {
                "jobs_pending": jobs_pending,
                "jobs_running": jobs_running,
                "jobs_completed": jobs_completed,
                "workers_healthy": workers_healthy,
                "workers_total": len(workers),
            }
        )

    def _api_actions(self, _request: Request) -> JSONResponse:
        """Return recent actions log."""
        actions = self._state.get_recent_actions(limit=50)
        return JSONResponse(
            [
                {
                    "timestamp_ms": a.timestamp_ms,
                    "action": a.action,
                    "job_id": str(a.job_id) if a.job_id else None,
                    "worker_id": str(a.worker_id) if a.worker_id else None,
                    "details": a.details,
                }
                for a in actions
            ]
        )

    def _api_workers(self, _request: Request) -> JSONResponse:
        """Return all workers with status."""
        workers = self._state.list_all_workers()
        return JSONResponse(
            [
                {
                    "worker_id": str(w.worker_id),
                    "address": w.address,
                    "healthy": w.healthy,
                    "running_jobs": len(w.running_jobs),
                    "consecutive_failures": w.consecutive_failures,
                    "last_heartbeat_ms": w.last_heartbeat_ms,
                }
                for w in workers
            ]
        )

    def _api_jobs(self, _request: Request) -> JSONResponse:
        """Return all jobs with status."""
        jobs = self._state.list_all_jobs()
        return JSONResponse(
            [
                {
                    "job_id": str(j.job_id),
                    "name": j.request.name,
                    "state": _job_state_name(j.state),
                    "worker_id": str(j.worker_id) if j.worker_id else None,
                    "error": j.error,
                    "submitted_at_ms": j.submitted_at_ms,
                    "started_at_ms": j.started_at_ms,
                    "finished_at_ms": j.finished_at_ms,
                }
                for j in jobs
            ]
        )

    def _health(self, _request: Request) -> JSONResponse:
        """Return health check status."""
        workers = self._state.list_all_workers()
        jobs = self._state.list_all_jobs()
        healthy_count = sum(1 for w in workers if w.healthy)

        return JSONResponse(
            {
                "status": "ok",
                "workers": len(workers),
                "healthy_workers": healthy_count,
                "jobs": len(jobs),
            }
        )

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
        if self._server:
            self._server.should_exit = True
