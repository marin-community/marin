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
from fluster.cluster_connect import WorkerServiceASGIApplication
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
  </style>
</head>
<body>
  <h1>Fluster Worker Dashboard</h1>
  <div id="stats"></div>
  <h2>Jobs</h2>
  <table id="jobs">
    <tr><th>ID</th><th>Status</th><th>Started</th><th>Finished</th><th>Error</th></tr>
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
        return `<tr>
          <td>${j.job_id.slice(0, 8)}...</td>
          <td class="status-${j.status}">${j.status}</td>
          <td>${started}</td>
          <td>${finished}</td>
          <td>${j.error || '-'}</td>
        </tr>`;
      }).join('');
      document.getElementById('jobs').innerHTML =
        '<tr><th>ID</th><th>Status</th><th>Started</th><th>Finished</th><th>Error</th></tr>' + tbody;
    }
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
        self._server = None

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        """Create Starlette application with all routes."""
        rpc_app = WorkerServiceASGIApplication(service=self._service)

        routes = [
            # Web dashboard
            Route("/", self._dashboard),
            # REST API (for dashboard)
            Route("/api/stats", self._stats),
            Route("/api/jobs", self._list_jobs),
            Route("/api/jobs/{job_id}", self._get_job),
            Route("/api/jobs/{job_id}/logs", self._get_logs),
            # Connect RPC
            Mount(rpc_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    async def _dashboard(self, _request: Request) -> HTMLResponse:
        """Serve web dashboard HTML."""
        return HTMLResponse(DASHBOARD_HTML)

    async def _stats(self, _request: Request) -> JSONResponse:
        """Return job statistics by status."""
        # Call canonical RPC method
        ctx = FakeRequestContext()
        response = await self._service.list_jobs(cluster_pb2.ListJobsRequest(), ctx)
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

    async def _list_jobs(self, _request: Request) -> JSONResponse:
        """List all jobs as JSON."""
        # Call canonical RPC method
        ctx = FakeRequestContext()
        response = await self._service.list_jobs(cluster_pb2.ListJobsRequest(), ctx)
        jobs = response.jobs

        return JSONResponse(
            [
                {
                    "job_id": j.job_id,
                    "status": self._status_name(j.state),
                    "started_at": j.started_at_ms,
                    "finished_at": j.finished_at_ms,
                    "error": j.error,
                }
                for j in jobs
            ]
        )

    async def _get_job(self, request: Request) -> JSONResponse:
        """Get single job by ID."""
        job_id = request.path_params["job_id"]

        # Call canonical RPC method
        ctx = FakeRequestContext()
        try:
            job = await self._service.get_job_status(cluster_pb2.GetStatusRequest(job_id=job_id), ctx)
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
            }
        )

    async def _get_logs(self, request: Request) -> JSONResponse:
        """Get logs with optional tail parameter."""
        job_id = request.path_params["job_id"]

        # Support ?tail=N for last N lines
        tail = request.query_params.get("tail")
        start_line = -int(tail) if tail else 0

        # Call canonical RPC method
        ctx = FakeRequestContext()
        log_filter = cluster_pb2.FetchLogsFilter(start_line=start_line)
        try:
            response = await self._service.fetch_logs(
                cluster_pb2.FetchLogsRequest(job_id=job_id, filter=log_filter), ctx
            )
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing jobs
            return JSONResponse({"error": "Not found"}, status_code=404)

        return JSONResponse(
            [
                {
                    "timestamp": entry.timestamp_ms,
                    "source": entry.source,
                    "data": entry.data,
                }
                for entry in response.logs
            ]
        )

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
        """Run server (async)."""
        import uvicorn

        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        if self._server:
            self._server.should_exit = True
