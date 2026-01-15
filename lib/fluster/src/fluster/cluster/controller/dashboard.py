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

# TODO: observability, gregate stats over jobs , log to stable storage

from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.types import JobId
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceWSGIApplication


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
        cluster_pb2.JOB_STATE_BUILDING: "building",
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
    .worker-link, .job-link { color: #2196F3; text-decoration: none; }
    .worker-link:hover, .job-link:hover { text-decoration: underline; }
    .status-building { color: #9b59b6; }
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
      <div class="stat-value" id="jobs-building">-</div>
      <div class="stat-label">Jobs Building</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="workers-healthy">-</div>
      <div class="stat-label">Workers Healthy</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="workers-total">-</div>
      <div class="stat-label">Workers Total</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" id="endpoints-count">-</div>
      <div class="stat-label">Endpoints</div>
    </div>
  </div>

  <h2>Recent Actions</h2>
  <div class="actions-log" id="actions"></div>

  <h2>Workers</h2>
  <table id="workers-table">
    <tr><th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th><th>Running Jobs</th><th>Last Heartbeat</th></tr>
  </table>

  <h2>Job Queue</h2>
  <table id="jobs-table">
    <tr><th>ID</th><th>Name</th><th>State</th><th>Resources</th><th>Worker</th><th>Error</th></tr>
  </table>

  <h2>Endpoints</h2>
  <table id="endpoints-table">
    <tr><th>Name</th><th>Address</th><th>Job</th><th>Namespace</th></tr>
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
        const [stats, actions, workers, jobs, endpoints] = await Promise.all([
          fetch('/api/stats').then(r => r.json()),
          fetch('/api/actions').then(r => r.json()),
          fetch('/api/workers').then(r => r.json()),
          fetch('/api/jobs').then(r => r.json()),
          fetch('/api/endpoints').then(r => r.json())
        ]);

        // Update stats
        document.getElementById('jobs-pending').textContent = stats.jobs_pending;
        document.getElementById('jobs-running').textContent = stats.jobs_running;
        document.getElementById('jobs-completed').textContent = stats.jobs_completed;
        document.getElementById('jobs-building').textContent = stats.jobs_building;
        document.getElementById('workers-healthy').textContent = stats.workers_healthy;
        document.getElementById('workers-total').textContent = stats.workers_total;
        document.getElementById('endpoints-count').textContent = stats.endpoints_count;

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
          const lastHb = w.last_heartbeat_ms
            ? new Date(w.last_heartbeat_ms).toLocaleString() : '-';
          const healthClass = w.healthy ? 'healthy' : 'unhealthy';
          const wid = escapeHtml(w.worker_id);
          const workerLink = w.address
            ? `<a href="http://${escapeHtml(w.address)}/" class="worker-link" target="_blank">${wid}</a>`
            : wid;
          const cpu = w.resources ? w.resources.cpu : '-';
          const memory = w.resources ? (w.resources.memory || '-') : '-';
          return `<tr>
            <td>${workerLink}</td>
            <td class="${healthClass}">${w.healthy ? 'Yes' : 'No'}</td>
            <td>${cpu}</td>
            <td>${memory}</td>
            <td>${w.running_jobs}</td>
            <td>${lastHb}</td>
          </tr>`;
        }).join('');
        const workersHeader = '<tr><th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th>' +
          '<th>Running Jobs</th><th>Last Heartbeat</th></tr>';
        document.getElementById('workers-table').innerHTML = workersHeader + workersHtml;

        // Update jobs table
        const jobsHtml = jobs.map(j => {
          const jid = escapeHtml(j.job_id);
          const jobLink = `<a href="/job/${jid}" class="job-link">${jid.slice(0,8)}...</a>`;
          const resources = j.resources
            ? `${j.resources.cpu} CPU, ${j.resources.memory || '-'}` : '-';
          return `<tr>
            <td>${jobLink}</td>
            <td>${escapeHtml(j.name)}</td>
            <td class="status-${j.state}">${escapeHtml(j.state)}</td>
            <td>${resources}</td>
            <td>${escapeHtml(j.worker_id) || '-'}</td>
            <td>${escapeHtml(j.error) || '-'}</td>
          </tr>`;
        }).join('');
        const jobsHeader = '<tr><th>ID</th><th>Name</th><th>State</th><th>Resources</th>' +
          '<th>Worker</th><th>Error</th></tr>';
        document.getElementById('jobs-table').innerHTML = jobsHeader + jobsHtml;

        // Update endpoints table
        const endpointsHtml = endpoints.map(e => {
          const eid = escapeHtml(e.job_id);
          const jobLink = `<a href="/job/${eid}" class="job-link">${eid.slice(0,8)}...</a>`;
          return `<tr>
            <td>${escapeHtml(e.name)}</td>
            <td>${escapeHtml(e.address)}</td>
            <td>${jobLink}</td>
            <td>${escapeHtml(e.namespace)}</td>
          </tr>`;
        }).join('');
        document.getElementById('endpoints-table').innerHTML =
          '<tr><th>Name</th><th>Address</th><th>Job</th><th>Namespace</th></tr>' + endpointsHtml;
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


JOB_DETAIL_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Job Detail - {{job_id}}</title>
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
    .back-link {
      color: #2196F3;
      text-decoration: none;
      margin-bottom: 20px;
      display: inline-block;
    }
    .back-link:hover { text-decoration: underline; }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }
    .info-card {
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-card h3 {
      margin-top: 0;
      color: #34495e;
      border-bottom: 1px solid #ecf0f1;
      padding-bottom: 10px;
    }
    .info-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #ecf0f1;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #7f8c8d; }
    .info-value { font-weight: 500; }
    .status-pending { color: #f39c12; }
    .status-building { color: #9b59b6; }
    .status-running { color: #3498db; }
    .status-succeeded { color: #27ae60; }
    .status-failed { color: #e74c3c; }
    .status-killed { color: #95a5a6; }
    .status-worker_failed { color: #9b59b6; }
    .log-tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
    }
    .log-tab {
      padding: 8px 16px;
      background: #ecf0f1;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    .log-tab.active {
      background: #3498db;
      color: white;
    }
    .log-container {
      background: #1e1e1e;
      color: #d4d4d4;
      padding: 15px;
      border-radius: 8px;
      font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
      font-size: 13px;
      max-height: 500px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .log-line { padding: 2px 0; }
    .log-stdout { color: #d4d4d4; }
    .log-stderr { color: #f48771; }
    .log-build { color: #9cdcfe; }
    .error-message {
      background: #fee;
      border: 1px solid #e74c3c;
      color: #c0392b;
      padding: 15px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
    .no-worker-warning {
      background: #fff3cd;
      border: 1px solid #ffc107;
      color: #856404;
      padding: 15px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <a href="/" class="back-link">&larr; Back to Dashboard</a>
  <h1>Job: {{job_id}}</h1>

  <div id="no-worker-warning" class="no-worker-warning" style="display: none;">
    No worker assigned to this job. Job may be pending or completed. Limited information available.
  </div>

  <div id="error-container"></div>

  <div class="info-grid">
    <div class="info-card">
      <h3>Status</h3>
      <div class="info-row">
        <span class="info-label">State</span>
        <span class="info-value" id="job-state">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Exit Code</span>
        <span class="info-value" id="job-exit-code">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Started</span>
        <span class="info-value" id="job-started">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Finished</span>
        <span class="info-value" id="job-finished">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Duration</span>
        <span class="info-value" id="job-duration">-</span>
      </div>
    </div>

    <div class="info-card">
      <h3>Resources</h3>
      <div class="info-row">
        <span class="info-label">Memory Used</span>
        <span class="info-value" id="resource-memory">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">CPU Usage</span>
        <span class="info-value" id="resource-cpu">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Disk Used</span>
        <span class="info-value" id="resource-disk">-</span>
      </div>
    </div>

    <div class="info-card">
      <h3>Build Info</h3>
      <div class="info-row">
        <span class="info-label">Image Tag</span>
        <span class="info-value" id="build-image">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Cache Status</span>
        <span class="info-value" id="build-cache">-</span>
      </div>
    </div>
  </div>

  <h2>Logs</h2>
  <div class="log-tabs">
    <button class="log-tab active" data-filter="all">ALL</button>
    <button class="log-tab" data-filter="stdout">STDOUT</button>
    <button class="log-tab" data-filter="stderr">STDERR</button>
    <button class="log-tab" data-filter="build">BUILD</button>
  </div>
  <div class="log-container" id="log-container">Loading logs...</div>

  <script>
    const jobId = '{{job_id}}';
    const workerAddress = '{{worker_address}}';
    let allLogs = [];
    let currentFilter = 'all';

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }

    function formatTimestamp(ms) {
      if (!ms) return '-';
      return new Date(ms).toLocaleString();
    }

    function formatDuration(startMs, endMs) {
      if (!startMs) return '-';
      const end = endMs || Date.now();
      const seconds = Math.floor((end - startMs) / 1000);
      if (seconds < 60) return `${seconds}s`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
      return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    function getStateClass(state) {
      const stateMap = {
        'pending': 'status-pending',
        'building': 'status-building',
        'running': 'status-running',
        'succeeded': 'status-succeeded',
        'failed': 'status-failed',
        'killed': 'status-killed',
        'worker_failed': 'status-worker_failed'
      };
      return stateMap[state] || '';
    }

    function renderLogs() {
      const container = document.getElementById('log-container');
      const filtered = currentFilter === 'all'
        ? allLogs
        : allLogs.filter(l => l.stream === currentFilter);

      if (filtered.length === 0) {
        container.innerHTML = '<div class="log-line">No logs available</div>';
        return;
      }

      container.innerHTML = filtered.map(l => {
        const streamClass = l.stream === 'stderr' ? 'log-stderr' :
                           l.stream === 'build' ? 'log-build' : 'log-stdout';
        return `<div class="log-line ${streamClass}">${escapeHtml(l.line)}</div>`;
      }).join('');

      container.scrollTop = container.scrollHeight;
    }

    async function fetchJobStatus() {
      if (!workerAddress) {
        document.getElementById('no-worker-warning').style.display = 'block';
        return;
      }

      try {
        const response = await fetch(`http://${workerAddress}/api/jobs/${jobId}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const job = await response.json();

        // Update status (worker returns 'status' not 'state')
        const stateEl = document.getElementById('job-state');
        stateEl.textContent = job.status || '-';
        stateEl.className = 'info-value ' + getStateClass(job.status);

        document.getElementById('job-exit-code').textContent =
          job.exit_code !== undefined && job.exit_code !== null ? job.exit_code : '-';
        document.getElementById('job-started').textContent =
          formatTimestamp(job.started_at);
        document.getElementById('job-finished').textContent =
          formatTimestamp(job.finished_at);
        document.getElementById('job-duration').textContent =
          formatDuration(job.started_at, job.finished_at);

        // Update resources (worker returns resources.memory_mb etc)
        if (job.resources) {
          document.getElementById('resource-memory').textContent =
            job.resources.memory_mb ? `${job.resources.memory_mb} MB (peak: ${job.resources.memory_peak_mb || 0})` : '-';
          document.getElementById('resource-cpu').textContent =
            job.resources.cpu_percent !== undefined ? `${job.resources.cpu_percent}%` : '-';
          document.getElementById('resource-disk').textContent =
            job.resources.disk_mb ? `${job.resources.disk_mb} MB` : '-';
        }

        // Update build info (worker returns build.from_cache, build.image_tag)
        if (job.build) {
          document.getElementById('build-image').textContent =
            job.build.image_tag || '-';
          document.getElementById('build-cache').textContent =
            job.build.from_cache ? 'Cache Hit' : 'Cache Miss';
        }

        // Show error if present
        if (job.error) {
          document.getElementById('error-container').innerHTML =
            `<div class="error-message"><strong>Error:</strong> ${escapeHtml(job.error)}</div>`;
        } else {
          document.getElementById('error-container').innerHTML = '';
        }
      } catch (e) {
        console.error('Failed to fetch job status:', e);
      }
    }

    async function fetchLogs() {
      if (!workerAddress) return;

      try {
        const response = await fetch(`http://${workerAddress}/api/jobs/${jobId}/logs`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const logs = await response.json();
        // Worker returns array directly with {timestamp, source, data}
        allLogs = Array.isArray(logs) ? logs.map(l => ({
          stream: l.source,
          line: `[${new Date(l.timestamp).toLocaleTimeString()}] ${l.data}`
        })) : [];
        renderLogs();
      } catch (e) {
        console.error('Failed to fetch logs:', e);
        document.getElementById('log-container').innerHTML =
          '<div class="log-line">Failed to load logs</div>';
      }
    }

    // Tab click handlers
    document.querySelectorAll('.log-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.log-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentFilter = tab.dataset.filter;
        renderLogs();
      });
    });

    async function refresh() {
      await Promise.all([fetchJobStatus(), fetchLogs()]);
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
            Route("/job/{job_id}", self._job_detail_page),
            # REST API (for dashboard)
            Route("/api/stats", self._api_stats),
            Route("/api/actions", self._api_actions),
            Route("/api/workers", self._api_workers),
            Route("/api/jobs", self._api_jobs),
            Route("/api/endpoints", self._api_endpoints),
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
        jobs_response = self._service.list_jobs(cluster_pb2.Controller.ListJobsRequest(), ctx)
        workers = self._state.list_all_workers()

        jobs_pending = sum(1 for j in jobs_response.jobs if j.state == cluster_pb2.JOB_STATE_PENDING)
        jobs_running = sum(1 for j in jobs_response.jobs if j.state == cluster_pb2.JOB_STATE_RUNNING)
        jobs_building = sum(1 for j in jobs_response.jobs if j.state == cluster_pb2.JOB_STATE_BUILDING)
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

        # Count endpoints for running jobs
        endpoints_count = sum(
            1
            for ep in self._state._endpoints.values()
            if (job := self._state.get_job(ep.job_id)) and job.state == cluster_pb2.JOB_STATE_RUNNING
        )

        return JSONResponse(
            {
                "jobs_pending": jobs_pending,
                "jobs_running": jobs_running,
                "jobs_building": jobs_building,
                "jobs_completed": jobs_completed,
                "workers_healthy": workers_healthy,
                "workers_total": len(workers),
                "endpoints_count": endpoints_count,
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
                    "resources": {
                        "cpu": w.resources.cpu if w.resources else 0,
                        "memory": w.resources.memory if w.resources else "",
                    },
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
                    "resources": {
                        "cpu": j.request.resources.cpu if j.request.resources else 0,
                        "memory": j.request.resources.memory if j.request.resources else "",
                    },
                }
                for j in jobs
            ]
        )

    def _api_endpoints(self, _request: Request) -> JSONResponse:
        """Return all active endpoints for RUNNING jobs."""
        endpoints = []
        for ep in self._state._endpoints.values():
            job = self._state.get_job(ep.job_id)
            if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                endpoints.append(
                    {
                        "endpoint_id": ep.endpoint_id,
                        "name": ep.name,
                        "address": ep.address,
                        "job_id": str(ep.job_id),
                        "metadata": dict(ep.metadata),
                    }
                )
        return JSONResponse(endpoints)

    def _job_detail_page(self, request: Request) -> HTMLResponse:
        """Serve job detail page - fetches from worker via JS."""
        job_id = request.path_params["job_id"]
        job = self._state.get_job(JobId(job_id))
        worker_address = ""
        if job and job.worker_id:
            worker = self._state.get_worker(job.worker_id)
            if worker:
                worker_address = worker.address
        return HTMLResponse(JOB_DETAIL_HTML.replace("{{job_id}}", job_id).replace("{{worker_address}}", worker_address))

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
