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

"""HTTP dashboard for controller visibility.

Provides a simple HTML dashboard showing the state of workers and jobs,
as well as a JSON health endpoint for monitoring.
"""

import html

from aiohttp import web

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState


def _job_state_name(state: int) -> str:
    """Convert job state integer to human-readable name.

    Args:
        state: JobState enum value

    Returns:
        Human-readable state name
    """
    state_map: dict[int, str] = {
        cluster_pb2.JOB_STATE_PENDING: "PENDING",
        cluster_pb2.JOB_STATE_RUNNING: "RUNNING",
        cluster_pb2.JOB_STATE_SUCCEEDED: "SUCCEEDED",
        cluster_pb2.JOB_STATE_FAILED: "FAILED",
        cluster_pb2.JOB_STATE_KILLED: "KILLED",
        cluster_pb2.JOB_STATE_WORKER_FAILED: "WORKER_FAILED",
    }
    return state_map.get(state, f"UNKNOWN({state})")


def create_dashboard_app(state: ControllerState) -> web.Application:
    """Create an aiohttp web application for the controller dashboard.

    The dashboard provides:
    - / - HTML page showing workers and jobs in tables
    - /health - JSON endpoint with worker/job counts

    Args:
        state: Controller state to display

    Returns:
        aiohttp Application with routes configured
    """
    app = web.Application()

    async def index(request: web.Request) -> web.Response:
        workers = state.list_all_workers()
        jobs = state.list_all_jobs()

        # Count workers by health status
        healthy_count = len([w for w in workers if w.healthy])

        # Generate HTML table rows for workers (escape user data to prevent XSS)
        worker_rows = "".join(
            f"<tr>"
            f"<td>{html.escape(str(w.worker_id))}</td>"
            f"<td>{html.escape(w.address)}</td>"
            f"<td>{'Yes' if w.healthy else 'No'}</td>"
            f"<td>{len(w.running_jobs)}</td>"
            f"</tr>"
            for w in workers
        )

        # Generate HTML table rows for jobs (escape user data to prevent XSS)
        job_rows = "".join(
            f"<tr>"
            f"<td>{html.escape(str(j.job_id))}</td>"
            f"<td>{html.escape(j.request.name)}</td>"
            f"<td>{_job_state_name(j.state)}</td>"
            f"<td>{html.escape(str(j.worker_id)) if j.worker_id else '-'}</td>"
            f"<td>{html.escape(j.error) if j.error else '-'}</td>"
            f"</tr>"
            for j in jobs
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Fluster Controller Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .summary {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Fluster Controller Dashboard</h1>

    <h2>Workers <span class="summary">({healthy_count} healthy / {len(workers)} total)</span></h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Address</th>
            <th>Healthy</th>
            <th>Running Jobs</th>
        </tr>
        {worker_rows}
    </table>

    <h2>Jobs <span class="summary">({len(jobs)} total)</span></h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>State</th>
            <th>Worker</th>
            <th>Error</th>
        </tr>
        {job_rows}
    </table>
</body>
</html>
"""
        return web.Response(text=html_content, content_type="text/html")

    async def health(request: web.Request) -> web.Response:
        workers = state.list_all_workers()
        jobs = state.list_all_jobs()

        healthy_count = len([w for w in workers if w.healthy])

        return web.json_response(
            {
                "status": "ok",
                "workers": len(workers),
                "healthy_workers": healthy_count,
                "jobs": len(jobs),
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/health", health)

    return app
