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

"""HTTP dashboard with Connect RPC and web UI.

The dashboard serves:
- Web UI at / (main dashboard)
- Web UI at /job/{job_id} (job detail page)
- Web UI at /vm/{vm_id} (VM detail page)
- Connect RPC at /iris.cluster.ControllerService/* (called directly by JS)
- Health check at /health

All data fetching happens via Connect RPC calls from the browser JavaScript.
The Python layer only serves HTML shells; all rendering is done client-side.
"""

import logging

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import html_shell, static_files_mount
from iris.logging import LogBuffer
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceWSGIApplication

logger = logging.getLogger(__name__)


class ControllerDashboard:
    """HTTP dashboard with Connect RPC and web UI.

    The dashboard serves a single-page web UI that fetches all data directly
    via Connect RPC calls to the ControllerService. This eliminates the need
    for a separate REST API layer and ensures the dashboard shows exactly
    what the RPC returns.
    """

    def __init__(
        self,
        service: ControllerServiceImpl,
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
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/", self._dashboard),
            Route("/job/{job_id}", self._job_detail_page),
            Route("/vm/{vm_id}", self._vm_detail_page),
            Route("/logs", self._logs_page),
            Route("/api/logs", self._api_logs),
            Route("/health", self._health),
            Mount(rpc_wsgi_app.path, app=rpc_app),
            static_files_mount(),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Controller", "/static/controller/app.js"))

    def _job_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Job Detail", "/static/controller/job-detail.js"))

    def _vm_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("VM Detail", "/static/controller/vm-detail.js"))

    def _logs_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Logs", "/static/shared/log-viewer.js"))

    def _api_logs(self, request: Request) -> JSONResponse:
        if not self._log_buffer:
            return JSONResponse([])
        prefix = request.query_params.get("prefix") or None
        try:
            limit = int(request.query_params.get("limit", "200"))
        except (ValueError, TypeError):
            limit = 200
        records = self._log_buffer.query(prefix=prefix, limit=limit)
        return JSONResponse(
            [
                {"timestamp": r.timestamp, "level": r.level, "logger_name": r.logger_name, "message": r.message}
                for r in records
            ]
        )

    def _health(self, _request: Request) -> JSONResponse:
        """Health check endpoint for controller availability."""
        workers_resp = self._service.list_workers(cluster_pb2.Controller.ListWorkersRequest(), None)
        jobs_resp = self._service.list_jobs(cluster_pb2.Controller.ListJobsRequest(), None)
        worker_count = len(workers_resp.workers)
        job_count = len(jobs_resp.jobs)

        response = {
            "status": "ok",
            "workers": worker_count,
            "jobs": job_count,
        }

        return JSONResponse(response)

    def run(self) -> None:
        uvicorn.run(self._app, host=self._host, port=self._port)

    async def run_async(self) -> None:
        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        if self._server:
            self._server.should_exit = True
