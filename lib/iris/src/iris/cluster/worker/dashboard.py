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
from iris.cluster.dashboard_common import html_shell, static_files_mount
from iris.rpc.cluster_connect import WorkerServiceWSGIApplication


class WorkerDashboard:
    """HTTP dashboard with Connect RPC and web UI."""

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
        self._server: uvicorn.Server | None = None

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        rpc_wsgi_app = WorkerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/health", self._health),
            Route("/", self._dashboard),
            Route("/task/{task_id:path}", self._task_detail_page),
            Route("/logs", self._logs_page),
            static_files_mount(),
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _logs_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Logs", "/static/worker/logs-page.js"))

    def _health(self, _request: Request) -> JSONResponse:
        """Simple health check endpoint for bootstrap and load balancers."""
        return JSONResponse({"status": "healthy"})

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Worker", "/static/worker/app.js"))

    def _task_detail_page(self, request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Task Detail", "/static/worker/task-detail.js"))

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
