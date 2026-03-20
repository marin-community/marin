# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP dashboard with Connect RPC and web UI for worker monitoring."""

from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.dashboard_common import html_shell, static_files_mount
from iris.rpc.auth import AuthInterceptor, NullAuthInterceptor, StaticTokenVerifier, TokenVerifier
from iris.rpc.cluster_connect import WorkerServiceWSGIApplication

WORKER_AUTH_USER = "system:worker"


class WorkerDashboard:
    """HTTP dashboard with Connect RPC and web UI."""

    def __init__(
        self,
        service: WorkerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_token: str = "",
    ):
        self._service = service
        self._host = host
        self._port = port
        self._auth_token = auth_token
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> Starlette:
        return self._app

    def _build_verifier(self) -> TokenVerifier | None:
        if not self._auth_token:
            return None
        return StaticTokenVerifier(
            tokens={self._auth_token: WORKER_AUTH_USER},
            roles={WORKER_AUTH_USER: "worker"},
        )

    def _create_app(self) -> Starlette:
        verifier = self._build_verifier()
        interceptors: list = []
        if verifier is not None:
            interceptors.append(AuthInterceptor(verifier))
        else:
            interceptors.append(NullAuthInterceptor())
        rpc_wsgi_app = WorkerServiceWSGIApplication(service=self._service, interceptors=interceptors)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/health", self._health),
            Route("/", self._dashboard),
            Route("/task/{task_id:path}", self._task_detail_page),
            Route("/status", self._status_page),
            static_files_mount(),
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _status_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Status", "worker"))

    def _health(self, _request: Request) -> JSONResponse:
        """Simple health check endpoint for bootstrap and load balancers."""
        return JSONResponse({"status": "healthy"})

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Worker", "worker"))

    def _task_detail_page(self, request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Task Detail", "worker"))
