# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP dashboard with Connect RPC and web UI.

The dashboard serves:
- Web UI at / (main dashboard with tabs: jobs, fleet, endpoints, autoscaler, logs, transactions)
- Web UI at /job/{job_id} (job detail page)
- Web UI at /worker/{id} (worker detail page)
- Connect RPC at /iris.cluster.ControllerService/* (called directly by JS)
- Health check at /health

All data fetching happens via Connect RPC calls from the browser JavaScript.
The Python layer only serves HTML shells; all rendering is done client-side.
"""

import logging

from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import html_shell, static_files_mount
from iris.rpc.cluster_connect import ControllerServiceWSGIApplication
from iris.rpc.interceptors import RequestTimingInterceptor

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
    ):
        self._service = service
        self._host = host
        self._port = port
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> Starlette:
        return self._app

    def _create_app(self) -> Starlette:
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service, interceptors=[RequestTimingInterceptor()])
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/", self._dashboard),
            Route("/job/{job_id:path}", self._job_detail_page),
            Route("/worker/{worker_id:path}", self._worker_detail_page),
            Route("/bundles/{bundle_id:str}.zip", self._bundle_download),
            Route("/health", self._health),
            Mount(rpc_wsgi_app.path, app=rpc_app),
            static_files_mount(),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Controller", "/static/controller/app.js"))

    def _job_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Job Detail", "/static/controller/job-detail.js"))

    def _worker_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Worker Detail", "/static/controller/worker-detail.js"))

    def _health(self, _request: Request) -> JSONResponse:
        """Health check endpoint for controller availability."""
        return JSONResponse({"status": "ok"})

    def _bundle_download(self, request: Request) -> Response:
        bundle_id = request.path_params["bundle_id"]
        data = self._service.bundle_zip(bundle_id)
        return Response(data, media_type="application/zip")
