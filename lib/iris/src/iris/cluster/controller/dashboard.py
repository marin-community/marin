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

import httpx
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import html_shell, static_files_mount
from iris.rpc.auth import SESSION_COOKIE, AuthInterceptor, NullAuthInterceptor, TokenVerifier
from iris.rpc.cluster_connect import ControllerServiceWSGIApplication
from iris.rpc.interceptors import RequestTimingInterceptor

logger = logging.getLogger(__name__)


_UNAUTHENTICATED_RPCS = {"Login", "GetAuthInfo"}


class _SelectiveAuthInterceptor:
    """Auth interceptor that skips authentication for specific RPC methods."""

    def __init__(self, verifier: TokenVerifier):
        self._inner = AuthInterceptor(verifier)

    def intercept_unary_sync(self, call_next, request, ctx):
        if ctx.method().name in _UNAUTHENTICATED_RPCS:
            return call_next(request, ctx)
        return self._inner.intercept_unary_sync(call_next, request, ctx)


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
        auth_verifier: TokenVerifier | None = None,
        auth_provider: str | None = None,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._auth_verifier = auth_verifier
        self._auth_provider = auth_provider
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> Starlette:
        return self._app

    def _create_app(self) -> Starlette:
        interceptors = [RequestTimingInterceptor()]
        if self._auth_provider is not None:
            interceptors.insert(0, _SelectiveAuthInterceptor(self._auth_verifier))
        else:
            interceptors.insert(0, NullAuthInterceptor(verifier=self._auth_verifier))
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service, interceptors=interceptors)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/", self._dashboard),
            Route("/auth/config", self._auth_config),
            Route("/auth/session", self._auth_session, methods=["POST"]),
            Route("/auth/logout", self._auth_logout, methods=["POST"]),
            Route("/job/{job_id:path}", self._job_detail_page),
            Route("/worker/{worker_id:path}", self._worker_detail_page),
            Route("/bundles/{bundle_id:str}.zip", self._bundle_download),
            Route("/health", self._health),
            Mount(rpc_wsgi_app.path, app=rpc_app),
            static_files_mount(),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, request: Request) -> Response:
        session_token = request.query_params.get("session_token")
        if session_token and self._auth_verifier is not None:
            try:
                self._auth_verifier.verify(session_token)
                response = RedirectResponse("/", status_code=302)
                response.set_cookie(
                    SESSION_COOKIE,
                    session_token,
                    httponly=True,
                    samesite="strict",
                    secure=request.url.scheme == "https",
                    path="/",
                )
                return response
            except ValueError:
                pass
        return HTMLResponse(html_shell("Iris Controller", "controller"))

    def _job_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Job Detail", "controller"))

    def _worker_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Worker Detail", "controller"))

    def _auth_config(self, request: Request) -> JSONResponse:
        """Unauthenticated endpoint telling the frontend whether auth is required."""
        has_session = SESSION_COOKIE in request.cookies
        return JSONResponse(
            {
                "auth_enabled": self._auth_provider is not None,
                "provider": self._auth_provider,
                "has_session": has_session,
            }
        )

    async def _auth_session(self, request: Request) -> JSONResponse:
        """Set auth cookie from bearer token."""
        body = await request.json()
        token = body.get("token", "").strip()
        if not token:
            return JSONResponse({"error": "token required"}, status_code=400)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )
        return response

    async def _auth_logout(self, request: Request) -> JSONResponse:
        """Clear auth cookie."""
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    def _health(self, _request: Request) -> JSONResponse:
        """Health check endpoint for controller availability."""
        return JSONResponse({"status": "ok"})

    def _bundle_download(self, request: Request) -> Response:
        # TODO(#3291): Add bearer token auth once Kubernetes init-containers
        # support Authorization headers. Currently bundle IDs are SHA-256 hashes
        # (256 bits of entropy) serving as capability URLs. Workers and K8s
        # init-containers fetch via stdlib urlopen with no auth header support.
        bundle_id = request.path_params["bundle_id"]
        try:
            data = self._service.bundle_zip(bundle_id)
        except FileNotFoundError:
            return Response(f"Bundle not found: {bundle_id}", status_code=404)
        return Response(data, media_type="application/zip")


class ProxyControllerDashboard:
    """Dashboard that proxies RPC calls to a remote Iris controller.

    Serves the same web UI locally but forwards all Connect RPC requests
    to an upstream controller at the given URL. Useful for viewing a remote
    controller's state without running a local controller instance.
    """

    def __init__(
        self,
        upstream_url: str,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._upstream_url = upstream_url.rstrip("/")
        self._host = host
        self._port = port
        self._client = httpx.AsyncClient(base_url=self._upstream_url, timeout=60.0)
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> Starlette:
        return self._app

    def _create_app(self) -> Starlette:
        routes = [
            Route("/", self._dashboard),
            Route("/job/{job_id:path}", self._job_detail_page),
            Route("/worker/{worker_id:path}", self._worker_detail_page),
            Route("/bundles/{bundle_id:str}.zip", self._proxy_bundle),
            Route("/health", self._health),
            Route("/iris.cluster.ControllerService/{method}", self._proxy_rpc, methods=["POST"]),
            static_files_mount(),
        ]
        return Starlette(routes=routes, on_shutdown=[self._client.aclose])

    def _proxy_html(self, dashboard_type: str) -> HTMLResponse:
        html = html_shell("Iris Controller (Proxy)", dashboard_type)
        banner = (
            '<div style="background:#f59e0b;color:#000;text-align:center;'
            "padding:4px 8px;font-size:13px;font-weight:600;position:fixed;"
            f'top:0;left:0;right:0;z-index:9999;">Proxy &rarr; {self._upstream_url}</div>'
            '<div style="height:28px;"></div>'
        )
        html = html.replace('<div id="app">', banner + '<div id="app">')
        return HTMLResponse(html)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return self._proxy_html("controller")

    def _job_detail_page(self, _request: Request) -> HTMLResponse:
        return self._proxy_html("controller")

    def _worker_detail_page(self, _request: Request) -> HTMLResponse:
        return self._proxy_html("controller")

    def _health(self, _request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _proxy_rpc(self, request: Request) -> Response:
        method = request.path_params["method"]
        body = await request.body()
        upstream_resp = await self._client.post(
            f"/iris.cluster.ControllerService/{method}",
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _proxy_bundle(self, request: Request) -> Response:
        bundle_id = request.path_params["bundle_id"]
        upstream_resp = await self._client.get(f"/bundles/{bundle_id}.zip")
        if upstream_resp.status_code != 200:
            return Response(upstream_resp.text, status_code=upstream_resp.status_code)
        return Response(upstream_resp.content, media_type="application/zip")
