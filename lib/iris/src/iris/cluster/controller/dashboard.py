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

Auth model:
- HTML shell routes are public — they contain no data, just the SPA skeleton.
- RPC routes have their own auth interceptor chain (AuthInterceptor / NullAuthInterceptor).
- Bundle downloads use capability URLs (SHA-256 hash = 256 bits of entropy).
- Auth endpoints (/auth/*) handle session management (CSRF-protected).
- Each route handler is annotated @public or @requires_auth. The middleware
  denies any route that lacks an annotation, so forgetting to annotate a new
  route is a safe failure.
"""

import logging
import os
from http.cookies import SimpleCookie
from collections.abc import Callable
from urllib.parse import urlparse

import httpx
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import html_shell, static_files_mount
from iris.rpc.auth import SESSION_COOKIE, AuthInterceptor, NullAuthInterceptor, TokenVerifier, extract_bearer_token
from iris.rpc.cluster_connect import ControllerServiceWSGIApplication
from iris.rpc.interceptors import RequestTimingInterceptor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route auth policy annotations
# ---------------------------------------------------------------------------

_AUTH_POLICY_ATTR = "_auth_policy"


def public(fn: Callable) -> Callable:
    """Mark a route handler as publicly accessible (no auth required)."""
    setattr(fn, _AUTH_POLICY_ATTR, "public")
    return fn


def requires_auth(fn: Callable) -> Callable:
    """Mark a route handler as requiring authentication via session cookie or Bearer token."""
    setattr(fn, _AUTH_POLICY_ATTR, "requires_auth")
    return fn


def _extract_token_from_scope(scope: Scope) -> str | None:
    """Extract auth token from ASGI scope (cookie or Authorization header)."""
    headers: dict[str, str] = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    cookie_header = headers.get("cookie", "")
    if not cookie_header:
        return None
    cookie = SimpleCookie(cookie_header)
    if SESSION_COOKIE in cookie:
        return cookie[SESSION_COOKIE].value
    return None


class _RouteAuthMiddleware:
    """ASGI middleware that enforces per-route auth policy annotations.

    Looks up the matched Starlette route's endpoint function and checks its
    @public / @requires_auth annotation. Routes without an annotation are
    denied (default-deny). RPC Mount routes and static file mounts are
    skipped (they have their own auth).
    """

    def __init__(self, app: Starlette, verifier: TokenVerifier):
        self._app = app
        self._verifier = verifier
        self._router = app.router

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self._app(scope, receive, send)

        policy = self._resolve_policy(scope)

        if policy == "public":
            return await self._app(scope, receive, send)

        if policy == "requires_auth":
            return await self._check_auth(scope, receive, send)

        # No policy (Mount for RPC/static, or unknown) — pass through.
        # RPC routes have their own interceptor; static mounts serve assets.
        if policy == "skip":
            return await self._app(scope, receive, send)

        # Default-deny: route exists but has no annotation.
        response = JSONResponse({"error": "authentication required"}, status_code=401)
        return await response(scope, receive, send)

    def _resolve_policy(self, scope: Scope) -> str:
        """Resolve the auth policy for the matched route."""
        from starlette.routing import Match

        for route in self._router.routes:
            if isinstance(route, Mount):
                if route.matches(scope)[0] != Match.NONE:
                    return "skip"
                continue
            if isinstance(route, Route):
                match_result, _ = route.matches(scope)
                if match_result == Match.FULL:
                    endpoint = route.endpoint
                    return getattr(endpoint, _AUTH_POLICY_ATTR, "deny")

        # No route matched — let Starlette handle 404.
        return "skip"

    async def _check_auth(self, scope: Scope, receive: Receive, send: Send) -> None:
        token = _extract_token_from_scope(scope)
        if token is None:
            response = JSONResponse({"error": "authentication required"}, status_code=401)
            return await response(scope, receive, send)
        try:
            identity = self._verifier.verify(token)
        except ValueError:
            response = JSONResponse({"error": "invalid session"}, status_code=401)
            return await response(scope, receive, send)
        scope["auth_identity"] = identity
        return await self._app(scope, receive, send)


_UNAUTHENTICATED_RPCS = {"Login", "GetAuthInfo"}


def _check_csrf(request: Request) -> bool:
    """Verify Origin header matches the request host for CSRF protection."""
    origin = request.headers.get("origin")
    if origin is None:
        referer = request.headers.get("referer")
        if referer is None:
            return False
        parsed = urlparse(referer)
        origin = f"{parsed.scheme}://{parsed.netloc}"

    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        proto = request.headers.get("x-forwarded-proto", "https")
        expected_origin = f"{proto}://{forwarded_host}"
    else:
        expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    return origin == expected_origin


class _SelectiveAuthInterceptor:
    """Auth interceptor that skips authentication for specific RPC methods."""

    def __init__(self, verifier: TokenVerifier):
        self._inner = AuthInterceptor(verifier)

    def intercept_unary_sync(self, call_next, request, ctx):
        if ctx.method().name in _UNAUTHENTICATED_RPCS:
            return call_next(request, ctx)
        return self._inner.intercept_unary_sync(call_next, request, ctx)


class _OptionalAuthInterceptor:
    """Auth interceptor for optional/gradual-adoption mode.

    Token-bearing requests are fully authenticated: valid tokens resolve to the
    real identity with its actual role; invalid tokens are rejected. Requests
    without a token fall through as anonymous/admin so existing unauthenticated
    clients keep working during the transition.

    Login and GetAuthInfo are always unauthenticated (same as mandatory mode).
    """

    def __init__(self, verifier: TokenVerifier):
        self._inner = AuthInterceptor(verifier)
        self._null = NullAuthInterceptor(verifier=verifier)

    def intercept_unary_sync(self, call_next, request, ctx):
        if ctx.method().name in _UNAUTHENTICATED_RPCS:
            return call_next(request, ctx)

        token = extract_bearer_token(ctx.request_headers())
        if token:
            # Token present — full auth (reject on failure).
            return self._inner.intercept_unary_sync(call_next, request, ctx)

        # No token — anonymous fallback for gradual adoption.
        return self._null.intercept_unary_sync(call_next, request, ctx)


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
        auth_optional: bool = False,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._auth_verifier = auth_verifier
        self._auth_provider = auth_provider
        self._auth_optional = auth_optional
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> ASGIApp:
        return self._app

    def _create_app(self) -> ASGIApp:
        interceptors = [RequestTimingInterceptor(include_traceback=bool(os.environ.get("IRIS_DEBUG")))]
        if self._auth_provider is not None and not self._auth_optional:
            interceptors.insert(0, _SelectiveAuthInterceptor(self._auth_verifier))
        elif self._auth_optional and self._auth_verifier is not None:
            # Optional mode: authenticate token-bearing requests fully (reject
            # invalid tokens), but allow unauthenticated requests through.
            interceptors.insert(0, _OptionalAuthInterceptor(self._auth_verifier))
        else:
            # Null-auth mode: no provider configured. Verify worker tokens
            # when present but treat everything as anonymous/admin.
            interceptors.insert(0, NullAuthInterceptor(verifier=self._auth_verifier))
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service, interceptors=interceptors)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/", self._dashboard),
            Route("/auth/session_bootstrap", self._session_bootstrap),
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
        app: Starlette | _RouteAuthMiddleware = Starlette(routes=routes)
        if self._auth_verifier is not None and self._auth_provider is not None and not self._auth_optional:
            app = _RouteAuthMiddleware(app, self._auth_verifier)
        return app

    @public
    def _dashboard(self, request: Request) -> Response:
        return HTMLResponse(html_shell("Iris Controller", "controller"))

    @public
    def _session_bootstrap(self, request: Request) -> Response:
        """Accept token via query param, set cookie, redirect to dashboard."""
        token = request.query_params.get("token", "")
        if not token or self._auth_verifier is None:
            return RedirectResponse("/", status_code=302)
        try:
            self._auth_verifier.verify(token)
        except ValueError:
            return JSONResponse({"error": "invalid token"}, status_code=401)
        response = RedirectResponse("/", status_code=302)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )
        return response

    @public
    def _job_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Job Detail", "controller"))

    @public
    def _worker_detail_page(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Worker Detail", "controller"))

    @public
    def _auth_config(self, request: Request) -> JSONResponse:
        """Unauthenticated endpoint telling the frontend whether auth is required."""
        has_session = SESSION_COOKIE in request.cookies
        provider_kind = "kubernetes" if self._service._controller.has_direct_provider else "worker"
        return JSONResponse(
            {
                "auth_enabled": self._auth_provider is not None,
                "provider": self._auth_provider,
                "has_session": has_session,
                "provider_kind": provider_kind,
                "optional": self._auth_optional,
            }
        )

    # Rate limiting is handled at the infrastructure layer via Cloudflare WAF rules.
    # See: https://developers.cloudflare.com/waf/rate-limiting-rules/
    @public
    async def _auth_session(self, request: Request) -> JSONResponse:
        """Set auth cookie from bearer token."""
        if not _check_csrf(request):
            return JSONResponse({"error": "CSRF check failed"}, status_code=403)
        body = await request.json()
        token = body.get("token", "").strip()
        if not token:
            return JSONResponse({"error": "token required"}, status_code=400)
        if self._auth_verifier is not None:
            try:
                self._auth_verifier.verify(token)
            except ValueError:
                return JSONResponse({"error": "invalid token"}, status_code=401)
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

    @public
    async def _auth_logout(self, request: Request) -> JSONResponse:
        """Clear auth cookie."""
        if not _check_csrf(request):
            return JSONResponse({"error": "CSRF check failed"}, status_code=403)
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @public
    def _health(self, _request: Request) -> JSONResponse:
        """Health check endpoint for controller availability."""
        return JSONResponse({"status": "ok"})

    @public
    def _bundle_download(self, request: Request) -> Response:
        # Bundle IDs are SHA-256 hashes (256 bits of entropy) serving as
        # capability URLs. Workers and K8s init-containers fetch via stdlib
        # urlopen with no auth header support.
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
