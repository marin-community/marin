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
from urllib.parse import urlparse

import httpx
from finelog.client import LogServiceProxy
from finelog.rpc.logging_connect import LogServiceWSGIApplication
from finelog.server import LogServiceImpl
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from iris.cluster.controller import endpoint_proxy
from iris.cluster.controller.actor_proxy import PROXY_ROUTE, ActorProxy
from iris.cluster.controller.endpoint_proxy import EndpointProxy
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import (
    _AUTH_POLICY_ATTR,
    favicon_route,
    html_shell,
    on_shutdown,
    public,
    requires_auth,
    static_files_mount,
)
from iris.rpc.auth import SESSION_COOKIE, NullAuthInterceptor, TokenVerifier, extract_bearer_token, resolve_auth
from iris.rpc.controller_connect import ControllerServiceWSGIApplication
from iris.rpc.interceptors import SLOW_RPC_THRESHOLD_MS, ConcurrencyLimitInterceptor, RequestTimingInterceptor
from iris.rpc.stats import RpcStatsCollector
from iris.rpc.stats_connect import StatsServiceWSGIApplication
from iris.rpc.stats_service import RpcStatsService

logger = logging.getLogger(__name__)


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


async def _enforce_http_auth(
    scope: Scope,
    receive: Receive,
    send: Send,
    verifier: TokenVerifier,
    optional: bool,
) -> bool:
    """Resolve auth for an ASGI scope; on failure send a 401 and return False.

    On success, sets ``scope["auth_identity"]`` if a verified identity is
    present and returns True. Shared by ``_RouteAuthMiddleware`` (which
    runs against route-annotated requests) and ``_SubdomainProxyMiddleware``
    (which intercepts before any route can match).
    """
    token = _extract_token_from_scope(scope)
    try:
        identity = resolve_auth(token, verifier, optional)
    except ValueError:
        response = JSONResponse({"error": "authentication required"}, status_code=401)
        await response(scope, receive, send)
        return False
    if identity is not None:
        scope["auth_identity"] = identity
    return True


class _RouteAuthMiddleware:
    """ASGI middleware that enforces per-route auth policy annotations.

    Looks up the matched Starlette route's endpoint function and checks its
    @public / @requires_auth annotation. Routes without an annotation are
    denied (default-deny). RPC Mount routes and static file mounts are
    skipped (they have their own auth).

    Uses resolve_auth() — the same policy function as the gRPC interceptor —
    so HTTP and gRPC layers agree on allow/deny for every token state.
    """

    def __init__(self, app: Starlette, verifier: TokenVerifier, optional: bool = False):
        self._app = app
        self._verifier = verifier
        self._optional = optional
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
        if not await _enforce_http_auth(scope, receive, send, self._verifier, self._optional):
            return
        await self._app(scope, receive, send)


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


class _DashboardAuthInterceptor:
    """RPC auth interceptor that uses resolve_auth() — same policy as HTTP middleware.

    Login and GetAuthInfo RPCs are always unauthenticated. All other RPCs go
    through resolve_auth(token, verifier, optional) which:
    - token present + valid → authenticated identity
    - token present + invalid → rejected
    - no token + optional → anonymous/admin fallback via NullAuthInterceptor
    - no token + required → rejected
    """

    def __init__(self, verifier: TokenVerifier, optional: bool = False):
        self._verifier = verifier
        self._optional = optional
        self._null = NullAuthInterceptor(verifier=verifier)

    def intercept_unary_sync(self, call_next, request, ctx):
        from iris.rpc.auth import _verified_identity

        if ctx.method().name in _UNAUTHENTICATED_RPCS:
            return call_next(request, ctx)

        token = extract_bearer_token(ctx.request_headers())
        try:
            identity = resolve_auth(token, self._verifier, self._optional)
        except ValueError as exc:
            from connectrpc.code import Code
            from connectrpc.errors import ConnectError

            if token is None:
                raise ConnectError(Code.UNAUTHENTICATED, str(exc)) from exc
            logger.warning("Authentication failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Authentication failed") from exc

        if identity is None:
            # Optional mode, no token — anonymous fallback.
            return self._null.intercept_unary_sync(call_next, request, ctx)

        reset_token = _verified_identity.set(identity)
        try:
            return call_next(request, ctx)
        finally:
            _verified_identity.reset(reset_token)


# DNS marker label that flags a Host as a per-endpoint subdomain. A request
# whose Host contains a ``proxy`` label routes the labels left of it to the
# endpoint proxy: ``<encoded_name>.proxy.<base>`` -> endpoint ``<encoded_name>``
# (with ``.`` -> ``/`` decoding, mirroring the path-style ``/proxy/<name>``
# route). Base-domain-agnostic: works for ``iris-dev.oa.dev``,
# ``iris.oa.dev``, or any other public host.
PROXY_HOST_LABEL = "proxy"


def _extract_proxy_subdomain(host: str) -> str | None:
    """Return the encoded endpoint name from a Host header, or None.

    Splits on ``.`` and looks for ``proxy`` as a label. Everything to the
    left of that label (rejoined with ``.``) is the encoded name.
    """
    if not host:
        return None
    bare = host.split(",", 1)[0].split(":", 1)[0].strip().lower()
    labels = bare.split(".")
    try:
        idx = labels.index(PROXY_HOST_LABEL)
    except ValueError:
        return None
    if idx == 0:
        return None
    return ".".join(labels[:idx])


class _SubdomainProxyMiddleware:
    """Dispatch ``<encoded_name>.proxy.<base>`` requests to the endpoint proxy.

    Subdomain requests don't match any Starlette route on the inner app,
    so :class:`_RouteAuthMiddleware`'s default-allow-on-no-route would
    leave them unauthenticated. This middleware therefore enforces auth
    itself — running ``resolve_auth(token, verifier, optional)`` with the
    same policy as the route-level ``@requires_auth`` annotations before
    dispatching to the proxy.

    Hosts without a ``proxy`` label pass through to the wrapped app
    unchanged.

    The encoded name (everything left of the ``proxy`` label) is decoded
    by the proxy using the same ``.`` -> ``/`` rule as the path-style
    route, so ``user.jobX.dash.proxy.iris-dev.oa.dev`` resolves to
    ``/user/jobX/dash``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        endpoint_proxy: EndpointProxy,
        auth_verifier: TokenVerifier | None = None,
        auth_optional: bool = False,
    ):
        self._app = app
        self._endpoint_proxy = endpoint_proxy
        self._auth_verifier = auth_verifier
        self._auth_optional = auth_optional

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self._app(scope, receive, send)

        encoded_name = _extract_proxy_subdomain(self._extract_host(scope))
        if encoded_name is None:
            return await self._app(scope, receive, send)

        if self._auth_verifier is not None:
            if not await _enforce_http_auth(scope, receive, send, self._auth_verifier, self._auth_optional):
                return

        request = Request(scope, receive=receive)
        response = await self._endpoint_proxy.dispatch(
            request,
            encoded_name=encoded_name,
            sub_path=request.url.path.lstrip("/"),
            proxy_prefix="",
        )
        await response(scope, receive, send)

    @staticmethod
    def _extract_host(scope: Scope) -> str:
        """Return the raw public-facing host header value.

        Trusts ``X-Forwarded-Host`` since uvicorn is configured with
        ``forwarded_allow_ips="*"``; the controller's only ingress is the
        IAP proxy.
        """
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        return headers.get("x-forwarded-host") or headers.get("host", "")


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
        log_service: LogServiceImpl | LogServiceProxy,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_verifier: TokenVerifier | None = None,
        auth_provider: str | None = None,
        auth_optional: bool = False,
    ):
        self._service = service
        self._log_service = log_service
        self._host = host
        self._port = port
        self._auth_verifier = auth_verifier
        self._auth_provider = auth_provider
        self._auth_optional = auth_optional
        # In-process RPC statistics. Fed by RequestTimingInterceptor on the
        # ControllerService chain only; LogService's chatty FetchLogs traffic
        # would dominate the numbers if included.
        self._stats_collector = RpcStatsCollector(slow_threshold_ms=SLOW_RPC_THRESHOLD_MS)
        self._app = self._create_app()

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> ASGIApp:
        return self._app

    def _create_app(self) -> ASGIApp:
        # Two timing interceptors: only the controller chain feeds the stats
        # collector, so the panel stays a clean view of ControllerService
        # traffic. The log server runs in a subprocess and will gain its own
        # collector separately; the controller-side LogService mount is a
        # legacy proxy whose forwarded calls would just add noise here.
        include_tb = bool(os.environ.get("IRIS_DEBUG"))
        controller_timing = RequestTimingInterceptor(include_traceback=include_tb, collector=self._stats_collector)
        log_timing = RequestTimingInterceptor(include_traceback=include_tb)
        if self._auth_provider is not None and self._auth_verifier is not None:
            auth_interceptor = _DashboardAuthInterceptor(self._auth_verifier, optional=self._auth_optional)
        else:
            # Null-auth mode: no provider configured. Verify worker tokens
            # when present but treat everything as anonymous/admin.
            auth_interceptor = NullAuthInterceptor(verifier=self._auth_verifier)
        controller_interceptors = [auth_interceptor, controller_timing]
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service, interceptors=controller_interceptors)

        # StatsService: reuses the auth interceptor (so non-admins can't read
        # sampled request previews) but skips RequestTimingInterceptor so the
        # stats endpoint itself doesn't pollute the numbers it reports.
        stats_wsgi_app = StatsServiceWSGIApplication(
            service=RpcStatsService(self._stats_collector),
            interceptors=[auth_interceptor],
        )
        stats_app = WSGIMiddleware(stats_wsgi_app)

        # PushLogs is kept on the controller as a forwarding proxy: older workers
        # cached /system/log-server -> controller URL, so we must accept their
        # pushes and forward them to the real log server. Forwarding happens
        # transparently because self._log_service is a LogServiceProxy whose
        # push_logs() calls the remote LogService over RPC.
        # Cap concurrent FetchLogs RPCs to avoid evicting the page cache with
        # parallel DuckDB scans. See duckdb_store.py for working-set caps.
        log_interceptors = [auth_interceptor, log_timing, ConcurrencyLimitInterceptor({"FetchLogs": 4})]
        log_wsgi_app = LogServiceWSGIApplication(service=self._log_service, interceptors=log_interceptors)
        log_app = WSGIMiddleware(log_wsgi_app)

        # Backward-compat: old clients call ControllerService/FetchLogs (removed
        # from the proto in the LogService migration).  Register the already-
        # intercepted LogService FetchLogs endpoint under the old path so the
        # Connect protocol handles encoding, compression, and auth correctly.
        # The LogService now lives under the finelog.logging proto package.
        _LOG_FETCH_ENDPOINT = "/finelog.logging.LogService/FetchLogs"
        _LOG_PUSH_ENDPOINT = "/finelog.logging.LogService/PushLogs"
        _COMPAT_FETCH_ENDPOINT = "/iris.cluster.ControllerService/FetchLogs"
        rpc_wsgi_app._endpoints[_COMPAT_FETCH_ENDPOINT] = log_wsgi_app._endpoints[_LOG_FETCH_ENDPOINT]

        # Backward-compat: clients/workers built before the finelog lift call
        # /iris.logging.LogService/{FetchLogs,PushLogs}. Wire bytes are identical
        # to /finelog.logging.LogService/*, so mount the same WSGI app at the
        # legacy prefix and register relative-path aliases.
        # connectrpc dispatch (_server_sync.py:206-210) first looks up PATH_INFO
        # directly; the existing /finelog.logging.LogService mount only hits via
        # the SCRIPT_NAME==self.path fallback. Adding relative keys lets the
        # first lookup succeed regardless of which mount handled the request.
        log_wsgi_app._endpoints["/FetchLogs"] = log_wsgi_app._endpoints[_LOG_FETCH_ENDPOINT]
        log_wsgi_app._endpoints["/PushLogs"] = log_wsgi_app._endpoints[_LOG_PUSH_ENDPOINT]
        _LEGACY_LOG_SERVICE_PATH = "/iris.logging.LogService"

        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        self._actor_proxy = ActorProxy(self._service._store)

        def _resolve_endpoint(name: str) -> str | None:
            # Task-registered endpoints live in the SQL store; system endpoints
            # (``/system/...``) live in an in-memory dict on the service.
            # Same fallback order as ListEndpoints' system-endpoint branch.
            row = self._service._store.endpoints.resolve(name)
            if row is not None:
                return row.address
            return self._service._system_endpoints.get(name)

        self._endpoint_proxy = EndpointProxy(_resolve_endpoint)

        @requires_auth
        async def _proxy_actor_rpc(request: Request) -> Response:
            return await self._actor_proxy.handle(request)

        @requires_auth
        async def _proxy_endpoint(request: Request) -> Response:
            name = request.path_params["endpoint_name"]
            return await self._endpoint_proxy.dispatch(
                request,
                encoded_name=name,
                sub_path=request.path_params["sub_path"],
                proxy_prefix=f"/proxy/{name}",
            )

        @requires_auth
        async def _proxy_endpoint_redirect(request: Request) -> Response:
            # ``/proxy/<name>`` (no trailing slash, no sub_path) needs a
            # redirect to ``/proxy/<name>/`` so upstream apps resolve their
            # relative assets correctly. We can't use Starlette's built-in
            # redirect_slashes=True: that builds an *absolute* Location from
            # scope["server"] / the Host header, which behind IAP is the
            # internal bind IP. A path-only Location resolves against the
            # browser's current origin, so no internal address leaks.
            name = request.path_params["endpoint_name"]
            query = f"?{request.url.query}" if request.url.query else ""
            return RedirectResponse(f"/proxy/{name}/{query}", status_code=307)

        routes = [
            Route("/", self._dashboard),
            favicon_route(),
            Route("/auth/session_bootstrap", self._session_bootstrap),
            Route("/auth/config", self._auth_config),
            Route("/auth/session", self._auth_session, methods=["POST"]),
            Route("/auth/logout", self._auth_logout, methods=["POST"]),
            Route("/job/{job_id:path}", self._job_detail_page),
            Route("/worker/{worker_id:path}", self._worker_detail_page),
            Route("/bundles/{bundle_id:str}.zip", self._bundle_download),
            Route("/blobs/{blob_id:str}", self._blob_download),
            Route("/health", self._health),
            Route(PROXY_ROUTE, _proxy_actor_rpc, methods=["POST"]),
            Route(
                "/proxy/{endpoint_name:str}",
                _proxy_endpoint_redirect,
                methods=list(endpoint_proxy.ALLOWED_METHODS),
            ),
            Route(
                endpoint_proxy.PROXY_ROUTE,
                _proxy_endpoint,
                methods=list(endpoint_proxy.ALLOWED_METHODS),
            ),
            Mount(log_wsgi_app.path, app=log_app),
            Mount(_LEGACY_LOG_SERVICE_PATH, app=log_app),
            Mount(rpc_wsgi_app.path, app=rpc_app),
            Mount(stats_wsgi_app.path, app=stats_app),
            static_files_mount(),
        ]

        app: Starlette | _RouteAuthMiddleware = Starlette(
            routes=routes,
            lifespan=on_shutdown(self._actor_proxy.close, self._endpoint_proxy.close),
        )
        # Starlette's default trailing-slash redirect builds an absolute
        # Location from ``scope["server"]`` (or the request's Host header).
        # Behind GCP IAP / a load balancer whose backend Host is the internal
        # bind IP, that absolute URL leaks ``http://10.x.x.x:10000/...`` back
        # to the browser — unreachable outside the VPC. Strict routing is
        # fine here: the SPA handles its own paths client-side and the API
        # surface is small enough that canonical URLs are easy to publish.
        # ``redirect_slashes`` is a Router attribute, not a Starlette ctor
        # kwarg, so we flip it after construction.
        app.router.redirect_slashes = False
        wrapped: ASGIApp = app
        if self._auth_verifier is not None and self._auth_provider is not None:
            wrapped = _RouteAuthMiddleware(app, self._auth_verifier, optional=self._auth_optional)
        # Subdomain dispatch wraps everything: subdomain requests don't match
        # any Starlette route, so _RouteAuthMiddleware would default-allow
        # them. This middleware enforces auth itself before forwarding.
        wrapped = _SubdomainProxyMiddleware(
            wrapped,
            endpoint_proxy=self._endpoint_proxy,
            auth_verifier=self._auth_verifier,
            auth_optional=self._auth_optional,
        )
        return wrapped

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

    @public
    def _blob_download(self, request: Request) -> Response:
        blob_id = request.path_params["blob_id"]
        try:
            data = self._service.blob_data(blob_id)
        except FileNotFoundError:
            return Response(f"Blob not found: {blob_id}", status_code=404)
        return Response(data, media_type="application/octet-stream")


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
            favicon_route(),
            Route("/job/{job_id:path}", self._job_detail_page),
            Route("/worker/{worker_id:path}", self._worker_detail_page),
            Route("/bundles/{bundle_id:str}.zip", self._proxy_bundle),
            Route("/blobs/{blob_id:str}", self._proxy_blob),
            Route("/health", self._health),
            Route("/auth/{path:path}", self._proxy_auth),
            Route("/iris.cluster.ControllerService/{method}", self._proxy_rpc, methods=["POST"]),
            Route("/finelog.logging.LogService/{method}", self._proxy_log_rpc, methods=["POST"]),
            static_files_mount(),
        ]

        return Starlette(routes=routes, lifespan=on_shutdown(self._client.aclose))

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

    async def _proxy_auth(self, request: Request) -> Response:
        path = request.path_params["path"]
        upstream_resp = await self._client.request(
            request.method,
            f"/auth/{path}",
            content=await request.body() if request.method in ("POST", "PUT") else None,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
        )

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

    async def _proxy_log_rpc(self, request: Request) -> Response:
        method = request.path_params["method"]
        body = await request.body()
        upstream_resp = await self._client.post(
            f"/finelog.logging.LogService/{method}",
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

    async def _proxy_blob(self, request: Request) -> Response:
        blob_id = request.path_params["blob_id"]
        upstream_resp = await self._client.get(f"/blobs/{blob_id}")
        if upstream_resp.status_code != 200:
            return Response(upstream_resp.text, status_code=upstream_resp.status_code)
        return Response(upstream_resp.content, media_type="application/octet-stream")
