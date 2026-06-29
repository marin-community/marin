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

import functools
import logging
import os
from urllib.parse import urlparse

import httpx
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import (
    NullAuthInterceptor,
    RequestAuthPolicy,
    extract_bearer_token,
    identity_scope,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Match, Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from iris.cluster.controller import endpoint_proxy
from iris.cluster.controller.backend import backend_descriptor
from iris.cluster.controller.endpoint_proxy import EndpointProxy
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
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
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.rpc.async_adapter import AsyncServiceAdapter
from iris.rpc.auth import SESSION_COOKIE, authorize_method
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceASGIApplication, EndpointServiceASGIApplication
from iris.rpc.interceptors import SLOW_RPC_THRESHOLD_MS, RequestTimingInterceptor
from iris.rpc.stats import RpcStatsCollector
from iris.rpc.stats_connect import StatsServiceASGIApplication
from iris.rpc.stats_service import RpcStatsService

logger = logging.getLogger(__name__)


def _scope_headers(scope: Scope) -> dict[str, str]:
    """Lowercase header dict from an ASGI scope."""
    return {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}


def _scope_client_address(scope: Scope) -> str | None:
    """Return the transport peer as ``host:port``, or None.

    This is uvicorn's ``scope["client"]`` — the genuine peer for a direct
    connection, or a forwarded value (port 0) when derived from
    ``X-Forwarded-For``. ``is_trusted_loopback`` relies on that distinction.
    """
    client = scope.get("client")
    if not client:
        return None
    return f"{client[0]}:{client[1]}"


async def _enforce_http_auth(
    scope: Scope,
    receive: Receive,
    send: Send,
    policy: RequestAuthPolicy,
) -> bool:
    """Resolve auth for an ASGI scope; on failure send a 401 and return False.

    On success, sets ``scope["auth_identity"]`` if a verified identity is
    present and returns True. Shared by ``_RouteAuthMiddleware`` (which
    runs against route-annotated requests) and ``_SubdomainProxyMiddleware``
    (which intercepts before any route can match).
    """
    headers = _scope_headers(scope)
    token = extract_bearer_token(headers, cookie_name=SESSION_COOKIE)
    try:
        identity = policy.resolve(
            token,
            client_address=_scope_client_address(scope),
            headers=headers,
        )
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

    def __init__(self, app: Starlette, policy: RequestAuthPolicy):
        self._app = app
        self._policy = policy
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

        for route in self._router.routes:
            if isinstance(route, Mount):
                if route.matches(scope)[0] != Match.NONE:
                    return "skip"
                continue
            if isinstance(route, Route):
                match_result, _ = route.matches(scope)
                if match_result == Match.FULL:
                    return getattr(route.endpoint, _AUTH_POLICY_ATTR, "deny")

        # No route matched — let Starlette handle 404.
        return "skip"

    async def _check_auth(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not await _enforce_http_auth(scope, receive, send, self._policy):
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


# Path scoping the session cookie. set/delete must use the same path or the
# browser will not match them, so both go through this constant.
SESSION_COOKIE_PATH = "/"


def _set_session_cookie(response: Response, token: str, request: Request) -> None:
    """Attach the session cookie with the standard security attributes.

    Centralizes the cookie flags so the bootstrap (redirect) and auth-session
    (fetch) paths cannot drift apart on security-sensitive attributes.
    """
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path=SESSION_COOKIE_PATH,
    )


class _DashboardAuthInterceptor:
    """RPC auth interceptor that uses the policy's authenticator stack — same
    policy as the HTTP middleware.

    Login and GetAuthInfo RPCs are always unauthenticated. All other RPCs go
    through ``policy.resolve`` (the ``[Jwt, IapAssertion?, Loopback]`` stack):
    - token present + valid → authenticated identity
    - token present + invalid → rejected
    - no token + loopback peer → anonymous/admin (loopback trust)
    - no token + optional → anonymous/admin fallback via NullAuthInterceptor
    - no token + required → rejected
    """

    def __init__(self, policy: RequestAuthPolicy):
        self._policy = policy
        self._null = NullAuthInterceptor(verifier=policy.verifier, cookie_name=SESSION_COOKIE)

    def _resolve_or_raise(self, ctx):
        """Returns (identity, fallback_to_null). Raises ConnectError on rejection."""

        headers = ctx.request_headers()
        token = extract_bearer_token(headers, cookie_name=SESSION_COOKIE)
        try:
            identity = self._policy.resolve(
                token,
                client_address=ctx.client_address(),
                headers=headers,
            )
        except ValueError as exc:
            if token is None:
                raise ConnectError(Code.UNAUTHENTICATED, str(exc)) from exc
            logger.warning("Authentication failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Authentication failed") from exc
        return identity

    def intercept_unary_sync(self, call_next, request, ctx):
        if ctx.method().name in _UNAUTHENTICATED_RPCS:
            return call_next(request, ctx)

        identity = self._resolve_or_raise(ctx)
        if identity is None:
            return self._null.intercept_unary_sync(call_next, request, ctx)

        authorize_method(identity, ctx.method().name)
        with identity_scope(identity):
            return call_next(request, ctx)

    async def intercept_unary(self, call_next, request, ctx):
        if ctx.method().name in _UNAUTHENTICATED_RPCS:
            return await call_next(request, ctx)

        identity = self._resolve_or_raise(ctx)
        if identity is None:
            return await self._null.intercept_unary(call_next, request, ctx)

        authorize_method(identity, ctx.method().name)
        with identity_scope(identity):
            return await call_next(request, ctx)


# DNS marker label that flags a Host as a per-endpoint subdomain. A request
# whose Host contains a ``proxy`` label routes the labels left of it to the
# endpoint proxy: ``<encoded_name>.proxy.<base>`` -> endpoint ``<encoded_name>``
# (with ``.`` -> ``/`` decoding, mirroring the path-style ``/proxy/<name>``
# route). Base-domain-agnostic: works for ``iris-dev.oa.dev``,
# ``iris.oa.dev``, or any other public host.
PROXY_HOST_LABEL = "proxy"

# Backward-compat for finelog clients built before logs moved behind the generic
# endpoint proxy: they resolve /system/log-server to the bare controller URL and
# POST to /finelog.logging.LogService/<method> directly. We forward those to the
# log server through the same EndpointProxy the dashboard uses, so no typed
# LogService forwarding mount is needed on the controller. The encoded name is
# the endpoint's wire name with the leading slash dropped and "/" -> ".".
_LOG_SERVICE_RPC_PREFIX = "finelog.logging.LogService"
_LOG_SERVER_PROXY_NAME = LOG_SERVER_ENDPOINT_NAME.strip("/").replace("/", ".")


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
    itself — running ``policy.resolve`` (the same authenticator stack as the
    route-level ``@requires_auth`` annotations) before dispatching to the proxy.

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
        auth_policy: RequestAuthPolicy = RequestAuthPolicy(),
    ):
        self._app = app
        self._endpoint_proxy = endpoint_proxy
        self._auth_policy = auth_policy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        encoded_name = _extract_proxy_subdomain(self._extract_host(scope))
        if encoded_name is None:
            await self._app(scope, receive, send)
            return

        if self._auth_policy.request_auth_enabled:
            if not await _enforce_http_auth(scope, receive, send, self._auth_policy):
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
        *,
        endpoint_service: EndpointServiceImpl | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_provider: str | None = None,
        auth_policy: RequestAuthPolicy = RequestAuthPolicy(),
    ):
        self._service = service
        # Defaults to the service's own backend; the two must share one instance
        # so a system endpoint registered on one is resolvable through the other.
        self._endpoint_service = endpoint_service or service.endpoint_service
        self._host = host
        self._port = port
        self._auth_provider = auth_provider
        self._auth_policy = auth_policy
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
        # Only the controller RPC chain feeds the stats collector. Finelog RPCs
        # use the generic endpoint proxy and are measured by the log server.
        include_tb = bool(os.environ.get("IRIS_DEBUG"))
        controller_timing = RequestTimingInterceptor(include_traceback=include_tb, collector=self._stats_collector)
        if self._auth_provider is not None and self._auth_policy.request_auth_enabled:
            auth_interceptor = _DashboardAuthInterceptor(self._auth_policy)
        else:
            # Null-auth mode: no provider configured. Verify worker tokens
            # when present but treat everything as anonymous/admin.
            auth_interceptor = NullAuthInterceptor(verifier=self._auth_policy.verifier, cookie_name=SESSION_COOKIE)
        controller_interceptors = [auth_interceptor, controller_timing]
        # @on_loop handlers run inline on the event loop; everything else
        # is dispatched to a thread by AsyncServiceAdapter.
        rpc_asgi_app = ControllerServiceASGIApplication(
            service=AsyncServiceAdapter(self._service),
            interceptors=controller_interceptors,
            compressions=IRIS_RPC_COMPRESSIONS,
        )

        # Leased service-discovery registry on its own wire surface. The legacy
        # ControllerService.{Register,Unregister,List}Endpoint RPCs forward into
        # the same backend in-process (see ControllerServiceImpl); new clients
        # call this service directly to learn their lease and renew.
        endpoint_rpc_app = EndpointServiceASGIApplication(
            service=AsyncServiceAdapter(self._endpoint_service),
            interceptors=controller_interceptors,
            compressions=IRIS_RPC_COMPRESSIONS,
        )

        # StatsService: reuses the auth interceptor (so non-admins can't read
        # sampled request previews) but skips RequestTimingInterceptor so the
        # stats endpoint itself doesn't pollute the numbers it reports.
        stats_app = StatsServiceASGIApplication(
            service=AsyncServiceAdapter(RpcStatsService(self._stats_collector)),
            interceptors=[auth_interceptor],
            compressions=IRIS_RPC_COMPRESSIONS,
        )

        self._endpoint_proxy = EndpointProxy(self._endpoint_service.resolve_endpoint)

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

        @requires_auth
        async def _legacy_log_service(request: Request) -> Response:
            # Forward pre-proxy clients' bare LogService calls to the log server
            # through the generic endpoint proxy (see _LOG_SERVER_PROXY_NAME).
            method = request.path_params["method"]
            return await self._endpoint_proxy.dispatch(
                request,
                encoded_name=_LOG_SERVER_PROXY_NAME,
                sub_path=f"{_LOG_SERVICE_RPC_PREFIX}/{method}",
                proxy_prefix="",
            )

        routes = [
            Route("/", self._dashboard),
            favicon_route(),
            Route("/auth/session_bootstrap", self._session_bootstrap),
            Route("/auth/config", self._auth_config),
            Route("/auth/session", self._auth_session, methods=["POST"]),
            Route("/auth/logout", self._auth_logout, methods=["POST"]),
            Route("/job/{job_id:path}", self._dashboard),
            Route("/worker/{worker_id:path}", self._dashboard),
            Route("/bundles/{bundle_id:str}.zip", self._bundle_download),
            Route("/blobs/{blob_id:str}", self._blob_download),
            Route("/health", self._health),
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
            Route(
                f"/{_LOG_SERVICE_RPC_PREFIX}/{{method}}",
                _legacy_log_service,
                methods=["POST"],
            ),
            Mount(rpc_asgi_app.path, app=rpc_asgi_app),
            Mount(endpoint_rpc_app.path, app=endpoint_rpc_app),
            Mount(stats_app.path, app=stats_app),
        ]
        routes.append(static_files_mount())

        app: Starlette | _RouteAuthMiddleware = Starlette(
            routes=routes,
            lifespan=on_shutdown(self._endpoint_proxy.close),
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
        if self._auth_policy.request_auth_enabled and self._auth_provider is not None:
            wrapped = _RouteAuthMiddleware(app, self._auth_policy)
        # Subdomain dispatch wraps everything: subdomain requests don't match
        # any Starlette route, so _RouteAuthMiddleware would default-allow
        # them. This middleware enforces auth itself before forwarding.
        wrapped = _SubdomainProxyMiddleware(
            wrapped,
            endpoint_proxy=self._endpoint_proxy,
            auth_policy=self._auth_policy,
        )
        return wrapped

    @public
    def _dashboard(self, _request: Request) -> HTMLResponse:
        # Vue Router handles client-side routing, so every SPA path serves the same shell.
        return HTMLResponse(html_shell("controller"))

    @public
    def _session_bootstrap(self, request: Request) -> Response:
        """Accept token via query param, set cookie, redirect to dashboard."""
        token = request.query_params.get("token", "")
        if not token or self._auth_policy.verifier is None:
            return RedirectResponse("/", status_code=302)
        try:
            self._auth_policy.verifier.verify(token)
        except ValueError:
            return JSONResponse({"error": "invalid token"}, status_code=401)
        response = RedirectResponse("/", status_code=302)
        _set_session_cookie(response, token, request)
        return response

    @public
    def _auth_config(self, request: Request) -> JSONResponse:
        """Unauthenticated endpoint telling the frontend whether auth is required."""
        has_session = SESSION_COOKIE in request.cookies
        descriptor = backend_descriptor(self._service.provider)
        return JSONResponse(
            {
                "auth_enabled": self._auth_provider is not None,
                "provider": self._auth_provider,
                "has_session": has_session,
                "backend": {
                    "name": descriptor.name,
                    "capabilities": descriptor.capabilities,
                },
                "optional": self._auth_policy.optional,
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
        if self._auth_policy.verifier is not None:
            try:
                self._auth_policy.verifier.verify(token)
            except ValueError:
                return JSONResponse({"error": "invalid token"}, status_code=401)
        response = JSONResponse({"ok": True})
        _set_session_cookie(response, token, request)
        return response

    @public
    async def _auth_logout(self, request: Request) -> JSONResponse:
        """Clear auth cookie."""
        if not _check_csrf(request):
            return JSONResponse({"error": "CSRF check failed"}, status_code=403)
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path=SESSION_COOKIE_PATH)
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
        # Vue Router handles client-side routing, so every SPA path serves the same shell.
        routes = [
            Route("/", self._dashboard),
            favicon_route(),
            Route("/job/{job_id:path}", self._dashboard),
            Route("/worker/{worker_id:path}", self._dashboard),
            Route(
                "/bundles/{bundle_id:str}.zip",
                functools.partial(
                    self._proxy_get, param="bundle_id", upstream="/bundles/{}.zip", media_type="application/zip"
                ),
            ),
            Route(
                "/blobs/{blob_id:str}",
                functools.partial(
                    self._proxy_get, param="blob_id", upstream="/blobs/{}", media_type="application/octet-stream"
                ),
            ),
            Route("/health", self._health),
            Route("/auth/{path:path}", self._proxy_auth),
            Route(
                "/iris.cluster.ControllerService/{method}",
                functools.partial(self._proxy_rpc_post, service="iris.cluster.ControllerService"),
                methods=["POST"],
            ),
            Route("/proxy/{path:path}", self._proxy_endpoint, methods=list(endpoint_proxy.ALLOWED_METHODS)),
            static_files_mount(),
        ]

        return Starlette(routes=routes, lifespan=on_shutdown(self._client.aclose))

    def _dashboard(self, _request: Request) -> HTMLResponse:
        html = html_shell("controller")
        banner = (
            '<div style="background:#f59e0b;color:#000;text-align:center;'
            "padding:4px 8px;font-size:13px;font-weight:600;position:fixed;"
            f'top:0;left:0;right:0;z-index:9999;">Proxy &rarr; {self._upstream_url}</div>'
            '<div style="height:28px;"></div>'
        )
        html = html.replace('<div id="app">', banner + '<div id="app">')
        return HTMLResponse(html)

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

    async def _proxy_rpc_post(self, request: Request, *, service: str) -> Response:
        """Forward a Connect-RPC POST to ``/<service>/<method>`` on the upstream."""
        method = request.path_params["method"]
        body = await request.body()
        upstream_resp = await self._client.post(
            f"/{service}/{method}",
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _proxy_endpoint(self, request: Request) -> Response:
        """Forward generic ``/proxy/<endpoint>/<sub_path>`` requests upstream.

        The dashboard's stats panels (live resource usage, status text, profile
        history) reach the bundled log server through
        ``/proxy/system.log-server/finelog.stats.StatsService/...``. The upstream
        controller already exposes the endpoint proxy at the same path, so we pass
        the request through verbatim (method, body, query, content-type).
        """
        path = request.path_params["path"]
        query = f"?{request.url.query}" if request.url.query else ""
        upstream_resp = await self._client.request(
            request.method,
            f"/proxy/{path}{query}",
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _proxy_get(self, request: Request, *, param: str, upstream: str, media_type: str) -> Response:
        """Forward a GET for a single path param to ``upstream`` (a format string)."""
        upstream_resp = await self._client.get(upstream.format(request.path_params[param]))
        if upstream_resp.status_code != 200:
            return Response(upstream_resp.text, status_code=upstream_resp.status_code)
        return Response(upstream_resp.content, media_type=media_type)
