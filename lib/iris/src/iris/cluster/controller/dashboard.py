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
- The native listener parses and authenticates every endpoint proxy request.
  Python sees only normalized federation decisions: received-job ownership
  checks and outgoing peer-token minting.
- One RequestAuthPolicy chain covers the Python RPC and HTTP routes: RPC mounts
  use PolicyAuthInterceptor and HTTP routes use RouteAuthMiddleware. Null-auth
  is a permissive chain, not a bypass.
- HTML shell routes are public — they contain no data, just the SPA skeleton.
- Bundle downloads use capability URLs (SHA-256 hash = 256 bits of entropy).
- Auth endpoints (/auth/*) handle session management (CSRF-protected).
- Each route handler is annotated @public or @requires_auth; an unannotated
  route is denied, so forgetting to annotate a new route is a safe failure.
"""

import functools
import logging
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from rigging.server_auth import (
    PolicyAuthInterceptor,
    RequestAuthPolicy,
    RouteAuthMiddleware,
    extract_bearer_token,
    public,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp

from iris.cluster.controller.auth import JwtTokenManager
from iris.cluster.controller.backend import backend_descriptor
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_proxy import FederatedEndpointHandoff
from iris.cluster.controller.native_proxy import (
    DECISION_SECRET_HEADER,
    DEFAULT_PROXY_TIMEOUT_SECONDS,
    PROXY_DECISION_PATH,
    PROXY_METHODS,
    PROXY_PREFIX_HEADER,
    PROXY_TIMEOUT_HEADER,
    UPSTREAM_AUTHORIZATION_HEADER,
    UPSTREAM_URL_HEADER,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import (
    favicon_route,
    html_shell,
    on_shutdown,
    static_files_mount,
)
from iris.cluster.types import JobName
from iris.rpc.async_adapter import AsyncServiceAdapter
from iris.rpc.auth import SESSION_COOKIE, authorize_method
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceASGIApplication, EndpointServiceASGIApplication
from iris.rpc.interceptors import SLOW_RPC_THRESHOLD_MS, RequestTimingInterceptor
from iris.rpc.stats import RpcStatsCollector
from iris.rpc.stats_connect import StatsServiceASGIApplication
from iris.rpc.stats_service import RpcStatsService

logger = logging.getLogger(__name__)

FederationOwnerCheck = Callable[[JobName, str], bool]


@dataclass(frozen=True, slots=True)
class _FederationDecision:
    direction: str
    encoded_name: str
    sub_path: str
    query: str
    proxy_prefix: str
    peer_id: str
    task_id: str | None
    local_upstream: str | None
    timeout_seconds: float | None


def _request_is_authenticated(policy: RequestAuthPolicy, request: Request) -> bool:
    headers = dict(request.headers)
    token = extract_bearer_token(headers, cookie_name=SESSION_COOKIE)
    client = request.client
    try:
        policy.resolve(
            token,
            client_address=f"{client.host}:{client.port}" if client else None,
            headers=headers,
        )
    except ValueError:
        return False
    return True


# Every control RPC is authenticated: users reach the controller only through IAP,
# which authenticates each request at the edge. No RPC is exempt from the policy.
_UNAUTHENTICATED_RPCS: frozenset[str] = frozenset()


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
        auth_provider: str | None = None,
        auth_policy: RequestAuthPolicy = RequestAuthPolicy(),
        jwt_manager: JwtTokenManager | None = None,
        federated_handoff: FederatedEndpointHandoff | None = None,
        federation_owner_check: FederationOwnerCheck | None = None,
        proxy_decision_secret: str | None = None,
    ):
        self._service = service
        # Defaults to the service's own backend; the two must share one instance
        # so a system endpoint registered on one is resolvable through the other.
        self._endpoint_service = endpoint_service or service.endpoint_service
        self._auth_provider = auth_provider
        self._auth_policy = auth_policy
        # The signing authority, for serving public keys at /.well-known/jwks.json
        # (None when the controller has no auth configured, so no signer exists).
        self._jwt_manager = jwt_manager
        # Forwards a /proxy request for a federated (remote) endpoint to the peer that
        # owns it; None on a cluster with no federation peers (no remote endpoints).
        self._federated_handoff = federated_handoff
        # Authorizes an inbound federation-peer's /proxy by job ownership; set on a
        # cluster that receives federation, None otherwise.
        self._federation_owner_check = federation_owner_check
        self._proxy_decision_secret = proxy_decision_secret
        # In-process RPC statistics. Fed by RequestTimingInterceptor on the
        # ControllerService chain only; LogService's chatty FetchLogs traffic
        # would dominate the numbers if included.
        self._stats_collector = RpcStatsCollector(slow_threshold_ms=SLOW_RPC_THRESHOLD_MS)
        self._app = self._create_app()

    @property
    def app(self) -> ASGIApp:
        return self._app

    @property
    def proxy_decision_secret(self) -> str:
        """Return the capability for private decisions, or fail when disabled."""
        if self._proxy_decision_secret is None:
            raise RuntimeError("native proxy decisions are not configured")
        return self._proxy_decision_secret

    def _create_app(self) -> ASGIApp:
        # Only the controller RPC chain feeds the stats collector. Finelog RPCs
        # use the generic endpoint proxy and are measured by the log server.
        include_tb = bool(os.environ.get("IRIS_DEBUG"))
        controller_timing = RequestTimingInterceptor(include_traceback=include_tb, collector=self._stats_collector)
        auth_interceptor = PolicyAuthInterceptor(
            self._auth_policy,
            cookie_name=SESSION_COOKIE,
            unauthenticated_methods=_UNAUTHENTICATED_RPCS,
            authorize=authorize_method,
        )
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

        @public
        async def _federation_decision(request: Request) -> Response:
            """Serve a normalized, Rust-authenticated federation handoff."""
            supplied_secret = request.headers.get(DECISION_SECRET_HEADER, "")
            if self._proxy_decision_secret is None or not secrets.compare_digest(
                supplied_secret,
                self._proxy_decision_secret,
            ):
                return JSONResponse({"error": "route not found"}, status_code=404)

            decision = _FederationDecision(**await request.json())
            headers = {
                PROXY_PREFIX_HEADER: decision.proxy_prefix,
                PROXY_TIMEOUT_HEADER: str(decision.timeout_seconds or DEFAULT_PROXY_TIMEOUT_SECONDS),
            }
            if decision.direction == "inbound":
                if (
                    decision.task_id is None
                    or decision.local_upstream is None
                    or self._federation_owner_check is None
                    or not self._federation_owner_check(
                        JobName.from_wire(decision.task_id).root_job,
                        decision.peer_id,
                    )
                ):
                    return JSONResponse({"error": "peer not authorized for this endpoint"}, status_code=403)
                headers[UPSTREAM_URL_HEADER] = decision.local_upstream
                return Response(status_code=204, headers=headers)

            if decision.direction != "outbound" or self._federated_handoff is None:
                return JSONResponse({"error": "federation proxy unavailable"}, status_code=502)
            target = self._federated_handoff.target(
                peer_id=decision.peer_id,
                encoded_name=decision.encoded_name,
                sub_path=decision.sub_path,
                query=decision.query,
            )
            if target is None:
                return JSONResponse({"error": f"Peer '{decision.peer_id}' unavailable"}, status_code=502)
            headers[UPSTREAM_URL_HEADER] = target.upstream_url
            headers[UPSTREAM_AUTHORIZATION_HEADER] = target.authorization
            return Response(status_code=204, headers=headers)

        routes = [
            Route("/", self._dashboard),
            favicon_route(),
            Route("/auth/config", self._auth_config),
            Route("/auth/session", self._auth_session, methods=["POST"]),
            Route("/auth/logout", self._auth_logout, methods=["POST"]),
            Route("/job/{job_id:path}", self._dashboard),
            Route("/worker/{worker_id:path}", self._dashboard),
            Route("/bundles/{bundle_id:str}.zip", self._bundle_download),
            Route("/blobs/{blob_id:str}", self._blob_download),
            Route("/health", self._health),
            Route("/.well-known/jwks.json", self._jwks),
            Route(PROXY_DECISION_PATH, _federation_decision, methods=["POST"]),
            Mount(rpc_asgi_app.path, app=rpc_asgi_app),
            Mount(endpoint_rpc_app.path, app=endpoint_rpc_app),
            Mount(stats_app.path, app=stats_app),
        ]
        routes.append(static_files_mount())

        app = Starlette(routes=routes)
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
        return RouteAuthMiddleware(app, self._auth_policy, cookie_name=SESSION_COOKIE)

    @public
    def _dashboard(self, _request: Request) -> HTMLResponse:
        # Vue Router handles client-side routing, so every SPA path serves the same shell.
        return HTMLResponse(html_shell("controller"))

    @public
    def _jwks(self, _request: Request) -> JSONResponse:
        """Public JWKS (this controller's current + retained-previous public keys).

        Public keys only — safe to serve unauthenticated. A federated finelog or
        peer resolves this controller's verification key by ``kid`` from here (or
        from an inline copy in its trust config). Empty when no signer exists (the
        controller has no auth configured).
        """
        if self._jwt_manager is None:
            return JSONResponse({"keys": []})
        return JSONResponse(self._jwt_manager.public_jwks())

    @public
    def _auth_config(self, request: Request) -> JSONResponse:
        """Report whether auth is required and whether this request is authenticated.

        Public endpoint the frontend reads before rendering to decide whether to
        show the login page. ``authenticated`` resolves the request through the
        same policy the RPC surface enforces, so a request carrying any accepted
        credential — a session cookie, a bearer token, or the signed IAP edge
        header — is reported as authenticated.
        """
        authenticated = _request_is_authenticated(self._auth_policy, request)
        descriptors = {bid: backend_descriptor(b) for bid, b in self._service.backends.items()}
        union_capabilities = sorted({cap for d in descriptors.values() for cap in d.capabilities})
        representative = backend_descriptor(self._service.provider)
        return JSONResponse(
            {
                "auth_enabled": self._auth_provider is not None,
                "provider": self._auth_provider,
                "authenticated": authenticated,
                # Union of every backend's capabilities gates which tabs the dashboard shows.
                "capabilities": union_capabilities,
                "backends": [
                    {"id": bid, "name": d.name, "capabilities": d.capabilities} for bid, d in descriptors.items()
                ],
                # Representative backend for the single-backend frontend path.
                "backend": {
                    "name": representative.name,
                    "capabilities": representative.capabilities,
                },
                "optional": self._auth_policy.allows_anonymous,
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
            Route("/proxy/{path:path}", self._proxy_endpoint, methods=list(PROXY_METHODS)),
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
