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
- One RequestAuthPolicy chain is applied everywhere, unconditionally: the
  RPC mounts through PolicyAuthInterceptor, the HTTP routes through
  RouteAuthMiddleware, and the /proxy routes through their per-endpoint
  access mode (_authorize_proxy). Null-auth is a permissive chain, not a
  bypass, so no surface branches on whether auth is "on".
- HTML shell routes are public — they contain no data, just the SPA skeleton.
- Bundle downloads use capability URLs (SHA-256 hash = 256 bits of entropy).
- Auth endpoints (/auth/*) handle session management (CSRF-protected).
- Each route handler is annotated @public or @requires_auth; an unannotated
  route is denied, so forgetting to annotate a new route is a safe failure.
"""

import functools
import logging
import os
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlparse

import httpx
from rigging.server_auth import (
    PolicyAuthInterceptor,
    RequestAuthPolicy,
    RouteAuthMiddleware,
    VerifiedIdentity,
    extract_bearer_token,
    public,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from iris.cluster.controller import endpoint_proxy
from iris.cluster.controller.auth import JwtTokenManager
from iris.cluster.controller.backend import backend_descriptor
from iris.cluster.controller.endpoint_proxy import EndpointProxy, FederatedEndpointProxy
from iris.cluster.controller.endpoint_service import EndpointServiceImpl, ResolvedEndpoint
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import (
    favicon_route,
    html_shell,
    on_shutdown,
    static_files_mount,
)
from iris.cluster.types import EndpointAccess, JobName
from iris.rpc.async_adapter import AsyncServiceAdapter
from iris.rpc.auth import FEDERATION_PEER_ROLE, SESSION_COOKIE, authorize_method
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceASGIApplication, EndpointServiceASGIApplication
from iris.rpc.interceptors import SLOW_RPC_THRESHOLD_MS, RequestTimingInterceptor
from iris.rpc.stats import RpcStatsCollector
from iris.rpc.stats_connect import StatsServiceASGIApplication
from iris.rpc.stats_service import RpcStatsService

logger = logging.getLogger(__name__)


def _resolve_request_identity(policy: RequestAuthPolicy, request: Request, token: str | None = None) -> VerifiedIdentity:
    """Resolve a Starlette request to a cluster identity via the auth policy.

    Lifts the bearer token from the Authorization header or session cookie when no
    explicit ``token`` is given, and forwards the peer address and headers the
    policy's authenticators need (the signed IAP header, CIDR, loopback). Raises
    ``ValueError`` when the request cannot be authenticated.
    """
    headers = dict(request.headers)
    if token is None:
        token = extract_bearer_token(headers, cookie_name=SESSION_COOKIE)
    client = request.client
    return policy.resolve(
        token,
        client_address=f"{client.host}:{client.port}" if client else None,
        headers=headers,
    )


# Answers whether ``peer_id`` handed this cluster the job rooted at ``root_job``
# (a RECEIVED federation handle). Gates a federation-peer's /proxy access to the
# endpoints of jobs it actually federated here. None on a cluster that receives no
# federation, where a federation-peer identity cannot arise.
FederationOwnerCheck = Callable[[JobName, str], bool]


def _authorize_federation_peer_proxy(
    resolved: ResolvedEndpoint | None,
    peer_id: str,
    owner_check: FederationOwnerCheck | None,
) -> Response | None:
    """Authorize a parent controller's forwarded /proxy for one of its federated jobs.

    The bearer proves the peer, not the end user: the parent already enforced the
    endpoint's access mode against the user before forwarding, so this cluster only
    confirms the endpoint belongs to a job this peer handed here. Scoped to the
    endpoint's root job so a trusted peer cannot reach another peer's or a local
    job's endpoint. ``/system/`` endpoints (no owning task) are never reachable this
    way.
    """
    if resolved is None or resolved.task_id is None:
        return JSONResponse({"error": "unknown endpoint"}, status_code=404)
    if owner_check is None or not owner_check(resolved.task_id.root_job, peer_id):
        return JSONResponse({"error": "peer not authorized for this endpoint"}, status_code=403)
    return None


def _authorize_proxy(
    request: Request,
    resolved: ResolvedEndpoint | None,
    policy: RequestAuthPolicy,
    *,
    token: str | None = None,
    federation_owner_check: FederationOwnerCheck | None = None,
) -> Response | None:
    """Authorize a ``/proxy`` request against its endpoint's access mode.

    Returns a deny ``Response`` (401/403) to send, or ``None`` when the request
    is allowed. ``resolved`` is the endpoint the request names (``None`` for an
    unknown name, which is treated as ``PRIVATE`` — the forwarding layer then
    404s). This is the *only* place an endpoint-scoped token is accepted.

    - ``LINK``: a scoped token must match this endpoint's wire name; a full
      cluster identity also passes.
    - ``PRIVATE`` (and unknown): a full cluster identity is required; a scoped
      token is rejected.
    - A ``federation-peer`` identity (a parent controller forwarding a federated
      endpoint's /proxy) is authorized by job ownership, not access mode — see
      :func:`_authorize_federation_peer_proxy`.

    ``token`` is the credential source: the URL-token fallback passes the token
    lifted from the path; otherwise the ``Authorization`` header / session
    cookie is used.
    """
    access = resolved.access if resolved is not None else EndpointAccess.ENDPOINT_ACCESS_PRIVATE
    try:
        identity = _resolve_request_identity(policy, request, token)
    except ValueError:
        return JSONResponse({"error": "authentication required"}, status_code=401)

    if identity.role == FEDERATION_PEER_ROLE:
        return _authorize_federation_peer_proxy(resolved, identity.user_id, federation_owner_check)

    scoped = identity.audience is not None
    if access == EndpointAccess.ENDPOINT_ACCESS_LINK:
        if scoped and identity.audience != resolved.name:
            return JSONResponse({"error": "token not valid for this endpoint"}, status_code=403)
        return None
    # PRIVATE (and unknown): full cluster identity only, never a scoped token.
    if scoped:
        return JSONResponse({"error": "endpoint-scoped token cannot access this endpoint"}, status_code=403)
    return None


class ProxyTargetResolver(Protocol):
    """What the proxy surfaces need from the endpoint service."""

    def resolve_proxy_target(self, encoded_name: str) -> ResolvedEndpoint | None: ...


def _resolve_and_authorize_proxy(
    request: Request,
    encoded_name: str,
    endpoint_service: ProxyTargetResolver,
    policy: RequestAuthPolicy,
    *,
    token: str | None = None,
    federation_owner_check: FederationOwnerCheck | None = None,
) -> tuple[ResolvedEndpoint | None, Response | None]:
    """Resolve a proxy request's target and authorize it against the endpoint's
    access mode. Returns ``(resolved, deny)``; send ``deny`` and stop when it is
    not None.
    """
    resolved = endpoint_service.resolve_proxy_target(encoded_name)
    deny = _authorize_proxy(request, resolved, policy, token=token, federation_owner_check=federation_owner_check)
    return resolved, deny


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

    Subdomain requests don't match any Starlette route on the inner app, so
    :class:`RouteAuthMiddleware` would pass them through unauthenticated. This
    middleware therefore applies the per-endpoint access check itself before
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
        endpoint_service: ProxyTargetResolver,
        auth_policy: RequestAuthPolicy = RequestAuthPolicy(),
        federation_owner_check: FederationOwnerCheck | None = None,
    ):
        self._app = app
        self._endpoint_proxy = endpoint_proxy
        self._endpoint_service = endpoint_service
        self._auth_policy = auth_policy
        self._federation_owner_check = federation_owner_check

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        encoded_name = _extract_proxy_subdomain(self._extract_host(scope))
        if encoded_name is None:
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        resolved, deny = _resolve_and_authorize_proxy(
            request,
            encoded_name,
            self._endpoint_service,
            self._auth_policy,
            federation_owner_check=self._federation_owner_check,
        )
        if deny is not None:
            await deny(scope, receive, send)
            return

        # Refuse an unknown or fail-closed-ambiguous name here rather than dial with
        # address=None (which would let EndpointProxy re-resolve and pick one of the
        # ambiguous rows), matching the path-style route's guard.
        if resolved is None:
            response: Response = JSONResponse({"error": f"No endpoint '{encoded_name}'"}, status_code=404)
            await response(scope, receive, send)
            return

        # Subdomain-style proxying of a federated endpoint is not supported: the
        # child rewrites redirects to its own /proxy/<name> path, which does not map
        # cleanly onto the bare-origin subdomain form. Use the path-style
        # /proxy/<name>/ URL for a remote endpoint instead. Fail closed rather than
        # serve broken navigation.
        if resolved.peer_id:
            response = JSONResponse(
                {"error": f"Endpoint '{encoded_name}' is federated; use its /proxy/{encoded_name}/ URL"},
                status_code=502,
            )
            await response(scope, receive, send)
            return

        response = await self._endpoint_proxy.dispatch(
            request,
            encoded_name=encoded_name,
            sub_path=request.url.path.lstrip("/"),
            proxy_prefix="",
            address=resolved.address,
            timeout_seconds=resolved.timeout_seconds,
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
        jwt_manager: JwtTokenManager | None = None,
        federated_proxy: FederatedEndpointProxy | None = None,
        federation_owner_check: FederationOwnerCheck | None = None,
    ):
        self._service = service
        # Defaults to the service's own backend; the two must share one instance
        # so a system endpoint registered on one is resolvable through the other.
        self._endpoint_service = endpoint_service or service.endpoint_service
        self._host = host
        self._port = port
        self._auth_provider = auth_provider
        self._auth_policy = auth_policy
        # The signing authority, for serving public keys at /.well-known/jwks.json
        # (None when the controller has no auth configured, so no signer exists).
        self._jwt_manager = jwt_manager
        # Forwards a /proxy request for a federated (remote) endpoint to the peer that
        # owns it; None on a cluster with no federation peers (no remote endpoints).
        self._federated_proxy = federated_proxy
        # Authorizes an inbound federation-peer's /proxy by job ownership; set on a
        # cluster that receives federation, None otherwise.
        self._federation_owner_check = federation_owner_check
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

        self._endpoint_proxy = EndpointProxy(self._endpoint_service.resolve_endpoint)

        async def _dispatch_resolved(
            request: Request, resolved: ResolvedEndpoint | None, *, encoded_name: str, sub_path: str, proxy_prefix: str
        ) -> Response:
            # resolve_proxy_target returns None for an unknown name and, fail-closed,
            # for an ambiguous one (a name shared by a local and a remote row, or two
            # peers). Refuse here rather than dial with address=None, which would let
            # EndpointProxy re-resolve via the single-row resolver and arbitrarily pick
            # one of the ambiguous rows — defeating the fail-closed guard.
            if resolved is None:
                return JSONResponse({"error": f"No endpoint '{encoded_name}'"}, status_code=404)
            # A remote endpoint forwards to the peer that owns it; a local one dials
            # its address directly.
            if resolved.peer_id:
                if self._federated_proxy is None:
                    return JSONResponse({"error": "federation proxy unavailable"}, status_code=502)
                return await self._federated_proxy.dispatch(
                    request,
                    peer_id=resolved.peer_id,
                    encoded_name=encoded_name,
                    sub_path=sub_path,
                    proxy_prefix=proxy_prefix,
                    timeout_seconds=resolved.timeout_seconds,
                )
            return await self._endpoint_proxy.dispatch(
                request,
                encoded_name=encoded_name,
                sub_path=sub_path,
                proxy_prefix=proxy_prefix,
                address=resolved.address,
                timeout_seconds=resolved.timeout_seconds,
            )

        # The proxy routes are @public so the route-annotation middleware does
        # not apply the whole-dashboard @requires_auth (which would over-grant a
        # served-model token the RPC surface). They enforce their own
        # per-endpoint access mode via _authorize_proxy instead.
        @public
        async def _proxy_endpoint(request: Request) -> Response:
            name = request.path_params["endpoint_name"]
            resolved, deny = _resolve_and_authorize_proxy(
                request,
                name,
                self._endpoint_service,
                self._auth_policy,
                federation_owner_check=self._federation_owner_check,
            )
            if deny is not None:
                return deny
            return await _dispatch_resolved(
                request,
                resolved,
                encoded_name=name,
                sub_path=request.path_params["sub_path"],
                proxy_prefix=f"/proxy/{name}",
            )

        @public
        async def _proxy_endpoint_token(request: Request) -> Response:
            # URL-token fallback for transports that can't set an Authorization
            # header: /proxy/t/<token>/<name>/<sub_path>. Same JWT as the header
            # form, lifted from the path and validated the same way.
            token = request.path_params["token"]
            name = request.path_params["endpoint_name"]
            resolved, deny = _resolve_and_authorize_proxy(
                request,
                name,
                self._endpoint_service,
                self._auth_policy,
                token=token,
                federation_owner_check=self._federation_owner_check,
            )
            if deny is not None:
                return deny
            return await _dispatch_resolved(
                request,
                resolved,
                encoded_name=name,
                sub_path=request.path_params["sub_path"],
                proxy_prefix=f"/proxy/t/{token}/{name}",
            )

        @public
        async def _proxy_endpoint_redirect(request: Request) -> Response:
            # ``/proxy/<name>`` (no trailing slash, no sub_path) needs a
            # redirect to ``/proxy/<name>/`` so upstream apps resolve their
            # relative assets correctly. We can't use Starlette's built-in
            # redirect_slashes=True: that builds an *absolute* Location from
            # scope["server"] / the Host header, which behind IAP is the
            # internal bind IP. A path-only Location resolves against the
            # browser's current origin, so no internal address leaks.
            name = request.path_params["endpoint_name"]
            _, deny = _resolve_and_authorize_proxy(
                request,
                name,
                self._endpoint_service,
                self._auth_policy,
                federation_owner_check=self._federation_owner_check,
            )
            if deny is not None:
                return deny
            query = f"?{request.url.query}" if request.url.query else ""
            return RedirectResponse(f"/proxy/{name}/{query}", status_code=307)

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
            Route(
                "/proxy/{endpoint_name:str}",
                _proxy_endpoint_redirect,
                methods=list(endpoint_proxy.ALLOWED_METHODS),
            ),
            # URL-token fallback — must precede PROXY_ROUTE, which would otherwise
            # swallow ``/proxy/t/...`` as endpoint ``t``. The ``t`` label is
            # reserved (endpoint names are never a bare ``t``).
            Route(
                "/proxy/t/{token:str}/{endpoint_name:str}/{sub_path:path}",
                _proxy_endpoint_token,
                methods=list(endpoint_proxy.ALLOWED_METHODS),
            ),
            Route(
                endpoint_proxy.PROXY_ROUTE,
                _proxy_endpoint,
                methods=list(endpoint_proxy.ALLOWED_METHODS),
            ),
            Mount(rpc_asgi_app.path, app=rpc_asgi_app),
            Mount(endpoint_rpc_app.path, app=endpoint_rpc_app),
            Mount(stats_app.path, app=stats_app),
        ]
        routes.append(static_files_mount())

        async def _close_proxies() -> None:
            await self._endpoint_proxy.close()
            if self._federated_proxy is not None:
                await self._federated_proxy.close()

        app = Starlette(
            routes=routes,
            lifespan=on_shutdown(_close_proxies),
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
        wrapped: ASGIApp = RouteAuthMiddleware(app, self._auth_policy, cookie_name=SESSION_COOKIE)
        # Subdomain dispatch wraps everything: subdomain requests don't match
        # any Starlette route, so RouteAuthMiddleware would pass them through.
        wrapped = _SubdomainProxyMiddleware(
            wrapped,
            endpoint_proxy=self._endpoint_proxy,
            endpoint_service=self._endpoint_service,
            auth_policy=self._auth_policy,
            federation_owner_check=self._federation_owner_check,
        )
        return wrapped

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
        try:
            _resolve_request_identity(self._auth_policy, request)
            authenticated = True
        except ValueError:
            authenticated = False
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
