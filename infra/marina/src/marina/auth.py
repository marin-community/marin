# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native IAP/loopback identity or explicit shared grants for stable app UUIDs.

Shared mode admits only annotated operations and has no loopback administrator
fallback. Native deployments retain their existing IAP and local policies.
"""

import uuid
from collections.abc import Callable

from rigging.grants import GrantDenied
from rigging.server_auth import (
    IapAssertionVerifier,
    JwtAuthenticator,
    RequestAuthPolicy,
    VerifiedIdentity,
    extract_bearer_token,
    scope_client_address,
    scope_headers,
)
from rigging.shared_grants import SharedGrantClient, SharedGrantUnavailable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Match, Router
from starlette.types import ASGIApp, Receive, Scope, Send

ADMIN_ROLE = "admin"


def build_policy(iap_audience: str | None, shared: SharedGrantClient | None = None) -> RequestAuthPolicy:
    """Select shared grants or the native IAP/loopback authentication chain."""
    if shared is not None:
        if iap_audience is not None:
            raise ValueError("Select either shared grants or IAP authentication")
        return RequestAuthPolicy(authenticators=(JwtAuthenticator(shared),), verifier=shared)
    verifier = IapAssertionVerifier(iap_audience, lambda _email: ADMIN_ROLE) if iap_audience else None
    return RequestAuthPolicy.enforcing(iap_assertion_verifier=verifier)


def identity_for(request: Request, policy: RequestAuthPolicy) -> VerifiedIdentity:
    """Resolve the identity of a request the route middleware already admitted."""
    headers = scope_headers(request.scope)
    return policy.resolve(
        extract_bearer_token(headers),
        client_address=scope_client_address(request.scope),
        headers=headers,
    )


def grant_action(action: str) -> Callable[[Callable], Callable]:
    """Declare a route's shared authority; undeclared routes deny shared callers."""

    def annotate(handler):
        handler.__marina_grant_action__ = action
        return handler

    return annotate


class SharedGrantMiddleware:
    """Check app UUIDs after route/name resolution, including aliases and APIs."""

    def __init__(
        self, app: ASGIApp, router: Router, grants: SharedGrantClient, resolve_name: Callable[[str], uuid.UUID]
    ):
        self.app = app
        self.router = router
        self.grants = grants
        self.resolve_name = resolve_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        for route in self.router.routes:
            matched, child = route.matches(scope)
            if matched is not Match.FULL:
                continue
            endpoint = child.get("endpoint")
            endpoint_name = getattr(endpoint, "__name__", "")
            if endpoint_name == "healthz":
                return await self.app(scope, receive, send)
            action = getattr(endpoint, "__marina_grant_action__", None)
            if action is None:
                return await JSONResponse({"error": "no shared grant authority for this route"}, status_code=403)(
                    scope, receive, send
                )
            token = extract_bearer_token(dict(request.headers))
            if token is None:
                return await JSONResponse({"error": "authentication required"}, status_code=401)(scope, receive, send)
            try:
                self.grants.access(token)
                if action not in {"directory", "identity"}:
                    params = child.get("path_params", {})
                    if "applet_id" in params:
                        applet_id = uuid.UUID(str(params["applet_id"]))
                    else:
                        name = params.get("name") or request.query_params.get("app")
                        if name is None:
                            raise GrantDenied("An application is required")
                        applet_id = await run_in_threadpool(self.resolve_name, name)
                    if (
                        action == "app.read"
                        and request.method not in {"GET", "HEAD"}
                        and endpoint_name not in {"query_applet", "versioned_applet_query"}
                    ):
                        raise GrantDenied("Application read access cannot invoke write APIs")
                    await run_in_threadpool(self.grants.authorize, token, action=action, resource=f"/apps/{applet_id}/")
            except ValueError:
                return await JSONResponse({"error": "grant does not authorize this application"}, status_code=403)(
                    scope, receive, send
                )
            except SharedGrantUnavailable:
                return await JSONResponse({"error": "shared authorization is unavailable"}, status_code=503)(
                    scope, receive, send
                )
            return await self.app(scope, receive, send)
        return await self.app(scope, receive, send)
