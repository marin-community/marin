# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared grant token exchange and live authorization for application services.

Only this exchange mints grant access tokens. Applications use the verifier and
live grant database to enforce action/resource bounds and immediate revocation.
Identity authentication and signing are injected so OpenBao remains the identity
and key authority, rather than making each application a token issuer.
"""

import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import parse_qs

import jwt
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from rigging.auth import TokenProvider
from rigging.grants import Grant, GrantDenied, GrantStore, Permission, permissions_allow, permissions_cover
from rigging.server_auth import extract_bearer_token
from rigging.token_authority import VerifiedClaims

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
GRANT_VERSION = 1
REFRESH_AUDIENCES = frozenset({"cw_s3"})


class IdentityVerifier(Protocol):
    """Authenticate a workload identity to a configured, trusted principal."""

    def verify(self, token: str) -> str: ...


class GrantTokenSigner(Protocol):
    """Mint a grant JWT with protected header typ=at+jwt using the shared authority."""

    def mint(self, claims: dict, *, audience: str, ttl_seconds: int) -> str: ...


class GrantTokenVerifier(Protocol):
    def verify(self, token: str) -> VerifiedClaims: ...


def _permission_claims(permissions: tuple[Permission, ...]) -> list[dict]:
    return [
        {"service": item.service, "actions": sorted(item.actions), "resource": item.resource} for item in permissions
    ]


class GrantAuthService:
    """Exchange authenticated identities for short-lived, audience-bound grants."""

    def __init__(
        self,
        store: GrantStore,
        identity_verifier: IdentityVerifier,
        signer: GrantTokenSigner,
        verifier: GrantTokenVerifier,
        *,
        max_token_ttl: int,
        now: Callable[[], float] = time.time,
    ):
        if max_token_ttl <= 0:
            raise ValueError("max_token_ttl must be positive")
        self.store = store
        self.identity_verifier = identity_verifier
        self._signer = signer
        self._verifier = verifier
        self._max_token_ttl = max_token_ttl
        self._now = now

    def exchange(self, subject_token: str, *, grant_id: str, audience: str) -> dict:
        """Authenticate the grant's actor, recheck policy, and mint one audience."""
        try:
            principal = self.identity_verifier.verify(subject_token)
        except ValueError as exc:
            raise GrantDenied("Invalid workload identity") from exc
        grant = self.store.active(grant_id)
        if principal != grant.actor:
            raise GrantDenied("Identity is outside the grant")
        return self.mint_for(grant, audience=audience)

    def mint_for(self, grant: Grant, *, audience: str, permissions: tuple[Permission, ...] | None = None) -> dict:
        """Mint for a grant already authenticated by a trusted service boundary."""
        grant = self.store.active(grant.id)
        principal = grant.actor
        if permissions is None:
            permissions = tuple(item for item in grant.permissions if item.service == audience)
        if (
            not permissions
            or any(item.service != audience for item in permissions)
            or not permissions_cover(grant.permissions, permissions)
        ):
            raise GrantDenied("Audience is outside the grant")
        # Floor the remaining lifetime so the signed token never outlives the grant.
        ttl = min(self._max_token_ttl, math.floor(grant.expires_at - self._now()))
        if ttl <= 0:
            raise GrantDenied("Grant lifetime has ended")
        token = self._signer.mint(
            {
                "sub": principal,
                "grant_version": GRANT_VERSION,
                "act": {"sub": principal},
                "owner": grant.owner,
                "grant_id": grant.id,
                "policy_revision": grant.policy_revision,
                "permissions": _permission_claims(permissions),
                "jti": uuid.uuid4().hex,
            },
            audience=audience,
            ttl_seconds=ttl,
        )
        self.store.active(grant.id)
        issued_expiry = jwt.decode(token, options={"verify_signature": False})["exp"]
        if issued_expiry > grant.expires_at or issued_expiry <= self._now():
            raise GrantDenied("Issuer latency exceeded the remaining grant lifetime")
        return {
            "access_token": token,
            "issued_token_type": ACCESS_TOKEN_TYPE,
            "token_type": "Bearer",
            "expires_in": ttl,
            "grant_id": grant.id,
        }

    def refresh(self, token: str, *, audience: str) -> dict:
        """Renew a storage access token without changing audience or bounds."""
        if audience not in REFRESH_AUDIENCES:
            raise GrantDenied("This audience requires a fresh workload identity")
        grant, permissions = _verified_access_grant(self._verifier, self.store, audience=audience, token=token)
        return self.mint_for(grant, audience=audience, permissions=permissions)

    def authorize(self, token: str, *, audience: str, action: str, resource: str) -> Grant:
        """Authorize an operation using the shared token verifier and live grants."""
        return GrantAuthorizer(self._verifier, self.store, audience=audience).authorize(
            token, action=action, resource=resource
        )

    def storage_token(self, grant: Grant) -> str:
        """Mint the cw_s3 grant token a separately scoped storage broker consumes."""
        return self.mint_for(grant, audience="cw_s3")["access_token"]

    def active(self, grant_id: str, actor: str) -> bool:
        """Return whether this actor still holds the grant."""
        try:
            return self.store.active(grant_id).actor == actor
        except GrantDenied:
            return False

    def revoke(self, grant_id: str, principal: str) -> None:
        self.store.revoke(grant_id, principal=principal)


class GrantAuthorizer:
    """Verify grant JWTs and current database bounds before a service operation."""

    def __init__(self, verifier: GrantTokenVerifier, store: GrantStore, *, audience: str):
        self._verifier = verifier
        self._store = store
        self._audience = audience

    def authorize(self, token: str, *, action: str, resource: str) -> Grant:
        """Return the active grant permitting this operation, or deny it."""
        grant, permissions = _verified_access_grant(self._verifier, self._store, audience=self._audience, token=token)
        if not permissions_allow(permissions, service=self._audience, action=action, resource=resource):
            raise GrantDenied("Operation is outside token bounds")
        return self._store.authorize(
            grant.id, actor=grant.actor, service=self._audience, action=action, resource=resource
        )


@dataclass(frozen=True)
class GrantAccess:
    """Verified token authority, separate from current runtime grant state."""

    grant_id: str
    owner: str
    actor: str
    audience: str
    policy_revision: str
    permissions: tuple[Permission, ...]
    expires_at: int

    def allows(self, action: str, resource: str) -> bool:
        return permissions_allow(self.permissions, service=self.audience, action=action, resource=resource)


class _TokenPermission(BaseModel):
    model_config = ConfigDict(strict=True)
    service: str
    actions: list[str]
    resource: str


class _AccessClaims(BaseModel):
    model_config = ConfigDict(strict=True)
    iss: str
    grant_version: int
    aud: str
    sub: str
    act: dict[str, str]
    owner: str
    grant_id: str
    policy_revision: str
    permissions: list[_TokenPermission]
    exp: int
    iat: int
    nbf: int | None = None
    jti: str


def verified_grant_token(verifier: GrantTokenVerifier, token: str, *, audience: str) -> GrantAccess:
    """Verify the signed cross-service token contract without a runtime lookup."""
    verified = verifier.verify(token)
    try:
        claims = _AccessClaims.model_validate(verified.claims)
        permissions = tuple(
            Permission(item.service, frozenset(item.actions), item.resource) for item in claims.permissions
        )
    except ValueError as exc:
        raise GrantDenied("Malformed grant token bounds") from exc
    if (
        jwt.get_unverified_header(token).get("typ") != "at+jwt"
        or claims.grant_version != GRANT_VERSION
        or claims.aud != audience
        or not all((claims.sub, claims.owner, claims.grant_id, claims.policy_revision, claims.jti))
        or claims.act != {"sub": claims.sub}
        or claims.iat < 0
        or (claims.nbf is not None and claims.nbf < 0)
        or claims.iat > time.time()
        or claims.exp <= claims.iat
        or not permissions
        or any(item.service != audience for item in permissions)
    ):
        raise GrantDenied("Invalid grant token identity, audience, or permissions")
    return GrantAccess(
        grant_id=claims.grant_id,
        owner=claims.owner,
        actor=claims.sub,
        audience=audience,
        policy_revision=claims.policy_revision,
        permissions=permissions,
        expires_at=claims.exp,
    )


def _verified_access_grant(
    verifier: GrantTokenVerifier, store: GrantStore, *, audience: str, token: str
) -> tuple[Grant, tuple[Permission, ...]]:
    access = verified_grant_token(verifier, token, audience=audience)
    grant = store.active(access.grant_id)
    if (
        access.actor != grant.actor
        or access.owner != grant.owner
        or access.policy_revision != grant.policy_revision
        or access.expires_at > grant.expires_at
        or not permissions_cover(grant.permissions, access.permissions)
    ):
        raise GrantDenied("Token does not match the current grant")
    return grant, access.permissions


class _PermissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str
    actions: frozenset[str]
    resource: str


class _GrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: str
    permissions: tuple[_PermissionRequest, ...]
    expires_at: float
    parent_id: str | None = None
    idempotency_key: str | None = None


class _AuthorizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audience: str
    action: str
    resource: str


def _grant_response(grant: Grant) -> dict:
    return {**asdict(grant), "permissions": _permission_claims(grant.permissions)}


def create_app(service: GrantAuthService) -> Starlette:
    """Mount grant creation/revocation and RFC 8693 token exchange endpoints."""

    async def authorize(request: Request) -> JSONResponse:
        token = extract_bearer_token(dict(request.headers))
        if token is None:
            raise GrantDenied("A grant access token is required")
        payload = _AuthorizationRequest.model_validate(await request.json())
        grant = await run_in_threadpool(
            service.authorize, token, audience=payload.audience, action=payload.action, resource=payload.resource
        )
        return JSONResponse(
            {
                "grant_id": grant.id,
                "actor": grant.actor,
                "owner": grant.owner,
                "expires_at": grant.expires_at,
                "policy_revision": grant.policy_revision,
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    def principal(request: Request) -> str:
        token = extract_bearer_token(dict(request.headers))
        if token is None:
            raise GrantDenied("Authentication required")
        try:
            return service.identity_verifier.verify(token)
        except ValueError as exc:
            raise GrantDenied("Invalid workload identity") from exc

    async def create(request: Request) -> JSONResponse:
        caller = await run_in_threadpool(principal, request)
        payload = _GrantRequest.model_validate(await request.json())
        grant = service.store.create(
            caller,
            actor=payload.actor,
            permissions=tuple(Permission(item.service, item.actions, item.resource) for item in payload.permissions),
            expires_at=payload.expires_at,
            parent_id=payload.parent_id,
            idempotency_key=payload.idempotency_key,
        )
        return JSONResponse(_grant_response(grant), status_code=201)

    async def revoke(request: Request) -> JSONResponse:
        caller = await run_in_threadpool(principal, request)
        await run_in_threadpool(service.store.revoke, request.path_params["grant_id"], principal=caller)
        return JSONResponse({"revoked": True})

    async def exchange(request: Request) -> JSONResponse:
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        form = parse_qs((await request.body()).decode("utf-8"), strict_parsing=True)
        allowed = {"grant_type", "subject_token_type", "subject_token", "audience", "grant_id", "requested_token_type"}
        if form.keys() - allowed or any(len(value) != 1 for value in form.values()):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        fields = {key: value[0] for key, value in form.items()}
        if (
            fields.get("grant_type") != TOKEN_EXCHANGE_GRANT
            or fields.get("subject_token_type") not in (JWT_TOKEN_TYPE, ACCESS_TOKEN_TYPE)
            or fields.get("requested_token_type", ACCESS_TOKEN_TYPE) != ACCESS_TOKEN_TYPE
            or not all(fields.get(key) for key in ("subject_token", "audience", "grant_id"))
        ):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        if fields["subject_token_type"] == JWT_TOKEN_TYPE:
            result = await run_in_threadpool(
                service.exchange, fields["subject_token"], grant_id=fields["grant_id"], audience=fields["audience"]
            )
        else:
            result = await run_in_threadpool(service.refresh, fields["subject_token"], audience=fields["audience"])
            if result["grant_id"] != fields["grant_id"]:
                raise GrantDenied("Refresh cannot change grants")
        return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    async def denied(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"error": "access_denied"}, status_code=403)

    async def invalid(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    return Starlette(
        routes=[
            Route("/v1/grants", create, methods=["POST"]),
            Route("/v1/authorize", authorize, methods=["POST"]),
            Route("/v1/grants/{grant_id}/revoke", revoke, methods=["POST"]),
            Route("/oauth/token", exchange, methods=["POST"]),
        ],
        exception_handlers={GrantDenied: denied, ValueError: invalid, ValidationError: invalid},
    )


class GrantTokenProvider:
    """Acquire grant access tokens using a standard Rigging identity provider.

    ``exchange`` is the transport boundary (for example an HTTPS form POST).
    Refresh always returns to the shared authority so it can enforce revocation.
    """

    def __init__(
        self,
        identity: TokenProvider,
        exchange: Callable[[Mapping[str, str]], dict],
        *,
        grant_id: str,
        audience: str,
    ):
        self._identity = identity
        self._exchange = exchange
        self._grant_id = grant_id
        self._audience = audience

    def get_token(self) -> str:
        identity_token = self._identity.get_token()
        if identity_token is None:
            raise GrantDenied("Workload identity is required")
        result = self._exchange(
            {
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token_type": JWT_TOKEN_TYPE,
                "subject_token": identity_token,
                "audience": self._audience,
                "grant_id": self._grant_id,
            }
        )
        return result["access_token"]
