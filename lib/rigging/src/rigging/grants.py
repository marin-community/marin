# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attenuating action/resource grants evaluated by an injected runtime authority."""

import hashlib
import math
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit


class GrantDenied(ValueError):
    """The authenticated principal does not hold the requested authority."""


def _validate_resource(resource: str, *, prefix: bool) -> None:
    parsed = urlsplit(resource)
    path = parsed.path
    if (
        not resource
        or not path.startswith("/")
        or (parsed.scheme and (not parsed.netloc or parsed.scheme != "s3"))
        or (not parsed.scheme and parsed.netloc)
        or parsed.query
        or parsed.fragment
        or (parsed.scheme and re.fullmatch(r"[a-z0-9][a-z0-9.-]*", parsed.netloc) is None)
        or any(ord(character) <= 32 for character in resource)
        or any(character in resource for character in ("%", "\\", "*", "?", "#", "\x00"))
        or any(segment in (".", "..") for segment in path.split("/"))
        or "//" in path
        or (prefix and not resource.endswith("/"))
    ):
        raise ValueError("Resources must be canonical absolute paths or s3:// URLs; bounds must end in /")


@dataclass(frozen=True)
class Permission:
    """Actions allowed beneath one resource prefix at one service audience."""

    service: str
    actions: frozenset[str]
    resource: str

    def __post_init__(self) -> None:
        if not self.service or not self.actions or any(not action or "*" in action for action in self.actions):
            raise ValueError("Permissions require an explicit service and nonempty exact actions")
        _validate_resource(self.resource, prefix=True)


def permissions_cover(bounds: Sequence[Permission], requested: Sequence[Permission]) -> bool:
    """Return whether each requested action and prefix is contained in bounds."""
    return all(
        all(
            any(
                bound.service == permission.service
                and action in bound.actions
                and permission.resource.startswith(bound.resource)
                for bound in bounds
            )
            for action in permission.actions
        )
        for permission in requested
    )


def permissions_allow(permissions: Sequence[Permission], *, service: str, action: str, resource: str) -> bool:
    """Check an operation against canonical, slash-delimited resource bounds."""
    try:
        _validate_resource(resource, prefix=False)
    except ValueError:
        return False
    return any(
        permission.service == service and action in permission.actions and resource.startswith(permission.resource)
        for permission in permissions
    )


class GrantPolicy(Protocol):
    """Deployment policy for complete grant bounds and concrete operations."""

    def allows_grant(self, principal: str, permissions: Sequence[Permission]) -> bool: ...

    def check(self, principal: str, service: str, action: str, resource: str) -> bool: ...


@dataclass(frozen=True)
class PrincipalAuthority:
    """Deployment-managed ownership and delegation limits for one principal."""

    principal: str
    owner: str
    max_ttl: int
    delegation_depth: int
    revision: str

    def __post_init__(self) -> None:
        if not self.principal or not self.owner or self.max_ttl <= 0 or self.delegation_depth < 0 or not self.revision:
            raise ValueError("Principal authorities require identities, a positive lifetime, and nonnegative depth")


@dataclass(frozen=True)
class Grant:
    """A runtime authority bound to one acting principal and its ancestor grant."""

    id: str
    owner: str
    actor: str
    permissions: tuple[Permission, ...]
    parent_id: str | None
    created_by: str
    created_at: float
    expires_at: float
    delegation_depth: int
    revoked_at: float | None
    policy_revision: str
    lease_expires_at: float | None = None

    @property
    def credential_expires_at(self) -> float:
        """Bound credential lifetime by both the absolute ceiling and live lease."""
        return min(self.expires_at, self.lease_expires_at) if self.lease_expires_at is not None else self.expires_at


class RuntimeGrants(Protocol):
    """Persist grants and evaluate their current bounds and complete ancestry."""

    def get(self, grant_id: str) -> Grant | None: ...

    def put(self, grant: Grant) -> Grant:
        """Atomically insert immutable grant data, or return its existing identity."""
        ...

    def is_active(self, grant: Grant, now: float) -> bool: ...

    def allows(self, grant: Grant, *, service: str, action: str, resource: str, now: float) -> bool: ...

    def renew(self, grant: Grant, *, expires_at: float) -> Grant:
        """Append an overlapping lease without changing immutable grant bounds."""
        ...

    def revoke(self, grant: Grant, now: float) -> None: ...

    def close(self) -> None: ...


def _request_id(principal: str, key: str) -> str:
    return hashlib.sha256(f"{len(principal)}:{principal}{key}".encode()).hexdigest()


def _matches_request(
    grant: Grant,
    *,
    actor: str,
    permissions: tuple[Permission, ...],
    expires_at: float,
    parent_id: str | None,
    lease_expires_at: float | None,
) -> bool:
    return (
        grant.actor == actor
        and (grant.lease_expires_at is None) == (lease_expires_at is None)
        and (lease_expires_at is None or grant.credential_expires_at <= lease_expires_at)
        and grant.parent_id == parent_id
        and grant.expires_at <= expires_at
        and permissions_cover(grant.permissions, permissions)
        and permissions_cover(permissions, grant.permissions)
    )


class GrantStore:
    """Validate issuance and standing policy against the runtime grant authority."""

    def __init__(
        self,
        runtime: RuntimeGrants,
        authorities: Sequence[PrincipalAuthority],
        policy: GrantPolicy,
        *,
        now: Callable[[], float] = time.time,
    ):
        self._authorities = {authority.principal: authority for authority in authorities}
        if len(self._authorities) != len(authorities):
            raise ValueError("Principal authorities must be unique")
        self._policy = policy
        self._now = now
        self.runtime = runtime

    def close(self) -> None:
        self.runtime.close()

    def _get(self, grant_id: str) -> Grant:
        grant = self.runtime.get(grant_id)
        if grant is None:
            raise GrantDenied("Unknown grant")
        return grant

    def _active(self, grant_id: str, now: float) -> tuple[Grant, Grant]:
        grant = self._get(grant_id)
        if not self.runtime.is_active(grant, now):
            raise GrantDenied("Grant is expired or revoked")
        ancestor = grant
        chain: list[Grant] = []
        while True:
            chain.append(ancestor)
            if ancestor.parent_id is None:
                break
            ancestor = self._get(ancestor.parent_id)
        authority = self._authorities.get(ancestor.created_by)
        if (
            authority is None
            or authority.owner != ancestor.owner
            or authority.revision != ancestor.policy_revision
            or ancestor.expires_at > ancestor.created_at + authority.max_ttl
            or ancestor.delegation_depth > authority.delegation_depth - (ancestor.actor != ancestor.created_by)
            or not all(self._policy.allows_grant(ancestor.created_by, item.permissions) for item in chain)
        ):
            raise GrantDenied("Standing policy no longer covers the grant")
        return grant, ancestor

    def active(self, grant_id: str) -> Grant:
        """Read an active grant, rejecting any expired or revoked ancestor."""
        grant, _ = self._active(grant_id, self._now())
        return grant

    def grant_for_request(self, *, parent_id: str, idempotency_key: str) -> Grant | None:
        """Find a delegated request for retirement, including expired or revoked grants."""
        parent = self._get(parent_id)
        grant = self.runtime.get(_request_id(parent.actor, idempotency_key))
        return grant if grant is not None and grant.parent_id == parent_id else None

    def authority_principal(self, grant_id: str) -> str:
        """Return the standing principal behind an active grant's delegation chain."""
        _, root = self._active(grant_id, self._now())
        return root.created_by

    def create(
        self,
        principal: str,
        *,
        actor: str,
        permissions: tuple[Permission, ...],
        expires_at: float,
        parent_id: str | None = None,
        idempotency_key: str | None = None,
        lease_expires_at: float | None = None,
    ) -> Grant:
        """Create a root from standing policy, or attenuate the caller's grant.

        ``principal`` must come from authenticated identity, never request JSON.
        A delegate may narrow actions, resources, lifetime, and remaining depth.
        The owner's identity is inherited from policy or the parent grant.
        """
        if not actor or not permissions:
            raise GrantDenied("A grant needs an actor and permissions")
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 128:
            raise GrantDenied("Idempotency keys require 1-128 characters")
        now = self._now()
        if lease_expires_at is not None and (
            not math.isfinite(lease_expires_at) or not now < lease_expires_at <= expires_at
        ):
            raise GrantDenied("Lease deadline is outside the grant lifetime")
        if not math.isfinite(expires_at) or expires_at <= now:
            raise GrantDenied("Grant lifetime has ended")
        grant_id = uuid.uuid4().hex if idempotency_key is None else _request_id(principal, idempotency_key)
        existing = self.runtime.get(grant_id)
        if existing is not None:
            if not _matches_request(
                existing,
                actor=actor,
                permissions=permissions,
                expires_at=expires_at,
                parent_id=parent_id,
                lease_expires_at=lease_expires_at,
            ):
                raise GrantDenied("Idempotency key belongs to a different grant request")
            return self.active(existing.id)
        if parent_id is None:
            authority = self._authorities.get(principal)
            if (
                authority is None
                or expires_at > now + authority.max_ttl
                or not self._policy.allows_grant(principal, permissions)
            ):
                raise GrantDenied("No standing authority covers this grant")
            owner, depth, revision = authority.owner, authority.delegation_depth, authority.revision
            if actor != principal:
                if depth == 0:
                    raise GrantDenied("Standing authority does not allow delegation")
                depth -= 1
        else:
            parent, root = self._active(parent_id, now)
            if principal != parent.actor:
                raise GrantDenied("Only the parent grant's actor may delegate it")
            if parent.delegation_depth == 0:
                raise GrantDenied("Grant does not allow further delegation")
            if expires_at > parent.credential_expires_at or not permissions_cover(parent.permissions, permissions):
                raise GrantDenied("Delegation would widen the parent grant")
            if not self._policy.allows_grant(root.created_by, permissions):
                raise GrantDenied("Standing policy does not allow this delegation")
            owner, depth, revision = parent.owner, parent.delegation_depth - 1, parent.policy_revision
        grant = self.runtime.put(
            Grant(
                id=grant_id,
                owner=owner,
                actor=actor,
                permissions=permissions,
                parent_id=parent_id,
                created_by=principal,
                created_at=now,
                expires_at=expires_at,
                delegation_depth=depth,
                revoked_at=None,
                policy_revision=revision,
                lease_expires_at=lease_expires_at,
            )
        )
        if not _matches_request(
            grant,
            actor=actor,
            permissions=permissions,
            expires_at=expires_at,
            parent_id=parent_id,
            lease_expires_at=lease_expires_at,
        ):
            raise GrantDenied("Idempotency key belongs to a different grant request")
        # Recheck after insertion so a concurrent ancestor revocation also denies issuance.
        return self.active(grant.id)

    def renew_delegated(self, grant_id: str, *, principal: str, expires_at: float) -> Grant:
        """Extend a live child lease at a trusted deputy boundary.

        This is absent from the public grant API. The caller authenticates
        liveness and serializes renewal with retirement. Compute the absolute
        deadline before remote work; never recompute it on retry.
        """
        grant = self.active(grant_id)
        if grant.parent_id is None or grant.lease_expires_at is None:
            raise GrantDenied("Only delegated leases can be renewed")
        parent = self.active(grant.parent_id)
        if (
            principal != parent.actor
            or not math.isfinite(expires_at)
            or expires_at > min(parent.expires_at, grant.expires_at)
        ):
            raise GrantDenied("Renewal is outside the parent authority")
        if expires_at <= self._now():
            raise GrantDenied("Renewal deadline has ended")
        if expires_at > grant.credential_expires_at:
            self.runtime.renew(grant, expires_at=expires_at)
        return self.active(grant.id)

    def authorize(self, grant_id: str, *, actor: str, service: str, action: str, resource: str) -> Grant:
        """Authorize one concrete service operation against current grant state."""
        _validate_resource(resource, prefix=False)
        grant, root = self._active(grant_id, self._now())
        if actor != grant.actor or not self.runtime.allows(
            grant, service=service, action=action, resource=resource, now=self._now()
        ):
            raise GrantDenied("Operation is outside the grant")
        if not self._policy.check(root.created_by, service, action, resource):
            raise GrantDenied("Standing policy does not allow this operation")
        return grant

    def revoke(self, grant_id: str, *, principal: str) -> None:
        """Stop authority and future refresh for this grant and its descendants."""
        grant = self._get(grant_id)
        if principal not in (grant.owner, grant.created_by, grant.actor):
            raise GrantDenied("Principal cannot revoke this grant")
        self.runtime.revoke(grant, self._now())
