# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared, attenuating action/resource grants with durable runtime revocation.

Standing policy is supplied by the deployment. Creating a session grant only
writes the runtime database; it never changes standing policy or provider IAM.
"""

import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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


def _encode_permissions(permissions: Sequence[Permission]) -> str:
    return json.dumps(
        [{"service": item.service, "actions": sorted(item.actions), "resource": item.resource} for item in permissions],
        separators=(",", ":"),
    )


def _decode_permissions(encoded: str) -> tuple[Permission, ...]:
    return tuple(
        Permission(item["service"], frozenset(item["actions"]), item["resource"]) for item in json.loads(encoded)
    )


def _grant_from_row(row: sqlite3.Row) -> Grant:
    return Grant(
        id=row["id"],
        owner=row["owner"],
        actor=row["actor"],
        permissions=_decode_permissions(row["permissions"]),
        parent_id=row["parent_id"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        delegation_depth=row["delegation_depth"],
        revoked_at=row["revoked_at"],
        policy_revision=row["policy_revision"],
    )


class GrantStore:
    """SQLite runtime grants; ancestor revocation applies on every decision.

    All grant mutations and decisions are serialized in transactions. A single
    local service can reopen the database after restart without losing grants
    or revocations. This store is not a replicated multi-host database.
    """

    def __init__(
        self,
        path: Path,
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
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS grants (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                actor TEXT NOT NULL,
                permissions TEXT NOT NULL,
                parent_id TEXT REFERENCES grants(id),
                created_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                delegation_depth INTEGER NOT NULL,
                revoked_at REAL,
                policy_revision TEXT NOT NULL
            )"""
        )

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _get(self, grant_id: str) -> Grant:
        row = self._connection.execute("SELECT * FROM grants WHERE id = ?", (grant_id,)).fetchone()
        if row is None:
            raise GrantDenied("Unknown grant")
        return _grant_from_row(row)

    def _active(self, grant_id: str, now: float) -> tuple[Grant, Grant]:
        grant = self._get(grant_id)
        ancestor = grant
        chain: list[Grant] = []
        while True:
            if ancestor.revoked_at is not None or ancestor.expires_at <= now:
                raise GrantDenied("Grant is expired or revoked")
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
        with self._transaction():
            grant, _ = self._active(grant_id, self._now())
            return grant

    def authority_principal(self, grant_id: str) -> str:
        """Return the standing principal behind an active grant's delegation chain."""
        with self._transaction():
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
    ) -> Grant:
        """Create a root from standing policy, or attenuate the caller's grant.

        ``principal`` must come from authenticated identity, never request JSON.
        A delegate may narrow actions, resources, lifetime, and remaining depth.
        The owner's identity is inherited from policy or the parent grant.
        """
        if not actor or not permissions:
            raise GrantDenied("A grant needs an actor and permissions")
        with self._transaction():
            now = self._now()
            if not math.isfinite(expires_at) or expires_at <= now:
                raise GrantDenied("Grant lifetime has ended")
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
                if expires_at > parent.expires_at or not permissions_cover(parent.permissions, permissions):
                    raise GrantDenied("Delegation would widen the parent grant")
                if not self._policy.allows_grant(root.created_by, permissions):
                    raise GrantDenied("Standing policy does not allow this delegation")
                owner, depth, revision = parent.owner, parent.delegation_depth - 1, parent.policy_revision
            grant = Grant(
                id=uuid.uuid4().hex,
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
            )
            self._connection.execute(
                "INSERT INTO grants VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    grant.id,
                    grant.owner,
                    grant.actor,
                    _encode_permissions(grant.permissions),
                    grant.parent_id,
                    grant.created_by,
                    grant.created_at,
                    grant.expires_at,
                    grant.delegation_depth,
                    grant.revoked_at,
                    grant.policy_revision,
                ),
            )
            return grant

    def authorize(self, grant_id: str, *, actor: str, service: str, action: str, resource: str) -> Grant:
        """Authorize one concrete service operation against current grant state."""
        _validate_resource(resource, prefix=False)
        with self._transaction():
            grant, root = self._active(grant_id, self._now())
            if actor != grant.actor or not permissions_allow(
                grant.permissions, service=service, action=action, resource=resource
            ):
                raise GrantDenied("Operation is outside the grant")
            if not self._policy.check(root.created_by, service, action, resource):
                raise GrantDenied("Standing policy does not allow this operation")
            return grant

    def revoke(self, grant_id: str, *, principal: str) -> None:
        """Stop authority and future refresh for this grant and its descendants."""
        with self._transaction():
            grant = self._get(grant_id)
            if principal not in (grant.owner, grant.created_by, grant.actor):
                raise GrantDenied("Principal cannot revoke this grant")
            self._connection.execute(
                "UPDATE grants SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?", (self._now(), grant_id)
            )
