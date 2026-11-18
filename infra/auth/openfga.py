# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standing policy and attenuating runtime authority in separate OpenFGA stores."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from urllib.parse import quote

import requests
from rigging.grants import Grant, Permission, PrincipalAuthority, permissions_allow, permissions_cover
from rigging.shared_grants import authority_origin

from infra.auth.secret_bindings import SecretBinding

ACTIONS = frozenset(
    {
        ("loom", "session.create"),
        ("iris", "job.submit"),
        ("iris", "job.read"),
        ("iris", "job.cancel"),
        ("finelog", "logs.query"),
        ("finelog", "logs.ingest"),
        ("marina", "app.read"),
        ("marina", "app.release"),
        ("cw_s3", "credentials.issue"),
        ("openbao", "secret.read"),
    }
)


class AuthorizationUnavailable(requests.RequestException):
    """A missing authorization decision denies the request without changing authority."""


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat()


def principal_user(principal: str) -> str:
    """Encode a Marin identity as one OpenFGA principal object ID."""
    return f"principal:{quote(principal, safe='')}"


class OpenFGARuntime:
    """Persist and evaluate runtime grants in an independently authorized store.

    Creation writes the immutable metadata singleton, actor, bounds and ancestry
    together. Revocation is monotonic and retains metadata for lifecycle cleanup.
    The gateway credential must be scoped to this store's Read, Write and Check.
    """

    def __init__(
        self,
        *,
        api_url: str,
        store_id: str,
        model_id: str,
        token_file: Path,
        ca_bundle: str | bool = True,
    ):
        self._url = f"{authority_origin(api_url)}/stores/{quote(store_id, safe='')}"
        self._model_id = model_id
        self._token_file = token_file
        self._ca_bundle = ca_bundle
        if not store_id or not model_id:
            raise ValueError("Runtime OpenFGA store and immutable model IDs are required")

    def _request(self, operation: str, payload: dict) -> dict:
        try:
            response = requests.post(
                f"{self._url}/{operation}",
                json=payload,
                headers={"Authorization": f"Bearer {self._token_file.read_text().strip()}"},
                timeout=10,
                verify=self._ca_bundle,
                allow_redirects=False,
            )
            if response.status_code not in {200, 201, 204}:
                raise AuthorizationUnavailable(f"OpenFGA runtime {operation} failed: HTTP {response.status_code}")
            result = response.json() if response.content else {}
            if not isinstance(result, dict):
                raise AuthorizationUnavailable("OpenFGA returned an invalid runtime response")
            return result
        except (requests.RequestException, OSError, ValueError) as error:
            raise AuthorizationUnavailable("OpenFGA runtime authority is unavailable") from error

    def get(self, grant_id: str) -> Grant | None:
        rows = []
        continuation = ""
        while True:
            result = self._request(
                "read",
                {
                    "tuple_key": {"object": f"grant:{grant_id}"},
                    "page_size": 100,
                    "continuation_token": continuation,
                    "consistency": "HIGHER_CONSISTENCY",
                },
            )
            rows.extend(row["key"] for row in result.get("tuples", []))
            continuation = result.get("continuation_token", "")
            if not continuation:
                break
        metadata = next((row for row in rows if row["relation"] == "enabled" and row["user"] == "principal:*"), None)
        if metadata is None:
            return None
        try:
            data = json.loads(metadata["condition"]["context"]["grant_json"])
            data["permissions"] = tuple(
                Permission(item["service"], frozenset(item["actions"]), item["resource"]) for item in data["permissions"]
            )
            grant = Grant(**data)
            if grant.lease_expires_at is not None:
                prefix = f"lease:{grant.id}_"
                deadlines = [
                    float.fromhex(row["user"].removeprefix(prefix))
                    for row in rows
                    if row["relation"] == "lease" and row["user"].startswith(prefix)
                ]
                if not deadlines:
                    raise ValueError("Leased grant has no persisted lease")
                grant = replace(grant, lease_expires_at=min(grant.expires_at, max(deadlines)))
            revoked = next((row for row in rows if row["relation"] == "revoked"), None)
            return replace(grant, revoked_at=None if revoked is None else revoked["condition"]["context"]["revoked_at"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuthorizationUnavailable("OpenFGA contains invalid runtime grant metadata") from error

    def put(self, grant: Grant) -> Grant:
        permissions = [
            {"service": item.service, "actions": sorted(item.actions), "resource": item.resource}
            for item in grant.permissions
        ]
        metadata = {**asdict(grant), "permissions": permissions}
        object_id = f"grant:{grant.id}"
        tuples = [
            {
                "user": "principal:*",
                "relation": "enabled",
                "object": object_id,
                "condition": {
                    "name": "live_grant",
                    "context": {
                        "expires_at": _timestamp(grant.expires_at),
                        "grant_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    },
                },
            },
            {"user": principal_user(grant.actor), "relation": "actor", "object": object_id},
            {
                "user": "principal:*",
                "relation": "bounds",
                "object": object_id,
                "condition": {"name": "grant_bounds", "context": {"permissions": permissions}},
            },
            {
                "user": "principal:*" if grant.parent_id is None else f"grant:{grant.parent_id}",
                "relation": "root" if grant.parent_id is None else "parent",
                "object": object_id,
            },
        ]
        if grant.lease_expires_at is None:
            tuples.append({"user": "principal:*", "relation": "unleased", "object": object_id})
        else:
            tuples.extend(self._lease_tuples(grant.id, grant.lease_expires_at))
        try:
            self._request("write", {"authorization_model_id": self._model_id, "writes": {"tuple_keys": tuples}})
        except AuthorizationUnavailable:
            # A conflicting or timed-out atomic write may already have committed.
            existing = self.get(grant.id)
            if existing is None:
                raise
            return existing
        return grant

    def _check(self, grant: Grant, relation: str, context: dict) -> bool:
        result = self._request(
            "check",
            {
                "authorization_model_id": self._model_id,
                "tuple_key": {
                    "user": principal_user(grant.actor),
                    "relation": relation,
                    "object": f"grant:{grant.id}",
                },
                "context": context,
                "consistency": "HIGHER_CONSISTENCY",
            },
        )
        if type(result.get("allowed")) is not bool:
            raise AuthorizationUnavailable("OpenFGA returned an invalid runtime authorization decision")
        return result["allowed"]

    def is_active(self, grant: Grant, now: float) -> bool:
        return self._check(grant, "valid", {"current_time": _timestamp(now)})

    def allows(self, grant: Grant, *, service: str, action: str, resource: str, now: float) -> bool:
        return self._check(
            grant,
            "allowed",
            {"current_time": _timestamp(now), "service": service, "action": action, "resource": resource},
        )

    def renew(self, grant: Grant, *, expires_at: float) -> Grant:
        """Append an overlapping lease without changing immutable grant metadata.

        A delayed write can only supply its original precomputed deadline.
        Separate revocation tuples remain authoritative across concurrent writes.
        """
        self._request(
            "write",
            {
                "authorization_model_id": self._model_id,
                "writes": {"tuple_keys": self._lease_tuples(grant.id, expires_at)},
            },
        )
        return replace(grant, lease_expires_at=expires_at)

    def _lease_tuples(self, grant_id: str, expires_at: float) -> list[dict]:
        lease_id = f"lease:{grant_id}_{float(expires_at).hex()}"
        return [
            {"user": lease_id, "relation": "lease", "object": f"grant:{grant_id}"},
            {
                "user": "principal:*",
                "relation": "active",
                "object": lease_id,
                "condition": {"name": "live_lease", "context": {"expires_at": _timestamp(expires_at)}},
            },
        ]

    def revoke(self, grant: Grant, now: float) -> None:
        try:
            self._request(
                "write",
                {
                    "authorization_model_id": self._model_id,
                    "writes": {
                        "tuple_keys": [
                            {
                                "user": "principal:*",
                                "relation": "revoked",
                                "object": f"grant:{grant.id}",
                                "condition": {"name": "revoked_grant", "context": {"revoked_at": now}},
                            }
                        ]
                    },
                },
            )
        except AuthorizationUnavailable:
            existing = self.get(grant.id)
            if existing is None or existing.revoked_at is None:
                raise

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class ProviderIdentity:
    """The complete, audited native capability of one provider identity."""

    name: str
    permissions: tuple[Permission, ...]


def scope_object(service: str, action: str, resource: str) -> str:
    """Keep action namespaces distinct even when their resource paths coincide."""
    return f"scope:{service}/{action}/{resource}"


def resource_parents(service: str, action: str, resource: str) -> list[dict]:
    """Construct ancestry from a canonical server-selected path, never caller tuples."""
    Permission(service, frozenset({action}), resource if resource.endswith("/") else resource + "/")
    if not resource.startswith("/"):
        raise ValueError("Application resources require canonical absolute paths")
    paths = [resource[: index + 1] for index, character in enumerate(resource) if character == "/"]
    if paths[-1] != resource:
        paths.append(resource)
    if len(paths) > 32:
        raise ValueError("Resource ancestry exceeds the supported depth")
    return [
        {
            "user": scope_object(service, action, parent),
            "relation": "parent",
            "object": scope_object(service, action, child),
        }
        for parent, child in pairwise(paths)
    ]


class OpenFGAPolicy:
    """Query a pinned OpenFGA model with current standing relationships.

    The configured endpoint must expose only Check to this credential. Model and
    tuple writes belong to the operator reconciliation endpoint. Tuple changes
    do not change grant revisions: each operation checks live membership again.
    """

    def __init__(
        self,
        *,
        api_url: str,
        store_id: str,
        model_id: str,
        token_file: Path,
        principals: Sequence[dict],
        providers: Sequence[ProviderIdentity],
        ca_bundle: str | bool = True,
        secrets: Sequence[SecretBinding] = (),
    ):
        self._url = f"{authority_origin(api_url)}/stores/{quote(store_id, safe='')}/check"
        self._model_id = model_id
        self._token_file = token_file
        self._ca_bundle = ca_bundle
        bundle = {
            "store_id": store_id,
            "model_id": model_id,
            "principals": list(principals),
            "secrets": [asdict(binding) for binding in secrets],
            "providers": [
                {
                    "name": provider.name,
                    "permissions": [
                        {"service": p.service, "actions": sorted(p.actions), "resource": p.resource}
                        for p in provider.permissions
                    ],
                }
                for provider in providers
            ],
        }
        self.revision = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.authorities = tuple(PrincipalAuthority(**p, revision=self.revision) for p in principals)
        self._principals = {authority.principal for authority in self.authorities}
        self._providers = {provider.name: provider for provider in providers}
        if len(self._principals) != len(self.authorities) or len(self._providers) != len(providers):
            raise ValueError("Principal and provider identity names must be unique")
        if not store_id or not model_id:
            raise ValueError("OpenFGA store and immutable model IDs are required")
        for provider in providers:
            if not provider.name or not provider.permissions or any(p.service != "cw_s3" for p in provider.permissions):
                raise ValueError("Provider identities require complete cw_s3 mappings")

    def _decision(self, principal: str, service: str, action: str, resource: str) -> bool:
        if principal not in self._principals or (service, action) not in ACTIONS:
            return False
        try:
            parents = [] if service == "cw_s3" else resource_parents(service, action, resource)
        except ValueError:
            return False
        payload = {
            "authorization_model_id": self._model_id,
            "tuple_key": {
                "user": principal_user(principal),
                "relation": "allowed",
                "object": scope_object(service, action, resource),
            },
            "consistency": "HIGHER_CONSISTENCY",
            "contextual_tuples": {"tuple_keys": parents},
        }
        try:
            response = requests.post(
                self._url,
                json=payload,
                headers={"Authorization": f"Bearer {self._token_file.read_text().strip()}"},
                timeout=10,
                verify=self._ca_bundle,
                allow_redirects=False,
            )
            if response.status_code != 200:
                raise AuthorizationUnavailable("OpenFGA did not return an authorization decision")
            result = response.json()
            if not isinstance(result, dict) or type(result.get("allowed")) is not bool:
                raise AuthorizationUnavailable("OpenFGA returned an invalid authorization decision")
            return result["allowed"]
        except (requests.RequestException, OSError, ValueError) as error:
            raise AuthorizationUnavailable("OpenFGA authorization is unavailable") from error

    def allows_provider(self, principal: str, binding: str) -> bool:
        return binding in self._providers and self._decision(principal, "cw_s3", "credentials.issue", binding)

    def check(self, principal: str, service: str, action: str, resource: str) -> bool:
        if service != "cw_s3" or action == "credentials.issue":
            return self._decision(principal, service, action, resource)
        return any(
            permissions_allow(provider.permissions, service=service, action=action, resource=resource)
            and self.allows_provider(principal, provider.name)
            for provider in self._providers.values()
        )

    def allows_grant(self, principal: str, permissions: Sequence[Permission]) -> bool:
        if not permissions:
            return False
        for permission in permissions:
            if permission.service != "cw_s3" and not all(
                self.check(principal, permission.service, action, permission.resource) for action in permission.actions
            ):
                return False
        storage = tuple(permission for permission in permissions if permission.service == "cw_s3")
        if not storage:
            return True
        approved = tuple(
            permission
            for provider in self._providers.values()
            if permissions_cover(storage, provider.permissions) and self.allows_provider(principal, provider.name)
            for permission in provider.permissions
        )
        return permissions_cover(approved, storage)
