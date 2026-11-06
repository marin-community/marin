# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standing authorization through OpenFGA; runtime grants only narrow its decisions."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.parse import quote

import requests
from rigging.grants import Permission, PrincipalAuthority, permissions_allow, permissions_cover
from rigging.shared_grants import authority_origin

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
    }
)


class AuthorizationUnavailable(requests.RequestException):
    """A missing standing decision denies the request without revoking its grant."""


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
    ):
        self._url = f"{authority_origin(api_url)}/stores/{quote(store_id, safe='')}/check"
        self._model_id = model_id
        self._token_file = token_file
        self._ca_bundle = ca_bundle
        bundle = {
            "store_id": store_id,
            "model_id": model_id,
            "principals": list(principals),
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
                "user": f"principal:{principal}",
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
