# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Central Cedar decisions over trusted principals and server-derived resources."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import cedarpy
from rigging.grants import Permission, PrincipalAuthority, permissions_allow, permissions_cover

_NAMESPACE = "Marin"
_READ_ONLY_ACTIONS = frozenset(
    {"s3:GetObject", "s3:GetObjectVersion", "s3:ListBucket", "s3:ListBucketVersions", "s3:GetBucketLocation"}
)
_PROVIDER_ACTION = "cw_s3.credentials.issue"


@dataclass(frozen=True)
class ProviderIdentity:
    """The complete, audited native capability of one configured provider identity."""

    name: str
    permissions: tuple[Permission, ...]


def _uid(kind: str, identifier: str) -> dict:
    return {"type": f"{_NAMESPACE}::{kind}", "id": identifier}


def _provider_entity(provider: ProviderIdentity) -> dict:
    return {
        "uid": _uid("CWIdentity", provider.name),
        "attrs": {
            "resources": sorted({permission.resource for permission in provider.permissions}),
            "actions": sorted({action for permission in provider.permissions for action in permission.actions}),
            "read_only": all(permission.actions <= _READ_ONLY_ACTIONS for permission in provider.permissions),
        },
        "parents": [],
    }


def _resource_entities(kind: str, resource: str) -> list[dict]:
    # Permission's existing parser rejects traversal, encoded separators and globs.
    Permission("resource-validation", frozenset({"check"}), resource if resource.endswith("/") else resource + "/")
    if not resource.startswith("/"):
        raise ValueError("Application resources require canonical absolute namespace paths")
    identifiers = [resource[: index + 1] for index, character in enumerate(resource) if character == "/"]
    if identifiers[-1] != resource:
        identifiers.append(resource)
    return [
        {
            "uid": _uid(kind, identifier),
            "attrs": {},
            "parents": [_uid(kind, identifiers[index - 1])] if index else [],
        }
        for index, identifier in enumerate(identifiers)
    ]


class CedarPolicy:
    """Compile standing policy once and deny any Cedar evaluation diagnostic.

    Data permissions for CoreWeave are represented only by complete configured
    provider identities. A Cedar permission on an object prefix is never treated
    as proof of the authority that a native storage credential would carry.
    """

    def __init__(
        self,
        schema: dict,
        policies: str,
        entities: list[dict],
        providers: Sequence[ProviderIdentity],
    ):
        if set(schema) != {_NAMESPACE}:
            raise ValueError("The policy adapter requires the Marin schema namespace")
        self._schema = cedarpy.Schema.from_json_str(json.dumps(schema))
        validation = cedarpy.validate_policies(policies, self._schema)
        if not validation.validation_passed:
            raise ValueError(f"Cedar policy does not validate against the schema: {validation.errors}")
        self._policies = cedarpy.PolicySet.from_str(policies)
        resource_types = schema[_NAMESPACE]["entityTypes"]
        prefix_types = {
            name
            for name, definition in resource_types.items()
            if name not in {"Principal", "Group", "CWIdentity"}
            and definition.get("memberOfTypes") == [name]
            and not definition.get("shape", {}).get("attributes", {})
        }
        self._actions = {}
        for identifier, definition in schema[_NAMESPACE]["actions"].items():
            service, separator, action = identifier.partition(".")
            applies = definition.get("appliesTo", {})
            resources = applies.get("resourceTypes", [])
            principals = applies.get("principalTypes", [])
            if (
                not service
                or not separator
                or not action
                or len(resources) != 1
                or resources[0] not in prefix_types | {"CWIdentity"}
                or principals != ["Principal"]
                or applies.get("context", {}).get("attributes", {})
                or (resources[0] == "CWIdentity") != (identifier == _PROVIDER_ACTION)
                or (service == "cw_s3" and identifier != _PROVIDER_ACTION)
            ):
                raise ValueError(f"Unsupported action or resource mapping: {identifier}")
            self._actions[(service, action)] = resources[0]
        self._providers = {provider.name: provider for provider in providers}
        if len(self._providers) != len(providers):
            raise ValueError("Provider identity names must be unique")
        for provider in providers:
            if (
                not provider.name
                or not provider.permissions
                or any(item.service != "cw_s3" for item in provider.permissions)
            ):
                raise ValueError("Provider identities require named, complete cw_s3 mappings")
        self._principals = set()
        for entity in entities:
            if entity["uid"]["type"] not in {f"{_NAMESPACE}::Principal", f"{_NAMESPACE}::Group"}:
                raise ValueError("Base entities may contain only trusted principals and groups")
            if entity["uid"]["type"] == f"{_NAMESPACE}::Principal":
                self._principals.add(entity["uid"]["id"])
        self._entities = cedarpy.Entities.from_json_str(
            json.dumps([*entities, *(_provider_entity(provider) for provider in providers)]), self._schema
        )
        bundle = {
            "schema": schema,
            "policies": policies,
            "entities": entities,
            "providers": [
                {
                    "name": provider.name,
                    "permissions": [
                        {
                            "service": permission.service,
                            "actions": sorted(permission.actions),
                            "resource": permission.resource,
                        }
                        for permission in provider.permissions
                    ],
                }
                for provider in providers
            ],
        }
        self.revision = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.authorities = tuple(
            PrincipalAuthority(
                principal=entity["uid"]["id"],
                owner=entity["attrs"]["owner"],
                max_ttl=entity["attrs"]["max_ttl"],
                delegation_depth=entity["attrs"]["delegation_depth"],
                revision=self.revision,
            )
            for entity in entities
            if entity["uid"]["type"] == f"{_NAMESPACE}::Principal"
        )

    def _decision(self, principal: str, service: str, action: str, resource: str) -> bool:
        kind = self._actions.get((service, action))
        if principal not in self._principals or kind is None:
            return False
        try:
            entities = self._entities
            if kind != "CWIdentity":
                entities = entities.with_added_json_str(json.dumps(_resource_entities(kind, resource)), self._schema)
            result = cedarpy.is_authorized(
                {
                    "principal": _uid("Principal", principal),
                    "action": _uid("Action", f"{service}.{action}"),
                    "resource": _uid(kind, resource),
                    "context": {},
                },
                self._policies,
                entities,
                self._schema,
            )
        except ValueError:
            return False
        return result.allowed and not result.diagnostics.errors

    def allows_provider(self, principal: str, binding: str) -> bool:
        """Authorize the entire configured identity, including its native rights."""
        return binding in self._providers and self._decision(principal, "cw_s3", "credentials.issue", binding)

    def check(self, principal: str, service: str, action: str, resource: str) -> bool:
        """Check one operation using trusted types and server-derived entities."""
        if service != "cw_s3" or action == "credentials.issue":
            return self._decision(principal, service, action, resource)
        return any(
            permissions_allow(provider.permissions, service=service, action=action, resource=resource)
            and self.allows_provider(principal, provider.name)
            for provider in self._providers.values()
        )

    def allows_grant(self, principal: str, permissions: Sequence[Permission]) -> bool:
        """Approve app bounds and exactly representable complete native mappings."""
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
