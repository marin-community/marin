# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Program-first import metadata and manifest validation for Marin IaC."""

import copy
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pulumi

IMPORT_ID_PLACEHOLDER = "<PLACEHOLDER>"

ImportManifest = dict[str, Any]


@dataclass(frozen=True)
class ImportIdentity:
    """Pulumi identity fields needed to match one generated import entry."""

    resource_type: str
    logical_name: str
    parent_type: str | None
    parent_name: str | None


@dataclass(frozen=True)
class ImportSpec:
    """The provider ID for one resource declared by the Pulumi program."""

    identity: ImportIdentity
    provider_id: str


@dataclass(frozen=True)
class ImportManifestSummary:
    """A principal-safe summary suitable for logs and pull-request comments."""

    imports_by_type: dict[str, int]
    unresolved_by_type: dict[str, int]
    component_count: int
    digest: str


class ImportRegistrar(Protocol):
    """Records provider import IDs without changing a resource's Pulumi options."""

    def register(
        self,
        resource: pulumi.Resource,
        *,
        parent: pulumi.Resource | None,
        provider_id: str,
    ) -> None: ...


class ImportCatalog:
    """Collect import IDs while an inline Pulumi program declares its resources."""

    def __init__(self) -> None:
        self._specs: dict[ImportIdentity, ImportSpec] = {}

    def register(
        self,
        resource: pulumi.Resource,
        *,
        parent: pulumi.Resource | None,
        provider_id: str,
    ) -> None:
        identity = ImportIdentity(
            resource_type=resource.pulumi_resource_type,
            logical_name=resource.pulumi_resource_name,
            parent_type=parent.pulumi_resource_type if parent is not None else None,
            parent_name=parent.pulumi_resource_name if parent is not None else None,
        )
        if identity in self._specs:
            raise ValueError(f"duplicate import identity: {identity}")
        self._specs[identity] = ImportSpec(identity=identity, provider_id=provider_id)

    @property
    def specs(self) -> tuple[ImportSpec, ...]:
        return tuple(self._specs.values())


class _NoImportRegistrar:
    def register(
        self,
        resource: pulumi.Resource,
        *,
        parent: pulumi.Resource | None,
        provider_id: str,
    ) -> None:
        return None


NO_IMPORTS: ImportRegistrar = _NoImportRegistrar()


def _manifest_resources(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    resources = manifest.get("resources")
    if not isinstance(resources, list) or not all(isinstance(resource, dict) for resource in resources):
        raise ValueError("import manifest must contain a resources list")
    return resources


def _manifest_name_table(manifest: Mapping[str, Any]) -> dict[str, str]:
    name_table = manifest.get("nameTable", {})
    if not isinstance(name_table, dict) or not all(
        isinstance(name, str) and isinstance(urn, str) for name, urn in name_table.items()
    ):
        raise ValueError("import manifest nameTable must map names to URNs")
    return name_table


def _logical_name(resource: Mapping[str, Any]) -> str:
    logical_name = resource.get("logicalName", resource.get("name"))
    if not isinstance(logical_name, str):
        raise ValueError("import resource has no logical name")
    return logical_name


def _type_and_name_from_urn(urn: str) -> tuple[str, str]:
    parts = urn.rsplit("::", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid Pulumi URN in import nameTable: {urn!r}")
    qualified_type, logical_name = parts[1], parts[2]
    return qualified_type.rsplit("$", 1)[-1], logical_name


def _parent_identity(
    resource: Mapping[str, Any],
    resources: list[dict[str, Any]],
    name_table: Mapping[str, str],
) -> tuple[str | None, str | None]:
    parent_ref = resource.get("parent")
    if parent_ref is None:
        return None, None
    if not isinstance(parent_ref, str):
        raise ValueError("import resource parent must be a name")

    parent_matches = [candidate for candidate in resources if candidate.get("name") == parent_ref]
    if len(parent_matches) > 1:
        raise ValueError(f"ambiguous import parent {parent_ref!r}")
    if parent_matches:
        parent = parent_matches[0]
        parent_type = parent.get("type")
        if not isinstance(parent_type, str):
            raise ValueError("import parent has no resource type")
        return parent_type, _logical_name(parent)

    parent_urn = name_table.get(parent_ref)
    if parent_urn is None:
        raise ValueError(f"import parent {parent_ref!r} is absent from resources and nameTable")
    return _type_and_name_from_urn(parent_urn)


def _manifest_identity(
    resource: Mapping[str, Any],
    resources: list[dict[str, Any]],
    name_table: Mapping[str, str],
) -> ImportIdentity:
    resource_type = resource.get("type")
    if not isinstance(resource_type, str):
        raise ValueError("import resource has no resource type")
    parent_type, parent_name = _parent_identity(resource, resources, name_table)
    return ImportIdentity(
        resource_type=resource_type,
        logical_name=_logical_name(resource),
        parent_type=parent_type,
        parent_name=parent_name,
    )


def resolve_import_manifest(
    generated: Mapping[str, Any],
    specs: Iterable[ImportSpec],
) -> ImportManifest:
    """Fill provider IDs in a Pulumi-generated Program-first import manifest."""

    resolved = copy.deepcopy(dict(generated))
    resources = _manifest_resources(resolved)
    name_table = _manifest_name_table(resolved)

    specs_by_identity: dict[ImportIdentity, ImportSpec] = {}
    for spec in specs:
        if spec.identity in specs_by_identity:
            raise ValueError(f"duplicate import identity: {spec.identity}")
        specs_by_identity[spec.identity] = spec

    for resource in resources:
        if resource.get("component") is True:
            continue
        if resource.get("id") != IMPORT_ID_PLACEHOLDER:
            raise ValueError("generated import resource did not contain an ID placeholder")
        identity = _manifest_identity(resource, resources, name_table)
        if spec := specs_by_identity.get(identity):
            resource["id"] = spec.provider_id
    return resolved


def _canonical_resource(resource: Mapping[str, Any]) -> str:
    return json.dumps(resource, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def import_manifest_digest(manifest: Mapping[str, Any]) -> str:
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def import_manifest_summary(manifest: Mapping[str, Any]) -> ImportManifestSummary:
    imports_by_type: Counter[str] = Counter()
    unresolved_by_type: Counter[str] = Counter()
    component_count = 0
    for resource in _manifest_resources(manifest):
        resource_type = resource.get("type")
        if not isinstance(resource_type, str):
            raise ValueError("import resource has no resource type")
        if resource.get("component") is True:
            component_count += 1
        elif resource.get("id") == IMPORT_ID_PLACEHOLDER:
            unresolved_by_type[resource_type] += 1
        else:
            imports_by_type[resource_type] += 1
    return ImportManifestSummary(
        imports_by_type=dict(sorted(imports_by_type.items())),
        unresolved_by_type=dict(sorted(unresolved_by_type.items())),
        component_count=component_count,
        digest=import_manifest_digest(manifest),
    )


def _validate_manifest_references(manifest: Mapping[str, Any]) -> None:
    resources = _manifest_resources(manifest)
    name_table = _manifest_name_table(manifest)
    resource_names = {resource.get("name") for resource in resources}
    for resource in resources:
        for field in ("parent", "provider"):
            reference = resource.get(field)
            if reference is not None and reference not in resource_names and reference not in name_table:
                raise ValueError(f"import {field} is absent from resources and nameTable")


def validate_reviewed_manifest(reviewed: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Require the reviewed manifest to be a ready, unchanged subset of current creates."""

    reviewed_name_table = _manifest_name_table(reviewed)
    current_name_table = _manifest_name_table(current)
    if reviewed_name_table != current_name_table:
        raise ValueError("reviewed import nameTable is stale")

    current_resources = Counter(_canonical_resource(resource) for resource in _manifest_resources(current))
    imported_leaf_count = 0
    for resource_index, resource in enumerate(_manifest_resources(reviewed)):
        encoded = _canonical_resource(resource)
        if current_resources[encoded] == 0:
            raise ValueError(f"reviewed import entry {resource_index} is stale or edited")
        current_resources[encoded] -= 1
        if resource.get("component") is not True:
            if resource.get("id") == IMPORT_ID_PLACEHOLDER:
                raise ValueError(f"unresolved import ID at entry {resource_index}")
            imported_leaf_count += 1
    if imported_leaf_count == 0:
        raise ValueError("reviewed import manifest contains no resources to import")
    _validate_manifest_references(reviewed)


def write_import_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a local import transaction file with owner-only permissions."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
