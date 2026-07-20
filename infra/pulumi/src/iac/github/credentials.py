# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog and audit GitHub Actions credentials declared in Pulumi stack config."""

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import yaml

BUILTIN_ACTIONS_SECRETS = frozenset({"GITHUB_TOKEN"})
SECRET_REFERENCE = re.compile(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)")
SECRET_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PINNED_GCP_SECRET = re.compile(
    r"^gcp-secret://projects/(?P<project>[^/]+)/secrets/(?P<secret>[^/]+)/versions/(?P<version>[1-9][0-9]*)$"
)
GITHUB_API_PAGE_SIZE = 100
CREDENTIAL_SCHEMA_VERSION = 1
CREDENTIAL_COMMON_FIELDS = frozenset(
    {
        "name",
        "scope",
        "presence",
        "source_kind",
        "source_ref",
        "disposition",
        "note",
    }
)


class CredentialScope(StrEnum):
    ORGANIZATION = "organization"
    REPOSITORY = "repository"
    ENVIRONMENT = "environment"


class OrganizationVisibility(StrEnum):
    ALL = "all"
    PRIVATE = "private"
    SELECTED = "selected"


class Presence(StrEnum):
    PRESENT = "present"
    REFERENCED_MISSING = "referenced-missing"


class SourceKind(StrEnum):
    GCP_SECRET = "gcp-secret"
    CURRENT_ENVIRONMENT = "current-environment"
    CURRENT_FILE = "current-file"
    CONFIGURATION = "configuration"
    GITHUB_METADATA = "github-metadata"
    MANUAL = "manual"


class Disposition(StrEnum):
    KEEP = "keep"
    REMOVE_CANDIDATE = "remove-candidate"
    MOVE_TO_VARIABLE = "move-to-variable"
    REPLACE_WITH_OIDC = "replace-with-oidc"
    REVIEW = "review"


def _credential_scope_fields(scope: CredentialScope) -> frozenset[str]:
    if scope is CredentialScope.ORGANIZATION:
        return frozenset({"visibility", "repositories"})
    if scope is CredentialScope.REPOSITORY:
        return frozenset({"repository"})
    return frozenset({"repository", "environment"})


def _organization_key(name: str) -> tuple[str, ...]:
    return (CredentialScope.ORGANIZATION, name)


def _repository_key(repository: str, name: str) -> tuple[str, ...]:
    return (CredentialScope.REPOSITORY, repository, name)


def _environment_key(repository: str, environment: str, name: str) -> tuple[str, ...]:
    return (CredentialScope.ENVIRONMENT, repository, environment, name)


@dataclass(frozen=True)
class ValueSource:
    kind: SourceKind
    ref: str

    @property
    def does_not_require_owner_recovery(self) -> bool:
        return self.kind is not SourceKind.MANUAL


@dataclass(frozen=True, kw_only=True)
class Credential:
    name: str
    presence: Presence
    source: ValueSource
    disposition: Disposition
    note: str = ""
    scope: ClassVar[CredentialScope]

    @property
    def key(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return ":".join(self.key)


@dataclass(frozen=True, kw_only=True)
class OrganizationCredential(Credential):
    visibility: OrganizationVisibility
    repositories: tuple[str, ...] = ()
    scope: ClassVar[CredentialScope] = CredentialScope.ORGANIZATION

    @property
    def key(self) -> tuple[str, ...]:
        return _organization_key(self.name)


@dataclass(frozen=True, kw_only=True)
class RepositoryCredential(Credential):
    repository: str
    scope: ClassVar[CredentialScope] = CredentialScope.REPOSITORY

    @property
    def key(self) -> tuple[str, ...]:
        return _repository_key(self.repository, self.name)


@dataclass(frozen=True, kw_only=True)
class EnvironmentCredential(Credential):
    repository: str
    environment: str
    scope: ClassVar[CredentialScope] = CredentialScope.ENVIRONMENT

    @property
    def key(self) -> tuple[str, ...]:
        return _environment_key(self.repository, self.environment, self.name)


@dataclass(frozen=True)
class CredentialManifest:
    organization: str
    repositories: tuple[str, ...]
    credentials: tuple[Credential, ...]


@dataclass(frozen=True)
class SecretReference:
    path: str
    line: int


@dataclass(frozen=True, kw_only=True)
class LiveSecret:
    name: str
    scope: ClassVar[CredentialScope]

    @property
    def key(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class OrganizationLiveSecret(LiveSecret):
    visibility: OrganizationVisibility
    repositories: tuple[str, ...] = ()
    scope: ClassVar[CredentialScope] = CredentialScope.ORGANIZATION

    @property
    def key(self) -> tuple[str, ...]:
        return _organization_key(self.name)


@dataclass(frozen=True, kw_only=True)
class RepositoryLiveSecret(LiveSecret):
    repository: str
    scope: ClassVar[CredentialScope] = CredentialScope.REPOSITORY

    @property
    def key(self) -> tuple[str, ...]:
        return _repository_key(self.repository, self.name)


@dataclass(frozen=True, kw_only=True)
class EnvironmentLiveSecret(LiveSecret):
    repository: str
    environment: str
    scope: ClassVar[CredentialScope] = CredentialScope.ENVIRONMENT

    @property
    def key(self) -> tuple[str, ...]:
        return _environment_key(self.repository, self.environment, self.name)


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str
    credential: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {"code": self.code, "detail": self.detail}
        if self.credential is not None:
            result["credential"] = self.credential
        return result


@dataclass(frozen=True)
class AuditReport:
    errors: tuple[Finding, ...]
    referenced_missing: tuple[str, ...]
    removal_candidates: tuple[str, ...]
    unreferenced: tuple[str, ...]
    shadowed: tuple[str, ...]
    available_sources: tuple[str, ...]
    manual_sources: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "errors": [finding.as_dict() for finding in self.errors],
            "referenced_missing": list(self.referenced_missing),
            "removal_candidates": list(self.removal_candidates),
            "unreferenced": list(self.unreferenced),
            "shadowed": list(self.shadowed),
            "available_sources": list(self.available_sources),
            "manual_sources": list(self.manual_sources),
        }


def _credential_from_dict(raw: dict) -> Credential:
    scope = CredentialScope(raw["scope"])
    unknown_fields = set(raw) - CREDENTIAL_COMMON_FIELDS - _credential_scope_fields(scope)
    if unknown_fields:
        raise ValueError(f"unknown credential fields: {sorted(unknown_fields)}")
    name = raw["name"]
    presence = Presence(raw["presence"])
    source = ValueSource(kind=SourceKind(raw["source_kind"]), ref=raw["source_ref"])
    disposition = Disposition(raw["disposition"])
    note = raw.get("note", "")
    if scope is CredentialScope.ORGANIZATION:
        credential: Credential = OrganizationCredential(
            name=name,
            presence=presence,
            source=source,
            disposition=disposition,
            note=note,
            visibility=OrganizationVisibility(raw["visibility"]),
            repositories=tuple(sorted(raw.get("repositories", []))),
        )
    elif scope is CredentialScope.REPOSITORY:
        credential = RepositoryCredential(
            name=name,
            presence=presence,
            source=source,
            disposition=disposition,
            note=note,
            repository=raw["repository"],
        )
    else:
        credential = EnvironmentCredential(
            name=name,
            presence=presence,
            source=source,
            disposition=disposition,
            note=note,
            repository=raw["repository"],
            environment=raw["environment"],
        )
    _validate_credential(credential)
    return credential


def _validate_credential(credential: Credential) -> None:
    if not SECRET_NAME.fullmatch(credential.name):
        raise ValueError(f"invalid GitHub secret name {credential.name!r}")
    if not credential.source.ref:
        raise ValueError(f"{credential.label} has an empty source_ref")
    if credential.source.kind is SourceKind.GCP_SECRET and not PINNED_GCP_SECRET.fullmatch(credential.source.ref):
        raise ValueError(f"{credential.label} must pin a numeric GCP Secret Manager version")
    if isinstance(credential, OrganizationCredential):
        if credential.visibility is OrganizationVisibility.SELECTED and not credential.repositories:
            raise ValueError(f"{credential.label} selects no repositories")
        if credential.visibility is not OrganizationVisibility.SELECTED and credential.repositories:
            raise ValueError(f"{credential.label} lists repositories without selected visibility")
    if credential.disposition is Disposition.REMOVE_CANDIDATE and credential.presence is not Presence.PRESENT:
        raise ValueError(f"{credential.label} cannot remove a credential that is not present")


def credential_manifest(
    *,
    schema_version: int,
    organization: str,
    repositories: list[str],
    credentials: list[dict],
) -> CredentialManifest:
    """Validate credential data read from a Pulumi stack."""
    if schema_version != CREDENTIAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported credential schema version {schema_version!r}")
    parsed_credentials = tuple(_credential_from_dict(raw) for raw in credentials)
    keys = [credential.key for credential in parsed_credentials]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate credential declarations: {duplicates}")
    parsed_repositories = tuple(sorted(repositories))
    known_repositories = {
        credential.repository
        for credential in parsed_credentials
        if isinstance(credential, RepositoryCredential | EnvironmentCredential)
    }
    if not known_repositories.issubset(parsed_repositories):
        raise ValueError(
            "credential repositories are missing from top-level repositories: "
            f"{known_repositories - set(parsed_repositories)}"
        )
    return CredentialManifest(
        organization=organization,
        repositories=parsed_repositories,
        credentials=parsed_credentials,
    )


def load_stack_manifest(path: Path) -> CredentialManifest:
    """Load credential declarations from a committed Pulumi stack YAML file."""
    data = yaml.safe_load(path.read_text())
    config = data["config"]
    return credential_manifest(
        schema_version=config["marin-github:schemaVersion"],
        organization=config["marin-github:organization"],
        repositories=config["marin-github:repositories"],
        credentials=config["marin-github:credentials"],
    )


def discover_secret_references(repo_root: Path) -> dict[str, tuple[SecretReference, ...]]:
    """Return Actions secret references in workflow and composite-action YAML."""
    references: dict[str, list[SecretReference]] = defaultdict(list)
    roots = (repo_root / ".github" / "workflows", repo_root / ".github" / "actions")
    paths = sorted(path for root in roots for suffix in ("*.yaml", "*.yml") for path in root.rglob(suffix))
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for match in SECRET_REFERENCE.finditer(line):
                references[match.group(1)].append(
                    SecretReference(path=str(path.relative_to(repo_root)), line=line_number)
                )
    return {name: tuple(found) for name, found in sorted(references.items())}


def _accessible_to_repository(credential: Credential, repository: str) -> bool:
    if isinstance(credential, RepositoryCredential):
        return credential.repository == repository
    if isinstance(credential, EnvironmentCredential):
        return False
    assert isinstance(credential, OrganizationCredential)
    if credential.visibility is OrganizationVisibility.ALL:
        return True
    if credential.visibility is OrganizationVisibility.PRIVATE:
        return False
    return repository in credential.repositories


def _resolved_credential(
    manifest: CredentialManifest,
    name: str,
    repository: str,
) -> Credential | None:
    candidates = [
        credential
        for credential in manifest.credentials
        if credential.name == name and _accessible_to_repository(credential, repository)
    ]
    scope_order = {
        CredentialScope.ENVIRONMENT: 0,
        CredentialScope.REPOSITORY: 1,
        CredentialScope.ORGANIZATION: 2,
    }
    candidates.sort(key=lambda credential: (credential.presence is not Presence.PRESENT, scope_order[credential.scope]))
    return candidates[0] if candidates else None


def _live_findings(manifest: CredentialManifest, live_secrets: tuple[LiveSecret, ...]) -> list[Finding]:
    findings: list[Finding] = []
    declarations = {credential.key: credential for credential in manifest.credentials}
    live = {secret.key: secret for secret in live_secrets}
    for key, credential in declarations.items():
        live_secret = live.get(key)
        if credential.presence is Presence.REFERENCED_MISSING:
            if live_secret is not None:
                findings.append(
                    Finding("expected-missing-present", "secret now exists; mark it present", credential.label)
                )
            continue
        if live_secret is None:
            findings.append(
                Finding("declared-secret-missing", "declared present but absent from GitHub", credential.label)
            )
            continue
        if isinstance(credential, OrganizationCredential):
            assert isinstance(live_secret, OrganizationLiveSecret)
            if live_secret.visibility != credential.visibility:
                findings.append(
                    Finding(
                        "visibility-drift",
                        f"expected {credential.visibility!r}, found {live_secret.visibility!r}",
                        credential.label,
                    )
                )
            if live_secret.repositories != credential.repositories:
                findings.append(
                    Finding(
                        "repository-access-drift",
                        f"expected {list(credential.repositories)!r}, found {list(live_secret.repositories)!r}",
                        credential.label,
                    )
                )
    for key in live:
        if key not in declarations:
            findings.append(
                Finding("unmanaged-live-secret", "present in GitHub but absent from the catalog", ":".join(key))
            )
    return findings


def audit_credentials(
    manifest: CredentialManifest,
    references: dict[str, tuple[SecretReference, ...]],
    live_secrets: tuple[LiveSecret, ...] | None = None,
) -> AuditReport:
    """Compare the catalog with workflow references and optional live metadata."""
    errors: list[Finding] = []
    consumers: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for name, found in references.items():
        if name in BUILTIN_ACTIONS_SECRETS:
            continue
        for repository in manifest.repositories:
            credential = _resolved_credential(manifest, name, repository)
            if credential is not None:
                consumers[credential.key].update(reference.path for reference in found)
                break
        else:
            locations = ", ".join(f"{reference.path}:{reference.line}" for reference in found)
            errors.append(Finding("undeclared-reference", f"{name} is referenced at {locations}", name))

    for credential in manifest.credentials:
        if credential.disposition is Disposition.REMOVE_CANDIDATE and consumers[credential.key]:
            errors.append(
                Finding(
                    "removal-candidate-referenced",
                    f"referenced by {sorted(consumers[credential.key])}",
                    credential.label,
                )
            )
    if live_secrets is not None:
        errors.extend(_live_findings(manifest, live_secrets))

    present = [credential for credential in manifest.credentials if credential.presence is Presence.PRESENT]
    unreferenced = sorted(credential.label for credential in present if not consumers[credential.key])
    removal_candidates = sorted(
        credential.label for credential in present if credential.disposition is Disposition.REMOVE_CANDIDATE
    )
    referenced_missing = sorted(
        credential.label
        for credential in manifest.credentials
        if credential.presence is Presence.REFERENCED_MISSING and consumers[credential.key]
    )
    shadowed = sorted(
        credential.label
        for credential in manifest.credentials
        if isinstance(credential, OrganizationCredential)
        and credential.presence is Presence.PRESENT
        and any(
            other.name == credential.name
            and isinstance(other, RepositoryCredential | EnvironmentCredential)
            and other.presence is Presence.PRESENT
            and _accessible_to_repository(credential, other.repository)
            for other in manifest.credentials
        )
    )
    return AuditReport(
        errors=tuple(errors),
        referenced_missing=tuple(referenced_missing),
        removal_candidates=tuple(removal_candidates),
        unreferenced=tuple(unreferenced),
        shadowed=tuple(shadowed),
        available_sources=tuple(
            sorted(c.label for c in manifest.credentials if c.source.does_not_require_owner_recovery)
        ),
        manual_sources=tuple(
            sorted(c.label for c in manifest.credentials if not c.source.does_not_require_owner_recovery)
        ),
    )


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _gh_paginated_items(endpoint: str, collection: str) -> tuple[dict, ...]:
    pages = _gh_json("api", "--paginate", "--slurp", endpoint)
    assert isinstance(pages, list)
    return tuple(item for page in pages for item in page[collection])


def github_secret_inventory(manifest: CredentialManifest) -> tuple[LiveSecret, ...]:
    """Return live organization, repository, and environment secret metadata."""
    secrets: list[LiveSecret] = []
    organization_secrets = _gh_json(
        "secret",
        "list",
        "--org",
        manifest.organization,
        "--json",
        "name,visibility",
    )
    assert isinstance(organization_secrets, list)
    for item in organization_secrets:
        repositories: tuple[str, ...] = ()
        visibility = OrganizationVisibility(item["visibility"])
        if visibility is OrganizationVisibility.SELECTED:
            selected_repositories = _gh_paginated_items(
                f"orgs/{manifest.organization}/actions/secrets/{item['name']}/repositories"
                f"?per_page={GITHUB_API_PAGE_SIZE}",
                "repositories",
            )
            repositories = tuple(sorted(repository["full_name"] for repository in selected_repositories))
        secrets.append(
            OrganizationLiveSecret(
                name=item["name"],
                visibility=visibility,
                repositories=repositories,
            )
        )

    for repository in manifest.repositories:
        repository_secrets = _gh_json(
            "secret",
            "list",
            "--repo",
            repository,
            "--json",
            "name",
        )
        assert isinstance(repository_secrets, list)
        secrets.extend(RepositoryLiveSecret(name=item["name"], repository=repository) for item in repository_secrets)
        environments = _gh_paginated_items(
            f"repos/{repository}/environments?per_page={GITHUB_API_PAGE_SIZE}",
            "environments",
        )
        for environment in environments:
            environment_secrets = _gh_json(
                "secret",
                "list",
                "--repo",
                repository,
                "--env",
                environment["name"],
                "--json",
                "name",
            )
            assert isinstance(environment_secrets, list)
            secrets.extend(
                EnvironmentLiveSecret(
                    name=item["name"],
                    repository=repository,
                    environment=environment["name"],
                )
                for item in environment_secrets
            )
    return tuple(secrets)
