# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema for GitHub Actions credentials declared in Pulumi stack config."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

SECRET_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PINNED_GCP_SECRET = re.compile(
    r"^gcp-secret://projects/(?P<project>[^/]+)/secrets/(?P<secret>[^/]+)/versions/(?P<version>[1-9][0-9]*)$"
)
CREDENTIAL_SCHEMA_VERSION = 1


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


def _organization_key(name: str) -> tuple[str, ...]:
    return (CredentialScope.ORGANIZATION, name)


def _repository_key(repository: str, name: str) -> tuple[str, ...]:
    return (CredentialScope.REPOSITORY, repository, name)


def _environment_key(repository: str, environment: str, name: str) -> tuple[str, ...]:
    return (CredentialScope.ENVIRONMENT, repository, environment, name)


class ValueSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SourceKind
    ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_gcp_version(self) -> Self:
        if self.kind is SourceKind.GCP_SECRET and not PINNED_GCP_SECRET.fullmatch(self.ref):
            raise ValueError("GCP Secret Manager sources must pin a numeric version")
        return self

    @property
    def does_not_require_owner_recovery(self) -> bool:
        return self.kind is not SourceKind.MANUAL


class Credential(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=SECRET_NAME)
    presence: Presence
    source: ValueSource
    disposition: Disposition
    note: str = ""

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if self.disposition is Disposition.REMOVE_CANDIDATE and self.presence is not Presence.PRESENT:
            raise ValueError("cannot remove a credential that is not present")
        return self

    @property
    def key(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return ":".join(self.key)


class OrganizationCredential(Credential):
    visibility: OrganizationVisibility
    repositories: tuple[str, ...] = ()
    scope: Literal[CredentialScope.ORGANIZATION] = CredentialScope.ORGANIZATION

    @field_validator("repositories")
    @classmethod
    def sort_repositories(cls, repositories: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(repositories))

    @model_validator(mode="after")
    def validate_visibility(self) -> Self:
        if self.visibility is OrganizationVisibility.SELECTED and not self.repositories:
            raise ValueError("selected visibility requires repositories")
        if self.visibility is not OrganizationVisibility.SELECTED and self.repositories:
            raise ValueError("repositories require selected visibility")
        return self

    @property
    def key(self) -> tuple[str, ...]:
        return _organization_key(self.name)


class RepositoryCredential(Credential):
    repository: str
    scope: Literal[CredentialScope.REPOSITORY] = CredentialScope.REPOSITORY

    @property
    def key(self) -> tuple[str, ...]:
        return _repository_key(self.repository, self.name)


class EnvironmentCredential(Credential):
    repository: str
    environment: str
    scope: Literal[CredentialScope.ENVIRONMENT] = CredentialScope.ENVIRONMENT

    @property
    def key(self) -> tuple[str, ...]:
        return _environment_key(self.repository, self.environment, self.name)


CredentialConfig = Annotated[
    OrganizationCredential | RepositoryCredential | EnvironmentCredential,
    Field(discriminator="scope"),
]


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
    recoverable_credentials: tuple[str, ...]
    manual_recovery_credentials: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "errors": [finding.as_dict() for finding in self.errors],
            "referenced_missing": list(self.referenced_missing),
            "removal_candidates": list(self.removal_candidates),
            "unreferenced": list(self.unreferenced),
            "shadowed": list(self.shadowed),
            "recoverable_credentials": list(self.recoverable_credentials),
            "manual_recovery_credentials": list(self.manual_recovery_credentials),
        }


def _credential_from_dict(raw: dict) -> Credential:
    normalized = dict(raw)
    normalized["source"] = {
        "kind": normalized.pop("source_kind"),
        "ref": normalized.pop("source_ref"),
    }
    return TypeAdapter(CredentialConfig).validate_python(normalized)


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
        schema_version=int(config["marin-github:schemaVersion"]),
        organization=config["marin-github:organization"],
        repositories=config["marin-github:repositories"],
        credentials=config["marin-github:credentials"],
    )
