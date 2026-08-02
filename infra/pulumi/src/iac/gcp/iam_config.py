# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load and edit the machine-readable GCP IAM declaration."""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpBucketIam,
    GcpCustomRole,
    GcpEncryptedMember,
    GcpIamCondition,
    GcpOwnedServiceAccount,
    GcpRoleGrant,
    GcpSecretIam,
    GcpServiceAccountIam,
)

IAM_DATA_PATH = Path(__file__).with_name("iam_data.yaml")

_PRINCIPAL_ID_RE = re.compile(r"human-(\d{3,})")
_SCHEMA_VERSION = 1
_PLAIN_MEMBER_PREFIXES = (
    "domain:",
    "group:",
    "principal://",
    "principalSet://",
    "projectEditor:",
    "projectOwner:",
    "projectViewer:",
    "serviceAccount:",
)
_PLAIN_SPECIAL_MEMBERS = frozenset({"allAuthenticatedUsers", "allUsers"})
_YAML_HEADER = """\
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Canonical non-authoritative GCP IAM declaration for hai-gcp-models. Human
# user principals are KMS ciphertexts declared once under `principals`; grants
# reference their opaque IDs so one principal cannot drift across roles.
#
# The owned service account and custom roles came from the retired
# infra/permissions project. Do not add custom roles found live unless Pulumi
# already owns them or their adoption has been reviewed separately.
#
# GCP does not support resource-based IAM Conditions for iam.googleapis.com
# permissions. Custom-role-management grants therefore remain project-wide.
"""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: MappingNode, deep: bool = False) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError(f"unhashable YAML mapping key at line {key_node.start_mark.line + 1}") from error
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class GcpPrincipal:
    """One opaque ID and its KMS-encrypted `user:<email>` principal."""

    principal_id: str
    ciphertext: str


@dataclass(frozen=True)
class GcpIamConfig:
    """The complete checked-in IAM declaration before Pulumi stack settings."""

    kms_location: str
    kms_key_ring: str
    kms_key: str
    principals: tuple[GcpPrincipal, ...]
    custom_roles: tuple[GcpCustomRole, ...]
    owned_service_accounts: tuple[GcpOwnedServiceAccount, ...]
    project_grants: tuple[GcpRoleGrant, ...]
    kms_grants: tuple[GcpRoleGrant, ...]
    secrets: tuple[GcpSecretIam, ...]
    buckets: tuple[GcpBucketIam, ...]
    artifact_repositories: tuple[GcpArtifactRepositoryIam, ...]
    service_accounts: tuple[GcpServiceAccountIam, ...]


@dataclass(frozen=True)
class GcpPrincipalRegistration:
    """The config and opaque principal selected by a registration request."""

    config: GcpIamConfig
    principal: GcpPrincipal
    created: bool


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{path} must be an integer")
    return value


def _fields(
    value: object,
    path: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, object]:
    mapping = _mapping(value, path)
    missing = required - mapping.keys()
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(sorted(missing))}")
    unknown = mapping.keys() - required - optional
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
    return mapping


def _parse_principals(value: object) -> tuple[GcpPrincipal, ...]:
    principals = _mapping(value, "principals")
    parsed = tuple(
        GcpPrincipal(principal_id=principal_id, ciphertext=_string(ciphertext, f"principals.{principal_id}"))
        for principal_id, ciphertext in principals.items()
    )
    for principal in parsed:
        if _PRINCIPAL_ID_RE.fullmatch(principal.principal_id) is None:
            raise ValueError(f"invalid principal ID {principal.principal_id!r}; expected human-NNN")
    ciphertexts = [principal.ciphertext for principal in parsed]
    if len(ciphertexts) != len(set(ciphertexts)):
        raise ValueError("principals contains duplicate ciphertexts")
    return parsed


def _parse_condition(value: object, path: str) -> GcpIamCondition:
    fields = _fields(
        value,
        path,
        required=frozenset({"title", "expression"}),
        optional=frozenset({"description"}),
    )
    return GcpIamCondition(
        title=_string(fields["title"], f"{path}.title"),
        expression=_string(fields["expression"], f"{path}.expression"),
        description=_string(fields.get("description", ""), f"{path}.description"),
    )


def _validate_plain_member(member: str, path: str) -> None:
    if member.startswith("user:"):
        raise ValueError(f"{path} contains a plaintext user principal")
    if member not in _PLAIN_SPECIAL_MEMBERS and not member.startswith(_PLAIN_MEMBER_PREFIXES):
        raise ValueError(f"{path} contains an invalid plain IAM member {member!r}")


def _parse_member(value: object, path: str, principals: dict[str, GcpEncryptedMember]) -> str | GcpEncryptedMember:
    if isinstance(value, str):
        _validate_plain_member(value, path)
        return value
    fields = _fields(value, path, required=frozenset({"principal"}))
    principal_id = _string(fields["principal"], f"{path}.principal")
    try:
        return principals[principal_id]
    except KeyError as error:
        raise ValueError(f"{path} references unknown principal {principal_id!r}") from error


def _parse_grant(value: object, path: str, principals: dict[str, GcpEncryptedMember]) -> GcpRoleGrant:
    fields = _fields(
        value,
        path,
        required=frozenset({"role", "members"}),
        optional=frozenset({"condition"}),
    )
    members = tuple(
        _parse_member(member, f"{path}.members[{index}]", principals)
        for index, member in enumerate(_sequence(fields["members"], f"{path}.members"))
    )
    if not members:
        raise ValueError(f"{path}.members must not be empty")
    if len(members) != len(set(members)):
        raise ValueError(f"{path}.members contains duplicates")
    condition_value = fields.get("condition")
    condition = _parse_condition(condition_value, f"{path}.condition") if condition_value is not None else None
    return GcpRoleGrant(
        role=_string(fields["role"], f"{path}.role"),
        members=members,
        condition=condition,
    )


def _parse_grants(value: object, path: str, principals: dict[str, GcpEncryptedMember]) -> tuple[GcpRoleGrant, ...]:
    return tuple(
        _parse_grant(grant, f"{path}[{index}]", principals) for index, grant in enumerate(_sequence(value, path))
    )


def _parse_custom_roles(value: object) -> tuple[GcpCustomRole, ...]:
    roles = []
    for index, raw_role in enumerate(_sequence(value, "custom_roles")):
        path = f"custom_roles[{index}]"
        fields = _fields(
            raw_role,
            path,
            required=frozenset({"role_id", "title", "description", "permissions"}),
        )
        roles.append(
            GcpCustomRole(
                role_id=_string(fields["role_id"], f"{path}.role_id"),
                title=_string(fields["title"], f"{path}.title"),
                description=_string(fields["description"], f"{path}.description"),
                permissions=tuple(
                    _string(permission, f"{path}.permissions[{permission_index}]")
                    for permission_index, permission in enumerate(
                        _sequence(fields["permissions"], f"{path}.permissions")
                    )
                ),
            )
        )
    return tuple(roles)


def _parse_owned_service_accounts(value: object) -> tuple[GcpOwnedServiceAccount, ...]:
    accounts = []
    for index, raw_account in enumerate(_sequence(value, "owned_service_accounts")):
        path = f"owned_service_accounts[{index}]"
        fields = _fields(raw_account, path, required=frozenset({"account_id", "display_name"}))
        accounts.append(
            GcpOwnedServiceAccount(
                account_id=_string(fields["account_id"], f"{path}.account_id"),
                display_name=_string(fields["display_name"], f"{path}.display_name"),
            )
        )
    return tuple(accounts)


def _parse_secrets(value: object, principals: dict[str, GcpEncryptedMember]) -> tuple[GcpSecretIam, ...]:
    secrets = []
    for index, raw_secret in enumerate(_sequence(value, "secrets")):
        path = f"secrets[{index}]"
        fields = _fields(raw_secret, path, required=frozenset({"secret", "grants"}))
        secrets.append(
            GcpSecretIam(
                secret=_string(fields["secret"], f"{path}.secret"),
                grants=_parse_grants(fields["grants"], f"{path}.grants", principals),
            )
        )
    return tuple(secrets)


def _parse_buckets(value: object, principals: dict[str, GcpEncryptedMember]) -> tuple[GcpBucketIam, ...]:
    buckets = []
    for index, raw_bucket in enumerate(_sequence(value, "buckets")):
        path = f"buckets[{index}]"
        fields = _fields(raw_bucket, path, required=frozenset({"bucket", "grants"}))
        buckets.append(
            GcpBucketIam(
                bucket=_string(fields["bucket"], f"{path}.bucket"),
                grants=_parse_grants(fields["grants"], f"{path}.grants", principals),
            )
        )
    return tuple(buckets)


def _parse_artifact_repositories(
    value: object, principals: dict[str, GcpEncryptedMember]
) -> tuple[GcpArtifactRepositoryIam, ...]:
    repositories = []
    for index, raw_repository in enumerate(_sequence(value, "artifact_repositories")):
        path = f"artifact_repositories[{index}]"
        fields = _fields(raw_repository, path, required=frozenset({"location", "repository", "grants"}))
        repositories.append(
            GcpArtifactRepositoryIam(
                location=_string(fields["location"], f"{path}.location"),
                repository=_string(fields["repository"], f"{path}.repository"),
                grants=_parse_grants(fields["grants"], f"{path}.grants", principals),
            )
        )
    return tuple(repositories)


def _parse_service_accounts(
    value: object, principals: dict[str, GcpEncryptedMember]
) -> tuple[GcpServiceAccountIam, ...]:
    accounts = []
    for index, raw_account in enumerate(_sequence(value, "service_accounts")):
        path = f"service_accounts[{index}]"
        fields = _fields(raw_account, path, required=frozenset({"email", "grants"}))
        accounts.append(
            GcpServiceAccountIam(
                email=_string(fields["email"], f"{path}.email"),
                grants=_parse_grants(fields["grants"], f"{path}.grants", principals),
            )
        )
    return tuple(accounts)


def load_iam_config(path: Path = IAM_DATA_PATH) -> GcpIamConfig:
    """Load and validate the complete IAM declaration."""
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    fields = _fields(
        raw,
        str(path),
        required=frozenset(
            {
                "kms",
                "schema_version",
                "principals",
                "custom_roles",
                "owned_service_accounts",
                "project_grants",
                "kms_grants",
                "secrets",
                "buckets",
                "artifact_repositories",
                "service_accounts",
            }
        ),
    )
    schema_version = _integer(fields["schema_version"], "schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported IAM schema version {schema_version}; expected {_SCHEMA_VERSION}")
    kms = _fields(
        fields["kms"],
        "kms",
        required=frozenset({"location", "key_ring", "key"}),
    )
    principal_records = _parse_principals(fields["principals"])
    principals = {principal.principal_id: GcpEncryptedMember(principal.ciphertext) for principal in principal_records}
    return GcpIamConfig(
        kms_location=_string(kms["location"], "kms.location"),
        kms_key_ring=_string(kms["key_ring"], "kms.key_ring"),
        kms_key=_string(kms["key"], "kms.key"),
        principals=principal_records,
        custom_roles=_parse_custom_roles(fields["custom_roles"]),
        owned_service_accounts=_parse_owned_service_accounts(fields["owned_service_accounts"]),
        project_grants=_parse_grants(fields["project_grants"], "project_grants", principals),
        kms_grants=_parse_grants(fields["kms_grants"], "kms_grants", principals),
        secrets=_parse_secrets(fields["secrets"], principals),
        buckets=_parse_buckets(fields["buckets"], principals),
        artifact_repositories=_parse_artifact_repositories(fields["artifact_repositories"], principals),
        service_accounts=_parse_service_accounts(fields["service_accounts"], principals),
    )


def _condition_data(condition: GcpIamCondition) -> dict[str, str]:
    data = {"title": condition.title, "expression": condition.expression}
    if condition.description:
        data["description"] = condition.description
    return data


def _member_data(member: str | GcpEncryptedMember, principal_ids: dict[str, str]) -> str | dict[str, str]:
    if isinstance(member, str):
        _validate_plain_member(member, "grant")
        return member
    try:
        return {"principal": principal_ids[member.ciphertext]}
    except KeyError as error:
        raise ValueError("grant contains an encrypted member missing from principals") from error


def _grant_data(grant: GcpRoleGrant, principal_ids: dict[str, str]) -> dict[str, object]:
    data: dict[str, object] = {
        "role": grant.role,
        "members": [_member_data(member, principal_ids) for member in grant.members],
    }
    if grant.condition is not None:
        data["condition"] = _condition_data(grant.condition)
    return data


def _grants_data(grants: tuple[GcpRoleGrant, ...], principal_ids: dict[str, str]) -> list[dict[str, object]]:
    return [_grant_data(grant, principal_ids) for grant in grants]


def iam_config_data(config: GcpIamConfig) -> dict[str, object]:
    """Return the canonical YAML-compatible representation."""
    principal_ids = {principal.ciphertext: principal.principal_id for principal in config.principals}
    if len(principal_ids) != len(config.principals):
        raise ValueError("principals contains duplicate ciphertexts")
    return {
        "schema_version": _SCHEMA_VERSION,
        "kms": {
            "location": config.kms_location,
            "key_ring": config.kms_key_ring,
            "key": config.kms_key,
        },
        "principals": {principal.principal_id: principal.ciphertext for principal in config.principals},
        "custom_roles": [
            {
                "role_id": role.role_id,
                "title": role.title,
                "description": role.description,
                "permissions": list(role.permissions),
            }
            for role in config.custom_roles
        ],
        "owned_service_accounts": [
            {"account_id": account.account_id, "display_name": account.display_name}
            for account in config.owned_service_accounts
        ],
        "project_grants": _grants_data(config.project_grants, principal_ids),
        "kms_grants": _grants_data(config.kms_grants, principal_ids),
        "secrets": [
            {
                "secret": secret.secret,
                "grants": _grants_data(secret.grants, principal_ids),
            }
            for secret in config.secrets
        ],
        "buckets": [
            {
                "bucket": bucket.bucket,
                "grants": _grants_data(bucket.grants, principal_ids),
            }
            for bucket in config.buckets
        ],
        "artifact_repositories": [
            {
                "location": repository.location,
                "repository": repository.repository,
                "grants": _grants_data(repository.grants, principal_ids),
            }
            for repository in config.artifact_repositories
        ],
        "service_accounts": [
            {
                "email": account.email,
                "grants": _grants_data(account.grants, principal_ids),
            }
            for account in config.service_accounts
        ],
    }


def write_iam_config(config: GcpIamConfig, path: Path = IAM_DATA_PATH) -> None:
    """Write a deterministic machine-editable IAM declaration."""
    yaml_text = yaml.safe_dump(
        iam_config_data(config),
        sort_keys=False,
        width=120,
        default_flow_style=False,
    )
    path.write_text(f"{_YAML_HEADER}{yaml_text}", encoding="utf-8")


def _replace_principal_grants(
    grants: tuple[GcpRoleGrant, ...], replacements: dict[str, str]
) -> tuple[GcpRoleGrant, ...]:
    return tuple(
        replace(
            grant,
            members=tuple(
                GcpEncryptedMember(replacements[member.ciphertext]) if isinstance(member, GcpEncryptedMember) else member
                for member in grant.members
            ),
        )
        for grant in grants
    )


def replace_principals(config: GcpIamConfig, principals: tuple[GcpPrincipal, ...]) -> GcpIamConfig:
    """Replace registry ciphertexts while preserving every grant reference."""
    old_by_id = {principal.principal_id: principal for principal in config.principals}
    new_by_id = {principal.principal_id: principal for principal in principals}
    if old_by_id.keys() != new_by_id.keys():
        raise ValueError("replacement principals must have the same IDs")
    replacements = {old_by_id[principal_id].ciphertext: new_by_id[principal_id].ciphertext for principal_id in old_by_id}
    return replace(
        config,
        principals=principals,
        project_grants=_replace_principal_grants(config.project_grants, replacements),
        kms_grants=_replace_principal_grants(config.kms_grants, replacements),
        secrets=tuple(
            replace(
                secret,
                grants=_replace_principal_grants(secret.grants, replacements),
            )
            for secret in config.secrets
        ),
        buckets=tuple(
            replace(
                bucket,
                grants=_replace_principal_grants(bucket.grants, replacements),
            )
            for bucket in config.buckets
        ),
        artifact_repositories=tuple(
            replace(
                repository,
                grants=_replace_principal_grants(repository.grants, replacements),
            )
            for repository in config.artifact_repositories
        ),
        service_accounts=tuple(
            replace(
                account,
                grants=_replace_principal_grants(account.grants, replacements),
            )
            for account in config.service_accounts
        ),
    )


def _next_principal_id(principals: tuple[GcpPrincipal, ...]) -> str:
    numbers = [
        int(match.group(1))
        for principal in principals
        if (match := _PRINCIPAL_ID_RE.fullmatch(principal.principal_id)) is not None
    ]
    return f"human-{max(numbers, default=0) + 1:03d}"


def register_principal(
    config: GcpIamConfig,
    email: str,
    *,
    decrypt_ciphertext: Callable[[str], str],
    encrypt_email: Callable[[str], str],
) -> GcpPrincipalRegistration:
    """Find one email's principal or add it to the encrypted registry."""
    normalized_email = f"user:{email.removeprefix('user:').strip()}"
    if "@" not in normalized_email:
        raise ValueError(f"expected an email, got {email!r}")

    principal = next(
        (candidate for candidate in config.principals if decrypt_ciphertext(candidate.ciphertext) == normalized_email),
        None,
    )
    if principal is None:
        principal = GcpPrincipal(
            principal_id=_next_principal_id(config.principals),
            ciphertext=encrypt_email(normalized_email.removeprefix("user:")),
        )
        config = replace(config, principals=(*config.principals, principal))
        return GcpPrincipalRegistration(config=config, principal=principal, created=True)
    return GcpPrincipalRegistration(config=config, principal=principal, created=False)


def grant_project_roles(
    config: GcpIamConfig,
    email: str,
    roles: tuple[str, ...],
    *,
    decrypt_ciphertext: Callable[[str], str],
    encrypt_email: Callable[[str], str],
) -> GcpIamConfig:
    """Grant project roles to one email, reusing or creating one encrypted principal."""
    if not roles:
        raise ValueError("at least one project role is required")
    registration = register_principal(
        config,
        email,
        decrypt_ciphertext=decrypt_ciphertext,
        encrypt_email=encrypt_email,
    )
    config = registration.config
    principal = registration.principal

    encrypted_member = GcpEncryptedMember(principal.ciphertext)
    project_grants = list(config.project_grants)
    for role in dict.fromkeys(roles):
        matches = [index for index, grant in enumerate(project_grants) if grant.role == role and grant.condition is None]
        if len(matches) > 1:
            raise ValueError(f"project role {role!r} has multiple unconditional grants")
        if not matches:
            project_grants.append(GcpRoleGrant(role=role, members=(encrypted_member,)))
            continue
        index = matches[0]
        grant = project_grants[index]
        if encrypted_member not in grant.members:
            project_grants[index] = replace(grant, members=(*grant.members, encrypted_member))
    project_grants.sort(key=lambda grant: (grant.role, grant.condition.title if grant.condition else ""))
    return replace(config, project_grants=tuple(project_grants))
