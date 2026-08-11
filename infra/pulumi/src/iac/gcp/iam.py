# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-authoritative GCP IAM grants for the hai-gcp-models project.

Covers every human, service account, and Google-managed service-agent binding on the project,
the shared KMS key, every secret, every bucket, every Artifact Registry repo, and who can
impersonate each service account (the flip side of what an account can access). Every resource
here is a non-authoritative `*IAMMember` grant, never `*IAMBinding`/`*IAMPolicy`: a partial or
mistaken authoritative binding would revoke access for members not yet captured here.

Replaces infra/permissions's `GcpDeployPermissions`, which covered only additive deploy-account
grants; those accounts' grants are folded into `project_grants`/`kms_grants`/etc. here rather
than kept as a separate component.
"""

import base64
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

import pulumi
import pulumi_gcp as gcp
from google.cloud import kms_v1

from iac.gcp.cloud_run import resource_slug
from iac.imports import NO_IMPORTS, ImportRegistrar

# Builds the common ResourceOptions for one owned resource or IAM member grant.
OptsFor = Callable[..., pulumi.ResourceOptions]

_ConditionArgsT = TypeVar("_ConditionArgsT")


class _ConditionArgsFactory(Protocol[_ConditionArgsT]):
    """The shape shared by every `gcp.*.*ConditionArgs` class: same constructor kwargs, a
    different concrete return type per resource type (they share no base class)."""

    def __call__(self, *, title: str, expression: str, description: str | None = None) -> _ConditionArgsT: ...


@dataclass(frozen=True)
class GcpIamCondition:
    """A CEL IAM condition scoping a grant, e.g. a resource-name prefix or an expiry time."""

    title: str
    expression: str
    description: str = ""


@dataclass(frozen=True)
class GcpEncryptedMember:
    """A `user:<email>` principal whose email is KMS ciphertext (base64) in source, decrypted
    once at plan time via the same key that encrypts this stack's own secrets. Only human users
    are ever encrypted — service accounts and groups aren't personal identities and stay plain
    strings. See `_resolve_encrypted_members`."""

    ciphertext: str


@dataclass(frozen=True)
class GcpRoleGrant:
    """One role bound to a set of members, optionally scoped by an IAM condition. A member is
    either a plain principal string or a `GcpEncryptedMember`, resolved to a plain `user:<email>`
    string before any resource is declared."""

    role: str
    members: tuple[str | GcpEncryptedMember, ...]
    condition: GcpIamCondition | None = None


@dataclass(frozen=True)
class GcpSecretIam:
    secret: str
    grants: tuple[GcpRoleGrant, ...]


@dataclass(frozen=True)
class GcpBucketIam:
    bucket: str
    grants: tuple[GcpRoleGrant, ...]


@dataclass(frozen=True)
class GcpArtifactRepositoryIam:
    location: str
    repository: str
    grants: tuple[GcpRoleGrant, ...]


@dataclass(frozen=True)
class GcpServiceAccountIam:
    """Grants on the service account resource itself: who can impersonate, mint tokens for, or
    administer it — the flip side of what that account can access."""

    email: str
    grants: tuple[GcpRoleGrant, ...]


@dataclass(frozen=True)
class GcpOwnedServiceAccount:
    """A service account this stack creates and owns end-to-end, as opposed to
    `GcpServiceAccountIam`, which only grants IAM on an account created out-of-band. Protected
    (unlike the rest of this component's resources): deleting the account this stack's own CI
    preview authenticates as would break `ops-iac-preview` with no automatic repair, since a
    freshly recreated account gets a new unique ID that existing IAM bindings don't reference."""

    account_id: str
    display_name: str


@dataclass(frozen=True)
class GcpCustomRole:
    """A project-scoped custom IAM role: one of the narrow "manage IAM on resource class X"
    roles this project grants to its own deploy accounts and operators, rather than a broad
    built-in role. Only roles a Pulumi program already owns belong here — a custom role found
    live but never declared in code needs its own review before adoption, not silent capture."""

    role_id: str
    title: str
    description: str
    permissions: tuple[str, ...]


@dataclass(frozen=True)
class GcpIamArgs:
    project: str
    kms_location: str
    kms_key_ring: str
    kms_key: str
    custom_roles: tuple[GcpCustomRole, ...]
    owned_service_accounts: tuple[GcpOwnedServiceAccount, ...]
    project_grants: tuple[GcpRoleGrant, ...]
    kms_grants: tuple[GcpRoleGrant, ...]
    secrets: tuple[GcpSecretIam, ...]
    buckets: tuple[GcpBucketIam, ...]
    artifact_repositories: tuple[GcpArtifactRepositoryIam, ...]
    service_accounts: tuple[GcpServiceAccountIam, ...]


def _grant_name(prefix: str, role: str, member: str, condition: GcpIamCondition | None) -> str:
    name = f"{prefix}-{resource_slug(role)}-{resource_slug(member)}"
    return f"{name}-{resource_slug(condition.title)}" if condition else name


def _condition_suffix(condition: GcpIamCondition | None) -> str:
    return f" {condition.title}" if condition else ""


def _crypto_key_id(args: GcpIamArgs) -> str:
    return (
        f"projects/{args.project}/locations/{args.kms_location}/keyRings/{args.kms_key_ring}/"
        f"cryptoKeys/{args.kms_key}"
    )


class _KmsMemberDecryptor:
    """Decrypts `GcpEncryptedMember` ciphertext into `user:<email>` strings via the shared
    marin-iac KMS key, memoized so a person appearing in many grants costs one KMS call, not
    one per occurrence."""

    def __init__(self, crypto_key_id: str) -> None:
        self._crypto_key_id = crypto_key_id
        self._client = kms_v1.KeyManagementServiceClient()
        self._cache: dict[str, str] = {}

    def __call__(self, member: GcpEncryptedMember) -> str:
        if member.ciphertext not in self._cache:
            response = self._client.decrypt(
                name=self._crypto_key_id,
                ciphertext=base64.b64decode(member.ciphertext),
            )
            self._cache[member.ciphertext] = f"user:{response.plaintext.decode('utf-8')}"
        return self._cache[member.ciphertext]


def _resolve_grants(grants: tuple[GcpRoleGrant, ...], decrypt: _KmsMemberDecryptor) -> tuple[GcpRoleGrant, ...]:
    return tuple(
        replace(
            grant,
            members=tuple(decrypt(m) if isinstance(m, GcpEncryptedMember) else m for m in grant.members),
        )
        for grant in grants
    )


def _resolve_encrypted_members(args: GcpIamArgs, decrypt: _KmsMemberDecryptor) -> GcpIamArgs:
    """Resolve every `GcpEncryptedMember` in `args` to a plain `user:<email>` string, so every
    `_grant_*_iam` function below only ever sees plain strings, exactly as before this existed."""
    return replace(
        args,
        project_grants=_resolve_grants(args.project_grants, decrypt),
        kms_grants=_resolve_grants(args.kms_grants, decrypt),
        secrets=tuple(replace(s, grants=_resolve_grants(s.grants, decrypt)) for s in args.secrets),
        buckets=tuple(replace(b, grants=_resolve_grants(b.grants, decrypt)) for b in args.buckets),
        artifact_repositories=tuple(
            replace(r, grants=_resolve_grants(r.grants, decrypt)) for r in args.artifact_repositories
        ),
        service_accounts=tuple(replace(a, grants=_resolve_grants(a.grants, decrypt)) for a in args.service_accounts),
    )


def _grant_resource(
    name_prefix: str,
    resource_ref: str,
    grant: GcpRoleGrant,
    member: str | GcpEncryptedMember,
) -> tuple[str, str, str]:
    """The logical name, principal, and provider ID for one IAM member grant.
    Asserts `member` is already resolved to `str` — every call site runs after
    `_resolve_encrypted_members`, so this is a safety net, not the resolution step, and narrows
    the type for callers that would otherwise still see `str | GcpEncryptedMember` from the
    grant's declared type. Each `_grant_*_iam` function still owns its own resource constructor
    call directly, since the six `gcp.*.*Member` types take different kwargs and have no common
    base."""
    assert isinstance(member, str), f"unresolved encrypted member reached _grant_resource: {member!r}"
    name = _grant_name(name_prefix, grant.role, member, grant.condition)
    import_id = f"{resource_ref} {grant.role} {member}{_condition_suffix(grant.condition)}"
    return name, member, import_id


def _create_service_accounts(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> list[gcp.serviceaccount.Account]:
    created = []
    for account in args.owned_service_accounts:
        import_id = (
            f"projects/{args.project}/serviceAccounts/{account.account_id}@{args.project}.iam.gserviceaccount.com"
        )
        resource = gcp.serviceaccount.Account(
            f"account-{resource_slug(account.account_id)}",
            project=args.project,
            account_id=account.account_id,
            display_name=account.display_name,
            opts=opts_for(protect=True),
        )
        imports.register(resource, parent=parent, provider_id=import_id)
        created.append(resource)
    return created


def _create_custom_roles(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> list[gcp.projects.IAMCustomRole]:
    created = []
    for role in args.custom_roles:
        import_id = f"projects/{args.project}/roles/{role.role_id}"
        resource = gcp.projects.IAMCustomRole(
            f"role-{resource_slug(role.role_id)}",
            project=args.project,
            role_id=role.role_id,
            title=role.title,
            description=role.description,
            permissions=list(role.permissions),
            opts=opts_for(),
        )
        imports.register(resource, parent=parent, provider_id=import_id)
        created.append(resource)
    return created


def _grant_project_iam(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> None:
    for grant in args.project_grants:
        for member in grant.members:
            name, member, import_id = _grant_resource("project", args.project, grant, member)
            resource = gcp.projects.IAMMember(
                name,
                project=args.project,
                role=grant.role,
                member=member,
                condition=_condition_args(grant.condition, gcp.projects.IAMMemberConditionArgs),
                opts=opts_for(),
            )
            imports.register(resource, parent=parent, provider_id=import_id)


def _grant_kms_iam(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> None:
    crypto_key_id = _crypto_key_id(args)
    for grant in args.kms_grants:
        for member in grant.members:
            name, member, import_id = _grant_resource("kms", crypto_key_id, grant, member)
            resource = gcp.kms.CryptoKeyIAMMember(
                name,
                crypto_key_id=crypto_key_id,
                role=grant.role,
                member=member,
                condition=_condition_args(grant.condition, gcp.kms.CryptoKeyIAMMemberConditionArgs),
                opts=opts_for(),
            )
            imports.register(resource, parent=parent, provider_id=import_id)


def _grant_secret_iam(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> None:
    for secret in args.secrets:
        secret_id = f"projects/{args.project}/secrets/{secret.secret}"
        for grant in secret.grants:
            for member in grant.members:
                name, member, import_id = _grant_resource(
                    f"secret-{resource_slug(secret.secret)}", secret_id, grant, member
                )
                resource = gcp.secretmanager.SecretIamMember(
                    name,
                    project=args.project,
                    secret_id=secret.secret,
                    role=grant.role,
                    member=member,
                    condition=_condition_args(grant.condition, gcp.secretmanager.SecretIamMemberConditionArgs),
                    opts=opts_for(),
                )
                imports.register(resource, parent=parent, provider_id=import_id)


def _grant_bucket_iam(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> None:
    for bucket in args.buckets:
        for grant in bucket.grants:
            for member in grant.members:
                name, member, import_id = _grant_resource(
                    f"bucket-{resource_slug(bucket.bucket)}", f"b/{bucket.bucket}", grant, member
                )
                resource = gcp.storage.BucketIAMMember(
                    name,
                    bucket=bucket.bucket,
                    role=grant.role,
                    member=member,
                    condition=_condition_args(grant.condition, gcp.storage.BucketIAMMemberConditionArgs),
                    opts=opts_for(),
                )
                imports.register(resource, parent=parent, provider_id=import_id)


def _grant_artifact_repository_iam(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> None:
    for repo in args.artifact_repositories:
        repo_path = f"projects/{args.project}/locations/{repo.location}/repositories/{repo.repository}"
        for grant in repo.grants:
            for member in grant.members:
                name, member, import_id = _grant_resource(
                    f"ar-{resource_slug(repo.location)}-{resource_slug(repo.repository)}",
                    repo_path,
                    grant,
                    member,
                )
                resource = gcp.artifactregistry.RepositoryIamMember(
                    name,
                    project=args.project,
                    location=repo.location,
                    repository=repo.repository,
                    role=grant.role,
                    member=member,
                    condition=_condition_args(grant.condition, gcp.artifactregistry.RepositoryIamMemberConditionArgs),
                    opts=opts_for(),
                )
                imports.register(resource, parent=parent, provider_id=import_id)


def _grant_service_account_iam(
    args: GcpIamArgs,
    opts_for: OptsFor,
    imports: ImportRegistrar,
    parent: pulumi.Resource,
) -> None:
    for account in args.service_accounts:
        service_account_id = f"projects/{args.project}/serviceAccounts/{account.email}"
        account_local = account.email.split("@", 1)[0]
        for grant in account.grants:
            for member in grant.members:
                name, member, import_id = _grant_resource(
                    f"sa-{resource_slug(account_local)}", service_account_id, grant, member
                )
                resource = gcp.serviceaccount.IAMMember(
                    name,
                    service_account_id=service_account_id,
                    role=grant.role,
                    member=member,
                    condition=_condition_args(grant.condition, gcp.serviceaccount.IAMMemberConditionArgs),
                    opts=opts_for(),
                )
                imports.register(resource, parent=parent, provider_id=import_id)


def _condition_args(
    condition: GcpIamCondition | None, args_cls: _ConditionArgsFactory[_ConditionArgsT]
) -> _ConditionArgsT | None:
    if condition is None:
        return None
    return args_cls(title=condition.title, expression=condition.expression, description=condition.description or None)


class GcpIam(pulumi.ComponentResource):
    """Every non-authoritative IAM grant on the hai-gcp-models project.

    Deliberately unprotected (unlike `GcpDeployPermissions`'s core deploy-auth resources):
    revoking an overbroad or stale grant found here is meant to be a plain code deletion plus
    `pulumi up`, not a `pulumi state unprotect` dance first.
    """

    def __init__(
        self,
        name: str,
        args: GcpIamArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:GcpIam", name, None, opts)

        # Decrypt every GcpEncryptedMember before declaring any resource, so everything below
        # only ever handles plain strings.
        args = _resolve_encrypted_members(args, _KmsMemberDecryptor(_crypto_key_id(args)))

        def opts_for(
            *,
            depends_on: list[pulumi.Resource] | None = None,
            protect: bool = False,
        ) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=gcp_provider,
                depends_on=depends_on,
                protect=protect,
            )

        # Owned service accounts and custom role definitions first: a grant referencing either
        # by name needs it to already exist on a from-scratch `up`. `depends_on` (not Python call
        # order, which Pulumi's engine ignores) is what actually enforces that — every grant
        # waits on both, mirroring GcpDeployPermissions's created_accounts -> grant_opts pattern.
        created_accounts = _create_service_accounts(args, opts_for, imports, self)
        created_roles = _create_custom_roles(args, opts_for, imports, self)

        def grant_opts_for() -> pulumi.ResourceOptions:
            return opts_for(depends_on=created_accounts + created_roles)

        _grant_project_iam(args, grant_opts_for, imports, self)
        _grant_kms_iam(args, grant_opts_for, imports, self)
        _grant_secret_iam(args, grant_opts_for, imports, self)
        _grant_bucket_iam(args, grant_opts_for, imports, self)
        _grant_artifact_repository_iam(args, grant_opts_for, imports, self)
        _grant_service_account_iam(args, grant_opts_for, imports, self)

        self.register_outputs({})
