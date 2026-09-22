# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM grants owned by the ``marin`` stack for the hai-gcp-models project.

Covers every human, service account, and Google-managed service-agent binding on the project,
the shared KMS key, every secret, every bucket, every Artifact Registry repo, backend-service
and Cloud Run IAP policies, and who can impersonate each service account. This is the sole
Pulumi owner for GCP IAM grants. Each ``*IAMBinding`` owns the complete member set for one role
and optional condition on its target resource.
"""

import base64
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

import pulumi
import pulumi_gcp as gcp
from google.cloud import kms_v1

from iac.gcp.cloud_run import resource_slug
from iac.imports import NO_IMPORTS, ImportRegistrar

_ConditionArgsT = TypeVar("_ConditionArgsT")
_TargetT = TypeVar("_TargetT", bound=Hashable)


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
class GcpBackendServiceIapIam:
    service: str
    iap_grants: tuple[GcpRoleGrant, ...]


@dataclass(frozen=True)
class GcpCloudRunIapIam:
    """IAP grants on a Cloud Run service, plus grants on the service itself (the IAP service
    agent's ``roles/run.invoker``, which IAP needs to forward admitted requests)."""

    location: str
    service: str
    iap_grants: tuple[GcpRoleGrant, ...]
    service_grants: tuple[GcpRoleGrant, ...] = ()


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
class GcpIamGrantSet:
    """IAM declarations owned by one deploy target and applied by the global stack."""

    project_grants: tuple[GcpRoleGrant, ...] = ()
    kms_grants: tuple[GcpRoleGrant, ...] = ()
    secrets: tuple[GcpSecretIam, ...] = ()
    buckets: tuple[GcpBucketIam, ...] = ()
    artifact_repositories: tuple[GcpArtifactRepositoryIam, ...] = ()
    service_accounts: tuple[GcpServiceAccountIam, ...] = ()
    backend_service_iap: tuple[GcpBackendServiceIapIam, ...] = ()
    cloud_run_iap: tuple[GcpCloudRunIapIam, ...] = ()


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
    backend_service_iap: tuple[GcpBackendServiceIapIam, ...]
    cloud_run_iap: tuple[GcpCloudRunIapIam, ...]


def merge_iam_grant_sets(args: GcpIamArgs, grant_sets: tuple[GcpIamGrantSet, ...]) -> GcpIamArgs:
    """Add deploy-target declarations to the global IAM resource graph."""
    return replace(
        args,
        project_grants=args.project_grants
        + tuple(grant for grant_set in grant_sets for grant in grant_set.project_grants),
        kms_grants=args.kms_grants + tuple(grant for grant_set in grant_sets for grant in grant_set.kms_grants),
        secrets=args.secrets + tuple(secret for grant_set in grant_sets for secret in grant_set.secrets),
        buckets=args.buckets + tuple(bucket for grant_set in grant_sets for bucket in grant_set.buckets),
        artifact_repositories=args.artifact_repositories
        + tuple(repository for grant_set in grant_sets for repository in grant_set.artifact_repositories),
        service_accounts=args.service_accounts
        + tuple(account for grant_set in grant_sets for account in grant_set.service_accounts),
        backend_service_iap=args.backend_service_iap
        + tuple(service for grant_set in grant_sets for service in grant_set.backend_service_iap),
        cloud_run_iap=args.cloud_run_iap
        + tuple(service for grant_set in grant_sets for service in grant_set.cloud_run_iap),
    )


@dataclass(frozen=True)
class _BindingDeclaration:
    logical_name: str
    members: tuple[str, ...]
    provider_id: str


@dataclass(frozen=True)
class _GcpIamContext:
    args: GcpIamArgs
    imports: ImportRegistrar
    provider: pulumi.ProviderResource
    parent: pulumi.Resource
    dependencies: tuple[pulumi.Resource, ...] = ()

    def options(self, *, protect: bool = False) -> pulumi.ResourceOptions:
        return pulumi.ResourceOptions(
            parent=self.parent,
            provider=self.provider,
            depends_on=list(self.dependencies),
            protect=protect,
        )

    def register(self, resource: pulumi.Resource, provider_id: str) -> None:
        self.imports.register(resource, parent=self.parent, provider_id=provider_id)


def _binding_name(prefix: str, role: str, condition: GcpIamCondition | None) -> str:
    name = f"{prefix}-{resource_slug(role)}"
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
        backend_service_iap=tuple(
            replace(
                service,
                iap_grants=_resolve_grants(service.iap_grants, decrypt),
            )
            for service in args.backend_service_iap
        ),
        cloud_run_iap=tuple(
            replace(
                service,
                iap_grants=_resolve_grants(service.iap_grants, decrypt),
            )
            for service in args.cloud_run_iap
        ),
    )


def _role_bindings(grants: tuple[GcpRoleGrant, ...]) -> tuple[GcpRoleGrant, ...]:
    """Merge declarations that target the same authoritative role and condition."""
    bindings: dict[tuple[str, str | None], tuple[GcpIamCondition | None, list[str]]] = {}
    for grant in grants:
        condition_title = grant.condition.title if grant.condition else None
        key = (grant.role, condition_title)
        condition, members = bindings.setdefault(key, (grant.condition, []))
        if condition != grant.condition:
            raise ValueError(f"conflicting condition {condition_title!r} for authoritative binding {grant.role!r}")
        for member in grant.members:
            assert isinstance(member, str), f"unresolved encrypted member reached _role_bindings: {member!r}"
            if member in members:
                raise ValueError(f"duplicate member {member!r} for authoritative binding {grant.role!r}")
            members.append(member)
    return tuple(
        GcpRoleGrant(role=role, members=tuple(members), condition=condition)
        for (role, _condition_title), (condition, members) in bindings.items()
    )


def _group_target_grants(
    declarations: Iterable[tuple[_TargetT, tuple[GcpRoleGrant, ...]]],
) -> tuple[tuple[_TargetT, tuple[GcpRoleGrant, ...]], ...]:
    """Merge grant sets declared for the same target resource."""
    grants_by_target: dict[_TargetT, list[GcpRoleGrant]] = {}
    for target, grants in declarations:
        grants_by_target.setdefault(target, []).extend(grants)
    return tuple((target, tuple(grants)) for target, grants in grants_by_target.items())


def _binding_declaration(
    name_prefix: str,
    resource_ref: str,
    grant: GcpRoleGrant,
) -> _BindingDeclaration:
    """Build one role-authoritative binding and its provider import ID."""
    assert all(
        isinstance(member, str) for member in grant.members
    ), f"unresolved encrypted member reached _binding_declaration: {grant.members!r}"
    name = _binding_name(name_prefix, grant.role, grant.condition)
    import_id = f"{resource_ref} {grant.role}{_condition_suffix(grant.condition)}"
    return _BindingDeclaration(logical_name=name, members=grant.members, provider_id=import_id)


def _create_service_accounts(context: _GcpIamContext) -> list[gcp.serviceaccount.Account]:
    created = []
    for account in context.args.owned_service_accounts:
        import_id = (
            f"projects/{context.args.project}/serviceAccounts/"
            f"{account.account_id}@{context.args.project}.iam.gserviceaccount.com"
        )
        resource = gcp.serviceaccount.Account(
            f"account-{resource_slug(account.account_id)}",
            project=context.args.project,
            account_id=account.account_id,
            display_name=account.display_name,
            opts=context.options(protect=True),
        )
        context.register(resource, import_id)
        created.append(resource)
    return created


def _create_custom_roles(context: _GcpIamContext) -> list[gcp.projects.IAMCustomRole]:
    created = []
    for role in context.args.custom_roles:
        import_id = f"projects/{context.args.project}/roles/{role.role_id}"
        resource = gcp.projects.IAMCustomRole(
            f"role-{resource_slug(role.role_id)}",
            project=context.args.project,
            role_id=role.role_id,
            title=role.title,
            description=role.description,
            permissions=list(role.permissions),
            opts=context.options(),
        )
        context.register(resource, import_id)
        created.append(resource)
    return created


def _grant_project_iam(context: _GcpIamContext) -> None:
    for grant in _role_bindings(context.args.project_grants):
        declaration = _binding_declaration("project", context.args.project, grant)
        resource = gcp.projects.IAMBinding(
            declaration.logical_name,
            project=context.args.project,
            role=grant.role,
            members=list(declaration.members),
            condition=_condition_args(grant.condition, gcp.projects.IAMBindingConditionArgs),
            opts=context.options(),
        )
        context.register(resource, declaration.provider_id)


def _grant_kms_iam(context: _GcpIamContext) -> None:
    crypto_key_id = _crypto_key_id(context.args)
    for grant in _role_bindings(context.args.kms_grants):
        declaration = _binding_declaration("kms", crypto_key_id, grant)
        resource = gcp.kms.CryptoKeyIAMBinding(
            declaration.logical_name,
            crypto_key_id=crypto_key_id,
            role=grant.role,
            members=list(declaration.members),
            condition=_condition_args(grant.condition, gcp.kms.CryptoKeyIAMBindingConditionArgs),
            opts=context.options(),
        )
        context.register(resource, declaration.provider_id)


def _grant_secret_iam(context: _GcpIamContext) -> None:
    targets = _group_target_grants((secret.secret, secret.grants) for secret in context.args.secrets)
    for secret, grants in targets:
        secret_id = f"projects/{context.args.project}/secrets/{secret}"
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(f"secret-{resource_slug(secret)}", secret_id, grant)
            resource = gcp.secretmanager.SecretIamBinding(
                declaration.logical_name,
                project=context.args.project,
                secret_id=secret,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(grant.condition, gcp.secretmanager.SecretIamBindingConditionArgs),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _grant_bucket_iam(context: _GcpIamContext) -> None:
    targets = _group_target_grants((bucket.bucket, bucket.grants) for bucket in context.args.buckets)
    for bucket, grants in targets:
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(f"bucket-{resource_slug(bucket)}", f"b/{bucket}", grant)
            resource = gcp.storage.BucketIAMBinding(
                declaration.logical_name,
                bucket=bucket,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(grant.condition, gcp.storage.BucketIAMBindingConditionArgs),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _grant_artifact_repository_iam(context: _GcpIamContext) -> None:
    targets = _group_target_grants(
        ((repository.location, repository.repository), repository.grants)
        for repository in context.args.artifact_repositories
    )
    for (location, repository), grants in targets:
        repo_path = f"projects/{context.args.project}/locations/{location}/repositories/{repository}"
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(
                f"ar-{resource_slug(location)}-{resource_slug(repository)}",
                repo_path,
                grant,
            )
            resource = gcp.artifactregistry.RepositoryIamBinding(
                declaration.logical_name,
                project=context.args.project,
                location=location,
                repository=repository,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(grant.condition, gcp.artifactregistry.RepositoryIamBindingConditionArgs),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _grant_service_account_iam(context: _GcpIamContext) -> None:
    targets = _group_target_grants((account.email, account.grants) for account in context.args.service_accounts)
    for account_email, grants in targets:
        service_account_id = f"projects/{context.args.project}/serviceAccounts/{account_email}"
        account_local = account_email.split("@", 1)[0]
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(f"sa-{resource_slug(account_local)}", service_account_id, grant)
            resource = gcp.serviceaccount.IAMBinding(
                declaration.logical_name,
                service_account_id=service_account_id,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(grant.condition, gcp.serviceaccount.IAMBindingConditionArgs),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _grant_cloud_run_iap(context: _GcpIamContext) -> None:
    targets = _group_target_grants(
        ((service.location, service.service), service.iap_grants) for service in context.args.cloud_run_iap
    )
    for (location, service), grants in targets:
        iap_service_id = f"projects/{context.args.project}/iap_web/cloud_run-{location}/services/{service}"
        iap_prefix = f"iap-run-service-{resource_slug(location)}-{resource_slug(service)}"
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(iap_prefix, iap_service_id, grant)
            resource = gcp.iap.WebCloudRunServiceIamBinding(
                declaration.logical_name,
                project=context.args.project,
                location=location,
                cloud_run_service_name=service,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(
                    grant.condition,
                    gcp.iap.WebCloudRunServiceIamBindingConditionArgs,
                ),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _grant_cloud_run_services(context: _GcpIamContext) -> None:
    targets = _group_target_grants(
        ((service.location, service.service), service.service_grants) for service in context.args.cloud_run_iap
    )
    for (location, service), grants in targets:
        service_id = f"projects/{context.args.project}/locations/{location}/services/{service}"
        prefix = f"run-service-{resource_slug(location)}-{resource_slug(service)}"
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(prefix, service_id, grant)
            resource = gcp.cloudrunv2.ServiceIamBinding(
                declaration.logical_name,
                project=context.args.project,
                location=location,
                name=service,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(grant.condition, gcp.cloudrunv2.ServiceIamBindingConditionArgs),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _grant_backend_service_iap(context: _GcpIamContext) -> None:
    targets = _group_target_grants((service.service, service.iap_grants) for service in context.args.backend_service_iap)
    for service, grants in targets:
        iap_service_id = f"projects/{context.args.project}/iap_web/compute/services/{service}"
        iap_prefix = f"iap-backend-service-{resource_slug(service)}"
        for grant in _role_bindings(grants):
            declaration = _binding_declaration(iap_prefix, iap_service_id, grant)
            resource = gcp.iap.WebBackendServiceIamBinding(
                declaration.logical_name,
                project=context.args.project,
                web_backend_service=service,
                role=grant.role,
                members=list(declaration.members),
                condition=_condition_args(
                    grant.condition,
                    gcp.iap.WebBackendServiceIamBindingConditionArgs,
                ),
                opts=context.options(),
            )
            context.register(resource, declaration.provider_id)


def _condition_args(
    condition: GcpIamCondition | None, args_cls: _ConditionArgsFactory[_ConditionArgsT]
) -> _ConditionArgsT | None:
    if condition is None:
        return None
    return args_cls(title=condition.title, expression=condition.expression, description=condition.description or None)


class GcpIam(pulumi.ComponentResource):
    """Every GCP IAM grant owned by the ``marin`` stack.

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
        context = _GcpIamContext(args=args, imports=imports, provider=gcp_provider, parent=self)

        # Owned service accounts and custom role definitions first: a grant referencing either
        # by name needs it to already exist on a from-scratch `up`. `depends_on` (not Python call
        # order, which Pulumi's engine ignores) is what actually enforces that — every grant
        # waits on both, mirroring GcpDeployPermissions's created_accounts -> grant_opts pattern.
        created_accounts = _create_service_accounts(context)
        created_roles = _create_custom_roles(context)
        grant_context = replace(context, dependencies=tuple(created_accounts + created_roles))

        _grant_project_iam(grant_context)
        _grant_kms_iam(grant_context)
        _grant_secret_iam(grant_context)
        _grant_bucket_iam(grant_context)
        _grant_artifact_repository_iam(grant_context)
        _grant_service_account_iam(grant_context)
        _grant_backend_service_iap(grant_context)
        _grant_cloud_run_iap(grant_context)
        _grant_cloud_run_services(grant_context)

        self.register_outputs({})
