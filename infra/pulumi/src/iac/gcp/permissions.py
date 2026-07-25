# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Additive GCP permissions for keyless GitHub service deployments."""

from dataclasses import dataclass
from enum import StrEnum

import pulumi
import pulumi_gcp as gcp

WORKLOAD_IDENTITY_USER = "roles/iam.workloadIdentityUser"
STATE_WRITER = "roles/storage.objectAdmin"
STATE_READER = "roles/storage.objectViewer"
STATE_LOCK_WRITER = "roles/storage.objectUser"
STATE_LOCKS_PREFIX = ".pulumi/locks/"
KMS_ENCRYPTER_DECRYPTER = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
KMS_DECRYPTER = "roles/cloudkms.cryptoKeyDecrypter"
SERVICE_ACCOUNT_TOKEN_CREATOR = "roles/iam.serviceAccountTokenCreator"
SECRET_METADATA_VIEWER = "roles/secretmanager.viewer"
SECRET_ACCESSOR = "roles/secretmanager.secretAccessor"
SECRET_IAM_ROLE_ID = "marinSecretIamManager"
SECRET_IAM_PERMISSIONS = (
    "secretmanager.secrets.get",
    "secretmanager.secrets.getIamPolicy",
    "secretmanager.secrets.setIamPolicy",
)
ARTIFACT_REGISTRY_WRITER = "roles/artifactregistry.writer"
GCP_RESOURCE_PREVIEWER_ROLE_ID = "marinGcpResourcePreviewer"
GCP_RESOURCE_PREVIEWER_PERMISSIONS = (
    "artifactregistry.locations.get",
    "artifactregistry.repositories.get",
    "compute.addresses.get",
    "compute.regions.get",
)
IAP_IAM_ROLE_ID = "marinIapIamManager"
IAP_IAM_PERMISSIONS = (
    "iap.webServices.getIamPolicy",
    "iap.webServices.setIamPolicy",
)


class GcpKmsAccess(StrEnum):
    """How far a deploy account's KMS grant reaches on the shared state key."""

    ENCRYPT_DECRYPT = "encrypt_decrypt"  # `pulumi up` — reads and writes stack secrets.
    DECRYPT_ONLY = "decrypt_only"  # `pulumi preview` — reads stack secrets, never writes them.


class GcpStateAccess(StrEnum):
    """How far a deploy account's grant reaches on the shared state bucket."""

    APPLY = "apply"  # `pulumi up` — full read/write on state objects.
    PREVIEW = "preview"  # `pulumi preview` — read state, write only the per-stack lock prefix.


@dataclass(frozen=True)
class GcpArtifactRegistryGrant:
    location: str
    repositories: tuple[str, ...]


@dataclass(frozen=True)
class GcpAutomationAccount:
    service_account: str
    # One GitHub OIDC subject per trigger this account must authenticate from (e.g. a
    # `pull_request` subject for preview plus a main-branch `ref:refs/heads/main` subject for
    # `workflow_dispatch`, whose subject follows the dispatching ref rather than the event name).
    github_subjects: tuple[str, ...]
    mint_id_tokens: bool = False
    kms_access: GcpKmsAccess = GcpKmsAccess.ENCRYPT_DECRYPT
    state_access: GcpStateAccess = GcpStateAccess.APPLY
    secret_metadata_viewer: bool = False
    secret_access_secrets: tuple[str, ...] = ()
    secret_iam_secrets: tuple[str, ...] = ()
    artifact_registry_grants: tuple[GcpArtifactRegistryGrant, ...] = ()
    gcp_resource_previewer: bool = False
    iap_iam_manager: bool = False
    # True for accounts this stack owns end-to-end (net-new, deploy-only identities). False
    # (default) for accounts that predate this stack and were created out-of-band — this
    # component only grants IAM on those, per the project README.
    create_account: bool = False
    display_name: str = ""


@dataclass(frozen=True)
class GcpDeployPermissionsArgs:
    project: str
    project_number: str
    workload_identity_pool: str
    state_bucket: str
    kms_location: str
    kms_key_ring: str
    kms_key: str
    accounts: tuple[GcpAutomationAccount, ...]


@dataclass(frozen=True)
class GcpDeployPermissionSet:
    account_id: str
    service_account_id: str
    service_account_member: str
    github_principals: tuple[str, ...]
    state_bucket: str
    crypto_key_id: str
    mint_id_tokens: bool
    kms_access: GcpKmsAccess
    state_access: GcpStateAccess


def _account_id(project: str, email: str) -> str:
    suffix = f"@{project}.iam.gserviceaccount.com"
    if not email.endswith(suffix):
        raise ValueError(f"service account {email!r} is not in project {project!r}")
    return email.removesuffix(suffix)


def _service_account_member(email: str) -> str:
    return f"serviceAccount:{email}"


def _github_principals(
    project_number: str, workload_identity_pool: str, github_subjects: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        f"principal://iam.googleapis.com/projects/{project_number}/locations/global/"
        f"workloadIdentityPools/{workload_identity_pool}/subject/{subject}"
        for subject in github_subjects
    )


def deploy_permission_sets(args: GcpDeployPermissionsArgs) -> tuple[GcpDeployPermissionSet, ...]:
    """Resolve stable IAM members and resource IDs for each deployment account."""
    crypto_key_id = (
        f"projects/{args.project}/locations/{args.kms_location}/keyRings/{args.kms_key_ring}/"
        f"cryptoKeys/{args.kms_key}"
    )
    return tuple(
        GcpDeployPermissionSet(
            account_id=_account_id(args.project, account.service_account),
            service_account_id=f"projects/{args.project}/serviceAccounts/{account.service_account}",
            service_account_member=_service_account_member(account.service_account),
            github_principals=_github_principals(
                args.project_number, args.workload_identity_pool, account.github_subjects
            ),
            state_bucket=args.state_bucket,
            crypto_key_id=crypto_key_id,
            mint_id_tokens=account.mint_id_tokens,
            kms_access=account.kms_access,
            state_access=account.state_access,
        )
        for account in args.accounts
    )


def _kms_role(access: GcpKmsAccess) -> str:
    if access is GcpKmsAccess.DECRYPT_ONLY:
        return KMS_DECRYPTER
    return KMS_ENCRYPTER_DECRYPTER


def _create_service_accounts(
    args: GcpDeployPermissionsArgs, opts: pulumi.ResourceOptions
) -> tuple[gcp.serviceaccount.Account, ...]:
    """Create the accounts this stack owns end-to-end (`create_account=True`)."""
    created = []
    for account in args.accounts:
        if not account.create_account:
            continue
        account_id = _account_id(args.project, account.service_account)
        created.append(
            gcp.serviceaccount.Account(
                f"{account_id}-account",
                account_id=account_id,
                project=args.project,
                display_name=account.display_name or account_id,
                opts=opts,
            )
        )
    return tuple(created)


def _grant_workload_identity(permission_set: GcpDeployPermissionSet, opts: pulumi.ResourceOptions) -> None:
    # Suffix only when an account has more than one subject: the unsuffixed "-github-main" name
    # is the pre-existing resource name for the accounts already live in state, and renaming it
    # would make Pulumi delete-then-recreate a protected resource.
    multiple = len(permission_set.github_principals) > 1
    for i, principal in enumerate(permission_set.github_principals):
        suffix = f"-{i}" if multiple else ""
        gcp.serviceaccount.IAMMember(
            f"{permission_set.account_id}-github-main{suffix}",
            service_account_id=permission_set.service_account_id,
            role=WORKLOAD_IDENTITY_USER,
            member=principal,
            opts=opts,
        )
        if permission_set.mint_id_tokens:
            gcp.serviceaccount.IAMMember(
                f"{permission_set.account_id}-github-main-id-token{suffix}",
                service_account_id=permission_set.service_account_id,
                role=SERVICE_ACCOUNT_TOKEN_CREATOR,
                member=principal,
                opts=opts,
            )


def _grant_state_access(permission_set: GcpDeployPermissionSet, opts: pulumi.ResourceOptions) -> None:
    if permission_set.state_access is GcpStateAccess.APPLY:
        gcp.storage.BucketIAMMember(
            f"{permission_set.account_id}-pulumi-state",
            bucket=permission_set.state_bucket,
            role=STATE_WRITER,
            member=permission_set.service_account_member,
            opts=opts,
        )
        return
    gcp.storage.BucketIAMMember(
        f"{permission_set.account_id}-pulumi-state-read",
        bucket=permission_set.state_bucket,
        role=STATE_READER,
        member=permission_set.service_account_member,
        opts=opts,
    )
    gcp.storage.BucketIAMMember(
        f"{permission_set.account_id}-pulumi-state-locks",
        bucket=permission_set.state_bucket,
        role=STATE_LOCK_WRITER,
        member=permission_set.service_account_member,
        condition=gcp.storage.BucketIAMMemberConditionArgs(
            title=f"{permission_set.account_id}-pulumi-locks",
            description="Create/delete stack lock objects only; no access to state content.",
            expression=(
                f'resource.name.startsWith("projects/_/buckets/{permission_set.state_bucket}'
                f'/objects/{STATE_LOCKS_PREFIX}")'
            ),
        ),
        opts=opts,
    )


def _grant_kms_access(permission_set: GcpDeployPermissionSet, opts: pulumi.ResourceOptions) -> None:
    gcp.kms.CryptoKeyIAMMember(
        f"{permission_set.account_id}-pulumi-kms",
        crypto_key_id=permission_set.crypto_key_id,
        role=_kms_role(permission_set.kms_access),
        member=permission_set.service_account_member,
        opts=opts,
    )


def _create_base_permissions(
    permission_sets: tuple[GcpDeployPermissionSet, ...],
    opts: pulumi.ResourceOptions,
) -> None:
    for permission_set in permission_sets:
        _grant_workload_identity(permission_set, opts)
        _grant_state_access(permission_set, opts)
        _grant_kms_access(permission_set, opts)


def _create_secret_permissions(args: GcpDeployPermissionsArgs, opts: pulumi.ResourceOptions) -> None:
    for account in args.accounts:
        account_id = _account_id(args.project, account.service_account)
        account_member = _service_account_member(account.service_account)
        if account.secret_metadata_viewer:
            gcp.projects.IAMMember(
                f"{account_id}-secret-metadata",
                project=args.project,
                role=SECRET_METADATA_VIEWER,
                member=account_member,
                opts=opts,
            )
        for secret in account.secret_access_secrets:
            gcp.secretmanager.SecretIamMember(
                f"{account_id}-{secret}-accessor",
                project=args.project,
                secret_id=secret,
                role=SECRET_ACCESSOR,
                member=account_member,
                opts=opts,
            )

    secret_iam_accounts = tuple(account for account in args.accounts if account.secret_iam_secrets)
    if not secret_iam_accounts:
        return

    secret_iam_role = gcp.projects.IAMCustomRole(
        "secret-iam-manager",
        project=args.project,
        role_id=SECRET_IAM_ROLE_ID,
        title="Marin Secret IAM Manager",
        description="Manage IAM policies on selected deployment secrets without reading payloads.",
        permissions=list(SECRET_IAM_PERMISSIONS),
        opts=opts,
    )
    for account in secret_iam_accounts:
        account_id = _account_id(args.project, account.service_account)
        for secret in account.secret_iam_secrets:
            gcp.secretmanager.SecretIamMember(
                f"{account_id}-{secret}-iam-manager",
                project=args.project,
                secret_id=secret,
                role=secret_iam_role.name,
                member=_service_account_member(account.service_account),
                opts=opts,
            )


def _create_artifact_registry_permissions(args: GcpDeployPermissionsArgs, opts: pulumi.ResourceOptions) -> None:
    for account in args.accounts:
        account_id = _account_id(args.project, account.service_account)
        for grant in account.artifact_registry_grants:
            for repository in grant.repositories:
                gcp.artifactregistry.RepositoryIamMember(
                    f"{account_id}-{repository}-writer",
                    project=args.project,
                    location=grant.location,
                    repository=repository,
                    role=ARTIFACT_REGISTRY_WRITER,
                    member=_service_account_member(account.service_account),
                    opts=opts,
                )


def _create_gcp_resource_preview_permissions(args: GcpDeployPermissionsArgs, opts: pulumi.ResourceOptions) -> None:
    preview_accounts = tuple(account for account in args.accounts if account.gcp_resource_previewer)
    if not preview_accounts:
        return

    preview_role = gcp.projects.IAMCustomRole(
        "gcp-resource-previewer",
        project=args.project,
        role_id=GCP_RESOURCE_PREVIEWER_ROLE_ID,
        title="Marin GCP Resource Previewer",
        description="Read the GCP resources declared by the marin-iac preview stack.",
        permissions=list(GCP_RESOURCE_PREVIEWER_PERMISSIONS),
        opts=opts,
    )
    for account in preview_accounts:
        gcp.projects.IAMMember(
            f"{_account_id(args.project, account.service_account)}-gcp-resource-previewer",
            project=args.project,
            role=preview_role.name,
            member=_service_account_member(account.service_account),
            opts=opts,
        )


def _create_iap_permissions(args: GcpDeployPermissionsArgs, opts: pulumi.ResourceOptions) -> None:
    iap_iam_accounts = tuple(account for account in args.accounts if account.iap_iam_manager)
    if not iap_iam_accounts:
        return

    iap_iam_role = gcp.projects.IAMCustomRole(
        "iap-iam-manager",
        project=args.project,
        role_id=IAP_IAM_ROLE_ID,
        title="Marin IAP IAM Manager",
        description="Manage IAP policies on web services without accessing them.",
        permissions=list(IAP_IAM_PERMISSIONS),
        opts=opts,
    )
    for account in iap_iam_accounts:
        gcp.projects.IAMMember(
            f"{_account_id(args.project, account.service_account)}-iap-iam-manager",
            project=args.project,
            role=iap_iam_role.name,
            member=_service_account_member(account.service_account),
            opts=opts,
        )


class GcpDeployPermissions(pulumi.ComponentResource):
    """Grant GitHub workflows access to deploy or preview through deploy accounts.

    Each account binds to its own GitHub OIDC subject(s) (e.g. a main-branch push for
    `pulumi up`, or a `pull_request` event for `pulumi preview`), scoped by
    `kms_access`/`state_access` to what that trigger needs. Most accounts are existing —
    this component only grants IAM on them — but one with `create_account=True` is owned
    end-to-end, including the `gcp.serviceaccount.Account` itself. The workload identity
    pool/provider, state bucket, and KMS key are always existing shared resources.
    """

    def __init__(
        self,
        name: str,
        args: GcpDeployPermissionsArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:GcpDeployPermissions", name, None, opts)
        child = pulumi.ResourceOptions(parent=self, provider=gcp_provider, protect=True)
        created_accounts = _create_service_accounts(args, child)
        # Grants on a just-created account must wait for the Account resource itself
        grant_opts = (
            pulumi.ResourceOptions(parent=self, provider=gcp_provider, protect=True, depends_on=list(created_accounts))
            if created_accounts
            else child
        )
        _create_base_permissions(deploy_permission_sets(args), grant_opts)
        _create_secret_permissions(args, grant_opts)
        _create_artifact_registry_permissions(args, grant_opts)
        _create_gcp_resource_preview_permissions(args, grant_opts)
        _create_iap_permissions(args, grant_opts)

        self.register_outputs({})
