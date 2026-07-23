# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Additive GCP permissions for keyless GitHub service deployments."""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp

WORKLOAD_IDENTITY_USER = "roles/iam.workloadIdentityUser"
STATE_WRITER = "roles/storage.objectAdmin"
KMS_ENCRYPTER_DECRYPTER = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
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
IAP_IAM_ROLE_ID = "marinIapIamManager"
IAP_IAM_PERMISSIONS = (
    "iap.webServices.getIamPolicy",
    "iap.webServices.setIamPolicy",
)


@dataclass(frozen=True)
class GcpArtifactRegistryGrant:
    location: str
    repositories: tuple[str, ...]


@dataclass(frozen=True)
class GcpDeployAccount:
    service_account: str
    mint_id_tokens: bool = False
    secret_metadata_viewer: bool = False
    secret_access_secrets: tuple[str, ...] = ()
    secret_iam_secrets: tuple[str, ...] = ()
    artifact_registry_grants: tuple[GcpArtifactRegistryGrant, ...] = ()
    iap_iam_manager: bool = False


@dataclass(frozen=True)
class GcpDeployPermissionsArgs:
    project: str
    project_number: str
    workload_identity_pool: str
    github_subject: str
    state_bucket: str
    kms_location: str
    kms_key_ring: str
    kms_key: str
    accounts: tuple[GcpDeployAccount, ...]


@dataclass(frozen=True)
class GcpDeployPermissionSet:
    account_id: str
    service_account_id: str
    service_account_member: str
    github_principal: str
    state_bucket: str
    crypto_key_id: str
    mint_id_tokens: bool


def _account_id(project: str, email: str) -> str:
    suffix = f"@{project}.iam.gserviceaccount.com"
    if not email.endswith(suffix):
        raise ValueError(f"service account {email!r} is not in project {project!r}")
    return email.removesuffix(suffix)


def _service_account_member(email: str) -> str:
    return f"serviceAccount:{email}"


def deploy_permission_sets(args: GcpDeployPermissionsArgs) -> tuple[GcpDeployPermissionSet, ...]:
    """Resolve stable IAM members and resource IDs for each deployment account."""
    github_principal = (
        f"principal://iam.googleapis.com/projects/{args.project_number}/locations/global/"
        f"workloadIdentityPools/{args.workload_identity_pool}/subject/{args.github_subject}"
    )
    crypto_key_id = (
        f"projects/{args.project}/locations/{args.kms_location}/keyRings/{args.kms_key_ring}/"
        f"cryptoKeys/{args.kms_key}"
    )
    return tuple(
        GcpDeployPermissionSet(
            account_id=_account_id(args.project, account.service_account),
            service_account_id=f"projects/{args.project}/serviceAccounts/{account.service_account}",
            service_account_member=_service_account_member(account.service_account),
            github_principal=github_principal,
            state_bucket=args.state_bucket,
            crypto_key_id=crypto_key_id,
            mint_id_tokens=account.mint_id_tokens,
        )
        for account in args.accounts
    )


def _create_base_permissions(
    permission_sets: tuple[GcpDeployPermissionSet, ...],
    opts: pulumi.ResourceOptions,
) -> None:
    for permission_set in permission_sets:
        gcp.serviceaccount.IAMMember(
            f"{permission_set.account_id}-github-main",
            service_account_id=permission_set.service_account_id,
            role=WORKLOAD_IDENTITY_USER,
            member=permission_set.github_principal,
            opts=opts,
        )
        if permission_set.mint_id_tokens:
            gcp.serviceaccount.IAMMember(
                f"{permission_set.account_id}-github-main-id-token",
                service_account_id=permission_set.service_account_id,
                role=SERVICE_ACCOUNT_TOKEN_CREATOR,
                member=permission_set.github_principal,
                opts=opts,
            )
        gcp.storage.BucketIAMMember(
            f"{permission_set.account_id}-pulumi-state",
            bucket=permission_set.state_bucket,
            role=STATE_WRITER,
            member=permission_set.service_account_member,
            opts=opts,
        )
        gcp.kms.CryptoKeyIAMMember(
            f"{permission_set.account_id}-pulumi-kms",
            crypto_key_id=permission_set.crypto_key_id,
            role=KMS_ENCRYPTER_DECRYPTER,
            member=permission_set.service_account_member,
            opts=opts,
        )


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
    """Grant main-branch GitHub workflows access to deploy through existing accounts.

    The component owns custom roles and non-authoritative IAM members. The service accounts,
    workload identity pool/provider, state bucket, and KMS key are existing shared resources.
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
        _create_base_permissions(deploy_permission_sets(args), child)
        _create_secret_permissions(args, child)
        _create_artifact_registry_permissions(args, child)
        _create_iap_permissions(args, child)

        self.register_outputs({})
