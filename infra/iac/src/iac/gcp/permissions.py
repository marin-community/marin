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
class GcpSecretGrant:
    service_account: str
    secrets: tuple[str, ...]


@dataclass(frozen=True)
class GcpArtifactRegistryGrant:
    service_account: str
    location: str
    repositories: tuple[str, ...]


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
    service_accounts: tuple[str, ...]
    id_token_service_accounts: frozenset[str] = frozenset()
    secret_metadata_viewers: frozenset[str] = frozenset()
    secret_iam_grants: tuple[GcpSecretGrant, ...] = ()
    secret_access_grants: tuple[GcpSecretGrant, ...] = ()
    artifact_registry_grants: tuple[GcpArtifactRegistryGrant, ...] = ()
    iap_iam_managers: frozenset[str] = frozenset()


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


def deploy_permission_sets(args: GcpDeployPermissionsArgs) -> tuple[GcpDeployPermissionSet, ...]:
    """Resolve stable IAM members and resource IDs for each deployment account."""
    unknown_id_token_accounts = args.id_token_service_accounts - set(args.service_accounts)
    if unknown_id_token_accounts:
        raise ValueError(
            f"id_token_service_accounts contains undeclared deploy accounts: {sorted(unknown_id_token_accounts)!r}"
        )
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
            account_id=_account_id(args.project, email),
            service_account_id=f"projects/{args.project}/serviceAccounts/{email}",
            service_account_member=f"serviceAccount:{email}",
            github_principal=github_principal,
            state_bucket=args.state_bucket,
            crypto_key_id=crypto_key_id,
            mint_id_tokens=email in args.id_token_service_accounts,
        )
        for email in args.service_accounts
    )


def _validate_permission_accounts(args: GcpDeployPermissionsArgs) -> None:
    deploy_accounts = set(args.service_accounts)
    secret_accounts = (
        args.secret_metadata_viewers
        | {grant.service_account for grant in args.secret_iam_grants}
        | {grant.service_account for grant in args.secret_access_grants}
    )
    permission_accounts = (
        secret_accounts | {grant.service_account for grant in args.artifact_registry_grants} | args.iap_iam_managers
    )
    unknown_accounts = permission_accounts - deploy_accounts
    if unknown_accounts:
        raise ValueError(f"permissions contain undeclared deploy accounts: {sorted(unknown_accounts)!r}")


class GcpDeployPermissions(pulumi.ComponentResource):
    """Grant main-branch GitHub workflows access to deploy through existing accounts.

    The component owns only non-authoritative IAM members. The service accounts, workload
    identity pool/provider, state bucket, and KMS key are existing shared resources.
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
        _validate_permission_accounts(args)
        for permission_set in deploy_permission_sets(args):
            gcp.serviceaccount.IAMMember(
                f"{permission_set.account_id}-github-main",
                service_account_id=permission_set.service_account_id,
                role=WORKLOAD_IDENTITY_USER,
                member=permission_set.github_principal,
                opts=child,
            )
            if permission_set.mint_id_tokens:
                gcp.serviceaccount.IAMMember(
                    f"{permission_set.account_id}-github-main-id-token",
                    service_account_id=permission_set.service_account_id,
                    role=SERVICE_ACCOUNT_TOKEN_CREATOR,
                    member=permission_set.github_principal,
                    opts=child,
                )
            gcp.storage.BucketIAMMember(
                f"{permission_set.account_id}-pulumi-state",
                bucket=permission_set.state_bucket,
                role=STATE_WRITER,
                member=permission_set.service_account_member,
                opts=child,
            )
            gcp.kms.CryptoKeyIAMMember(
                f"{permission_set.account_id}-pulumi-kms",
                crypto_key_id=permission_set.crypto_key_id,
                role=KMS_ENCRYPTER_DECRYPTER,
                member=permission_set.service_account_member,
                opts=child,
            )

        for email in args.secret_metadata_viewers:
            gcp.projects.IAMMember(
                f"{_account_id(args.project, email)}-secret-metadata",
                project=args.project,
                role=SECRET_METADATA_VIEWER,
                member=f"serviceAccount:{email}",
                opts=child,
            )

        if args.secret_iam_grants:
            secret_iam_role = gcp.projects.IAMCustomRole(
                "secret-iam-manager",
                project=args.project,
                role_id=SECRET_IAM_ROLE_ID,
                title="Marin Secret IAM Manager",
                description="Manage IAM policies on selected deployment secrets without reading payloads.",
                permissions=list(SECRET_IAM_PERMISSIONS),
                opts=child,
            )
            for grant in args.secret_iam_grants:
                account_id = _account_id(args.project, grant.service_account)
                for secret in grant.secrets:
                    gcp.secretmanager.SecretIamMember(
                        f"{account_id}-{secret}-iam-manager",
                        project=args.project,
                        secret_id=secret,
                        role=secret_iam_role.name,
                        member=f"serviceAccount:{grant.service_account}",
                        opts=child,
                    )

        for grant in args.secret_access_grants:
            account_id = _account_id(args.project, grant.service_account)
            for secret in grant.secrets:
                gcp.secretmanager.SecretIamMember(
                    f"{account_id}-{secret}-accessor",
                    project=args.project,
                    secret_id=secret,
                    role=SECRET_ACCESSOR,
                    member=f"serviceAccount:{grant.service_account}",
                    opts=child,
                )

        for grant in args.artifact_registry_grants:
            account_id = _account_id(args.project, grant.service_account)
            for repository in grant.repositories:
                gcp.artifactregistry.RepositoryIamMember(
                    f"{account_id}-{repository}-writer",
                    project=args.project,
                    location=grant.location,
                    repository=repository,
                    role=ARTIFACT_REGISTRY_WRITER,
                    member=f"serviceAccount:{grant.service_account}",
                    opts=child,
                )

        if args.iap_iam_managers:
            iap_iam_role = gcp.projects.IAMCustomRole(
                "iap-iam-manager",
                project=args.project,
                role_id=IAP_IAM_ROLE_ID,
                title="Marin IAP IAM Manager",
                description="Manage IAP policies on web services without accessing them.",
                permissions=list(IAP_IAM_PERMISSIONS),
                opts=child,
            )
            for email in args.iap_iam_managers:
                gcp.projects.IAMMember(
                    f"{_account_id(args.project, email)}-iap-iam-manager",
                    project=args.project,
                    role=iap_iam_role.name,
                    member=f"serviceAccount:{email}",
                    opts=child,
                )

        self.register_outputs({})
