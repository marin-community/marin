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

        self.register_outputs({})
