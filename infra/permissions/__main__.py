# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for shared deployment IAM grants."""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.permissions import (
    GcpArtifactRegistryGrant,
    GcpDeployAccount,
    GcpDeployPermissions,
    GcpDeployPermissionsArgs,
    GcpKmsAccess,
    GcpStateAccess,
)


def main() -> None:
    config = pulumi.Config()
    project = config.require("project")
    deploy_accounts = config.require_object("deploy_accounts")
    provider = gcp.Provider("gcp", project=project)
    GcpDeployPermissions(
        "deploy",
        GcpDeployPermissionsArgs(
            project=project,
            project_number=config.require("project_number"),
            workload_identity_pool=config.require("workload_identity_pool"),
            state_bucket=config.require("state_bucket"),
            kms_location=config.require("kms_location"),
            kms_key_ring=config.require("kms_key_ring"),
            kms_key=config.require("kms_key"),
            accounts=tuple(
                GcpDeployAccount(
                    service_account=account["service_account"],
                    github_subject=account["github_subject"],
                    mint_id_tokens=account.get("mint_id_tokens", False),
                    kms_access=GcpKmsAccess(account.get("kms_access", GcpKmsAccess.ENCRYPT_DECRYPT.value)),
                    state_access=GcpStateAccess(account.get("state_access", GcpStateAccess.APPLY.value)),
                    secret_metadata_viewer=account.get("secret_metadata_viewer", False),
                    secret_access_secrets=tuple(account.get("secret_access_secrets", [])),
                    secret_iam_secrets=tuple(account.get("secret_iam_secrets", [])),
                    artifact_registry_grants=tuple(
                        GcpArtifactRegistryGrant(
                            location=grant["location"],
                            repositories=tuple(grant["repositories"]),
                        )
                        for grant in account.get("artifact_registry_grants", [])
                    ),
                    iap_iam_manager=account.get("iap_iam_manager", False),
                    create_account=account.get("create_account", False),
                    display_name=account.get("display_name", ""),
                )
                for account in deploy_accounts
            ),
        ),
        gcp_provider=provider,
    )


main()
