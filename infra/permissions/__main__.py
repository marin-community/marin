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
            github_subject=config.require("github_subject"),
            state_bucket=config.require("state_bucket"),
            kms_location=config.require("kms_location"),
            kms_key_ring=config.require("kms_key_ring"),
            kms_key=config.require("kms_key"),
            accounts=tuple(
                GcpDeployAccount(
                    service_account=account["service_account"],
                    mint_id_tokens=account.get("mint_id_tokens", False),
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
                )
                for account in deploy_accounts
            ),
        ),
        gcp_provider=provider,
    )


main()
