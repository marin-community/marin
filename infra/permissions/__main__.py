# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for shared deployment IAM grants."""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.permissions import (
    GcpArtifactRegistryGrant,
    GcpDeployPermissions,
    GcpDeployPermissionsArgs,
    GcpSecretGrant,
)


def main() -> None:
    config = pulumi.Config()
    project = config.require("project")
    secret_iam_grants = config.get_object("secret_iam_grants") or {}
    secret_access_grants = config.get_object("secret_access_grants") or {}
    artifact_registry_grants = config.get_object("artifact_registry_grants") or []
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
            service_accounts=tuple(config.require_object("deploy_service_accounts")),
            id_token_service_accounts=frozenset(config.get_object("id_token_service_accounts") or []),
            secret_metadata_viewers=frozenset(config.get_object("secret_metadata_viewers") or []),
            secret_iam_grants=tuple(
                GcpSecretGrant(service_account=service_account, secrets=tuple(secrets))
                for service_account, secrets in secret_iam_grants.items()
            ),
            secret_access_grants=tuple(
                GcpSecretGrant(service_account=service_account, secrets=tuple(secrets))
                for service_account, secrets in secret_access_grants.items()
            ),
            artifact_registry_grants=tuple(
                GcpArtifactRegistryGrant(
                    service_account=grant["service_account"],
                    location=grant["location"],
                    repositories=tuple(grant["repositories"]),
                )
                for grant in artifact_registry_grants
            ),
            iap_iam_managers=frozenset(config.get_object("iap_iam_managers") or []),
        ),
        gcp_provider=provider,
    )


main()
