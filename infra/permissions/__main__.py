# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for shared deployment IAM grants."""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.permissions import GcpDeployPermissions, GcpDeployPermissionsArgs


def main() -> None:
    config = pulumi.Config()
    project = config.require("project")
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
        ),
        gcp_provider=provider,
    )


main()
