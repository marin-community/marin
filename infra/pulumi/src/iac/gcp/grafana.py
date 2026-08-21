# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the Grafana deploy target."""

from collections.abc import Mapping

from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpCloudRunIapIam,
    GcpEncryptedMember,
    GcpIamGrantSet,
    GcpRoleGrant,
    GcpSecretIam,
)

_REGION = "us-central1"


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return Grafana grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    runtime_account = f"serviceAccount:marin-grafana@{project}.iam.gserviceaccount.com"
    loom_account = f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com"
    secret_grants = (
        GcpRoleGrant(
            role=f"projects/{project}/roles/marinSecretIamManager",
            members=(deploy_account,),
        ),
        GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime_account,)),
    )
    # The live-policy audit for #8455 found no SMTP accessor grant. Keep it out of this transfer
    # until the follow-up policy review decides whether Grafana should receive it.
    return GcpIamGrantSet(
        project_grants=(
            GcpRoleGrant(role="roles/cloudsql.client", members=(runtime_account,)),
            GcpRoleGrant(role="roles/compute.viewer", members=(runtime_account,)),
        ),
        secrets=tuple(
            GcpSecretIam(secret=secret, grants=secret_grants)
            for secret in (
                "cloudsql-grafana-password",
                "marin-grafana-cw-read-token",
                "marin-grafana-github-app-private-key",
                "marin-grafana-slack-bot-token",
            )
        ),
        artifact_repositories=(
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository="marin-grafana",
                grants=(
                    GcpRoleGrant(
                        role=f"projects/{project}/roles/marinArtifactRegistryIamManager",
                        members=(principals["human-070"],),
                    ),
                    GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy_account,)),
                ),
            ),
        ),
        cloud_run_iap=(
            GcpCloudRunIapIam(
                location=_REGION,
                service="marin-grafana",
                iap_grants=(
                    GcpRoleGrant(
                        role="roles/iap.httpsResourceAccessor",
                        members=(
                            "domain:openathena.ai",
                            loom_account,
                            principals["human-012"],
                            principals["human-071"],
                        ),
                    ),
                ),
            ),
        ),
    )
