# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the Echo deploy target."""

from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpCloudRunIapIam,
    GcpIamGrantSet,
    GcpRoleGrant,
    GcpSecretIam,
)

_REGION = "us-central1"
_OPENATHENA_GROUP = "group:eng-all@openathena.ai"


def iam_grants(project: str) -> GcpIamGrantSet:
    """Return Echo grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    echo_api_account = f"serviceAccount:echo-api@{project}.iam.gserviceaccount.com"
    echo_sync_account = f"serviceAccount:echo-sync@{project}.iam.gserviceaccount.com"
    loom_account = f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com"
    return GcpIamGrantSet(
        project_grants=(
            GcpRoleGrant(
                role="roles/cloudsql.client",
                members=(_OPENATHENA_GROUP, echo_api_account, echo_sync_account),
            ),
            GcpRoleGrant(
                role="roles/cloudsql.instanceUser",
                members=(_OPENATHENA_GROUP, echo_api_account, echo_sync_account),
            ),
        ),
        secrets=(
            GcpSecretIam(
                secret="marinmirror-token",
                grants=(
                    GcpRoleGrant(
                        role=f"projects/{project}/roles/marinSecretIamManager",
                        members=(deploy_account,),
                    ),
                    GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(echo_sync_account,)),
                ),
            ),
        ),
        artifact_repositories=(
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository="echo-api",
                grants=(GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy_account,)),),
            ),
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository="echo-sync",
                grants=(GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy_account,)),),
            ),
        ),
        cloud_run_iap=(
            GcpCloudRunIapIam(
                location=_REGION,
                service="echo-api",
                iap_grants=(
                    GcpRoleGrant(
                        role="roles/iap.httpsResourceAccessor",
                        members=("domain:openathena.ai", loom_account),
                    ),
                ),
            ),
        ),
    )
