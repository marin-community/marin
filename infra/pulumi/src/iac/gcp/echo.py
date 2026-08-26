# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the Echo deploy target."""

from collections.abc import Mapping

from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpCloudRunIapIam,
    GcpEncryptedMember,
    GcpIamGrantSet,
    GcpRoleGrant,
    GcpSecretIam,
)
from iac.gcp.iap import IAP_ACCESSOR_ROLE, shared_iap_accessors

_REGION = "us-central1"
_OPENATHENA_GROUP = "group:eng-all@openathena.ai"
_API_SERVICE = "echo-api"
_SYNC_JOB = "echo-sync"
_MIRROR_TOKEN_SECRET = "marinmirror-token"


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return Echo grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    echo_api_account = f"serviceAccount:{_API_SERVICE}@{project}.iam.gserviceaccount.com"
    echo_sync_account = f"serviceAccount:{_SYNC_JOB}@{project}.iam.gserviceaccount.com"
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
                secret=_MIRROR_TOKEN_SECRET,
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
                repository=_API_SERVICE,
                grants=(GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy_account,)),),
            ),
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository=_SYNC_JOB,
                grants=(GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy_account,)),),
            ),
        ),
        cloud_run_iap=(
            GcpCloudRunIapIam(
                location=_REGION,
                service=_API_SERVICE,
                iap_grants=(
                    GcpRoleGrant(
                        role=IAP_ACCESSOR_ROLE,
                        members=(*shared_iap_accessors(project, principals), "domain:openathena.ai", loom_account),
                    ),
                ),
            ),
        ),
    )
