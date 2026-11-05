# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""InferenceX deploy-target grants; activation belongs to the global IAM stack."""

from collections.abc import Mapping
from dataclasses import replace

from iac.gcp import marina
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
    """Reuse Marina's human viewers and give ingestion its scoped machine access."""
    runtime = f"serviceAccount:marin-inferencex@{project}.iam.gserviceaccount.com"
    ingest = f"serviceAccount:github-iris@{project}.iam.gserviceaccount.com"
    deploy = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    marina_automation = {
        deploy,
        f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com",
        f"serviceAccount:iris-controller@{project}.iam.gserviceaccount.com",
    }
    viewers = marina.iam_grants(project, principals).cloud_run_iap[0].iap_grants
    viewers = tuple(
        replace(grant, members=(*(member for member in grant.members if member not in marina_automation), ingest))
        for grant in viewers
    )
    iap_agent = "serviceAccount:service-748532799086@gcp-sa-iap.iam.gserviceaccount.com"
    return GcpIamGrantSet(
        project_grants=(GcpRoleGrant(role="roles/cloudsql.client", members=(runtime, ingest)),),
        secrets=(
            GcpSecretIam(
                secret="cloudsql-inferencex-reader-password",
                grants=(GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime,)),),
            ),
            GcpSecretIam(
                secret="cloudsql-inferencex-writer-password",
                grants=(GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(ingest,)),),
            ),
            GcpSecretIam(
                secret="marin-inferencex-invalidate-secret",
                grants=(GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime, ingest)),),
            ),
        ),
        artifact_repositories=(
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository="marin-inferencex",
                grants=(
                    GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy,)),
                    GcpRoleGrant(role="roles/artifactregistry.reader", members=(ingest,)),
                ),
            ),
        ),
        cloud_run_iap=(
            GcpCloudRunIapIam(
                location=_REGION,
                service="marin-inferencex",
                iap_grants=viewers,
                service_grants=(GcpRoleGrant(role="roles/run.invoker", members=(iap_agent,)),),
            ),
        ),
    )
