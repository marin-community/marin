# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""InferenceX deploy-target grants; activation belongs to the global IAM stack."""

from collections.abc import Mapping
from dataclasses import replace

from iac.gcp import grafana
from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpCloudRunIapIam,
    GcpEncryptedMember,
    GcpIamGrantSet,
    GcpRoleGrant,
    GcpSecretIam,
)


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Reuse Grafana's viewers and give the ingest identity its scoped machine access."""
    runtime = f"serviceAccount:marin-inferencex@{project}.iam.gserviceaccount.com"
    ingest = f"serviceAccount:github-iris@{project}.iam.gserviceaccount.com"
    deploy = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    viewers = grafana.iam_grants(project, principals).cloud_run_iap[0].iap_grants
    viewers = tuple(replace(grant, members=(*grant.members, ingest)) for grant in viewers)
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
                location="us-central1",
                repository="marin-inferencex",
                grants=(
                    GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy,)),
                    GcpRoleGrant(role="roles/artifactregistry.reader", members=(ingest,)),
                ),
            ),
        ),
        cloud_run_iap=(GcpCloudRunIapIam(location="us-central1", service="marin-inferencex", iap_grants=viewers),),
    )
