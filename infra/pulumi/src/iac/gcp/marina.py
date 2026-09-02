# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the Marina deploy target."""

from collections.abc import Mapping

from iac.gcp.iam import GcpArtifactRepositoryIam, GcpCloudRunIapIam, GcpEncryptedMember, GcpIamGrantSet, GcpRoleGrant

_REGION = "us-central1"
_SERVICE = "marina"


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return Marina grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    return GcpIamGrantSet(
        artifact_repositories=(
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository=_SERVICE,
                grants=(GcpRoleGrant(role="roles/artifactregistry.writer", members=(deploy_account,)),),
            ),
        ),
        cloud_run_iap=(
            GcpCloudRunIapIam(
                location=_REGION,
                service=_SERVICE,
                iap_grants=(
                    GcpRoleGrant(
                        role="roles/iap.httpsResourceAccessor",
                        members=(
                            "domain:openathena.ai",
                            principals["human-014"],
                            principals["human-032"],
                            principals["human-024"],
                            principals["human-012"],
                            principals["human-067"],
                            principals["human-021"],
                            principals["human-006"],
                        ),
                    ),
                ),
            ),
        ),
    )
