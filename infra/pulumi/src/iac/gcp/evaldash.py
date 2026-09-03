# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the EvalDash deploy target."""

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
_SERVICE = "marin-evaldash"
_SECRETS = (
    "cloudsql-evals-password",
    "cw-object-storage-key-id",
    "cw-object-storage-key-secret",
)


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return EvalDash grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    runtime_account = f"serviceAccount:{_SERVICE}@{project}.iam.gserviceaccount.com"
    loom_account = f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com"
    return GcpIamGrantSet(
        project_grants=(
            GcpRoleGrant(role="roles/cloudsql.client", members=(runtime_account,)),
            GcpRoleGrant(role="roles/compute.viewer", members=(runtime_account,)),
            GcpRoleGrant(role="roles/storage.objectViewer", members=(runtime_account,)),
        ),
        secrets=tuple(
            GcpSecretIam(
                secret=secret,
                grants=(GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime_account,)),),
            )
            for secret in _SECRETS
        ),
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
                            f"serviceAccount:iris-controller@{project}.iam.gserviceaccount.com",
                            "serviceAccount:ravwojdyla@rav-openathena.iam.gserviceaccount.com",
                            principals["human-014"],
                            principals["human-032"],
                            principals["human-024"],
                            principals["human-012"],
                            principals["human-067"],
                            principals["human-021"],
                            principals["human-006"],
                            "domain:openathena.ai",
                            loom_account,
                            principals["human-064"],
                            principals["human-070"],
                        ),
                    ),
                ),
            ),
        ),
    )
