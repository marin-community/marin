# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the Loom deploy target."""

from iac.gcp.iam import GcpArtifactRepositoryIam, GcpIamGrantSet, GcpRoleGrant, GcpSecretIam

_REGION = "us-central1"


def iam_grants(project: str) -> GcpIamGrantSet:
    """Return Loom grants for composition into the global IAM stack."""
    runtime_account = f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com"
    return GcpIamGrantSet(
        project_grants=tuple(
            GcpRoleGrant(role=role, members=(runtime_account,))
            for role in (
                "roles/cloudsql.client",
                "roles/cloudsql.instanceUser",
                "roles/iam.securityReviewer",
                "roles/logging.logWriter",
                "roles/secretmanager.secretAccessor",
                "roles/secretmanager.secretVersionAdder",
                "roles/storage.objectUser",
                "roles/viewer",
            )
        ),
        kms_grants=(GcpRoleGrant(role="roles/cloudkms.cryptoKeyEncrypterDecrypter", members=(runtime_account,)),),
        secrets=(
            GcpSecretIam(
                secret="LOOM_DOTENV",
                grants=(GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime_account,)),),
            ),
        ),
        artifact_repositories=(
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository="loom",
                grants=(GcpRoleGrant(role="roles/artifactregistry.reader", members=(runtime_account,)),),
            ),
        ),
    )
