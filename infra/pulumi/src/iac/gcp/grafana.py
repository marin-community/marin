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
from iac.gcp.iap import IAP_ACCESSOR_ROLE

_REGION = "us-central1"
_SERVICE = "marin-grafana"
_SECRETS = (
    "cloudsql-grafana-password",
    "marin-grafana-cw-read-token",
    "marin-grafana-github-app-private-key",
    "marin-grafana-slack-bot-token",
)


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return Grafana grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    runtime_account = f"serviceAccount:{_SERVICE}@{project}.iam.gserviceaccount.com"
    loom_account = f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com"
    secret_grants = (
        GcpRoleGrant(
            role=f"projects/{project}/roles/marinSecretIamManager",
            members=(deploy_account,),
        ),
        GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime_account,)),
    )
    # The live-policy audit found no SMTP accessor grant. Omit it until policy review approves
    # granting Grafana access.
    return GcpIamGrantSet(
        project_grants=(
            GcpRoleGrant(role="roles/cloudsql.client", members=(runtime_account,)),
            GcpRoleGrant(role="roles/compute.viewer", members=(runtime_account,)),
        ),
        secrets=tuple(GcpSecretIam(secret=secret, grants=secret_grants) for secret in _SECRETS),
        artifact_repositories=(
            GcpArtifactRepositoryIam(
                location=_REGION,
                repository=_SERVICE,
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
                service=_SERVICE,
                iap_grants=(
                    GcpRoleGrant(
                        role=IAP_ACCESSOR_ROLE,
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
                            principals["human-071"],
                        ),
                    ),
                ),
            ),
        ),
    )
