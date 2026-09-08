# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP IAM declarations required by the Marina deploy target.

One runtime service account serves every app and runs its manifest-declared Cloud Run jobs. It is
the only account that writes the ``marina`` database; people read it through a Cloud SQL
group login that ``marina migrate`` grants. The grants below are the union of what the
hosted apps need: Cloud SQL login, the record buckets evaldash indexes, compute listing for
Iris discovery, the CoreWeave storage keys, and the marinmirror token for sync.
"""

from collections.abc import Mapping

from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpBucketIam,
    GcpCloudRunIapIam,
    GcpEncryptedMember,
    GcpIamGrantSet,
    GcpRoleGrant,
    GcpSecretIam,
)

_REGION = "us-central1"
_PROJECT_NUMBER = "748532799086"
_SERVICE = "marina"
_DATA_BUCKET = "marin-marina"
_MIRROR_TOKEN_SECRET = "marinmirror-token"
_COREWEAVE_SECRETS = ("cw-object-storage-key-id", "cw-object-storage-key-secret")
# The Cloud SQL group login `marina migrate` grants read on every app schema.
_READER_GROUP = "group:eng-all@openathena.ai"


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return Marina grants for composition into the global IAM stack."""
    deploy_account = f"serviceAccount:marin-cd-cloud-run-deploy@{project}.iam.gserviceaccount.com"
    runtime_account = f"serviceAccount:{_SERVICE}@{project}.iam.gserviceaccount.com"
    loom_account = f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com"
    iap_agent = f"serviceAccount:service-{_PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com"
    return GcpIamGrantSet(
        project_grants=(
            # Marina's Pulumi program manages one Scheduler trigger per app-declared runner.
            GcpRoleGrant(role="roles/cloudscheduler.admin", members=(deploy_account,)),
            GcpRoleGrant(role="roles/cloudsql.client", members=(runtime_account, _READER_GROUP)),
            GcpRoleGrant(role="roles/cloudsql.instanceUser", members=(runtime_account, _READER_GROUP)),
            GcpRoleGrant(role="roles/compute.viewer", members=(runtime_account,)),
            GcpRoleGrant(role="roles/storage.objectViewer", members=(runtime_account,)),
            # Cloud Scheduler invokes manifest-declared runners as the runtime account.
            GcpRoleGrant(role="roles/run.invoker", members=(runtime_account,)),
        ),
        secrets=(
            GcpSecretIam(
                secret=_MIRROR_TOKEN_SECRET,
                grants=(
                    GcpRoleGrant(role=f"projects/{project}/roles/marinSecretIamManager", members=(deploy_account,)),
                    GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime_account,)),
                ),
            ),
            *(
                GcpSecretIam(
                    secret=secret,
                    grants=(GcpRoleGrant(role="roles/secretmanager.secretAccessor", members=(runtime_account,)),),
                )
                for secret in _COREWEAVE_SECRETS
            ),
        ),
        buckets=(
            GcpBucketIam(
                bucket=_DATA_BUCKET,
                grants=(
                    GcpRoleGrant(role="roles/storage.objectAdmin", members=(deploy_account, "domain:openathena.ai")),
                ),
            ),
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
                service_grants=(GcpRoleGrant(role="roles/run.invoker", members=(iap_agent,)),),
                iap_grants=(
                    GcpRoleGrant(
                        role="roles/iap.httpsResourceAccessor",
                        members=(
                            "domain:openathena.ai",
                            loom_account,
                            f"serviceAccount:iris-controller@{project}.iam.gserviceaccount.com",
                            "serviceAccount:ravwojdyla@rav-openathena.iam.gserviceaccount.com",
                            principals["human-014"],
                            principals["human-032"],
                            principals["human-024"],
                            principals["human-012"],
                            principals["human-067"],
                            principals["human-021"],
                            principals["human-006"],
                            principals["human-064"],
                        ),
                    ),
                ),
            ),
        ),
    )
