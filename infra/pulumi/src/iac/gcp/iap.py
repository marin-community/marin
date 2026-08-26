# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-service IAP grants owned by the global IAM stack."""

from collections.abc import Mapping

from iac.gcp.iam import GcpBackendServiceIapIam, GcpEncryptedMember, GcpIamGrantSet, GcpRoleGrant

IAP_ACCESSOR_ROLE = "roles/iap.httpsResourceAccessor"


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return authoritative IAP grants for the GCE Iris backends."""
    return GcpIamGrantSet(
        backend_service_iap=(
            GcpBackendServiceIapIam(
                service="iris-marin-be",
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
                            "domain:stanford.edu",
                            f"serviceAccount:rav-agent@{project}.iam.gserviceaccount.com",
                            principals["human-054"],
                            principals["human-064"],
                            principals["human-070"],
                            f"serviceAccount:github-iris@{project}.iam.gserviceaccount.com",
                            f"serviceAccount:iris-ci-smoke@{project}.iam.gserviceaccount.com",
                            f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com",
                            principals["human-061"],
                            principals["human-073"],
                            principals["human-074"],
                        ),
                    ),
                ),
            ),
            GcpBackendServiceIapIam(
                service="iris-marin-dev-be",
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
                            "domain:stanford.edu",
                            f"serviceAccount:rav-agent@{project}.iam.gserviceaccount.com",
                            principals["human-054"],
                            principals["human-064"],
                            principals["human-070"],
                        ),
                    ),
                ),
            ),
        ),
    )
