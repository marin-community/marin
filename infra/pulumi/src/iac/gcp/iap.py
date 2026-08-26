# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared and backend-service IAP grants owned by the global IAM stack."""

from collections.abc import Mapping

from iac.gcp.iam import GcpBackendServiceIapIam, GcpEncryptedMember, GcpIamGrantSet, GcpRoleGrant

_ACCESSOR_ROLE = "roles/iap.httpsResourceAccessor"
_SHARED_HUMAN_ACCESSOR_IDS = (
    "human-014",
    "human-032",
    "human-024",
    "human-012",
    "human-067",
    "human-021",
    "human-006",
)


def shared_iap_accessors(
    project: str,
    principals: Mapping[str, GcpEncryptedMember],
) -> tuple[str | GcpEncryptedMember, ...]:
    """Return the accessors retained from the retired project-wide IAP grant."""
    return (
        f"serviceAccount:iris-controller@{project}.iam.gserviceaccount.com",
        "serviceAccount:ravwojdyla@rav-openathena.iam.gserviceaccount.com",
        *(principals[principal_id] for principal_id in _SHARED_HUMAN_ACCESSOR_IDS),
    )


def iam_grants(project: str, principals: Mapping[str, GcpEncryptedMember]) -> GcpIamGrantSet:
    """Return authoritative IAP grants for the GCE Iris backends."""
    shared_accessors = shared_iap_accessors(project, principals)
    return GcpIamGrantSet(
        backend_service_iap=(
            GcpBackendServiceIapIam(
                service="iris-marin-be",
                iap_grants=(
                    GcpRoleGrant(
                        role=_ACCESSOR_ROLE,
                        members=(
                            *shared_accessors,
                            "domain:openathena.ai",
                            "domain:stanford.edu",
                            f"serviceAccount:github-iris@{project}.iam.gserviceaccount.com",
                            f"serviceAccount:iris-ci-smoke@{project}.iam.gserviceaccount.com",
                            f"serviceAccount:loom-vm@{project}.iam.gserviceaccount.com",
                            f"serviceAccount:rav-agent@{project}.iam.gserviceaccount.com",
                            principals["human-054"],
                            principals["human-061"],
                            principals["human-064"],
                            principals["human-070"],
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
                        role=_ACCESSOR_ROLE,
                        members=(
                            *shared_accessors,
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
