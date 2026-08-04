# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast

import pulumi
from iac.gcp import iam as iam_module
from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpBucketIam,
    GcpCustomRole,
    GcpIam,
    GcpIamArgs,
    GcpOwnedServiceAccount,
    GcpRoleGrant,
    GcpSecretIam,
    GcpServiceAccountIam,
)

TEST_PROJECT = "example"
CUSTOM_ROLE_ID = "customViewer"
OWNED_SERVICE_ACCOUNT_ID = "worker"
OWNED_SERVICE_ACCOUNT_IMPORT_ID = (
    f"projects/{TEST_PROJECT}/serviceAccounts/{OWNED_SERVICE_ACCOUNT_ID}@{TEST_PROJECT}.iam.gserviceaccount.com"
)
CUSTOM_ROLE_TYPE = "gcp:projects/iAMCustomRole:IAMCustomRole"
OWNED_SERVICE_ACCOUNT_TYPE = "gcp:serviceaccount/account:Account"
ARTIFACT_REPOSITORY_IAM_MEMBER_TYPE = "gcp:artifactregistry/repositoryIamMember:RepositoryIamMember"
KMS_IAM_MEMBER_TYPE = "gcp:kms/cryptoKeyIAMMember:CryptoKeyIAMMember"
PROJECT_IAM_MEMBER_TYPE = "gcp:projects/iAMMember:IAMMember"
SECRET_IAM_MEMBER_TYPE = "gcp:secretmanager/secretIamMember:SecretIamMember"
SERVICE_ACCOUNT_IAM_MEMBER_TYPE = "gcp:serviceaccount/iAMMember:IAMMember"
BUCKET_IAM_MEMBER_TYPE = "gcp:storage/bucketIAMMember:BucketIAMMember"
IAM_MEMBER_TYPES = frozenset(
    {
        ARTIFACT_REPOSITORY_IAM_MEMBER_TYPE,
        KMS_IAM_MEMBER_TYPE,
        PROJECT_IAM_MEMBER_TYPE,
        SECRET_IAM_MEMBER_TYPE,
        SERVICE_ACCOUNT_IAM_MEMBER_TYPE,
        BUCKET_IAM_MEMBER_TYPE,
    }
)


def _grant() -> GcpRoleGrant:
    return GcpRoleGrant(role="roles/viewer", members=("serviceAccount:reader@example.com",))


def _args() -> GcpIamArgs:
    return GcpIamArgs(
        project=TEST_PROJECT,
        kms_location="us-central1",
        kms_key_ring="test-key-ring",
        kms_key="test-key",
        custom_roles=(
            GcpCustomRole(
                role_id=CUSTOM_ROLE_ID,
                title="Custom viewer",
                description="Read test resources.",
                permissions=("resourcemanager.projects.get",),
            ),
        ),
        owned_service_accounts=(GcpOwnedServiceAccount(account_id=OWNED_SERVICE_ACCOUNT_ID, display_name="Worker"),),
        project_grants=(_grant(),),
        kms_grants=(_grant(),),
        secrets=(GcpSecretIam(secret="test-secret", grants=(_grant(),)),),
        buckets=(GcpBucketIam(bucket="test-bucket", grants=(_grant(),)),),
        artifact_repositories=(
            GcpArtifactRepositoryIam(location="us-central1", repository="test-repository", grants=(_grant(),)),
        ),
        service_accounts=(GcpServiceAccountIam(email="target@example.iam.gserviceaccount.com", grants=(_grant(),)),),
        adopt=True,
    )


def _resource_recorder(
    resource_type: str,
    options_by_type: dict[str, pulumi.ResourceOptions],
) -> Callable[..., pulumi.Resource]:
    def record(_name: str, **kwargs) -> pulumi.Resource:
        options_by_type[resource_type] = kwargs["opts"]
        return object.__new__(pulumi.Resource)

    return record


def test_adoption_imports_owned_resources_without_reimporting_iam_members(monkeypatch):
    options_by_type: dict[str, pulumi.ResourceOptions] = {}
    constructors = (
        (
            iam_module.gcp.artifactregistry,
            "RepositoryIamMember",
            ARTIFACT_REPOSITORY_IAM_MEMBER_TYPE,
        ),
        (iam_module.gcp.kms, "CryptoKeyIAMMember", KMS_IAM_MEMBER_TYPE),
        (iam_module.gcp.projects, "IAMMember", PROJECT_IAM_MEMBER_TYPE),
        (iam_module.gcp.secretmanager, "SecretIamMember", SECRET_IAM_MEMBER_TYPE),
        (iam_module.gcp.serviceaccount, "IAMMember", SERVICE_ACCOUNT_IAM_MEMBER_TYPE),
        (iam_module.gcp.storage, "BucketIAMMember", BUCKET_IAM_MEMBER_TYPE),
        (iam_module.gcp.projects, "IAMCustomRole", CUSTOM_ROLE_TYPE),
        (iam_module.gcp.serviceaccount, "Account", OWNED_SERVICE_ACCOUNT_TYPE),
    )
    for namespace, name, resource_type in constructors:
        monkeypatch.setattr(namespace, name, _resource_recorder(resource_type, options_by_type))

    monkeypatch.setattr(iam_module.kms_v1, "KeyManagementServiceClient", object)
    monkeypatch.setattr(pulumi.ComponentResource, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GcpIam, "register_outputs", lambda *_args, **_kwargs: None)

    GcpIam("iam", _args(), gcp_provider=cast(pulumi.ProviderResource, None))

    assert set(options_by_type) == IAM_MEMBER_TYPES | {CUSTOM_ROLE_TYPE, OWNED_SERVICE_ACCOUNT_TYPE}
    assert all(options_by_type[resource_type].import_ is None for resource_type in IAM_MEMBER_TYPES)
    assert options_by_type[CUSTOM_ROLE_TYPE].import_ == f"projects/{TEST_PROJECT}/roles/{CUSTOM_ROLE_ID}"
    assert options_by_type[OWNED_SERVICE_ACCOUNT_TYPE].import_ == OWNED_SERVICE_ACCOUNT_IMPORT_ID
