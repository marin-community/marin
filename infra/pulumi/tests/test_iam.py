# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast

import pulumi
from iac.gcp import iam as iam_module
from iac.gcp.iam import (
    GcpArtifactRepositoryIam,
    GcpBucketIam,
    GcpCloudRunIapIam,
    GcpCustomRole,
    GcpIam,
    GcpIamArgs,
    GcpIamCondition,
    GcpOwnedServiceAccount,
    GcpRoleGrant,
    GcpSecretIam,
    GcpServiceAccountIam,
)
from iac.imports import ImportCatalog

TEST_PROJECT = "example"
CUSTOM_ROLE_ID = "customViewer"
OWNED_SERVICE_ACCOUNT_ID = "worker"
KMS_RESOURCE_ID = f"projects/{TEST_PROJECT}/locations/us-central1/keyRings/test-key-ring/cryptoKeys/test-key"
CUSTOM_ROLE_TYPE = "gcp:projects/iAMCustomRole:IAMCustomRole"
OWNED_SERVICE_ACCOUNT_TYPE = "gcp:serviceaccount/account:Account"
ARTIFACT_REPOSITORY_IAM_BINDING_TYPE = "gcp:artifactregistry/repositoryIamBinding:RepositoryIamBinding"
KMS_IAM_BINDING_TYPE = "gcp:kms/cryptoKeyIAMBinding:CryptoKeyIAMBinding"
PROJECT_IAM_BINDING_TYPE = "gcp:projects/iAMBinding:IAMBinding"
SECRET_IAM_BINDING_TYPE = "gcp:secretmanager/secretIamBinding:SecretIamBinding"
SERVICE_ACCOUNT_IAM_BINDING_TYPE = "gcp:serviceaccount/iAMBinding:IAMBinding"
BUCKET_IAM_BINDING_TYPE = "gcp:storage/bucketIAMBinding:BucketIAMBinding"
IAP_CLOUD_RUN_SERVICE_IAM_BINDING_TYPE = "gcp:iap/webCloudRunServiceIamBinding:WebCloudRunServiceIamBinding"
IAM_BINDING_TYPES = frozenset(
    {
        ARTIFACT_REPOSITORY_IAM_BINDING_TYPE,
        KMS_IAM_BINDING_TYPE,
        PROJECT_IAM_BINDING_TYPE,
        SECRET_IAM_BINDING_TYPE,
        SERVICE_ACCOUNT_IAM_BINDING_TYPE,
        BUCKET_IAM_BINDING_TYPE,
        IAP_CLOUD_RUN_SERVICE_IAM_BINDING_TYPE,
    }
)


def _grant(member: str = "serviceAccount:reader@example.com") -> GcpRoleGrant:
    return GcpRoleGrant(role="roles/viewer", members=(member,))


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
        project_grants=(_grant(), _grant("serviceAccount:writer@example.com")),
        kms_grants=(_grant(),),
        secrets=(
            GcpSecretIam(secret="test-secret", grants=(_grant(),)),
            GcpSecretIam(secret="test-secret", grants=(_grant("serviceAccount:writer@example.com"),)),
        ),
        buckets=(
            GcpBucketIam(
                bucket="test-bucket",
                grants=(
                    GcpRoleGrant(
                        role="roles/viewer",
                        members=("serviceAccount:reader@example.com",),
                        condition=GcpIamCondition(
                            title="expires-2027",
                            description="Temporary test access.",
                            expression="request.time < timestamp('2027-01-01T00:00:00Z')",
                        ),
                    ),
                ),
            ),
        ),
        artifact_repositories=(
            GcpArtifactRepositoryIam(location="us-central1", repository="test-repository", grants=(_grant(),)),
        ),
        service_accounts=(GcpServiceAccountIam(email="target@example.iam.gserviceaccount.com", grants=(_grant(),)),),
        cloud_run_iap=(
            GcpCloudRunIapIam(
                location="us-central1",
                service="test-service",
                iap_grants=(_grant(),),
            ),
        ),
    )


def _resource_recorder(
    resource_type: str,
    options_by_type: dict[str, pulumi.ResourceOptions],
    arguments_by_type: dict[str, dict[str, object]],
) -> Callable[..., pulumi.Resource]:
    def record(logical_name: str, **kwargs) -> pulumi.Resource:
        options_by_type[resource_type] = kwargs["opts"]
        arguments_by_type[resource_type] = kwargs
        resource = object.__new__(pulumi.Resource)
        resource._type = resource_type
        resource._name = logical_name
        return resource

    return record


def test_gcp_iam_catalogs_provider_ids_without_in_program_imports(monkeypatch):
    options_by_type: dict[str, pulumi.ResourceOptions] = {}
    arguments_by_type: dict[str, dict[str, object]] = {}
    constructors = (
        (
            iam_module.gcp.artifactregistry,
            "RepositoryIamBinding",
            ARTIFACT_REPOSITORY_IAM_BINDING_TYPE,
        ),
        (iam_module.gcp.kms, "CryptoKeyIAMBinding", KMS_IAM_BINDING_TYPE),
        (iam_module.gcp.projects, "IAMBinding", PROJECT_IAM_BINDING_TYPE),
        (iam_module.gcp.secretmanager, "SecretIamBinding", SECRET_IAM_BINDING_TYPE),
        (iam_module.gcp.serviceaccount, "IAMBinding", SERVICE_ACCOUNT_IAM_BINDING_TYPE),
        (iam_module.gcp.storage, "BucketIAMBinding", BUCKET_IAM_BINDING_TYPE),
        (iam_module.gcp.projects, "IAMCustomRole", CUSTOM_ROLE_TYPE),
        (iam_module.gcp.serviceaccount, "Account", OWNED_SERVICE_ACCOUNT_TYPE),
        (iam_module.gcp.iap, "WebCloudRunServiceIamBinding", IAP_CLOUD_RUN_SERVICE_IAM_BINDING_TYPE),
    )
    for namespace, name, resource_type in constructors:
        monkeypatch.setattr(namespace, name, _resource_recorder(resource_type, options_by_type, arguments_by_type))

    monkeypatch.setattr(iam_module.kms_v1, "KeyManagementServiceClient", object)

    def initialize_component(resource, resource_type, name, *_args, **_kwargs):
        resource._type = resource_type
        resource._name = name

    monkeypatch.setattr(pulumi.ComponentResource, "__init__", initialize_component)
    monkeypatch.setattr(GcpIam, "register_outputs", lambda *_args, **_kwargs: None)

    imports = ImportCatalog()
    GcpIam("iam", _args(), gcp_provider=cast(pulumi.ProviderResource, None), imports=imports)

    assert set(options_by_type) == IAM_BINDING_TYPES | {CUSTOM_ROLE_TYPE, OWNED_SERVICE_ACCOUNT_TYPE}
    assert all(options.import_ is None for options in options_by_type.values())
    assert arguments_by_type[PROJECT_IAM_BINDING_TYPE]["members"] == [
        "serviceAccount:reader@example.com",
        "serviceAccount:writer@example.com",
    ]
    assert arguments_by_type[SECRET_IAM_BINDING_TYPE]["members"] == [
        "serviceAccount:reader@example.com",
        "serviceAccount:writer@example.com",
    ]
    assert {spec.identity.resource_type: spec.provider_id for spec in imports.specs} == {
        PROJECT_IAM_BINDING_TYPE: f"{TEST_PROJECT} roles/viewer",
        KMS_IAM_BINDING_TYPE: f"{KMS_RESOURCE_ID} roles/viewer",
        SECRET_IAM_BINDING_TYPE: f"projects/{TEST_PROJECT}/secrets/test-secret roles/viewer",
        BUCKET_IAM_BINDING_TYPE: "b/test-bucket roles/viewer expires-2027",
        ARTIFACT_REPOSITORY_IAM_BINDING_TYPE: (
            f"projects/{TEST_PROJECT}/locations/us-central1/repositories/test-repository roles/viewer"
        ),
        SERVICE_ACCOUNT_IAM_BINDING_TYPE: (
            f"projects/{TEST_PROJECT}/serviceAccounts/target@example.iam.gserviceaccount.com roles/viewer"
        ),
        IAP_CLOUD_RUN_SERVICE_IAM_BINDING_TYPE: (
            f"projects/{TEST_PROJECT}/iap_web/cloud_run-us-central1/services/test-service roles/viewer"
        ),
        CUSTOM_ROLE_TYPE: f"projects/{TEST_PROJECT}/roles/{CUSTOM_ROLE_ID}",
        OWNED_SERVICE_ACCOUNT_TYPE: (
            f"projects/{TEST_PROJECT}/serviceAccounts/"
            f"{OWNED_SERVICE_ACCOUNT_ID}@{TEST_PROJECT}.iam.gserviceaccount.com"
        ),
    }
