# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
import pulumi_gcp as gcp
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
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks

TEST_PROJECT = "example"
CUSTOM_ROLE_ID = "customViewer"
OWNED_SERVICE_ACCOUNT_ID = "worker"
IAM_MEMBER_TYPES = frozenset(
    {
        "gcp:artifactregistry/repositoryIamMember:RepositoryIamMember",
        "gcp:kms/cryptoKeyIAMMember:CryptoKeyIAMMember",
        "gcp:projects/iAMMember:IAMMember",
        "gcp:secretmanager/secretIamMember:SecretIamMember",
        "gcp:serviceaccount/iAMMember:IAMMember",
        "gcp:storage/bucketIAMMember:BucketIAMMember",
    }
)


class AdoptionMocks(Mocks):
    def new_resource(self, args: MockResourceArgs):
        if args.typ in IAM_MEMBER_TYPES:
            assert not args.resource_id
        elif args.typ == "gcp:projects/iAMCustomRole:IAMCustomRole":
            assert args.resource_id == f"projects/{TEST_PROJECT}/roles/{CUSTOM_ROLE_ID}"
        elif args.typ == "gcp:serviceaccount/account:Account":
            assert args.resource_id == (
                f"projects/{TEST_PROJECT}/serviceAccounts/"
                f"{OWNED_SERVICE_ACCOUNT_ID}@{TEST_PROJECT}.iam.gserviceaccount.com"
            )
        return args.resource_id or f"{args.name}_id", args.inputs

    def call(self, args: MockCallArgs) -> tuple[dict, list[tuple[str, str]] | None]:
        return args.args, []


def _grant() -> GcpRoleGrant:
    return GcpRoleGrant(role="roles/viewer", members=("serviceAccount:reader@example.com",))


@pulumi.runtime.test
def test_adoption_imports_owned_resources_without_reimporting_iam_members(monkeypatch):
    monkeypatch.setattr(iam_module.kms_v1, "KeyManagementServiceClient", object)
    mocks = AdoptionMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    provider = gcp.Provider("gcp", project=TEST_PROJECT)

    iam = GcpIam(
        "iam",
        GcpIamArgs(
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
        ),
        gcp_provider=provider,
    )
    return iam.urn.apply(lambda _: None)
