# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
import pulumi_gcp as gcp
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks

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

IAM_MEMBER_TYPES = {
    "gcp:artifactregistry/repositoryIamMember:RepositoryIamMember",
    "gcp:kms/cryptoKeyIAMMember:CryptoKeyIAMMember",
    "gcp:projects/iAMMember:IAMMember",
    "gcp:secretmanager/secretIamMember:SecretIamMember",
    "gcp:serviceaccount/iAMMember:IAMMember",
    "gcp:storage/bucketIAMMember:BucketIAMMember",
}


class AdoptionMocks(Mocks):
    def new_resource(self, args: MockResourceArgs):
        if args.typ in IAM_MEMBER_TYPES:
            assert not args.resource_id
        elif args.typ == "gcp:projects/iAMCustomRole:IAMCustomRole":
            assert args.resource_id == "projects/example/roles/customViewer"
        elif args.typ == "gcp:serviceaccount/account:Account":
            assert args.resource_id == "projects/example/serviceAccounts/worker@example.iam.gserviceaccount.com"
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
    provider = gcp.Provider("gcp", project="example")

    iam = GcpIam(
        "iam",
        GcpIamArgs(
            project="example",
            kms_location="us-central1",
            kms_key_ring="test-key-ring",
            kms_key="test-key",
            custom_roles=(
                GcpCustomRole(
                    role_id="customViewer",
                    title="Custom viewer",
                    description="Read test resources.",
                    permissions=("resourcemanager.projects.get",),
                ),
            ),
            owned_service_accounts=(
                GcpOwnedServiceAccount(account_id="worker", display_name="Worker"),
            ),
            project_grants=(_grant(),),
            kms_grants=(_grant(),),
            secrets=(GcpSecretIam(secret="test-secret", grants=(_grant(),)),),
            buckets=(GcpBucketIam(bucket="test-bucket", grants=(_grant(),)),),
            artifact_repositories=(
                GcpArtifactRepositoryIam(location="us-central1", repository="test-repository", grants=(_grant(),)),
            ),
            service_accounts=(
                GcpServiceAccountIam(email="target@example.iam.gserviceaccount.com", grants=(_grant(),)),
            ),
            adopt=True,
        ),
        gcp_provider=provider,
    )
    return iam.urn.apply(lambda _: None)
