# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import fnmatch

import pulumi
import pulumi_coreweave as coreweave
import pytest
from iac.auth.broker import coreweave_identity_metadata
from iac.coreweave.identity import CoreweaveIdentity, CoreweaveIdentityPolicies, identity_policy_statements
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks
from rigging.grants import Permission

ISSUER = "https://auth.example.org/v1/identity/oidc"
SUBJECT = "ad14b571-2449-4fd5-974e-f525ed965370"


class RecordingMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        return args.name, args.inputs

    def call(self, args: MockCallArgs):
        return args.args


def test_coreweave_identity_metadata_excludes_credential_references():
    metadata = coreweave_identity_metadata(
        [
            {
                "name": "checkpoint-reader",
                "subject": SUBJECT,
                "permissions": [
                    {"service": "cw_s3", "actions": ["s3:GetObject"], "resource": "s3://checkpoints/run-1/"}
                ],
                "role": "marin-coreweave",
                "token_file": "/var/run/marin-auth/private-identity/token",
            }
        ]
    )

    assert metadata == {
        "checkpoint-reader": {
            "subject": SUBJECT,
            "permissions": [{"service": "cw_s3", "actions": ["s3:GetObject"], "resource": "s3://checkpoints/run-1/"}],
        }
    }


@pulumi.runtime.test
def test_coreweave_policies_resolve_identity_metadata_with_stable_declared_resource_names():
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-buckets", stack="test", preview=False)
    identities = {
        "reader": pulumi.Output.from_input(
            CoreweaveIdentity(SUBJECT, (Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://checkpoints/run-1/"),))
        ),
        "writer": pulumi.Output.from_input(
            CoreweaveIdentity(
                "583c8758-fd78-4de7-9a3c-3866343dc9ea",
                (Permission("cw_s3", frozenset({"s3:PutObject"}), "s3://checkpoints/run-2/"),),
            )
        ),
    }
    component = CoreweaveIdentityPolicies(
        "auth-identities",
        issuer=pulumi.Output.from_input(ISSUER),
        identities=identities,
        coreweave_provider=coreweave.Provider("test"),
    )

    def check(_: list) -> None:
        policies = {
            resource.name: resource.inputs
            for resource in mocks.resources
            if resource.typ
            == "coreweave:index/objectStorageOrganizationAccessPolicy:ObjectStorageOrganizationAccessPolicy"
        }
        assert set(policies) == {"auth-identities-reader", "auth-identities-writer"}
        assert policies["auth-identities-reader"]["name"] == "marin-auth-reader"
        assert policies["auth-identities-reader"]["statements"][1] == {
            "name": "objects-0",
            "effect": "Allow",
            "actions": ["s3:GetObject"],
            "resources": ["checkpoints/run-1/*"],
            "principals": [f"role/{ISSUER}:{SUBJECT}"],
        }
        assert policies["auth-identities-writer"]["statements"][1] == {
            "name": "objects-0",
            "effect": "Allow",
            "actions": ["s3:PutObject"],
            "resources": ["checkpoints/run-2/*"],
            "principals": [f"role/{ISSUER}:583c8758-fd78-4de7-9a3c-3866343dc9ea"],
        }

    return pulumi.Output.all(*(policy.statements for policy in component.policies.values())).apply(check)


def test_coreweave_identity_policy_scopes_object_access_and_credential_creation():
    identity = CoreweaveIdentity(
        subject=SUBJECT,
        permissions=(Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://checkpoints/experiments/run-1/"),),
    )
    statements = identity_policy_statements(ISSUER, identity)

    def allowed(action: str, resource: str, principal: str) -> bool:
        return any(
            statement["effect"] == "Allow"
            and action in statement["actions"]
            and principal in statement["principals"]
            and any(fnmatch.fnmatchcase(resource, pattern) for pattern in statement["resources"])
            for statement in statements
        )

    principal = f"role/{ISSUER}:{SUBJECT}"
    assert allowed("cwobject:CreateAccessKeyOIDC", "*", principal)
    assert allowed("s3:GetObject", "checkpoints/experiments/run-1/model.bin", principal)
    assert not allowed("s3:GetObject", "checkpoints/experiments/run-10/model.bin", principal)
    assert not allowed("s3:GetObject", "different-bucket/experiments/run-1/model.bin", principal)
    assert not allowed("s3:GetObject", "checkpoints/experiments/run-1/model.bin", "role/another-identity")
    assert not allowed("s3:PutObject", "checkpoints/experiments/run-1/model.bin", principal)
    assert not allowed("s3:ListBucket", "checkpoints", principal)


def test_coreweave_identity_policy_rejects_unrepresentable_list_prefix_scope():
    identity = CoreweaveIdentity(
        subject=SUBJECT,
        permissions=(Permission("cw_s3", frozenset({"s3:ListBucket"}), "s3://checkpoints/experiments/run-1/"),),
    )
    with pytest.raises(ValueError, match="cannot be restricted to an object prefix"):
        identity_policy_statements(ISSUER, identity)


def test_coreweave_identity_bucket_actions_use_short_bucket_name():
    identity = CoreweaveIdentity(
        subject=SUBJECT,
        permissions=(Permission("cw_s3", frozenset({"s3:ListBucket", "s3:GetObject"}), "s3://checkpoints/"),),
    )
    statements = identity_policy_statements(ISSUER, identity)
    assert {(tuple(statement["actions"]), tuple(statement["resources"])) for statement in statements[1:]} == {
        (("s3:GetObject",), ("checkpoints/*",)),
        (("s3:ListBucket",), ("checkpoints",)),
    }
