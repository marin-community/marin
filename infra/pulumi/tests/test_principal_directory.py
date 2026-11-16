# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import pulumi
import pulumi_gcp as gcp
from google.cloud import kms_v1
from iac.gcp.iam import GcpIam
from iac.gcp.iam_config import load_iam_config
from iac.gcp.iam_targets import global_iam_args
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks


class DirectoryMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        return f"{args.name}_id", args.inputs

    def call(self, args: MockCallArgs):
        return args.args, []


@pulumi.runtime.test
def test_iam_publishes_registry_as_secret_directory_with_controller_read_access(monkeypatch):
    config = load_iam_config()
    expected = {
        principal.principal_id: f"person-{index}@example.com" for index, principal in enumerate(config.principals)
    }
    decrypted = {
        base64.b64decode(principal.ciphertext): expected[principal.principal_id] for principal in config.principals
    }

    class FakeKms:
        def decrypt(self, *, name: str, ciphertext: bytes):
            return kms_v1.DecryptResponse(plaintext=decrypted[ciphertext].encode())

    monkeypatch.setattr(kms_v1, "KeyManagementServiceClient", FakeKms)
    mocks = DirectoryMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    versions = []
    secret_version = gcp.secretmanager.SecretVersion

    def record_version(*args, **kwargs):
        version = secret_version(*args, **kwargs)
        versions.append(version)
        return version

    monkeypatch.setattr(gcp.secretmanager, "SecretVersion", record_version)
    bindings = []
    secret_binding = gcp.secretmanager.SecretIamBinding

    def record_binding(*args, **kwargs):
        binding = secret_binding(*args, **kwargs)
        if kwargs["secret_id"] == "iris-principal-directory":
            bindings.append(binding)
        return binding

    monkeypatch.setattr(gcp.secretmanager, "SecretIamBinding", record_binding)
    GcpIam("iam", global_iam_args("example", config), gcp_provider=gcp.Provider("gcp", project="example"))

    def check(values):
        payload, secret, _ = values
        assert secret is True
        assert json.loads(payload) == expected
        container = next(resource for resource in mocks.resources if resource.typ == "gcp:secretmanager/secret:Secret")
        assert container.inputs["project"] == "example"
        assert container.inputs["secretId"] == "iris-principal-directory"
        binding = next(
            resource
            for resource in mocks.resources
            if resource.typ == "gcp:secretmanager/secretIamBinding:SecretIamBinding"
            and resource.inputs["secretId"] == "iris-principal-directory"
        )
        assert binding.inputs["role"] == "roles/secretmanager.secretAccessor"
        assert "serviceAccount:iris-controller@hai-gcp-models.iam.gserviceaccount.com" in binding.inputs["members"]

    return pulumi.Output.all(
        versions[0].secret_data, pulumi.Output.is_secret(versions[0].secret_data), bindings[0].urn
    ).apply(check)
