# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pulumi
import pytest
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks


class RecordingMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        outputs = dict(args.inputs)
        outputs.setdefault("name", args.name)
        if args.typ == "gcp:compute/address:Address":
            outputs["address"] = "203.0.113.10"
        if args.typ == "gcp:serviceaccount/account:Account":
            outputs["email"] = f"{args.name}@example.iam.gserviceaccount.com"
            outputs["uniqueId"] = "11223344556677889900"
            outputs["unique_id"] = "11223344556677889900"
        if args.typ == "gcp:storage/bucket:Bucket":
            outputs["url"] = f"gs://{args.name}"
        if args.typ == "docker-build:index:Image":
            outputs["digest"] = "sha256:" + "a" * 64
            outputs["ref"] = f"{outputs['tags'][0]}@sha256:" + "a" * 64
        return f"{args.name}_id", outputs

    def call(self, args: MockCallArgs) -> tuple[dict, list[tuple[str, str]] | None]:
        outputs = dict(args.args)
        if args.token == "gcp:artifactregistry/getDockerImage:getDockerImage":
            image = "us-central1-docker.pkg.dev/example/loom/loom@sha256:" + "b" * 64
            outputs["selfLink"] = image
            outputs["self_link"] = image
        return outputs, []


mocks = RecordingMocks()
pulumi.runtime.set_mocks(mocks, project="marin-loom", stack="test", preview=False)

from infra.loom.infrastructure import (  # noqa: E402
    DeploymentConfig,
    WorkloadIdentityConfig,
    _profile_manifest,
    _validated_image_reference,
    create_infrastructure,
)


def deployment_config(*, build_commit: str | None = None, runtime_rollout_enabled: bool = False) -> DeploymentConfig:
    return DeploymentConfig(
        project="example",
        region="us-central1",
        zone="us-central1-a",
        domain="loom.example.com",
        operator_cidr="203.0.113.7/32",
        dns_zone_id="cloudflare-zone",
        build_commit=build_commit,
        runtime_rollout_enabled=runtime_rollout_enabled,
        git_ref="0123456789abcdef0123456789abcdef01234567",
        dotenv_secret_version=3,
        prune_deployment=True,
        profiles={
            "ops": {
                "agent": "codex",
                "protocol": "acp",
                "class": "automation",
                "strict": True,
                "envClear": True,
                "env": {"KUBECONFIG": {"secretRef": "projects/example/secrets/ops-kubeconfig/versions/latest"}},
            }
        },
        workloads=(
            WorkloadIdentityConfig.parse(
                {
                    "name": "marin-ops",
                    "profile": "ops",
                    "serviceAccountId": "loom-marin-ops",
                }
            ),
        ),
    )


def make_infrastructure(*, build_commit: str | None = None, runtime_rollout_enabled: bool = False):
    mocks.resources.clear()
    return create_infrastructure(
        deployment_config(build_commit=build_commit, runtime_rollout_enabled=runtime_rollout_enabled)
    )


def by_name(name: str) -> MockResourceArgs:
    return next(resource for resource in mocks.resources if resource.name == name)


def field(inputs: dict, snake: str, camel: str):
    return inputs.get(snake, inputs.get(camel))


def test_build_and_runtime_rollout_require_commits() -> None:
    base = deployment_config()
    with pytest.raises(ValueError, match="40-character gitRef"):
        DeploymentConfig(
            **{
                **base.__dict__,
                "runtime_rollout_enabled": True,
                "git_ref": "main",
            }
        )
    with pytest.raises(ValueError, match="buildCommit"):
        DeploymentConfig(**{**base.__dict__, "build_commit": "main"})
    with pytest.raises(ValueError, match="staged before"):
        DeploymentConfig(
            **{
                **base.__dict__,
                "build_commit": base.git_ref,
                "runtime_rollout_enabled": True,
            }
        )


def test_empty_runtime_policy_cannot_prune_existing_profiles() -> None:
    base = deployment_config()
    with pytest.raises(ValueError, match="non-empty runtime policy"):
        DeploymentConfig(
            **{
                **base.__dict__,
                "prune_deployment": True,
                "profiles": {},
                "workloads": (),
                "github_federations": (),
            }
        )


def test_release_reference_must_be_the_expected_registry_digest() -> None:
    valid = "us-central1-docker.pkg.dev/example/loom/loom@sha256:" + "a" * 64
    assert _validated_image_reference(valid, "example", "us-central1") == valid
    with pytest.raises(ValueError, match="expected Loom image digest"):
        _validated_image_reference("us-central1-docker.pkg.dev/example/loom/loom:main", "example", "us-central1")


def test_profile_manifest_accepts_secret_references_but_rejects_values() -> None:
    profiles, references = _profile_manifest(
        {
            "ops": {
                "agent": "codex",
                "env": {"OPS_TOKEN": {"secretRef": "projects/example/secrets/ops-token/versions/7"}},
            }
        }
    )
    assert profiles[0]["env"] == [{"name": "OPS_TOKEN", "secret_ref": "projects/example/secrets/ops-token/versions/7"}]
    assert references == [("example", "ops-token")]
    with pytest.raises(ValueError, match="full secretRef"):
        _profile_manifest({"ops": {"agent": "codex", "env": {"OPS_TOKEN": "plaintext"}}})


@pulumi.runtime.test
def test_phase_a_models_durable_resources_without_secret_payloads():
    infrastructure = make_infrastructure()

    def check(_: object) -> None:
        resource_types = {resource.typ for resource in mocks.resources}
        assert "gcp:secretmanager/secretVersion:SecretVersion" not in resource_types
        assert "gcp:iam/workloadIdentityPool:WorkloadIdentityPool" not in resource_types
        assert "gcp:iam/workloadIdentityPoolProvider:WorkloadIdentityPoolProvider" not in resource_types

        vm = by_name("loom")
        attached = field(vm.inputs, "attached_disks", "attachedDisks")
        assert attached is not None
        assert len(attached) == 1
        assert field(attached[0], "device_name", "deviceName") == "loom-data"
        assert field(attached[0], "auto_delete", "autoDelete") is not True
        assert vm.inputs["metadata"]["image-mode"] == "disabled"
        assert vm.inputs["metadata"]["dotenv-secret-version"] == "3"
        assert "startup-script" in vm.inputs["metadata"]
        assert "metadataStartupScript" not in vm.inputs
        assert "metadata_startup_script" not in vm.inputs
        assert field(vm.inputs, "allow_stopping_for_update", "allowStoppingForUpdate") is False

        secret_reader = by_name("loom-vm-secret-reader")
        assert secret_reader.inputs["role"] == "roles/secretmanager.secretAccessor"
        assert by_name("loom-backups").inputs["publicAccessPrevention"] == "enforced"

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_phase_a_creates_the_repository_without_building_an_image():
    infrastructure = make_infrastructure()

    def check(_: object) -> None:
        repository = by_name("loom-images")
        docker_config = field(repository.inputs, "docker_config", "dockerConfig")
        assert field(docker_config, "immutable_tags", "immutableTags") is True
        assert not any(resource.typ == "docker-build:index:Image" for resource in mocks.resources)

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_release_build_precedes_the_runtime_rollout():
    infrastructure = make_infrastructure(build_commit="0123456789abcdef0123456789abcdef01234567")

    def check(_: object) -> None:
        image = by_name("loom-release-image")
        assert image.inputs["buildOnPreview"] is False
        assert image.inputs["context"]["location"].endswith(".git#0123456789abcdef0123456789abcdef01234567")
        assert image.inputs["platforms"] == ["linux/amd64"]
        assert image.inputs["buildArgs"] == {"CARGO_PROFILE": "release"}
        assert image.inputs["labels"] == {
            "org.opencontainers.image.revision": "0123456789abcdef0123456789abcdef01234567",
            "org.opencontainers.image.source": "https://github.com/marin-community/loom.git",
        }
        assert image.inputs["push"] is True
        vm = by_name("loom")
        assert field(vm.inputs, "allow_stopping_for_update", "allowStoppingForUpdate") is False
        assert vm.inputs["metadata"]["image-mode"] == "disabled"

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_dns_matches_the_existing_unproxied_cloudflare_record():
    infrastructure = make_infrastructure()

    def check(_: object) -> None:
        record = by_name("loom-dns-address")
        assert record.inputs["name"] == "loom.example.com"
        assert record.inputs["type"] == "A"
        assert record.inputs["ttl"] == 300
        assert record.inputs["proxied"] is False
        assert record.inputs["content"] == "203.0.113.10"

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_release_rollout_pins_metadata_to_the_reviewed_commit_and_digest():
    infrastructure = make_infrastructure(runtime_rollout_enabled=True)

    def check(_: object) -> None:
        metadata = by_name("loom").inputs["metadata"]
        assert (
            field(
                by_name("loom").inputs,
                "allow_stopping_for_update",
                "allowStoppingForUpdate",
            )
            is True
        )
        assert metadata["git-ref"] == "0123456789abcdef0123456789abcdef01234567"
        assert metadata["image-mode"] == "pull"
        assert metadata["ar-image"].endswith("@sha256:" + "b" * 64)
        assert not any(resource.typ == "docker-build:index:Image" for resource in mocks.resources)

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_next_release_can_build_while_the_previous_commit_stays_deployed():
    infrastructure = make_infrastructure(build_commit="f" * 40, runtime_rollout_enabled=True)

    def check(_: object) -> None:
        image = by_name("loom-release-image")
        assert image.inputs["tags"] == ["us-central1-docker.pkg.dev/example/loom/loom:" + "f" * 40]
        metadata = by_name("loom").inputs["metadata"]
        assert metadata["git-ref"] == "0123456789abcdef0123456789abcdef01234567"
        assert metadata["ar-image"].endswith("@sha256:" + "b" * 64)

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_profiles_workloads_and_monitoring_render_to_vm_metadata():
    infrastructure = make_infrastructure()

    def check(_: object) -> None:
        names = {resource.name for resource in mocks.resources}
        assert {"loom-workload-marin-ops", "loom-readiness", "loom-readiness-failed", "loom-operations"} <= names
        manifest = json.loads(by_name("loom").inputs["metadata"]["loom-deployment"])
        assert manifest["prune"] is True
        assert manifest["profiles"][0]["profile"]["name"] == "ops"
        assert manifest["federations"][0]["subject"] == "11223344556677889900"

    return infrastructure.dashboard.id.apply(check)
