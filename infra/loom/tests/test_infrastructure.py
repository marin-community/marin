# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

import pulumi
import pytest
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks

from infra.loom.infrastructure import (
    DeploymentConfig,
    GitHubFederationConfig,
    ProfileConfig,
    WorkloadIdentityConfig,
    _deployment_profiles,
    _validated_image_reference,
    create_infrastructure,
)


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
        if args.typ == "docker-build:index:Image":
            outputs["digest"] = "sha256:" + "a" * 64
            outputs["ref"] = f"{outputs['tags'][0]}@sha256:" + "a" * 64
        return f"{args.name}_id", outputs

    def call(self, args: MockCallArgs) -> tuple[dict, list[tuple[str, str]] | None]:
        outputs = dict(args.args)
        if args.token == "gcp:serviceaccount/getAccount:getAccount":
            outputs["email"] = f"{args.args['accountId']}@example.iam.gserviceaccount.com"
            outputs["uniqueId"] = "99887766554433221100"
        return outputs, []


def deployment_config() -> DeploymentConfig:
    return DeploymentConfig(
        project="example",
        region="us-central1",
        zone="us-central1-a",
        domain="loom.example.com",
        operator_cidr="203.0.113.7/32",
        dns_zone_id="cloudflare-zone",
        build_context="/tmp/loom-source",
        network="default",
        instance_name="loom",
        vm_service_account_name="loom-vm",
        machine_type="e2-highmem-4",
        boot_disk_gb=100,
        data_disk_gb=500,
        dotenv_secret_version=3,
        snapshot_retention_days=14,
        prune_deployment=True,
        profiles=(
            ProfileConfig.parse(
                "ops",
                {
                    "agent": "codex",
                    "protocol": "acp",
                    "class": "automation",
                    "strict": True,
                    "envClear": True,
                    "env": {"KUBECONFIG": {"secretRef": "projects/example/secrets/ops-kubeconfig/versions/latest"}},
                },
            ),
        ),
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


def infrastructure_and_mocks():
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-loom", stack="test", preview=False)
    infrastructure = create_infrastructure(deployment_config())
    return infrastructure, mocks


def by_name(mocks: RecordingMocks, name: str) -> MockResourceArgs:
    return next(resource for resource in mocks.resources if resource.name == name)


def field(inputs: dict, snake: str, camel: str):
    return inputs.get(snake, inputs.get(camel))


def test_empty_runtime_policy_cannot_prune_existing_profiles() -> None:
    base = deployment_config()
    with pytest.raises(ValueError, match="non-empty runtime policy"):
        replace(base, prune_deployment=True, profiles=(), workloads=(), github_federations=())


def test_domain_is_a_canonical_hostname() -> None:
    with pytest.raises(ValueError, match="canonical hostname"):
        replace(deployment_config(), domain="https://loom.example.com/")


def test_github_federations_require_unique_names_and_known_profiles() -> None:
    base = deployment_config()
    unknown = GitHubFederationConfig.parse(
        {"name": "ci", "repositoryId": "1", "workflowRef": "build.yml", "profile": "missing"}
    )
    with pytest.raises(ValueError, match="unknown profile"):
        replace(base, github_federations=(unknown,))
    duplicate = replace(unknown, profile="ops")
    with pytest.raises(ValueError, match="duplicate GitHub federation"):
        replace(base, github_federations=(duplicate, duplicate))


def test_release_reference_must_be_the_expected_registry_digest() -> None:
    canonical = "us-central1-docker.pkg.dev/example/loom/loom@sha256:" + "a" * 64
    tagged = "us-central1-docker.pkg.dev/example/loom/loom:latest@sha256:" + "a" * 64
    assert _validated_image_reference(canonical, "example", "us-central1") == canonical
    assert _validated_image_reference(tagged, "example", "us-central1") == tagged
    with pytest.raises(ValueError, match="expected Loom image digest"):
        _validated_image_reference("us-central1-docker.pkg.dev/example/loom/loom:main", "example", "us-central1")


def test_profile_manifest_accepts_secret_references_but_rejects_values() -> None:
    profiles, references = _deployment_profiles(
        (
            ProfileConfig.parse(
                "ops",
                {
                    "agent": "codex",
                    "env": {"OPS_TOKEN": {"secretRef": "projects/example/secrets/ops-token/versions/7"}},
                },
            ),
        )
    )
    assert profiles[0]["env"] == [{"name": "OPS_TOKEN", "secret_ref": "projects/example/secrets/ops-token/versions/7"}]
    assert references == [("example", "ops-token")]
    with pytest.raises(ValueError, match="full secretRef"):
        ProfileConfig.parse("ops", {"agent": "codex", "env": {"OPS_TOKEN": "plaintext"}})


@pulumi.runtime.test
def test_deployment_models_durable_resources_without_secret_payloads():
    infrastructure, mocks = infrastructure_and_mocks()

    def check(_: object) -> None:
        resource_types = {resource.typ for resource in mocks.resources}
        assert "gcp:secretmanager/secretVersion:SecretVersion" not in resource_types
        assert "gcp:iam/workloadIdentityPool:WorkloadIdentityPool" not in resource_types
        assert "gcp:iam/workloadIdentityPoolProvider:WorkloadIdentityPoolProvider" not in resource_types

        vm = by_name(mocks, "loom")
        attached = field(vm.inputs, "attached_disks", "attachedDisks")
        assert attached is not None
        assert len(attached) == 1
        assert field(attached[0], "device_name", "deviceName") == "loom-data"
        assert field(attached[0], "auto_delete", "autoDelete") is not True
        assert vm.inputs["metadata"]["dotenv-secret-version"] == "3"
        assert "startup-script" in vm.inputs["metadata"]
        assert "loom-compose" in vm.inputs["metadata"]
        assert "loom-caddyfile" in vm.inputs["metadata"]
        assert "metadataStartupScript" not in vm.inputs
        assert "metadata_startup_script" not in vm.inputs
        assert field(vm.inputs, "allow_stopping_for_update", "allowStoppingForUpdate") is False

        secret_reader = by_name(mocks, "loom-vm-secret-reader")
        assert secret_reader.inputs["role"] == "roles/secretmanager.secretAccessor"
        log_writer = by_name(mocks, "loom-vm-log-writer")
        assert log_writer.inputs["role"] == "roles/logging.logWriter"
        assert log_writer.inputs["member"] == "serviceAccount:loom-vm@example.iam.gserviceaccount.com"

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_local_tree_build_drives_the_runtime_rollout():
    infrastructure, mocks = infrastructure_and_mocks()

    def check(_: object) -> None:
        image = by_name(mocks, "loom-release-image")
        assert image.inputs["buildOnPreview"] is True
        assert image.inputs["context"]["location"] == "/tmp/loom-source"
        assert image.inputs["platforms"] == ["linux/amd64"]
        assert image.inputs["buildArgs"] == {"CARGO_PROFILE": "release"}
        assert image.inputs["labels"] == {
            "org.opencontainers.image.source": "https://github.com/marin-community/loom.git"
        }
        assert image.inputs["tags"] == ["us-central1-docker.pkg.dev/example/loom/loom:latest"]
        assert image.inputs["push"] is True

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_dns_matches_the_existing_unproxied_cloudflare_record():
    infrastructure, mocks = infrastructure_and_mocks()

    def check(_: object) -> None:
        record = by_name(mocks, "loom-dns-address")
        assert record.inputs["name"] == "loom.example.com"
        assert record.inputs["type"] == "A"
        assert record.inputs["ttl"] == 300
        assert record.inputs["proxied"] is False
        assert record.inputs["content"] == "203.0.113.10"

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_release_rollout_pins_metadata_to_the_built_image_digest():
    infrastructure, mocks = infrastructure_and_mocks()

    def check(_: object) -> None:
        metadata = by_name(mocks, "loom").inputs["metadata"]
        assert metadata["loom-image"].endswith("@sha256:" + "a" * 64)
        activation = by_name(mocks, "loom-activate")
        triggers = activation.inputs["triggers"]
        assert "loom_id" in triggers
        serialized_metadata = json.loads(next(trigger for trigger in triggers if trigger != "loom_id"))
        assert serialized_metadata == metadata

    return infrastructure.activation.id.apply(check)


@pulumi.runtime.test
def test_profiles_and_workloads_render_to_vm_metadata():
    infrastructure, mocks = infrastructure_and_mocks()

    def check(_: object) -> None:
        assert by_name(mocks, "loom-workload-marin-ops").typ == "gcp:serviceaccount/account:Account"
        manifest = json.loads(by_name(mocks, "loom").inputs["metadata"]["loom-deployment"])
        assert manifest["prune"] is True
        assert manifest["profiles"][0]["profile"]["name"] == "ops"
        assert manifest["federations"][0]["subject"] == "11223344556677889900"

    return infrastructure.instance.id.apply(check)


@pulumi.runtime.test
def test_existing_service_account_can_be_bound_to_a_workload_profile():
    base = deployment_config()
    grafana = WorkloadIdentityConfig.parse(
        {
            "name": "grafana-alerts",
            "profile": "ops",
            "serviceAccountId": "marin-grafana",
            "createServiceAccount": False,
        }
    )
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-loom", stack="test", preview=False)
    infrastructure = create_infrastructure(replace(base, workloads=(grafana,)))

    def check(_: object) -> None:
        assert not any(resource.name == "loom-workload-grafana-alerts" for resource in mocks.resources)
        manifest = json.loads(by_name(mocks, "loom").inputs["metadata"]["loom-deployment"])
        mapping = manifest["federations"][0]
        assert mapping["service_account"] == "marin-grafana@example.iam.gserviceaccount.com"
        assert mapping["subject"] == "99887766554433221100"
        assert mapping["profiles"] == ["ops"]

    return infrastructure.instance.id.apply(check)
