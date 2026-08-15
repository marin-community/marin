# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

import pulumi
import pytest
import yaml
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks

from infra.loom.infrastructure import (
    ROOT,
    DeploymentConfig,
    GitHubFederationConfig,
    HomeFileConfig,
    ProfileConfig,
    WorkloadIdentityConfig,
    _deployment_manifest,
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
        dotenv_secret_version=3,
        snapshot_retention_days=14,
        vm_project_roles=("roles/cloudsql.client", "roles/cloudsql.instanceUser"),
        vm_pulumi_kms_keys=(
            "projects/example/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key",
        ),
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
                    "instructionsFile": "profiles/ops/AGENTS.md",
                    "env": {"KUBECONFIG": {"secretRef": "projects/example/secrets/ops-kubeconfig/versions/latest"}},
                },
            ),
        ),
        home_files=(
            HomeFileConfig.parse(
                ".kube/coreweave-iris",
                {
                    "secretRef": "projects/example/secrets/coreweave-iris-kubeconfig/versions/4",
                    "mode": "0600",
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


def test_vm_permissions_require_canonical_unique_resource_names() -> None:
    with pytest.raises(ValueError, match="vmProjectRoles"):
        replace(deployment_config(), vm_project_roles=("cloudsql.client",))
    with pytest.raises(ValueError, match="vmPulumiKmsKeys"):
        replace(
            deployment_config(),
            vm_pulumi_kms_keys=(
                "projects/example/locations/us-central1/keyRings/key/cryptoKeys/key",
                "projects/example/locations/us-central1/keyRings/key/cryptoKeys/key",
            ),
        )


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


def test_profile_manifest_renders_github_repositories_and_secret_references() -> None:
    profiles, references = _deployment_profiles(
        (
            ProfileConfig.parse(
                "ops",
                {
                    "agent": "codex",
                    "githubRepositories": ["marin-community/marin", "marin-community/vllm"],
                    "mcpAccess": {"mode": "all", "groups": []},
                    "env": {"OPS_TOKEN": {"secretRef": "projects/example/secrets/ops-token/versions/7"}},
                },
            ),
        )
    )
    assert profiles[0]["profile"]["github_repositories"] == [
        "marin-community/marin",
        "marin-community/vllm",
    ]
    assert profiles[0]["profile"]["mcp_access"] == {"mode": "all", "groups": []}
    assert profiles[0]["env"] == [{"name": "OPS_TOKEN", "secret_ref": "projects/example/secrets/ops-token/versions/7"}]
    assert references == [("example", "ops-token")]
    with pytest.raises(ValueError, match="full secretRef"):
        ProfileConfig.parse("ops", {"agent": "codex", "env": {"OPS_TOKEN": "plaintext"}})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("/root/.kube/config", {"secretRef": "projects/example/secrets/kube/versions/1"}),
        (".kube/../config", {"secretRef": "projects/example/secrets/kube/versions/1"}),
        (".kube/config", {"secretRef": "projects/example/secrets/kube/versions/latest"}),
        (".kube/config", {"secretRef": "projects/example/secrets/kube/versions/1"}),
        (".kube/config", {"secretRef": "projects/example/secrets/kube/versions/1", "mode": "0644"}),
    ],
)
def test_home_files_reject_unsafe_paths_unpinned_versions_and_open_modes(path: str, value: object) -> None:
    with pytest.raises(ValueError, match="homeFiles"):
        HomeFileConfig.parse(path, value)


def test_home_files_reject_overlapping_targets() -> None:
    base = deployment_config()
    nested = replace(base.home_files[0], path=".kube/coreweave-iris/context")
    with pytest.raises(ValueError, match="overlapping path"):
        replace(base, home_files=(*base.home_files, nested))


def test_deployment_manifest_preserves_unicode_profile_instructions() -> None:
    profile = ProfileConfig.parse("github", {"agent": "codex", "instructions": "Prefix comments with 🤖"})
    config = replace(deployment_config(), profiles=(profile,), workloads=(), github_federations=())
    profiles, _ = _deployment_profiles(config.profiles)
    manifest = _deployment_manifest(config, profiles, [])
    assert "🤖" in manifest
    assert json.loads(manifest)["profiles"][0]["profile"]["instructions"] == "Prefix comments with 🤖"


def test_profile_instructions_reject_ambiguous_or_external_sources() -> None:
    with pytest.raises(ValueError, match="only one"):
        ProfileConfig.parse(
            "slack",
            {
                "agent": "codex",
                "instructions": "inline",
                "instructionsFile": "profiles/slack/AGENTS.md",
            },
        )
    with pytest.raises(ValueError, match="under"):
        ProfileConfig.parse("slack", {"agent": "codex", "instructionsFile": "../../AGENTS.md"})


@pytest.mark.parametrize(
    "mcp_access",
    [
        {"mode": "groups", "groups": []},
        {"mode": "all", "groups": ["messaging"]},
        {"mode": "unknown", "groups": []},
    ],
)
def test_profile_mcp_access_rejects_invalid_selections(mcp_access: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="mcpAccess"):
        ProfileConfig.parse("ops", {"agent": "codex", "mcpAccess": mcp_access})


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
        assert not attached
        boot_disk = field(vm.inputs, "boot_disk", "bootDisk")
        assert boot_disk is not None
        assert field(boot_disk, "auto_delete", "autoDelete") is False
        root_disk = by_name(mocks, "loom-root")
        assert root_disk.typ == "gcp:compute/disk:Disk"
        assert root_disk.inputs["name"] == "loom"
        assert boot_disk["source"] == "loom-root_id"
        snapshot_attachment = by_name(mocks, "loom-snapshot-policy")
        assert snapshot_attachment.inputs["disk"] == "loom"
        metadata = vm.inputs["metadata"]
        assert metadata["dotenv-secret-version"] == "3"
        assert json.loads(metadata["docker-daemon-config"]) == {
            "data-root": "/var/lib/docker",
            "default-ulimits": {"core": {"Name": "core", "Hard": 0, "Soft": 0}},
        }
        assert yaml.safe_load(metadata["loom-compose"])["services"]["loom"]["working_dir"] == "/home/app"
        assert "data-disk-device" not in metadata
        assert "startup-script" in metadata
        assert "loom-compose" in metadata
        assert "loom-caddyfile" in metadata
        assert "loom-home-file-materializer" in metadata
        assert json.loads(metadata["loom-home-files"]) == [
            {
                "mode": "0600",
                "path": ".kube/coreweave-iris",
                "project": "example",
                "secret": "coreweave-iris-kubeconfig",
                "version": "4",
            }
        ]
        assert "metadataStartupScript" not in vm.inputs
        assert "metadata_startup_script" not in vm.inputs
        assert field(vm.inputs, "allow_stopping_for_update", "allowStoppingForUpdate") is False

        secret_reader = by_name(mocks, "loom-vm-secret-reader")
        assert secret_reader.inputs["role"] == "roles/secretmanager.secretAccessor"
        runtime_secret_readers = [
            resource for resource in mocks.resources if resource.name.startswith("loom-profile-secret-")
        ]
        assert {field(resource.inputs, "secret_id", "secretId") for resource in runtime_secret_readers} == {
            "coreweave-iris-kubeconfig",
            "ops-kubeconfig",
        }
        log_writer = by_name(mocks, "loom-vm-log-writer")
        assert log_writer.inputs["role"] == "roles/logging.logWriter"
        assert log_writer.inputs["member"] == "serviceAccount:loom-vm@example.iam.gserviceaccount.com"
        project_roles = {
            resource.inputs["role"] for resource in mocks.resources if resource.name.startswith("loom-vm-project-role-")
        }
        assert project_roles == {"roles/cloudsql.client", "roles/cloudsql.instanceUser"}
        kms_grant = next(resource for resource in mocks.resources if resource.name.startswith("loom-vm-pulumi-kms-"))
        assert kms_grant.inputs["role"] == "roles/cloudkms.cryptoKeyEncrypterDecrypter"
        assert field(kms_grant.inputs, "crypto_key_id", "cryptoKeyId") == (
            "projects/example/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key"
        )

    return infrastructure.activation.id.apply(check)


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
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-loom", stack="test", preview=False)
    infrastructure = create_infrastructure(replace(deployment_config(), settings=(("slack.profile", "ops"),)))

    def check(_: object) -> None:
        assert by_name(mocks, "loom-workload-marin-ops").typ == "gcp:serviceaccount/account:Account"
        manifest = json.loads(by_name(mocks, "loom").inputs["metadata"]["loom-deployment"])
        assert manifest["prune"] is True
        assert manifest["settings"] == {"slack.profile": "ops"}
        assert manifest["profiles"][0]["profile"]["name"] == "ops"
        assert manifest["profiles"][0]["profile"]["instructions"] == (
            (ROOT / "profiles/ops/AGENTS.md").read_text().strip()
        )
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
