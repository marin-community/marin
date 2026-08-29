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
        machine_type="c4d-highmem-4",
        boot_disk_name="loom-hyperdisk",
        boot_disk_type="hyperdisk-balanced",
        boot_disk_gb=100,
        boot_disk_iops=3000,
        boot_disk_throughput=140,
        boot_disk_snapshot="loom-pre-c4d-hyperdisk-20260816",
        dotenv_secret_version=3,
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


def test_profile_secrets_must_share_the_deployment_project() -> None:
    profile = ProfileConfig.parse(
        "ops",
        {"agent": "codex", "env": {"TOKEN": {"secretRef": "projects/other/secrets/token/versions/latest"}}},
    )

    with pytest.raises(ValueError, match="secretRef must use project 'example'"):
        replace(deployment_config(), profiles=(profile,))


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


def test_fork_ferry_workflow_stays_within_loom_profile_capacity() -> None:
    stack = yaml.safe_load((ROOT / "Pulumi.marin-loom.yaml").read_text())
    workflow = yaml.safe_load((ROOT.parent.parent / ".github/workflows/ops-fork-ferry.yaml").read_text())
    max_concurrent = stack["config"]["marin-loom:profiles"]["fork-ferry"]["maxConcurrent"]
    units = workflow["jobs"]["ferry"]["strategy"]["matrix"]["include"]

    assert len(units) <= max_concurrent


def test_loom_pr_review_launcher_never_executes_pull_request_code() -> None:
    workflow_path = ROOT.parent.parent / ".github/workflows/ops-loom-review.yaml"
    workflow = yaml.safe_load(workflow_path.read_text())
    trigger = workflow.get("on", workflow.get(True))
    job = workflow["jobs"]["review"]

    assert set(trigger) == {"pull_request_target"}
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    checkout, launch = job["steps"]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["sparse-checkout"] == ".github/actions/launch-loom-run"
    assert launch["uses"] == "./.github/actions/launch-loom-run"
    assert "head.repo.full_name == github.repository" in job["if"]


def test_loom_launch_action_uses_registered_automation_endpoint() -> None:
    action_path = ROOT.parent.parent / ".github/actions/launch-loom-run/action.yaml"
    action = yaml.safe_load(action_path.read_text())
    launch_script = action["runs"]["steps"][0]["run"]

    assert '"$LOOM_URL/api/runs/create"' in launch_script


def test_loom_launch_action_bounds_scratch_attachments_before_upload() -> None:
    action_path = ROOT.parent.parent / ".github/actions/launch-loom-run/action.yaml"
    action = yaml.safe_load(action_path.read_text())
    scratch_input = action["inputs"]["scratch-file"]
    launch_script = action["runs"]["steps"][0]["run"]

    # The composite action's shell is the deployment boundary. These checks
    # guard ordering and data transport that cannot be exercised without the
    # GitHub OIDC runtime.
    assert scratch_input["required"] is False
    assert scratch_input["default"] == ""
    assert '[[ -L "$SCRATCH_FILE" || ! -f "$SCRATCH_FILE" ]]' in launch_script
    assert 'scratch_path=$(realpath -- "$SCRATCH_FILE")' in launch_script
    assert '"$workspace"/*)' in launch_script
    assert "scratch_bytes > 26214400" in launch_script
    assert 'base64 --wrap=0 -- "$scratch_path"' in launch_script
    assert ".session.scratch" in launch_script
    assert '--rawfile scratch_content "$scratch_content_file"' in launch_script
    assert '--data-binary @"$request_file"' in launch_script
    assert '-d "$request"' not in launch_script


def _codehealth_refinement_workflow() -> tuple[dict, str, dict[str, dict]]:
    workflow_path = ROOT.parent.parent / ".github/workflows/ops-codehealth-refinement.yaml"
    workflow_text = workflow_path.read_text()
    workflow = yaml.safe_load(workflow_text)
    steps = {step["name"]: step for step in workflow["jobs"]["refine"]["steps"]}
    return workflow, workflow_text, steps


def test_codehealth_refinement_workflow_collects_one_complete_corpus() -> None:
    workflow, workflow_text, steps = _codehealth_refinement_workflow()
    trigger = workflow.get("on", workflow.get(True))

    assert set(trigger) == {"schedule", "workflow_dispatch"}
    assert workflow["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
        "id-token": "write",
    }
    assert "OPENAI_API_KEY" not in workflow_text
    assert "NIGHTSHIFT" not in workflow_text

    checkout = steps["Checkout repository"]
    assert checkout["with"]["persist-credentials"] is False
    setup = steps["Set up code-health environment"]
    assert setup["uses"] == "./.github/actions/codehealth-setup"
    assert setup["with"] == {"gcp-credentials-json": "${{ secrets.IRIS_CI_GCP_SA_KEY }}"}
    setup_action = yaml.safe_load((ROOT.parent.parent / ".github/actions/codehealth-setup/action.yaml").read_text())
    assert set(setup_action["inputs"]) == {"gcp-credentials-json"}
    export_step = steps["Export frozen review corpus"]
    assert export_step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    export = export_step["run"]
    assert "infra.codehealth.review_corpus export" in export
    assert "--days 30" in export
    assert "--github-concurrency" not in export
    validate = steps["Validate corpus"]["run"]
    assert "infra.codehealth.review_corpus validate refinement-corpus" in validate


def test_codehealth_refinement_workflow_removes_credentials_before_handoff() -> None:
    workflow, workflow_text, steps = _codehealth_refinement_workflow()
    step_names = [step["name"] for step in workflow["jobs"]["refine"]["steps"]]
    remove_credentials = steps["Remove export credentials"]["run"]
    assert '"$GITHUB_WORKSPACE"/*)' in remove_credentials
    assert 'rm -f -- "$CREDENTIAL_PATH"' in remove_credentials
    archive = steps["Archive corpus"]
    assert "env" not in archive
    assert "tar --create --gzip --file refinement-corpus.tar.gz refinement-corpus" in archive["run"]
    assert step_names.index("Remove export credentials") < step_names.index("Archive corpus")
    assert step_names.index("Archive corpus") < step_names.index("Launch refinement analysis")

    upload = steps["Upload frozen corpus"]
    assert upload["with"]["path"] == "refinement-corpus.tar.gz"
    launch = steps["Launch refinement analysis"]
    assert launch["uses"] == "./.github/actions/launch-loom-run"
    assert "env" not in launch
    assert launch["with"]["profile"] == "${{ vars.LOOM_CODEHEALTH_REFINEMENT_PROFILE }}"
    assert launch["with"]["channel"] == "codehealth-refinement"
    assert launch["with"]["scratch-file"] == "refinement-corpus.tar.gz"
    assert "IRIS_CI_GCP_SSH_KEY" not in workflow_text
    assert "google_compute_engine" not in workflow_text


def test_codehealth_refinement_workflow_delegates_read_only_analysis() -> None:
    _, _, steps = _codehealth_refinement_workflow()
    launch = steps["Launch refinement analysis"]
    goal = " ".join(launch["with"]["goal"].split())
    assert "Do not query live GitHub, Finelog, or any other network source" in goal
    assert "at most five subagents" in goal
    assert "three distinct pull requests" in goal
    assert "benchmark labels hidden" in goal
    assert "no production findings" in goal
    assert "not production recall" in goal
    assert "codehealth-refinement-analysis" in goal
    assert "codehealth-refinement-benchmark-predictions" in goal
    assert "compact Slack rendering" in goal
    assert "typed result" in goal


def test_codehealth_refinement_profile_is_read_only_and_credential_free() -> None:
    stack = yaml.safe_load((ROOT / "Pulumi.marin-loom.yaml").read_text())
    config = stack["config"]
    profile = config["marin-loom:profiles"]["codehealth-refinement"]
    federations = {federation["name"]: federation for federation in config["marin-loom:githubFederations"]}
    instructions = (ROOT / profile["instructionsFile"]).read_text()

    assert profile["class"] == "automation"
    assert profile["strict"] is True
    assert profile["envClear"] is True
    assert profile["ambientAllowlist"] == []
    assert profile["mode"] == "plan"
    # Loom's restricted profiles remove the Agent tool. Codex plan mode keeps
    # the required subagents while enforcing a read-only runtime.
    assert profile["restricted"] is False
    assert profile["githubRepositories"] == []
    assert profile["maxConcurrent"] == 1
    assert profile["mcpAccess"] == {"mode": "groups", "groups": ["artifact", "channel"]}
    assert "Do not query GitHub, Finelog" in instructions
    assert "same frozen archive" in instructions
    assert "tar -xOzf" in instructions
    assert "`benchmark/labels.jsonl`" in instructions
    assert "synthetic regression check" in instructions
    assert "infra/codehealth/refinement_report.py" in instructions
    assert "codehealth-refinement-analysis" in instructions
    assert "codehealth-refinement-benchmark-predictions" in instructions
    assert "only permitted writes" in instructions

    federation = federations["codehealth-refinement"]
    assert federation == {
        "name": "codehealth-refinement",
        "repositoryId": "775839592",
        "workflowRef": "marin-community/marin/.github/workflows/ops-codehealth-refinement.yaml@refs/heads/main",
        "profile": "codehealth-refinement",
        "serviceTag": "codehealth-refinement",
        "ref": "refs/heads/main",
    }


def test_release_reference_must_be_the_expected_registry_digest() -> None:
    canonical = "us-central1-docker.pkg.dev/example/loom/loom@sha256:" + "a" * 64
    tagged = "us-central1-docker.pkg.dev/example/loom/loom:latest@sha256:" + "a" * 64
    assert _validated_image_reference(canonical, "example", "us-central1") == canonical
    assert _validated_image_reference(tagged, "example", "us-central1") == tagged
    with pytest.raises(ValueError, match="expected Loom image digest"):
        _validated_image_reference("us-central1-docker.pkg.dev/example/loom/loom:main", "example", "us-central1")


def test_profile_manifest_renders_github_repositories_and_secret_references() -> None:
    profiles = _deployment_profiles(
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
    with pytest.raises(ValueError, match="exactly one of value or secretRef"):
        ProfileConfig.parse("ops", {"agent": "codex", "env": {"OPS_TOKEN": "plaintext"}})


def test_deployment_manifest_preserves_unicode_profile_instructions() -> None:
    profile = ProfileConfig.parse("github", {"agent": "codex", "instructions": "Prefix comments with 🤖"})
    config = replace(deployment_config(), profiles=(profile,), workloads=(), github_federations=())
    profiles = _deployment_profiles(config.profiles)
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
        assert "gcp:compute/resourcePolicy:ResourcePolicy" not in resource_types
        assert "gcp:compute/diskResourcePolicyAttachment:DiskResourcePolicyAttachment" not in resource_types

        vm = by_name(mocks, "loom")
        attached = field(vm.inputs, "attached_disks", "attachedDisks")
        assert not attached
        boot_disk = field(vm.inputs, "boot_disk", "bootDisk")
        assert boot_disk is not None
        assert field(boot_disk, "auto_delete", "autoDelete") is False
        assert field(boot_disk, "interface", "interface") == "NVME"
        root_disk = by_name(mocks, "loom-primary-root")
        assert root_disk.typ == "gcp:compute/disk:Disk"
        assert root_disk.inputs["name"] == "loom-hyperdisk"
        assert root_disk.inputs["type"] == "hyperdisk-balanced"
        assert field(root_disk.inputs, "provisioned_iops", "provisionedIops") == 3000
        assert field(root_disk.inputs, "provisioned_throughput", "provisionedThroughput") == 140
        assert root_disk.inputs["snapshot"] == "loom-pre-c4d-hyperdisk-20260816"
        assert boot_disk["source"] == "loom-primary-root_id"
        network_interface = field(vm.inputs, "network_interfaces", "networkInterfaces")[0]
        assert field(network_interface, "nic_type", "nicType") == "GVNIC"
        assert field(vm.inputs, "machine_type", "machineType") == "c4d-highmem-4"
        assert field(vm.inputs, "reservation_affinity", "reservationAffinity")["type"] == "NO_RESERVATION"
        scheduling = field(vm.inputs, "scheduling", "scheduling")
        assert field(scheduling, "automatic_restart", "automaticRestart") is True
        assert field(scheduling, "on_host_maintenance", "onHostMaintenance") == "MIGRATE"
        assert scheduling["preemptible"] is False
        assert field(scheduling, "provisioning_model", "provisioningModel") == "STANDARD"
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
        assert "metadataStartupScript" not in vm.inputs
        assert "metadata_startup_script" not in vm.inputs
        assert field(vm.inputs, "allow_stopping_for_update", "allowStoppingForUpdate") is False

        iam_resource_types = {
            "gcp:artifactregistry/repositoryIamMember:RepositoryIamMember",
            "gcp:kms/cryptoKeyIAMMember:CryptoKeyIAMMember",
            "gcp:projects/iAMMember:IAMMember",
            "gcp:secretmanager/secretIamMember:SecretIamMember",
        }
        assert iam_resource_types.isdisjoint(resource.typ for resource in mocks.resources)

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
