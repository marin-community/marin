# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarative resources and release placement for loom.oa.dev."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_command as command
import pulumi_docker_build as docker_build
import pulumi_gcp as gcp

ROOT = Path(__file__).resolve().parent
DEFAULT_REGION = "us-central1"
DEFAULT_NETWORK = "default"
DEFAULT_INSTANCE_NAME = "loom"
DEFAULT_VM_SERVICE_ACCOUNT = "loom-vm"
DEFAULT_MACHINE_TYPE = "e2-highmem-4"
DEFAULT_BOOT_DISK_GB = 100
DEFAULT_DATA_DISK_GB = 500
DEFAULT_REPO_URL = "https://github.com/marin-community/loom.git"
DEFAULT_GIT_REF = "main"
DEFAULT_DOTENV_SECRET_VERSION = 1
DEFAULT_SNAPSHOT_RETENTION_DAYS = 14
DEFAULT_BACKUP_RETENTION_DAYS = 30
ARTIFACT_REPOSITORY_ID = "loom"
ARTIFACT_IMAGE_NAME = "loom"
DOTENV_SECRET_ID = "LOOM_DOTENV"
LOOM_PORT = 7878
SECRET_ACCESSOR_ROLE = "roles/secretmanager.secretAccessor"
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
WEB_FIREWALL_TAG = "loom-web"
SSH_FIREWALL_TAG = "loom-ssh"
STARTUP_SCRIPT = (ROOT / "startup-script.sh").read_text()
RUNTIME_COMPOSE = (ROOT / "runtime/docker-compose.yml").read_text()
RUNTIME_CADDYFILE = (ROOT / "runtime/Caddyfile").read_text()
BACKUP_SCRIPT = (ROOT / "runtime/backup-sqlite.sh").read_text()


class RuntimeMode(StrEnum):
    """Whether Pulumi preserves an adopted runtime or actively reconciles it."""

    ADOPT = "adopt"
    MANAGED = "managed"


def _positive_config_int(value: int | None, default: int, name: str) -> int:
    """Default only an absent Pulumi integer; reject explicit zero/negatives."""
    resolved = default if value is None else value
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _artifact_image_path(project: str, region: str) -> str:
    return f"{region}-docker.pkg.dev/{project}/{ARTIFACT_REPOSITORY_ID}/{ARTIFACT_IMAGE_NAME}"


def _service_account_member(email: str) -> str:
    return f"serviceAccount:{email}"


def _validated_image_reference(value: str, project: str, region: str) -> str:
    prefix = f"{_artifact_image_path(project, region)}@sha256:"
    digest = value.removeprefix(prefix)
    if value == digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Artifact Registry did not resolve gitRef to the expected Loom image digest")
    return value


SECRET_REF = re.compile(
    r"^projects/(?P<project>[a-z0-9-]+)/secrets/(?P<secret>[A-Za-z0-9_-]+)/versions/(?:latest|[0-9]+)$"
)


@dataclass(frozen=True)
class WorkloadIdentityConfig:
    name: str
    profile: str
    service_tag: str
    service_account_id: str

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> WorkloadIdentityConfig:
        name = str(value.get("name", "")).strip()
        profile = str(value.get("profile", "")).strip()
        service_tag = str(value.get("serviceTag", name)).strip()
        account_id = str(value.get("serviceAccountId", f"loom-{name}")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", account_id):
            raise ValueError(f"invalid serviceAccountId for workload {name!r}")
        if not name or not profile or not service_tag:
            raise ValueError("workloads require name, profile, and serviceTag")
        if not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,62}[a-z0-9])?", name):
            raise ValueError(f"invalid workload name {name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", service_tag):
            raise ValueError(f"invalid serviceTag for workload {name!r}")
        return cls(name, profile, service_tag, account_id)


@dataclass(frozen=True)
class ProfileSecretConfig:
    name: str
    secret_ref: str
    project: str
    secret: str

    @classmethod
    def parse(cls, name: str, value: object, profile: str) -> ProfileSecretConfig:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name.startswith(("LOOM_", "WEAVER_")):
            raise ValueError(f"profile {profile!r} has invalid environment name {name!r}")
        if not isinstance(value, dict):
            raise ValueError(f"profile {profile!r} env {name!r} must use a full secretRef")
        secret_ref = str(value.get("secretRef", "")).strip()
        match = SECRET_REF.fullmatch(secret_ref)
        if not match:
            raise ValueError(f"profile {profile!r} env {name!r} must use a full secretRef")
        return cls(name, secret_ref, match.group("project"), match.group("secret"))

    def manifest(self) -> dict[str, str]:
        return {"name": self.name, "secret_ref": self.secret_ref}


def _string_tuple(value: object, field: str, profile: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"profile {profile!r} {field} must be a list of strings")
    return tuple(value)


def _optional_int(value: object, field: str, profile: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"profile {profile!r} {field} must be an integer")
    return value


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    agent: str
    description: str
    model: str
    effort: str
    protocol: str
    mode: str
    session_class: str
    strict: bool
    env_clear: bool
    ambient_allowlist: tuple[str, ...]
    idle_archive_secs: int | None
    max_concurrent: int
    turn_budget: int | None
    prelude: str
    restricted: bool
    allowed_tools: tuple[str, ...]
    env: tuple[ProfileSecretConfig, ...]

    @classmethod
    def parse(cls, name: str, value: Mapping[str, object]) -> ProfileConfig:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
            raise ValueError(f"invalid profile name {name!r}")
        agent = str(value.get("agent", "")).strip()
        if not agent:
            raise ValueError(f"profile {name!r} requires agent")
        raw_env = value.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError(f"profile {name!r} env must be an object")
        env = tuple(ProfileSecretConfig.parse(str(key), item, name) for key, item in sorted(raw_env.items()))
        return cls(
            name=name,
            agent=agent,
            description=str(value.get("description", "")),
            model=str(value.get("model", "")),
            effort=str(value.get("effort", "")),
            protocol=str(value.get("protocol", "")),
            mode=str(value.get("mode", "auto")),
            session_class=str(value.get("class", "interactive")),
            strict=bool(value.get("strict", False)),
            env_clear=bool(value.get("envClear", False)),
            ambient_allowlist=_string_tuple(value.get("ambientAllowlist", []), "ambientAllowlist", name),
            idle_archive_secs=_optional_int(value.get("idleArchiveSeconds"), "idleArchiveSeconds", name),
            max_concurrent=int(value.get("maxConcurrent", 0)),
            turn_budget=_optional_int(value.get("turnBudget"), "turnBudget", name),
            prelude=str(value.get("prelude", "weaver")),
            restricted=bool(value.get("restricted", False)),
            allowed_tools=_string_tuple(value.get("allowedTools", []), "allowedTools", name),
            env=env,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "agent_kind": self.agent,
            "model": self.model,
            "effort": self.effort,
            "protocol": self.protocol,
            "mode": self.mode,
            "class": self.session_class,
            "strict": self.strict,
            "env_clear": self.env_clear,
            "ambient_allowlist": list(self.ambient_allowlist),
            "idle_archive_secs": self.idle_archive_secs,
            "max_concurrent": self.max_concurrent,
            "turn_budget": self.turn_budget,
            "prelude": self.prelude,
            "restricted": self.restricted,
            "allowed_tools": list(self.allowed_tools),
        }


@dataclass(frozen=True)
class GitHubFederationConfig:
    name: str
    repository_id: str
    workflow_ref: str
    profile: str
    service_tag: str
    event_name: str | None
    ref_pattern: str | None

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> GitHubFederationConfig:
        required = {
            field: str(value.get(key, "")).strip()
            for field, key in (
                ("name", "name"),
                ("repository_id", "repositoryId"),
                ("workflow_ref", "workflowRef"),
                ("profile", "profile"),
            )
        }
        if not all(required.values()):
            raise ValueError("githubFederations require name, repositoryId, workflowRef, and profile")
        event = value.get("event")
        ref = value.get("ref")
        return cls(
            **required,
            service_tag=str(value.get("serviceTag", "github-actions")),
            event_name=None if event is None else str(event),
            ref_pattern=None if ref is None else str(ref),
        )

    def manifest(self, audience: str) -> dict[str, object]:
        return {
            "name": self.name,
            "provider": "github",
            "issuer": "https://token.actions.githubusercontent.com",
            "audience": audience,
            "service_tag": self.service_tag,
            "repository_id": self.repository_id,
            "workflow_ref": self.workflow_ref,
            "event_name": self.event_name,
            "ref_pattern": self.ref_pattern,
            "profiles": [self.profile],
        }


def _profile_manifest(
    profiles: tuple[ProfileConfig, ...],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Return Loom profile manifests and their ``(project, secret)`` IAM targets."""
    result: list[dict[str, Any]] = []
    secret_refs: list[tuple[str, str]] = []
    for profile in sorted(profiles, key=lambda item: item.name):
        result.append({"profile": profile.manifest(), "env": [item.manifest() for item in profile.env]})
        secret_refs.extend((item.project, item.secret) for item in profile.env)
    return result, secret_refs


def _google_federation_mapping(
    workload: WorkloadIdentityConfig,
    audience: str,
    email: str,
    subject: str,
) -> dict[str, Any]:
    """Render the exact two-claim Google identity binding Loom verifies."""
    return {
        "name": workload.name,
        "provider": "google",
        "issuer": "https://accounts.google.com",
        "audience": audience,
        "subject": str(subject),
        "service_account": email,
        "service_tag": workload.service_tag,
        "profiles": [workload.profile],
    }


@dataclass(frozen=True)
class DeploymentConfig:
    project: str
    region: str
    zone: str
    domain: str
    operator_cidr: str
    dns_zone_id: str
    build_commit: str | None = None
    runtime_mode: RuntimeMode = RuntimeMode.ADOPT
    buildx_builder: str | None = None
    network: str = DEFAULT_NETWORK
    instance_name: str = DEFAULT_INSTANCE_NAME
    vm_service_account_name: str = DEFAULT_VM_SERVICE_ACCOUNT
    machine_type: str = DEFAULT_MACHINE_TYPE
    boot_disk_gb: int = DEFAULT_BOOT_DISK_GB
    data_disk_gb: int = DEFAULT_DATA_DISK_GB
    repo_url: str = DEFAULT_REPO_URL
    git_ref: str = DEFAULT_GIT_REF
    dotenv_secret_version: int = DEFAULT_DOTENV_SECRET_VERSION
    prune_deployment: bool = False
    profiles: tuple[ProfileConfig, ...] = ()
    workloads: tuple[WorkloadIdentityConfig, ...] = ()
    github_federations: tuple[GitHubFederationConfig, ...] = ()
    snapshot_retention_days: int = DEFAULT_SNAPSHOT_RETENTION_DAYS
    backup_retention_days: int = DEFAULT_BACKUP_RETENTION_DAYS

    def __post_init__(self) -> None:
        if self.build_commit is not None and not GIT_COMMIT.fullmatch(self.build_commit):
            raise ValueError("buildCommit must be a 40-character Git commit")
        if self.runtime_mode is RuntimeMode.MANAGED and not GIT_COMMIT.fullmatch(self.git_ref):
            raise ValueError("managed runtime requires a 40-character gitRef")
        if self.runtime_mode is RuntimeMode.MANAGED and self.build_commit == self.git_ref:
            raise ValueError("buildCommit must be staged before the same gitRef is activated")
        for name, value in (
            ("bootDiskGb", self.boot_disk_gb),
            ("dataDiskGb", self.data_disk_gb),
            ("snapshotRetentionDays", self.snapshot_retention_days),
            ("backupRetentionDays", self.backup_retention_days),
            ("dotenvSecretVersion", self.dotenv_secret_version),
        ):
            _positive_config_int(value, value, name)
        profile_names = {profile.name for profile in self.profiles}
        workload_names: set[str] = set()
        for workload in self.workloads:
            if workload.name in workload_names:
                raise ValueError(f"duplicate workload name {workload.name!r}")
            if workload.profile not in profile_names:
                raise ValueError(f"workload {workload.name!r} references unknown profile {workload.profile!r}")
            workload_names.add(workload.name)
        federation_names: set[str] = set()
        for federation in self.github_federations:
            if federation.name in federation_names:
                raise ValueError(f"duplicate GitHub federation name {federation.name!r}")
            if federation.profile not in profile_names:
                raise ValueError(
                    f"GitHub federation {federation.name!r} references unknown profile {federation.profile!r}"
                )
            federation_names.add(federation.name)
        if self.prune_deployment and not (self.profiles or self.workloads or self.github_federations):
            raise ValueError("pruneDeployment requires a non-empty runtime policy")

    @classmethod
    def from_pulumi(cls) -> DeploymentConfig:
        config = pulumi.Config()
        gcp_config = pulumi.Config("gcp")
        project = gcp_config.require("project")
        region = config.get("region") or DEFAULT_REGION
        raw_profiles = config.get_object("profiles") or {}
        if not isinstance(raw_profiles, dict):
            raise ValueError("profiles must be an object")
        raw_workloads = config.get_object("workloads") or []
        if not isinstance(raw_workloads, list):
            raise ValueError("workloads must be a list")
        raw_github_federations = config.get_object("githubFederations") or []
        if not isinstance(raw_github_federations, list):
            raise ValueError("githubFederations must be a list")
        profiles = []
        for name, value in raw_profiles.items():
            if not isinstance(value, dict):
                raise ValueError(f"profile {name!r} must be an object")
            profiles.append(ProfileConfig.parse(str(name), value))
        workloads = []
        for value in raw_workloads:
            if not isinstance(value, dict):
                raise ValueError("each workload must be an object")
            workloads.append(WorkloadIdentityConfig.parse(value))
        github_federations = []
        for value in raw_github_federations:
            if not isinstance(value, dict):
                raise ValueError("each GitHub federation must be an object")
            github_federations.append(GitHubFederationConfig.parse(value))
        return cls(
            project=project,
            region=region,
            zone=config.get("zone") or f"{region}-a",
            domain=config.require("domain"),
            operator_cidr=config.require("operatorCidr"),
            dns_zone_id=config.require("dnsZoneId"),
            build_commit=config.get("buildCommit"),
            runtime_mode=RuntimeMode(config.get("runtimeMode") or RuntimeMode.ADOPT),
            buildx_builder=config.get("buildxBuilder"),
            network=config.get("network") or DEFAULT_NETWORK,
            instance_name=config.get("instanceName") or DEFAULT_INSTANCE_NAME,
            vm_service_account_name=config.get("vmServiceAccountName") or DEFAULT_VM_SERVICE_ACCOUNT,
            machine_type=config.get("machineType") or DEFAULT_MACHINE_TYPE,
            boot_disk_gb=_positive_config_int(config.get_int("bootDiskGb"), DEFAULT_BOOT_DISK_GB, "bootDiskGb"),
            data_disk_gb=_positive_config_int(config.get_int("dataDiskGb"), DEFAULT_DATA_DISK_GB, "dataDiskGb"),
            repo_url=config.get("repoUrl") or DEFAULT_REPO_URL,
            git_ref=config.get("gitRef") or DEFAULT_GIT_REF,
            dotenv_secret_version=_positive_config_int(
                config.get_int("dotenvSecretVersion"), DEFAULT_DOTENV_SECRET_VERSION, "dotenvSecretVersion"
            ),
            prune_deployment=config.get_bool("pruneDeployment") or False,
            profiles=tuple(profiles),
            workloads=tuple(workloads),
            github_federations=tuple(github_federations),
            snapshot_retention_days=_positive_config_int(
                config.get_int("snapshotRetentionDays"), DEFAULT_SNAPSHOT_RETENTION_DAYS, "snapshotRetentionDays"
            ),
            backup_retention_days=_positive_config_int(
                config.get_int("backupRetentionDays"), DEFAULT_BACKUP_RETENTION_DAYS, "backupRetentionDays"
            ),
        )


@dataclass(frozen=True)
class Infrastructure:
    instance: gcp.compute.Instance
    activation: command.local.Command | None


def _enable_apis(project: str) -> list[gcp.projects.Service]:
    services = (
        "artifactregistry.googleapis.com",
        "compute.googleapis.com",
        "iam.googleapis.com",
        "iamcredentials.googleapis.com",
        "secretmanager.googleapis.com",
        "sts.googleapis.com",
        "storage.googleapis.com",
    )
    return [
        gcp.projects.Service(
            f"api-{service.split('.')[0]}",
            project=project,
            service=service,
            disable_on_destroy=False,
        )
        for service in services
    ]


def create_infrastructure(config: DeploymentConfig) -> Infrastructure:
    """Create loom's GCP resource graph and export its operator-facing values."""
    apis = _enable_apis(config.project)
    api_options = pulumi.ResourceOptions(depends_on=apis)

    vm_account = gcp.serviceaccount.Account(
        "loom-vm",
        project=config.project,
        account_id=config.vm_service_account_name,
        display_name="loom standalone VM",
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    profile_manifest, profile_secret_refs = _profile_manifest(config.profiles)
    audience = f"https://{config.domain.rstrip('/')}"
    workload_mapping_outputs: list[pulumi.Output[dict[str, Any]]] = []
    workload_client_outputs: list[pulumi.Output[dict[str, str]]] = []
    for workload in config.workloads:
        resource_name = re.sub(r"[^a-z0-9-]", "-", workload.name.lower())
        account = gcp.serviceaccount.Account(
            f"loom-workload-{resource_name}",
            project=config.project,
            account_id=workload.service_account_id,
            display_name=f"Loom workload: {workload.name}",
            opts=api_options,
        )
        workload_mapping_outputs.append(
            pulumi.Output.all(account.email, account.unique_id).apply(
                lambda values, workload=workload: _google_federation_mapping(workload, audience, values[0], values[1])
            )
        )
        workload_client_outputs.append(
            account.email.apply(
                lambda email, workload=workload: {
                    "name": workload.name,
                    "serviceAccount": email,
                    "loomUrl": audience,
                    "tokenAudience": audience,
                    "profile": workload.profile,
                    "serviceTag": workload.service_tag,
                }
            )
        )
    github_mappings = [mapping.manifest(audience) for mapping in config.github_federations]

    def render_deployment_manifest(workload_mappings: list[dict[str, Any]]) -> str:
        return json.dumps(
            {
                "profiles": profile_manifest,
                "federations": github_mappings + workload_mappings,
                "prune": config.prune_deployment,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    deployment_manifest: pulumi.Input[str]
    if workload_mapping_outputs:
        deployment_manifest = pulumi.Output.all(*workload_mapping_outputs).apply(
            lambda workload_mappings: render_deployment_manifest(list(workload_mappings))
        )
    else:
        deployment_manifest = render_deployment_manifest([])
    artifact_repository = gcp.artifactregistry.Repository(
        "loom-images",
        project=config.project,
        location=config.region,
        repository_id=ARTIFACT_REPOSITORY_ID,
        format="DOCKER",
        docker_config={"immutable_tags": True},
        description="Immutable loom deployment images",
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    vm_image_reader = gcp.artifactregistry.RepositoryIamMember(
        "loom-vm-image-reader",
        project=config.project,
        location=artifact_repository.location,
        repository=artifact_repository.repository_id,
        role="roles/artifactregistry.reader",
        member=vm_account.email.apply(_service_account_member),
    )
    web_firewall = gcp.compute.Firewall(
        "loom-web",
        project=config.project,
        network=config.network,
        name=f"{config.instance_name}-allow-web",
        direction="INGRESS",
        source_ranges=["0.0.0.0/0"],
        target_tags=[WEB_FIREWALL_TAG],
        # Preserve provider-normalized ordering from the imported firewall so
        # equivalent policy does not produce a permanent diff.
        allows=[
            {"protocol": "tcp", "ports": ["443"]},
            {"protocol": "udp", "ports": ["443"]},
            {"protocol": "tcp", "ports": ["80"]},
        ],
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    ssh_firewall = gcp.compute.Firewall(
        "loom-ssh",
        project=config.project,
        network=config.network,
        name=f"{config.instance_name}-allow-ssh",
        direction="INGRESS",
        source_ranges=[config.operator_cidr],
        target_tags=[SSH_FIREWALL_TAG],
        allows=[{"protocol": "tcp", "ports": ["22"]}],
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    address = gcp.compute.Address(
        "loom-address",
        project=config.project,
        region=config.region,
        name=f"{config.instance_name}-ip",
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )

    dns_record = cloudflare.DnsRecord(
        "loom-dns-address",
        zone_id=config.dns_zone_id,
        name=config.domain.rstrip("."),
        type="A",
        content=address.address,
        ttl=300,
        proxied=False,
        opts=pulumi.ResourceOptions(protect=True),
    )

    data_disk = gcp.compute.Disk(
        "loom-data",
        project=config.project,
        zone=config.zone,
        name=f"{config.instance_name}-data",
        type="pd-balanced",
        size=config.data_disk_gb,
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    snapshot_policy = gcp.compute.ResourcePolicy(
        "loom-data-snapshots",
        project=config.project,
        region=config.region,
        name=f"{config.instance_name}-data-daily",
        snapshot_schedule_policy={
            "schedule": {"daily_schedule": {"days_in_cycle": 1, "start_time": "04:00"}},
            "retention_policy": {
                "max_retention_days": config.snapshot_retention_days,
                "on_source_disk_delete": "KEEP_AUTO_SNAPSHOTS",
            },
            "snapshot_properties": {"storage_locations": config.region},
        },
        opts=api_options,
    )
    snapshot_attachment = gcp.compute.DiskResourcePolicyAttachment(
        "loom-data-snapshot-policy",
        project=config.project,
        zone=config.zone,
        disk=data_disk.name,
        name=snapshot_policy.name,
    )

    dotenv_secret = gcp.secretmanager.Secret(
        "loom-dotenv",
        project=config.project,
        secret_id=DOTENV_SECRET_ID,
        replication={"auto": {}},
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    vm_secret_reader = gcp.secretmanager.SecretIamMember(
        "loom-vm-secret-reader",
        project=config.project,
        secret_id=dotenv_secret.secret_id,
        role=SECRET_ACCESSOR_ROLE,
        member=vm_account.email.apply(_service_account_member),
    )
    profile_secret_readers = []
    for secret_project, secret_name in sorted(set(profile_secret_refs)):
        suffix = hashlib.sha256(f"{secret_project}/{secret_name}".encode()).hexdigest()[:10]
        profile_secret_readers.append(
            gcp.secretmanager.SecretIamMember(
                f"loom-profile-secret-{suffix}",
                project=secret_project,
                secret_id=secret_name,
                role=SECRET_ACCESSOR_ROLE,
                member=vm_account.email.apply(_service_account_member),
                opts=api_options,
            )
        )

    backup_bucket = gcp.storage.Bucket(
        "loom-backups",
        project=config.project,
        location=config.region,
        name=pulumi.Output.format("{}-{}-backups", config.project, config.instance_name),
        uniform_bucket_level_access=True,
        public_access_prevention="enforced",
        versioning={"enabled": True},
        lifecycle_rules=[
            {
                "action": {"type": "Delete"},
                "condition": {"age": config.backup_retention_days},
            }
        ],
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    vm_backup_writer = gcp.storage.BucketIAMMember(
        "loom-vm-backup-writer",
        bucket=backup_bucket.name,
        role="roles/storage.objectCreator",
        member=vm_account.email.apply(_service_account_member),
    )

    built_image: docker_build.Image | None = None
    if config.build_commit is not None:
        image_tag = f"{_artifact_image_path(config.project, config.region)}:{config.build_commit}"
        built_image = docker_build.Image(
            "loom-release-image",
            context=docker_build.BuildContextArgs(location=f"{config.repo_url}#{config.build_commit}"),
            build_args={"CARGO_PROFILE": "release"},
            labels={
                "org.opencontainers.image.revision": config.build_commit,
                "org.opencontainers.image.source": config.repo_url,
            },
            platforms=[docker_build.Platform.LINUX_AMD64],
            tags=[image_tag],
            push=True,
            build_on_preview=False,
            builder=(
                docker_build.BuilderConfigArgs(name=config.buildx_builder) if config.buildx_builder is not None else None
            ),
            opts=pulumi.ResourceOptions(depends_on=[artifact_repository], retain_on_delete=True),
        )

    image: pulumi.Input[str] = ""
    if config.runtime_mode is RuntimeMode.MANAGED:
        released_image = gcp.artifactregistry.get_docker_image_output(
            project=config.project,
            location=config.region,
            repository_id=ARTIFACT_REPOSITORY_ID,
            image_name=f"{ARTIFACT_IMAGE_NAME}:{config.git_ref}",
            opts=pulumi.InvokeOutputOptions(depends_on=[artifact_repository]),
        )
        image = released_image.self_link.apply(
            lambda value: _validated_image_reference(value, config.project, config.region)
        )
    metadata = {
        "loom-domain": config.domain,
        "release-commit": config.git_ref,
        "loom-image": image,
        "backup-bucket": backup_bucket.name,
        "dotenv-secret-version": str(config.dotenv_secret_version),
        "dotenv-secret-id": DOTENV_SECRET_ID,
        "loom-port": str(LOOM_PORT),
        "loom-deployment": deployment_manifest,
        "loom-compose": RUNTIME_COMPOSE,
        "loom-caddyfile": RUNTIME_CADDYFILE,
        "loom-backup-script": BACKUP_SCRIPT,
        "startup-script": STARTUP_SCRIPT,
    }
    dependencies: list[pulumi.Resource] = [
        web_firewall,
        ssh_firewall,
        data_disk,
        dotenv_secret,
        vm_secret_reader,
        vm_image_reader,
        backup_bucket,
        vm_backup_writer,
        snapshot_attachment,
        *profile_secret_readers,
    ]
    dependencies.append(dns_record)
    ignored_instance_changes = (
        ["metadata"]
        if config.runtime_mode is RuntimeMode.ADOPT
        else ['metadata["ssh-keys"]', 'metadata["enable-osconfig"]']
    )
    instance_options = pulumi.ResourceOptions(
        depends_on=dependencies,
        protect=True,
        ignore_changes=ignored_instance_changes,
    )
    instance = gcp.compute.Instance(
        "loom",
        project=config.project,
        zone=config.zone,
        name=config.instance_name,
        machine_type=config.machine_type,
        tags=[WEB_FIREWALL_TAG, SSH_FIREWALL_TAG],
        boot_disk={
            "auto_delete": True,
            "initialize_params": {
                "image": "debian-cloud/debian-12",
                "size": config.boot_disk_gb,
                "type": "pd-balanced",
            },
        },
        attached_disks=[
            # GCE preserves separately attached persistent disks when an
            # instance is deleted; the disk resource is protected as well.
            {
                "source": data_disk.id,
                "device_name": "loom-data",
                "mode": "READ_WRITE",
            }
        ],
        network_interfaces=[
            {
                "network": config.network,
                "access_configs": [{"nat_ip": address.address}],
            }
        ],
        metadata=metadata,
        service_account={
            "email": vm_account.email,
            "scopes": ["cloud-platform"],
        },
        allow_stopping_for_update=False,
        deletion_protection=True,
        opts=instance_options,
    )

    activation = None
    if config.runtime_mode is RuntimeMode.MANAGED:
        activation = command.local.Command(
            "loom-activate",
            create="./activate.sh",
            update="./activate.sh",
            dir=".",
            environment={
                "LOOM_PROJECT": config.project,
                "LOOM_ZONE": config.zone,
                "LOOM_INSTANCE": config.instance_name,
                "LOOM_DOMAIN": config.domain,
            },
            triggers=[
                instance.id,
                config.git_ref,
                config.dotenv_secret_version,
                image,
                deployment_manifest,
                STARTUP_SCRIPT,
                RUNTIME_COMPOSE,
                RUNTIME_CADDYFILE,
                BACKUP_SCRIPT,
            ],
            opts=pulumi.ResourceOptions(depends_on=[instance, dns_record]),
        )

    pulumi.export("address", address.address)
    pulumi.export("url", f"https://{config.domain}/")
    pulumi.export("instanceName", instance.name)
    pulumi.export("zone", config.zone)
    pulumi.export("artifactImage", image)
    pulumi.export("builtImage", built_image.ref if built_image is not None else "")
    pulumi.export("buildCommit", config.build_commit or "")
    pulumi.export("gitRef", config.git_ref)
    pulumi.export("runtimeMode", config.runtime_mode)
    pulumi.export("dotenvSecretVersion", config.dotenv_secret_version)
    pulumi.export("backupBucket", backup_bucket.url)
    pulumi.export("tokenAudience", audience)
    pulumi.export("profileNames", sorted(profile.name for profile in config.profiles))
    pulumi.export(
        "workloadClients",
        pulumi.Output.all(*workload_client_outputs) if workload_client_outputs else [],
    )
    return Infrastructure(
        instance=instance,
        activation=activation,
    )
