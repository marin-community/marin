# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarative resources and release placement for loom.oa.dev."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_command as command
import pulumi_docker_build as docker_build
import pulumi_gcp as gcp

ROOT = Path(__file__).resolve().parent
DEFAULT_DISK_TYPE = "pd-balanced"
REPOSITORY_URL = "https://github.com/marin-community/loom.git"
ARTIFACT_REPOSITORY_ID = "loom"
ARTIFACT_IMAGE_NAME = "loom"
DOTENV_SECRET_ID = "LOOM_DOTENV"
LOOM_PORT = 7878
DATA_DISK_DEVICE_NAME = "loom-data"
SECRET_ACCESSOR_ROLE = "roles/secretmanager.secretAccessor"
LOG_WRITER_ROLE = "roles/logging.logWriter"
SERVICE_ACCOUNT_MEMBER = "serviceAccount:{}"
WEB_FIREWALL_TAG = "loom-web"
SSH_FIREWALL_TAG = "loom-ssh"
STARTUP_SCRIPT = (ROOT / "startup-script.sh").read_text()
RUNTIME_COMPOSE = (ROOT / "runtime/docker-compose.yml").read_text()
RUNTIME_CADDYFILE = (ROOT / "runtime/Caddyfile").read_text()


def _positive_config_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _artifact_image_path(project: str, region: str) -> str:
    return f"{region}-docker.pkg.dev/{project}/{ARTIFACT_REPOSITORY_ID}/{ARTIFACT_IMAGE_NAME}"


def _validated_image_reference(value: str, project: str, region: str) -> str:
    image_path = re.escape(_artifact_image_path(project, region))
    if not re.fullmatch(rf"{image_path}(?::[^@]+)?@sha256:[0-9a-f]{{64}}", value):
        raise ValueError("Docker did not produce the expected Loom image digest")
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
    create_service_account: bool

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> WorkloadIdentityConfig:
        name = str(value.get("name", "")).strip()
        profile = str(value.get("profile", "")).strip()
        service_tag = str(value.get("serviceTag", name)).strip()
        account_id = str(value.get("serviceAccountId", f"loom-{name}")).strip()
        create_account = value.get("createServiceAccount", True)
        if not isinstance(create_account, bool):
            raise ValueError(f"createServiceAccount for workload {name!r} must be a boolean")
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", account_id):
            raise ValueError(f"invalid serviceAccountId for workload {name!r}")
        if not name or not profile or not service_tag:
            raise ValueError("workloads require name, profile, and serviceTag")
        if not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,62}[a-z0-9])?", name):
            raise ValueError(f"invalid workload name {name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", service_tag):
            raise ValueError(f"invalid serviceTag for workload {name!r}")
        return cls(name, profile, service_tag, account_id, create_account)


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


def _deployment_profiles(
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
    build_context: str
    network: str
    instance_name: str
    vm_service_account_name: str
    machine_type: str
    boot_disk_gb: int
    data_disk_gb: int
    dotenv_secret_version: int
    snapshot_retention_days: int
    prune_deployment: bool = False
    profiles: tuple[ProfileConfig, ...] = ()
    workloads: tuple[WorkloadIdentityConfig, ...] = ()
    github_federations: tuple[GitHubFederationConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.domain != self.domain.strip().rstrip(".") or "://" in self.domain or "/" in self.domain:
            raise ValueError("domain must be a canonical hostname without a scheme, path, or trailing dot")
        for name, value in (
            ("bootDiskGb", self.boot_disk_gb),
            ("dataDiskGb", self.data_disk_gb),
            ("snapshotRetentionDays", self.snapshot_retention_days),
            ("dotenvSecretVersion", self.dotenv_secret_version),
        ):
            _positive_config_int(value, name)
        profile_names = {profile.name for profile in self.profiles}
        workload_names: set[str] = set()
        for workload in self.workloads:
            _validate_profile_reference("workload", workload.name, workload.profile, workload_names, profile_names)
        federation_names: set[str] = set()
        for federation in self.github_federations:
            _validate_profile_reference(
                "GitHub federation", federation.name, federation.profile, federation_names, profile_names
            )
        if self.prune_deployment and not (self.profiles or self.workloads or self.github_federations):
            raise ValueError("pruneDeployment requires a non-empty runtime policy")

    @property
    def public_url(self) -> str:
        return f"https://{self.domain}"

    @classmethod
    def from_pulumi(cls) -> DeploymentConfig:
        config = pulumi.Config()
        gcp_config = pulumi.Config("gcp")
        project = gcp_config.require("project")
        source_path = config.get("buildContext")
        source = REPOSITORY_URL
        if source_path is not None:
            local_source = Path(source_path).expanduser().resolve()
            if not (local_source / "Dockerfile").is_file():
                raise ValueError(f"buildContext does not contain a Dockerfile: {local_source}")
            source = str(local_source)
        region = config.require("region")
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
            zone=config.require("zone"),
            domain=config.require("domain"),
            operator_cidr=config.require("operatorCidr"),
            dns_zone_id=config.require("dnsZoneId"),
            build_context=source,
            network=config.require("network"),
            instance_name=config.require("instanceName"),
            vm_service_account_name=config.require("vmServiceAccountName"),
            machine_type=config.require("machineType"),
            boot_disk_gb=config.require_int("bootDiskGb"),
            data_disk_gb=config.require_int("dataDiskGb"),
            dotenv_secret_version=config.require_int("dotenvSecretVersion"),
            prune_deployment=config.get_bool("pruneDeployment") or False,
            profiles=tuple(profiles),
            workloads=tuple(workloads),
            github_federations=tuple(github_federations),
            snapshot_retention_days=config.require_int("snapshotRetentionDays"),
        )


@dataclass(frozen=True)
class Infrastructure:
    instance: gcp.compute.Instance
    activation: command.local.Command


def _validate_profile_reference(
    kind: str,
    name: str,
    profile: str,
    seen_names: set[str],
    profile_names: set[str],
) -> None:
    if name in seen_names:
        raise ValueError(f"duplicate {kind} name {name!r}")
    if profile not in profile_names:
        raise ValueError(f"{kind} {name!r} references unknown profile {profile!r}")
    seen_names.add(name)


def _enable_apis(project: str) -> list[gcp.projects.Service]:
    services = (
        "artifactregistry.googleapis.com",
        "compute.googleapis.com",
        "iam.googleapis.com",
        "iamcredentials.googleapis.com",
        "secretmanager.googleapis.com",
        "sts.googleapis.com",
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


@dataclass(frozen=True)
class NetworkResources:
    web_firewall: gcp.compute.Firewall
    ssh_firewall: gcp.compute.Firewall
    address: gcp.compute.Address
    dns_record: cloudflare.DnsRecord


def _create_network(config: DeploymentConfig, apis: list[gcp.projects.Service]) -> NetworkResources:
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
        name=config.domain,
        type="A",
        content=address.address,
        ttl=300,
        proxied=False,
        opts=pulumi.ResourceOptions(protect=True),
    )
    return NetworkResources(web_firewall, ssh_firewall, address, dns_record)


@dataclass(frozen=True)
class DataResources:
    disk: gcp.compute.Disk
    snapshot_attachment: gcp.compute.DiskResourcePolicyAttachment


def _create_data_disk(config: DeploymentConfig, apis: list[gcp.projects.Service]) -> DataResources:
    data_disk = gcp.compute.Disk(
        "loom-data",
        project=config.project,
        zone=config.zone,
        name=f"{config.instance_name}-data",
        type=DEFAULT_DISK_TYPE,
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
        opts=pulumi.ResourceOptions(depends_on=apis),
    )
    snapshot_attachment = gcp.compute.DiskResourcePolicyAttachment(
        "loom-data-snapshot-policy",
        project=config.project,
        zone=config.zone,
        disk=data_disk.name,
        name=snapshot_policy.name,
    )
    return DataResources(data_disk, snapshot_attachment)


@dataclass(frozen=True)
class ImageResources:
    image: docker_build.Image
    reference: pulumi.Output[str]
    vm_reader: gcp.artifactregistry.RepositoryIamMember


def _create_image(
    config: DeploymentConfig,
    apis: list[gcp.projects.Service],
    vm_account: gcp.serviceaccount.Account,
) -> ImageResources:
    repository = gcp.artifactregistry.Repository(
        "loom-images",
        project=config.project,
        location=config.region,
        repository_id=ARTIFACT_REPOSITORY_ID,
        format="DOCKER",
        description="Loom deployment images",
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    vm_reader = gcp.artifactregistry.RepositoryIamMember(
        "loom-vm-image-reader",
        project=config.project,
        location=repository.location,
        repository=repository.repository_id,
        role="roles/artifactregistry.reader",
        member=pulumi.Output.format(SERVICE_ACCOUNT_MEMBER, vm_account.email),
    )
    image_tag = f"{_artifact_image_path(config.project, config.region)}:latest"
    image = docker_build.Image(
        "loom-release-image",
        context=docker_build.BuildContextArgs(location=config.build_context),
        build_args={"CARGO_PROFILE": "release"},
        labels={"org.opencontainers.image.source": REPOSITORY_URL},
        platforms=[docker_build.Platform.LINUX_AMD64],
        tags=[image_tag],
        build_on_preview=True,
        push=True,
        opts=pulumi.ResourceOptions(depends_on=[repository]),
    )
    reference = image.ref.apply(lambda value: _validated_image_reference(value, config.project, config.region))
    return ImageResources(image, reference, vm_reader)


@dataclass(frozen=True)
class RuntimePolicyResources:
    audience: str
    manifest: pulumi.Input[str]
    workload_clients: list[pulumi.Output[dict[str, str]]]
    profile_secret_refs: list[tuple[str, str]]


def _workload_service_account(
    config: DeploymentConfig,
    workload: WorkloadIdentityConfig,
    api_options: pulumi.ResourceOptions,
) -> tuple[pulumi.Output[str], pulumi.Output[str]]:
    if workload.create_service_account:
        resource_name = re.sub(r"[^a-z0-9-]", "-", workload.name.lower())
        account = gcp.serviceaccount.Account(
            f"loom-workload-{resource_name}",
            project=config.project,
            account_id=workload.service_account_id,
            display_name=f"Loom workload: {workload.name}",
            opts=api_options,
        )
        return account.email, account.unique_id

    existing = gcp.serviceaccount.get_account_output(
        account_id=workload.service_account_id,
        project=config.project,
    )
    return (
        existing.apply(lambda account: account.email),
        existing.apply(lambda account: account.unique_id),
    )


def _create_runtime_policy(
    config: DeploymentConfig,
    api_options: pulumi.ResourceOptions,
) -> RuntimePolicyResources:
    profiles, profile_secret_refs = _deployment_profiles(config.profiles)
    audience = config.public_url
    workload_mappings: list[pulumi.Output[dict[str, Any]]] = []
    workload_clients: list[pulumi.Output[dict[str, str]]] = []
    for workload in config.workloads:
        email, unique_id = _workload_service_account(config, workload, api_options)
        workload_mappings.append(
            pulumi.Output.all(email, unique_id).apply(
                lambda values, workload=workload: _google_federation_mapping(workload, audience, values[0], values[1])
            )
        )
        workload_clients.append(
            email.apply(
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

    def render(workload_values: list[dict[str, Any]]) -> str:
        return json.dumps(
            {
                "profiles": profiles,
                "federations": github_mappings + workload_values,
                "prune": config.prune_deployment,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    manifest: pulumi.Input[str]
    if workload_mappings:
        manifest = pulumi.Output.all(*workload_mappings).apply(lambda values: render(list(values)))
    else:
        manifest = render([])
    return RuntimePolicyResources(audience, manifest, workload_clients, profile_secret_refs)


@dataclass(frozen=True)
class SecretResources:
    secret: gcp.secretmanager.Secret
    vm_reader: gcp.secretmanager.SecretIamMember
    profile_readers: list[gcp.secretmanager.SecretIamMember]


def _create_secrets(
    config: DeploymentConfig,
    apis: list[gcp.projects.Service],
    api_options: pulumi.ResourceOptions,
    vm_account: gcp.serviceaccount.Account,
    profile_secret_refs: list[tuple[str, str]],
) -> SecretResources:
    secret = gcp.secretmanager.Secret(
        "loom-dotenv",
        project=config.project,
        secret_id=DOTENV_SECRET_ID,
        replication={"auto": {}},
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    vm_reader = gcp.secretmanager.SecretIamMember(
        "loom-vm-secret-reader",
        project=config.project,
        secret_id=secret.secret_id,
        role=SECRET_ACCESSOR_ROLE,
        member=pulumi.Output.format(SERVICE_ACCOUNT_MEMBER, vm_account.email),
    )
    profile_readers = []
    for secret_project, secret_name in sorted(set(profile_secret_refs)):
        suffix = hashlib.sha256(f"{secret_project}/{secret_name}".encode()).hexdigest()[:10]
        profile_readers.append(
            gcp.secretmanager.SecretIamMember(
                f"loom-profile-secret-{suffix}",
                project=secret_project,
                secret_id=secret_name,
                role=SECRET_ACCESSOR_ROLE,
                member=pulumi.Output.format(SERVICE_ACCOUNT_MEMBER, vm_account.email),
                opts=api_options,
            )
        )
    return SecretResources(secret, vm_reader, profile_readers)


@dataclass(frozen=True)
class InstanceResources:
    instance: gcp.compute.Instance
    metadata: dict[str, pulumi.Input[str]]


def _create_instance(
    config: DeploymentConfig,
    vm_account: gcp.serviceaccount.Account,
    vm_log_writer: gcp.projects.IAMMember,
    network: NetworkResources,
    data: DataResources,
    image: ImageResources,
    secrets: SecretResources,
    runtime_policy: RuntimePolicyResources,
) -> InstanceResources:
    metadata: dict[str, pulumi.Input[str]] = {
        "loom-domain": config.domain,
        "loom-image": image.reference,
        "dotenv-secret-version": str(config.dotenv_secret_version),
        "dotenv-secret-id": DOTENV_SECRET_ID,
        "loom-port": str(LOOM_PORT),
        "data-disk-device": DATA_DISK_DEVICE_NAME,
        "loom-deployment": runtime_policy.manifest,
        "loom-compose": RUNTIME_COMPOSE,
        "loom-caddyfile": RUNTIME_CADDYFILE,
        "startup-script": STARTUP_SCRIPT,
    }
    dependencies: list[pulumi.Resource] = [
        network.web_firewall,
        network.ssh_firewall,
        network.dns_record,
        data.disk,
        data.snapshot_attachment,
        secrets.secret,
        secrets.vm_reader,
        image.vm_reader,
        vm_log_writer,
        *secrets.profile_readers,
    ]
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
                "type": DEFAULT_DISK_TYPE,
            },
        },
        attached_disks=[
            # GCE preserves separately attached persistent disks when an
            # instance is deleted; the disk resource is protected as well.
            {
                "source": data.disk.id,
                "device_name": DATA_DISK_DEVICE_NAME,
                "mode": "READ_WRITE",
            }
        ],
        network_interfaces=[
            {
                "network": config.network,
                "access_configs": [{"nat_ip": network.address.address}],
            }
        ],
        metadata=metadata,
        service_account={
            "email": vm_account.email,
            "scopes": ["cloud-platform"],
        },
        allow_stopping_for_update=False,
        deletion_protection=True,
        opts=pulumi.ResourceOptions(
            depends_on=dependencies,
            protect=True,
            ignore_changes=['metadata["ssh-keys"]', 'metadata["enable-osconfig"]'],
        ),
    )
    return InstanceResources(instance, metadata)


def _create_activation(
    config: DeploymentConfig,
    instance: InstanceResources,
    dns_record: cloudflare.DnsRecord,
) -> command.local.Command:
    return command.local.Command(
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
        triggers=[instance.instance.id, pulumi.Output.json_dumps(instance.metadata)],
        opts=pulumi.ResourceOptions(depends_on=[instance.instance, dns_record]),
    )


def _export_outputs(
    config: DeploymentConfig,
    instance: gcp.compute.Instance,
    network: NetworkResources,
    image: ImageResources,
    runtime_policy: RuntimePolicyResources,
) -> None:
    pulumi.export("address", network.address.address)
    pulumi.export("url", f"{config.public_url}/")
    pulumi.export("instanceName", instance.name)
    pulumi.export("zone", config.zone)
    pulumi.export("artifactImage", image.reference)
    pulumi.export("builtImage", image.image.ref)
    pulumi.export("dotenvSecretVersion", config.dotenv_secret_version)
    pulumi.export("tokenAudience", runtime_policy.audience)
    pulumi.export("profileNames", sorted(profile.name for profile in config.profiles))
    pulumi.export(
        "workloadClients",
        pulumi.Output.all(*runtime_policy.workload_clients) if runtime_policy.workload_clients else [],
    )


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
    vm_log_writer = gcp.projects.IAMMember(
        "loom-vm-log-writer",
        project=config.project,
        role=LOG_WRITER_ROLE,
        member=pulumi.Output.format(SERVICE_ACCOUNT_MEMBER, vm_account.email),
        opts=api_options,
    )
    runtime_policy = _create_runtime_policy(config, api_options)
    image = _create_image(config, apis, vm_account)
    network = _create_network(config, apis)
    data = _create_data_disk(config, apis)
    secrets = _create_secrets(config, apis, api_options, vm_account, runtime_policy.profile_secret_refs)
    instance = _create_instance(config, vm_account, vm_log_writer, network, data, image, secrets, runtime_policy)
    activation = _create_activation(config, instance, network.dns_record)
    _export_outputs(config, instance.instance, network, image, runtime_policy)
    return Infrastructure(instance.instance, activation)
