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
import pulumi_github as github
from iac.gcp.firewall import FirewallPort, GcpFirewallRuleArgs, create_firewall_rule

ROOT = Path(__file__).resolve().parent
REPOSITORY_OWNER = "marin-community"
REPOSITORY_NAME = "loom"
REPOSITORY_BRANCH = "main"
REPOSITORY_URL = f"https://github.com/{REPOSITORY_OWNER}/{REPOSITORY_NAME}.git"
ARTIFACT_REPOSITORY_ID = "loom"
ARTIFACT_IMAGE_NAME = "loom"
DOTENV_SECRET_ID = "LOOM_DOTENV"
LOOM_PORT = 7878
DOCKER_ROOT = "/var/lib/docker"
SECRET_ACCESSOR_ROLE = "roles/secretmanager.secretAccessor"
LOG_WRITER_ROLE = "roles/logging.logWriter"
KMS_ENCRYPTER_DECRYPTER_ROLE = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
RESOURCE_HASH_LENGTH = 10
SERVICE_ACCOUNT_MEMBER = "serviceAccount:{}"
WEB_FIREWALL_TAG = "loom-web"
SSH_FIREWALL_TAG = "loom-ssh"
FIREWALL_PRIORITY = 1000
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
STARTUP_SCRIPT = (ROOT / "startup-script.sh").read_text()
DOCKER_DAEMON_CONFIG = (
    json.dumps(
        {
            "data-root": DOCKER_ROOT,
            "default-ulimits": {
                "core": {
                    "Name": "core",
                    "Hard": 0,
                    "Soft": 0,
                }
            },
        },
        indent=2,
    )
    + "\n"
)
RUNTIME_COMPOSE = (ROOT / "runtime/docker-compose.yml").read_text()
RUNTIME_CADDYFILE = (ROOT / "runtime/Caddyfile").read_text()
MCP_ACCESS_NONE = "none"
MCP_ACCESS_ALL = "all"
MCP_ACCESS_GROUPS = "groups"
MCP_ACCESS_MODES = frozenset({MCP_ACCESS_NONE, MCP_ACCESS_ALL, MCP_ACCESS_GROUPS})
IAM_ROLE = re.compile(r"^(?:roles/[A-Za-z0-9_.]+|projects/[a-z][a-z0-9-]+/roles/[A-Za-z][A-Za-z0-9_.]+)$")
KMS_CRYPTO_KEY = re.compile(
    r"^projects/[a-z][a-z0-9-]+/locations/[a-z0-9-]+/" r"keyRings/[A-Za-z0-9_-]+/cryptoKeys/[A-Za-z0-9_-]+$"
)


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


def _git_context_at_revision(revision: str) -> str:
    if not GIT_COMMIT.fullmatch(revision):
        raise ValueError(f"GitHub returned an invalid Loom revision: {revision!r}")
    return f"{REPOSITORY_URL}#{revision}"


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


def _profile_instructions(value: Mapping[str, object], profile: str) -> str:
    inline = value.get("instructions")
    source = value.get("instructionsFile")
    if inline is not None and source is not None:
        raise ValueError(f"profile {profile!r} must use only one of instructions or instructionsFile")
    if source is None:
        if inline is None:
            return ""
        if not isinstance(inline, str):
            raise ValueError(f"profile {profile!r} instructions must be a string")
        return inline.strip()
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"profile {profile!r} instructionsFile must be a relative path")
    path = (ROOT / source).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"profile {profile!r} instructionsFile must name a file under {ROOT}")
    return path.read_text().strip()


def _validated_string_tuple(value: object, field: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and pattern.fullmatch(item) for item in value):
        raise ValueError(f"{field} must be a list of canonical resource names")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


@dataclass(frozen=True)
class McpAccessConfig:
    mode: str
    groups: tuple[str, ...]

    @classmethod
    def parse(cls, value: object, profile: str) -> McpAccessConfig:
        if not isinstance(value, dict):
            raise ValueError(f"profile {profile!r} mcpAccess must be an object")
        mode = str(value.get("mode", MCP_ACCESS_NONE)).strip()
        groups = _string_tuple(value.get("groups", []), "mcpAccess.groups", profile)
        if mode not in MCP_ACCESS_MODES:
            raise ValueError(f"profile {profile!r} mcpAccess.mode must be none, all, or groups")
        if mode != MCP_ACCESS_GROUPS and groups:
            raise ValueError(f"profile {profile!r} mcpAccess.groups requires mode groups")
        if mode == MCP_ACCESS_GROUPS and not groups:
            raise ValueError(f"profile {profile!r} mcpAccess mode groups requires at least one group")
        return cls(mode, groups)

    def manifest(self) -> dict[str, object]:
        return {"mode": self.mode, "groups": list(self.groups)}


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
    instructions: str
    restricted: bool
    github_repositories: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    mcp_access: McpAccessConfig
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
            instructions=_profile_instructions(value, name),
            restricted=bool(value.get("restricted", False)),
            github_repositories=_string_tuple(value.get("githubRepositories", []), "githubRepositories", name),
            allowed_tools=_string_tuple(value.get("allowedTools", []), "allowedTools", name),
            mcp_access=McpAccessConfig.parse(value.get("mcpAccess", {}), name),
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
            "instructions": self.instructions,
            "restricted": self.restricted,
            "github_repositories": list(self.github_repositories),
            "allowed_tools": list(self.allowed_tools),
            "mcp_access": self.mcp_access.manifest(),
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
    boot_disk_name: str
    boot_disk_type: str
    boot_disk_gb: int
    boot_disk_iops: int
    boot_disk_throughput: int
    boot_disk_snapshot: str
    dotenv_secret_version: int
    prune_deployment: bool = False
    settings: tuple[tuple[str, str | int | bool], ...] = ()
    profiles: tuple[ProfileConfig, ...] = ()
    workloads: tuple[WorkloadIdentityConfig, ...] = ()
    github_federations: tuple[GitHubFederationConfig, ...] = ()
    vm_project_roles: tuple[str, ...] = ()
    vm_pulumi_kms_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.domain != self.domain.strip().rstrip(".") or "://" in self.domain or "/" in self.domain:
            raise ValueError("domain must be a canonical hostname without a scheme, path, or trailing dot")
        for name, value in (
            ("bootDiskGb", self.boot_disk_gb),
            ("bootDiskIops", self.boot_disk_iops),
            ("bootDiskThroughput", self.boot_disk_throughput),
            ("dotenvSecretVersion", self.dotenv_secret_version),
        ):
            _positive_config_int(value, name)
        _validated_string_tuple(list(self.vm_project_roles), "vmProjectRoles", IAM_ROLE)
        _validated_string_tuple(list(self.vm_pulumi_kms_keys), "vmPulumiKmsKeys", KMS_CRYPTO_KEY)
        profile_names = {profile.name for profile in self.profiles}
        workload_names: set[str] = set()
        for workload in self.workloads:
            _validate_profile_reference("workload", workload.name, workload.profile, workload_names, profile_names)
        federation_names: set[str] = set()
        for federation in self.github_federations:
            _validate_profile_reference(
                "GitHub federation", federation.name, federation.profile, federation_names, profile_names
            )
        if self.prune_deployment and not (self.settings or self.profiles or self.workloads or self.github_federations):
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
        raw_settings = config.get_object("settings") or {}
        if not isinstance(raw_settings, dict):
            raise ValueError("settings must be an object")
        settings: list[tuple[str, str | int | bool]] = []
        for key, value in sorted(raw_settings.items()):
            if not isinstance(key, str) or not key.strip() or not isinstance(value, (str, int, bool)):
                raise ValueError("settings must map non-empty string keys to string, integer, or boolean values")
            settings.append((key, value))
        raw_workloads = config.get_object("workloads") or []
        if not isinstance(raw_workloads, list):
            raise ValueError("workloads must be a list")
        raw_github_federations = config.get_object("githubFederations") or []
        if not isinstance(raw_github_federations, list):
            raise ValueError("githubFederations must be a list")
        vm_project_roles = _validated_string_tuple(config.get_object("vmProjectRoles") or [], "vmProjectRoles", IAM_ROLE)
        vm_pulumi_kms_keys = _validated_string_tuple(
            config.get_object("vmPulumiKmsKeys") or [], "vmPulumiKmsKeys", KMS_CRYPTO_KEY
        )
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
            boot_disk_name=config.require("bootDiskName"),
            boot_disk_type=config.require("bootDiskType"),
            boot_disk_gb=config.require_int("bootDiskGb"),
            boot_disk_iops=config.require_int("bootDiskIops"),
            boot_disk_throughput=config.require_int("bootDiskThroughput"),
            boot_disk_snapshot=config.require("bootDiskSnapshot"),
            dotenv_secret_version=config.require_int("dotenvSecretVersion"),
            prune_deployment=config.get_bool("pruneDeployment") or False,
            settings=tuple(settings),
            profiles=tuple(profiles),
            workloads=tuple(workloads),
            github_federations=tuple(github_federations),
            vm_project_roles=vm_project_roles,
            vm_pulumi_kms_keys=vm_pulumi_kms_keys,
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
    web_firewall = create_firewall_rule(
        "loom-web",
        GcpFirewallRuleArgs(
            project=config.project,
            network=config.network,
            name=f"{config.instance_name}-allow-web",
            priority=FIREWALL_PRIORITY,
            source_ranges=("0.0.0.0/0",),
            target_tags=(WEB_FIREWALL_TAG,),
            # Preserve provider-normalized ordering from the imported firewall so
            # equivalent policy does not produce a permanent diff.
            allows=(
                FirewallPort(protocol="tcp", ports=("443",)),
                FirewallPort(protocol="udp", ports=("443",)),
                FirewallPort(protocol="tcp", ports=("80",)),
            ),
        ),
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    ssh_firewall = create_firewall_rule(
        "loom-ssh",
        GcpFirewallRuleArgs(
            project=config.project,
            network=config.network,
            name=f"{config.instance_name}-allow-ssh",
            priority=FIREWALL_PRIORITY,
            source_ranges=(config.operator_cidr,),
            target_tags=(SSH_FIREWALL_TAG,),
            allows=(FirewallPort(protocol="tcp", ports=("22",)),),
        ),
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


def _create_root_disk(config: DeploymentConfig, apis: list[gcp.projects.Service]) -> gcp.compute.Disk:
    return gcp.compute.Disk(
        "loom-primary-root",
        project=config.project,
        zone=config.zone,
        name=config.boot_disk_name,
        snapshot=config.boot_disk_snapshot,
        type=config.boot_disk_type,
        size=config.boot_disk_gb,
        provisioned_iops=config.boot_disk_iops,
        provisioned_throughput=config.boot_disk_throughput,
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True, retain_on_delete=True),
    )


@dataclass(frozen=True)
class ImageResources:
    image: docker_build.Image
    reference: pulumi.Output[str]
    vm_reader: gcp.artifactregistry.RepositoryIamMember


def _resolved_build_context(config: DeploymentConfig) -> str:
    if config.build_context != REPOSITORY_URL:
        return config.build_context

    source_provider = github.Provider("loom-source", owner=REPOSITORY_OWNER)
    branch = github.get_branch(
        repository=REPOSITORY_NAME,
        branch=REPOSITORY_BRANCH,
        opts=pulumi.InvokeOptions(provider=source_provider),
    )
    return _git_context_at_revision(branch.sha)


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
        context=docker_build.BuildContextArgs(location=_resolved_build_context(config)),
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


def _deployment_manifest(
    config: DeploymentConfig,
    profiles: list[dict[str, Any]],
    workload_values: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "settings": dict(config.settings),
            "profiles": profiles,
            "federations": (
                [mapping.manifest(config.public_url) for mapping in config.github_federations] + workload_values
            ),
            "prune": config.prune_deployment,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
    manifest: pulumi.Input[str]
    if workload_mappings:
        manifest = pulumi.Output.all(*workload_mappings).apply(
            lambda values: _deployment_manifest(config, profiles, list(values))
        )
    else:
        manifest = _deployment_manifest(config, profiles, [])
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
        suffix = hashlib.sha256(f"{secret_project}/{secret_name}".encode()).hexdigest()[:RESOURCE_HASH_LENGTH]
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
    root_disk: gcp.compute.Disk,
    image: ImageResources,
    secrets: SecretResources,
    runtime_policy: RuntimePolicyResources,
    vm_permissions: list[pulumi.Resource],
) -> InstanceResources:
    metadata: dict[str, pulumi.Input[str]] = {
        "loom-domain": config.domain,
        "loom-image": image.reference,
        "dotenv-secret-version": str(config.dotenv_secret_version),
        "dotenv-secret-id": DOTENV_SECRET_ID,
        "loom-port": str(LOOM_PORT),
        "loom-deployment": runtime_policy.manifest,
        "docker-daemon-config": DOCKER_DAEMON_CONFIG,
        "loom-compose": RUNTIME_COMPOSE,
        "loom-caddyfile": RUNTIME_CADDYFILE,
        "startup-script": STARTUP_SCRIPT,
    }
    dependencies: list[pulumi.Resource] = [
        network.web_firewall,
        network.ssh_firewall,
        network.dns_record,
        root_disk,
        secrets.secret,
        secrets.vm_reader,
        image.vm_reader,
        vm_log_writer,
        *vm_permissions,
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
            "auto_delete": False,
            "interface": "NVME",
            "source": root_disk.id,
        },
        network_interfaces=[
            {
                "network": config.network,
                "nic_type": "GVNIC",
                "access_configs": [{"nat_ip": network.address.address}],
            }
        ],
        metadata=metadata,
        service_account={
            "email": vm_account.email,
            "scopes": ["cloud-platform"],
        },
        reservation_affinity={"type": "NO_RESERVATION"},
        scheduling={
            "automatic_restart": True,
            "on_host_maintenance": "MIGRATE",
            "preemptible": False,
            "provisioning_model": "STANDARD",
        },
        allow_stopping_for_update=False,
        deletion_protection=True,
        opts=pulumi.ResourceOptions(
            depends_on=dependencies,
            delete_before_replace=True,
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


def _create_vm_permissions(
    config: DeploymentConfig,
    vm_account: gcp.serviceaccount.Account,
    api_options: pulumi.ResourceOptions,
) -> list[pulumi.Resource]:
    member = pulumi.Output.format(SERVICE_ACCOUNT_MEMBER, vm_account.email)
    grants: list[pulumi.Resource] = []
    for role in sorted(config.vm_project_roles):
        suffix = hashlib.sha256(role.encode()).hexdigest()[:RESOURCE_HASH_LENGTH]
        grants.append(
            gcp.projects.IAMMember(
                f"loom-vm-project-role-{suffix}",
                project=config.project,
                role=role,
                member=member,
                opts=api_options,
            )
        )
    for crypto_key in sorted(config.vm_pulumi_kms_keys):
        suffix = hashlib.sha256(crypto_key.encode()).hexdigest()[:RESOURCE_HASH_LENGTH]
        grants.append(
            gcp.kms.CryptoKeyIAMMember(
                f"loom-vm-pulumi-kms-{suffix}",
                crypto_key_id=crypto_key,
                role=KMS_ENCRYPTER_DECRYPTER_ROLE,
                member=member,
                opts=api_options,
            )
        )
    return grants


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
        "githubFederationProfiles",
        {federation.name: federation.profile for federation in config.github_federations},
    )
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
    vm_permissions = _create_vm_permissions(config, vm_account, api_options)
    runtime_policy = _create_runtime_policy(config, api_options)
    image = _create_image(config, apis, vm_account)
    network = _create_network(config, apis)
    root_disk = _create_root_disk(config, apis)
    secrets = _create_secrets(config, apis, api_options, vm_account, runtime_policy.profile_secret_refs)
    instance = _create_instance(
        config,
        vm_account,
        vm_log_writer,
        network,
        root_disk,
        image,
        secrets,
        runtime_policy,
        vm_permissions,
    )
    activation = _create_activation(config, instance, network.dns_record)
    _export_outputs(config, instance.instance, network, image, runtime_policy)
    return Infrastructure(instance.instance, activation)
