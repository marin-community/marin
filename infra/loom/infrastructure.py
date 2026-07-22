# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarative Google Cloud resources for the single-host Loom deployment.

Keep the resource graph in ``create_infrastructure``: tests can run it under
Pulumi mocks without credentials, while ``__main__.py`` is intentionally only
the stack entry point.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_docker_build as docker_build
import pulumi_gcp as gcp

ROOT = Path(__file__).resolve().parent


def _positive_config_int(value: int | None, default: int, name: str) -> int:
    """Default only an absent Pulumi integer; reject explicit zero/negatives."""
    resolved = default if value is None else value
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _validated_image_reference(value: str, project: str, region: str) -> str:
    prefix = f"{region}-docker.pkg.dev/{project}/loom/loom@sha256:"
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
    def parse(cls, value: dict[str, Any]) -> WorkloadIdentityConfig:
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


def _profile_manifest(
    profiles: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Translate stack-friendly camelCase profiles into Loom's REST contract."""
    result: list[dict[str, Any]] = []
    secret_refs: list[tuple[str, str]] = []
    for name, raw in sorted(profiles.items()):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
            raise ValueError(f"invalid profile name {name!r}")
        agent = str(raw.get("agent", "")).strip()
        if not agent:
            raise ValueError(f"profile {name!r} requires agent")
        profile = {
            "name": name,
            "description": str(raw.get("description", "")),
            "agent_kind": agent,
            "model": str(raw.get("model", "")),
            "effort": str(raw.get("effort", "")),
            "protocol": str(raw.get("protocol", "")),
            "mode": str(raw.get("mode", "auto")),
            "class": str(raw.get("class", "interactive")),
            "strict": bool(raw.get("strict", False)),
            "env_clear": bool(raw.get("envClear", False)),
            "ambient_allowlist": list(raw.get("ambientAllowlist", [])),
            "idle_archive_secs": raw.get("idleArchiveSeconds"),
            "max_concurrent": int(raw.get("maxConcurrent", 0)),
            "turn_budget": raw.get("turnBudget"),
            "prelude": str(raw.get("prelude", "weaver")),
            "restricted": bool(raw.get("restricted", False)),
            "allowed_tools": raw.get("allowedTools", []),
        }
        env = []
        for env_name, env_value in sorted(dict(raw.get("env", {})).items()):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) or env_name.startswith(("LOOM_", "WEAVER_")):
                raise ValueError(f"profile {name!r} has invalid environment name {env_name!r}")
            if not isinstance(env_value, dict):
                raise ValueError(f"profile {name!r} env {env_name!r} must use a full secretRef")
            secret_ref = str(env_value.get("secretRef", "")).strip()
            match = SECRET_REF.fullmatch(secret_ref)
            if not match:
                raise ValueError(f"profile {name!r} env {env_name!r} must use a full secretRef")
            env.append({"name": env_name, "secret_ref": secret_ref})
            secret_refs.append((match.group("project"), match.group("secret")))
        result.append({"profile": profile, "env": env})
    return result, secret_refs


def _github_federation_manifest(mappings: list[dict[str, Any]], audience: str) -> list[dict[str, Any]]:
    result = []
    for raw in mappings:
        name = str(raw.get("name", "")).strip()
        repository_id = str(raw.get("repositoryId", "")).strip()
        workflow_ref = str(raw.get("workflowRef", "")).strip()
        profile = str(raw.get("profile", "")).strip()
        if not all((name, repository_id, workflow_ref, profile)):
            raise ValueError("githubFederations require name, repositoryId, workflowRef, and profile")
        result.append(
            {
                "name": name,
                "provider": "github",
                "issuer": "https://token.actions.githubusercontent.com",
                "audience": audience,
                "service_tag": str(raw.get("serviceTag", "github-actions")),
                "repository_id": repository_id,
                "workflow_ref": workflow_ref,
                "event_name": raw.get("event"),
                "ref_pattern": raw.get("ref"),
                "profiles": [profile],
            }
        )
    return result


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
    runtime_rollout_enabled: bool = False
    buildx_builder: str | None = None
    network: str = "default"
    instance_name: str = "loom"
    vm_service_account_name: str = "loom-vm"
    machine_type: str = "e2-highmem-4"
    boot_disk_gb: int = 100
    data_disk_gb: int = 500
    repo_url: str = "https://github.com/marin-community/loom.git"
    git_ref: str = "main"
    dotenv_secret_version: int = 1
    prune_deployment: bool = False
    profiles: dict[str, dict[str, Any]] | None = None
    workloads: tuple[WorkloadIdentityConfig, ...] = ()
    github_federations: tuple[dict[str, Any], ...] = ()
    alert_emails: tuple[str, ...] = ()
    snapshot_retention_days: int = 14
    backup_retention_days: int = 30

    def __post_init__(self) -> None:
        if self.build_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", self.build_commit):
            raise ValueError("buildCommit must be a 40-character Git commit")
        if self.runtime_rollout_enabled and not re.fullmatch(r"[0-9a-f]{40}", self.git_ref):
            raise ValueError("runtime rollout requires a 40-character gitRef")
        if self.runtime_rollout_enabled and self.build_commit == self.git_ref:
            raise ValueError("buildCommit must be staged before the same gitRef is activated")
        for name, value in (
            ("bootDiskGb", self.boot_disk_gb),
            ("dataDiskGb", self.data_disk_gb),
            ("snapshotRetentionDays", self.snapshot_retention_days),
            ("backupRetentionDays", self.backup_retention_days),
            ("dotenvSecretVersion", self.dotenv_secret_version),
        ):
            _positive_config_int(value, value, name)
        profile_names = set((self.profiles or {}).keys())
        workload_names: set[str] = set()
        for workload in self.workloads:
            if workload.name in workload_names:
                raise ValueError(f"duplicate workload name {workload.name!r}")
            if workload.profile not in profile_names:
                raise ValueError(f"workload {workload.name!r} references unknown profile {workload.profile!r}")
            workload_names.add(workload.name)
        if self.prune_deployment and not (self.profiles or self.workloads or self.github_federations):
            raise ValueError("pruneDeployment requires a non-empty runtime policy")

    @classmethod
    def from_pulumi(cls) -> DeploymentConfig:
        config = pulumi.Config()
        gcp_config = pulumi.Config("gcp")
        project = gcp_config.require("project")
        region = config.get("region") or "us-central1"
        return cls(
            project=project,
            region=region,
            zone=config.get("zone") or f"{region}-a",
            domain=config.require("domain"),
            operator_cidr=config.require("operatorCidr"),
            dns_zone_id=config.require("dnsZoneId"),
            build_commit=config.get("buildCommit"),
            runtime_rollout_enabled=config.get_bool("runtimeRolloutEnabled") or False,
            buildx_builder=config.get("buildxBuilder"),
            network=config.get("network") or "default",
            instance_name=config.get("instanceName") or "loom",
            vm_service_account_name=config.get("vmServiceAccountName") or "loom-vm",
            machine_type=config.get("machineType") or "e2-highmem-4",
            boot_disk_gb=_positive_config_int(config.get_int("bootDiskGb"), 100, "bootDiskGb"),
            data_disk_gb=_positive_config_int(config.get_int("dataDiskGb"), 500, "dataDiskGb"),
            repo_url=config.get("repoUrl") or "https://github.com/marin-community/loom.git",
            git_ref=config.get("gitRef") or "main",
            dotenv_secret_version=_positive_config_int(config.get_int("dotenvSecretVersion"), 1, "dotenvSecretVersion"),
            prune_deployment=config.get_bool("pruneDeployment") or False,
            profiles=dict(config.get_object("profiles") or {}),
            workloads=tuple(WorkloadIdentityConfig.parse(value) for value in list(config.get_object("workloads") or [])),
            github_federations=tuple(dict(value) for value in list(config.get_object("githubFederations") or [])),
            alert_emails=tuple(config.get_object("alertEmails") or []),
            snapshot_retention_days=_positive_config_int(
                config.get_int("snapshotRetentionDays"), 14, "snapshotRetentionDays"
            ),
            backup_retention_days=_positive_config_int(config.get_int("backupRetentionDays"), 30, "backupRetentionDays"),
        )


@dataclass(frozen=True)
class Infrastructure:
    address: gcp.compute.Address
    instance: gcp.compute.Instance
    artifact_repository: gcp.artifactregistry.Repository
    backup_bucket: gcp.storage.Bucket
    workload_accounts: tuple[gcp.serviceaccount.Account, ...]
    uptime_check: gcp.monitoring.UptimeCheckConfig
    dashboard: gcp.monitoring.Dashboard


def _enable_apis(project: str) -> list[gcp.projects.Service]:
    services = (
        "artifactregistry.googleapis.com",
        "compute.googleapis.com",
        "iam.googleapis.com",
        "iamcredentials.googleapis.com",
        "logging.googleapis.com",
        "monitoring.googleapis.com",
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
    profile_manifest, profile_secret_refs = _profile_manifest(config.profiles or {})
    audience = f"https://{config.domain.rstrip('/')}"
    workload_accounts: list[gcp.serviceaccount.Account] = []
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
        workload_accounts.append(account)
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
    github_mappings = _github_federation_manifest(list(config.github_federations), audience)

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
    vm_observability_grants = [
        gcp.projects.IAMMember(
            f"loom-vm-{suffix}",
            project=config.project,
            role=role,
            member=vm_account.email.apply(lambda email: f"serviceAccount:{email}"),
            opts=api_options,
        )
        for suffix, role in (
            ("metric-writer", "roles/monitoring.metricWriter"),
            ("log-writer", "roles/logging.logWriter"),
        )
    ]

    artifact_repository = gcp.artifactregistry.Repository(
        "loom-images",
        project=config.project,
        location=config.region,
        repository_id="loom",
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
        member=vm_account.email.apply(lambda email: f"serviceAccount:{email}"),
    )
    web_firewall = gcp.compute.Firewall(
        "loom-web",
        project=config.project,
        network=config.network,
        name=f"{config.instance_name}-allow-web",
        direction="INGRESS",
        source_ranges=["0.0.0.0/0"],
        target_tags=["loom-web"],
        # Match the legacy resource's provider-normalized ordering so Phase A
        # adoption does not update an already-equivalent live firewall.
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
        target_tags=["loom-ssh"],
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
        secret_id="LOOM_DOTENV",
        replication={"auto": {}},
        opts=pulumi.ResourceOptions(depends_on=apis, protect=True),
    )
    vm_secret_reader = gcp.secretmanager.SecretIamMember(
        "loom-vm-secret-reader",
        project=config.project,
        secret_id=dotenv_secret.secret_id,
        role="roles/secretmanager.secretAccessor",
        member=vm_account.email.apply(lambda email: f"serviceAccount:{email}"),
    )
    profile_secret_readers = []
    for secret_project, secret_name in sorted(set(profile_secret_refs)):
        suffix = hashlib.sha256(f"{secret_project}/{secret_name}".encode()).hexdigest()[:10]
        profile_secret_readers.append(
            gcp.secretmanager.SecretIamMember(
                f"loom-profile-secret-{suffix}",
                project=secret_project,
                secret_id=secret_name,
                role="roles/secretmanager.secretAccessor",
                member=vm_account.email.apply(lambda email: f"serviceAccount:{email}"),
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
        member=vm_account.email.apply(lambda email: f"serviceAccount:{email}"),
    )

    built_image: docker_build.Image | None = None
    if config.build_commit is not None:
        image_tag = f"{config.region}-docker.pkg.dev/{config.project}/loom/loom:{config.build_commit}"
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
    image_mode = "disabled"
    if config.runtime_rollout_enabled:
        released_image = gcp.artifactregistry.get_docker_image_output(
            project=config.project,
            location=config.region,
            repository_id="loom",
            image_name=f"loom:{config.git_ref}",
            opts=pulumi.InvokeOutputOptions(depends_on=[artifact_repository]),
        )
        image = released_image.self_link.apply(
            lambda value: _validated_image_reference(value, config.project, config.region)
        )
        image_mode = "pull"
    metadata = {
        "loom-domain": config.domain,
        "repo-url": config.repo_url,
        "git-ref": config.git_ref,
        "image-mode": image_mode,
        "ar-image": image,
        "backup-bucket": backup_bucket.name,
        "dotenv-secret-version": str(config.dotenv_secret_version),
        "loom-deployment": deployment_manifest,
        "startup-script": (ROOT / "startup-script.sh").read_text(),
        "loom-ops-agent-config": (
            """metrics:
  receivers:
    loom:
      type: prometheus
      config:
        scrape_configs:
          - job_name: loom
            scrape_interval: 30s
            static_configs:
              - targets: [\"127.0.0.1:7878\"]
  service:
    pipelines:
      loom:
        receivers: [loom]
"""
        ),
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
        *vm_observability_grants,
    ]
    dependencies.append(dns_record)
    ignored_instance_changes = (
        ["metadata"] if not config.runtime_rollout_enabled else ['metadata["ssh-keys"]', 'metadata["enable-osconfig"]']
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
        tags=["loom-web", "loom-ssh"],
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
        allow_stopping_for_update=config.runtime_rollout_enabled,
        deletion_protection=True,
        opts=instance_options,
    )

    uptime_check = gcp.monitoring.UptimeCheckConfig(
        "loom-readiness",
        project=config.project,
        display_name="Loom readiness",
        period="60s",
        timeout="10s",
        log_check_failures=True,
        http_check={
            "path": "/api/ready",
            "port": 443,
            "request_method": "GET",
            "use_ssl": True,
            "validate_ssl": True,
        },
        monitored_resource={
            "type": "uptime_url",
            "labels": {"project_id": config.project, "host": config.domain},
        },
        opts=pulumi.ResourceOptions(depends_on=[instance]),
    )
    notification_channels = [
        gcp.monitoring.NotificationChannel(
            f"loom-alert-email-{index}",
            project=config.project,
            display_name=f"Loom operator {email}",
            type="email",
            labels={"email_address": email},
            opts=api_options,
        )
        for index, email in enumerate(config.alert_emails)
    ]
    readiness_alert = gcp.monitoring.AlertPolicy(
        "loom-readiness-failed",
        project=config.project,
        display_name="Loom is not ready",
        combiner="OR",
        notification_channels=[channel.name for channel in notification_channels],
        conditions=[
            {
                "display_name": "Public readiness probes are failing",
                "condition_threshold": {
                    "filter": (
                        'metric.type="monitoring.googleapis.com/uptime_check/check_passed" '
                        'AND resource.type="uptime_url" '
                        f'AND resource.label.host="{config.domain}"'
                    ),
                    "duration": "120s",
                    "comparison": "COMPARISON_LT",
                    "threshold_value": 1,
                    "aggregations": [
                        {
                            "alignment_period": "120s",
                            "per_series_aligner": "ALIGN_NEXT_OLDER",
                            "cross_series_reducer": "REDUCE_COUNT_TRUE",
                            "group_by_fields": ["resource.label.host"],
                        }
                    ],
                    "evaluation_missing_data": "EVALUATION_MISSING_DATA_ACTIVE",
                },
            }
        ],
        alert_strategy={"auto_close": "1800s"},
        documentation={
            "content": (
                f"Loom readiness at {audience}/api/ready has failed for two minutes. "
                "Inspect /api/diagnostics and the loom startup journal."
            ),
            "mime_type": "text/markdown",
        },
        opts=pulumi.ResourceOptions(depends_on=[uptime_check]),
    )
    dashboard = gcp.monitoring.Dashboard(
        "loom-operations",
        project=config.project,
        dashboard_json=json.dumps(
            {
                "displayName": "Loom operations",
                "mosaicLayout": {
                    "columns": 12,
                    "tiles": [
                        {
                            "xPos": 0,
                            "yPos": 0,
                            "width": 6,
                            "height": 4,
                            "widget": {
                                "title": "Current sessions by state",
                                "xyChart": {
                                    "dataSets": [
                                        {
                                            "plotType": "LINE",
                                            "targetAxis": "Y1",
                                            "timeSeriesQuery": {
                                                "prometheusQuery": "sum by (status, profile) (loom_sessions_current)"
                                            },
                                        }
                                    ],
                                    "yAxis": {"label": "sessions", "scale": "LINEAR"},
                                },
                            },
                        },
                        {
                            "xPos": 6,
                            "yPos": 0,
                            "width": 6,
                            "height": 4,
                            "widget": {
                                "title": "Automation runs by state",
                                "xyChart": {
                                    "dataSets": [
                                        {
                                            "plotType": "STACKED_BAR",
                                            "targetAxis": "Y1",
                                            "timeSeriesQuery": {
                                                "prometheusQuery": (
                                                    "sum by (status, source, service, profile) "
                                                    "(loom_automation_runs_current)"
                                                )
                                            },
                                        }
                                    ],
                                    "yAxis": {"label": "runs", "scale": "LINEAR"},
                                },
                            },
                        },
                    ],
                },
            },
            sort_keys=True,
        ),
        # The Monitoring API injects `name` and `etag` into dashboardJson. The
        # provider then proposes deleting them on every preview even when the
        # authored dashboard is unchanged.
        opts=pulumi.ResourceOptions(
            depends_on=[instance, readiness_alert],
            ignore_changes=["dashboardJson"],
        ),
    )

    pulumi.export("address", address.address)
    pulumi.export("url", f"https://{config.domain}/")
    pulumi.export("instanceName", instance.name)
    pulumi.export("zone", config.zone)
    pulumi.export("artifactImage", image)
    pulumi.export("builtImage", built_image.ref if built_image is not None else "")
    pulumi.export("buildCommit", config.build_commit or "")
    pulumi.export("gitRef", config.git_ref)
    pulumi.export("runtimeRolloutEnabled", config.runtime_rollout_enabled)
    pulumi.export("dotenvSecretVersion", config.dotenv_secret_version)
    pulumi.export("backupBucket", backup_bucket.url)
    pulumi.export("tokenAudience", audience)
    pulumi.export("profileNames", sorted((config.profiles or {}).keys()))
    pulumi.export(
        "workloadClients",
        pulumi.Output.all(*workload_client_outputs) if workload_client_outputs else [],
    )
    pulumi.export("monitoringDashboard", dashboard.id)
    return Infrastructure(
        address=address,
        instance=instance,
        artifact_repository=artifact_repository,
        backup_bucket=backup_bucket,
        workload_accounts=tuple(workload_accounts),
        uptime_check=uptime_check,
        dashboard=dashboard,
    )
