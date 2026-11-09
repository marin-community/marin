# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated CoreWeave Loom staging with native shared-grant policy declarations."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

import pulumi
import pulumi_kubernetes as k8s

from iac.auth.broker import BROKER_LABELS
from iac.auth.openbao import dns_egress

LOOM_LABELS = {"app.kubernetes.io/name": "loom-cw"}


def loom_deployment_manifest(
    *,
    issuer: str,
    public_keys: list[str],
    profiles: list[dict],
    profile_pins: dict,
    exposure: Literal["bootstrap", "enrolling", "serving"],
) -> dict:
    """Render the native reconcile document without credentials or implicit models."""
    names = set()
    for entry in profiles:
        profile = entry["profile"]
        name = profile["name"]
        if name in names:
            raise ValueError("Loom deployment profile names must be unique")
        names.add(name)
        if not (
            profile["strict"] is True
            and profile["env_clear"] is True
            and profile["restricted"] is True
            and profile["class"] == "automation"
            and profile["protocol"] == "acp"
            and profile["model"]
            and profile.get("runtime_permissions", []) == []
            and profile.get("ambient_allowlist", []) == []
            and profile["mcp_access"] == {"mode": "none", "groups": []}
        ):
            raise ValueError(
                "Loom staging requires restricted ACP automation profiles with explicit models "
                "and no shell or custom MCP permissions"
            )
        if entry.get("env", []):
            raise ValueError("Profile environment is configured outside this credential-free deployment manifest")
    if exposure != "bootstrap":
        if not names or set(profile_pins) != names or not public_keys:
            raise ValueError(
                "Serving requires public signing keys and exact composition pins for every reviewed profile"
            )
        for pin in profile_pins.values():
            path = PurePosixPath(pin["cwd"])
            if not path.is_relative_to("/home/app/workspace") or ".." in path.parts:
                raise ValueError("Loom profile roots must be inside the retained /home/app/workspace directory")
            if pin["revision"] <= 0 or not pin["resolver_revision"]:
                raise ValueError("Loom serving requires reviewed profile and resolver revisions")
    shared = {
        "issuer": issuer,
        "public_keys": public_keys,
        "authorize_url": f"{issuer}/v1/authorize",
        "credentials_url": f"{issuer}/v1/loom/credentials",
        "ca_file": "/etc/loom-ca/ca.crt",
        "profiles": profile_pins,
    }
    return {
        "settings": {
            "auth.trust_loopback": False,
            "auth.shared_grants": json.dumps(shared, sort_keys=True) if exposure != "bootstrap" else "",
            "connector.enabled": False,
            "slack.status_updates": False,
        },
        "profiles": profiles,
        "remote_mcps": [],
        "federations": [],
        "watches": [],
        "oauth_clients": [],
        "prune": True,
    }


@dataclass(frozen=True)
class LoomArgs:
    namespace: str
    management_namespace: pulumi.Input[str]
    image: str
    proxy_image: str
    exposure: Literal["bootstrap", "enrolling", "serving"]
    storage_class: str
    storage_size: str
    hostname: str
    ingress_tls_secret: str
    listener_tls_secret: str
    listener_tls_server_name: str
    ingress_namespace: str
    ingress_pod_labels: dict[str, str]
    dns_cidrs: tuple[str, ...]
    https_egress_cidrs: tuple[str, ...]
    deployment: pulumi.Input[dict]
    env_secret_refs: dict[str, dict[str, str]]
    ca_secret: str


class Loom(pulumi.ComponentResource):
    """Run local restricted sessions without host mounts or Kubernetes credentials."""

    def __init__(
        self,
        name: str,
        args: LoomArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:auth:Loom", name, None, opts)

        def distinct_namespace(value: str) -> str:
            if args.namespace == value:
                raise ValueError("Loom sessions require a separate namespace from the management authority")
            return value

        management_namespace = pulumi.Output.from_input(args.management_namespace).apply(distinct_namespace)
        if any("@sha256:" not in image for image in (args.image, args.proxy_image)) or not args.ingress_pod_labels:
            raise ValueError("Loom requires a digest-pinned image and explicit ingress pod selector")
        permitted_env = {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "LOOM_GITHUB_CLIENT_ID",
            "LOOM_GITHUB_CLIENT_SECRET",
            "LOOM_GITHUB_APP_PRIVATE_KEY",
            "LOOM_GITHUB_WEBHOOK_SECRET",
        }
        if not set(args.env_secret_refs) <= permitted_env:
            raise ValueError("Loom external environment Secret references must name reviewed provider/OAuth credentials")
        namespace = k8s.core.v1.Namespace(
            f"{name}-namespace",
            metadata={"name": args.namespace, "labels": {"pod-security.kubernetes.io/enforce": "restricted"}},
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, protect=True, retain_on_delete=True),
        )
        child = pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[namespace])
        account = k8s.core.v1.ServiceAccount(
            f"{name}-account",
            metadata={"name": "loom", "namespace": args.namespace},
            automount_service_account_token=False,
            opts=child,
        )
        deployment_json = pulumi.Output.json_dumps(args.deployment, sort_keys=True)
        health_probe = {
            "exec": {"command": ["curl", "--fail", "--silent", "--max-time", "2", "http://127.0.0.1:7878/api/health"]},
            "timeout_seconds": 3,
        }
        self.config = k8s.core.v1.ConfigMap(
            f"{name}-deployment",
            metadata={"name": "loom-deployment", "namespace": args.namespace},
            data={
                "deployment.json": deployment_json,
                "Caddyfile": (
                    "{\n  admin off\n  auto_https off\n}\n:8443 {\n"
                    "  tls /etc/loom-tls/tls.crt /etc/loom-tls/tls.key\n"
                    "  reverse_proxy 127.0.0.1:7878 {\n"
                    "    header_up X-Loom-Forwarded true\n  }\n}\n"
                ),
            },
            opts=child,
        )
        self.stateful_set = k8s.apps.v1.StatefulSet(
            f"{name}-server",
            metadata={"name": "loom", "namespace": args.namespace},
            spec={
                "service_name": "loom",
                "replicas": 1,
                "update_strategy": {"type": "RollingUpdate"},
                "persistent_volume_claim_retention_policy": {"when_deleted": "Retain", "when_scaled": "Retain"},
                "selector": {"match_labels": LOOM_LABELS},
                "template": {
                    "metadata": {
                        "labels": LOOM_LABELS,
                        "annotations": {
                            "marin.dev/deployment-sha256": deployment_json.apply(
                                lambda value: hashlib.sha256(value.encode()).hexdigest()
                            )
                        },
                    },
                    "spec": {
                        "service_account_name": account.metadata["name"],
                        "automount_service_account_token": False,
                        "security_context": {
                            "run_as_non_root": True,
                            "run_as_user": 1000,
                            "run_as_group": 1000,
                            "fs_group": 1000,
                            "seccomp_profile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "loom",
                                "image": args.image,
                                # Preserve the image's tini + CLI installation entrypoint.
                                "args": ["loom", "server", "run", "--addr", "127.0.0.1:7878"],
                                "ports": [{"name": "http", "container_port": 7878}],
                                "env": [
                                    {"name": "LOOM_RUNNER", "value": "local"},
                                    {"name": "HOME", "value": "/home/app"},
                                    {"name": "WEAVER_HOME", "value": "/home/app/.weaver"},
                                    {"name": "SSL_CERT_FILE", "value": "/etc/loom-ca/ca.crt"},
                                    {"name": "REQUESTS_CA_BUNDLE", "value": "/etc/loom-ca/ca.crt"},
                                    *[
                                        {
                                            "name": env,
                                            "value_from": {"secret_key_ref": {"name": ref["name"], "key": ref["key"]}},
                                        }
                                        for env, ref in sorted(args.env_secret_refs.items())
                                    ],
                                ],
                                "security_context": {
                                    "allow_privilege_escalation": False,
                                    "read_only_root_filesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "resources": {
                                    "requests": {"cpu": "500m", "memory": "1Gi"},
                                    "limits": {"cpu": "4", "memory": "8Gi"},
                                },
                                "startup_probe": {
                                    **health_probe,
                                    "failure_threshold": 120,
                                    "period_seconds": 10,
                                },
                                "readiness_probe": health_probe,
                                "liveness_probe": health_probe,
                                "volume_mounts": [
                                    {"name": "home", "mount_path": "/home/app"},
                                    {"name": "tmp", "mount_path": "/tmp"},
                                    {"name": "deployment", "mount_path": "/etc/loom-deployment", "read_only": True},
                                    {"name": "ca", "mount_path": "/etc/loom-ca", "read_only": True},
                                ],
                            },
                            {
                                "name": "tls-proxy",
                                "image": args.proxy_image,
                                "command": ["caddy"],
                                "args": ["run", "--config", "/etc/loom-deployment/Caddyfile", "--adapter", "caddyfile"],
                                "ports": [{"name": "https", "container_port": 8443}],
                                "security_context": {
                                    "allow_privilege_escalation": False,
                                    "read_only_root_filesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "resources": {
                                    "requests": {"cpu": "50m", "memory": "64Mi"},
                                    "limits": {"cpu": "500m", "memory": "256Mi"},
                                },
                                "readiness_probe": {
                                    "http_get": {"path": "/api/health", "port": "https", "scheme": "HTTPS"}
                                },
                                "volume_mounts": [
                                    {"name": "deployment", "mount_path": "/etc/loom-deployment", "read_only": True},
                                    {"name": "tls", "mount_path": "/etc/loom-tls", "read_only": True},
                                    {"name": "proxy-data", "mount_path": "/data"},
                                    {"name": "proxy-config", "mount_path": "/config"},
                                ],
                            },
                        ],
                        "volumes": [
                            {"name": "tmp", "empty_dir": {}},
                            {"name": "proxy-data", "empty_dir": {}},
                            {"name": "proxy-config", "empty_dir": {}},
                            {"name": "deployment", "config_map": {"name": self.config.metadata["name"]}},
                            {
                                "name": "ca",
                                "secret": {
                                    "secret_name": args.ca_secret,
                                    "items": [{"key": "ca.crt", "path": "ca.crt"}],
                                },
                            },
                            {"name": "tls", "secret": {"secret_name": args.listener_tls_secret, "default_mode": 0o440}},
                        ],
                    },
                },
                "volume_claim_templates": [
                    {
                        "metadata": {"name": "home"},
                        "spec": {
                            "access_modes": ["ReadWriteOnce"],
                            "storage_class_name": args.storage_class,
                            "resources": {"requests": {"storage": args.storage_size}},
                        },
                    }
                ],
            },
            opts=pulumi.ResourceOptions(
                parent=self, provider=k8s_provider, depends_on=[namespace], protect=True, retain_on_delete=True
            ),
        )
        service = k8s.core.v1.Service(
            f"{name}-api",
            metadata={
                "name": "loom",
                "namespace": args.namespace,
                "annotations": {
                    "traefik.ingress.kubernetes.io/service.serversscheme": "https",
                    "traefik.ingress.kubernetes.io/service.serverstransport": f"{args.namespace}-loom@kubernetescrd",
                },
            },
            spec={"selector": LOOM_LABELS, "ports": [{"name": "https", "port": 8443, "target_port": "https"}]},
            opts=child,
        )
        ingress_peer = {
            "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": args.ingress_namespace}},
            "pod_selector": {"match_labels": args.ingress_pod_labels},
        }
        broker_peer = {
            "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": management_namespace}},
            "pod_selector": {"match_labels": BROKER_LABELS},
        }
        self.network_policy = k8s.networking.v1.NetworkPolicy(
            f"{name}-network",
            metadata={"name": "loom", "namespace": args.namespace},
            spec={
                "pod_selector": {"match_labels": LOOM_LABELS},
                "policy_types": ["Ingress", "Egress"],
                "ingress": (
                    [{"from": [ingress_peer, broker_peer], "ports": [{"port": 8443}]}]
                    if args.exposure == "serving"
                    else []
                ),
                "egress": [
                    {"to": [broker_peer], "ports": [{"port": 8090}]},
                    dns_egress(args.dns_cidrs),
                    *(
                        [
                            {
                                "to": [{"ip_block": {"cidr": cidr}} for cidr in args.https_egress_cidrs],
                                "ports": [{"port": 443}],
                            }
                        ]
                        if args.https_egress_cidrs
                        else []
                    ),
                ],
            },
            opts=child,
        )
        self.ingress = None
        if args.exposure == "serving":
            transport = k8s.apiextensions.CustomResource(
                f"{name}-transport",
                api_version="traefik.io/v1alpha1",
                kind="ServersTransport",
                metadata={"name": "loom", "namespace": args.namespace},
                spec={"serverName": args.listener_tls_server_name, "rootCAsSecrets": [args.listener_tls_secret]},
                opts=child,
            )
            self.ingress = k8s.networking.v1.Ingress(
                f"{name}-ingress",
                metadata={
                    "name": "loom",
                    "namespace": args.namespace,
                    "annotations": {
                        "traefik.ingress.kubernetes.io/router.entrypoints": "websecure",
                        "traefik.ingress.kubernetes.io/router.tls": "true",
                    },
                },
                spec={
                    "ingress_class_name": "traefik",
                    "tls": [{"hosts": [args.hostname], "secret_name": args.ingress_tls_secret}],
                    "rules": [
                        {
                            "host": args.hostname,
                            "http": {
                                "paths": [
                                    {
                                        "path": "/",
                                        "path_type": "Prefix",
                                        "backend": {
                                            "service": {"name": service.metadata["name"], "port": {"number": 8443}}
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                },
                opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[namespace, transport]),
            )
        self.register_outputs(
            {
                "namespace": namespace.metadata["name"],
                "deployment_config": self.config.metadata["name"],
                "exposure": args.exposure,
            }
        )
