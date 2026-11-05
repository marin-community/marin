# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deployment-owned grant policy and a singleton broker with a durable runtime ledger."""

import hashlib
from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s

from iac.auth.openbao import APP_LABELS, dns_egress

BROKER_LABELS = {"app.kubernetes.io/name": "marin-auth"}
TOKEN_DIRECTORY = "/var/run/marin-auth"
DATABASE_PATH = "/var/lib/marin-auth/grants.db"


def coreweave_identity_metadata(bindings: list[dict]) -> dict:
    """Publish the complete provider scope without private credential references."""
    metadata = {}
    subjects = set()
    for binding in bindings:
        if binding["name"] in metadata or binding["subject"] in subjects:
            raise ValueError("Each storage binding name and subject must have one complete policy mapping")
        subjects.add(binding["subject"])
        metadata[binding["name"]] = {
            "subject": binding["subject"],
            "permissions": binding["permissions"],
        }
    return metadata


@dataclass(frozen=True)
class GrantBrokerArgs:
    namespace: str
    image: str
    storage_class: str
    storage_size: str
    policy: dict
    token_secret_names: tuple[str, ...]
    listener_tls_secret: str
    ca_secret: str
    client_namespace: str
    client_pod_labels: dict[str, str]
    dns_cidrs: tuple[str, ...]
    https_egress_cidrs: tuple[str, ...]


class GrantBroker(pulumi.ComponentResource):
    """Deploy the shared exchange; runtime grants never mutate this ConfigMap."""

    def __init__(
        self,
        name: str,
        args: GrantBrokerArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:auth:GrantBroker", name, None, opts)
        if "@sha256:" not in args.image:
            raise ValueError("Grant broker image must be pinned by digest")
        if not args.client_pod_labels:
            raise ValueError("Grant broker requires explicit client pod labels")
        if {"ca", "tls"} & set(args.token_secret_names):
            raise ValueError("Token Secret names ca and tls conflict with broker TLS mounts")
        if args.policy["database"] != DATABASE_PATH:
            raise ValueError("Grant broker must store its runtime ledger on the retained PVC")
        if args.policy["ca_bundle"] != f"{TOKEN_DIRECTORY}/ca/ca.crt":
            raise ValueError("Grant broker must verify service TLS using the mounted CA bundle")
        expected_paths = {f"{TOKEN_DIRECTORY}/{secret}/token" for secret in args.token_secret_names}
        configured_paths = {
            args.policy["broker_token_file"],
            *(binding["token_file"] for binding in args.policy["storage"]),
        }
        if args.policy["loom"] is not None:
            configured_paths.add(args.policy["loom"]["observer_token_file"])
        if configured_paths != expected_paths:
            raise ValueError("Every broker/storage token file must correspond to an explicit external Secret mount")
        if args.policy["broker_token_file"] in {binding["token_file"] for binding in args.policy["storage"]}:
            raise ValueError("Broker signing and storage identities require separate token mounts")
        child = pulumi.ResourceOptions(parent=self, provider=k8s_provider)
        policy_json = pulumi.Output.json_dumps(args.policy, sort_keys=True)
        config = k8s.core.v1.ConfigMap(
            f"{name}-policy",
            metadata={"name": "marin-auth-policy", "namespace": args.namespace},
            data={"policy.yaml": policy_json},
            opts=child,
        )
        account = k8s.core.v1.ServiceAccount(
            f"{name}-account",
            metadata={"name": "marin-auth", "namespace": args.namespace},
            automount_service_account_token=False,
            opts=child,
        )
        token_mounts = [
            {"name": f"token-{index}", "mount_path": f"{TOKEN_DIRECTORY}/{secret}", "read_only": True}
            for index, secret in enumerate(args.token_secret_names)
        ]
        token_volumes = [
            {
                "name": f"token-{index}",
                "secret": {"secret_name": secret, "default_mode": 0o440, "items": [{"key": "token", "path": "token"}]},
            }
            for index, secret in enumerate(args.token_secret_names)
        ]
        self.stateful_set = k8s.apps.v1.StatefulSet(
            f"{name}-server",
            metadata={"name": "marin-auth", "namespace": args.namespace},
            spec={
                "service_name": "marin-auth",
                "replicas": 1,
                "update_strategy": {"type": "RollingUpdate"},
                "persistent_volume_claim_retention_policy": {"when_deleted": "Retain", "when_scaled": "Retain"},
                "selector": {"match_labels": BROKER_LABELS},
                "template": {
                    "metadata": {
                        "labels": BROKER_LABELS,
                        "annotations": {
                            "marin.dev/policy-sha256": policy_json.apply(
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
                                "name": "broker",
                                "image": args.image,
                                "command": ["python", "-m", "infra.auth.service"],
                                "working_dir": "/app",
                                "args": [
                                    "--config",
                                    "/etc/marin-auth/policy.yaml",
                                    "--host",
                                    "0.0.0.0",
                                    "--port",
                                    "8090",
                                    "--ssl-certfile",
                                    f"{TOKEN_DIRECTORY}/tls/tls.crt",
                                    "--ssl-keyfile",
                                    f"{TOKEN_DIRECTORY}/tls/tls.key",
                                ],
                                "ports": [{"name": "https", "container_port": 8090}],
                                "security_context": {
                                    "allow_privilege_escalation": False,
                                    "read_only_root_filesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"cpu": "1", "memory": "512Mi"},
                                },
                                "readiness_probe": {
                                    "http_get": {"path": "/healthz", "port": "https", "scheme": "HTTPS"}
                                },
                                "liveness_probe": {"tcp_socket": {"port": "https"}, "initial_delay_seconds": 30},
                                "volume_mounts": [
                                    {"name": "data", "mount_path": "/var/lib/marin-auth"},
                                    {"name": "policy", "mount_path": "/etc/marin-auth", "read_only": True},
                                    {"name": "ca", "mount_path": f"{TOKEN_DIRECTORY}/ca", "read_only": True},
                                    {"name": "tls", "mount_path": f"{TOKEN_DIRECTORY}/tls", "read_only": True},
                                    {"name": "tmp", "mount_path": "/tmp"},
                                    *token_mounts,
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "policy", "config_map": {"name": config.metadata["name"]}},
                            {
                                "name": "ca",
                                "secret": {
                                    "secret_name": args.ca_secret,
                                    "items": [{"key": "ca.crt", "path": "ca.crt"}],
                                },
                            },
                            {"name": "tls", "secret": {"secret_name": args.listener_tls_secret, "default_mode": 0o440}},
                            {"name": "tmp", "empty_dir": {}},
                            *token_volumes,
                        ],
                    },
                },
                "volume_claim_templates": [
                    {
                        "metadata": {"name": "data"},
                        "spec": {
                            "access_modes": ["ReadWriteOnce"],
                            "storage_class_name": args.storage_class,
                            "resources": {"requests": {"storage": args.storage_size}},
                        },
                    }
                ],
            },
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, protect=True, retain_on_delete=True),
        )
        service = k8s.core.v1.Service(
            f"{name}-api",
            metadata={"name": "marin-auth", "namespace": args.namespace},
            spec={"selector": BROKER_LABELS, "ports": [{"name": "https", "port": 8090, "target_port": "https"}]},
            opts=child,
        )
        client_peer = {
            "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": args.client_namespace}},
            "pod_selector": {"match_labels": args.client_pod_labels},
        }
        self.network_policy = k8s.networking.v1.NetworkPolicy(
            f"{name}-network",
            metadata={"name": "marin-auth", "namespace": args.namespace},
            spec={
                "pod_selector": {"match_labels": BROKER_LABELS},
                "policy_types": ["Ingress", "Egress"],
                "ingress": [{"from": [client_peer], "ports": [{"port": 8090}]}],
                "egress": [
                    {"to": [{"pod_selector": {"match_labels": APP_LABELS}}], "ports": [{"port": 8200}]},
                    *([{"to": [client_peer], "ports": [{"port": 8443}]}] if args.policy["loom"] is not None else []),
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
        self.address = service.metadata.apply(lambda metadata: f"https://{metadata['name']}.{args.namespace}.svc:8090")
        self.register_outputs({"address": self.address, "policy": config.metadata["name"]})
