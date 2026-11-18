# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent OpenBao servers with a private API and public OIDC verification endpoints."""

import hashlib
import json
import re
from dataclasses import dataclass
from ipaddress import ip_network

import pulumi
import pulumi_kubernetes as k8s

OPENBAO_API_PORT = 8200
OPENBAO_CLUSTER_PORT = 8201
OIDC_PATH = "/v1/identity/oidc"
DISCOVERY_PATHS = (f"{OIDC_PATH}/.well-known/openid-configuration", f"{OIDC_PATH}/.well-known/keys")
APP_LABELS = {"app.kubernetes.io/name": "openbao"}


def dns_egress(cidrs: tuple[str, ...]) -> dict:
    """Allow approved resolver addresses and the observed node-local DNS pods.

    Cilium can enforce Service-DNAT traffic against the resolver's pod identity
    rather than the original Service address.
    """
    if not cidrs or any(ip_network(cidr).num_addresses != 1 for cidr in cidrs):
        raise ValueError("DNS egress requires explicit resolver /32 or /128 CIDRs")
    return {
        "to": [
            *({"ip_block": {"cidr": cidr}} for cidr in cidrs),
            {
                "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": "kube-system"}},
                "pod_selector": {"match_labels": {"k8s-app": "node-local-dns"}},
            },
        ],
        "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
    }


def https_egress_spec(labels: dict[str, str], dns_cidrs: tuple[str, ...], domains: tuple[str, ...]) -> dict:
    """Route DNS through Cilium so HTTPS follows only the approved names."""
    if any(
        len(domain) > 253
        or "." not in domain
        or domain.rsplit(".", 1)[-1].isdigit()
        or any(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) is None for label in domain.split("."))
        for domain in domains
    ):
        raise ValueError("HTTPS egress requires exact lowercase DNS names")
    dns_ports = [
        {
            "ports": [{"port": "53", "protocol": "UDP"}, {"port": "53", "protocol": "TCP"}],
            "rules": {"dns": [{"matchPattern": "*"}]},
        }
    ]
    return {
        "endpointSelector": {"matchLabels": labels},
        "egress": [
            {"toCIDR": list(dns_cidrs), "toPorts": dns_ports},
            {
                "toEndpoints": [
                    {
                        "matchLabels": {
                            "k8s:io.kubernetes.pod.namespace": "kube-system",
                            "k8s:k8s-app": "node-local-dns",
                        }
                    }
                ],
                "toPorts": dns_ports,
            },
            {
                "toFQDNs": [{"matchName": domain} for domain in domains],
                "toPorts": [{"ports": [{"port": "443", "protocol": "TCP"}]}],
            },
        ],
    }


@dataclass(frozen=True)
class OpenBaoArgs:
    namespace: str
    image: str
    public_hostname: str
    database_secret: str
    database_ca_secret: str
    database_host: str
    database_namespace: str
    database_cluster: str
    listener_tls_secret: str
    listener_tls_server_name: str
    ingress_tls_secret: str
    ingress_namespace: str
    ingress_pod_labels: dict[str, str]
    client_namespace: str
    client_pod_labels: dict[str, str]
    operator_groups: tuple[str, ...]
    dns_cidrs: tuple[str, ...]
    https_egress_cidrs: tuple[str, ...]
    https_egress_domains: tuple[str, ...] = ()
    replicas: int = 3
    anti_affinity: str = "required"
    discovery_mode: str = "ingress"


def openbao_config(args: OpenBaoArgs) -> dict:
    """Return listener and PostgreSQL configuration containing only public metadata."""
    return {
        "ui": False,
        "disable_mlock": True,
        "api_addr": f"https://openbao.{args.namespace}.svc:{OPENBAO_API_PORT}",
        "listener": {
            "tcp": {
                "address": f"[::]:{OPENBAO_API_PORT}",
                "cluster_address": f"[::]:{OPENBAO_CLUSTER_PORT}",
                "tls_min_version": "tls12",
                "tls_cert_file": "/openbao/tls/tls.crt",
                "tls_key_file": "/openbao/tls/tls.key",
            }
        },
        "storage": {"postgresql": {"ha_enabled": True}},
    }


class OpenBao(pulumi.ComponentResource):
    """Deploy Shamir-sealed servers backed by external PostgreSQL credentials."""

    def __init__(
        self,
        name: str,
        args: OpenBaoArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:auth:OpenBao", name, None, opts)
        if args.replicas < 1:
            raise ValueError("OpenBao requires at least one replica")
        if not all(
            (
                args.database_secret,
                args.database_ca_secret,
                args.database_host,
                args.database_namespace,
                args.database_cluster,
            )
        ):
            raise ValueError(
                "OpenBao requires external PostgreSQL credential, CA, host, namespace and cluster references"
            )
        if args.anti_affinity not in {"required", "preferred"}:
            raise ValueError("OpenBao anti_affinity must be required or preferred")
        if args.anti_affinity == "preferred" and args.namespace != "marin-auth-staging":
            raise ValueError("Shared-host OpenBao placement is restricted to synthetic staging")
        if args.discovery_mode not in {"ingress", "private"}:
            raise ValueError("OpenBao discovery mode must be ingress or private")
        if not args.operator_groups or not args.client_pod_labels or not args.ingress_pod_labels:
            raise ValueError("OpenBao requires explicit operator groups and client/ingress pod selectors")
        if "@sha256:" not in args.image:
            raise ValueError("OpenBao image must be pinned by digest")
        namespace = k8s.core.v1.Namespace(
            f"{name}-namespace",
            metadata={
                "name": args.namespace,
                "labels": {"pod-security.kubernetes.io/enforce": "restricted"},
            },
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, protect=True, retain_on_delete=True),
        )
        child = pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[namespace])
        service_account = k8s.core.v1.ServiceAccount(
            f"{name}-service-account",
            metadata={"name": "openbao", "namespace": args.namespace},
            automount_service_account_token=False,
            opts=child,
        )
        role = k8s.rbac.v1.Role(
            f"{name}-operator",
            metadata={"name": "marin-openbao-operator", "namespace": args.namespace},
            rules=[
                {"api_groups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]},
                {"api_groups": [""], "resources": ["pods/portforward"], "verbs": ["create"]},
            ],
            opts=child,
        )
        k8s.rbac.v1.RoleBinding(
            f"{name}-operators",
            metadata={"name": "marin-openbao-operators", "namespace": args.namespace},
            role_ref={"api_group": "rbac.authorization.k8s.io", "kind": "Role", "name": role.metadata["name"]},
            subjects=[
                {"api_group": "rbac.authorization.k8s.io", "kind": "Group", "name": group}
                for group in args.operator_groups
            ],
            opts=child,
        )
        config_json = json.dumps(openbao_config(args), sort_keys=True)
        config = k8s.core.v1.ConfigMap(
            f"{name}-config",
            metadata={"name": "openbao-config", "namespace": args.namespace},
            data={"config.json": config_json},
            opts=child,
        )
        k8s.core.v1.Service(
            f"{name}-internal",
            metadata={
                "name": "openbao-internal",
                "namespace": args.namespace,
                "annotations": {"pulumi.com/skipAwait": "true"},
            },
            spec={
                "cluster_ip": "None",
                "publish_not_ready_addresses": True,
                "selector": APP_LABELS,
                "ports": [{"name": "api", "port": OPENBAO_API_PORT}, {"name": "cluster", "port": OPENBAO_CLUSTER_PORT}],
            },
            opts=child,
        )
        service = k8s.core.v1.Service(
            f"{name}-api",
            metadata={
                "name": "openbao",
                "namespace": args.namespace,
                "annotations": {
                    "pulumi.com/skipAwait": "true",
                    "traefik.ingress.kubernetes.io/service.serversscheme": "https",
                    "traefik.ingress.kubernetes.io/service.serverstransport": f"{args.namespace}-openbao@kubernetescrd",
                },
            },
            spec={"selector": APP_LABELS, "ports": [{"name": "https", "port": OPENBAO_API_PORT, "target_port": "api"}]},
            opts=child,
        )
        self.stateful_set = k8s.apps.v1.StatefulSet(
            f"{name}-servers",
            metadata={
                "name": "openbao",
                "namespace": args.namespace,
                # Fresh servers remain sealed until operators initialize them outside Pulumi.
                "annotations": {"pulumi.com/skipAwait": "true"},
            },
            spec={
                "service_name": "openbao-internal",
                "replicas": args.replicas,
                "pod_management_policy": "Parallel",
                "update_strategy": {"type": "OnDelete"},
                "persistent_volume_claim_retention_policy": {"when_deleted": "Retain", "when_scaled": "Retain"},
                "selector": {"match_labels": APP_LABELS},
                "template": {
                    "metadata": {
                        "labels": APP_LABELS,
                        "annotations": {"marin.dev/config-sha256": hashlib.sha256(config_json.encode()).hexdigest()},
                    },
                    "spec": {
                        "service_account_name": service_account.metadata["name"],
                        "automount_service_account_token": False,
                        "termination_grace_period_seconds": 60,
                        "security_context": {
                            "run_as_non_root": True,
                            "run_as_user": 1000,
                            "run_as_group": 1000,
                            "fs_group": 1000,
                            "seccomp_profile": {"type": "RuntimeDefault"},
                        },
                        "affinity": {
                            "pod_anti_affinity": {
                                (
                                    "required_during_scheduling_ignored_during_execution"
                                    if args.anti_affinity == "required"
                                    else "preferred_during_scheduling_ignored_during_execution"
                                ): [
                                    (
                                        {
                                            "label_selector": {"match_labels": APP_LABELS},
                                            "topology_key": "kubernetes.io/hostname",
                                        }
                                        if args.anti_affinity == "required"
                                        else {
                                            "weight": 100,
                                            "pod_affinity_term": {
                                                "label_selector": {"match_labels": APP_LABELS},
                                                "topology_key": "kubernetes.io/hostname",
                                            },
                                        }
                                    )
                                ]
                            }
                        },
                        "containers": [
                            {
                                "name": "openbao",
                                "image": args.image,
                                "command": ["bao"],
                                "args": ["server", "-config=/openbao/config/config.json"],
                                "ports": [
                                    {"name": "api", "container_port": OPENBAO_API_PORT},
                                    {"name": "cluster", "container_port": OPENBAO_CLUSTER_PORT},
                                ],
                                "env": [
                                    {"name": "PGHOST", "value": args.database_host},
                                    {"name": "PGPORT", "value": "5432"},
                                    {"name": "PGDATABASE", "value": "openbao"},
                                    {"name": "PGSSLMODE", "value": "verify-full"},
                                    {"name": "PGSSLROOTCERT", "value": "/openbao/database-ca/ca.crt"},
                                    *(
                                        {
                                            "name": variable,
                                            "value_from": {"secret_key_ref": {"name": args.database_secret, "key": key}},
                                        }
                                        for variable, key in (("PGUSER", "username"), ("PGPASSWORD", "password"))
                                    ),
                                    {"name": "POD_IP", "value_from": {"field_ref": {"field_path": "status.podIP"}}},
                                    {"name": "BAO_CLUSTER_ADDR", "value": f"https://$(POD_IP):{OPENBAO_CLUSTER_PORT}"},
                                    {"name": "BAO_ADDR", "value": f"https://127.0.0.1:{OPENBAO_API_PORT}"},
                                    {"name": "BAO_TLS_SERVER_NAME", "value": args.listener_tls_server_name},
                                    {"name": "BAO_CACERT", "value": "/openbao/tls/ca.crt"},
                                    {"name": "HOME", "value": "/tmp"},
                                ],
                                "security_context": {
                                    "allow_privilege_escalation": False,
                                    "read_only_root_filesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "resources": {
                                    "requests": {"cpu": "250m", "memory": "512Mi"},
                                    "limits": {"cpu": "2", "memory": "2Gi"},
                                },
                                "readiness_probe": {
                                    "exec": {"command": ["bao", "status"]},
                                    "period_seconds": 5,
                                    "timeout_seconds": 3,
                                },
                                # A sealed server is alive; restarting it cannot unseal it.
                                "liveness_probe": {"tcp_socket": {"port": "api"}, "initial_delay_seconds": 30},
                                "volume_mounts": [
                                    {"name": "database-ca", "mount_path": "/openbao/database-ca", "read_only": True},
                                    {"name": "config", "mount_path": "/openbao/config", "read_only": True},
                                    {"name": "tls", "mount_path": "/openbao/tls", "read_only": True},
                                    {"name": "tmp", "mount_path": "/tmp"},
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "database-ca",
                                "secret": {
                                    "secret_name": args.database_ca_secret,
                                    "items": [{"key": "ca.crt", "path": "ca.crt"}],
                                },
                            },
                            {"name": "config", "config_map": {"name": config.metadata["name"]}},
                            {"name": "tls", "secret": {"secret_name": args.listener_tls_secret, "default_mode": 0o440}},
                            {"name": "tmp", "empty_dir": {}},
                        ],
                    },
                },
            },
            opts=pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=[namespace],
                protect=True,
                retain_on_delete=True,
                # Existing immutable Raft claim templates and retained PVCs remain available for rollback.
                ignore_changes=["spec.volumeClaimTemplates"],
            ),
        )
        k8s.policy.v1.PodDisruptionBudget(
            f"{name}-quorum",
            metadata={"name": "openbao", "namespace": args.namespace},
            spec={"max_unavailable": 1, "selector": {"match_labels": APP_LABELS}},
            opts=child,
        )
        ingress_peer = {
            "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": args.ingress_namespace}},
            "pod_selector": {"match_labels": args.ingress_pod_labels},
        }
        client_peer = {
            "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": args.client_namespace}},
            "pod_selector": {"match_labels": args.client_pod_labels},
        }
        resolver_egress = dns_egress(args.dns_cidrs)
        self.network_policy = k8s.networking.v1.NetworkPolicy(
            f"{name}-network",
            metadata={"name": "openbao", "namespace": args.namespace},
            spec={
                "pod_selector": {"match_labels": APP_LABELS},
                "policy_types": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [{"pod_selector": {"match_labels": APP_LABELS}}],
                        "ports": [{"port": OPENBAO_API_PORT}, {"port": OPENBAO_CLUSTER_PORT}],
                    },
                    {"from": [ingress_peer, client_peer], "ports": [{"port": OPENBAO_API_PORT}]},
                ],
                "egress": [
                    {
                        "to": [{"pod_selector": {"match_labels": APP_LABELS}}],
                        "ports": [{"port": OPENBAO_API_PORT}, {"port": OPENBAO_CLUSTER_PORT}],
                    },
                    {
                        "to": [
                            {
                                "namespace_selector": {
                                    "match_labels": {"kubernetes.io/metadata.name": args.database_namespace}
                                },
                                "pod_selector": {"match_labels": {"cnpg.io/cluster": args.database_cluster}},
                            }
                        ],
                        "ports": [{"port": 5432}],
                    },
                    *([] if args.https_egress_domains else [resolver_egress]),
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
        if args.https_egress_domains:
            # An overlapping L4 DNS allow would bypass Cilium's DNS proxy.
            self.https_egress_policy = k8s.apiextensions.CustomResource(
                f"{name}-https-egress",
                api_version="cilium.io/v2",
                kind="CiliumNetworkPolicy",
                metadata={"name": "openbao-https-egress", "namespace": args.namespace},
                spec=https_egress_spec(APP_LABELS, args.dns_cidrs, args.https_egress_domains),
                opts=child,
            )
        if args.discovery_mode == "private":
            self.issuer = pulumi.Output.from_input(f"https://openbao.{args.namespace}.svc:{OPENBAO_API_PORT}{OIDC_PATH}")
            self.api_address = pulumi.Output.from_input(f"https://openbao.{args.namespace}.svc:{OPENBAO_API_PORT}")
            self.register_outputs({"issuer": self.issuer, "api_address": self.api_address})
            return
        transport = k8s.apiextensions.CustomResource(
            f"{name}-transport",
            api_version="traefik.io/v1alpha1",
            kind="ServersTransport",
            metadata={"name": "openbao", "namespace": args.namespace},
            spec={"serverName": args.listener_tls_server_name, "rootCAsSecrets": [args.listener_tls_secret]},
            opts=child,
        )
        self.discovery_ingress = k8s.networking.v1.Ingress(
            f"{name}-discovery",
            metadata={
                "name": "openbao-discovery",
                "namespace": args.namespace,
                "annotations": {
                    "pulumi.com/skipAwait": "true",
                    "traefik.ingress.kubernetes.io/router.entrypoints": "websecure",
                    "traefik.ingress.kubernetes.io/router.tls": "true",
                },
            },
            spec={
                "ingress_class_name": "traefik",
                "tls": [{"hosts": [args.public_hostname], "secret_name": args.ingress_tls_secret}],
                "rules": [
                    {
                        "host": args.public_hostname,
                        "http": {
                            "paths": [
                                {
                                    "path": path,
                                    "path_type": "Exact",
                                    "backend": {
                                        "service": {
                                            "name": service.metadata["name"],
                                            "port": {"number": OPENBAO_API_PORT},
                                        }
                                    },
                                }
                                for path in DISCOVERY_PATHS
                            ]
                        },
                    }
                ],
            },
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[namespace, transport]),
        )
        self.issuer = pulumi.Output.from_input(f"https://{args.public_hostname}{OIDC_PATH}")
        self.api_address = pulumi.Output.from_input(f"https://openbao.{args.namespace}.svc:{OPENBAO_API_PORT}")
        self.register_outputs(
            {"issuer": self.issuer, "api_address": self.api_address, "namespace": namespace.metadata["name"]}
        )
