# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declare a small HTTPS storage proxy without handling credential values.

The upstream server streams one durable prefix. The existing Kubernetes secret
supplies a dedicated read-only object-store key; providers use ambient credentials.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_docker_build as docker_build
import pulumi_kubernetes as k8s
from iac.config import CLOUDFLARE_OA_DEV_ZONE_ID, load_iris_config, load_provisioning
from iris.cluster.platforms.k8s.network_manifests import DEFAULT_TRAEFIK_NAMESPACE
from rigging.filesystem.storage_path import StoragePath

NAME = "shellsim-ports"
NAMESPACE = "iris"
READONLY_SECRET = "shellsim-ports-readonly"
IMAGE_TAG = "ghcr.io/marin-community/iris-task:shellsim-ports-proxy"


@dataclass(frozen=True)
class ProxyConfig:
    cluster: str
    hostname: str
    source_revision: str
    prefix: str
    replicas: int = 2
    cpu_request: str = "100m"
    cpu_limit: str = "1"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"


def stack_config() -> ProxyConfig:
    """Read public deployment inputs without resolving secret values."""
    config = pulumi.Config()
    source_revision = config.require("source_revision")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be a full Git SHA")
    return ProxyConfig(
        cluster=config.require("cluster"),
        hostname=config.require("hostname"),
        source_revision=source_revision,
        prefix=config.require("prefix"),
        replicas=config.get_int("replicas") or 2,
        cpu_request=config.get("cpu_request") or "100m",
        cpu_limit=config.get("cpu_limit") or "1",
        memory_request=config.get("memory_request") or "128Mi",
        memory_limit=config.get("memory_limit") or "512Mi",
    )


def deploy(config: ProxyConfig) -> pulumi.Output[str]:
    """Declare image, workload, network policy, ingress, and DNS in this leaf stack."""
    prefix = StoragePath(config.prefix)
    if prefix.scheme != "s3":
        raise ValueError("ports proxy requires a CoreWeave S3 prefix")
    bucket_hostname = f"{prefix.bucket}.cwobject.com"
    cluster = load_iris_config(config.cluster)
    platform = cluster.platform.coreweave
    provisioning = load_provisioning(config.cluster).coreweave
    if platform is None or not platform.kube_context or provisioning is None:
        raise ValueError("ports proxy requires a CoreWeave cluster with an explicit Kubernetes context")
    if not os.environ.get("KUBECONFIG"):
        raise ValueError("KUBECONFIG must identify the reviewed CoreWeave kubeconfig")
    if provisioning.federation_dns is None:
        raise ValueError("ports proxy requires the existing CoreWeave ingress DNS target")
    provider = k8s.Provider("ports-k8s", kubeconfig="", context=platform.kube_context)
    options = pulumi.ResourceOptions(provider=provider)
    image = docker_build.Image(
        "ports-proxy-image",
        context=docker_build.BuildContextArgs(location=str(Path(__file__).parent)),
        build_args={"SOURCE_REVISION": config.source_revision},
        platforms=[docker_build.Platform.LINUX_AMD64],
        tags=[IMAGE_TAG],
        build_on_preview=False,
        push=True,
    )
    labels = {"app.kubernetes.io/name": NAME}
    deployment = k8s.apps.v1.Deployment(
        NAME,
        metadata={"name": NAME, "namespace": NAMESPACE},
        spec={
            "replicas": config.replicas,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "automountServiceAccountToken": False,
                    "securityContext": {"runAsNonRoot": True, "runAsUser": 65532, "runAsGroup": 65532},
                    "containers": [
                        {
                            "name": "proxy",
                            "image": image.ref,
                            "args": ["--prefix", config.prefix],
                            "ports": [{"name": "http", "containerPort": 8080}],
                            "envFrom": [{"secretRef": {"name": READONLY_SECRET}}],
                            "resources": {
                                "requests": {"cpu": config.cpu_request, "memory": config.memory_request},
                                "limits": {"cpu": config.cpu_limit, "memory": config.memory_limit},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "readinessProbe": {"tcpSocket": {"port": "http"}},
                        }
                    ],
                },
            },
        },
        opts=options,
    )
    service = k8s.core.v1.Service(
        NAME,
        metadata={"name": NAME, "namespace": NAMESPACE},
        spec={"selector": labels, "ports": [{"name": "http", "port": 8080, "targetPort": "http"}]},
        opts=pulumi.ResourceOptions(provider=provider, depends_on=[deployment]),
    )
    network_policy = k8s.apiextensions.CustomResource(
        "ports-proxy-network",
        api_version="cilium.io/v2",
        kind="CiliumNetworkPolicy",
        metadata={"name": NAME, "namespace": NAMESPACE},
        spec={
            "endpointSelector": {"matchLabels": labels},
            "ingress": [
                {
                    "fromEndpoints": [
                        {
                            "matchLabels": {
                                "k8s:io.kubernetes.pod.namespace": DEFAULT_TRAEFIK_NAMESPACE,
                                "k8s:app.kubernetes.io/name": "traefik",
                            }
                        },
                        {
                            "matchLabels": {
                                "k8s:io.kubernetes.pod.namespace": NAMESPACE,
                                "k8s:shellsim.oa.dev/ports-client": "trusted",
                            }
                        },
                    ],
                    "toPorts": [{"ports": [{"port": "8080", "protocol": "TCP"}]}],
                }
            ],
            "egress": [
                {
                    "toEndpoints": [
                        {
                            "matchLabels": {
                                "k8s:io.kubernetes.pod.namespace": "kube-system",
                                "k8s:k8s-app": "node-local-dns",
                            }
                        }
                    ],
                    "toPorts": [
                        {
                            "ports": [{"port": "53", "protocol": "ANY"}],
                            "rules": {"dns": [{"matchPattern": "*"}]},
                        }
                    ],
                },
                {
                    "toFQDNs": [{"matchName": bucket_hostname}],
                    "toPorts": [{"ports": [{"port": "443", "protocol": "TCP"}]}],
                },
            ],
        },
        opts=options,
    )
    ingress = k8s.networking.v1.Ingress(
        "ports-proxy-https",
        metadata={
            "name": NAME,
            "namespace": NAMESPACE,
            "annotations": {"cert-manager.io/cluster-issuer": provisioning.ingress.active_cluster_issuer},
        },
        spec={
            "ingressClassName": provisioning.ingress.ingress_class,
            "tls": [{"hosts": [config.hostname], "secretName": "shellsim-ports-tls"}],
            "rules": [
                {
                    "host": config.hostname,
                    "http": {
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {"service": {"name": NAME, "port": {"number": 8080}}},
                            }
                        ]
                    },
                }
            ],
        },
        opts=pulumi.ResourceOptions(provider=provider, depends_on=[service]),
    )
    dns_provider = cloudflare.Provider("ports-cloudflare")
    dns = cloudflare.DnsRecord(
        "ports-cname",
        zone_id=CLOUDFLARE_OA_DEV_ZONE_ID,
        name=config.hostname,
        type="CNAME",
        content=provisioning.federation_dns.target,
        ttl=300,
        proxied=False,
        opts=pulumi.ResourceOptions(provider=dns_provider),
    )
    return pulumi.Output.all(deployment.urn, service.urn, network_policy.urn, ingress.urn, dns.urn).apply(
        lambda _: f"https://{config.hostname}"
    )
