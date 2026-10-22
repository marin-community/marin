# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the CoreWeave OpenBao pilot without reading credentials or contacting a cluster."""

import json
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

NAME = "openbao"
AUTHORITY_LABEL = "marin.community/authority"
API_PORT = 8200
RAFT_PORT = 8201


class Exposure(StrEnum):
    INTERNAL = "internal"
    TLS_PASSTHROUGH = "tls-passthrough"


class OpenBaoDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster: str
    namespace: str
    hostname: str
    image: str
    storage_class: str
    authority_pool: str
    exposure: Exposure
    replicas: int = 1
    storage_gb: int = 20
    controller_namespace: str = "iris"
    ingress_namespace: str = "traefik"

    @field_validator("image")
    @classmethod
    def pinned_image(cls, value: str) -> str:
        if not re.fullmatch(r"ghcr.io/openbao/openbao@sha256:[0-9a-f]{64}", value):
            raise ValueError("OpenBao requires an immutable official image digest")
        return value

    @field_validator("replicas")
    @classmethod
    def raft_size(cls, value: int) -> int:
        if value not in {1, 3, 5}:
            raise ValueError("Use one bootstrap server, three pilot voters or five main voters")
        return value

    @field_validator("namespace", "hostname", "authority_pool", "storage_class")
    @classmethod
    def dns_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value):
            raise ValueError("Expected a non-empty DNS name")
        return value


def openbao_manifests(config: OpenBaoDeployment) -> list[dict]:
    """Build a sealed Raft service; initialization and joining remain operator actions."""
    namespace = config.namespace
    labels = {"app.kubernetes.io/name": NAME}
    peer_name = f"{NAME}-internal"
    peer_dns = f"{peer_name}.{namespace}.svc.cluster.local"
    api_dns = f"{NAME}.{namespace}.svc.cluster.local"
    quorum = config.replicas // 2 + 1

    def resource(api_version: str, kind: str, name: str, **body: object) -> dict:
        return {
            "apiVersion": api_version,
            "kind": kind,
            "metadata": {"name": name, "namespace": namespace, "labels": labels},
            **body,
        }

    server_config = {
        "ui": False,
        # Raft uses mmap. The host must disable swap; IPC_LOCK is unnecessary.
        "disable_mlock": True,
        "disable_standby_reads": True,
        "storage": {"raft": {"path": "/openbao/data"}},
        "listener": {
            "tcp": {
                "address": f"[::]:{API_PORT}",
                "cluster_address": f"[::]:{RAFT_PORT}",
                "tls_cert_file": "/openbao/tls/tls.crt",
                "tls_key_file": "/openbao/tls/tls.key",
                "tls_min_version": "tls12",
                "tls_auto_reload": True,
            }
        },
    }
    manifests = [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace,
                "labels": {
                    "pod-security.kubernetes.io/enforce": "restricted",
                    "pod-security.kubernetes.io/enforce-version": "v1.30",
                },
            },
        },
        resource("v1", "ServiceAccount", NAME, automountServiceAccountToken=False),
        resource("cert-manager.io/v1", "Issuer", "bootstrap", spec={"selfSigned": {}}),
        resource(
            "cert-manager.io/v1",
            "Certificate",
            "ca",
            spec={
                "isCA": True,
                "commonName": f"{namespace}-ca",
                "secretName": "openbao-ca",
                "duration": "8760h",
                "privateKey": {"algorithm": "ECDSA", "size": 256},
                "issuerRef": {"name": "bootstrap", "kind": "Issuer"},
            },
        ),
        resource("cert-manager.io/v1", "Issuer", NAME, spec={"ca": {"secretName": "openbao-ca"}}),
        resource(
            "cert-manager.io/v1",
            "Certificate",
            NAME,
            spec={
                "secretName": "openbao-tls",
                "duration": "720h",
                "renewBefore": "168h",
                "dnsNames": [config.hostname, api_dns, peer_dns, f"*.{peer_dns}"],
                "privateKey": {"algorithm": "ECDSA", "size": 256, "rotationPolicy": "Always"},
                "issuerRef": {"name": NAME, "kind": "Issuer"},
            },
        ),
        resource("v1", "ConfigMap", NAME, data={"server.json": json.dumps(server_config, indent=2)}),
        resource(
            "v1",
            "Service",
            peer_name,
            spec={
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": labels,
                "ports": [
                    {"name": "https", "port": API_PORT},
                    {"name": "raft", "port": RAFT_PORT},
                ],
            },
        ),
        resource(
            "v1",
            "Service",
            NAME,
            spec={"selector": labels, "ports": [{"name": "https", "port": API_PORT}]},
        ),
        resource(
            "apps/v1",
            "StatefulSet",
            NAME,
            spec={
                "serviceName": peer_name,
                "replicas": config.replicas,
                # All processes may start sealed; join and unseal them one at a time.
                "podManagementPolicy": "Parallel",
                "updateStrategy": {"type": "OnDelete"},
                "persistentVolumeClaimRetentionPolicy": {"whenDeleted": "Retain", "whenScaled": "Retain"},
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "serviceAccountName": NAME,
                        "automountServiceAccountToken": False,
                        "nodeSelector": {AUTHORITY_LABEL: config.authority_pool},
                        "tolerations": [
                            {
                                "key": AUTHORITY_LABEL,
                                "operator": "Equal",
                                "value": config.authority_pool,
                                "effect": "NoSchedule",
                            }
                        ],
                        "affinity": {
                            "podAntiAffinity": {
                                "requiredDuringSchedulingIgnoredDuringExecution": [
                                    {"labelSelector": {"matchLabels": labels}, "topologyKey": "kubernetes.io/hostname"}
                                ]
                            }
                        },
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 100,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "terminationGracePeriodSeconds": 60,
                        "containers": [
                            {
                                "name": NAME,
                                "image": config.image,
                                "command": ["bao", "server", "-config=/openbao/config/server.json"],
                                "env": [
                                    {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                                    {"name": "BAO_RAFT_NODE_ID", "value": "$(POD_NAME)"},
                                    {"name": "BAO_API_ADDR", "value": f"https://{config.hostname}"},
                                    {"name": "BAO_CLUSTER_ADDR", "value": f"https://$(POD_NAME).{peer_dns}:{RAFT_PORT}"},
                                    {"name": "BAO_CACERT", "value": "/openbao/tls/ca.crt"},
                                    {"name": "BAO_ADDR", "value": f"https://$(POD_NAME).{peer_dns}:{API_PORT}"},
                                ],
                                "ports": [
                                    {"name": "https", "containerPort": API_PORT},
                                    {"name": "raft", "containerPort": RAFT_PORT},
                                ],
                                "resources": {
                                    "requests": {"cpu": "500m", "memory": "1Gi"},
                                    "limits": {"cpu": "2", "memory": "2Gi"},
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                # Sealed and uninitialized processes stay running for the ceremony.
                                "readinessProbe": {
                                    "httpGet": {
                                        "scheme": "HTTPS",
                                        "port": API_PORT,
                                        "path": "/v1/sys/health?standbyok=true",
                                    },
                                    "periodSeconds": 5,
                                },
                                "volumeMounts": [
                                    {"name": "data", "mountPath": "/openbao/data"},
                                    {"name": "config", "mountPath": "/openbao/config", "readOnly": True},
                                    {"name": "tls", "mountPath": "/openbao/tls", "readOnly": True},
                                    {"name": "tmp", "mountPath": "/tmp"},
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "config", "configMap": {"name": NAME}},
                            {"name": "tls", "secret": {"secretName": "openbao-tls", "defaultMode": 0o440}},
                            {"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "64Mi"}},
                        ],
                    },
                },
                "volumeClaimTemplates": [
                    {
                        "metadata": {"name": "data"},
                        "spec": {
                            "accessModes": ["ReadWriteOnce"],
                            "storageClassName": config.storage_class,
                            "resources": {"requests": {"storage": f"{config.storage_gb}Gi"}},
                        },
                    }
                ],
            },
        ),
        resource(
            "policy/v1", "PodDisruptionBudget", NAME, spec={"minAvailable": quorum, "selector": {"matchLabels": labels}}
        ),
        resource(
            "networking.k8s.io/v1",
            "NetworkPolicy",
            NAME,
            spec={
                "podSelector": {"matchLabels": labels},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [{"podSelector": {"matchLabels": labels}}],
                        "ports": [{"port": API_PORT}, {"port": RAFT_PORT}],
                    },
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"kubernetes.io/metadata.name": config.controller_namespace}
                                },
                                "podSelector": {"matchLabels": {"app": "iris-controller"}},
                            },
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"kubernetes.io/metadata.name": config.ingress_namespace}
                                },
                                "podSelector": {"matchLabels": {"app.kubernetes.io/name": "traefik"}},
                            },
                        ],
                        "ports": [{"port": API_PORT}],
                    },
                ],
                "egress": [
                    {
                        "to": [{"podSelector": {"matchLabels": labels}}],
                        "ports": [{"port": API_PORT}, {"port": RAFT_PORT}],
                    },
                    {
                        "to": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}}}],
                        "ports": [{"port": 53, "protocol": "UDP"}, {"port": 53, "protocol": "TCP"}],
                    },
                    {"to": [{"ipBlock": {"cidr": "0.0.0.0/0", "except": ["169.254.0.0/16"]}}], "ports": [{"port": 443}]},
                ],
            },
        ),
    ]
    if config.exposure is Exposure.TLS_PASSTHROUGH:
        manifests.append(
            resource(
                "traefik.io/v1alpha1",
                "IngressRouteTCP",
                NAME,
                spec={
                    "entryPoints": ["websecure"],
                    "routes": [
                        {"match": f"HostSNI(`{config.hostname}`)", "services": [{"name": NAME, "port": API_PORT}]}
                    ],
                    "tls": {"passthrough": True},
                },
            )
        )
    return manifests
