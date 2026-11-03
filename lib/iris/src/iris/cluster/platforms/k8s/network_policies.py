# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operator-managed network isolation for Iris task pods."""

import ipaddress

from iris.cluster.platforms.k8s.constants import (
    CONTAINER_GROUP_NETWORK,
    EGRESS_LABEL,
    EGRESS_NETWORK_POLICY_PREFIX,
)
from iris.cluster.runtime.sandbox import EGRESS_BLOCKED_CIDRS, TaskNetwork

CONTAINER_GROUP_NETWORK_POLICY_NAME = f"{EGRESS_NETWORK_POLICY_PREFIX}{CONTAINER_GROUP_NETWORK}"
_NAMESPACE_NAME_LABEL = "kubernetes.io/metadata.name"
_DNS_NAMESPACE = "kube-system"
_DNS_PORT = 53


def _internet_except(service_cidr: str) -> list[str]:
    blocked = [ipaddress.ip_network(cidr) for cidr in EGRESS_BLOCKED_CIDRS]
    if service_cidr:
        services = ipaddress.ip_network(service_cidr)
        if not any(services.subnet_of(network) for network in blocked if network.version == services.version):
            blocked.append(services)
    return [str(network) for network in blocked]


def build_egress_network_policy(namespace: str, network: TaskNetwork, service_cidr: str = "") -> dict:
    """Deny ingress and permit only DNS, plus public IPv4 for INTERNET tasks."""
    if network is TaskNetwork.CLUSTER:
        raise ValueError("cluster-network pods carry no egress NetworkPolicy")
    egress: list[dict] = [
        {
            "to": [{"namespaceSelector": {"matchLabels": {_NAMESPACE_NAME_LABEL: _DNS_NAMESPACE}}}],
            "ports": [{"protocol": "UDP", "port": _DNS_PORT}, {"protocol": "TCP", "port": _DNS_PORT}],
        }
    ]
    if network is TaskNetwork.INTERNET:
        egress.append({"to": [{"ipBlock": {"cidr": "0.0.0.0/0", "except": _internet_except(service_cidr)}}]})
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": f"{EGRESS_NETWORK_POLICY_PREFIX}{network.value}", "namespace": namespace},
        "spec": {
            "podSelector": {"matchLabels": {EGRESS_LABEL: network.value}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": egress,
        },
    }


def build_container_group_network_policy(namespace: str) -> dict:
    """Deny all ingress and egress for container groups sharing pod loopback."""
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": CONTAINER_GROUP_NETWORK_POLICY_NAME, "namespace": namespace},
        "spec": {
            "podSelector": {"matchLabels": {EGRESS_LABEL: CONTAINER_GROUP_NETWORK}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [],
        },
    }


DRIVER_POD_UID_LABEL = "iris.pod_uid"
DRIVER_OWNER_LABEL = "iris.driver_pod_uid"
DRIVER_PORT_LABEL = "iris.driver_port"


def driver_connection_policy(namespace: str, pod_name: str, pod_uid: str, attempt_uid: str, port: int) -> dict:
    """Permit one declared endpoint from one immutable owning driver Pod."""
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": f"iris-driver-{pod_uid}-{port}",
            "namespace": namespace,
            "ownerReferences": [{"apiVersion": "v1", "kind": "Pod", "name": pod_name, "uid": pod_uid}],
        },
        "spec": {
            "podSelector": {"matchLabels": {DRIVER_OWNER_LABEL: pod_uid, DRIVER_PORT_LABEL: str(port)}},
            "policyTypes": ["Ingress"],
            "ingress": [
                {
                    "from": [
                        {
                            "podSelector": {
                                "matchLabels": {
                                    DRIVER_POD_UID_LABEL: pod_uid,
                                    "iris.attempt_uid": attempt_uid,
                                }
                            }
                        }
                    ],
                    "ports": [{"protocol": "TCP", "port": port}],
                }
            ],
        },
    }
