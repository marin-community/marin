# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Namespace-scoped controller and task permissions for the synthetic CW proof."""

import pulumi
import pulumi_kubernetes as k8s

from iac.auth.broker import BROKER_LABELS
from iac.auth.openbao import APP_LABELS, dns_egress

AUTH_NAMESPACE = "marin-auth-staging"
TASK_NAMESPACE = "marin-auth-task-staging"
PROOF_LABELS = {"app.kubernetes.io/name": "auth-staging-proof", "marin.dev/auth-client": "true"}
TASK_LABELS = {"iris.managed": "true", "iris.runtime": "iris-kubernetes", "iris.egress": "internet"}


def staging_permissions(
    provider: pulumi.ProviderResource,
    *,
    dns_cidrs: tuple[str, ...],
    api_cidrs: tuple[str, ...],
    namespace_dependency: pulumi.Resource,
) -> None:
    """Grant the proof controller task-namespace access, without production Iris roles."""
    options = pulumi.ResourceOptions(provider=provider, depends_on=[namespace_dependency])
    namespace = k8s.core.v1.Namespace(
        "auth-staging-tasks",
        metadata={
            "name": TASK_NAMESPACE,
            "labels": {
                "pod-security.kubernetes.io/enforce": "restricted",
                "pod-security.kubernetes.io/audit": "restricted",
                "pod-security.kubernetes.io/warn": "restricted",
            },
        },
        opts=pulumi.ResourceOptions(provider=provider, protect=True, retain_on_delete=True),
    )
    account = k8s.core.v1.ServiceAccount(
        "auth-staging-proof",
        metadata={"name": "auth-staging-proof", "namespace": AUTH_NAMESPACE},
        automount_service_account_token=False,
        opts=options,
    )
    scoped = pulumi.ResourceOptions(provider=provider, depends_on=[namespace])
    role = k8s.rbac.v1.Role(
        "auth-staging-task-controller",
        metadata={"name": "auth-staging-task-controller", "namespace": TASK_NAMESPACE},
        rules=[
            {
                "api_groups": [""],
                "resources": ["pods", "pods/log", "configmaps", "events"],
                "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
            },
            {"api_groups": [""], "resources": ["secrets"], "verbs": ["get", "create", "patch", "delete"]},
            {"api_groups": [""], "resources": ["pods/exec"], "verbs": ["create"]},
            {
                "api_groups": ["policy"],
                "resources": ["poddisruptionbudgets"],
                "verbs": ["get", "list", "create", "update", "patch", "delete"],
            },
            {
                "api_groups": ["kueue.x-k8s.io"],
                "resources": ["workloads"],
                "verbs": ["get", "list", "watch", "delete"],
            },
        ],
        opts=scoped,
    )
    subject = {"kind": "ServiceAccount", "name": account.metadata["name"], "namespace": AUTH_NAMESPACE}
    k8s.apiextensions.CustomResource(
        "auth-staging-local-queue",
        api_version="kueue.x-k8s.io/v1beta1",
        kind="LocalQueue",
        metadata={"name": "auth-staging", "namespace": TASK_NAMESPACE},
        spec={"clusterQueue": "iris-cq"},
        opts=scoped,
    )
    k8s.rbac.v1.RoleBinding(
        "auth-staging-task-controller",
        metadata={"namespace": TASK_NAMESPACE},
        role_ref={"api_group": "rbac.authorization.k8s.io", "kind": "Role", "name": role.metadata["name"]},
        subjects=[subject],
        opts=scoped,
    )
    nodes = k8s.rbac.v1.ClusterRole(
        "auth-staging-node-observer",
        metadata={"name": "auth-staging-node-observer"},
        rules=[{"api_groups": [""], "resources": ["nodes"], "verbs": ["get", "list", "watch"]}],
        opts=options,
    )
    k8s.rbac.v1.ClusterRoleBinding(
        "auth-staging-node-observer",
        metadata={"name": "auth-staging-node-observer"},
        role_ref={"api_group": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": nodes.metadata["name"]},
        subjects=[subject],
        opts=options,
    )
    broker_peer = {
        "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": AUTH_NAMESPACE}},
        "pod_selector": {"match_labels": BROKER_LABELS},
    }
    task_peer = {
        "namespace_selector": {"match_labels": {"kubernetes.io/metadata.name": TASK_NAMESPACE}},
        "pod_selector": {"match_labels": TASK_LABELS},
    }
    k8s.networking.v1.NetworkPolicy(
        "auth-staging-task-boundary",
        metadata={"namespace": TASK_NAMESPACE},
        spec={"pod_selector": {}, "policy_types": ["Ingress", "Egress"], "ingress": [], "egress": []},
        opts=scoped,
    )
    k8s.networking.v1.NetworkPolicy(
        "auth-staging-task-broker",
        metadata={"namespace": TASK_NAMESPACE},
        spec={
            "pod_selector": {"match_labels": TASK_LABELS},
            "policy_types": ["Egress"],
            "egress": [{"to": [broker_peer], "ports": [{"protocol": "TCP", "port": 8090}]}, dns_egress(dns_cidrs)],
        },
        opts=scoped,
    )
    k8s.networking.v1.NetworkPolicy(
        "auth-staging-broker-task-clients",
        metadata={"namespace": AUTH_NAMESPACE},
        spec={
            "pod_selector": {"match_labels": BROKER_LABELS},
            "policy_types": ["Ingress"],
            "ingress": [{"from": [task_peer], "ports": [{"protocol": "TCP", "port": 8090}]}],
        },
        opts=options,
    )
    proof_peer = {"pod_selector": {"match_labels": PROOF_LABELS}}
    k8s.networking.v1.NetworkPolicy(
        "auth-staging-proof-bao",
        metadata={"namespace": AUTH_NAMESPACE},
        spec={
            "pod_selector": {"match_labels": APP_LABELS},
            "policy_types": ["Ingress"],
            "ingress": [{"from": [proof_peer], "ports": [{"protocol": "TCP", "port": 8200}]}],
        },
        opts=options,
    )
    k8s.networking.v1.NetworkPolicy(
        "auth-staging-proof-network",
        metadata={"namespace": AUTH_NAMESPACE},
        spec={
            "pod_selector": {"match_labels": PROOF_LABELS},
            "policy_types": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [
                {"to": [{"pod_selector": {"match_labels": BROKER_LABELS}}], "ports": [{"port": 8090}]},
                {"to": [{"pod_selector": {"match_labels": APP_LABELS}}], "ports": [{"port": 8200}]},
                {
                    "to": [
                        {
                            "namespace_selector": {
                                "match_labels": {"kubernetes.io/metadata.name": "marin-auth-db-staging"}
                            },
                            "pod_selector": {"match_labels": {"cnpg.io/podRole": "instance"}},
                        }
                    ],
                    "ports": [{"port": 5432}],
                },
                {"to": [{"ip_block": {"cidr": cidr}} for cidr in api_cidrs], "ports": [{"port": 443}, {"port": 6443}]},
                dns_egress(dns_cidrs),
            ],
        },
        opts=options,
    )
