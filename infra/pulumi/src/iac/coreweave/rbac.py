# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IrisRbac — the Namespace + controller RBAC ceded from the K8s platform's ensure_rbac().

IaC is the owner of these resources now. Manifests come from the shared
`iris.cluster.platforms.k8s.rbac_manifests` builders, so IaC and any imperative caller
(e.g. the GPU gang smoke harness) render identically.
"""

from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s
from iris.cluster.platforms.k8s.credential_manifests import credential_secret_rbac
from iris.cluster.platforms.k8s.rbac_manifests import (
    cluster_role_binding_manifest,
    cluster_role_manifest,
    cluster_role_name,
    namespace_manifest,
    node_agent_rbac_manifests,
    service_account_manifest,
)

from iac.config import RbacSpec
from iac.imports import NO_IMPORTS, ImportRegistrar

GRAFANA_OBSERVER_ROLE = "marin-grafana-node-reader"
LOOM_SESSION_ROLE = "marin-loom-session-connect"


def _managed_auth_user_subjects(usernames: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "User",
            "name": username,
        }
        for username in usernames
    ]


@dataclass(frozen=True)
class IrisRbacArgs:
    namespace: str  # from kubernetes_provider.namespace
    spec: RbacSpec
    credential_broker: bool = False


class IrisRbac(pulumi.ComponentResource):
    """Namespace and separate controller and node-agent identities and grants."""

    def __init__(
        self,
        name: str,
        args: IrisRbacArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:IrisRbac", name, None, opts)
        role_name = cluster_role_name(args.namespace)

        def child_opts(depends_on: list | None = None) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
            )

        namespace_resource = namespace_manifest(args.namespace)
        namespace = k8s.core.v1.Namespace(
            "namespace",
            metadata=namespace_resource["metadata"],
            spec=namespace_resource["spec"],
            opts=child_opts(),
        )
        imports.register(namespace, parent=self, provider_id=args.namespace)
        # Exposed so other addons that create objects in this namespace (e.g. TraefikAddon's
        # federation Middleware/Ingress) can depend_on it — Pulumi has no ordering guarantee
        # between sibling ComponentResources otherwise, and a fresh cluster has no namespace yet.
        self.namespace = namespace
        sa_manifest = service_account_manifest(args.namespace, args.spec.service_account)
        service_account = k8s.core.v1.ServiceAccount(
            "service-account",
            metadata=sa_manifest["metadata"],
            opts=child_opts(depends_on=[namespace]),
        )
        imports.register(
            service_account,
            parent=self,
            provider_id=f"{args.namespace}/{args.spec.service_account}",
        )
        role_manifest = cluster_role_manifest(role_name)
        cluster_role = k8s.rbac.v1.ClusterRole(
            "cluster-role",
            metadata=role_manifest["metadata"],
            rules=role_manifest["rules"],
            opts=child_opts(),
        )
        imports.register(cluster_role, parent=self, provider_id=role_name)
        binding_manifest = cluster_role_binding_manifest(role_name, args.namespace, args.spec.service_account)
        cluster_role_binding = k8s.rbac.v1.ClusterRoleBinding(
            "cluster-role-binding",
            metadata=binding_manifest["metadata"],
            role_ref=binding_manifest["roleRef"],
            subjects=binding_manifest["subjects"],
            opts=child_opts(depends_on=[cluster_role, service_account]),
        )
        imports.register(cluster_role_binding, parent=self, provider_id=role_name)
        if args.credential_broker:
            secret_role, secret_binding = credential_secret_rbac(args.namespace, args.spec.service_account)
            session_role = k8s.rbac.v1.Role(
                "task-session-secrets",
                metadata=secret_role["metadata"],
                rules=secret_role["rules"],
                opts=child_opts(depends_on=[namespace]),
            )
            k8s.rbac.v1.RoleBinding(
                "task-session-secrets-binding",
                metadata=secret_binding["metadata"],
                role_ref=secret_binding["roleRef"],
                subjects=secret_binding["subjects"],
                opts=child_opts(depends_on=[session_role, service_account]),
            )

        agent_account_manifest, agent_role_manifest, agent_binding_manifest = node_agent_rbac_manifests(args.namespace)
        agent_account = k8s.core.v1.ServiceAccount(
            "node-agent-service-account",
            metadata=agent_account_manifest["metadata"],
            opts=child_opts(depends_on=[namespace]),
        )
        imports.register(
            agent_account,
            parent=self,
            provider_id=f"{args.namespace}/{agent_account_manifest['metadata']['name']}",
        )
        agent_role = k8s.rbac.v1.ClusterRole(
            "node-agent-cluster-role",
            metadata=agent_role_manifest["metadata"],
            rules=agent_role_manifest["rules"],
            opts=child_opts(),
        )
        imports.register(agent_role, parent=self, provider_id=agent_role_manifest["metadata"]["name"])
        agent_binding = k8s.rbac.v1.ClusterRoleBinding(
            "node-agent-cluster-role-binding",
            metadata=agent_binding_manifest["metadata"],
            role_ref=agent_binding_manifest["roleRef"],
            subjects=agent_binding_manifest["subjects"],
            opts=child_opts(depends_on=[agent_role, agent_account]),
        )
        imports.register(agent_binding, parent=self, provider_id=agent_binding_manifest["metadata"]["name"])
        self.register_outputs({})


@dataclass(frozen=True)
class GrafanaObserverRbacArgs:
    usernames: tuple[str, ...]


def grafana_observer_manifests(usernames: tuple[str, ...]) -> tuple[dict, dict]:
    """Return the cluster-inventory read role and its token-specific binding."""
    labels = {
        "app.kubernetes.io/name": "marin-grafana",
        "app.kubernetes.io/component": "k8s-observer",
    }
    role = {
        "metadata": {"name": GRAFANA_OBSERVER_ROLE, "labels": labels},
        "rules": [
            {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "list", "watch"]},
            {
                "apiGroups": ["compute.coreweave.com"],
                "resources": ["nodepools"],
                "verbs": ["get", "list", "watch"],
            },
        ],
    }
    binding = {
        "metadata": {"name": GRAFANA_OBSERVER_ROLE, "labels": labels},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": GRAFANA_OBSERVER_ROLE,
        },
        "subjects": _managed_auth_user_subjects(usernames),
    }
    return role, binding


class GrafanaObserverRbac(pulumi.ComponentResource):
    """Node and NodePool read access for Grafana's CoreWeave Managed Auth identity."""

    def __init__(
        self,
        name: str,
        args: GrafanaObserverRbacArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:GrafanaObserverRbac", name, None, opts)
        role_manifest, binding_manifest = grafana_observer_manifests(args.usernames)

        def child_opts(depends_on: list | None = None) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
            )

        cluster_role = k8s.rbac.v1.ClusterRole(
            "cluster-role",
            metadata=role_manifest["metadata"],
            rules=role_manifest["rules"],
            opts=child_opts(),
        )
        imports.register(cluster_role, parent=self, provider_id=GRAFANA_OBSERVER_ROLE)
        cluster_role_binding = k8s.rbac.v1.ClusterRoleBinding(
            "cluster-role-binding",
            metadata=binding_manifest["metadata"],
            role_ref=binding_manifest["roleRef"],
            subjects=binding_manifest["subjects"],
            opts=child_opts(depends_on=[cluster_role]),
        )
        imports.register(cluster_role_binding, parent=self, provider_id=GRAFANA_OBSERVER_ROLE)
        self.register_outputs({})


@dataclass(frozen=True)
class LoomSessionRbacArgs:
    namespace: str
    usernames: tuple[str, ...]
    namespace_dependency: pulumi.Resource


def loom_session_manifests(namespace: str, usernames: tuple[str, ...]) -> tuple[dict, dict]:
    """Return namespaced Pod connect access for Loom's CoreWeave identities."""
    labels = {
        "app.kubernetes.io/name": "marin-loom",
        "app.kubernetes.io/component": "session-connect",
    }
    metadata = {"name": LOOM_SESSION_ROLE, "namespace": namespace, "labels": labels}
    role = {
        "metadata": metadata,
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["pods/exec", "pods/portforward"],
                "verbs": ["create"],
            }
        ],
    }
    binding = {
        "metadata": metadata,
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": LOOM_SESSION_ROLE,
        },
        "subjects": _managed_auth_user_subjects(usernames),
    }
    return role, binding


class LoomSessionRbac(pulumi.ComponentResource):
    """Pod exec and port-forward access for Loom in one Iris namespace."""

    def __init__(
        self,
        name: str,
        args: LoomSessionRbacArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:LoomSessionRbac", name, None, opts)
        role_manifest, binding_manifest = loom_session_manifests(args.namespace, args.usernames)

        def child_opts(depends_on: list[pulumi.Resource] | None = None) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=depends_on)

        role = k8s.rbac.v1.Role(
            "role",
            metadata=role_manifest["metadata"],
            rules=role_manifest["rules"],
            opts=child_opts(depends_on=[args.namespace_dependency]),
        )
        k8s.rbac.v1.RoleBinding(
            "role-binding",
            metadata=binding_manifest["metadata"],
            role_ref=binding_manifest["roleRef"],
            subjects=binding_manifest["subjects"],
            opts=child_opts(depends_on=[role]),
        )
        self.register_outputs({})
