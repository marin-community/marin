# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IrisRbac — the Namespace + controller RBAC ceded from the K8s platform's ensure_rbac().

IaC is the owner of these resources now (the Iris `ensure_rbac` is being deleted). The
ClusterRole grants exactly the permissions the `iris-controller` ServiceAccount uses at
runtime — verified against the live cluster, since the coreweave.md §5 table is stale (it
lists only 4 of the 8 rules).
"""

from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s

from iac.config import RbacSpec

# The permissions the iris-controller ServiceAccount needs at runtime. Until the cede
# (spec §4), iris.cluster.platforms.k8s.controller.ensure_rbac still applies an identical
# ClusterRole on every `cluster start`, so these rules must stay byte-identical to it; the
# cede deletes ensure_rbac and leaves this the sole copy, closing the drift window. If the
# controller gains a new permission need before then, update BOTH lists.
_CLUSTER_ROLE_RULES = [
    {
        "apiGroups": ["compute.coreweave.com"],
        "resources": ["nodepools"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {
        # Bound via ClusterRoleBinding, so this grants pod access in ALL namespaces —
        # required for blocker eviction in kubernetes_provider.preempt_namespaces.
        "apiGroups": [""],
        "resources": ["pods", "pods/exec", "pods/log"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "list", "watch"]},
    {
        "apiGroups": [""],
        "resources": ["configmaps"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {"apiGroups": ["metrics.k8s.io"], "resources": ["pods"], "verbs": ["get", "list"]},
    {
        "apiGroups": ["policy"],
        "resources": ["poddisruptionbudgets"],
        "verbs": ["get", "list", "create", "update", "patch", "delete"],
    },
    {
        # Kueue gang admission: Iris deletes the per-pod-group Workload to release a
        # torn-down gang's reserved quota.
        "apiGroups": ["kueue.x-k8s.io"],
        "resources": ["workloads"],
        "verbs": ["get", "list", "watch", "delete"],
    },
    {
        # Iris creates iris-{production,interactive,batch} PriorityClass objects at startup.
        "apiGroups": ["scheduling.k8s.io"],
        "resources": ["priorityclasses"],
        "verbs": ["get", "create", "update", "patch", "delete"],
    },
]


@dataclass(frozen=True)
class IrisRbacArgs:
    namespace: str  # from kubernetes_provider.namespace
    spec: RbacSpec
    # Adoption mode: stamp import_=<live id> on each resource so `pulumi preview` shows the
    # real adoption diff (provider- and parent-correct) instead of planning creates. Set via
    # the `marin-iac:import` stack flag. See spec.md §4.
    adopt: bool = False


class IrisRbac(pulumi.ComponentResource):
    """Namespace, iris-controller ServiceAccount, and namespace-qualified ClusterRole +
    ClusterRoleBinding (`iris-controller-<namespace>`, to allow multiple Iris instances on
    one CKS cluster)."""

    def __init__(
        self,
        name: str,
        args: IrisRbacArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:IrisRbac", name, None, opts)
        role_name = f"iris-controller-{args.namespace}"

        def child_opts(import_id: str, depends_on: list | None = None) -> pulumi.ResourceOptions:
            # k8s import IDs: cluster-scoped => "<name>"; namespaced => "<namespace>/<name>".
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
                import_=import_id if args.adopt else None,
            )

        namespace = k8s.core.v1.Namespace(
            "namespace",
            metadata={"name": args.namespace},
            opts=child_opts(args.namespace),
        )
        # Exposed so other addons that create objects in this namespace (e.g. TraefikAddon's
        # federation Middleware/Ingress) can depend_on it — Pulumi has no ordering guarantee
        # between sibling ComponentResources otherwise, and a fresh cluster has no namespace yet.
        self.namespace = namespace
        service_account = k8s.core.v1.ServiceAccount(
            "service-account",
            metadata={"name": args.spec.service_account, "namespace": args.namespace},
            opts=child_opts(f"{args.namespace}/{args.spec.service_account}", depends_on=[namespace]),
        )
        cluster_role = k8s.rbac.v1.ClusterRole(
            "cluster-role",
            metadata={"name": role_name},
            rules=_CLUSTER_ROLE_RULES,
            opts=child_opts(role_name),
        )
        k8s.rbac.v1.ClusterRoleBinding(
            "cluster-role-binding",
            metadata={"name": role_name},
            role_ref={"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": role_name},
            subjects=[
                {
                    "kind": "ServiceAccount",
                    "name": args.spec.service_account,
                    "namespace": args.namespace,
                }
            ],
            opts=child_opts(role_name, depends_on=[cluster_role, service_account]),
        )
        self.register_outputs({})
