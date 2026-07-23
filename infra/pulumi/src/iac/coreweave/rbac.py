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
from iris.cluster.platforms.k8s.rbac_manifests import (
    cluster_role_binding_manifest,
    cluster_role_manifest,
    cluster_role_name,
    namespace_manifest,
    service_account_manifest,
)

from iac.config import RbacSpec


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
        role_name = cluster_role_name(args.namespace)

        def child_opts(import_id: str, depends_on: list | None = None) -> pulumi.ResourceOptions:
            # k8s import IDs: cluster-scoped => "<name>"; namespaced => "<namespace>/<name>".
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
                import_=import_id if args.adopt else None,
            )

        namespace_resource = namespace_manifest(args.namespace)
        namespace = k8s.core.v1.Namespace(
            "namespace",
            metadata=namespace_resource["metadata"],
            spec=namespace_resource["spec"],
            opts=child_opts(args.namespace),
        )
        # Exposed so other addons that create objects in this namespace (e.g. TraefikAddon's
        # federation Middleware/Ingress) can depend_on it — Pulumi has no ordering guarantee
        # between sibling ComponentResources otherwise, and a fresh cluster has no namespace yet.
        self.namespace = namespace
        sa_manifest = service_account_manifest(args.namespace, args.spec.service_account)
        service_account = k8s.core.v1.ServiceAccount(
            "service-account",
            metadata=sa_manifest["metadata"],
            opts=child_opts(f"{args.namespace}/{args.spec.service_account}", depends_on=[namespace]),
        )
        role_manifest = cluster_role_manifest(role_name)
        cluster_role = k8s.rbac.v1.ClusterRole(
            "cluster-role",
            metadata=role_manifest["metadata"],
            rules=role_manifest["rules"],
            opts=child_opts(role_name),
        )
        binding_manifest = cluster_role_binding_manifest(role_name, args.namespace, args.spec.service_account)
        k8s.rbac.v1.ClusterRoleBinding(
            "cluster-role-binding",
            metadata=binding_manifest["metadata"],
            role_ref=binding_manifest["roleRef"],
            subjects=binding_manifest["subjects"],
            opts=child_opts(role_name, depends_on=[cluster_role, service_account]),
        )
        self.register_outputs({})
