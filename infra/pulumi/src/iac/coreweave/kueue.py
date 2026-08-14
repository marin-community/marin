# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provision Kueue admission resources for an Iris CoreWeave cluster."""

from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s
from iris.cluster.platforms.k8s.kueue_manifests import (
    CW_REPO_URL,
    OPERATOR_NS,
    RELEASE_DEFAULT,
    RESOURCE_FLAVOR_NAME,
    TOPOLOGIES,
    build_cks_values,
    build_cluster_queue,
    build_resource_flavor,
    build_topology_cr,
)
from iris.cluster.platforms.k8s.types import IRIS_PRIORITY_CLASS_SYSTEM, iris_priority_class_manifest

from iac.config import KueueProvisioningSpec
from iac.imports import NO_IMPORTS, ImportRegistrar

# cks-kueue chart coordinates. The installer resolves `latest`; IaC pins the version so the
# release is reproducible. Bump this in lockstep with a chart upgrade.
CKS_KUEUE_CHART = "cks-kueue"
CKS_KUEUE_VERSION = "1.4.0"
# The Topology CRD's served apiVersion (install_kueue.py reads it from the live CRD; it is
# v1beta1 for cks-kueue 1.4.0).
TOPOLOGY_API_VERSION = "kueue.x-k8s.io/v1beta1"
MANAGER_DEPLOYMENT = "kueue-controller-manager"


@dataclass(frozen=True)
class KueueAddonArgs:
    namespace: str  # webhook scope + LocalQueue namespace, from kubernetes_provider.namespace
    cluster_queue: str  # from kubernetes_provider.kueue.cluster_queue
    spec: KueueProvisioningSpec


class KueueAddon(pulumi.ComponentResource):
    """Kueue gang-admission substrate for one Iris cluster.

    The webhooks are scoped to `args.namespace` (an unscoped webhook fail-closes CNI/system
    pods and deadlocks node delivery cluster-wide). The kueue-controller-manager is pinned to
    the `iris-system` PriorityClass via a Patch, because the chart exposes no priorityClassName
    value; the pin keeps a user pod from preempting the admission webhook.
    """

    def __init__(
        self,
        name: str,
        args: KueueAddonArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:KueueAddon", name, None, opts)

        def child_opts(depends_on: list | None = None) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
            )

        client_connection = args.spec.client_connection
        helm_values = build_cks_values(
            [args.namespace],
            manager_memory_limit=args.spec.manager_memory_limit,
            client_connection_qps=client_connection.qps,
            client_connection_burst=client_connection.burst,
        )

        # The cks-kueue Helm release. Webhooks scoped to args.namespace via the manager config.
        release = k8s.helm.v3.Release(
            "kueue",
            name=RELEASE_DEFAULT,
            chart=CKS_KUEUE_CHART,
            version=CKS_KUEUE_VERSION,
            namespace=OPERATOR_NS,
            create_namespace=True,
            repository_opts=k8s.helm.v3.RepositoryOptsArgs(repo=CW_REPO_URL),
            values=helm_values,
            opts=child_opts(),
        )
        # Helm Release import IDs are "<namespace>/<release-name>".
        imports.register(release, parent=self, provider_id=f"{OPERATOR_NS}/{RELEASE_DEFAULT}")

        # Topology CRs (infiniband + multinode-nvlink-ib) — applied out-of-band by the installer
        # because the chart renders them at a no-longer-served apiVersion.
        topologies = []
        for topology_name, levels in TOPOLOGIES.items():
            manifest = build_topology_cr(topology_name, levels, TOPOLOGY_API_VERSION)
            topology = k8s.apiextensions.CustomResource(
                f"topology-{topology_name}",
                api_version=TOPOLOGY_API_VERSION,
                kind="Topology",
                metadata=manifest["metadata"],
                spec=manifest["spec"],
                opts=child_opts(depends_on=[release]),
            )
            imports.register(topology, parent=self, provider_id=topology_name)
            topologies.append(topology)

        flavor_manifest = build_resource_flavor(args.spec.flavor_topology)
        resource_flavor = k8s.apiextensions.CustomResource(
            "resource-flavor",
            api_version=flavor_manifest["apiVersion"],
            kind=flavor_manifest["kind"],
            metadata=flavor_manifest["metadata"],
            spec=flavor_manifest["spec"],
            opts=child_opts(depends_on=[release, *topologies]),
        )
        imports.register(resource_flavor, parent=self, provider_id=RESOURCE_FLAVOR_NAME)

        queue_manifest = build_cluster_queue(args.cluster_queue)
        cluster_queue = k8s.apiextensions.CustomResource(
            "cluster-queue",
            api_version=queue_manifest["apiVersion"],
            kind=queue_manifest["kind"],
            metadata=queue_manifest["metadata"],
            spec=queue_manifest["spec"],
            opts=child_opts(depends_on=[release, resource_flavor]),
        )
        imports.register(cluster_queue, parent=self, provider_id=args.cluster_queue)

        # The iris-system PriorityClass and the manager's pin to it.
        priority_manifest = iris_priority_class_manifest(IRIS_PRIORITY_CLASS_SYSTEM)
        priority_class = k8s.scheduling.v1.PriorityClass(
            "iris-system",
            metadata=priority_manifest["metadata"],
            value=priority_manifest["value"],
            preemption_policy=priority_manifest["preemptionPolicy"],
            global_default=priority_manifest["globalDefault"],
            description=priority_manifest["description"],
            opts=child_opts(),
        )
        imports.register(priority_class, parent=self, provider_id=IRIS_PRIORITY_CLASS_SYSTEM)
        # Pin the chart-managed manager Deployment to iris-system. A Patch (server-side apply)
        # rather than an import: the chart owns the Deployment, IaC owns only this one field.
        k8s.apps.v1.DeploymentPatch(
            "kueue-manager-priority",
            metadata={"name": MANAGER_DEPLOYMENT, "namespace": OPERATOR_NS},
            spec={"template": {"spec": {"priorityClassName": IRIS_PRIORITY_CLASS_SYSTEM}}},
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[release]),
        )
        self.register_outputs({})
