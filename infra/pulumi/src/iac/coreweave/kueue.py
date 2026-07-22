# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KueueAddon — the cks-kueue Helm release plus the cluster-scoped Kueue objects Iris needs.

Reproduces what `lib/iris/scripts/install_kueue.py --with-queues` installs, so an already-
installed cluster adopts with no change: the `cks-kueue` Helm release (webhooks scoped to the
Iris namespace), the `infiniband` + `multinode-nvlink-ib` Topology CRs, the `cw-ib` and
`cw-cpu` ResourceFlavors, the ClusterQueue, the `iris-system` PriorityClass, and the out-of-band
pin of the kueue-controller-manager to that PriorityClass. Manifest shapes come from the shared
`iris.cluster.platforms.k8s.kueue_manifests` builders, so IaC and the script render identically.
"""

from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s
from iris.cluster.platforms.k8s.kueue_manifests import (
    CPU_RESOURCE_FLAVOR_NAME,
    CW_REPO_URL,
    OPERATOR_NS,
    RELEASE_DEFAULT,
    RESOURCE_FLAVOR_NAME,
    TOPOLOGIES,
    build_cks_values,
    build_cluster_queue,
    build_cpu_resource_flavor,
    build_resource_flavor,
    build_topology_cr,
)
from iris.cluster.platforms.k8s.types import IRIS_PRIORITY_CLASS_SYSTEM, iris_priority_class_manifest

from iac.config import KueueProvisioningSpec

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
    # Adoption mode: stamp import_ on each cluster-scoped object so `pulumi preview` shows the
    # real adoption diff instead of planning creates. Set via the `marin-iac:import` flag.
    adopt: bool = False


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
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:KueueAddon", name, None, opts)

        def child_opts(import_id: str | None = None, depends_on: list | None = None) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
                import_=import_id if (args.adopt and import_id) else None,
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
            values=build_cks_values([args.namespace], manager_memory_limit=args.spec.manager_memory_limit),
            # helm Release import id is "<namespace>/<release-name>".
            opts=child_opts(f"{OPERATOR_NS}/{RELEASE_DEFAULT}"),
        )

        # Topology CRs (infiniband + multinode-nvlink-ib) — applied out-of-band by the installer
        # because the chart renders them at a no-longer-served apiVersion.
        topologies = []
        for topology_name, levels in TOPOLOGIES.items():
            manifest = build_topology_cr(topology_name, levels, TOPOLOGY_API_VERSION)
            topologies.append(
                k8s.apiextensions.CustomResource(
                    f"topology-{topology_name}",
                    api_version=TOPOLOGY_API_VERSION,
                    kind="Topology",
                    metadata=manifest["metadata"],
                    spec=manifest["spec"],
                    opts=child_opts(topology_name, depends_on=[release]),
                )
            )

        flavor_manifest = build_resource_flavor(args.spec.flavor_topology)
        resource_flavor = k8s.apiextensions.CustomResource(
            "resource-flavor",
            api_version=flavor_manifest["apiVersion"],
            kind=flavor_manifest["kind"],
            metadata=flavor_manifest["metadata"],
            spec=flavor_manifest["spec"],
            opts=child_opts(RESOURCE_FLAVOR_NAME, depends_on=[release, *topologies]),
        )

        # The selector-less cw-cpu flavor CPU-only pods match (they route through Kueue too).
        cpu_flavor_manifest = build_cpu_resource_flavor()
        cpu_resource_flavor = k8s.apiextensions.CustomResource(
            "cpu-resource-flavor",
            api_version=cpu_flavor_manifest["apiVersion"],
            kind=cpu_flavor_manifest["kind"],
            metadata=cpu_flavor_manifest["metadata"],
            spec=cpu_flavor_manifest["spec"],
            opts=child_opts(CPU_RESOURCE_FLAVOR_NAME, depends_on=[release]),
        )

        queue_manifest = build_cluster_queue(args.cluster_queue)
        k8s.apiextensions.CustomResource(
            "cluster-queue",
            api_version=queue_manifest["apiVersion"],
            kind=queue_manifest["kind"],
            metadata=queue_manifest["metadata"],
            spec=queue_manifest["spec"],
            opts=child_opts(args.cluster_queue, depends_on=[release, resource_flavor, cpu_resource_flavor]),
        )

        # The iris-system PriorityClass and the manager's pin to it.
        priority_manifest = iris_priority_class_manifest(IRIS_PRIORITY_CLASS_SYSTEM)
        k8s.scheduling.v1.PriorityClass(
            "iris-system",
            metadata=priority_manifest["metadata"],
            value=priority_manifest["value"],
            preemption_policy=priority_manifest["preemptionPolicy"],
            global_default=priority_manifest["globalDefault"],
            description=priority_manifest["description"],
            opts=child_opts(IRIS_PRIORITY_CLASS_SYSTEM),
        )
        # Pin the chart-managed manager Deployment to iris-system. A Patch (server-side apply)
        # rather than an import: the chart owns the Deployment, IaC owns only this one field.
        k8s.apps.v1.DeploymentPatch(
            "kueue-manager-priority",
            metadata={"name": MANAGER_DEPLOYMENT, "namespace": OPERATOR_NS},
            spec={"template": {"spec": {"priorityClassName": IRIS_PRIORITY_CLASS_SYSTEM}}},
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[release]),
        )
        self.register_outputs({})
