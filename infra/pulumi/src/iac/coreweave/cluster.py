# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreweaveCluster — the reserved NodePools for an existing CKS cluster.

Renders the cluster's NodePools as `compute.coreweave.com/v1alpha1 NodePool` custom
resources, the objects Iris applies today, and targets an existing CKS cluster and its
kubeconfig.

The CKS cluster object (`coreweave_cks_cluster` + VPC) stays outside Pulumi. Managing or
adopting it would need the CoreWeave Terraform provider bridged into Pulumi
(`pulumi package add terraform-provider coreweave/coreweave`) and CoreWeave API credentials,
neither of which exists here (design.md Open Questions; gaps.md row 3). `CksClusterSpec`
(`args.cluster`, exported below) records that externally-provisioned cluster as in-tree config.
"""

from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s
from iris.cluster.platforms.k8s.nodepool_manifests import nodepool_manifest

from iac.config import CksClusterSpec
from iac.nodepools import NodePoolSpec


@dataclass(frozen=True)
class CoreweaveClusterArgs:
    cluster: CksClusterSpec
    nodepools: list[NodePoolSpec]
    # Adoption mode: stamp import_=<nodepool name> on each NodePool so `pulumi preview` shows
    # the real adoption diff instead of planning creates. Set via `marin-iac:import`. §4.
    adopt: bool = False


def _nodepool_manifest(nodepool: NodePoolSpec) -> dict:
    # targetNodes is required at create; we seed it to minNodes and hand the live count to
    # CoreWeave's autoscaler thereafter via ignore_changes on spec.targetNodes (see below).
    return nodepool_manifest(
        nodepool.name,
        nodepool.instance_type,
        node_labels=nodepool.node_labels,
        min_nodes=nodepool.min_nodes,
        max_nodes=nodepool.max_nodes,
        target_nodes=nodepool.min_nodes,
        target_racks=nodepool.target_racks,
        autoscaling=nodepool.autoscaling,
    )


class CoreweaveCluster(pulumi.ComponentResource):
    """The reserved NodePools for one Iris cluster.

    Node-based pools are declared with `ignore_changes=["spec.targetNodes"]` so CoreWeave's
    autoscaler may move the live node count within [minNodes, maxNodes] without Pulumi
    reverting it — IaC owns the envelope, not the runtime count. Rack-based (NVL72) pools
    don't autoscale, so IaC owns `spec.targetRacks` outright (no ignore_changes).
    """

    def __init__(
        self,
        name: str,
        args: CoreweaveClusterArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:CoreweaveCluster", name, None, opts)

        for nodepool in args.nodepools:
            manifest = _nodepool_manifest(nodepool)
            # Node-based pools cede the live count to CoreWeave's autoscaler; rack-based pools
            # have no autoscaler, so IaC keeps targetRacks reconciled.
            ignore_changes = [] if nodepool.target_racks is not None else ["spec.targetNodes"]
            # NodePools are cluster-scoped, so the k8s import ID is just the object name.
            k8s.apiextensions.CustomResource(
                f"nodepool-{nodepool.name}",
                api_version=manifest["apiVersion"],
                kind=manifest["kind"],
                metadata=manifest["metadata"],
                spec=manifest["spec"],
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=k8s_provider,
                    ignore_changes=ignore_changes,
                    import_=nodepool.name if args.adopt else None,
                ),
            )
        # Exported so `pulumi stack output` names the CKS cluster this stack's NodePools
        # live on; the cluster object itself is not Pulumi-managed (see module docstring).
        self.register_outputs({"cluster_name": args.cluster.name, "cluster_zone": args.cluster.zone})
