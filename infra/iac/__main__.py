# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin IaC.

Reads the target cluster from stack config (`marin-iac:cluster`), loads its Iris config +
typed `provisioning:` section, and builds the provider-appropriate components. Minimal cut:
CoreWeave RBAC + reserved NodePools (enough to `pulumi preview`). Kueue/Traefik/object-storage
and the CKS cluster object are the next slices.
"""

import os
import sys

# Make the `iac` package importable without a separate install step: Pulumi runs this file
# from infra/iac/, so add its src/ to the path. Deps (pulumi, pulumi-kubernetes) and
# marin-iris/marin-rigging come from the shared repo virtualenv (marin-iac is a workspace
# member; run `uv sync --all-packages`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pulumi
import pulumi_kubernetes as k8s
from iac.config import Provider, load_iris_config, load_provisioning
from iac.coreweave.cluster import CoreweaveCluster, CoreweaveClusterArgs
from iac.coreweave.rbac import IrisRbac, IrisRbacArgs
from iac.nodepools import derive_nodepools

DEFAULT_NAMESPACE = "iris"


def _build_coreweave(cluster: str, *, adopt: bool) -> None:
    provisioning = load_provisioning(cluster)
    assert provisioning.coreweave is not None  # guaranteed by load_provisioning
    coreweave_provisioning = provisioning.coreweave
    iris_config = load_iris_config(cluster)

    # Single-source: namespace + queue name come from the Iris config, not provisioning.
    kubernetes_provider = iris_config.kubernetes_provider
    if kubernetes_provider and kubernetes_provider.namespace:
        namespace = kubernetes_provider.namespace
    else:
        namespace = DEFAULT_NAMESPACE

    platform_coreweave = iris_config.platform.coreweave
    if platform_coreweave is None or not platform_coreweave.kubeconfig_path:
        raise ValueError(
            f"cluster {cluster!r} has no platform.coreweave.kubeconfig_path; "
            "the minimal IaC cut needs an out-of-cluster kubeconfig to target"
        )
    kubeconfig_path = os.path.expanduser(platform_coreweave.kubeconfig_path)
    k8s_provider = k8s.Provider("cw-k8s", kubeconfig=kubeconfig_path)

    CoreweaveCluster(
        "cluster",
        CoreweaveClusterArgs(
            cluster=coreweave_provisioning.cluster,
            region=coreweave_provisioning.region,
            nodepools=derive_nodepools(iris_config),
            adopt=adopt,
        ),
        k8s_provider=k8s_provider,
    )
    IrisRbac(
        "rbac",
        IrisRbacArgs(namespace=namespace, spec=coreweave_provisioning.rbac, adopt=adopt),
        k8s_provider=k8s_provider,
    )


def main() -> None:
    config = pulumi.Config("marin-iac")
    cluster = config.require("cluster")
    # Adoption recon: `pulumi config set marin-iac:import true` stamps import_ on every
    # resource so `pulumi preview` shows the real adoption diff (provider- and parent-correct)
    # instead of planning creates. Never run `pulumi up` through a destructive NodePool diff.
    adopt = config.get_bool("import") or False
    provider = load_provisioning(cluster).provider
    if provider is Provider.COREWEAVE:
        _build_coreweave(cluster, adopt=adopt)
    else:
        raise NotImplementedError(f"provider {provider!r} not yet implemented in iac")


main()
