# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
import pulumi_kubernetes as k8s
import pytest
from iac.config import load_iris_config, load_provisioning
from iac.coreweave.kueue import KueueAddon, KueueAddonArgs, accelerator_nominal_quotas, cpu_flavor_node_labels
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks

# Each real cluster that configures a Kueue CPU flavor: scale group and the
# Iris scale-group label key its NodePools carry.
CPU_FLAVOR_CLUSTERS = {
    "cw-us-east-08a": ("cpu-erapids", "iris-cw-use08a-scale-group"),
    "cw-us-east-02a": ("cpu-genoa", "iris-cw-use02a-scale-group"),
    "cw-rno2a": ("cpu-turin", "iris-cw-rno2a-scale-group"),
}


class RecordingKubernetes(Mocks):
    def __init__(self) -> None:
        self.resources: dict[str, dict] = {}

    def new_resource(self, args: MockResourceArgs):
        self.resources[args.name] = args.inputs
        return f"{args.name}_id", args.inputs

    def call(self, args: MockCallArgs):
        return args.args, []


def deploy_kueue_addon(cluster: str) -> tuple[RecordingKubernetes, pulumi.Output]:
    """Render KueueAddon for a real cluster config.

    Returns the mocks and an Output that resolves once every custom resource is
    registered; assertions must run inside ``output.apply(...)``.
    """
    mocks = RecordingKubernetes()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    config = load_iris_config(cluster)
    provisioning = load_provisioning(cluster).coreweave
    assert provisioning is not None
    cpu_scale_group = provisioning.kueue.cpu_scale_group
    registered: list[pulumi.Resource] = []
    custom_resource = k8s.apiextensions.CustomResource

    def record_resource(*args, **kwargs):
        resource = custom_resource(*args, **kwargs)
        registered.append(resource)
        return resource

    k8s.apiextensions.CustomResource = record_resource
    try:
        KueueAddon(
            "kueue",
            KueueAddonArgs(
                namespace="iris",
                cluster_queue="iris-cq",
                spec=provisioning.kueue,
                nominal_quotas=accelerator_nominal_quotas(config),
                cpu_node_labels=(
                    cpu_flavor_node_labels(config, cpu_scale_group) if cpu_scale_group is not None else None
                ),
            ),
            k8s_provider=k8s.Provider("k8s", kubeconfig=""),
        )
    finally:
        k8s.apiextensions.CustomResource = custom_resource
    return mocks, pulumi.Output.all(*(resource.urn for resource in registered))


def quota_by_resource(flavor: dict) -> dict[str, str]:
    return {entry["name"]: entry["nominalQuota"] for entry in flavor["resources"]}


@pytest.mark.parametrize("cluster", sorted(CPU_FLAVOR_CLUSTERS))
@pulumi.runtime.test
def test_kueue_deployment_publishes_cpu_admission_for_each_configured_cluster(cluster):
    mocks, registered = deploy_kueue_addon(cluster)
    scale_group, scale_group_label = CPU_FLAVOR_CLUSTERS[cluster]
    accelerator_quotas = accelerator_nominal_quotas(load_iris_config(cluster))
    provisioning = load_provisioning(cluster).coreweave
    assert provisioning is not None
    flavor_topology = provisioning.kueue.flavor_topology

    def check(_):
        queue = mocks.resources["cluster-queue"]["spec"]
        flavors = queue["resourceGroups"][0]["flavors"]
        gpu_flavor = mocks.resources["resource-flavor"]["spec"]
        assert gpu_flavor == {"nodeLabels": {"iris.kueue": "true"}, "topologyName": flavor_topology}
        assert mocks.resources["topology-cpu"]["spec"] == {"levels": [{"nodeLabel": "kubernetes.io/hostname"}]}
        cpu_flavor = mocks.resources["resource-flavor-cpu"]["spec"]
        assert cpu_flavor["topologyName"] == "iris-cpu-hostname"
        assert cpu_flavor["nodeLabels"] == {"iris.kueue": "true", scale_group_label: scale_group}
        assert [f["name"] for f in flavors] == ["iris-cpu", "cw-tas"]
        cpu_quotas = quota_by_resource(flavors[0])
        assert cpu_quotas["nvidia.com/gpu"] == cpu_quotas["rdma/ib"] == "0"
        # cw-tas keeps its binding accelerator quotas and the non-binding sentinels;
        # only the CPU flavor is non-binding on accelerators.
        gpu_quotas = quota_by_resource(flavors[1])
        assert gpu_quotas["nvidia.com/gpu"] == accelerator_quotas["nvidia.com/gpu"]
        assert gpu_quotas["rdma/ib"] == accelerator_quotas["rdma/ib"]
        assert gpu_quotas["cpu"] == "1G" and gpu_quotas["memory"] == "1Pi" and gpu_quotas["ephemeral-storage"] == "1Pi"
        # Kueue validates that every flavor lists coveredResources in one order.
        assert [r["name"] for r in flavors[0]["resources"]] == [r["name"] for r in flavors[1]["resources"]]

    return registered.apply(check)


@pulumi.runtime.test
def test_kueue_deployment_omits_cpu_admission_without_a_cpu_scale_group():
    mocks, registered = deploy_kueue_addon("cw-us-west-04a")

    def check(_):
        flavors = mocks.resources["cluster-queue"]["spec"]["resourceGroups"][0]["flavors"]
        assert [f["name"] for f in flavors] == ["cw-tas"]
        assert "resource-flavor-cpu" not in mocks.resources
        assert "topology-cpu" not in mocks.resources

    return registered.apply(check)
