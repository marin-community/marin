# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
import pulumi_kubernetes as k8s
import pytest
from iac.config import load_iris_config, load_provisioning
from iac.coreweave.kueue import KueueAddon, KueueAddonArgs, accelerator_nominal_quotas, cpu_flavor_node_labels
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks


class RecordingKubernetes(Mocks):
    def __init__(self) -> None:
        self.resources: dict[str, dict] = {}

    def new_resource(self, args: MockResourceArgs):
        self.resources[args.name] = args.inputs
        return f"{args.name}_id", args.inputs

    def call(self, args: MockCallArgs):
        return args.args, []


@pytest.mark.parametrize("cpu_scale_group", [None, "cpu-erapids"])
@pulumi.runtime.test
def test_kueue_deployment_publishes_cpu_admission_only_when_configured(cpu_scale_group, monkeypatch):
    mocks = RecordingKubernetes()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    config = load_iris_config("cw-us-east-08a")
    provisioning = load_provisioning("cw-us-east-08a").coreweave
    assert provisioning is not None
    registered: list[pulumi.Resource] = []
    custom_resource = k8s.apiextensions.CustomResource

    def record_resource(*args, **kwargs):
        resource = custom_resource(*args, **kwargs)
        registered.append(resource)
        return resource

    monkeypatch.setattr(k8s.apiextensions, "CustomResource", record_resource)
    KueueAddon(
        "kueue",
        KueueAddonArgs(
            namespace="iris",
            cluster_queue="iris-cq",
            spec=provisioning.kueue,
            nominal_quotas=accelerator_nominal_quotas(config),
            cpu_node_labels=cpu_flavor_node_labels(config, cpu_scale_group) if cpu_scale_group is not None else None,
        ),
        k8s_provider=k8s.Provider("k8s", kubeconfig=""),
    )

    def check(_):
        queue = mocks.resources["cluster-queue"]["spec"]
        flavors = queue["resourceGroups"][0]["flavors"]
        gpu_flavor = mocks.resources["resource-flavor"]["spec"]
        assert gpu_flavor == {"nodeLabels": {"iris.kueue": "true"}, "topologyName": "multinode-nvlink-ib"}
        if cpu_scale_group is None:
            assert [f["name"] for f in flavors] == ["cw-tas"]
            assert "resource-flavor-cpu" not in mocks.resources
            return
        assert mocks.resources["topology-cpu"]["spec"] == {"levels": [{"nodeLabel": "kubernetes.io/hostname"}]}
        cpu_flavor = mocks.resources["resource-flavor-cpu"]["spec"]
        assert cpu_flavor["topologyName"] == "iris-cpu-hostname"
        assert cpu_flavor["nodeLabels"] == {"iris.kueue": "true", "iris-cw-use08a-scale-group": "cpu-erapids"}
        assert [f["name"] for f in flavors] == ["iris-cpu", "cw-tas"]
        gpu_quotas = {r["name"]: r["nominalQuota"] for r in flavors[1]["resources"]}
        assert gpu_quotas["nvidia.com/gpu"] == gpu_quotas["rdma/ib"] == "864"

    return pulumi.Output.all(*(resource.urn for resource in registered)).apply(check)
