# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check physical locality before model startup."""

import json
import sys
import uuid
from ctypes import POINTER, cast
from types import SimpleNamespace

import pynvml
import pytest
from ray._private.label_utils import parse_node_labels_string
from skyrlrecipe.locality import LocalityPlan

from cloud.iris import task_runtime
from cloud.iris.role_topology import (
    FABRIC_LABEL,
    HOST_LABEL,
    FabricSample,
    RoleTopology,
    node_physical_labels,
    probe_fabric_info,
)

CLUSTER_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def fabric_info(**overrides):
    """Four GPUs on one healthy NVLink fabric, with every GPU carrying the same overrides."""
    fields = {
        "cluster_uuid": CLUSTER_UUID,
        "clique_id": 1,
        "state": pynvml.NVML_GPU_FABRIC_STATE_COMPLETED,
        "status": pynvml.NVML_SUCCESS,
    }
    fields.update(overrides)
    return tuple(FabricSample(**fields) for _ in range(4))


def role_nodes(topology):
    return [
        {
            "Alive": True,
            "Labels": {
                **topology.node_labels(rank),
                **node_physical_labels(rank=rank, gpus_per_node=4, host=f"host-{rank}", fabric_info=fabric_info()),
            },
        }
        for rank in range(topology.num_nodes)
    ]


@pytest.fixture
def audit(tmp_path, monkeypatch):
    def run(topology, nodes):
        monkeypatch.setitem(
            sys.modules,
            "ray",
            SimpleNamespace(
                init=lambda **kwargs: None,
                shutdown=lambda: None,
                nodes=lambda: nodes,
            ),
        )
        task_runtime.verify_role_placement("fixture", topology, tmp_path)
        return json.loads((tmp_path / "role-topology.json").read_text())

    return run


def test_healthy_single_node_writes_a_verified_report(audit):
    topology = RoleTopology(("policy",), LocalityPlan((1,), (0,)))
    report = audit(topology, role_nodes(topology))
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["fabric"] == f"{CLUSTER_UUID}.1"


def test_healthy_multi_node_groups_may_share_a_fabric(audit):
    topology = RoleTopology(("policy", "inference_0"), LocalityPlan((2, 1), (0, 1)))
    report = audit(topology, role_nodes(topology))
    assert len(report["nodes"]) == 3
    assert len({row["host"] for row in report["nodes"]}) == 3
    assert {row["fabric"] for row in report["nodes"]} == {f"{CLUSTER_UUID}.1"}


@pytest.mark.parametrize(
    "label,value", [(FABRIC_LABEL, f"{CLUSTER_UUID}.2"), (HOST_LABEL, "host-0")], ids=["split-group", "duplicate-host"]
)
def test_invalid_physical_placement_is_rejected(audit, tmp_path, label, value):
    topology = RoleTopology(("policy",), LocalityPlan((2,), (0,)))
    nodes = role_nodes(topology)
    nodes[1]["Labels"][label] = value
    with pytest.raises(ValueError):
        audit(topology, nodes)
    assert not (tmp_path / "role-topology.json").exists()


@pytest.mark.parametrize(
    "overrides",
    [{"state": pynvml.NVML_GPU_FABRIC_STATE_IN_PROGRESS}, {"status": pynvml.NVML_ERROR_UNINITIALIZED}],
    ids=["in-progress-fabric", "failed-fabric-attach"],
)
def test_unhealthy_node_fabric_is_rejected(overrides):
    with pytest.raises(ValueError):
        node_physical_labels(rank=0, gpus_per_node=4, host="host-0", fabric_info=fabric_info(**overrides))


def test_ray_accepts_physical_fabric_label():
    labels = node_physical_labels(rank=0, gpus_per_node=4, host="host-0", fabric_info=fabric_info())
    serialized = ",".join(f"{key}={value}" for key, value in labels.items())
    assert parse_node_labels_string(serialized) == labels


def test_fabric_probe_reads_versioned_nvml_output(monkeypatch):
    sample = fabric_info()[0]

    def write_fabric_info(_handle, output):
        info = cast(output, POINTER(pynvml.c_nvmlGpuFabricInfoV_t)).contents
        assert info.version == pynvml.nvmlGpuFabricInfo_v2
        info.clusterUuid[:] = sample.cluster_uuid.bytes
        info.cliqueId = sample.clique_id
        info.state = sample.state
        info.status = sample.status

    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetGpuFabricInfoV", write_fabric_info)

    assert probe_fabric_info() == (sample,)


def test_wrong_gpu_count_is_rejected():
    with pytest.raises(ValueError):
        node_physical_labels(rank=0, gpus_per_node=4, host="host-0", fabric_info=fabric_info()[:3])
