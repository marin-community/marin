"""Role-local placement from a launch recipe through physical discovery and Ray bundles."""

from collections import Counter
import json

import pytest
import ray
import yaml
from omegaconf import OmegaConf
from ray.cluster_utils import Cluster
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from cloud.iris.launch_config import load_launch_config
from cloud.iris.role_topology import (
    FABRIC_RESOURCE_PREFIX,
    GPU_RESOURCE_PREFIX,
    derive_role_topology,
    node_role_resources,
    validate_role_nodes,
)
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.iris.task_runtime import wait_for_nodes
from marinskyrl.role_placement import role_bundles, verify_role_placement


def launch_topology(tmp_path, *, nodes=48, policy_nodes=16, engines=2, dp=64, width=4, variant="GB200"):
    raw = raw_launch_config()
    raw["iris"]["role_topology"] = "nvlink.domain"
    raw["iris"]["allocation"].update(num_nodes=nodes, gpus_per_node=width, gpu_variant=variant)
    raw["skyrl"]["trainer"]["placement"].update(
        colocate_all=False,
        policy_num_nodes=policy_nodes,
        policy_num_gpus_per_node=width,
    )
    raw["skyrl"]["generator"].update(
        num_inference_engines=engines,
        inference_engine_data_parallel_size=dp,
        inference_engine_expert_parallel_size=dp,
    )
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    return derive_role_topology(OmegaConf.to_container(config, resolve=True))


def physical_resources(topology, rank):
    devices = "".join(
        f"<gpu><uuid>GPU-{rank}-{index}</uuid><fabric><state>Completed</state><status>Success</status>"
        f"<clusterUuid>cluster</clusterUuid><cliqueId>{rank // topology.nodes_per_domain}</cliqueId></fabric></gpu>"
        for index in range(topology.gpus_per_node)
    )
    return node_role_resources(
        topology=topology,
        rank=rank,
        host=f"host-{rank}",
        fabric_xml=f"<nvidia_smi_log>{devices}</nvidia_smi_log>",
    )


def physical_nodes(topology):
    return [
        {
            "NodeID": f"ray-{rank}",
            "Alive": True,
            "Resources": {"GPU": topology.gpus_per_node, **physical_resources(topology, rank)},
        }
        for rank in range(topology.num_nodes)
    ]


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        ({}, {"policy": (64, "cluster:0"), "inference_0": (64, "cluster:1"), "inference_1": (64, "cluster:2")}),
        (
            {"nodes": 32, "dp": 32},
            {"policy": (64, "cluster:0"), "inference_0": (32, "cluster:1"), "inference_1": (32, "cluster:1")},
        ),
        (
            {"nodes": 16, "policy_nodes": 8, "dp": 16},
            {"policy": (32, "cluster:0"), "inference_0": (16, "cluster:0"), "inference_1": (16, "cluster:0")},
        ),
        (
            {"nodes": 4, "policy_nodes": 1, "dp": 6},
            {"policy": (4, "cluster:0"), "inference_0": (6, "cluster:0"), "inference_1": (6, "cluster:0")},
        ),
        (
            {"nodes": 6, "policy_nodes": 1, "engines": 3, "dp": 6},
            {
                "policy": (4, "cluster:0"),
                "inference_0": (6, "cluster:0"),
                "inference_1": (6, "cluster:0"),
                "inference_2": (6, "cluster:0"),
            },
        ),
        (
            {"nodes": 2, "policy_nodes": 1, "dp": 4, "width": 8, "variant": "H100"},
            {"policy": (8, "host:host-0"), "inference_0": (4, "host:host-1"), "inference_1": (4, "host:host-1")},
        ),
    ],
)
def test_configured_groups_stay_within_domains_and_can_share_one(tmp_path, geometry, expected):
    topology = launch_topology(tmp_path, **geometry)
    rows = validate_role_nodes(physical_nodes(topology), topology=topology)
    counts = Counter()
    fabrics = {role: set() for role in expected}
    for row in rows:
        counts.update(row["roles"])
        if "policy" in row["roles"]:
            assert row["roles"]["policy"] == topology.gpus_per_node
        for role in row["roles"]:
            fabrics[role].add(row["fabric"])
    assert counts == {role: count for role, (count, _) in expected.items()}
    assert fabrics == {role: {fabric} for role, (_, fabric) in expected.items()}
    assert len({gpu for row in rows for gpu in row["gpu_uuids"]}) == topology.num_nodes * topology.gpus_per_node


@pytest.mark.parametrize("wrong_layout", ["split_policy", "overlapping_slices", "duplicate_gpu"])
def test_physical_gate_rejects_incorrect_placement_despite_correct_counts(tmp_path, monkeypatch, wrong_layout):
    topology = launch_topology(tmp_path)
    nodes = physical_nodes(topology)
    if wrong_layout == "split_policy":
        resources = nodes[0]["Resources"]
        del resources[FABRIC_RESOURCE_PREFIX + "cluster:0"]
        resources[FABRIC_RESOURCE_PREFIX + "cluster:1"] = 4
    elif wrong_layout == "overlapping_slices":
        for node in nodes[16:32]:
            resources = node["Resources"]
            del resources[FABRIC_RESOURCE_PREFIX + "cluster:1"]
            resources[FABRIC_RESOURCE_PREFIX + "cluster:0"] = 4
    else:
        resources = nodes[1]["Resources"]
        del resources[GPU_RESOURCE_PREFIX + "GPU-1-0"]
        resources[GPU_RESOURCE_PREFIX + "GPU-0-0"] = 1
    # Ray is the remote I/O boundary. Exercise the actual pre-driver physical gate.
    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray, "nodes", lambda: nodes)
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="NVLink domains|duplicate physical GPUs"):
        wait_for_nodes("ray.test", 48, 1, role_topology=topology)
    assert not (tmp_path / "role-topology.json").exists()


def test_physical_gate_writes_a_complete_receipt_before_the_driver(tmp_path, monkeypatch):
    topology = launch_topology(tmp_path)
    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray, "nodes", lambda: physical_nodes(topology))
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"GPU": 192})
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    wait_for_nodes("ray.test", 48, 1, role_topology=topology)
    receipt = json.loads((tmp_path / "role-topology.json").read_text())
    assert {row["fabric"] for row in receipt["nodes"]} == {"cluster:0", "cluster:1", "cluster:2"}
    assert len(receipt["nodes"]) == 48


def test_invalid_group_geometry_is_rejected_before_ray_can_wait_for_capacity(tmp_path):
    # Native 24-host scheduling gives two 12-host domains. A 64-GPU trainer
    # cannot fit either, although the combined 96-GPU allocation fits two racks.
    with pytest.raises(ValueError, match="policy needs 64 GPUs"):
        launch_topology(tmp_path, nodes=24, dp=16)


def test_omitted_locality_preserves_colocated_launches_and_unlabelled_bundles(tmp_path):
    raw = raw_launch_config()
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    assert derive_role_topology(OmegaConf.to_container(config, resolve=True)) is None
    assert role_bundles("policy", [{"GPU": 8, "CPU": 8}], {"GPU": 8}) == [{"GPU": 8, "CPU": 8}]


@pytest.mark.slow
def test_real_ray_actors_use_disjoint_gpus_in_their_assigned_domains(tmp_path, monkeypatch):
    # Synthetic GPUs exercise real Ray reservations and actors on this CPU host;
    # physical NVLink and CUDA communication are qualified by the separate live probe.
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    topology = launch_topology(tmp_path, nodes=2, policy_nodes=1, dp=4, width=8, variant="H100")
    cluster = Cluster()
    try:
        for rank in range(2):
            cluster.add_node(
                num_cpus=8,
                num_gpus=8,
                resources=physical_resources(topology, rank),
                object_store_memory=100 * 1024**2,
                include_dashboard=False,
            )
        ray.init(address=cluster.address)
        validate_role_nodes(ray.nodes(), topology=topology)

        @ray.remote(num_cpus=1, num_gpus=4)
        class Worker:
            def identity(self):
                return ray.get_runtime_context().get_node_id(), ray.get_gpu_ids()

        actors = {}
        for role, count in (("policy", 8), ("inference_0", 4), ("inference_1", 4)):
            bundles = [{"GPU": 4, "CPU": 1}] * (count // 4)
            pg = placement_group(role_bundles(role, bundles, ray.cluster_resources()), "PACK")
            ray.get(pg.ready(), timeout=60)
            assignments = verify_role_placement(pg, role)
            handles = [
                Worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=index,
                    )
                ).remote()
                for index in range(len(bundles))
            ]
            identities = ray.get([handle.identity.remote() for handle in handles], timeout=60)
            assert [node_id for node_id, _ in identities] == [assignments[index] for index in range(len(bundles))]
            actors[role] = identities
        assert len({node_id for node_id, _ in actors["policy"]}) == 1
        assert {node_id for node_id, _ in actors["inference_0"]} == {node_id for node_id, _ in actors["inference_1"]}
        assert {node_id for node_id, _ in actors["policy"]}.isdisjoint(node_id for node_id, _ in actors["inference_0"])
        assert (
            len({(node_id, gpu) for identities in actors.values() for node_id, gpus in identities for gpu in gpus})
            == 16
        )
    finally:
        ray.shutdown()
        cluster.shutdown()
