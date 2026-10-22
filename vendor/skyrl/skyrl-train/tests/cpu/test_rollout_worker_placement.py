# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import Counter
from collections.abc import Iterator

import pytest
import ray
from ray.cluster_utils import Cluster
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl_train.rollouts.workers import (
    NODE_ID_LABEL,
    RayNode,
    RolloutWorkerPlacementError,
    RolloutWorkerResources,
    live_ray_nodes,
    reserve_rollout_workers,
    rollout_worker_nodes,
)


def _nodes(*cpus: float) -> list[RayNode]:
    return [RayNode(f"node-{index}", f"10.0.0.{index}", count) for index, count in enumerate(cpus)]


def test_workers_spread_evenly_and_deterministically_over_every_node():
    # The incident topology: 12 workers of 8 CPUs on a six-node run, which used to land on the driver's node.
    nodes = _nodes(*[128] * 6)

    assignment = rollout_worker_nodes(nodes, num_workers=12, cpus_per_worker=8)

    assert Counter(assignment) == {node.node_id: 2 for node in nodes}
    assert rollout_worker_nodes(list(reversed(nodes)), num_workers=12, cpus_per_worker=8) == assignment


def test_a_node_takes_only_the_workers_its_cpus_hold():
    small, large, cpu_less = _nodes(8, 64, 0)

    assignment = rollout_worker_nodes([small, large, cpu_less], num_workers=4, cpus_per_worker=8)

    assert Counter(assignment) == {small.node_id: 1, large.node_id: 3}


def test_a_single_node_hosts_every_worker():
    (node,) = _nodes(32)

    assert rollout_worker_nodes([node], num_workers=4, cpus_per_worker=8) == [node.node_id] * 4


def test_workers_beyond_the_cluster_cpus_are_rejected():
    with pytest.raises(RolloutWorkerPlacementError):
        rollout_worker_nodes(_nodes(16, 8), num_workers=4, cpus_per_worker=8)


@ray.remote(num_cpus=0)
class _NodeProbe:
    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


def _resources(num_workers: int, placement_timeout_seconds: float) -> RolloutWorkerResources:
    return RolloutWorkerResources(
        num_workers=num_workers,
        cpus_per_worker=1,
        executor_threads=1,
        progress_timeout_seconds=60,
        start_interval_seconds=0,
        placement_timeout_seconds=placement_timeout_seconds,
    )


@pytest.fixture(scope="module")
def three_node_cluster() -> Iterator[Cluster]:
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 2, "include_dashboard": False})
    for _ in range(2):
        cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    try:
        yield cluster
    finally:
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.slow
def test_reserved_workers_run_on_their_assigned_nodes(three_node_cluster):
    resources = _resources(num_workers=6, placement_timeout_seconds=60)
    expected = rollout_worker_nodes(live_ray_nodes(), resources.num_workers, resources.cpus_per_worker)

    group = asyncio.run(reserve_rollout_workers(resources))
    try:
        probes = [
            _NodeProbe.options(
                num_cpus=1, scheduling_strategy=PlacementGroupSchedulingStrategy(group, placement_group_bundle_index=i)
            ).remote()
            for i in range(resources.num_workers)
        ]
        placed = ray.get([probe.node_id.remote() for probe in probes])
    finally:
        remove_placement_group(group)

    assert placed == expected
    assert len(set(placed)) == 3


@pytest.mark.slow
def test_reservation_fails_when_a_planned_node_has_no_free_cpus(three_node_cluster):
    # Another tenant holds every CPU on one node, so the CPU totals admit a one-worker-per-node plan that Ray
    # cannot grant.
    busy_node = live_ray_nodes()[0].node_id
    tenant = placement_group([{"CPU": 2}], bundle_label_selector=[{NODE_ID_LABEL: busy_node}])
    ray.get(tenant.ready(), timeout=60)
    try:
        with pytest.raises(RolloutWorkerPlacementError):
            asyncio.run(reserve_rollout_workers(_resources(num_workers=3, placement_timeout_seconds=2)))
    finally:
        remove_placement_group(tenant)
