from collections import Counter
from collections.abc import Mapping, Sequence
import socket

import torch
from ray.util.placement_group import PlacementGroup, placement_group_table

from marinskyrl.inference_placement import (
    InferenceReplicaPlacement,
    InferenceWorkerPlacement,
    validate_inference_replica_topology,
)
from skyrl_train.utils import get_reordered_bundle_indices
from skyrl_train.utils.placement_geometry import colocated_engine_bundle_indices


def inference_worker_placement(
    *,
    dp_rank: int,
    dp_world_size: int,
    ep_rank: int,
    ep_world_size: int,
    pp_rank: int = 0,
    pp_world_size: int = 1,
) -> InferenceWorkerPlacement:
    """The calling worker's host, GPU UUID and ranks."""
    return InferenceWorkerPlacement(
        host=socket.gethostname(),
        gpu_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        ep_rank=ep_rank,
        ep_world_size=ep_world_size,
        torch_rank=torch.distributed.get_rank(),
        torch_world_size=torch.distributed.get_world_size(),
        pp_rank=pp_rank,
        pp_world_size=pp_world_size,
    )


def inference_bundle_nodes(
    placement_groups: Sequence[PlacementGroup],
    *,
    data_parallel_size: int,
    node_gpu_capacities: Mapping[str, int],
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
) -> list[list[str]]:
    """Return each replica's bundle nodes and check placement completeness and GPU capacity.

    Bundle ``dp * PP + pp`` of a replica's group belongs to stage ``pp`` of data-parallel rank ``dp``.
    Each stage must fit on one node, except TP=PP=1 replicas with EP=DP>1.
    """
    cross_node_ep = pipeline_parallel_size == 1 and expert_parallel_size == data_parallel_size > 1
    bundle_nodes = []
    per_replica = data_parallel_size * pipeline_parallel_size
    for replica, pg in enumerate(placement_groups):
        bundles = placement_group_table(pg)["bundles_to_node_id"]
        if set(bundles) != set(range(per_replica)):
            raise ValueError(f"Inference replica {replica} has incomplete placement bundles")
        if not cross_node_ep:
            for stage in range(pipeline_parallel_size):
                on_stage = {bundles[dp * pipeline_parallel_size + stage] for dp in range(data_parallel_size)}
                if len(on_stage) != 1:
                    raise ValueError(f"Inference replica {replica} stage {stage} placement spans nodes")
        bundle_nodes.append([bundles[index] for index in range(per_replica)])
    for node_id, count in Counter(node for nodes in bundle_nodes for node in nodes).items():
        if not node_id or count > node_gpu_capacities.get(node_id, 0):
            raise ValueError(f"Inference replicas exceed GPU capacity on placement node {node_id}")
    return bundle_nodes


def verified_inference_replica_placements(
    reports: Sequence[Sequence[Mapping[str, str | int]]],
    *,
    bundle_nodes: Sequence[Sequence[str]],
    node_hosts: Mapping[str, str],
    relative_rank_offsets: Sequence[int],
    data_parallel_size: int,
    expert_parallel_size: int,
    pipeline_parallel_size: int = 1,
) -> list[InferenceReplicaPlacement]:
    """Match each worker's report to its bundle and check the topology.

    ``reports`` has one list per data-parallel actor, with one report per worker.
    """
    if len(reports) != len(bundle_nodes) * data_parallel_size or len(reports) != len(relative_rank_offsets):
        raise ValueError("Incomplete inference replica reports")
    placements = []
    for index, (report, offset) in enumerate(zip(reports, relative_rank_offsets, strict=True)):
        if len(report) != pipeline_parallel_size:
            raise ValueError(f"Expected {pipeline_parallel_size} worker reports per DP actor, got {len(report)}")
        replica, dp_rank = divmod(index, data_parallel_size)
        for row in report:
            worker = InferenceWorkerPlacement(**row)
            placements.append(
                InferenceReplicaPlacement(
                    replica=replica,
                    node_id=bundle_nodes[replica][dp_rank * pipeline_parallel_size + worker.pp_rank]
                    if 0 <= worker.pp_rank < pipeline_parallel_size
                    else "",
                    bundle_index=dp_rank * pipeline_parallel_size + worker.pp_rank,
                    worker=worker,
                    weight_receiver_rank=1 + offset + worker.torch_rank,
                )
            )
    validate_inference_replica_topology(
        placements,
        num_replicas=len(bundle_nodes),
        data_parallel_size=data_parallel_size,
        expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        node_hosts=node_hosts,
    )
    return placements


def colocated_engine_bundle_layout(
    shared_pg: PlacementGroup | None,
    *,
    num_inference_engines: int,
    data_parallel_size: int,
    tensor_pipeline_size: int,
) -> list[list[int]]:
    """Resolve node-atomic TP/PP slices, or return no slices when disaggregated."""

    if shared_pg is None:
        return []
    bundle_to_node_ids = placement_group_table(shared_pg)["bundles_to_node_id"]
    node_counts = Counter(bundle_to_node_ids.values())
    if not node_counts or len(set(node_counts.values())) != 1:
        raise ValueError(f"Colocated placement bundles must be uniform across nodes; got {dict(node_counts)}")
    gpus_per_node = next(iter(node_counts.values()))
    reordered_bundle_indices = get_reordered_bundle_indices(shared_pg)
    return [
        colocated_engine_bundle_indices(
            reordered_bundle_indices=reordered_bundle_indices,
            engine_index=engine_index,
            data_parallel_rank=data_parallel_rank,
            tensor_pipeline_size=tensor_pipeline_size,
            data_parallel_size=data_parallel_size,
            gpus_per_node=gpus_per_node,
        )
        for engine_index in range(num_inference_engines)
        for data_parallel_rank in range(data_parallel_size)
    ]
