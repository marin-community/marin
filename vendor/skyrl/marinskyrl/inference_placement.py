"""Where each vLLM inference worker runs, as the worker itself reports it.

Each worker reports its host, GPU UUID and ranks. The engine factory checks the reports
against the bundles Ray allocated, including a single serving worker for expert-block sync.
This module imports nothing from the trainer, because config validation imports it before
any model module.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marinskyrl.runtime_options import WeightSyncTransport


def validate_expert_block_transport(config: Mapping[str, Any]) -> None:
    """Reject ``expert_block`` unless the config meets its requirements.

    It needs the megatron strategy at TP=1 and ETP=1, local TP=1 vLLM engines with EP=DP>=1, and
    the NCCL weight-sync backend.
    """
    generator = config["generator"]
    transport = generator["weight_sync_transport"]
    choices = [item.value for item in WeightSyncTransport]
    if transport not in choices:
        raise ValueError(f"generator.weight_sync_transport must be one of {choices}, not {transport!r}")
    if transport != WeightSyncTransport.EXPERT_BLOCK:
        return
    trainer = config["trainer"]
    problems = []
    if trainer["strategy"] != "megatron":
        problems.append("the policy must train with the megatron strategy")
    else:
        megatron = trainer["policy"]["megatron_config"]
        if megatron["tensor_model_parallel_size"] != 1:
            problems.append("the policy must use tensor_model_parallel_size 1")
        if megatron["expert_tensor_parallel_size"] not in (None, 1):
            problems.append("the policy must use expert_tensor_parallel_size 1")
        # Trainer and engine EP sizes may differ. Whether each divides the expert count is
        # checked when the ranks report.
        if megatron["expert_model_parallel_size"] < 1 or generator["inference_engine_expert_parallel_size"] < 1:
            problems.append("expert-parallel sizes must be positive")
    if generator["weight_sync_backend"] != "nccl":
        problems.append("generator.weight_sync_backend must be nccl")
    # The transport matches trainer ranks to the checked worker placements, which the engine
    # factory records for local TP=1 vLLM engines, including DP=1.
    if generator["backend"] != "vllm" or not generator["run_engines_locally"] or trainer["placement"]["colocate_all"]:
        problems.append("the engines must be local, non-colocated vLLM engines")
    tp_pp_size = (
        generator["inference_engine_tensor_parallel_size"] * generator["inference_engine_pipeline_parallel_size"]
    )
    if generator.get("inference_engine_mp_backend", False) and tp_pp_size > 1:
        problems.append("the engines must use the Ray executor, not the mp backend")
    if generator["inference_engine_tensor_parallel_size"] != 1:
        problems.append("the engines must use TP=1")
    if generator["inference_engine_data_parallel_size"] < 1:
        problems.append("the engines must use positive DP")
    if generator["inference_engine_expert_parallel_size"] != generator["inference_engine_data_parallel_size"]:
        problems.append("the engines must use EP equal to DP")
    if int(generator["expert_block_sync"]["timeout_seconds"]) <= 0:
        problems.append("generator.expert_block_sync.timeout_seconds must be positive")
    if problems:
        raise ValueError("generator.weight_sync_transport=expert_block requires: " + "; ".join(problems))


@dataclass(frozen=True)
class InferenceWorkerPlacement:
    """What one vLLM worker reports about itself."""

    host: str
    gpu_uuid: str
    dp_rank: int
    dp_world_size: int
    ep_rank: int
    ep_world_size: int
    torch_rank: int
    torch_world_size: int
    pp_rank: int = 0
    pp_world_size: int = 1


@dataclass(frozen=True)
class InferenceReplicaPlacement:
    """A worker's report joined to its replica and placement bundle."""

    replica: int
    node_id: str
    bundle_index: int
    worker: InferenceWorkerPlacement
    weight_receiver_rank: int


def validate_inference_replica_topology(
    placements: Sequence[InferenceReplicaPlacement],
    *,
    num_replicas: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    pipeline_parallel_size: int = 1,
    node_hosts: Mapping[str, str],
) -> None:
    """Check each worker against its bundle, GPU identity and requested parallel groups.

    A worker's bundle index is ``dp_rank * PP + pp_rank``, which is also its torch rank in the engine.
    """
    per_replica = data_parallel_size * pipeline_parallel_size
    total = num_replicas * per_replica
    if len(placements) != total:
        raise ValueError(f"Expected {total} inference workers, observed {len(placements)}")
    if {row.replica for row in placements} != set(range(num_replicas)):
        raise ValueError("Inference replica indices are incomplete")
    gpu_uuids = {row.worker.gpu_uuid for row in placements}
    if len(gpu_uuids) != total or "" in gpu_uuids:
        raise ValueError("Inference workers must have distinct, nonempty physical GPU UUIDs")
    if {row.weight_receiver_rank for row in placements} != set(range(1, total + 1)):
        raise ValueError("Inference weight receiver ranks must be unique and cover the broadcast group")
    # A dense model has no EP group: every worker reports size 1.
    expert_parallel = expert_parallel_size > 1 and any(row.worker.ep_world_size != 1 for row in placements)
    cross_node_ep = expert_parallel and pipeline_parallel_size == 1 and expert_parallel_size == data_parallel_size
    expected_ep_size = data_parallel_size if expert_parallel else 1
    for replica in range(num_replicas):
        rows = [row for row in placements if row.replica == replica]
        if len(rows) != per_replica or {row.bundle_index for row in rows} != set(range(per_replica)):
            raise ValueError(f"Inference replica {replica} has incomplete placement bundles")
        for stage in range(pipeline_parallel_size):
            stage_rows = [row for row in rows if row.worker.pp_rank == stage]
            if not cross_node_ep and (
                len({row.node_id for row in stage_rows}) != 1 or len({row.worker.host for row in stage_rows}) != 1
            ):
                raise ValueError(f"Inference replica {replica} stage {stage} spans nodes")
        for row in rows:
            worker = row.worker
            if node_hosts.get(row.node_id) != worker.host:
                raise ValueError(f"Inference replica {replica} worker host disagrees with its placement node")
            if worker.pp_world_size != pipeline_parallel_size or not 0 <= worker.pp_rank < pipeline_parallel_size:
                raise ValueError(f"Inference replica {replica} has an unexpected PP rank or world size")
            expected_rank = worker.dp_rank * pipeline_parallel_size + worker.pp_rank
            if row.bundle_index != expected_rank or worker.torch_rank != expected_rank:
                raise ValueError(f"Inference replica {replica} worker ranks disagree with its placement bundle")
            if worker.dp_world_size != data_parallel_size or worker.torch_world_size != per_replica:
                raise ValueError(f"Inference replica {replica} has an unexpected DP/torch world size")
            expected_ep_rank = worker.dp_rank if expert_parallel else 0
            if worker.ep_world_size != expected_ep_size or worker.ep_rank != expected_ep_rank:
                raise ValueError(f"Inference replica {replica} has an unexpected EP rank or world size")
            if row.weight_receiver_rank != 1 + replica * per_replica + worker.torch_rank:
                raise ValueError(f"Inference replica {replica} has an incorrect weight receiver rank")
