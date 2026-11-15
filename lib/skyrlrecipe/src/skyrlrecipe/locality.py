# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack model GPU requirements into groups that each fit one topology domain."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class RoleTopologyMode(StrEnum):
    NVLINK_DOMAIN = "nvlink.domain"


@dataclass(frozen=True)
class LocalityPlan:
    group_sizes: tuple[int, ...]
    role_groups: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return sum(self.group_sizes)

    def group_for_rank(self, rank: int) -> int:
        if rank < 0:
            raise ValueError(f"Invalid task rank: {rank}")
        for index, size in enumerate(self.group_sizes):
            if rank < size:
                return index
            rank -= size
        raise ValueError("Task rank exceeds the locality allocation")


def plan_locality_groups(role_gpus: Sequence[int], *, gpus_per_node: int, max_nodes_per_domain: int) -> LocalityPlan:
    """Place roles in order, reusing space in the first domain that fits.

    Put the trainer first, followed by equally sized inference engines. Each
    domain rounds its GPU demand up to whole hosts independently.
    """
    capacity = gpus_per_node * max_nodes_per_domain
    occupied: list[int] = []
    role_groups = []
    for count in role_gpus:
        if not 0 < count <= capacity:
            raise ValueError(f"A local model group needs {count} GPUs; one domain holds {capacity}")
        group = next((index for index, used in enumerate(occupied) if used + count <= capacity), len(occupied))
        if group == len(occupied):
            occupied.append(0)
        occupied[group] += count
        role_groups.append(group)
    return LocalityPlan(
        group_sizes=tuple(-(-count // gpus_per_node) for count in occupied),
        role_groups=tuple(role_groups),
    )


def plan_nvlink_roles(
    *,
    policy_gpus: int,
    num_inference_engines: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    inference_backend: str,
    colocate_all: bool,
    colocate_policy_ref: bool,
    gpus_per_node: int,
    max_nodes_per_domain: int,
) -> LocalityPlan:
    """Validate the supported role geometry and count whole hosts per domain."""
    if colocate_all or not colocate_policy_ref:
        raise ValueError("NVLink role placement requires separate policy and inference GPUs with a shared reference")
    if num_inference_engines and (
        inference_backend != "vllm"
        or tensor_parallel_size != 1
        or pipeline_parallel_size != 1
        or expert_parallel_size != data_parallel_size
    ):
        raise ValueError("NVLink role placement requires vLLM TP1/PP1 EP=DP inference engines")
    return plan_locality_groups(
        [policy_gpus] + [data_parallel_size] * num_inference_engines,
        gpus_per_node=gpus_per_node,
        max_nodes_per_domain=max_nodes_per_domain,
    )
