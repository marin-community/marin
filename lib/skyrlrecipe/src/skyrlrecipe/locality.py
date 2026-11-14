# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack model GPU requirements into groups that each fit one topology domain."""

from collections.abc import Sequence
from dataclasses import dataclass


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
