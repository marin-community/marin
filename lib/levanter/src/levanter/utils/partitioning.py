# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax

from levanter.utils.types import ResourceMapping

from haliax.partitioning import (
    current_thread_local_mapping,
    infer_resource_partitions,
    named_jit,
    physical_axis_name,
    physical_axis_size,
    pspec_for,
    pspec_for_axis,
    round_axis_for_partitioning,
    sharding_for_axis,
    shard_map,
)


def round_vocab_axis_for_partitioning(vocab_size: int, axis_mapping: ResourceMapping | None):
    """Create and round a vocab axis to match the active partitioning constraints."""
    return round_axis_for_partitioning(hax.Axis("vocab", vocab_size), axis_mapping)


__all__ = [
    "current_thread_local_mapping",
    "infer_resource_partitions",
    "named_jit",
    "physical_axis_name",
    "physical_axis_size",
    "pspec_for",
    "pspec_for_axis",
    "round_vocab_axis_for_partitioning",
    "round_axis_for_partitioning",
    "sharding_for_axis",
    "shard_map",
]
