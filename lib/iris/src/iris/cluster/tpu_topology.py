# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU topology table and lookup helpers.

Leaf module shared by :mod:`iris.cluster.types` and
:mod:`iris.cluster.constraints` so the topology table can be read from
either without inducing a top-level import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TpuTopologyInfo:
    """TPU topology configuration."""

    name: str
    chip_count: int
    host_count: int
    vm_count: int
    chips_per_vm: int


TPU_TOPOLOGIES: list[TpuTopologyInfo] = [
    # https://cloud.google.com/tpu/docs/v4
    TpuTopologyInfo("v4-8", 4, 1, 1, 4),
    TpuTopologyInfo("v4-16", 8, 2, 2, 4),
    TpuTopologyInfo("v4-32", 16, 4, 4, 4),
    TpuTopologyInfo("v4-64", 32, 8, 8, 4),
    TpuTopologyInfo("v4-128", 64, 16, 16, 4),
    TpuTopologyInfo("v4-256", 128, 32, 32, 4),
    TpuTopologyInfo("v4-512", 256, 64, 64, 4),
    TpuTopologyInfo("v4-1024", 512, 128, 128, 4),
    TpuTopologyInfo("v4-2048", 1024, 256, 256, 4),
    TpuTopologyInfo("v4-4096", 2048, 512, 512, 4),
    # https://cloud.google.com/tpu/docs/v5e
    TpuTopologyInfo("v5litepod-1", 1, 1, 1, 1),
    TpuTopologyInfo("v5litepod-2", 2, 1, 1, 2),
    TpuTopologyInfo("v5litepod-4", 4, 1, 1, 4),
    TpuTopologyInfo("v5litepod-8", 8, 1, 1, 8),
    TpuTopologyInfo("v5litepod-16", 16, 2, 4, 4),
    TpuTopologyInfo("v5litepod-32", 32, 4, 8, 4),
    TpuTopologyInfo("v5litepod-64", 64, 8, 16, 4),
    TpuTopologyInfo("v5litepod-128", 128, 16, 32, 4),
    TpuTopologyInfo("v5litepod-256", 256, 32, 64, 4),
    # https://cloud.google.com/tpu/docs/v5p
    TpuTopologyInfo("v5p-8", 4, 1, 1, 4),
    TpuTopologyInfo("v5p-16", 8, 2, 2, 4),
    TpuTopologyInfo("v5p-32", 16, 4, 4, 4),
    TpuTopologyInfo("v5p-64", 32, 8, 8, 4),
    TpuTopologyInfo("v5p-128", 64, 16, 16, 4),
    TpuTopologyInfo("v5p-256", 128, 32, 32, 4),
    TpuTopologyInfo("v5p-512", 256, 64, 64, 4),
    TpuTopologyInfo("v5p-1024", 512, 128, 128, 4),
    TpuTopologyInfo("v5p-2048", 1024, 256, 256, 4),
    TpuTopologyInfo("v5p-4096", 2048, 512, 512, 4),
    TpuTopologyInfo("v5p-8192", 4096, 1024, 1024, 4),
    TpuTopologyInfo("v5p-12288", 6144, 1536, 1536, 4),
    # https://cloud.google.com/tpu/docs/v6e
    TpuTopologyInfo("v6e-1", 1, 1, 1, 1),
    TpuTopologyInfo("v6e-4", 4, 1, 1, 4),
    TpuTopologyInfo("v6e-8", 8, 1, 1, 8),
    TpuTopologyInfo("v6e-16", 16, 4, 4, 4),
    TpuTopologyInfo("v6e-32", 32, 8, 8, 4),
    TpuTopologyInfo("v6e-64", 64, 16, 16, 4),
    TpuTopologyInfo("v6e-128", 128, 32, 32, 4),
    TpuTopologyInfo("v6e-256", 256, 64, 64, 4),
]


TPU_FAMILY_VARIANT_PREFIX: dict[str, str] = {
    "v4": "v4",
    "v5e": "v5litepod",
    "v5p": "v5p",
    "v6e": "v6e",
}


def tpu_variant_name(family: str, size: int) -> str:
    """Build the device_variant string for a TPU family and chip-count size.

    >>> tpu_variant_name("v5e", 16)
    'v5litepod-16'
    >>> tpu_variant_name("v6e", 32)
    'v6e-32'
    """
    prefix = TPU_FAMILY_VARIANT_PREFIX.get(family)
    if prefix is None:
        raise ValueError(f"Unknown TPU family '{family}'. Known families: {sorted(TPU_FAMILY_VARIANT_PREFIX)}")
    return f"{prefix}-{size}"


def get_tpu_topology(tpu_type: str) -> TpuTopologyInfo:
    """Get TPU topology by type name."""
    for config in TPU_TOPOLOGIES:
        if config.name == tpu_type:
            return config
    raise ValueError(f"Unknown TPU type: {tpu_type}")
