# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class Flash4CuteKernelConfig:
    forward_tile: tuple[int, int]
    backward_tile: tuple[int, int]
    num_threads: int


def flash4_cute_kernel_config(
    head_dim: int,
    *,
    arch: int,
) -> Flash4CuteKernelConfig:
    arch_family = arch // 10
    if arch_family == 10:
        return Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
        )
    if arch_family == 12:
        return Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
        )
    if arch_family == 8:
        return Flash4CuteKernelConfig(
            forward_tile=(128, 64),
            backward_tile=(128, 64),
            num_threads=128,
        )
    if arch_family == 9:
        return Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(128, 64),
            num_threads=128,
        )
    raise NotImplementedError(f"FA4/CuTe attention does not support SM{arch}.")
