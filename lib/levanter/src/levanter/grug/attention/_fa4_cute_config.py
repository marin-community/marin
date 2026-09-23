# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum

# Query heads per KV head validated for the native SM100 forward and backward kernels.
SM100_GQA_RATIOS = (4, 6, 8)
# Q/K and V head dimension supported by the native SM100 kernels.
SM100_HEAD_DIM = 128


def runs_sm100_kernels(arch: int) -> bool:
    """Return whether upstream FA4 runs its SM100 kernels on this compute capability (any 10.x)."""
    return arch // 10 == 10


@dataclass(frozen=True)
class Flash4CuteSm100ForwardConfig:
    """Upstream Blackwell tile and number of pipelined query stages."""

    tile: tuple[int, int]
    q_stage: int


@dataclass(frozen=True)
class Flash4CuteSm100BackwardConfig:
    """Validated native Blackwell backward schedule."""

    tile: tuple[int, int]
    zero_fill_threads: int
    postprocess_threads: int


class Sm90BackwardSchedule(StrEnum):
    DENSE = "dense"
    CAUSAL_OR_LOCAL = "causal_or_local"


@dataclass(frozen=True)
class Flash4CuteSm90BackwardConfig:
    """Native Hopper backward schedule target from upstream FA4."""

    tile: tuple[int, int]
    num_threads: int
    num_stages_q: int
    num_stages_do: int
    num_stages_pds: int
    sdp_swap_ab: bool
    dkv_swap_ab: bool
    dq_swap_ab: bool
    atom_layout_m_sdp: int
    atom_layout_n_dkv: int
    atom_layout_m_dq: int
    num_warp_groups: int
    dq_single_wg: bool = False


@dataclass(frozen=True)
class Flash4CuteKernelConfig:
    forward_tile: tuple[int, int]
    backward_tile: tuple[int, int]
    num_threads: int
    backward_arch: int | None = None
    sm90_backward: Flash4CuteSm90BackwardConfig | None = None
    sm100_backward: Flash4CuteSm100BackwardConfig | None = None
    sm100_forward: Flash4CuteSm100ForwardConfig | None = None


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
            backward_arch=120,
        )
    if arch_family == 12:
        return Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
            backward_arch=120,
        )
    if arch_family == 8:
        return Flash4CuteKernelConfig(
            forward_tile=(128, 64),
            backward_tile=(128, 64),
            num_threads=128,
            backward_arch=80,
        )
    if arch_family == 9:
        sm90_backward = sm90_flash4_cute_backward_config(
            head_dim,
            head_dim_v=head_dim,
            schedule=Sm90BackwardSchedule.CAUSAL_OR_LOCAL,
        )
        return Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
            backward_arch=90,
            sm90_backward=sm90_backward,
        )
    raise NotImplementedError(f"FA4/CuTe attention does not support SM{arch}.")


def sm100_flash4_cute_kernel_config() -> Flash4CuteKernelConfig:
    """Return the native SM100 forward and backward configuration for BF16 D128 GQA."""
    # The port fields are required by Flash4CuteKernelConfig but unused: the backend always runs
    # sm100_forward and sm100_backward and rejects layouts they do not support.
    return Flash4CuteKernelConfig(
        forward_tile=(128, 64),
        backward_tile=(64, 64),
        num_threads=128,
        backward_arch=100,
        sm100_forward=Flash4CuteSm100ForwardConfig(tile=(128, 128), q_stage=2),
        sm100_backward=Flash4CuteSm100BackwardConfig(
            tile=(128, 128),
            zero_fill_threads=512,
            postprocess_threads=128,
        ),
    )


def sm90_flash4_cute_backward_config(
    head_dim: int,
    *,
    head_dim_v: int | None = None,
    schedule: Sm90BackwardSchedule,
    sparse_block_size_q: int | None = None,
) -> Flash4CuteSm90BackwardConfig:
    """Return the upstream FA4 SM90 backward schedule target."""
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    if head_dim <= 64:
        return Flash4CuteSm90BackwardConfig(
            tile=(128, 128),
            num_threads=384,
            num_stages_q=2,
            num_stages_do=2,
            num_stages_pds=2,
            sdp_swap_ab=True,
            dkv_swap_ab=False,
            dq_swap_ab=False,
            atom_layout_m_sdp=1,
            atom_layout_n_dkv=2,
            atom_layout_m_dq=2,
            num_warp_groups=2,
        )
    if head_dim <= 96:
        return Flash4CuteSm90BackwardConfig(
            tile=(64, 128),
            num_threads=384,
            num_stages_q=2,
            num_stages_do=2,
            num_stages_pds=2,
            sdp_swap_ab=True,
            dkv_swap_ab=False,
            dq_swap_ab=False,
            atom_layout_m_sdp=1,
            atom_layout_n_dkv=2,
            atom_layout_m_dq=1,
            num_warp_groups=2,
            dq_single_wg=True,
        )
    if head_dim <= 128:
        tile_m = 64 if schedule == Sm90BackwardSchedule.CAUSAL_OR_LOCAL else 80
        if sparse_block_size_q is not None and sparse_block_size_q % tile_m != 0:
            tile_m = 64
        return Flash4CuteSm90BackwardConfig(
            tile=(tile_m, 128),
            num_threads=384,
            num_stages_q=2,
            num_stages_do=2,
            num_stages_pds=2,
            sdp_swap_ab=True,
            dkv_swap_ab=False,
            dq_swap_ab=tile_m % 64 != 0,
            atom_layout_m_sdp=1,
            atom_layout_n_dkv=2,
            atom_layout_m_dq=1,
            num_warp_groups=2,
        )
    if head_dim <= 192:
        return Flash4CuteSm90BackwardConfig(
            tile=(64, 96),
            num_threads=384,
            num_stages_q=2,
            num_stages_do=2 if head_dim_v <= 128 else 1,
            num_stages_pds=1,
            sdp_swap_ab=False,
            dkv_swap_ab=True,
            dq_swap_ab=False,
            atom_layout_m_sdp=1,
            atom_layout_n_dkv=2,
            atom_layout_m_dq=1,
            num_warp_groups=2,
        )
    return Flash4CuteSm90BackwardConfig(
        tile=(64, 64),
        num_threads=384,
        num_stages_q=1,
        num_stages_do=1,
        num_stages_pds=1,
        sdp_swap_ab=False,
        dkv_swap_ab=False,
        dq_swap_ab=False,
        atom_layout_m_sdp=1,
        atom_layout_n_dkv=1,
        atom_layout_m_dq=1,
        num_warp_groups=2,
    )
