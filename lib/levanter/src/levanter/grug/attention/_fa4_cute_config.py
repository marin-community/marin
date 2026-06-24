# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


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
            is_causal_or_local=True,
        )
        return Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
            backward_arch=90,
            sm90_backward=sm90_backward,
        )
    raise NotImplementedError(f"FA4/CuTe attention does not support SM{arch}.")


def sm90_flash4_cute_backward_config(
    head_dim: int,
    *,
    head_dim_v: int | None = None,
    is_causal_or_local: bool,
    sparse_block_size_q: int | None = None,
) -> Flash4CuteSm90BackwardConfig:
    """Return the upstream FA4 SM90 backward schedule target.

    This is the native Hopper producer/consumer WGMMA schedule we need to port
    for Grug segmented masks. It is intentionally separate from
    ``Flash4CuteKernelConfig.backward_tile`` because the production path still
    uses the SM120-compatible segmented fallback until this schedule is ported.
    """
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
        tile_m = 64 if is_causal_or_local else 80
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
