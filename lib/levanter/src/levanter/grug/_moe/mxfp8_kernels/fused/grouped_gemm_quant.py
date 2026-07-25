# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# ruff: noqa

"""
MoE Block-Scaled Grouped GEMM Kernel with Quantization Support.

Supports:
    - Static / Dynamic persistent tile scheduling (MoEPersistentTileScheduler)
    - Dense (contiguous 3-D B) / Discrete (per-expert pointer array B) weight layout
    - FP8/FP4 output quantization with row/column scale factors (SFD)
    - Optional bias and routing-probability (prob) fusion
    - AMAX reduction for FP8 calibration

This module contains only the kernel class.
MoE scheduler components live in moe_persistent_scheduler.py / moe_sched_extension.py / moe_utils.py.
"""

from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Float32, Int32, AddressSpace
from .moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from .moe_utils import (
    compute_expert_token_range,
    MoEWeightMode,
    TensormapWorkspace,
    store_tma_desc,
)
from .moe_sched_extension import (
    DiscreteWeightScaledGemmSchedExtension,
    ContiguousAndConsistentGroupedGemmSchedExtension,
)
from .moe_kernel_helpers import (
    fmin,
    fmax,
    atomic_max_float32,
    compute_stages,
    compute_grid,
    can_implement,
    amax_reduction_per_thread,
    epilog_gmem_copy_and_partition,
    get_dtype_rcp_limits,
)


class BlockScaledMoEGroupedGemmQuantKernel:
    """Block-scaled grouped GEMM kernel with MoE tile scheduling and quantization.

    Supports both dense and discrete weight layouts, static and dynamic
    scheduling, and quantized output with row/column scale factors.

    :param sf_vec_size: Scale-factor vector size (16 or 32).
    :param acc_dtype: Accumulator data type (Float32).
    :param use_2cta_instrs: Use 2-CTA MMA instructions.
    :param mma_tiler_mn: MMA tile shape (M, N).
    :param cluster_shape_mn: Cluster shape (M, N).
    :param vectorized_f32: Use packed FP32 arithmetic.
    :param generate_sfd: Generate output scale factors.
    :param discrete_col_sfd: Use discrete column SFD layout.
    :param enable_bias: Fuse bias addition.
    :param expert_cnt: Number of experts.
    :param weight_mode: ``MoEWeightMode.DENSE`` or ``MoEWeightMode.DISCRETE``.
    :param use_dynamic_sched: Enable dynamic tile scheduling.
    """

    FIX_PAD_SIZE = 256

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        cd_major: str,
        m_aligned: int,
    ) -> bool:
        return can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            acc_dtype,
            d_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            cd_major,
            m_aligned,
            fix_pad_size=BlockScaledMoEGroupedGemmQuantKernel.FIX_PAD_SIZE,
        )

    def __init__(
        self,
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        generate_sfd: bool,
        discrete_col_sfd: bool,
        enable_bias: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        use_dynamic_sched: bool = False,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(
                f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}). " f"Supported mma_tiler_mn[0] values: 128, 256."
            )
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")

        self.sf_vec_size = sf_vec_size
        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.bias_load_warp_id = 7 if enable_bias else None
        self.threads_per_warp = 32
        all_warps = [
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_warp_id,
            self.sched_warp_id,
        ]
        warps_wo_sched = [*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id]
        if enable_bias:
            all_warps.append(self.bias_load_warp_id)
            warps_wo_sched.append(self.bias_load_warp_id)
        self.threads_per_cta = self.threads_per_warp * len(all_warps)
        self.threads_wo_sched = self.threads_per_warp * len(warps_wo_sched)

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.vectorized_f32 = vectorized_f32
        self.generate_sfd = generate_sfd
        self.discrete_col_sfd = discrete_col_sfd
        self.enable_bias = enable_bias

        self.weight_mode = weight_mode
        self.use_dynamic_sched = use_dynamic_sched

        self.epilogue_use_functor = False

        self.num_epilog_warps = len(self.epilog_warp_id)

    # ------------------------------------------------------------------
    # _setup_attributes
    # ------------------------------------------------------------------

    def _setup_attributes(self):
        """Configure MMA / tile / stage / SMEM layouts from GEMM inputs."""

        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.mma_tiler_d = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk_d = (
            self.mma_tiler_d[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_d[1],
            self.mma_tiler_d[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = (128, 32)

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_d_stage,
            self.num_tile_stage,
            self.num_bias_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.d_dtype,
            self.d_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
            self.generate_sfd,
            self.bias_dtype if self.enable_bias else None,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage,
        )

        if self.enable_bias:
            self.bias_smem_layout_staged = cute.make_layout(
                (self.mma_tiler[1], self.num_bias_stage),
                stride=(1, self.mma_tiler[1]),
            )
        else:
            self.bias_smem_layout_staged = cute.make_layout((1, 1))

        self.overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256
        self.epilogue_prefetch_more = False

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.epi_tile_n_required = cute.size(self.epi_tile[1])
        self.iter_acc_early_release_in_epilogue = (self.num_sf_tmem_cols + self.epi_tile_n_required - 1) // self.epi_tile_n_required - 1

    # ------------------------------------------------------------------
    # _compute_stages (with bias support)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        d_dtype,
        d_layout,
        sf_dtype,
        sf_vec_size,
        num_smem_capacity,
        occupancy,
        generate_sfd,
        bias_dtype,
    ):
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_d_stage = 2 if generate_sfd else 1
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        d_bytes = d_bytes_per_stage * num_d_stage * (2 if generate_sfd else 1)
        amax_bytes = 4 * cute.size_in_bytes(cutlass.Float32, cute.make_layout((1,))) if d_dtype == cutlass.BFloat16 else 0

        if bias_dtype is not None:
            num_bias_stage = 2
            bias_epi_tile_n = mma_tiler_mnk[1]
            bias_bytes = bias_epi_tile_n * num_bias_stage * (bias_dtype.width // 8)
        else:
            num_bias_stage = 0
            bias_bytes = 0

        epi_bytes = d_bytes + amax_bytes + bias_bytes
        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_d_stage, num_tile_stage, num_bias_stage

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def get_desc_workspace_bytes(self) -> int:
        if self.weight_mode == MoEWeightMode.DISCRETE:
            from .moe_utils import DiscreteWeightTensormapConstructor

            return DiscreteWeightTensormapConstructor.get_workspace_size(self.expert_cnt)
        return 0

    def get_workspace_bytes(self) -> int:
        desc_workspace_bytes = self.get_desc_workspace_bytes()
        dynamic_sched_bytes = 4 if self.use_dynamic_sched else 0
        return desc_workspace_bytes + dynamic_sched_bytes

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes()
        return cute.make_ptr(
            cutlass.Int32,
            counter_addr,
            AddressSpace.gmem,
            assumed_align=4,
        )

    # ------------------------------------------------------------------
    # helper_kernel: pre-main-kernel initialization
    #   - discrete weight: build per-expert B/SFB TMA descriptors
    #   - dynamic sched: reset the atomic tile counter
    # ------------------------------------------------------------------

    @cute.kernel
    def helper_kernel(
        self,
        # Discrete-only params (unused in dense mode, but must be present for signature)
        ptrs_b: cute.Pointer,
        ptrs_sfb: cute.Pointer,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        tiled_mma_arg: cute.TiledMma,
        tiled_mma_sfb_arg: cute.TiledMma,
        b_smem_layout_arg,
        sfb_smem_layout_arg,
        cluster_layout_vmnk_shape_arg: cutlass.Constexpr,
        cluster_layout_sfb_vmnk_shape_arg: cutlass.Constexpr,
    ):
        """Pre-main-kernel initialization.

        Launched with grid=(expert_cnt, 1, 1) for discrete mode, or
        grid=(1, 1, 1) for dense+dynamic mode.

        Discrete weight: each block builds B/SFB TMA descriptors for one expert.
        Dynamic sched: block 0 resets the atomic tile counter to 0.
        """
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            b_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_arg.thr_id)
            sfb_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma_arg.thr_id)

            b_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_b.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
            )
            sfb_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_sfb.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
            )

            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                stride_n = b_stride_size
                stride_k = c1_64
            else:
                stride_n = c1_64
                stride_k = b_stride_size

            b_ptr_val = b_ptr_tensor[expert_idx]
            b_ptr = cute.make_ptr(self.b_dtype, b_ptr_val, AddressSpace.gmem)
            b_tensor_i = cute.make_tensor(
                b_ptr,
                cute.make_layout((n, k, c1), stride=(stride_n, stride_k, c0)),
            )
            tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_tma_op_arg,
                b_tensor_i,
                b_smem_layout_arg,
                self.mma_tiler,
                tiled_mma_arg,
                cluster_layout_vmnk_shape_arg,
            )
            workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            store_tma_desc(tma_atom_b, workspace.get_ptr("b", expert_idx))

            sfb_ptr_val = sfb_ptr_tensor[expert_idx]
            sfb_ptr = cute.make_ptr(self.sf_dtype, sfb_ptr_val, AddressSpace.gmem)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb_tensor_i = cute.make_tensor(sfb_ptr, sfb_layout)
            tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_tma_op_arg,
                sfb_tensor_i,
                sfb_smem_layout_arg,
                self.mma_tiler_sfb,
                tiled_mma_sfb_arg,
                cluster_layout_sfb_vmnk_shape_arg,
                internal_type=cutlass.Uint64,
            )
            store_tma_desc(tma_atom_sfb, workspace.get_ptr("sfb", expert_idx))

        if cutlass.const_expr(self.use_dynamic_sched):
            if expert_idx == cutlass.Int32(0):
                sched_counter = cute.make_tensor(
                    self._get_sched_counter_ptr(workspace_ptr),
                    cute.make_layout(1),
                )
                sched_counter[0] = cutlass.Int32(0)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b,  # Dense: cute.Tensor (N,K,L) | Discrete: cute.Pointer to int64[]
        sfb,  # Dense: cute.Tensor         | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,
        d: cute.Tensor,
        d_col: Optional[cute.Tensor],
        sfa: cute.Tensor,
        sfd_row_tensor: Optional[cute.Tensor],
        sfd_col_tensor: Optional[cute.Tensor],
        amax_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        row_scale: Optional[cute.Tensor],
        bias: Optional[cute.Tensor],
        prob: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM.

        Dense mode: ``b`` and ``sfb`` are 3-D cute.Tensor (N, K, L).
        Discrete mode: ``b`` and ``sfb`` are cute.Pointer to device int64[]
        arrays of per-expert base addresses; ``n``, ``k``, ``b_stride_size``,
        ``b_major_mode`` describe the uniform per-expert layout.
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.d_layout = utils.LayoutEnum.from_tensor(d)
        self.bias_dtype = bias.element_type if cutlass.const_expr(self.enable_bias) else cutlass.BFloat16

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        else:
            self.b_major_mode = b_major_mode

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/B dtype must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # ---- SFA layout ----
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        # ---- B / SFB setup (mode-dependent) ----
        # Save the call-arg b/sfb before the discrete branch overwrites them
        # with template tensors.  helper_kernel needs the original Pointers.
        b_from_call_arg = b
        sfb_from_call_arg = sfb
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
            sfb = cute.make_tensor(sfb.iterator, sfb_layout)
        else:
            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                b_template_stride = (b_stride_size, c1_64, c0)
            else:
                b_template_stride = (c1_64, b_stride_size, c0)
            b_template_layout = cute.make_layout((n, k, c1), stride=b_template_stride)
            b_ptr_typed = cute.make_ptr(self.b_dtype, b.toint(), AddressSpace.gmem, assumed_align=16)
            b = cute.make_tensor(b_ptr_typed, b_template_layout)

            sfb_ptr_typed = cute.make_ptr(self.sf_dtype, sfb.toint(), AddressSpace.gmem, assumed_align=16)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb = cute.make_tensor(sfb_ptr_typed, sfb_layout)

        # ---- SFD setup ----
        self.generate_sfd = sfd_row_tensor is not None and norm_const_tensor is not None
        if cutlass.const_expr(self.generate_sfd == False):
            self.discrete_col_sfd = False
        if cutlass.const_expr(self.generate_sfd):
            sfd_row_layout = blockscaled_utils.tile_atom_to_shape_SF(d.shape, self.sf_vec_size)
            sfd_row_tensor = cute.make_tensor(sfd_row_tensor.iterator, sfd_row_layout)
            sfd_col_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
                d.shape,
                (1, 2, 3),
            )
            if cutlass.const_expr(self.discrete_col_sfd):
                sfd_col_layout = sfd_row_layout
            sfd_col_tensor = cute.make_tensor(sfd_col_tensor.iterator, sfd_col_layout)

        self.generate_amax = amax_tensor is not None

        # ---- TMA atoms ----
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d,
            d_smem_layout,
            self.epi_tile,
        )
        tma_atom_d_col, tma_tensor_d_col = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d_col,
            d_smem_layout,
            self.epi_tile,
        )

        # ---- Helper kernel: TMA desc init (discrete) + sched counter reset (dynamic) ----
        _need_helper = cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE or self.use_dynamic_sched)
        if cutlass.const_expr(_need_helper):
            _helper_grid_x = self.expert_cnt if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else 1
            _helper_args = (
                b_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                sfb_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                n if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                k if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                b_stride_size if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int64(0),
                b_major_mode if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else self.b_major_mode,
                workspace_ptr,
                tiled_mma,
                tiled_mma_sfb,
                b_smem_layout,
                sfb_smem_layout,
                self.cluster_layout_vmnk.shape,
                self.cluster_layout_sfb_vmnk.shape,
            )
            self.helper_kernel(*_helper_args).launch(
                grid=(_helper_grid_x, 1, 1),
                block=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
            )

        # ---- Grid computation via MoE scheduler ----
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            b_n, b_k, b_l = cute.shape(b)  # B is (N, K, L)
            sched_expert_shape = (self.expert_cnt, b_n, b_k)
        else:
            sched_expert_shape = (self.expert_cnt, n, k)

        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=sched_expert_shape,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )
        self.sched_params, grid = compute_grid(sched_params, max_active_clusters, self.use_2cta_instrs)

        self.buffer_align_bytes = 1024

        # ---- Shared storage ----
        sD_col_size = cute.cosize(self.d_smem_layout_staged.outer) if self.generate_sfd else 0
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            scheduler: SchedulerStorage
            if cutlass.const_expr(self.enable_bias):
                bias_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_bias_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.d_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sD_col: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, sD_col_size],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            if cutlass.const_expr(self.enable_bias):
                sBias: cute.struct.Align[
                    cute.struct.MemRange[self.bias_dtype, cute.cosize(self.bias_smem_layout_staged)],
                    16,
                ]
            sAmax: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_epilog_warps],
                4,
            ]

        self.shared_storage = SharedStorage

        # ---- Launch ----
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_d_col,
            tma_tensor_d_col,
            sfd_row_tensor,
            sfd_col_tensor,
            norm_const_tensor,
            amax_tensor,
            padded_offsets,
            alpha,
            row_scale,
            bias,
            prob,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.d_smem_layout_staged,
            self.bias_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            max_number_threads=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def amax_reduction_per_warp_and_cta(self, amax_fp32, warp_idx, amax_smem, amax_gmem):
        warp_amax = cute.arch.warp_redux_sync(
            value=amax_fp32,
            kind="fmax",
            mask_and_clamp=0xFFFFFFFF,
            nan=True,
        )
        if cute.arch.lane_idx() == 0:
            amax_smem[warp_idx] = cutlass.Float32(warp_amax)
        self.epilog_sync_barrier.arrive_and_wait()
        if warp_idx == self.epilog_warp_id[0] and cute.arch.lane_idx() == 0:
            block_amax = cutlass.Float32(0.0)
            for i in cutlass.range(self.num_epilog_warps):
                warp_amax_val = amax_smem[i]
                block_amax = cute.arch.fmax(block_amax, warp_amax_val)
            _ = atomic_max_float32(ptr=amax_gmem, value=block_amax)

    @cute.jit
    def quant_sfd_row(self, tile_idx, tiled_copy_r2s, src, pvscale, norm_const, rcp_limit, tRSrD):
        tTR_rAcc_frg = cute.logical_divide(src, cute.make_layout(self.sf_vec_size))
        acc_frg = tTR_rAcc_frg.load()
        abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
        abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
        pvscale_f32x4 = cute.make_rmem_tensor(4, cutlass.Float32)
        sfd_f8x4 = cute.make_rmem_tensor(4, self.sf_dtype)
        tmp_f32 = abs_acc_frg[None, 0].reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0) * rcp_limit * norm_const
        if tile_idx == 0:
            pvscale[0] = tmp_f32
        elif tile_idx == 1:
            pvscale[1] = tmp_f32
        elif tile_idx == 2:
            pvscale[2] = tmp_f32
        elif tile_idx == 3:
            pvscale[3] = tmp_f32
        pvscale_f32x4[0] = tmp_f32
        sfd_f8x4.store(pvscale_f32x4.load().to(self.sf_dtype))
        pvscale_f32x4.store(sfd_f8x4.load().to(cutlass.Float32))
        qpvscale_up = pvscale_f32x4[0]
        fp32_max = cutlass.Float32(3.40282346638528859812e38)
        acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
        acc_scale = fmin(acc_scale, fp32_max, nan=True)
        if cutlass.const_expr(self.vectorized_f32):
            vec = tTR_rAcc_frg[None, 0]
            for ei in cutlass.range_constexpr(0, self.sf_vec_size, 2):
                vec[ei], vec[ei + 1] = cute.arch.mul_packed_f32x2(
                    (vec[ei], vec[ei + 1]),
                    (acc_scale, acc_scale),
                    rnd="rn",
                    ftz=False,
                )
        else:
            vec = tTR_rAcc_frg[None, 0]
            for ei in cutlass.range_constexpr(self.sf_vec_size):
                vec[ei] = vec[ei] * acc_scale
        acc_vec = tiled_copy_r2s.retile(src).load()
        tRSrD.store(acc_vec.to(self.d_dtype))

    @cute.jit
    def quant_sfd_col(self, tile_idx, tiled_copy_r2s, src, pvscale, norm_const, rcp_limit, tRSrD):
        tTR_rAcc_frg = cute.logical_divide(src, cute.make_layout(self.sf_vec_size))
        acc_frg = tTR_rAcc_frg.load()
        abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
        acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
        tmp_f32 = cutlass.Float32(0.0)
        for vi in cutlass.range_constexpr(acc_frg.shape[0]):
            max_value_original = (
                cutlass.Float32(
                    cute.arch.warp_redux_sync(
                        value=acc_frg[vi, 0],
                        kind="fmax",
                        mask_and_clamp=0xFFFFFFFF,
                        nan=True,
                    )
                )
                * rcp_limit
                * norm_const
            )
            max_value_vec = cute.full(4, max_value_original, dtype=cutlass.Float32)
            max_value_vec_f8 = max_value_vec.to(cutlass.Float8E8M0FNU)
            max_value_vec_f32_chunked = max_value_vec_f8.to(cutlass.Float32)
            max_value = max_value_vec_f32_chunked[0]
            tidx = cute.arch.thread_idx()[0]
            if tidx % 32 == vi:
                tmp_f32 = max_value
            acc_scale_col = cutlass.Float32(0.0)
            if max_value_vec_f32_chunked[0] == 0.000000:
                acc_scale_col = cutlass.Float32(0.0)
            else:
                acc_scale_col = norm_const * cute.arch.rcp_approx(max_value_vec_f32_chunked[0])
            fp32_max = cutlass.Float32(3.40282346638528859812e38)
            acc_scale_col = fmin(acc_scale_col, fp32_max)
            tTR_rAcc_frg[vi] = tTR_rAcc_frg[vi] * acc_scale_col
        pvscale[None, None, tile_idx][0] = tmp_f32
        acc_vec = tiled_copy_r2s.retile(src).load()
        tRSrD.store(acc_vec.to(self.d_dtype))

    @cute.jit
    def tile_info_to_mn_idx(self, tile_info: cute.Tensor):
        m_idx = tile_info[1] * cute.size(self.cta_tile_shape_mnk[0])
        n_idx = tile_info[2] * cute.size(self.cta_tile_shape_mnk[1])
        return m_idx, n_idx

    @cute.jit
    def create_and_partition_new_SFDCol(self, tile_info, mSFDCol_mnl, padded_offsets):
        m_idx, n_idx = self.tile_info_to_mn_idx(tile_info)
        expert_idx = tile_info[0]
        cumsum_tokens, tokens_this_group = compute_expert_token_range(padded_offsets, expert_idx)
        n_total = cute.size(mSFDCol_mnl.shape[1])

        sf_tile_idx_begin = cumsum_tokens // cute.size(mSFDCol_mnl.shape[0][0])
        mSFDCol_mnl_new_ptr = mSFDCol_mnl[(None, sf_tile_idx_begin), None, 0].iterator

        sfd_col_quant_layout = cute.tile_to_shape(
            blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
            (tokens_this_group, n_total, mSFDCol_mnl.shape[2]),
            (1, 2, 3),
        )
        regPerSubtile = 4
        sfd_tile = (cute.make_layout(128), cute.make_layout(32 * regPerSubtile))
        mSFDCol_mnl_new = cute.make_tensor(mSFDCol_mnl_new_ptr, sfd_col_quant_layout)
        gSFDCol_mnl_new = cute.local_tile(mSFDCol_mnl_new, sfd_tile, (None, None, None))

        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((1,), order=(0,))
        copy_atom_sfd_col_quant = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gSFDCol_mnl_new.element_type,
            num_bits_per_copy=8,
        )
        tiled_copy_sfd_col_quant = cute.make_tiled_copy_tv(
            copy_atom_sfd_col_quant,
            thr_layout,
            val_layout,
        )
        tidx = cute.arch.thread_idx()[0]
        thr_copy_sfd_col_quant = tiled_copy_sfd_col_quant.get_slice(tidx)
        tCgSFDCol_mnl = thr_copy_sfd_col_quant.partition_D(cute.filter_zeros(gSFDCol_mnl_new))
        tCgSFDCol_mnl = cute.filter_zeros(tCgSFDCol_mnl)
        return tCgSFDCol_mnl

    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gD_mnl, epi_tile, use_2cta_instrs):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rD, tidx, sD):
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    # ------------------------------------------------------------------
    # GPU device kernel
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_d_col: cute.CopyAtom,
        mD_col_mnl: cute.Tensor,
        mSFDRow_mnl: Optional[cute.Tensor],
        mSFDCol_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        mAmax_tensor: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        row_scale: Optional[cute.Tensor],
        mBias_nl: Optional[cute.Tensor],
        prob: cute.Tensor,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        bias_smem_layout_staged: Optional[cute.Layout],
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel for persistent MoE grouped GEMM with quantization."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_d)
            if cutlass.const_expr(self.generate_sfd):
                cpasync.prefetch_descriptor(tma_atom_d_col)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        total_token = padded_offsets[self.expert_cnt - 1]

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * 1)
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_wo_sched)
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        if cutlass.const_expr(self.enable_bias):
            bias_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp)
            bias_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            bias_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.bias_mbar_ptr.data_ptr(),
                num_stages=self.num_bias_stage,
                producer_group=bias_pipeline_producer_group,
                consumer_group=bias_pipeline_consumer_group,
            )
            sBias = storage.sBias.get_tensor(bias_smem_layout_staged)

        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sD_col = sD
        if cutlass.const_expr(self.generate_sfd):
            sD_col = storage.sD_col.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        amax_layout = cute.make_layout((self.num_epilog_warps,))
        sAmax = storage.sAmax.get_tensor(amax_layout)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        # Multicast masks — must create ALL when any mcast or 2CTA is active
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        # MMA partition (for tCtAcc_fake shape computation only)
        thr_mma_common = tiled_mma.get_slice(0)
        tCsA_common = thr_mma_common.partition_A(sA)
        tCsB_common = thr_mma_common.partition_B(sB)
        tCsA_common = cute.filter_zeros(tCsA_common)
        tCsB_common = cute.filter_zeros(tCsB_common)

        # SMEM fragments for MMA (used by MMA warp)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # TMEM accumulator shape
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage_overlapped))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster sync before warp specialization
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        if total_token <= 0:
            cute.arch.nvvm.exit()

        # ==============================================================
        # Scheduler warp (MoE Persistent Tile Scheduler)
        # ==============================================================
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)
            while work_tile_info.is_valid_tile:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = work_tile_info.expert_idx
                    sInfo[(1, tile_info_producer_state.index)] = work_tile_info.tile_m_idx
                    sInfo[(2, tile_info_producer_state.index)] = work_tile_info.tile_n_idx
                    sInfo[(3, tile_info_producer_state.index)] = work_tile_info.k_tile_cnt
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # ==============================================================
        # Bias load warp
        # ==============================================================
        if cutlass.const_expr(self.enable_bias):
            if warp_idx == self.bias_load_warp_id:
                bias_ext = self._make_extension(workspace_ptr)
                bias_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_bias_stage)
                tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
                bias_g2s_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                    self.bias_dtype,
                    num_bits_per_copy=128,
                )
                bias_g2s_tiled = cute.make_tiled_copy_tv(
                    bias_g2s_atom,
                    cute.make_layout((32,)),
                    cute.make_layout((8,)),
                )
                thr_bias_g2s = bias_g2s_tiled.get_slice(cute.arch.lane_idx())
                tBs_sBias = thr_bias_g2s.partition_D(sBias)

                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    bias_producer_state.reset_count()
                    work_tile_info = MoEWorkTileInfo(
                        expert_idx=tile_info[0],
                        tile_m_idx=tile_info[1],
                        tile_n_idx=tile_info[2],
                        k_tile_cnt=tile_info[3],
                    )
                    bias_ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)
                    real_bias, _ = bias_ext.get_gmem_tensor("bias", mBias_nl, padded_offsets, work_tile_info)
                    gBias_expert = cute.local_tile(real_bias, cute.slice_(self.mma_tiler[:2], (0, None)), (None, None))
                    bias_tile = gBias_expert[(None, work_tile_info.tile_n_idx, 0)]
                    bias_identity_tensor = cute.make_identity_tensor(bias_tile.shape)
                    bias_partitioned_by_g2s = thr_bias_g2s.partition_S(bias_tile)
                    bias_coord_partitioned_by_g2s = thr_bias_g2s.partition_S(bias_identity_tensor)

                    residue_n = sched_params.intermediate - work_tile_info.tile_n_idx * self.cta_tile_shape_mnk[1]
                    bias_pred_tensor = cute.make_rmem_tensor(bias_coord_partitioned_by_g2s[(None, 0)].shape, cutlass.Boolean)
                    for vi in cutlass.range_constexpr(cute.size(bias_pred_tensor)):
                        bias_pred_tensor[vi] = cute.elem_less(bias_coord_partitioned_by_g2s[(vi, 0)], (residue_n,))
                    bias_pred_tensor = bias_pred_tensor[((0, None),)]

                    bias_pipeline.producer_acquire(bias_producer_state)
                    cute.copy(bias_g2s_tiled, bias_partitioned_by_g2s[(None, 0)], tBs_sBias[(None, 0, bias_producer_state.index)], pred=bias_pred_tensor)
                    bias_pipeline.producer_commit(bias_producer_state)
                    bias_producer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()
                bias_pipeline.producer_tail(bias_producer_state)

        # ==============================================================
        # DMA / TMA load warp
        # ==============================================================
        if warp_idx == self.tma_warp_id:
            ext = self._make_extension(workspace_ptr)
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, desc_ptr_b = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)
                real_sfa, _ = ext.get_gmem_tensor("sfa", mSFA_mkl, padded_offsets, work_tile_info)
                real_sfb, desc_ptr_sfb = ext.get_gmem_tensor("sfb", mSFB_nkl, padded_offsets, work_tile_info)

                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                gSFA_mkl = cute.local_tile(real_sfa, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gSFB_nkl = cute.local_tile(real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))

                # MMA partition on gmem tensors
                thr_mma_dma = tiled_mma.get_slice(mma_tile_coord_v)
                thr_mma_sfb_dma = tiled_mma_sfb.get_slice(mma_tile_coord_v)
                tCgA = thr_mma_dma.partition_A(gA_mkl)
                tCgB = thr_mma_dma.partition_B(gB_nkl)
                tCgSFA = thr_mma_dma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb_dma.partition_B(gSFB_nkl)

                # TMA partition A
                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                # TMA partition B
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                # TMA partition SFA
                sfa_cta_layout = a_cta_layout
                tAsSFA, tAgSFA = cpasync.tma_partition(
                    tma_atom_sfa,
                    block_in_cluster_coord_vmnk[2],
                    sfa_cta_layout,
                    cute.group_modes(sSFA, 0, 3),
                    cute.group_modes(tCgSFA, 0, 3),
                )
                tAsSFA = cute.filter_zeros(tAsSFA)
                tAgSFA = cute.filter_zeros(tAgSFA)
                # TMA partition SFB
                sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
                tBsSFB, tBgSFB = cpasync.tma_partition(
                    tma_atom_sfb,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFB, 0, 3),
                    cute.group_modes(tCgSFB, 0, 3),
                )
                tBsSFB = cute.filter_zeros(tBsSFB)
                tBgSFB = cute.filter_zeros(tBgSFB)

                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_m, None, 0)]
                slice_n = mma_tile_coord_n
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_n // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar, mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_bar, mcast_mask=b_full_mcast_mask, tma_desc_ptr=desc_ptr_b)
                    cute.copy(tma_atom_sfa, tAgSFA_k, tAsSFA_pipe, tma_bar_ptr=tma_bar, mcast_mask=sfa_full_mcast_mask)
                    cute.copy(tma_atom_sfb, tBgSFB_k, tBsSFB_pipe, tma_bar_ptr=tma_bar, mcast_mask=sfb_full_mcast_mask, tma_desc_ptr=desc_ptr_sfb)

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==============================================================
        # MMA warp
        # ==============================================================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA TMEM tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # SFB TMEM tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # S2T copy partition for SFA/SFB
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acd_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                k_tile_cnt = tile_info[3]

                # Peek AB buffer full
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek Acc buffer empty
                acd_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                mma_tile_coord_mnl = (
                    tile_info[1] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[2],
                    tile_info[0],
                )

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acd_producer_state.phase ^ 1
                else:
                    acc_stage_index = acd_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acd_producer_state, peek_acc_empty_status)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if ab_consumer_state_next.count < k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state = ab_consumer_state_next

                if is_leader_cta:
                    acc_pipeline.producer_commit(acd_producer_state)

                acd_producer_state.advance()
                if acd_producer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acd_producer_state)

        # ==============================================================
        # Epilogue warps
        # ==============================================================
        if warp_idx < self.mma_warp_id and total_token > 0:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)

            # Shape-only partition on global tensor (invariant setup for t2r copy atom)
            gD_mnl_shape = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
            tCgD_shape = thr_mma_epi.partition_C(gD_mnl_shape)

            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCgD_shape,
                epi_tile,
                use_2cta_instrs,
            )

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rD,
                epi_tidx,
                sD,
            )
            tTR_rD_col = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD_col, tRS_sD_col = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rD_col,
                epi_tidx,
                sD_col,
            )

            if cutlass.const_expr(self.generate_sfd):
                norm_const = norm_const_tensor[0]
                regPerSubtile = 4
                sfd_row_tile = (cute.make_layout(128), cute.make_layout(32 * regPerSubtile))
                gSFDRow_mnl = cute.local_tile(mSFDRow_mnl, sfd_row_tile, (None, None, None))
                thr_copy_t2r_local = tiled_copy_t2r.get_slice(tidx)
                tCgSFDRow_mnl = thr_copy_t2r_local.partition_D(gSFDRow_mnl)
                tCgSFDRow_mnl = cute.filter_zeros(tCgSFDRow_mnl)
                tCrSFDRow = cute.make_rmem_tensor(tCgSFDRow_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype)
                tCrSFDRow_pvscale = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)
                d_rcp_limits = get_dtype_rcp_limits(self.d_dtype)

                sfd_col_tile = sfd_row_tile
                gSFDCol_mnl = cute.local_tile(mSFDCol_mnl, sfd_col_tile, (None, None, None))
                thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
                val_layout = cute.make_ordered_layout((1,), order=(0,))
                copy_atom_sfd_col = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gSFDCol_mnl.element_type, num_bits_per_copy=8)
                tiled_copy_sfd_col = cute.make_tiled_copy_tv(copy_atom_sfd_col, thr_layout, val_layout)
                thr_copy_sfd_col = tiled_copy_sfd_col.get_slice(tidx)
                tCgSFDCol_mnl = thr_copy_sfd_col.partition_D(cute.filter_zeros(gSFDCol_mnl))
                tCgSFDCol_mnl = cute.filter_zeros(tCgSFDCol_mnl)
                tCrSFDCol = cute.make_rmem_tensor(tCgSFDRow_mnl[(None, None, None, 0, 0, 0)].shape, self.sf_dtype)
                tCrSFDCol_pvscale = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)

            epi_ext = self._make_extension(workspace_ptr)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            d_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
            d_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_d_stage, producer_group=d_producer_group)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            if cutlass.const_expr(self.enable_bias):
                bias_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_bias_stage)
                bias_s2r_tom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.bias_dtype, num_bits_per_copy=128)
                tTR_rBias = cute.make_rmem_tensor(cute.make_layout(self.epi_tile[1]), self.bias_dtype)

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                expert_idx = epi_work_tile_info.expert_idx
                epi_ext.update_expert_info(padded_offsets, expert_idx)

                alpha_val = alpha[expert_idx]

                if cutlass.const_expr(self.enable_bias):
                    bias_consumer_state.reset_count()
                    bias_pipeline.consumer_wait(bias_consumer_state)
                    sBias_stage = sBias[(None, bias_consumer_state.index)]
                    sBias_subtiles = cute.flat_divide(sBias_stage, cute.make_layout(self.epi_tile[1]))

                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                real_d_col = real_d
                if cutlass.const_expr(self.generate_sfd):
                    real_d_col, _ = epi_ext.get_gmem_tensor("d_col", mD_col_mnl, padded_offsets, epi_work_tile_info)

                thr_mma_epi_loop = tiled_mma.get_slice(mma_tile_coord_v)

                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                tCgD_loop = thr_mma_epi_loop.partition_C(gD_mnl_loop)
                _, bSG_sD, bSG_gD_partitioned = epilog_gmem_copy_and_partition(
                    epi_tidx,
                    tma_atom_d,
                    tCgD_loop,
                    epi_tile,
                    sD,
                )

                gD_col_mnl_loop = gD_mnl_loop
                tCgD_col_loop = tCgD_loop
                if cutlass.const_expr(self.generate_sfd):
                    gD_col_mnl_loop = cute.local_tile(real_d_col, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgD_col_loop = thr_mma_epi_loop.partition_C(gD_col_mnl_loop)
                _, bSG_sD_col, bSG_gD_col_partitioned = epilog_gmem_copy_and_partition(
                    epi_tidx,
                    tma_atom_d_col,
                    tCgD_col_loop,
                    epi_tile,
                    sD_col,
                )

                epi_mma_tile_coord = (
                    epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    epi_work_tile_info.tile_n_idx,
                    0,
                )
                bSG_gD = bSG_gD_partitioned[(None, None, None, *epi_mma_tile_coord)]
                bSG_gD_col = bSG_gD_col_partitioned[(None, None, None, *epi_mma_tile_coord)]
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))
                bSG_gD_col = cute.group_modes(bSG_gD_col, 1, cute.rank(bSG_gD_col))

                if cutlass.const_expr(self.generate_sfd):
                    tCgSFDRow_mn = tCgSFDRow_mnl[(None, None, None, None, None, 0)]
                    tCgSFDCol_mnl_new = tCgSFDCol_mnl
                    if cutlass.const_expr(self.discrete_col_sfd):
                        tCgSFDCol_mnl_new = self.create_and_partition_new_SFDCol(tile_info, mSFDCol_mnl, padded_offsets)
                    tCgSFDCol_mn = tCgSFDCol_mnl_new[(None, None, None, None, None, 0)]

                if cutlass.const_expr(self.generate_amax):
                    thread_tile_amax = cutlass.Float32(0.0)

                mPosition = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0] + tidx
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mProb = real_prob[mPosition, 0, 0]
                acc_scale = cutlass.Float32(alpha_val)
                # Optional per-row epilogue scale, applied together with the
                # per-expert alpha before output conversion.
                if cutlass.const_expr(row_scale is not None):
                    real_row_scale, _ = epi_ext.get_gmem_tensor(
                        "row_scale",
                        row_scale,
                        padded_offsets,
                        epi_work_tile_info,
                    )
                    acc_scale = acc_scale * real_row_scale[mPosition]

                # C1 fix: phase-based acc stage indexing for overlapping_accum
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_consumer_state.index

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                acc_pipeline.consumer_wait(acc_consumer_state)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - subtile_idx

                    # C1 fix: fence + early release for overlapping_accum
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if cutlass.const_expr(self.enable_bias):
                        # m7 fix: use real_subtile_idx directly (matches contiguous)
                        sBias_sub = sBias_subtiles[(None, real_subtile_idx)]
                        cute.copy(bias_s2r_tom, sBias_sub, tTR_rBias)
                        bias_vec = tTR_rBias.load()
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                                bias_f32_0 = bias_vec[i].to(cutlass.Float32)
                                bias_f32_1 = bias_vec[i + 1].to(cutlass.Float32)
                                bias_f32_0, bias_f32_1 = cute.arch.mul_packed_f32x2(
                                    (mProb, mProb),
                                    (bias_f32_0, bias_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                                tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rAcc[i], tTR_rAcc[i + 1]),
                                    (acc_scale, acc_scale),
                                    (bias_f32_0, bias_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tTR_rAcc[i] = tTR_rAcc[i] * acc_scale + bias_vec[i].to(cutlass.Float32) * mProb
                    else:
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                                tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.mul_packed_f32x2(
                                    (tTR_rAcc[i], tTR_rAcc[i + 1]),
                                    (acc_scale, acc_scale),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tTR_rAcc[i] = tTR_rAcc[i] * acc_scale

                    acc_vec = tTR_rAcc.load()
                    if cutlass.const_expr(not self.enable_bias):
                        tCompute = cute.make_rmem_tensor(acc_vec.shape, self.acc_dtype)
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                                    (acc_vec[i], acc_vec[i + 1]),
                                    (mProb, mProb),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tCompute[i] = acc_vec[i] * mProb
                    else:
                        tCompute = tTR_rAcc

                    if cutlass.const_expr(self.generate_amax):
                        thread_tile_amax = amax_reduction_per_thread(tCompute, thread_tile_amax)

                    if cutlass.const_expr(self.generate_sfd):
                        tCompute_col = cute.make_rmem_tensor(tCompute.layout, tCompute.element_type)
                        tCompute_col.store(tCompute.load())
                        self.quant_sfd_row(
                            real_subtile_idx % 4,
                            tiled_copy_r2s,
                            tCompute,
                            tCrSFDRow_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD,
                        )
                        self.quant_sfd_col(
                            real_subtile_idx % 4,
                            tiled_copy_r2s,
                            tCompute_col,
                            tCrSFDCol_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD_col,
                        )
                        # SFD M tile = cta_tile_m = 128; tile_m_idx is CTA-level per-expert
                        global_sfd_m = epi_work_tile_info.tile_m_idx + epi_ext.token_offset // self.cta_tile_shape_mnk[0]
                        if cutlass.const_expr(self.mma_tiler[1] == 256):
                            sfd_n = epi_work_tile_info.tile_n_idx * 2 + (real_subtile_idx >> 2)
                        else:
                            sfd_n = epi_work_tile_info.tile_n_idx
                        sfd_row_idx_mn = (global_sfd_m, sfd_n)
                        sfd_col_idx_mn = sfd_row_idx_mn
                        if cutlass.const_expr(self.discrete_col_sfd):
                            sfd_col_idx_mn = (
                                epi_work_tile_info.tile_m_idx,
                                sfd_n,
                            )
                        tCgSFDRow = tCgSFDRow_mn[(None, None, None, *sfd_row_idx_mn)]
                        tCgSFDCol = tCgSFDCol_mn[(None, None, None, *sfd_col_idx_mn)]
                        if subtile_idx == 3 or subtile_idx == 7:
                            if sfd_row_idx_mn[1] * 32 * regPerSubtile < cute.size(cute.shape(mSFDRow_mnl.layout, mode=[1])):
                                tCrSFDRow.store(tCrSFDRow_pvscale.load().to(self.sf_dtype))
                                cute.autovec_copy(tCrSFDRow, tCgSFDRow)
                            if sfd_col_idx_mn[1] * 32 * regPerSubtile < cute.size(cute.shape(mSFDCol_mnl.layout, mode=[1])):
                                tCrSFDCol.store(tCrSFDCol_pvscale.load().to(self.sf_dtype))
                                cute.autovec_copy(tCrSFDCol, tCgSFDCol)
                    else:
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rD.store(acc_vec.to(self.d_dtype))

                    d_buffer = num_prev_subtiles % self.num_d_stage
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, d_buffer)])
                    if cutlass.const_expr(self.generate_sfd):
                        cute.copy(tiled_copy_r2s, tRS_rD_col, tRS_sD_col[(None, None, None, d_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_d, bSG_sD[(None, d_buffer)], bSG_gD[(None, real_subtile_idx)])
                        if cutlass.const_expr(self.generate_sfd):
                            cute.copy(tma_atom_d_col, bSG_sD_col[(None, d_buffer)], bSG_gD_col[(None, real_subtile_idx)])
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                if cutlass.const_expr(self.enable_bias):
                    bias_pipeline.consumer_release(bias_consumer_state)
                    bias_consumer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                if cutlass.const_expr(self.generate_amax):
                    gAmax = mAmax_tensor[(expert_idx, None)].iterator.llvm_ptr
                    self.amax_reduction_per_warp_and_cta(thread_tile_amax, warp_idx, sAmax, gAmax)

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            d_pipeline.producer_tail()

    # ------------------------------------------------------------------
    # Internal: create extension based on weight_mode
    # ------------------------------------------------------------------

    @cute.jit
    def _make_extension(self, workspace_ptr):
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            desc_workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            return DiscreteWeightScaledGemmSchedExtension(
                tensormap_ctor=desc_workspace,
                sf_vec_size=self.sf_vec_size,
            )
        else:
            return ContiguousAndConsistentGroupedGemmSchedExtension(
                sf_vec_size=self.sf_vec_size,
            )
