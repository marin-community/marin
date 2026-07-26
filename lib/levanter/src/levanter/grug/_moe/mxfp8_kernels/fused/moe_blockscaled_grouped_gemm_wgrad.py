# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# ruff: noqa

"""
MoE Block-Scaled Grouped GEMM Kernel — Weight Gradient (2Dx2D).

Computes:  A(hidden, tokens_sum) x B(tokens_sum, intermediate)
        -> C(experts, hidden, intermediate)

where C is the weight gradient.  K (tokens) varies per expert;
M (hidden) and N (intermediate) are fixed across all experts.

Supports:
    - CLC-based dynamic persistent tile scheduling
    - Dense (contiguous 3-D C) / Discrete (per-expert pointer array C) output
    - accumulate_on_output (TMA reduce for atomic accumulation)
    - NVFP4 global_scale support
    - k_tile_cnt == 0 handling (zero output for empty experts)

This module contains only the kernel class.
Scheduler: moe_persistent_scheduler.py (CLC mode, scenario="2Dx2D")
Extension: moe_sched_extension.py (WgradDense / WgradDiscrete)
"""

from typing import Type, Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
)
from .moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from .moe_utils import (
    MoEWeightMode,
    WGradInputOrder,
    WgradSfTensormapConstructor,
)
from .moe_sched_extension import (
    WgradScaledGemmSchedExtension,
)
from .moe_kernel_helpers import (
    compute_stages_wgrad,
)


class BlockScaledMoEGroupedGemmWgradKernel:
    """Block-scaled grouped GEMM kernel for MoE weight gradient (2Dx2D).

    :param sf_vec_size: Scale-factor vector size (16 or 32).
    :param acc_dtype: Accumulator data type (Float32).
    :param use_2cta_instrs: Use 2-CTA MMA instructions.
    :param mma_tiler_mn: MMA tile shape (M, N).
    :param cluster_shape_mn: Cluster shape (M, N).
    :param accumulate_on_output: Use TMA reduce for atomic accumulation.
    :param expert_cnt: Number of experts.
    :param weight_mode: ``MoEWeightMode.DENSE`` or ``MoEWeightMode.DISCRETE`` for output.
    """

    def __init__(
        self,
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        accumulate_on_output: bool = False,
        expert_cnt: int = 1,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        input_order: WGradInputOrder = WGradInputOrder.Tensor2D,
    ):
        self.sf_vec_size = sf_vec_size
        self.expert_cnt = expert_cnt
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.accumulate_on_output = accumulate_on_output
        self.weight_mode = weight_mode
        self.input_order = input_order

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_id,
            )
        )

        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def get_workspace_bytes(self) -> int:
        return WgradSfTensormapConstructor.get_workspace_size(self.input_order, self.weight_mode, self.expert_cnt)

    # ------------------------------------------------------------------
    # _setup_attributes
    # ------------------------------------------------------------------

    def _setup_attributes(self) -> None:
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        mma_tiler_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_tiler_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_tiler_k,
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

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = compute_stages_wgrad(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )
        self.num_sched_stages = 2

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
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        self.overlapping_accum = self.cta_tile_shape_mnk[1] == 256
        self.num_acc_pipeline_stages = 1 if self.overlapping_accum else self.num_acc_stage

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage - (self.num_sf_tmem_cols if self.overlapping_accum else 0)

        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

    # ------------------------------------------------------------------
    # MMA helpers
    # ------------------------------------------------------------------

    def _create_tiled_mma(self):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

    def _create_tiled_mma_sfb(self):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # (hidden, tokens_sum) — activation^T
        mat_b: cute.Tensor,  # (tokens_sum, intermediate) — activation
        scale_a: cute.Tensor,  # SFA (assembled block-scaled layout)
        scale_b: cute.Tensor,  # SFB (assembled block-scaled layout)
        out,  # Dense: cute.Tensor (experts, hidden, intermediate)
        # Discrete: cute.Pointer to int64[]
        offs,  # Union[cute.Tensor, cute.Pointer] (experts,) cumsum end offsets, int32
        workspace: cute.Tensor,  # expert-wise TMA desc (discrete only)
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        global_scale_a: Optional[cute.Tensor] = None,
        global_scale_b: Optional[cute.Tensor] = None,
        # Discrete-only: template tensor for a single expert's output (M, N) or (M, N, 1)
        out_single_expert: Optional[cute.Tensor] = None,
    ) -> None:

        # This should be removed after from_dlpack fix.
        if cutlass.const_expr(mat_a.iterator.dtype.width < 8):
            mat_a = cute.make_tensor(
                mat_a.iterator,
                cute.recast_layout(mat_a.iterator.dtype.width, 8, mat_a.layout),
            )
        if cutlass.const_expr(mat_b.iterator.dtype.width < 8):
            mat_b = cute.make_tensor(
                mat_b.iterator,
                cute.recast_layout(mat_b.iterator.dtype.width, 8, mat_b.layout),
            )

        # =================================================================
        # Step 1: Transform to GEMM domain (2Dx2D)
        # =================================================================
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # mat_a: (hidden, tokens_sum) -> A: (M=hidden, K=tokens_sum, L=1)
        hidden, tokens_sum = mat_a.shape
        a_gemm = cute.make_tensor(
            mat_a.iterator,
            cute.make_layout(
                (hidden, tokens_sum, c1),
                stride=(mat_a.stride[0], mat_a.stride[1], c0),
            ),
        )
        # mat_b: (tokens_sum, intermediate) -> B: (N=intermediate, K=tokens_sum, L=1)
        tokens_sum_b, intermediate = mat_b.shape
        b_gemm = cute.make_tensor(
            mat_b.iterator,
            cute.make_layout(
                (intermediate, tokens_sum_b, c1),
                stride=(mat_b.stride[1], mat_b.stride[0], c0),
            ),
        )

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            # out: (experts, hidden, intermediate) -> C: (M=hidden, N=intermediate, L=experts)
            experts, hidden_c, intermediate_c = out.shape
            c_gemm = cute.make_tensor(
                out.iterator,
                cute.make_layout(
                    (hidden_c, intermediate_c, experts),
                    stride=(out.stride[1], out.stride[2], out.stride[0]),
                ),
            )
            expert_cnt = experts
        else:
            # Discrete: out is a Pointer to int64[] of per-expert base addresses
            expert_cnt = self.expert_cnt
            # Normalize out_single_expert to rank-3 (M, N, 1) if rank-2
            if cutlass.const_expr(cute.rank(out_single_expert.layout) == 2):
                out_single_expert = cute.make_tensor(
                    out_single_expert.iterator,
                    cute.make_layout(
                        (*out_single_expert.shape, c1),
                        stride=(*out_single_expert.stride, c0),
                    ),
                )
            c_gemm = out_single_expert

        intermediate_dim = intermediate
        hidden_dim = hidden

        # SFA: (hidden_padded, tokens_sum_padded_sf)
        hidden_padded = scale_a.shape[0]
        tokens_sum_padded = scale_a.shape[1] * self.sf_vec_size
        sfa_gemm = cute.make_tensor(
            scale_a.iterator,
            blockscaled_utils.tile_atom_to_shape_SF((hidden_padded, tokens_sum_padded, c1), self.sf_vec_size),
        )
        # SFB: (intermediate_padded, tokens_sum_padded_sf)
        intermediate_padded = scale_b.shape[0]
        sfb_gemm = cute.make_tensor(
            scale_b.iterator,
            blockscaled_utils.tile_atom_to_shape_SF((intermediate_padded, tokens_sum_padded, c1), self.sf_vec_size),
        )

        # =================================================================
        # Step 2: Infer dtypes and major modes
        # =================================================================
        self.a_dtype = a_gemm.element_type
        self.b_dtype = b_gemm.element_type
        self.c_dtype = c_gemm.element_type
        self.sf_dtype = sfa_gemm.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_gemm).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_gemm).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_gemm)

        # =================================================================
        # Step 3: Setup kernel attributes
        # =================================================================
        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        # =================================================================
        # Step 4: Create TMA ops
        # =================================================================

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

        # TMA ops for SFA/SFB (atoms built after helper kernel)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)

        # TMA store/reduce C
        if cutlass.const_expr(self.accumulate_on_output):
            c_tma_op = cpasync.CopyReduceBulkTensorTileS2GOp()
        else:
            c_tma_op = cpasync.CopyBulkTensorTileS2GOp()

        # =================================================================
        # Step 5: Scheduler params and grid
        # =================================================================
        sched_params = MoESchedulerParams(
            scenario="2Dx2D",
            expert_shape=(expert_cnt, intermediate_dim, hidden_dim),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        grid = MoESchedulerParams.get_grid_shape(sched_params, max_active_clusters)

        # =================================================================
        # Step 6: Launch helper kernel (both Dense and Discrete)
        # =================================================================
        # Builds expert-wise SFA/SFB TMA descs (both modes) + C descs
        # (Discrete only). The WgradSfTensormapConstructor is created inside
        # the kernel body from the raw params — it has too many Constexpr
        # fields for MLIR serialization as a kernel argument.
        # Dense passes None for C-related params; no if-else branch needed.
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        epi_smem_layout_helper = cute.select(self.c_smem_layout_staged, mode=[0, 1]) if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None
        if cutlass.const_expr(self.input_order == WGradInputOrder.TensorRagged):
            a_gemm_helper = a_gemm
            b_gemm_helper = b_gemm
            a_op_helper = a_op
            b_op_helper = b_op
            a_smem_layout_helper = a_smem_layout
            b_smem_layout_helper = b_smem_layout
        else:
            a_gemm_helper = None
            b_gemm_helper = None
            a_op_helper = None
            b_op_helper = None
            a_smem_layout_helper = None
            b_smem_layout_helper = None

        self.helper_kernel(
            sfa_gemm,
            sfb_gemm,
            offs,
            workspace.iterator,
            sfa_op,
            sfa_smem_layout,
            sfb_op,
            sfb_smem_layout,
            tiled_mma,
            tiled_mma_sfb,
            self.cluster_layout_vmnk.shape,
            self.cluster_layout_sfb_vmnk.shape,
            out if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            c_gemm if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            c_tma_op if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            epi_smem_layout_helper,
            self.epi_tile if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            a_gemm_helper,
            b_gemm_helper,
            a_op_helper,
            b_op_helper,
            a_smem_layout_helper,
            b_smem_layout_helper,
        ).launch(
            grid=(expert_cnt, 1, 1),
            block=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

        # Build A, B, SFA, SFB, C TMA atoms AFTER the helper kernel launch.
        # make_tiled_tma_atom_*() stores smem_layout on the TMA op object.
        # Because the ops are passed as Constexpr to the helper kernel, the
        # helper's own make_tiled_tma_atom_*() calls contaminate them with
        # kernel-region block arguments.  Creating fresh atoms here re-sets
        # the ops' smem_layout to valid host-region values.
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_gemm,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_gemm,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            c_tma_op,
            c_gemm,
            epi_smem_layout,
            self.epi_tile,
        )

        # =================================================================
        # Step 7: Launch main kernel
        # =================================================================
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
            tma_atom_c,
            tma_tensor_c,
            a_gemm,
            b_gemm,
            c_gemm,
            sfa_gemm,
            sfb_gemm,
            offs,
            sched_params,
            workspace.iterator,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            global_scale_a,
            global_scale_b,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    # ------------------------------------------------------------------
    # helper_kernel (expert-wise TMA desc init via construct_and_write)
    # ------------------------------------------------------------------

    @cute.kernel
    def helper_kernel(
        self,
        sfa_gemm: cute.Tensor,
        sfb_gemm: cute.Tensor,
        offs: cute.Tensor,
        workspace_ptr,
        sfa_tma_op: cutlass.Constexpr,
        sfa_smem_layout,
        sfb_tma_op: cutlass.Constexpr,
        sfb_smem_layout,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk_shape: cutlass.Constexpr,
        cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
        c_ptrs=None,
        c_single_expert=None,
        c_tma_op: cutlass.Constexpr = None,
        epi_smem_layout=None,
        epi_tile=None,
        a_tensor=None,
        b_tensor=None,
        a_tma_op: cutlass.Constexpr = None,
        b_tma_op: cutlass.Constexpr = None,
        a_smem_layout=None,
        b_smem_layout=None,
    ):
        """Build per-expert TMA descriptors (SFA/SFB for all modes, + C for Discrete).

        Launched with grid=(expert_cnt, 1, 1).
        Each block builds TMA descriptors for one expert.
        """
        from .moe_utils import WgradSfTensormapConstructor

        ctor = WgradSfTensormapConstructor(
            sf_vec_size=self.sf_vec_size,
            weight_mode=self.weight_mode,
            input_order=self.input_order,
            sfa_tma_op=sfa_tma_op,
            sfb_tma_op=sfb_tma_op,
            sfa_smem_layout=sfa_smem_layout,
            sfb_smem_layout=sfb_smem_layout,
            tiled_mma=tiled_mma,
            tiled_mma_sfb=tiled_mma_sfb,
            mma_tiler=self.mma_tiler,
            mma_tiler_sfb=self.mma_tiler_sfb,
            cluster_layout_vmnk_shape=cluster_layout_vmnk_shape,
            cluster_layout_sfb_vmnk_shape=cluster_layout_sfb_vmnk_shape,
            sfa_tensor=sfa_gemm,
            sfb_tensor=sfb_gemm,
            offs=offs,
            workspace_ptr=workspace_ptr,
            a_tma_op=a_tma_op,
            b_tma_op=b_tma_op,
            a_smem_layout=a_smem_layout,
            b_smem_layout=b_smem_layout,
            a_major_mode=self.a_major_mode,
            b_major_mode=self.b_major_mode,
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tma_op=c_tma_op,
            epi_smem_layout=epi_smem_layout,
            epi_tile=epi_tile,
            c_ptrs=c_ptrs,
            c_single_expert=c_single_expert,
            expert_cnt=self.expert_cnt,
        )
        expert_idx = cute.arch.block_idx()[0]
        ctor.construct_and_write(expert_idx)

    # ------------------------------------------------------------------
    # kernel (GPU device kernel)
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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        a_gemm: cute.Tensor,
        b_gemm: cute.Tensor,
        c_gemm: cute.Tensor,
        sfa_gemm: cute.Tensor,
        sfb_gemm: cute.Tensor,
        offs: cute.Tensor,
        sched_params: MoESchedulerParams,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        global_scale_a: Optional[cute.Tensor],
        global_scale_b: Optional[cute.Tensor],
    ):
        """GPU device kernel for MoE wgrad with block scaling and CLC scheduling."""

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # =================================================================
        # SharedStorage
        # =================================================================
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_sched_stages, use_dynamic_sched=True)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_pipeline_stages * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sched_storage = storage.scheduler

        # =================================================================
        # Pipelines
        # =================================================================

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * 32 * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Scheduler pipeline (sched warp -> tma/mma/epi warps)
        sched_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        num_sched_consumer_threads = 32 * len((self.tma_warp_id, self.mma_warp_id, *self.epilogue_warp_id))
        sched_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_sched_consumer_threads)
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_sched_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            defer_sync=True,
        )

        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Scheduler (CLC-based for 2Dx2D)
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            offs,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=None,
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # Cluster barrier sync after init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # =================================================================
        # SMEM tensors
        # =================================================================
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        if cutlass.const_expr(self.overlapping_accum):
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

        # Scheduler buf tensor for sched_pipeline broadcast
        sched_buf_ptr = sched_storage.sInfo.data_ptr()
        sched_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)
        sched_buf_tensor = cute.make_tensor(
            sched_buf_ptr,
            cute.make_layout((4, self.num_sched_stages), stride=(1, 4)),
        )

        # Build extension
        from .moe_utils import TensormapWorkspace, WgradSfTensormapConstructor

        slot_names = WgradSfTensormapConstructor.slot_names(self.input_order, self.weight_mode)
        desc_workspace = TensormapWorkspace(workspace_ptr, slot_names)
        ext = WgradScaledGemmSchedExtension(
            tensormap_ctor=desc_workspace,
            sf_vec_size=self.sf_vec_size,
            weight_mode=self.weight_mode,
            input_order=self.input_order,
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =================================================================
        # Scheduler warp (warp 6)
        # =================================================================
        if warp_idx == self.sched_warp_id:
            sched_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_sched_stages)

            work_tile_info = scheduler.initial_work_tile_info()
            # with cute.arch.elect_one():
            #     cute.printf("work_tile_info: [{}, {}, {}, {}]", work_tile_info.expert_idx, work_tile_info.tile_m_idx, work_tile_info.tile_n_idx, work_tile_info.k_tile_cnt)

            sched_pipeline.producer_acquire(sched_producer_state)
            rmem = work_tile_info.to_rmem_tensor()
            cute.copy(
                sched_copy_atom,
                rmem,
                sched_buf_tensor[(None, sched_producer_state.index)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            sched_producer_state.advance()

            work_tile_info = scheduler.advance_to_next_work()
            while work_tile_info.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                rmem = work_tile_info.to_rmem_tensor()
                cute.copy(
                    sched_copy_atom,
                    rmem,
                    sched_buf_tensor[(None, sched_producer_state.index)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()

                work_tile_info = scheduler.advance_to_next_work()

            sched_pipeline.producer_acquire(sched_producer_state)
            sentinel = MoEWorkTileInfo(
                cutlass.Int32(-1),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            )
            rmem = sentinel.to_rmem_tensor()
            cute.copy(
                sched_copy_atom,
                rmem,
                sched_buf_tensor[(None, sched_producer_state.index)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            sched_pipeline.producer_tail(sched_producer_state)

        # =================================================================
        # TMA load warp (warp 5)
        # =================================================================
        if warp_idx == self.tma_warp_id:
            a_full_mcast_mask = None
            b_full_mcast_mask = None
            sfa_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(offs, work_tile_info.expert_idx)

                real_a, desc_ptr_a = ext.get_gmem_tensor(
                    "a",
                    mA_mkl,
                    offs,
                    work_tile_info,
                )
                real_b, desc_ptr_b = ext.get_gmem_tensor(
                    "b",
                    mB_nkl,
                    offs,
                    work_tile_info,
                )
                real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                    "sfa",
                    mSFA_mkl,
                    offs,
                    work_tile_info,
                )
                real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                    "sfb",
                    mSFB_nkl,
                    offs,
                    work_tile_info,
                )

                # with cute.arch.elect_one():
                #     cute.printf(
                #         "TMA: expert={} tile_m={} tile_n={} k_cnt={} sfb_desc={} real_sfb_ptr={}\n",
                #         work_tile_info.expert_idx,
                #         work_tile_info.tile_m_idx,
                #         work_tile_info.tile_n_idx,
                #         k_tile_cnt,
                #         desc_ptr_sfb,
                #         real_sfb.iterator,
                #     )

                gA_mkl = cute.local_tile(
                    real_a,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                gB_nkl = cute.local_tile(
                    real_b,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (None, None, None),
                )
                gSFA_mkl = cute.local_tile(
                    real_sfa,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                gSFB_nkl = cute.local_tile(
                    real_sfb,
                    cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                    (None, None, None),
                )

                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
                tCgA = thr_mma.partition_A(gA_mkl)
                tCgB = thr_mma.partition_B(gB_nkl)
                tCgSFA = thr_mma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
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

                mma_tile_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]
                slice_n = work_tile_info.tile_n_idx
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = work_tile_info.tile_n_idx // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_a,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_b,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, handle.count)],
                        tAsSFA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_sfa,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, handle.count)],
                        tBsSFB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_sfb,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()
            ab_producer.tail()

        # =================================================================
        # MMA warp (warp 4) — identical to source file
        # =================================================================
        if warp_idx == self.mma_warp_id:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

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

            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                if is_leader_cta:
                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        offset = cutlass.Int32((work_tile_info.tile_n_idx % 2) * 2)
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        s2t_stage_coord = (None, None, None, None, handle.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB_mma],
                            tCtAcc,
                        )
                        handle.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        # =================================================================
        # SMEM tensor C (allocated after MMA section)
        # =================================================================
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        # =================================================================
        # Epilogue warps (warps 0-3)
        # =================================================================
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)

            epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )

            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)

            num_tiles_executed = cutlass.Int32(0)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(offs, work_tile_info.expert_idx)

                real_c, desc_ptr_c = ext.get_gmem_tensor(
                    "c",
                    mC_mnl,
                    offs,
                    work_tile_info,
                )

                gC_mnl = cute.local_tile(
                    real_c,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                tCgC = thr_mma.partition_C(gC_mnl)
                tCgC_transformed = transform_partitioned_tensor_layout(tCgC)

                mma_tile_coord_mnl = (
                    work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    work_tile_info.tile_n_idx,
                    cutlass.Int32(0),
                )

                tiled_copy_t2r, tTR_tAcc_base_epi, tTR_rAcc = epilogue_tmem_copy_and_partition(
                    self,
                    tidx,
                    tCtAcc_transformed,
                    tCgC_transformed,
                    epi_tile,
                    use_2cta_instrs,
                )
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC)

                tCgC_epi = cute.flat_divide(tCgC_transformed, epi_tile)
                bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.group_modes(tCgC_epi, 0, 2),
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = True if acc_stage_index == 0 else False
                else:
                    acc_stage_index = acc_consumer_state.index

                tTR_tAcc = tTR_tAcc_base_epi[(None, None, None, None, None, acc_stage_index)]

                if k_tile_cnt > 0:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if cutlass.const_expr(global_scale_a is not None):
                    expert_idx = work_tile_info.expert_idx
                    current_scale_a_iter = global_scale_a.iterator + expert_idx
                    current_scale_b_iter = global_scale_b.iterator + expert_idx
                    current_scale_b_iter = global_scale_b.iterator + expert_idx
                    alpha = cute.arch.load(current_scale_a_iter.llvm_ptr, cutlass.Float32) * cute.arch.load(current_scale_b_iter.llvm_ptr, cutlass.Float32)
                else:
                    alpha = None

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx

                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    if k_tile_cnt > 0:
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            if k_tile_cnt > 0:
                                acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()

                    acc_vec = cute.zeros_like(tiled_copy_r2s.retile(tTR_rAcc), dtype=tTR_rAcc._dtype)
                    if k_tile_cnt > 0:
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(global_scale_a is not None):
                        acc_vec = acc_vec * alpha
                    acc_vec = acc_vec.to(self.c_dtype)
                    tRS_rC.store(acc_vec)

                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_sync_barrier.arrive_and_wait()

                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                            tma_desc_ptr=desc_ptr_c,
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(not self.overlapping_accum):
                    if k_tile_cnt > 0:
                        acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                num_tiles_executed += cutlass.Int32(1)

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
