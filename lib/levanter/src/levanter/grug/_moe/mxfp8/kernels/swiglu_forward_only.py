# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Includes software developed at NVIDIA CORPORATION (https://www.nvidia.com/): cudnn-frontend.
#
# `SwigluKernel.__call__` and `SwigluKernel.kernel` are copied from nvidia-cudnn-frontend 1.31.0,
# cudnn/gemm/cutedsl/grouped/swiglu/grouped_gemm_swiglu_quant.py,
# BlockScaledContiguousGroupedGemmKernel.__call__ and .kernel, with upstream's module imports made
# absolute. `__init__` extends upstream's; every other method is inherited from that class unchanged.
# Modified by Marin: a static `forward_only` option skips the C (pre-activation) store and the
# column-quantized D and its scales, and writes D and its row scales exactly as the full kernel does.
#
# Formatted with the repository's black and ruff settings; upstream lint findings are suppressed.
# ruff: noqa: E712, F841
# pyrefly: ignore-errors

"""The cudnn-frontend fused SwiGLU grouped GEMM with a forward-only variant."""

from typing import Optional, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator
from cudnn.gemm.cutedsl.grouped.canonical import kernel_facing_b, kernel_facing_mx, kernel_facing_prob
from cudnn.gemm.cutedsl.grouped.swiglu.grouped_gemm_swiglu_quant import BlockScaledContiguousGroupedGemmKernel
from cudnn.gemm.cutedsl.grouped.utils import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
    search_expert_idx_full,
    search_expert_idx_incremental,
    silu_f32,
)
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05


class SwigluKernel(BlockScaledContiguousGroupedGemmKernel):
    """Upstream's SwiGLU kernel with a static option to store only what the next forward GEMM reads.

    With ``forward_only`` the kernel writes D (the row-quantized SwiGLU output) and its row scale
    factors bit for bit as the full kernel does, and skips C (the pre-activation), the
    column-quantized D and its scale factors, which only a backward reads. The call then takes None
    for ``c``, ``d_col`` and ``sfd_col_tensor``.
    """

    def __init__(self, *, forward_only: bool, **kwargs):
        super().__init__(**kwargs)
        self.forward_only = forward_only

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: Optional[cute.Tensor],
        d: cute.Tensor,
        d_col: Optional[cute.Tensor],
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        sfd_row_tensor: Optional[cute.Tensor],
        sfd_col_tensor: Optional[cute.Tensor],
        amax_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        prob: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C, None when forward_only
        :type c: Optional[cute.Tensor]
        :param d: Output tensor D
        :type d: cute.Tensor
        :param d_col: Output tensor D column quantized, None when forward_only
        :type d_col: Optional[cute.Tensor]
        :param sfa: Scale factor tensor A
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param sfd_row_tensor: Scale factor tensor D
        :type sfd_row_tensor: Optional[cute.Tensor]
        :param sfd_col_tensor: Scale factor tensor D
        :type sfd_col_tensor: Optional[cute.Tensor]
        :param amax_tensor: Absolute maximum value tensor
        :type amax_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Norm constant tensor
        :type norm_const_tensor: Optional[cute.Tensor]
        :param padded_offsets: End offset for each expert after padding, shape (expert_cnt,).
            padded_offsets[i] indicates where expert[i] ends in the padded A tensor.
            Expert i processes A[padded_offsets[i-1]:padded_offsets[i], :] (with padded_offsets[-1]=0).
            Must be 16-byte aligned and each offset must be divisible by FIX_PAD_SIZE.
        :type padded_offsets: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Canonical-rank operands (grouped/canonical.py) normalize to the kernel-facing views.
        a = kernel_facing_mx(a)
        b = kernel_facing_b(b)
        c = kernel_facing_mx(c)
        d = kernel_facing_mx(d)
        d_col = kernel_facing_mx(d_col)
        prob = kernel_facing_prob(prob)

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(b).mma_major_mode()
        self.d_layout = LayoutEnum.from_tensor(d)
        if cutlass.const_expr(self.forward_only):
            # Modified by Marin: no C is stored. Its unused shared-memory buffer keeps the full
            # kernel's bf16 row-major size, so both variants run the same pipeline stages.
            self.c_dtype = cutlass.BFloat16
            self.c_layout = self.d_layout
        else:
            self.c_dtype = c.element_type
            self.c_layout = LayoutEnum.from_tensor(c)

        # Compute grid size
        m, n, l = cute.shape(d)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        # Setup sfd tensor by filling D tensor to scale factor atom layout
        self.generate_sfd = sfd_row_tensor is not None and norm_const_tensor is not None
        # Modified by Marin: whether the column-quantized D and its scale factors are stored.
        self.generate_sfd_col = self.generate_sfd and not self.forward_only
        # discrete_col_sfd is enabled via __init__, padded_offsets contains the same info as m_split_cumsum
        if cutlass.const_expr(self.generate_sfd_col == False):
            self.discrete_col_sfd = False
        if cutlass.const_expr(self.generate_sfd):
            sfd_row_layout = blockscaled_utils.tile_atom_to_shape_SF(d.shape, self.sf_vec_size)
            sfd_row_tensor = cute.make_tensor(sfd_row_tensor.iterator, sfd_row_layout)
        if cutlass.const_expr(self.generate_sfd_col):
            sfd_col_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
                d.shape,
                (1, 2, 3),
            )
            if cutlass.const_expr(self.discrete_col_sfd):
                sfd_col_layout = sfd_row_layout
            sfd_col_tensor = cute.make_tensor(sfd_col_tensor.iterator, sfd_col_layout)

        self.generate_amax = amax_tensor is not None

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

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
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

        # Setup TMA load for A
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

        # Setup TMA load for B
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

        # Setup TMA load for SFA
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

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
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

        # Setup TMA store for C
        tma_atom_c, tma_tensor_c = None, None
        if cutlass.const_expr(not self.forward_only):
            c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                c_smem_layout,
                self.epi_tile_c,
            )

        # Setup TMA store for D
        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d,
            d_smem_layout,
            self.epi_tile,
        )
        tma_atom_d_col, tma_tensor_d_col = None, None
        if cutlass.const_expr(not self.forward_only):
            tma_atom_d_col, tma_tensor_d_col = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                d_col,
                d_smem_layout,
                self.epi_tile,
            )

        # Compute grid size
        # When use_mono_increase_expert_idx is enabled, use N-major rasterization
        # (raster_along_m=False) to ensure M coordinates are monotonically non-decreasing
        output_shape = (m, n, l)
        raster_along_m = not self.use_mono_increase_expert_idx
        self.tile_sched_params, grid = self._compute_grid(
            output_shape,
            self.cta_tile_shape_mnk_d,
            self.cluster_shape_mn,
            max_active_clusters,
            raster_along_m=raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorageFP8:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sD_col: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Amax reduction shared memory (one FP32 per epilogue warp)
            # Use smaller alignment for amax since it's only 16 bytes
            sAmax: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_epilog_warps],
                1,  # byte alignment
            ]
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                1,  # byte alignment
            ]

        @cute.struct
        class SharedStorageFP4:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Amax reduction shared memory (one FP32 per epilogue warp)
            # Use smaller alignment for amax since it's only 16 bytes
            sAmax: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_epilog_warps],
                1,  # byte alignment
            ]
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                1,  # byte alignment
            ]

        if cutlass.const_expr(self.generate_sfd):
            self.shared_storage = SharedStorageFP8
        else:
            self.shared_storage = SharedStorageFP4

        # Launch the kernel synchronously
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
            prob,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
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
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_d_col: Optional[cute.CopyAtom],
        mD_col_mnl: Optional[cute.Tensor],
        mSFDRow_mnl: Optional[cute.Tensor],
        mSFDCol_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        mAmax_tensor: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        prob: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.forward_only):
                cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_d)
            if cutlass.const_expr(self.generate_sfd_col):
                cpasync.prefetch_descriptor(tma_atom_d_col)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
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

        # Initialize acc_pipeline (barrier) and states
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

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/D/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        # placeholder again
        sD_col = sD
        if cutlass.const_expr(self.generate_sfd_col):
            sD_col = storage.sD_col.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # Shared memory for amax reduction (one FP32 per epilogue warp)
        # Simple 1D layout. The allocation always here if no amax is generated,
        # as the overhead is minimal and we want to keep the code simple.
        amax_layout = cute.make_layout((self.num_epilog_warps,))
        sAmax = storage.sAmax.get_tensor(amax_layout)
        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        if cutlass.const_expr(self.discrete_col_sfd):
            info_layout = cute.make_layout((6, self.num_tile_stage), stride=(1, 6))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        # (bM, bN, loopM, loopN, loopL)
        if cutlass.const_expr(not self.forward_only):
            gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gD_mnl = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
        # placeholder, it will be eventually removed by compiler as we won't do any store to it in FP4 mode
        gD_col_mnl = gD_mnl
        if cutlass.const_expr(self.generate_sfd_col):
            gD_col_mnl = cute.local_tile(
                mD_col_mnl,
                cute.slice_(self.mma_tiler_d, (None, None, 0)),
                (None, None, None),
            )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/D
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        if cutlass.const_expr(not self.forward_only):
            tCgC = thr_mma.partition_C(gC_mnl)
        tCgD = thr_mma.partition_C(gD_mnl)
        # placeholder, same as above
        tCgD_col = tCgD
        if cutlass.const_expr(self.generate_sfd_col):
            tCgD_col = thr_mma.partition_C(gD_col_mnl)
        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)

        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )

        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/D
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage_overlapped))
            # (MMA, MMA_M, MMA_N, STAGE)
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
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            lane_id = tidx % 32

            # ============================================================
            # 1. Parallel load padded_offsets into registers
            # ============================================================
            # Each thread loads ceil(expert_cnt / 32) offsets
            # Layout: padded_offsets_in_lanes[i] corresponds to global offsets[lane_id + i * 32]
            num_offsets_per_thread: cutlass.Constexpr = self.num_offsets_per_thread
            expert_cnt: cutlass.Constexpr = self.expert_cnt

            padded_offsets_in_lanes = cute.make_rmem_tensor((num_offsets_per_thread,), cutlass.Int32)

            for i in cutlass.range(num_offsets_per_thread, unroll_full=True):
                idx = lane_id + i * 32
                if idx < expert_cnt:
                    padded_offsets_in_lanes[i] = padded_offsets[idx]
                else:
                    padded_offsets_in_lanes[i] = cutlass.Int32(0x7FFFFFFF)  # MAX_INT

            # ============================================================
            # 2. Get total_m and num_valid_tiles using warp shuffle
            # ============================================================
            last_offset_round: cutlass.Constexpr = (expert_cnt - 1) // 32
            last_offset_lane: cutlass.Constexpr = (expert_cnt - 1) % 32

            # Shuffle to get total_m from the lane holding the last offset
            total_m = cute.arch.shuffle_sync(
                padded_offsets_in_lanes[last_offset_round],
                offset=last_offset_lane,
                mask=0xFFFFFFFF,
                mask_and_clamp=31,
            )

            mma_tile_m: cutlass.Constexpr = self.mma_tiler[0]
            num_valid_tiles = total_m // mma_tile_m

            # ============================================================
            # 3. Persistent tile scheduling loop
            # ============================================================
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            # For incremental expert lookup (when use_mono_increase_expert_idx=True):
            # Initialize prev_expert_idx to -1 and prev_expert_end to 0
            # These will be updated after each tile is processed
            use_mono_increase: cutlass.Constexpr = self.use_mono_increase_expert_idx
            prev_expert_idx = cutlass.Int32(-1)
            prev_expert_end = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                tile_m_start = mma_tile_coord_m * mma_tile_m

                if mma_tile_coord_m < num_valid_tiles:
                    # ============================================================
                    # 4. Warp-parallel expert_idx lookup using ballot + ffs
                    # ============================================================
                    if cutlass.const_expr(use_mono_increase):
                        # Incremental search: O(1) average, exploits expert_idx monotonicity
                        expert_idx, prev_expert_idx, prev_expert_end = search_expert_idx_incremental(
                            padded_offsets_in_lanes,
                            tile_m_start,
                            num_offsets_per_thread,
                            prev_expert_idx,
                            prev_expert_end,
                        )
                    else:
                        # Full search: O(E/32) complexity
                        expert_idx = search_expert_idx_full(
                            padded_offsets_in_lanes,
                            tile_m_start,
                            num_offsets_per_thread,
                        )

                    # ============================================================
                    # 5. Compute tokens_in_group and tokens_presum from padded_offsets
                    # ============================================================
                    tokens_presum_this_group = cutlass.Int32(0)
                    tokens_in_group = cutlass.Int32(0)

                    if cutlass.const_expr(self.discrete_col_sfd):
                        # Get tokens_presum_this_group (start of current expert)
                        if expert_idx > 0:
                            prev_expert_round = (expert_idx - 1) // 32
                            prev_expert_lane = (expert_idx - 1) % 32
                            tokens_presum_this_group = cute.arch.shuffle_sync(
                                padded_offsets_in_lanes[prev_expert_round],
                                offset=prev_expert_lane,
                                mask=0xFFFFFFFF,
                                mask_and_clamp=31,
                            )

                        # Get tokens_presum_next_group (end of current expert)
                        cur_expert_round = expert_idx // 32
                        cur_expert_lane = expert_idx % 32
                        tokens_presum_next_group = cute.arch.shuffle_sync(
                            padded_offsets_in_lanes[cur_expert_round],
                            offset=cur_expert_lane,
                            mask=0xFFFFFFFF,
                            mask_and_clamp=31,
                        )

                        tokens_in_group = tokens_presum_next_group - tokens_presum_this_group

                    # ============================================================
                    # 6. Write tile info to shared memory
                    # ============================================================
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                        sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(work_tile.is_valid_tile)
                        if cutlass.const_expr(self.discrete_col_sfd):
                            # number of tokens in this group
                            sInfo[(4, tile_info_producer_state.index)] = tokens_in_group
                            # token prefix sum of this group
                            sInfo[(5, tile_info_producer_state.index)] = tokens_presum_this_group
                        # fence view async shared
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )

                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Send invalid tile signal to consumers
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, 0)]

                # Apply SFB slicing when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
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

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB
            #
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

            #
            # Persistent tile scheduling loop
            #
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acd_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acd_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acd_producer_state.phase ^ 1
                else:
                    acc_stage_index = acd_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Apply TMEM pointer offset shift when cta_tile_shape_n=192 or cta_tile_shape_n=64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acd_producer_state, peek_acc_empty_status)
                #
                # Mma mainloop
                #

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acd_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acd_producer_state.advance()
                if acd_producer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acd_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgD, epi_tile, use_2cta_instrs)

            # Modified by Marin: the C and column-quantized D partitions exist only without
            # forward_only. They do not rebind the TMA atom arguments, which the partition returns as is.
            if cutlass.const_expr(not self.forward_only):
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, self.epi_tile_c, sC
                )[1:]

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rD, epi_tidx, sD)
            (
                tma_atom_d,
                bSG_sD,
                bSG_gD_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD, epi_tile, sD)

            if cutlass.const_expr(not self.forward_only):
                tTR_rD_col = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.d_dtype)
                tiled_copy_r2s, tRS_rD_col, tRS_sD_col = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rD_col, epi_tidx, sD_col
                )
                bSG_sD_col, bSG_gD_col_partitioned = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_d_col, tCgD_col, epi_tile, sD_col
                )[1:]

            if cutlass.const_expr(self.generate_sfd):
                norm_const = norm_const_tensor[0]
                regPerSubtile = self.mma_tiler_d[1] // 32
                sfd_row_tile = (
                    cute.make_layout(128),
                    cute.make_layout(32 * regPerSubtile),
                )
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFDRow_mnl = cute.local_tile(mSFDRow_mnl, sfd_row_tile, (None, None, None))
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFDRow_mnl = thr_copy_t2r.partition_D(gSFDRow_mnl)
                tCgSFDRow_mnl = cute.filter_zeros(tCgSFDRow_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFDRow = cute.make_rmem_tensor(tCgSFDRow_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype)
                tCrSFDRow_pvscale = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)
                d_rcp_limits = self.get_dtype_rcp_limits(self.d_dtype)

            if cutlass.const_expr(self.generate_sfd_col):
                # both SFDs are stored in row major mode.
                sfd_col_tile = sfd_row_tile
                gSFDCol_mnl = cute.local_tile(mSFDCol_mnl, sfd_col_tile, (None, None, None))
                thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
                val_layout = cute.make_ordered_layout((1,), order=(0,))
                copy_atom_sfd_col = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    gSFDCol_mnl.element_type,
                    num_bits_per_copy=8,
                )
                tiled_copy_sfd_col = cute.make_tiled_copy_tv(copy_atom_sfd_col, thr_layout, val_layout)
                thr_copy_sfd_col = tiled_copy_sfd_col.get_slice(tidx)
                tCgSFDCol_mnl = thr_copy_sfd_col.partition_D(cute.filter_zeros(gSFDCol_mnl))
                tCgSFDCol_mnl = cute.filter_zeros(tCgSFDCol_mnl)
                tCrSFDCol = cute.make_rmem_tensor(tCgSFDRow_mnl[(None, None, None, 0, 0, 0)].shape, self.sf_dtype)
                tCrSFDCol_pvscale = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)
                tCrSFDCol_qpvscale_up_fp32 = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            d_pipeline = None
            # Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )
            d_col_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            if cutlass.const_expr(self.discrete_col_sfd):
                tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(tile_info.shape[0], unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                if cutlass.const_expr(not self.forward_only):
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            mma_tile_coord_mnl[0],
                            mma_tile_coord_mnl[1],
                            0,
                        )
                    ]
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    bSG_gD_col = bSG_gD_col_partitioned[
                        (
                            None,
                            None,
                            None,
                            mma_tile_coord_mnl[0],
                            mma_tile_coord_mnl[1],
                            0,
                        )
                    ]
                    bSG_gD_col = cute.group_modes(bSG_gD_col, 1, cute.rank(bSG_gD_col))
                bSG_gD = bSG_gD_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                if cutlass.const_expr(self.generate_sfd):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgSFDRow_mn = tCgSFDRow_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]
                if cutlass.const_expr(self.generate_sfd_col):
                    tCgSFDCol_mnl_new = tCgSFDCol_mnl
                    if cutlass.const_expr(self.discrete_col_sfd):
                        tCgSFDCol_mnl_new = self.create_and_partition_new_SFDCol(tile_info, mSFDCol_mnl)
                    tCgSFDCol_mn = tCgSFDCol_mnl_new[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                if cutlass.const_expr(self.generate_amax):
                    thread_tile_amax = cutlass.Float32(0.0)

                #
                # Get PROB
                # Note, it always assumes T2R_M/EPI_M is 1, otherwise it will break the result.
                #
                if cutlass.const_expr(prob is not None):
                    mPosition = tile_info[0] * self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape) + tidx
                    mProb = prob[mPosition, 0, 0].to(cutlass.Float32)
                else:
                    mProb = cutlass.Float32(1.0)

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(0, subtile_cnt, 2, unroll=1):
                    real_subtile_idx = subtile_idx // 2
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            # Subtile always iterates on N dimension as we only have 4x1DP tmem load pattern for cta_tile_m = 128 cases.
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - subtile_idx // 2
                            )

                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]
                    tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                    #
                    # Async arrive accumulator buffer empty ealier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    #
                    # Apply alpha
                    #
                    if cutlass.const_expr(self.vector_f32):
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                            tTR_rAcc_up[i], tTR_rAcc_up[i + 1] = cute.arch.mul_packed_f32x2(
                                (tTR_rAcc_up[i], tTR_rAcc_up[i + 1]),
                                (
                                    cutlass.Float32(alpha_val),
                                    cutlass.Float32(alpha_val),
                                ),
                                rnd="rn",
                                ftz=False,
                            )
                            tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1] = cute.arch.mul_packed_f32x2(
                                (tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1]),
                                (
                                    cutlass.Float32(alpha_val),
                                    cutlass.Float32(alpha_val),
                                ),
                                rnd="rn",
                                ftz=False,
                            )
                    else:
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                            tTR_rAcc_up[i] = tTR_rAcc_up[i] * cutlass.Float32(alpha_val)
                            tTR_rAcc_gate[i] = tTR_rAcc_gate[i] * cutlass.Float32(alpha_val)

                    #
                    # Store to C tensor
                    #
                    if cutlass.const_expr(not self.forward_only):
                        self.store_c(
                            tiled_copy_r2s,
                            tma_atom_c,
                            warp_idx,
                            tTR_rAcc_up,
                            tTR_rAcc_gate,
                            tRS_rC,
                            tRS_sC,
                            bSG_gC,
                            bSG_sC,
                            c_pipeline,
                            num_prev_subtiles,
                            real_subtile_idx,
                        )

                    acc_vec_up = tTR_rAcc_up.load()
                    acc_vec_gate = tTR_rAcc_gate.load()

                    # SwiGlu
                    tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                    if cutlass.const_expr(self.vector_f32):
                        # SwiGlu Packed Version
                        LOG2_E = cutlass.Float32(1.4426950408889634)
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                            tCompute_log2e = cute.arch.mul_packed_f32x2(
                                (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                (-LOG2_E, -LOG2_E),
                                rnd="rn",
                                ftz=False,
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.add_packed_f32x2(
                                (
                                    cute.math.exp2(tCompute_log2e[0], fastmath=True),
                                    cute.math.exp2(tCompute_log2e[1], fastmath=True),
                                ),
                                (1.0, 1.0),
                            )
                            tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                            tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                rnd="rn",
                                ftz=False,
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (acc_vec_up[i], acc_vec_up[i + 1]),
                                rnd="rn",
                                ftz=False,
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (mProb, mProb),
                                rnd="rn",
                                ftz=False,
                            )
                    else:
                        # SwiGlu Unpacked Version
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                            tCompute[i] = acc_vec_up[i] * silu_f32(acc_vec_gate[i], fastmath=True)
                            tCompute[i] = tCompute[i] * mProb

                    #
                    # Generate amax
                    #
                    if cutlass.const_expr(self.generate_amax):
                        thread_tile_amax = self.amax_reduction_per_thread(tCompute, thread_tile_amax)

                    if cutlass.const_expr(self.generate_sfd):
                        if cutlass.const_expr(self.generate_sfd_col):
                            tCompute_col = cute.make_rmem_tensor(tCompute.layout, tCompute.element_type)
                            tCompute_col.store(tCompute.load())
                        #
                        # Generate row major SFD
                        #
                        self.quant_sfd_row(
                            real_subtile_idx,
                            tiled_copy_r2s,
                            tCompute,
                            tCrSFDRow_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD,
                            tile_info,
                        )

                        # Assume subtile partitioned always happens on n dimension
                        sfd_row_idx_mn = (
                            tile_info[0],
                            tile_info[1],
                        )
                        tCgSFDRow = tCgSFDRow_mn[
                            (
                                None,
                                None,
                                None,
                                *sfd_row_idx_mn,
                            )
                        ]

                        if cutlass.const_expr(self.generate_sfd_col):
                            #
                            # Generate col major SFD
                            #
                            self.quant_sfd_col(
                                real_subtile_idx,
                                tiled_copy_r2s,
                                tCompute_col,
                                tCrSFDCol_pvscale,
                                norm_const,
                                d_rcp_limits,
                                tRS_rD_col,
                                tile_info,
                            )
                            sfd_col_idx_mn = sfd_row_idx_mn
                            if cutlass.const_expr(self.discrete_col_sfd):
                                sfd_col_idx_mn = (
                                    tile_info[0] - tile_info[5] // 128,
                                    tile_info[1],
                                )
                            tCgSFDCol = tCgSFDCol_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfd_col_idx_mn,
                                )
                            ]

                        if subtile_idx == subtile_cnt - 2:
                            if sfd_row_idx_mn[1] * 32 * regPerSubtile < cute.size(
                                cute.shape(mSFDRow_mnl.layout, mode=[1])
                            ):
                                tCrSFDRow.store(tCrSFDRow_pvscale.load().to(self.sf_dtype))
                                cute.autovec_copy(tCrSFDRow, tCgSFDRow)
                            if cutlass.const_expr(self.generate_sfd_col):
                                if sfd_col_idx_mn[1] * 32 * regPerSubtile < cute.size(
                                    cute.shape(mSFDCol_mnl.layout, mode=[1])
                                ):
                                    tCrSFDCol.store(tCrSFDCol_pvscale.load().to(self.sf_dtype))
                                    cute.autovec_copy(tCrSFDCol, tCgSFDCol)
                    else:
                        #
                        # Convert to D type
                        #
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rD.store(acc_vec.to(self.d_dtype))

                    #
                    # Store D to shared memory
                    #
                    d_buffer = num_prev_subtiles % self.num_d_stage
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD,
                        tRS_sD[(None, None, None, d_buffer)],
                    )
                    if cutlass.const_expr(self.generate_sfd_col):
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_col,
                            tRS_sD_col[(None, None, None, d_buffer)],
                        )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    #
                    # TMA store D to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, d_buffer)],
                            bSG_gD[(None, real_subtile_idx)],
                        )
                        if cutlass.const_expr(self.generate_sfd_col):
                            cute.copy(
                                tma_atom_d_col,
                                bSG_sD_col[(None, d_buffer)],
                                bSG_gD_col[(None, real_subtile_idx)],
                            )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        if cutlass.const_expr(self.forward_only):
                            # Modified by Marin: the C store's acquire also waits on these TMA store
                            # groups. Without it, wait here before a later subtile reuses this D buffer.
                            d_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(tile_info.shape[0], unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                # Perform amax reduction after all subtiles are processed
                if cutlass.const_expr(self.generate_amax):
                    gAmax = mAmax_tensor[(expert_idx, None)].iterator.llvm_ptr  # First element
                    self.amax_reduction_per_warp_and_cta(thread_tile_amax, warp_idx, sAmax, gAmax)

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C/D store complete
            #
            c_pipeline.producer_tail()
            d_pipeline.producer_tail()

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
