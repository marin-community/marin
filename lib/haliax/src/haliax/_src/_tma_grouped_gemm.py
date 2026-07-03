# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import os
from inspect import isclass
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import jax
import jax.numpy as jnp
from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

"""
Grouped GEMM (C_g = A_g * B_g for each group g) for the NVIDIA Hopper architecture
using CuTe DSL.

Vendored from NVIDIA CUTLASS 4.5.2 (``examples/python/CuTeDSL/hopper/grouped_gemm.py``),
torch harness stripped, with these local patches for the haliax ``cute_ragged_dot``
FFI (``cutlass.jax.cutlass_call``) path:
  * The broken raw-NVVM non-mcast TMA-load helper is removed entirely; all A/B
    loads go through the standard ``cute.copy`` TMA path (the NVVM helper hit
    ``CUDA_ERROR_ILLEGAL_INSTRUCTION`` under the 4.5.2 toolchain).
  * The tile scheduler is the modern ``cutlass.utils.StaticPersistentGroupTileScheduler``
    (whose group search returns a real ``found`` validity), NOT the deprecated
    ``GroupedGemmTileSchedulerHelper`` (which hangs when the tile total is
    over-estimated). This lets a static upper bound on ``total_num_clusters``
    predicate surplus tiles away -- required for dynamic per-group token counts
    under ``jax.jit``.
  * ``_FixedTensorMapManager`` routes the per-group tensormap publish through the
    stable GMEM update path.
  * An ``mScale`` device operand is threaded into the epilogue: the f32
    accumulator is DIVIDED by ``mScale[0]`` before the FP8/bf16 output cast (the
    haliax dequantize convention).

This kernel extends hopper/dense_gemm_persistent.py with per-group TMA tensor map updates
and a group-aware persistent tile scheduler (StaticPersistentGroupTileScheduler).

Key features:
    - WGMMA + TMA + persistent warp-specialized kernel (inherited from dense_gemm_persistent)
    - Per-group A/B/C TMA descriptor updates (tensor map) via GMEM or SMEM mode
    - DMA warp group: loads A/B tiles, updates tensor maps A/B on group boundary
    - MMA warp group: performs WGMMA, updates tensor map C on group boundary, stores C

Constraints (same as dense_gemm_persistent.py plus):
* 8-bit (fp8) or 16-bit (fp16/bf16) inputs; 8-bit A/B may differ (E4M3/E5M2)
* l (batch) must be 1 for each group
* CTA tile M: 64/128, N: 64/128/256
* Cluster shape M/N: power of 2, total <= 4
* Contiguous dim must be 16-byte aligned
"""


class _FixedTensorMapManager(utils.TensorMapManager):
    """Per-group tensormap manager that publishes updates through the GMEM path.

    In SMEM update mode the base class stages the descriptor in SMEM; this
    override always publishes via ``update_tma_descriptor`` on the GMEM
    descriptor + ``fence_tma_desc_release`` -- the proven-stable path on the
    4.5.2 toolchain. ``tensormap_smem_ptr`` is unused (kept for the base-class
    call signature).
    """

    @_dsl_user_op
    @cute.jit
    def update_tensormap(
        self,
        tensor_gmem,
        tma_copy_atom,
        tensormap_gmem_ptr,
        warp_id: int,
        tensormap_smem_ptr,
        *,
        loc=None,
        ip=None,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx(loc=loc, ip=ip), loc=loc, ip=ip)
        if warp_idx == warp_id:
            with cute.arch.elect_one(loc=loc, ip=ip):
                cute.arch.cp_async_bulk_commit_group(loc=loc, ip=ip)
                cute.arch.cp_async_bulk_wait_group(0, read=True, loc=loc, ip=ip)
            cute.arch.sync_warp(loc=loc, ip=ip)
            for atom, tensor, gptr in zip(tma_copy_atom, tensor_gmem, tensormap_gmem_ptr):
                cute.nvgpu.cpasync.update_tma_descriptor(atom, tensor, gptr, loc=loc, ip=ip)
            cute.arch.sync_warp(loc=loc, ip=ip)
            cute.nvgpu.cpasync.fence_tma_desc_release(loc=loc, ip=ip)


class HopperGroupedGemmPersistentKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float16
          A and B must have the same data type
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout
        - Int8/Uint8
          A and B can have different types (Int8/Uint8)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32/Float16 (for all floating point inputs)
        - Int32 (for Int8/Uint8 inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperGroupedGemmPersistentKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    bytes_per_tensormap = 128
    num_tensormaps = 3  # A, B, C

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
        wgrad: bool = False,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        :param wgrad: When True, the ragged axis is the GEMM contraction (token-M
            weight gradient): A/B share ONE aligned full-buffer base and the per-group
            token offset is folded into the TMA element coordinate, with the per-group
            descriptor contraction extent bounding the ragged tail (TMA zero-fill).
        :type wgrad: bool
        """

        # Ragged-contraction (weight-gradient) mode. In this mode the per-group token
        # offset is an element coordinate into a shared full-tensor A/B descriptor
        # (whose base stays 16B-aligned) instead of a per-group base-pointer advance,
        # and the descriptor's contraction extent = offset + M_g zero-fills the tail.
        self.wgrad = wgrad

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (2, 1, 1) if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128 else (1, 1, 1)
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (self.num_dma_warp_groups + self.num_mma_warp_groups) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_mma_threads)

        # Grouped GEMM: tensor map update mode
        self.tensormap_update_mode = tensormap_update_mode
        # Delegate A/B tensor map init to MMA warp for better latency hiding (SMEM mode)
        self.delegate_tensormap_ab_init = tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        # barrier_id=2 (barrier_id=1 is already used by epilog_sync_barrier)
        # Only the load warp (32 threads) + all MMA threads participate:
        # DMA warps 1-3 are idle and never reach this barrier.
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_threads + 32,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Cluster tile shape used by group tile scheduler
        self.cluster_tile_shape_mnk = (
            self.tile_shape_mnk[0] * self.cluster_shape_mn[0],
            self.tile_shape_mnk[1] * self.cluster_shape_mn[1],
            self.tile_shape_mnk[2],
        )

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        scale: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
        offsets: cute.Tensor = None,
    ):
        """Execute the grouped GEMM operation.

        :param initial_a: Carries dtype+majorness only (shape irrelevant).
        :param initial_b: Carries dtype+majorness only (shape irrelevant).
        :param initial_c: Carries dtype+majorness only (shape irrelevant).
        :param group_count: Number of GEMM groups (compile-time constant).
        :param problem_shape_mnkl: Device tensor of shape (G, 4) Int32 with (M,N,K,L) per group.
        :param strides_abc: Device tensor of shape (G, 3, 2) Int32 with strides per group.
        :param tensor_address_abc: Device tensor of shape (G, 3) Int64 with base ptrs per group.
        :param scale: Device tensor of shape (1,) Float32; the epilogue divides the accumulator by scale[0].
        :param total_num_clusters: Total clusters across all groups (compile-time constant).
        :param tensormap_cute_tensor: Tensor map workspace, shape (num_sms, 3, 16) Int64.
        :param max_active_clusters: Max active clusters (compile-time constant).
        :param stream: CUDA stream.
        """

        # Setup static attributes from initial tensor dtype/layout
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(initial_a)
        self.b_layout = utils.LayoutEnum.from_tensor(initial_b)
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)

        if cutlass.const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16, float8, or int8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            initial_a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            initial_b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            initial_c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            total_num_clusters,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        # Number of Int64 words needed for the SMEM tensor map buffer (0 in GMEM mode)
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else HopperGroupedGemmPersistentKernel.num_tensormaps
            * HopperGroupedGemmPersistentKernel.bytes_per_tensormap
            // 8
        )

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[cutlass.Int64, self.size_tensormap_in_i64]
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            scale,
            tensormap_cute_tensor,
            offsets,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        scale: cute.Tensor,
        tensormaps: cute.Tensor,
        offsets: cute.Tensor = None,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param offsets: (wgrad only) per-group exclusive-prefix-sum token offset
            (E,) Int32; the group's contraction slice is ``[offsets[g], offsets[g]+M_g)``
            of the shared token axis. Folded into the A/B TMA element coordinate.
        :type offsets: cute.Tensor

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Parameters for the persistent tile scheduler
        :type tile_sched_params: utils.PersistentTileSchedulerParams
        """

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch Tma desc
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Generate smem tensor A/B
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # Partition global tensor for TiledMMA_A/B/C
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Setup per-SM tensor map pointers (shared by DMA and MMA warps)
        #
        grid_dim = cute.arch.grid_dim()
        bid = cute.arch.block_idx()
        sm_idx = bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]

        tensormap_manager = _FixedTensorMapManager(
            self.tensormap_update_mode,
            HopperGroupedGemmPersistentKernel.bytes_per_tensormap,
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(sm_idx, 0, None)].iterator)
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(sm_idx, 1, None)].iterator)
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(sm_idx, 2, None)].iterator)

        # SMEM buffer pointers for tensor maps (only non-None in SMEM mode)
        if cutlass.const_expr(self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM):
            smem_tm_base = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = smem_tm_base
            tensormap_b_smem_ptr = smem_tm_base + HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8
            tensormap_c_smem_ptr = smem_tm_base + 2 * HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8
        else:
            tensormap_a_smem_ptr = None
            tensormap_b_smem_ptr = None
            tensormap_c_smem_ptr = None

        tile_sched_params_for_sched = tile_sched_params

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

        #
        # DMA warp group (load A/B with TMA, update tensor maps A/B per group)
        #
        if warp_idx == self.load_warp_id:
            # Initialize tensor maps A/B (either here or delegated to MMA warp)
            if cutlass.const_expr(not self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(tma_atom_a, tensormap_a_ptr, self.load_warp_id)
                tensormap_manager.init_tensormap_from_atom(tma_atom_b, tensormap_b_ptr, self.load_warp_id)
                tensormap_manager.fence_tensormap_initialization()
            else:
                # Delegate path: wait for MMA warp to finish A/B tensor map init.
                # Must be unconditional (before the tile loop) so every CTA
                # participates even when it processes zero tiles.
                self.tensormap_ab_init_barrier.arrive_and_wait()

            last_group_idx = cutlass.Int32(-1)

            # Create a per-warp scheduler (same state — each warp runs its own instance)
            tile_sched = utils.StaticPersistentGroupTileScheduler.create(
                tile_sched_params_for_sched,
                bid,
                grid_dim,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
                group_count,
                problem_sizes_mnkl,
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                cur_group_idx = grouped_info.group_idx
                cur_k_tile_cnt = grouped_info.cta_tile_count_k

                if cur_k_tile_cnt != 0:
                    is_group_changed = cur_group_idx != last_group_idx

                    if is_group_changed:
                        real_a = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.a_dtype,
                            (
                                grouped_info.problem_shape_m,
                                grouped_info.problem_shape_n,
                                grouped_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            0,
                            offsets,
                        )
                        real_b = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.b_dtype,
                            (
                                grouped_info.problem_shape_m,
                                grouped_info.problem_shape_n,
                                grouped_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            1,
                            offsets,
                        )
                        tensormap_manager.update_tensormap(
                            (real_a, real_b),
                            (tma_atom_a, tma_atom_b),
                            (tensormap_a_ptr, tensormap_b_ptr),
                            self.load_warp_id,
                            (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                        )
                        tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_ptr)

                    mma_tile_coord_mnl = (
                        grouped_info.cta_tile_idx_m,
                        grouped_info.cta_tile_idx_n,
                        0,
                    )
                    # Wgrad: the group's tokens are the contraction slice
                    # [off, off+M_g) of the SHARED A/B buffer. Fold off into the TMA
                    # element coordinate (base stays the 16B-aligned buffer origin);
                    # the per-group descriptor extent = off+M_g zero-fills the tail.
                    if cutlass.const_expr(self.wgrad):
                        off_a = offsets[cur_group_idx]
                        gA_off = cute.local_tile(
                            cute.domain_offset((0, off_a, 0), mA_mkl),
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (None, None, None),
                        )
                        gB_off = cute.local_tile(
                            cute.domain_offset((0, off_a, 0), mB_nkl),
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (None, None, None),
                        )
                        _, tAgA_cur = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_a,
                            a_cta_crd,
                            a_cta_layout,
                            cute.group_modes(sA, 0, 2),
                            cute.group_modes(gA_off, 0, 2),
                        )
                        _, tBgB_cur = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_b,
                            b_cta_crd,
                            b_cta_layout,
                            cute.group_modes(sB, 0, 2),
                            cute.group_modes(gB_off, 0, 2),
                        )
                    else:
                        tAgA_cur = tAgA
                        tBgB_cur = tBgB
                    tAgA_slice = tAgA_cur[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                    tBgB_slice = tBgB_cur[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                    # Loop-invariant generic TMA descriptor pointers for cute.copy
                    # (the per-group tensormap is updated in gmem above).
                    tma_a_desc_ptr_copy = tensormap_manager.get_tensormap_ptr(
                        tensormap_a_ptr, cute.AddressSpace.generic
                    )
                    tma_b_desc_ptr_copy = tensormap_manager.get_tensormap_ptr(
                        tensormap_b_ptr, cute.AddressSpace.generic
                    )
                    mainloop_producer_state.reset_count()
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        # Standard cute.copy TMA load; handles the multicast mask
                        # for multi-CTA clusters and is the only supported load path.
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, k_tile)],
                            tAsA[(None, mainloop_producer_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                            mcast_mask=a_mcast_mask,
                            tma_desc_ptr=tma_a_desc_ptr_copy,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, k_tile)],
                            tBsB[(None, mainloop_producer_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                            mcast_mask=b_mcast_mask,
                            tma_desc_ptr=tma_b_desc_ptr_copy,
                        )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()
                else:
                    pass  # k_tile_cnt == 0: tensor map init already done before loop

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        #
        # MMA warp group (WGMMA + epilogue, update tensor map C per group)
        #
        if not is_dma_warp_group:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            # MMA warp always initializes tensor map C
            tensormap_manager.init_tensormap_from_atom(tma_atom_c, tensormap_c_ptr, self.epi_store_warp_id)
            # When delegating, MMA warp also initializes A/B and signals DMA warp
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(tma_atom_a, tensormap_a_ptr, self.epi_store_warp_id)
                tensormap_manager.init_tensormap_from_atom(tma_atom_b, tensormap_b_ptr, self.epi_store_warp_id)
                self.tensormap_ab_init_barrier.arrive_and_wait()

            tensormap_manager.fence_tensormap_initialization()

            tile_sched = utils.StaticPersistentGroupTileScheduler.create(
                tile_sched_params_for_sched,
                bid,
                grid_dim,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
                group_count,
                problem_sizes_mnkl,
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Partition for epilogue
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )

            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(),
                    4,
                ),
                self.c_dtype,
            )

            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s,
                tiled_copy_C_Atom,
            )

            # (R2S, R2S_M, R2S_N, PIPE_D)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group)
            # (t)hread-partition for (r)egister to (s)mem copy (tRS_)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            # (R2S, R2S_M, R2S_N)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            # Allocate D registers.
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1

            # Initialize tma store pipeline
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            last_group_idx_mma = cutlass.Int32(-1)

            # Output scale is uniform across all tiles; read it once (haliax
            # dequantize convention: the epilogue DIVIDES the accumulator by it).
            out_scale = self.acc_dtype(scale[0])

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                cur_group_idx = grouped_info.group_idx
                cur_k_tile_cnt = grouped_info.cta_tile_count_k

                # Per-group tensor map C update (only epi_store warp issues it)
                is_group_changed = cur_group_idx != last_group_idx_mma
                if is_group_changed and warp_idx == self.epi_store_warp_id:
                    real_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_info.problem_shape_m,
                            grouped_info.problem_shape_n,
                            grouped_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,
                        offsets,
                    )
                    tensormap_manager.update_tensormap(
                        (real_c,),
                        (tma_atom_c,),
                        (tensormap_c_ptr,),
                        self.epi_store_warp_id,
                        (tensormap_c_smem_ptr,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_c_ptr)

                mma_tile_coord_mnl = (
                    grouped_info.cta_tile_idx_m,
                    grouped_info.cta_tile_idx_n,
                    0,
                )
                gC_mnl_slice = gC_mnl[(None, None, *mma_tile_coord_mnl)]

                # MAINLOOP
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                prologue_mma_cnt = cutlass.min(k_pipe_mmas, cur_k_tile_cnt)

                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()

                for k_tile in cutlass.range(prologue_mma_cnt, cur_k_tile_cnt, 1, unroll=1):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    # Wait on the wgmma barrier for WGMMA to complete
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                # Epilogue
                tCgC_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                # thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition,
                )

                epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))

                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Scale (divide) in registers, then convert to the output dtype.
                    acc_vec = tRS_rD.load() / out_scale
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    # Copy from D registers to shared memory
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    # Copy from shared memory to global memory (TMA store with updated desc)
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr, cute.AddressSpace.generic
                            ),
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                    self.epilog_sync_barrier.arrive_and_wait()

                last_group_idx_mma = cur_group_idx
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
        offsets: cute.Tensor = None,
    ):
        """Construct a global tensor for tensormap update from per-group metadata.

        :param group_idx: Index of the current group.
        :param dtype: Element type of the tensor (A, B, or C).
        :param problem_shape_mnk: (M, N, K) of the current group.
        :param strides_abc: Tensor of strides, shape (G, 3, 2), dtype Int32.
        :param tensor_address_abc: Tensor of base ptrs, shape (G, 3), dtype Int64.
        :param tensor_index: 0=A, 1=B, 2=C.
        :param offsets: (wgrad only) per-group token offsets; the A/B descriptor
            contraction extent is set to ``offsets[group_idx] + K`` so TMA zero-fills
            every token at or past the group's ragged end.
        """
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)):
            raise TypeError(f"dtype must be a type of cutlass.Numeric, got {type(dtype)}")
        tensor_gmem_ptr = cute.make_ptr(dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16)

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # Wgrad A/B share one aligned full-buffer descriptor; the contraction extent
        # is offset+M_g so coordinates at/after the group's ragged end are zero-filled.
        k_ext = problem_shape_mnk[2]
        if cutlass.const_expr(self.wgrad):
            if cutlass.const_expr(tensor_index != 2):
                k_ext = offsets[group_idx] + problem_shape_mnk[2]

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k_ext, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k_ext, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage

        mbar_helpers_bytes = 1024

        ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple]:
        """Compute tile scheduler params and grid shape for grouped GEMM.

        :param total_num_clusters: Total clusters across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: (tile_sched_params, grid)
        :rtype: tuple
        """
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(problem_shape_ntile_mnl, (*cluster_shape_mn, 1))
        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        if a_dtype.width != b_dtype.width:
            is_valid = False
        if not a_dtype.is_same_kind(b_dtype):
            is_valid = False

        # for 8-bit types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and A type
        if a_dtype not in acc_ab_compatibility[acc_dtype]:
            is_valid = False

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[acc_dtype]:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid


# --------------------------------------------------------------------------- #
# JAX (cutlass_call) adapter
#
# Wraps HopperGroupedGemmPersistentKernel as a JAX custom call. A device-side
# prologue kernel (launched first on the same stream) fills ALL per-group
# metadata -- problem_shape_mnkl, strides_abc, tensor_address_abc -- from the
# traced ``group_sizes`` and the operand base pointers, so the whole thing runs
# under ``jax.jit`` with dynamic per-expert token counts and no host round-trip.
# --------------------------------------------------------------------------- #

# H100 has 132 SMs at occupancy 1 for this (128,256) 3-warpgroup fp8 kernel; the
# tensormap workspace is sized by the total CTA count and the persistent grid is
# capped at 132 / cluster_size clusters. HardwareInfo() needs a live CUDA context
# on the compile thread (unavailable on the FFI path), so this is hardcoded.
_H100_SMS = 132
_TILE_SHAPE_MN = (128, 256)
_TENSORMAP_BYTES = HopperGroupedGemmPersistentKernel.bytes_per_tensormap  # 128
_NUM_TENSORMAPS = HopperGroupedGemmPersistentKernel.num_tensormaps  # 3 (A, B, C)
_TENSORMAP_I64_WORDS = _TENSORMAP_BYTES // 8  # 16 Int64 words per tensormap
# 128-bit contiguous alignment for the fp8 TMA operands (16 fp8 elems).
_FP8_TMA_VEC = 16
# CuTe DSL target arch for Hopper (H100) — exported via CUTE_DSL_ARCH and passed as
# compile_options to every cutlass_call. All three uses must agree.
_HOPPER_CUTE_ARCH = "sm_90a"


def ensure_hopper_arch() -> None:
    """Resolve the CuTe target arch once and fail fast on non-Hopper GPUs.

    GPU detection on the FFI compile thread can silently default to ``sm_100a``
    (Blackwell) -> a cubin that will not load on an H100. Derive the arch from
    the live JAX device and export ``CUTE_DSL_ARCH`` if the caller has not set
    it. Only called on the GPU backend (guarded by ``cute_available``).
    """
    dev = jax.devices()[0]
    cc = dev.compute_capability  # "9.0" on H100
    if not str(cc).startswith("9"):
        raise RuntimeError(
            "cute_ragged_dot TMA kernel requires a Hopper (sm_90) GPU; got device "
            f"{dev} with compute_capability={cc!r}"
        )
    os.environ.setdefault("CUTE_DSL_ARCH", _HOPPER_CUTE_ARCH)


def _cluster_tile_mn(cluster_shape_mn):
    return (_TILE_SHAPE_MN[0] * cluster_shape_mn[0], _TILE_SHAPE_MN[1] * cluster_shape_mn[1])


def _total_num_clusters_upper_bound(m_total, n, cluster_shape_mn, group_count):
    """Static upper bound on cluster tiles across all (ragged) groups.

    Per-group M is ragged with fixed total ``m_total``, so
    ``sum_g ceil_div(M_g, ctm) <= ceil_div(m_total, ctm) + (group_count - 1)``.
    Times the (fixed) N cluster-tile count. Surplus tiles are predicated away by
    the scheduler's ``found`` check, so an over-estimate is safe (never a hang).
    """
    ctm, ctn = _cluster_tile_mn(cluster_shape_mn)
    m_cluster_tiles = (m_total + ctm - 1) // ctm + (group_count - 1)
    n_cluster_tiles = (n + ctn - 1) // ctn
    return m_cluster_tiles * n_cluster_tiles


def _build_tma_launcher(
    *,
    cluster_shape_mn,
    group_count,
    n,
    k,
    total_num_clusters,
    max_active_clusters,
    a_bytes,
    b_bytes,
    c_bytes,
):
    """Build the stream-first ``@cute.jit`` adapter launcher for ``cutlass_call``."""
    # One prologue block; >= group_count threads, rounded up to a warp.
    addr_threads = max(32, ((group_count + 31) // 32) * 32)

    class TmaGroupedLauncher:
        @cute.jit
        def __call__(
            self,
            stream,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mScale: cute.Tensor,
            mInitA: cute.Tensor,
            mInitB: cute.Tensor,
            mInitC: cute.Tensor,
            mC: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddrI32: cute.Tensor,
            mTmapI32: cute.Tensor,
        ):
            # Reinterpret the Int32-doubled scratch buffers as real Int64 (the
            # library must run in a default x64-off JAX process; JAX would
            # otherwise truncate an Int64 buffer to 32 bits).
            addr_i64 = cute.recast_tensor(mAddrI32, cutlass.Int64)  # (E, 3)
            tmap_i64 = cute.recast_tensor(mTmapI32, cutlass.Int64)  # (num_sms, 3, 16)

            # 1) Device prologue: fill problem_shape / strides / addresses.
            self.fill_metadata(mA, mB, mC, mGroupSizes, mProblemShape, mStrides, addr_i64).launch(
                grid=[1, 1, 1], block=[addr_threads, 1, 1], stream=stream
            )

            # 2) Stock persistent TMA grouped GEMM (ordered after the prologue on
            #    the same stream).
            kernel = HopperGroupedGemmPersistentKernel(
                cutlass.Float32,
                _TILE_SHAPE_MN,
                cluster_shape_mn,
                tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
            )
            kernel(
                mInitA,
                mInitB,
                mInitC,
                group_count,
                mProblemShape,
                mStrides,
                addr_i64,
                mScale,
                total_num_clusters,
                tmap_i64,
                max_active_clusters,
                stream,
            )

        @cute.kernel
        def fill_metadata(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddr: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx < group_count:
                g = tidx
                # Exclusive prefix sum of token counts as branch-free dataflow
                # (group_count <= a few hundred; each thread scans all groups).
                off = cutlass.Int32(0)
                for i in cutlass.range_constexpr(group_count):
                    pred = (cutlass.Int32(i) < g).to(cutlass.Int32)
                    off = off + mGroupSizes[i] * pred
                m_g = mGroupSizes[g]

                mProblemShape[g, 0] = m_g
                mProblemShape[g, 1] = cutlass.Int32(n)
                mProblemShape[g, 2] = cutlass.Int32(k)
                mProblemShape[g, 3] = cutlass.Int32(1)

                # A[M,K] k-major -> (K,1); B[N,K] k-major -> (K,1); C[M,N] n-major -> (N,1).
                mStrides[g, 0, 0] = cutlass.Int32(k)
                mStrides[g, 0, 1] = cutlass.Int32(1)
                mStrides[g, 1, 0] = cutlass.Int32(k)
                mStrides[g, 1, 1] = cutlass.Int32(1)
                mStrides[g, 2, 0] = cutlass.Int32(n)
                mStrides[g, 2, 1] = cutlass.Int32(1)

                off64 = cutlass.Int64(off)
                g64 = cutlass.Int64(g)
                base_a = cutlass.Int64(mA.iterator.toint())
                base_b = cutlass.Int64(mB.iterator.toint())
                base_c = cutlass.Int64(mC.iterator.toint())
                mAddr[g, 0] = base_a + off64 * (k * a_bytes)
                mAddr[g, 1] = base_b + g64 * (n * k * b_bytes)
                mAddr[g, 2] = base_c + off64 * (n * c_bytes)

    return TmaGroupedLauncher()


def _dtype_bytes(jax_dtype) -> int:
    return jnp.dtype(jax_dtype).itemsize


def tma_grouped_gemm(a, b, group_sizes, *, out_dtype, out_scale, cluster_shape_mn=(2, 1)):
    """Forward grouped GEMM ``a[M,K] . b[E,N,K] -> [M,N]`` (contract K) via the
    stock Hopper TMA warp-specialized persistent kernel.

    ``a``/``b`` are k-major 8-bit (E4M3 forward, E5M2xE4M3 dgrad). The epilogue
    DIVIDES the f32 accumulator by ``out_scale[0]`` (haliax dequantize
    convention). ``group_sizes`` may be traced (dynamic per-expert token counts).
    """
    ensure_hopper_arch()
    e, n, k = b.shape
    m = a.shape[0]
    a_dtype, b_dtype = a.dtype, b.dtype

    max_active_clusters = _H100_SMS // (cluster_shape_mn[0] * cluster_shape_mn[1])
    total_num_clusters = _total_num_clusters_upper_bound(m, n, cluster_shape_mn, e)

    launcher = _build_tma_launcher(
        cluster_shape_mn=cluster_shape_mn,
        group_count=e,
        n=n,
        k=k,
        total_num_clusters=total_num_clusters,
        max_active_clusters=max_active_clusters,
        a_bytes=_dtype_bytes(a_dtype),
        b_bytes=_dtype_bytes(b_dtype),
        c_bytes=_dtype_bytes(out_dtype),
    )

    ts = cjax.TensorSpec
    a_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    b_spec = ts(mode=(0, 1, 2), divisibility=(1, 1, _FP8_TMA_VEC), static=True)
    gs_spec = ts(mode=(0,), static=True)
    scale_spec = ts(mode=(0,), static=True)
    # Initials carry dtype + majorness only; CRITICAL: static=False. Static tiny
    # extents make CuTe canonicalize the size-1 Rest modes, collapsing tile-coord
    # math for every real tile beyond the dummy extent -> fast garbage output.
    init_a_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_b_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_c_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, 1), static=False)
    c_spec = ts(mode=(0, 1), divisibility=(1, 1), static=True)
    ps_spec = ts(mode=(0, 1), static=True)
    st_spec = ts(mode=(0, 1, 2), static=True)
    addr_spec = ts(mode=(0, 1), static=True)
    tmap_spec = ts(mode=(0, 1, 2), static=True)

    out_shapes = (
        jax.ShapeDtypeStruct((m, n), out_dtype),
        jax.ShapeDtypeStruct((e, 4), jnp.int32),  # problem_shape_mnkl
        jax.ShapeDtypeStruct((e, 3, 2), jnp.int32),  # strides_abc
        jax.ShapeDtypeStruct((e, 2 * _NUM_TENSORMAPS), jnp.int32),  # tensor_address (i32-doubled)
        jax.ShapeDtypeStruct(
            (_H100_SMS, _NUM_TENSORMAPS, 2 * _TENSORMAP_I64_WORDS), jnp.int32
        ),  # tensormap workspace (i32-doubled)
    )

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=out_shapes,
        input_spec=(a_spec, b_spec, gs_spec, scale_spec, init_a_spec, init_b_spec, init_c_spec),
        output_spec=(c_spec, ps_spec, st_spec, addr_spec, tmap_spec),
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_HOPPER_CUTE_ARCH),),
    )

    init_a = jnp.zeros((1, 128, 128), a_dtype)
    init_b = jnp.zeros((1, 128, 128), b_dtype)
    init_c = jnp.zeros((1, 128, 128), out_dtype)
    gs = group_sizes.astype(jnp.int32)
    out = call(a, b, gs, out_scale, init_a, init_b, init_c)
    return out[0]


# --------------------------------------------------------------------------- #
# Wgrad (token-M-contracting weight gradient) adapter
#
# ``drhs[g] = a_t[:, g-slice] @ b_t[:, g-slice]^T`` with the ragged TOKEN axis as
# the GEMM contraction. Unlike the forward, the group's data is the contraction
# slice of a SHARED packed buffer, so a per-group base-pointer advance (offsets[g]
# elements, 1 byte each for fp8) is NOT 16B-aligned. Instead A/B keep the aligned
# full-buffer base and the token offset is an element coordinate; the per-group
# descriptor contraction extent (offset+M_g) zero-fills the ragged tail via TMA.
# --------------------------------------------------------------------------- #


def _build_tma_wgrad_launcher(
    *,
    cluster_shape_mn,
    group_count,
    n,
    k_hidden,
    m_total,
    total_num_clusters,
    max_active_clusters,
    c_bytes,
):
    """Build the stream-first ``@cute.jit`` wgrad adapter launcher for ``cutlass_call``.

    The GEMM per group is ``(M=k_hidden, N=n, K=M_g)`` -- output rows are the hidden
    dim, output cols are ``n``, and the contraction is the ragged token count.
    """
    addr_threads = max(32, ((group_count + 31) // 32) * 32)

    class TmaWgradLauncher:
        @cute.jit
        def __call__(
            self,
            stream,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mScale: cute.Tensor,
            mInitA: cute.Tensor,
            mInitB: cute.Tensor,
            mInitC: cute.Tensor,
            mC: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddrI32: cute.Tensor,
            mTmapI32: cute.Tensor,
            mOffsets: cute.Tensor,
        ):
            addr_i64 = cute.recast_tensor(mAddrI32, cutlass.Int64)  # (E, 3)
            tmap_i64 = cute.recast_tensor(mTmapI32, cutlass.Int64)  # (num_sms, 3, 16)

            # 1) Device prologue: fill problem_shape / strides / addresses / offsets.
            self.fill_metadata(mA, mB, mC, mGroupSizes, mProblemShape, mStrides, addr_i64, mOffsets).launch(
                grid=[1, 1, 1], block=[addr_threads, 1, 1], stream=stream
            )

            # 2) Stock persistent TMA grouped GEMM in ragged-contraction (wgrad) mode.
            kernel = HopperGroupedGemmPersistentKernel(
                cutlass.Float32,
                _TILE_SHAPE_MN,
                cluster_shape_mn,
                tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
                wgrad=True,
            )
            kernel(
                mInitA,
                mInitB,
                mInitC,
                group_count,
                mProblemShape,
                mStrides,
                addr_i64,
                mScale,
                total_num_clusters,
                tmap_i64,
                max_active_clusters,
                stream,
                mOffsets,
            )

        @cute.kernel
        def fill_metadata(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddr: cute.Tensor,
            mOffsets: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx < group_count:
                g = tidx
                # Exclusive prefix sum of the 16-token-ROUNDED group sizes: every
                # group starts on a 16-token boundary so its TMA element coordinate
                # (folded offset) is 16B-aligned. Matches the host repack exactly.
                v = cutlass.Int32(_FP8_TMA_VEC)
                off = cutlass.Int32(0)
                for i in cutlass.range_constexpr(group_count):
                    pred = (cutlass.Int32(i) < g).to(cutlass.Int32)
                    padded_i = ((mGroupSizes[i] + v - cutlass.Int32(1)) // v) * v
                    off = off + padded_i * pred
                m_g = mGroupSizes[g]

                # GEMM problem per group: (M=k_hidden rows, N, K=M_g tokens, L=1).
                mProblemShape[g, 0] = cutlass.Int32(k_hidden)
                mProblemShape[g, 1] = cutlass.Int32(n)
                mProblemShape[g, 2] = m_g
                mProblemShape[g, 3] = cutlass.Int32(1)

                # A=a_t[k_hidden, m_total] token(=k)-major -> row stride m_total, k stride 1.
                # B=b_t[n, m_total] token(=k)-major -> row stride m_total, k stride 1.
                # C=out[k_hidden, n] n-major -> row stride n, col stride 1.
                mStrides[g, 0, 0] = cutlass.Int32(m_total)
                mStrides[g, 0, 1] = cutlass.Int32(1)
                mStrides[g, 1, 0] = cutlass.Int32(m_total)
                mStrides[g, 1, 1] = cutlass.Int32(1)
                mStrides[g, 2, 0] = cutlass.Int32(n)
                mStrides[g, 2, 1] = cutlass.Int32(1)

                mOffsets[g] = off

                g64 = cutlass.Int64(g)
                base_a = cutlass.Int64(mA.iterator.toint())
                base_b = cutlass.Int64(mB.iterator.toint())
                base_c = cutlass.Int64(mC.iterator.toint())
                # A/B share the aligned full-buffer base (offset folded into the TMA
                # coordinate); C advances one dense [k_hidden, n] slab per expert.
                mAddr[g, 0] = base_a
                mAddr[g, 1] = base_b
                mAddr[g, 2] = base_c + g64 * (k_hidden * n * c_bytes)

    return TmaWgradLauncher()


def _padded_group_offsets(group_sizes):
    """Exclusive-prefix token offsets after rounding each group up to 16 tokens.

    Returned offsets are all multiples of 16 (the TMA innermost-coordinate
    granularity for fp8). Mirrors the device prologue's offset accumulation so the
    host repack and the kernel agree on where each group starts.
    """
    gs = group_sizes.astype(jnp.int32)
    padded_sizes = ((gs + _FP8_TMA_VEC - 1) // _FP8_TMA_VEC) * _FP8_TMA_VEC
    dst_off = jnp.cumsum(padded_sizes) - padded_sizes  # exclusive prefix sum
    src_off = jnp.cumsum(gs) - gs
    return gs, dst_off, src_off


def _pad_token_groups_16(a_t, b_t, group_sizes, e, m):
    """Repack token-major ``a_t[K,M]``/``b_t[N,M]`` so each group starts on a
    16-token boundary, zero-filling the sub-16 gap after each group.

    Returns ``(a_pad, b_pad, m_total)`` where ``m_total`` is the static padded
    width (a multiple of 16). The pad columns are exact fp8 zero every call
    (gather ``mode='fill'``) so nothing stale leaks across XLA buffer reuse.
    """
    gs, dst_off, src_off = _padded_group_offsets(group_sizes)
    # Static worst case: each of the E groups adds <16 pad tokens; round the total
    # up to 16 so the padded row stride is itself 16B-aligned (TMA row start).
    m_total = ((m + _FP8_TMA_VEC * e + _FP8_TMA_VEC - 1) // _FP8_TMA_VEC) * _FP8_TMA_VEC
    positions = jnp.arange(m_total, dtype=jnp.int32)
    grp = jnp.clip(jnp.searchsorted(dst_off, positions, side="right") - 1, 0, e - 1)
    local = positions - dst_off[grp]
    is_real = local < gs[grp]
    # Out-of-bounds source index for pad slots -> gather fills 0.
    src = jnp.where(is_real, src_off[grp] + local, m)
    a_pad = jnp.take(a_t, src, axis=1, mode="fill", fill_value=0)
    b_pad = jnp.take(b_t, src, axis=1, mode="fill", fill_value=0)
    return a_pad, b_pad, m_total


def tma_grouped_wgrad(a_t, b_t, group_sizes, *, out_dtype, out_scale, cluster_shape_mn=(1, 1)):
    """Weight-gradient grouped GEMM ``a_t[K,M] . b_t[N,M] -> [E,K,N]`` contracting the
    ragged token axis M, via the stock Hopper TMA warp-specialized persistent kernel.

    ``a_t``/``b_t`` are token-major 8-bit (activations E4M3, output-grad E5M2). M is
    the packed, non-tile-aligned per-group contraction; the epilogue DIVIDES the f32
    accumulator by ``out_scale[0]`` (haliax dequantize convention). ``group_sizes``
    may be traced (dynamic per-expert token counts).

    Cluster ``(1,1)`` (no multicast): the GEMM M dim is the (fixed, often small) hidden
    dimension, which for K < cluster_tile_M would leave a fully out-of-range CTA in a
    B-multicast cluster (malformed launch). The multicast win is marginal here and the
    ragged axis is the contraction, so a single-CTA cluster is both safe and simplest.
    """
    ensure_hopper_arch()
    k_hidden, m = a_t.shape
    n = b_t.shape[0]
    e = group_sizes.shape[0]
    a_dtype, b_dtype = a_t.dtype, b_t.dtype

    # 16-token group padding (the TMA 16B innermost-coordinate constraint). Each
    # group's token slice is the GEMM contraction, folded into the A/B TMA element
    # coordinate; that coordinate -- the group's exclusive-prefix token offset --
    # must be a multiple of 16 fp8 elements (=16B) or the load faults. Repack the
    # token axis so every group starts on a 16-token boundary, zero-filling the
    # <16-token gap after each group. Zero pads add exact +0.0 to the f32
    # accumulator (the descriptor extent = off+M_g stops the load at the ragged
    # end anyway), so the result is bit-identical to the unpadded packing.
    a_t, b_t, m_total = _pad_token_groups_16(a_t, b_t, group_sizes, e, m)

    max_active_clusters = _H100_SMS // (cluster_shape_mn[0] * cluster_shape_mn[1])
    # The ragged axis is the contraction: the M/N (k_hidden, n) tile grid is uniform
    # per group, so the cluster-tile total is EXACT (no ragged-M surplus tiles).
    ctm, ctn = _cluster_tile_mn(cluster_shape_mn)
    total_num_clusters = e * ((k_hidden + ctm - 1) // ctm) * ((n + ctn - 1) // ctn)

    launcher = _build_tma_wgrad_launcher(
        cluster_shape_mn=cluster_shape_mn,
        group_count=e,
        n=n,
        k_hidden=k_hidden,
        m_total=m_total,
        total_num_clusters=total_num_clusters,
        max_active_clusters=max_active_clusters,
        c_bytes=_dtype_bytes(out_dtype),
    )

    ts = cjax.TensorSpec
    a_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    b_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    gs_spec = ts(mode=(0,), static=True)
    scale_spec = ts(mode=(0,), static=True)
    init_a_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_b_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_c_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, 1), static=False)
    c_spec = ts(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    ps_spec = ts(mode=(0, 1), static=True)
    st_spec = ts(mode=(0, 1, 2), static=True)
    addr_spec = ts(mode=(0, 1), static=True)
    tmap_spec = ts(mode=(0, 1, 2), static=True)
    off_spec = ts(mode=(0,), static=True)

    out_shapes = (
        jax.ShapeDtypeStruct((e, k_hidden, n), out_dtype),
        jax.ShapeDtypeStruct((e, 4), jnp.int32),  # problem_shape_mnkl
        jax.ShapeDtypeStruct((e, 3, 2), jnp.int32),  # strides_abc
        jax.ShapeDtypeStruct((e, 2 * _NUM_TENSORMAPS), jnp.int32),  # tensor_address (i32-doubled)
        jax.ShapeDtypeStruct((_H100_SMS, _NUM_TENSORMAPS, 2 * _TENSORMAP_I64_WORDS), jnp.int32),  # tensormap ws
        jax.ShapeDtypeStruct((e,), jnp.int32),  # per-group token offsets
    )

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=out_shapes,
        input_spec=(a_spec, b_spec, gs_spec, scale_spec, init_a_spec, init_b_spec, init_c_spec),
        output_spec=(c_spec, ps_spec, st_spec, addr_spec, tmap_spec, off_spec),
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_HOPPER_CUTE_ARCH),),
    )

    init_a = jnp.zeros((1, 128, 128), a_dtype)
    init_b = jnp.zeros((1, 128, 128), b_dtype)
    init_c = jnp.zeros((1, 128, 128), out_dtype)
    gs = group_sizes.astype(jnp.int32)
    out = call(a_t, b_t, gs, out_scale, init_a, init_b, init_c)
    return out[0]
