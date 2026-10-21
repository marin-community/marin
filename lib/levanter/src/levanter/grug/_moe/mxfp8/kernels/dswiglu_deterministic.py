# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Includes software developed at NVIDIA CORPORATION (https://www.nvidia.com/): cudnn-frontend.
#
# `DSwigluKernel.kernel` is copied from nvidia-cudnn-frontend 1.31.0,
# cudnn/gemm/cutedsl/grouped/dswiglu/grouped_gemm_dswiglu_quant.py,
# BlockScaledContiguousGroupedGemmKernel.kernel, with upstream's module imports made absolute.
# Every other method is inherited from that class unchanged.
# Modified by Marin: `dprob` is an (M, num_N_tiles, 1) float32 tensor of per-N-tile partials written
# with plain stores, replacing the float32 atomic add across N tiles (deterministic, no zero-init).
#
# Formatted with the repository's black and ruff settings; upstream lint findings are suppressed.
# ruff: noqa: E712, F841
# pyrefly: ignore-errors

"""The cudnn-frontend fused dSwiGLU grouped GEMM with a deterministic routing-weight gradient."""

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cudnn._cutlass_compat import SmemAllocator, TmemAllocator
from cudnn.gemm.cutedsl.grouped.dswiglu.grouped_gemm_dswiglu_quant import BlockScaledContiguousGroupedGemmKernel
from cudnn.gemm.cutedsl.grouped.utils import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
    search_expert_idx_full,
    search_expert_idx_incremental,
)
from cutlass.cute.nvgpu import cpasync, tcgen05


class DSwigluKernel(BlockScaledContiguousGroupedGemmKernel):
    """Upstream's dSwiGLU kernel, its per-row routing-weight gradient split into per-N-tile partials.

    ``dprob`` is an ``(M, ceil(N / cta_tile_n), 1)`` float32 tensor. The epilogue thread that owns a
    row in an N tile stores that tile's partial of ``<silu(gate) * up, dh>`` into the tile's column,
    so every element is written exactly once and the caller sums the columns. Upstream instead adds
    each partial into an ``(M, 1, 1)`` tensor with a float32 atomic, whose order varies between runs.
    """

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
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
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
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(not self.store_d_directly):
                cpasync.prefetch_descriptor(tma_atom_d)
            if cutlass.const_expr(self.generate_sfd):
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

        # Load C pipeline
        # Threads/warps participating in tma store pipeline
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.epilog_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=self.tma_c_load_bytes,
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
        sD = None
        if cutlass.const_expr(not self.store_d_directly):
            sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sD_col = None
        if cutlass.const_expr(self.generate_sfd):
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
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gD_mnl = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
        if cutlass.const_expr(self.generate_sfd):
            # (bM, bN, loopM, loopN, loopL)
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
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgD = thr_mma.partition_C(gD_mnl)
        if cutlass.const_expr(self.generate_sfd):
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
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgD, epi_tile, use_2cta_instrs)

            tTR_rC1 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rC2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_s2r, tRS_rC1, tRS_rC2, tRS_sC = self.epilog_smem_copy_and_partition_load(
                tiled_copy_t2r, tTR_rC1, tTR_rC2, epi_tidx, sC
            )

            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

            tTR_rD1 = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tTR_rD2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD1, tRS_rD2, tRS_sD = self.epilog_smem_copy_and_partition_store(
                tiled_copy_t2r, tTR_rD1, tTR_rD2, epi_tidx, sD
            )
            if cutlass.const_expr(self.store_d_directly):
                d_epilogue_subtile = (
                    cute.make_layout(128),
                    cute.make_layout(self.mma_tiler[1] * 2),
                )
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gD_sub_mnl = cute.local_tile(mD_mnl, d_epilogue_subtile, (None, None, None))
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgD_mnl = thr_copy_t2r.partition_D(gD_sub_mnl)
                tCgD_mnl = cute.filter_zeros(tCgD_mnl)
            else:
                (
                    bSG_sD,
                    bSG_gD_partitioned,
                ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD, epi_tile, sD)

            if cutlass.const_expr(self.generate_sfd):
                tTR_rD1_col = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
                tTR_rD2_col = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
                (
                    tiled_copy_r2s,
                    tRS_rD1_col,
                    tRS_rD2_col,
                    tRS_sD_col,
                ) = self.epilog_smem_copy_and_partition_store(
                    tiled_copy_t2r, tTR_rD1_col, tTR_rD2_col, epi_tidx, sD_col
                )
                (
                    bSG_sD_col,
                    bSG_gD_partitioned_col,
                ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d_col, tCgD_col, epi_tile, sD_col)

                norm_const = cutlass.Float32(norm_const_tensor[0])
                regPerSubtile = 4
                sfd_row_tile = (
                    cute.make_layout(128),
                    cute.make_layout(32 * regPerSubtile),
                )
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFDRow_mnl = cute.local_tile(mSFDRow_mnl, sfd_row_tile, (None, None, None))
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFDRow_mnl = thr_copy_t2r.partition_D(gSFDRow_mnl)
                tCgSFDRow_mnl = cute.filter_zeros(tCgSFDRow_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFDRow = cute.make_rmem_tensor(tCgSFDRow_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype)
                tCrSFDRow_pvscale = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)
                d_rcp_limits = self.get_dtype_rcp_limits(self.d_dtype)

                gSFDCol_mnl = cute.local_tile(mSFDCol_mnl, sfd_row_tile, (None, None, None))
                thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
                val_layout = cute.make_ordered_layout((1,), order=(0,))
                copy_atom_sfd_col_quant = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    gSFDCol_mnl.element_type,
                    num_bits_per_copy=8,
                )
                tiled_copy_sfd_col_quant = cute.make_tiled_copy_tv(copy_atom_sfd_col_quant, thr_layout, val_layout)
                thr_copy_sfd_col_quant = tiled_copy_sfd_col_quant.get_slice(tidx)
                tCgSFDCol_mnl = thr_copy_sfd_col_quant.partition_D(cute.filter_zeros(gSFDCol_mnl))
                tCgSFDCol_mnl = cute.filter_zeros(tCgSFDCol_mnl)
                tCrSFDCol = cute.make_rmem_tensor(tCrSFDRow.layout, tCrSFDRow.element_type)
                tCrSFDCol_pvscale = cute.make_rmem_tensor_like(tCrSFDRow_pvscale.layout, cutlass.Float32)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Load C pipeline
            c_pipeline_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_c_stage
            )

            # Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            d_pipeline = None
            if cutlass.const_expr(not self.store_d_directly):
                num_d_stages = self.num_d_stage // 2
                d_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=num_d_stages,
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
                    tile_info[1] * 2,
                    tile_info[2],
                )

                #
                # Get alpha for current group
                #
                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]
                beta_val = beta[expert_idx]

                #
                # Slice to per mma tile index
                #

                if cutlass.const_expr(not self.store_d_directly):
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
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

                if cutlass.const_expr(self.generate_sfd):
                    bSG_gD_col = bSG_gD_partitioned_col[
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
                mPosition = tile_info[0] * self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape) + tidx
                mProb = prob[mPosition, 0, 0].to(cutlass.Float32)
                if cutlass.const_expr(self.generate_dprob):
                    dProbVal = cutlass.Float32(0.0)

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Initialize thread-local amax accumulator for this tile
                # Use 0.0 as initial value since we're computing absolute maximum
                if cutlass.const_expr(self.generate_amax):
                    thread_tile_amax_1 = cutlass.Float32(0.0)
                    thread_tile_amax_2 = cutlass.Float32(0.0)

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                tTR_rAcc_0 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
                tTR_rAcc_1 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    real_subtile_idx = subtile_idx
                    real_subtile_idx_next = subtile_idx + 1
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            # Subtile always iterates on N dimension as we only have 4x1DP tmem load pattern for cta_tile_m = 128 cases.
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - subtile_idx
                            real_subtile_idx_next = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - subtile_idx - 1
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    if cutlass.const_expr(self.epilogue_prefetch_more):
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                        tTR_tAcc_mn_next = tTR_tAcc[(None, None, None, real_subtile_idx_next)]
                        if subtile_idx % 2 == 0:
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc_0)
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn_next, tTR_rAcc_1)
                            tTR_rAcc = tTR_rAcc_0
                        else:
                            tTR_rAcc = tTR_rAcc_1
                    else:
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty ealier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        bReleaseAcc = False
                        if cutlass.const_expr(self.epilogue_prefetch_more):
                            bReleaseAcc = subtile_idx == (self.iter_acc_early_release_in_epilogue // 2)
                        else:
                            bReleaseAcc = subtile_idx == (self.iter_acc_early_release_in_epilogue)
                        if bReleaseAcc:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    # Wait for C1/C2 load to complete
                    c_pipeline.consumer_wait(c_pipeline_consumer_state)
                    cute.copy(
                        tiled_copy_s2r,
                        tRS_sC[(None, None, None, c_pipeline_consumer_state.index)],
                        tRS_rC1,
                    )
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    c_pipeline.consumer_release(c_pipeline_consumer_state)
                    c_pipeline_consumer_state.advance()
                    c_pipeline.consumer_wait(c_pipeline_consumer_state)
                    cute.copy(
                        tiled_copy_s2r,
                        tRS_sC[(None, None, None, c_pipeline_consumer_state.index)],
                        tRS_rC2,
                    )
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    c_pipeline.consumer_release(c_pipeline_consumer_state)
                    c_pipeline_consumer_state.advance()

                    LOG2_E = cutlass.Float32(1.4426950408889634)
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc)
                    ab1_vec_load = tiled_copy_r2s.retile(tRS_rC1)
                    ab2_vec_load = tiled_copy_r2s.retile(tRS_rC2)
                    if cutlass.const_expr(self.generate_dprob):
                        dprob_swiglu = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)

                    #
                    # Apply alpha, act, and prob
                    #
                    square_alpha = alpha_val * alpha_val
                    if cutlass.const_expr(self.vectorized_f32):
                        d1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                        d2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                        for i in cutlass.range(0, cute.size(acc_vec), 2, unroll_full=True):
                            # Apply scaling factors for FP8
                            (
                                acc_vec[i + 0],
                                acc_vec[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (acc_vec[i + 0], acc_vec[i + 1]),
                                (square_alpha, square_alpha),
                                rnd="rn",
                                ftz=False,
                            )
                            ab1_vec_acc_type = cute.arch.mul_packed_f32x2(
                                (
                                    ab1_vec_load[i + 0].to(self.acc_dtype),
                                    ab1_vec_load[i + 1].to(self.acc_dtype),
                                ),
                                (beta_val, beta_val),
                                rnd="rn",
                                ftz=False,
                            )
                            ab2_vec_acc_type = cute.arch.mul_packed_f32x2(
                                (
                                    ab2_vec_load[i + 0].to(self.acc_dtype),
                                    ab2_vec_load[i + 1].to(self.acc_dtype),
                                ),
                                (beta_val, beta_val),
                                rnd="rn",
                                ftz=False,
                            )
                            sig_rcp_0, sig_rcp_1 = cute.arch.mul_packed_f32x2(
                                (ab1_vec_acc_type),
                                (-LOG2_E, -LOG2_E),
                                rnd="rn",
                                ftz=False,
                            )
                            sig_rcp_0, sig_rcp_1 = cute.arch.add_packed_f32x2(
                                (
                                    cute.math.exp2(sig_rcp_0, fastmath=True),
                                    cute.math.exp2(sig_rcp_1, fastmath=True),
                                ),
                                (1.0, 1.0),
                                rnd="rn",
                                ftz=False,
                            )
                            sig = (
                                cute.arch.rcp_approx(sig_rcp_0),
                                cute.arch.rcp_approx(sig_rcp_1),
                            )
                            swish = cute.arch.mul_packed_f32x2(
                                ab1_vec_acc_type,
                                sig,
                                rnd="rn",
                                ftz=False,
                            )
                            # calculate dprob
                            if cutlass.const_expr(self.generate_dprob):
                                (
                                    dprob_swiglu[i + 0],
                                    dprob_swiglu[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (ab2_vec_acc_type[0], ab2_vec_acc_type[1]),
                                    swish,
                                )
                                (
                                    dprob_swiglu[i + 0],
                                    dprob_swiglu[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (dprob_swiglu[i + 0], dprob_swiglu[i + 1]),
                                    (acc_vec[i + 0], acc_vec[i + 1]),
                                )
                            # calculate dswiglu
                            acc_vec_prob = cute.arch.mul_packed_f32x2(
                                (acc_vec[i + 0], acc_vec[i + 1]),
                                (mProb, mProb),
                            )
                            # calculate d2_vec
                            (
                                d2_vec[i + 0],
                                d2_vec[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (acc_vec_prob[0], acc_vec_prob[1]),
                                swish,
                                rnd="rn",
                                ftz=False,
                            )
                            # calculate d1_vec
                            (
                                d1_vec[i + 0],
                                d1_vec[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (acc_vec_prob[0], acc_vec_prob[1]),
                                (ab2_vec_acc_type[0], ab2_vec_acc_type[1]),
                                rnd="rn",
                                ftz=False,
                            )
                            (
                                d1_vec[i + 0],
                                d1_vec[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (d1_vec[i + 0], d1_vec[i + 1]),
                                sig,
                                rnd="rn",
                                ftz=False,
                            )
                            dsig = cute.arch.mul_packed_f32x2(
                                ab1_vec_acc_type,
                                (1 - sig[0], 1 - sig[1]),
                                rnd="rn",
                                ftz=False,
                            )
                            dsig_add_1 = cute.arch.add_packed_f32x2(
                                (dsig[0], dsig[1]),
                                (1.0, 1.0),
                                rnd="rn",
                                ftz=False,
                            )
                            (
                                d1_vec[i + 0],
                                d1_vec[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (d1_vec[i + 0], d1_vec[i + 1]),
                                dsig_add_1,
                                rnd="rn",
                                ftz=False,
                            )
                        d1_vec = d1_vec.load()
                        d2_vec = d2_vec.load()
                        if cutlass.const_expr(self.generate_dprob):
                            dprob_swiglu = dprob_swiglu.load()
                    else:
                        acc_vec = acc_vec.load()
                        ab1_vec_load = ab1_vec_load.load()
                        ab2_vec_load = ab2_vec_load.load()

                        acc_vec = acc_vec * square_alpha  # apply scale for A*B
                        ab1_vec_load = ab1_vec_load * beta_val  # apply scale for C
                        ab2_vec_load = ab2_vec_load * beta_val  # apply scale for C

                        sig_rcp = (1 + cute.math.exp(-1 * ab1_vec_load, True)).to(self.acc_dtype)
                        res = cute.make_rmem_tensor(sig_rcp.shape, cutlass.Float32)
                        res.store(sig_rcp)
                        # let every res[?] be cute.arch.rcp_approx(res[?])
                        [res.__setitem__(i, cute.arch.rcp_approx(res[i])) for i in range(cute.size(res.shape))]
                        sig = res.load()
                        swish = ab1_vec_load * sig

                        # calculate dprob
                        if cutlass.const_expr(self.generate_dprob):
                            dprob_swiglu = ab2_vec_load * swish
                            dprob_swiglu = acc_vec * dprob_swiglu

                        # calculate dswiglu
                        d2_vec = acc_vec * mProb * swish
                        d1_vec = acc_vec * mProb * ab2_vec_load * sig * (1 + ab1_vec_load * (1 - sig))

                    if cutlass.const_expr(self.generate_dprob):
                        # dprob sum reduction
                        dProbVal += dprob_swiglu.reduce(
                            cute.ReductionOp.ADD,
                            cutlass.Float32(0.0),
                            0,
                        )

                    #
                    # Generate amax
                    #
                    if cutlass.const_expr(self.generate_amax):
                        thread_tile_amax_1 = self.amax_reduction_per_thread(d1_vec, thread_tile_amax_1)
                        thread_tile_amax_2 = self.amax_reduction_per_thread(d2_vec, thread_tile_amax_2)

                    #
                    # Generate SFD
                    #
                    if cutlass.const_expr(self.generate_sfd):
                        #
                        # Generate row major SFD
                        #
                        self.quant_sfd_row(
                            (real_subtile_idx * 2 + 0) % 4,
                            tiled_copy_r2s,
                            d1_vec,
                            tCrSFDRow_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD1,
                        )
                        self.quant_sfd_col(
                            (real_subtile_idx * 2 + 0) % 4,
                            tiled_copy_r2s,
                            d1_vec,
                            tCrSFDCol_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD1_col,
                        )
                        self.quant_sfd_row(
                            (real_subtile_idx * 2 + 1) % 4,
                            tiled_copy_r2s,
                            d2_vec,
                            tCrSFDRow_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD2,
                        )
                        self.quant_sfd_col(
                            (real_subtile_idx * 2 + 1) % 4,
                            tiled_copy_r2s,
                            d2_vec,
                            tCrSFDCol_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD2_col,
                        )

                        if subtile_idx % 2 == 1:
                            # Assume subtile partitioned always happens on n dimension
                            sfd_row_idx_mn = (
                                tile_info[0] * self.epi_tile_cnt[0] + 0,
                                tile_info[1] * self.epi_tile_cnt[1] // 2 + (real_subtile_idx // 2),
                            )
                            sfd_col_idx_mn = sfd_row_idx_mn
                            if cutlass.const_expr(self.discrete_col_sfd):
                                sfd_col_idx_mn = (
                                    tile_info[0] * self.epi_tile_cnt[0] + 0 - tile_info[5] // 128,
                                    tile_info[1] * self.epi_tile_cnt[1] // 2 + (real_subtile_idx // 2),
                                )

                            tCgSFDRow = tCgSFDRow_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfd_row_idx_mn,
                                )
                            ]
                            tCgSFDCol = tCgSFDCol_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfd_col_idx_mn,
                                )
                            ]
                            if cutlass.const_expr(not self.use_fp8_ptx_cvt):
                                tCrSFDRow.store(tCrSFDRow_pvscale.load().to(self.sf_dtype))
                                tCrSFDCol.store(tCrSFDCol_pvscale.load().to(self.sf_dtype))
                            else:
                                self.cvt_f32x4_to_f8x4(tCrSFDRow_pvscale, tCrSFDRow)
                                self.cvt_f32x4_to_f8x4(tCrSFDCol_pvscale, tCrSFDCol)
                            if sfd_row_idx_mn[1] * 32 * regPerSubtile < cute.size(
                                cute.shape(mSFDRow_mnl.layout, mode=[1])
                            ):
                                cute.autovec_copy(tCrSFDRow, tCgSFDRow)
                            if sfd_col_idx_mn[1] * 32 * regPerSubtile < cute.size(
                                cute.shape(mSFDCol_mnl.layout, mode=[1])
                            ):
                                cute.autovec_copy(tCrSFDCol, tCgSFDCol)
                    else:
                        #
                        # Convert to D type
                        #
                        tRS_rD1.store(d1_vec.to(self.d_dtype))
                        tRS_rD2.store(d2_vec.to(self.d_dtype))

                    #
                    # Store D
                    #
                    if cutlass.const_expr(self.store_d_directly):
                        self.epilog_sync_barrier.arrive_and_wait()
                        d_idx_mn = (tile_info[0], tile_info[1])
                        tCgD1 = tCgD_mnl[
                            (
                                None,
                                0,  # T2R_M
                                2 * real_subtile_idx + 0,  # T2R_M
                                *d_idx_mn,  # RestM/N
                                0,  # RestL
                            )
                        ]
                        tCgD2 = tCgD_mnl[
                            (
                                None,
                                0,  # T2R_M
                                2 * real_subtile_idx + 1,  # T2R_M
                                *d_idx_mn,  # RestM/N
                                0,  # RestL
                            )
                        ]
                        # cute.autovec_copy(tRS_rD1, tCgD1)
                        # cute.autovec_copy(tRS_rD2, tCgD2)
                        self.store_global_memory_256b(tCgD1, tRS_rD1)
                        self.store_global_memory_256b(tCgD2, tRS_rD2)
                    else:
                        if warp_idx == self.epilog_warp_id[0]:
                            d_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        d1_buffer = num_prev_subtiles % self.num_d_stage
                        num_prev_subtiles = num_prev_subtiles + 1
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD1,
                            tRS_sD[(None, None, None, d1_buffer)],
                        )
                        if cutlass.const_expr(self.generate_sfd):
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD1_col,
                                tRS_sD_col[(None, None, None, d1_buffer)],
                            )
                        d2_buffer = num_prev_subtiles % self.num_d_stage
                        num_prev_subtiles = num_prev_subtiles + 1
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD2,
                            tRS_sD[(None, None, None, d2_buffer)],
                        )
                        if cutlass.const_expr(self.generate_sfd):
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD2_col,
                                tRS_sD_col[(None, None, None, d2_buffer)],
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
                                bSG_sD[(None, d1_buffer)],
                                bSG_gD[(None, 2 * real_subtile_idx + 0)],
                            )
                            cute.copy(
                                tma_atom_d,
                                bSG_sD[(None, d2_buffer)],
                                bSG_gD[(None, 2 * real_subtile_idx + 1)],
                            )
                            if cutlass.const_expr(self.generate_sfd):
                                cute.copy(
                                    tma_atom_d_col,
                                    bSG_sD_col[(None, d1_buffer)],
                                    bSG_gD_col[(None, 2 * real_subtile_idx + 0)],
                                )
                                cute.copy(
                                    tma_atom_d_col,
                                    bSG_sD_col[(None, d2_buffer)],
                                    bSG_gD_col[(None, 2 * real_subtile_idx + 1)],
                                )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            d_pipeline.producer_commit()
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
                    gAmax1 = mAmax_tensor[(expert_idx, 0, None)].iterator.llvm_ptr  # First element
                    gAmax2 = mAmax_tensor[(expert_idx, 1, None)].iterator.llvm_ptr  # Second element
                    self.amax_reduction_per_warp_and_cta(
                        thread_tile_amax_1,
                        warp_idx,
                        sAmax,
                        gAmax1,
                    )
                    self.amax_reduction_per_warp_and_cta(
                        thread_tile_amax_2,
                        warp_idx,
                        sAmax,
                        gAmax2,
                    )

                if cutlass.const_expr(self.generate_dprob):
                    # Modified by Marin: each (row, N tile) pair is visited by exactly one epilogue
                    # thread, so store this tile's partial into its own column instead of
                    # atomically accumulating across N tiles. The caller sums the columns, which
                    # makes dprob deterministic and removes the need to zero-initialize it.
                    dprob[(mPosition, tile_info[1], 0)] = dProbVal

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for D store complete
            #
            if cutlass.const_expr(not self.store_d_directly):
                d_pipeline.producer_tail()
        #
        # Specialized epilog load warp
        #
        if warp_idx == self.epilog_load_tma_id:
            ## M 1024, N 512
            ## tCgC (((32,128),1),1,8,4,2,1) : (((1@0,1@1),0),0,32@0,256@1,512@0,1@2)
            ## bGS_gC_mnl  (((32,128),1),1,8,EPI_M,EPI_N,L) : (((1@0,1@1),0),0,32@0,256@1,512@0,1@2)
            ## bGS_sC  ((4096, 1), (1, 4)) : ((1, 0), (0, 4096))
            (
                bGS_sC,
                bGS_gC_mnl,
            ) = self.epilog_gmem_copy_and_partition(tidx, tma_atom_c, tCgC, epi_tile, sC)
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
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

            c_pipeline_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_c_stage
            )
            is_reverse = True
            while is_valid_tile:
                # if it needs to be reversed
                if cutlass.const_expr(self.overlapping_accum):
                    reverse_subtile = is_reverse
                    is_reverse = not is_reverse
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1] * 2,
                    tile_info[2],
                )
                bGS_gC = bGS_gC_mnl[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))
                subtile_cnt = cute.size(bGS_gC.shape, mode=[1])
                for subtile_idx in cutlass.range(subtile_cnt, unroll=1):
                    # Check real subtile index
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            # Subtile always iterates on N dimension as we only have 4x1DP tmem load pattern for cta_tile_m = 128 cases.
                            real_subtile_idx = subtile_cnt - 1 - subtile_idx
                    # Load C from global memory to shared memory
                    c_pipeline.producer_acquire(c_pipeline_producer_state)
                    cute.copy(
                        tma_atom_c,
                        bGS_gC[(None, 2 * real_subtile_idx + 0)],
                        bGS_sC[(None, c_pipeline_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                    )
                    c_pipeline_producer_state.advance()
                    c_pipeline.producer_acquire(c_pipeline_producer_state)
                    cute.copy(
                        tma_atom_c,
                        bGS_gC[(None, 2 * real_subtile_idx + 1)],
                        bGS_sC[(None, c_pipeline_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                    )
                    c_pipeline_producer_state.advance()

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
            # Wait C buffer tail complete
            #
            c_pipeline.producer_tail(c_pipeline_producer_state)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
