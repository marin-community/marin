# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

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

from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import T
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector, arith

from .utils import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
    search_expert_idx_full,
    search_expert_idx_incremental,
    fmin,
    atomic_max_float32,
    atomic_add_float32,
)

"""
High-performance persistent blockscaled contiguous grouped dense GEMM (D = alpha * (SFA * A) * (SFB * B)) example for the NVIDIA Blackwell architecture
using CUTE DSL.
- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension
- Matrix D is MxNx1, D can be row-major("N"), ValidM is composed of valid m in different groups
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively

Matrix A/D Memory Layout Diagrams:

   ```
    Group 0    Group 1   Group 2
   -+---------+---------+---------+
    |         |         |         |
   K| ValidM0 | ValidM1 | ValidM2 |
    |         |         |         |
   -+---------+---------+---------+
    |<-        ValidM           ->|
   ```
   Note: the Group(L) dimension will be flatted into M dimension, and the rest Group(L) size is 1.
         each ValidM will be aligned to 256 or 128. The alignment is determined by the mma_tiler_mn parameter.
         For NVFP4, 2CTA, the alignment is 256. For NVFP4, 1CTA, the alignment is 128.

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. SCALE warp: Load scaleA and scaleB matrices from global memory (GMEM) to shared memory (SMEM) using non-TMA operations.
2. MMA warp: 
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Apply alpha and update the final accumulator Final = alpha * acc
    - Type convert Final matrix to output type.
    - Store D matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

.. code-block:: bash

    python examples/blackwell/contiguous_blockscaled_grouped_gemm.py         \
      --ab_dtype Float4E2M1FN --d_dtype BFloat16 --acc_dtype Float32            \
      --sf_dtype Float8E4M3FN --sf_vec_size 16                                   \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --mnkl 256,4096,7168,1 --use_2cta_instrs --m_aligned 256


Constraints:
* Supported input data types: mxf8, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/D tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the A/D matrices and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(FIX_PAD_SIZE-1), example: 4096*8 + 8*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - D matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding:
  - Scheduler warp loads padded_offsets and calculates num_valid_tiles from padded_offsets[-1]
  - Uses warp-parallel search (ballot + ffs) to find expert_idx for each tile
  - Only valid tiles (tile_m_start < padded_offsets[-1]) are written to tile_info pipeline
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in D matrix will not be written by the kernel
"""


class BlockScaledContiguousGroupedGemmKernel:
    """This class implements batched matrix multiplication (D = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported D data types:
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
        - FIX_PAD_SIZE (256) must be divisible by mma_tiler_mn[0]
        - m_aligned parameter in create_mask() MUST equal FIX_PAD_SIZE (256)
        - Each padded_offsets[i] will be a multiple of FIX_PAD_SIZE (guaranteed by m_aligned == FIX_PAD_SIZE)

    :note: New Interface (padded_offsets):
        Instead of tile_idx_to_expert_idx, num_non_exiting_tiles, and m_split_cumsum, users now provide:
        - padded_offsets: shape (expert_cnt,), where padded_offsets[i] is the end position
          of expert[i] in the padded A tensor.
        - Expert i processes A[padded_offsets[i-1]:padded_offsets[i], :] (with padded_offsets[-1]=0)

    """

    # Fixed pad size for user-side padding (decoupled from kernel tile size)
    FIX_PAD_SIZE = 256

    def __init__(
        self,
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        discrete_col_sfd: bool,
        expert_cnt: int,
        use_mono_increase_expert_idx: bool = True,
    ):
        """Initializes the configuration for a Blackwell blockscaled dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Expert Count:
            - expert_cnt: Number of experts for MoE grouped GEMM.

        4.  Monotonic Expert Index (optional optimization):
            - use_mono_increase_expert_idx: When enabled, uses N-major rasterization
              to ensure expert_idx is monotonically non-decreasing across consecutive
              tiles processed by the same CTA. This enables an optimized incremental
              expert lookup that can be O(1) in the common case instead of O(E/32).

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param expert_cnt: Number of experts (compile-time constant).
        :type expert_cnt: int
        :param use_mono_increase_expert_idx: Enable monotonic expert index optimization.
            Uses N-major rasterization to guarantee expert_idx monotonicity. Default: True.
        :type use_mono_increase_expert_idx: bool
        """
        self.sf_vec_size = sf_vec_size
        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.use_mono_increase_expert_idx = use_mono_increase_expert_idx
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        # Compute number of offsets each thread in scheduler warp needs to hold
        # Each thread handles ceil(expert_cnt / 32) offsets
        self.num_offsets_per_thread = (expert_cnt + 31) // 32

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.epilog_load_tma_id = 6
        self.sched_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.epilog_load_tma_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.epilog_load_tma_id,
            )
        )
        # TODO: Do we need to reallocate register?
        # self.num_regs_uniform_warps = 64
        # self.num_regs_sched_warps = 64
        # self.num_regs_epilogue_warps = 216

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
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

        # Amax reduction configuration
        self.num_epilog_warps = len(self.epilog_warp_id)

        self.discrete_col_sfd = discrete_col_sfd

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/D stage counts in shared memory
        - Computing A/B/D shared memory layout
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Configure tiled mma
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

        # Compute mma/cluster/tile shapes
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
        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Set epilogue subtile
        self.epi_tile = (128, 32)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_d[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_d[1] // self.epi_tile[1],
        )

        # enable direct store D when it is NVFP4 input, BFlat16 output
        self.store_d_directly = False  # self.d_dtype.width == 16 and self.a_dtype.width == 4

        # Setup A/B/D/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_d_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.d_dtype,
            self.d_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
            self.store_d_directly,
        )

        # Compute A/B/D/Scale shared memory layout
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

        if cutlass.const_expr(not self.store_d_directly):
            self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.d_dtype,
                self.d_layout,
                self.epi_tile,
                self.num_d_stage,
            )
        else:
            self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.d_dtype,
                self.d_layout,
                self.epi_tile,
                1,
            )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256

        # To prefetch more accumulator when overlapping_accum is enabled in epilogue
        self.epilogue_prefetch_more = self.d_dtype.width == 8 and self.a_dtype.width == 8

        # To generate dprob
        self.generate_dprob = True

        # Use ptx fp8 fp32 convert
        self.use_fp8_ptx_cvt = True

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.epi_tile_n_required = cute.size(self.epi_tile[1])
        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = (self.num_sf_tmem_cols + self.epi_tile_n_required - 1) // self.epi_tile_n_required - 1

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
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
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
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
        :param c: Output tensor C
        :type c: cute.Tensor
        :param d: Output tensor D
        :type d: cute.Tensor
        :param d_col: Output tensor D column quantized
        :type d_col: cute.Tensor
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
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.d_layout = utils.LayoutEnum.from_tensor(d)

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

        # Compute grid size
        m, n, l = cute.shape(d)

        # Setup sfd tensor by filling D tensor to scale factor atom layout
        self.generate_sfd = (
            self.a_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN)
            and self.sf_dtype == cutlass.Float8E8M0FNU
            and self.d_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN)
        )
        # discrete_col_sfd is enabled via __init__, padded_offsets contains the same info as m_split_cumsum
        if cutlass.const_expr(self.generate_sfd == False):
            self.discrete_col_sfd = False
        if cutlass.const_expr(self.generate_sfd):
            output_sfd_shape = (m, n, l)
            sfd_layout = blockscaled_utils.tile_atom_to_shape_SF(output_sfd_shape, self.sf_vec_size)
            sfd_row_tensor = cute.make_tensor(sfd_row_tensor.iterator, sfd_layout)
            sfd_col_quant_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
                output_sfd_shape,
                (1, 2, 3),
            )
            if cutlass.const_expr(self.discrete_col_sfd):
                sfd_col_quant_layout = sfd_layout
            sfd_col_tensor = cute.make_tensor(sfd_col_tensor.iterator, sfd_col_quant_layout)

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
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        self.tma_c_load_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            c,
            c_smem_layout,
            self.epi_tile,
        )

        # Setup TMA store for D
        tma_atom_d = None
        tma_tensor_d = None
        if cutlass.const_expr(not self.store_d_directly):
            d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
            tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                d,
                d_smem_layout,
                self.epi_tile,
            )
        else:
            tma_tensor_d = d

        tma_atom_d_col = None
        tma_tensor_d_col = None
        if cutlass.const_expr(self.generate_sfd):
            tma_atom_d_col, tma_tensor_d_col = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                d_col,
                d_smem_layout,
                self.epi_tile,
            )

        # Compute grid size
        # When use_mono_increase_expert_idx is enabled, use N-major rasterization
        # (raster_along_m=False) to ensure M coordinates are monotonically non-decreasing
        output_shape = (m, n // 2, l)
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
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
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

        # Define shared storage for kernel
        @cute.struct
        class SharedStorageFP4:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
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
                4,  # byte alignment
            ]
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                4,  # byte alignment
            ]

        @cute.struct
        class SharedStorageFP4_DirectStoreD:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
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
                4,  # byte alignment
            ]
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                4,  # byte alignment
            ]

        if cutlass.const_expr(self.generate_sfd):
            self.shared_storage = SharedStorageFP8
        else:
            if cutlass.const_expr(self.store_d_directly):
                self.shared_storage = SharedStorageFP4_DirectStoreD
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
            beta,
            prob,
            dprob,
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

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def amax_reduction_per_thread(self, vec_fp32, amax_fp32) -> None:
        vec_fp32_ssa = vec_fp32
        abs_acc_values_ir = cutlass._mlir.dialects.math.absf(vec_fp32_ssa.ir_value())
        abs_acc_values = type(vec_fp32_ssa)(abs_acc_values_ir, vec_fp32_ssa.shape, vec_fp32_ssa.dtype)
        subtile_amax = abs_acc_values.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        return cute.arch.fmax(amax_fp32, subtile_amax)

    @cute.jit
    def amax_reduction_per_warp_and_cta(self, amax_fp32, warp_idx, amax_smem, amax_gmem) -> None:
        # Warp-level reduction using wrapper function
        warp_amax = cute.arch.warp_redux_sync(
            value=amax_fp32,
            kind="fmax",
            mask_and_clamp=0xFFFFFFFF,
            nan=True,
        )
        # Each epilogue warp's lane 0 writes warp amax to shared memory
        if cute.arch.lane_idx() == 0:
            amax_smem[warp_idx] = cutlass.Float32(warp_amax)

        # Ensure all epilogue warps complete their writes before block reduction
        self.epilog_sync_barrier.arrive_and_wait()

        # Block-level reduction: only first epilogue warp's lane 0 handles this
        if warp_idx == self.epilog_warp_id[0] and cute.arch.lane_idx() == 0:
            block_amax = cutlass.Float32(0.0)
            for i in cutlass.range(self.num_epilog_warps):
                warp_amax_val = amax_smem[i]
                block_amax = cute.arch.fmax(block_amax, warp_amax_val)

            # Global atomic max (accumulates across all tiles for final tensor amax)
            _ = atomic_max_float32(ptr=amax_gmem, value=block_amax)

        # Ensure all epilogue warps complete their writes before global reduction
        self.epilog_sync_barrier.arrive_and_wait()

    @cute.jit
    def cvt_f32x4_to_f8x4_pack_i32(self, fp32x4, fp8_type, loc=None, ip=None):
        fp32x4 = fp32x4.load()
        src_vec4 = fp32x4.ir_value(loc=loc, ip=ip) if hasattr(fp32x4, "ir_value") else fp32x4

        src0 = Float32(vector.extract(src_vec4, [], [0])).ir_value(loc=loc, ip=ip)
        src1 = Float32(vector.extract(src_vec4, [], [1])).ir_value(loc=loc, ip=ip)
        src2 = Float32(vector.extract(src_vec4, [], [2])).ir_value(loc=loc, ip=ip)
        src3 = Float32(vector.extract(src_vec4, [], [3])).ir_value(loc=loc, ip=ip)

        cvt_instruction = ""
        if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
            cvt_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
        elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
            cvt_instruction = "cvt.rn.satfinite.e4m3x2.f32"
        else:
            with cute.arch.elect_one():
                cute.printf("error: unsupported fp8 element type")
            return

        asm_tmpl = (
            "{\n"
            "  .reg .b16 lo;\n"
            "  .reg .b16 hi;\n"
            f"  {cvt_instruction} lo, $2, $1;\n"
            f"  {cvt_instruction} hi, $4, $3;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}"
        )
        packed_i32 = llvm.inline_asm(
            T.i32(),
            [src0, src1, src2, src3],
            asm_tmpl,
            "=r,f,f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        return packed_i32

    @cute.jit
    def cvt_f32x4_to_f8x4(self, fp32x4, fp8x4, loc=None, ip=None):
        packed_i32 = self.cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8x4.element_type)
        fp8x4_i32 = cute.recast_tensor(fp8x4, cutlass.Int32)
        fp8x4_i32[0] = cutlass.Int32(packed_i32)
        return

    @cute.jit
    def cvt_f32_to_f8_to_f32(self, fp32x1, fp8_type, loc=None, ip=None):
        src_fp32 = Float32(fp32x1).ir_value(loc=loc, ip=ip)

        cvt_instruction_downcast = ""
        cvt_instruction_upcast = ""
        if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
            cvt_instruction_downcast = "cvt.rp.satfinite.ue8m0x2.f32"
            cvt_instruction_upcast = "cvt.rn.bf16x2.ue8m0x2"
        elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
            cvt_instruction_downcast = "cvt.rn.satfinite.e4m3x2.f32"
            cvt_instruction_upcast = "cvt.rn.bf16x2.e4m3x2"
        else:
            with cute.arch.elect_one():
                cute.printf("error: unsupported fp8 element type")
            return

        asm_tmpl = "{\n" "  .reg .b16 bf_lo;\n" f"  {cvt_instruction_downcast} bf_lo, 0f00000000, $1;\n" f"  {cvt_instruction_upcast}  $0, bf_lo;\n" "}"
        packed_i32 = llvm.inline_asm(
            T.i32(),
            [src_fp32],
            asm_tmpl,
            "=r,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        vec_bf16_ty = ir.Type.parse("vector<2xbf16>")
        bf2_lo = llvm.bitcast(vec_bf16_ty, packed_i32, loc=loc, ip=ip)
        h0 = vector.extract(bf2_lo, [], [0], loc=loc, ip=ip)
        dst_f32 = arith.extf(Float32.mlir_type, h0, loc=loc, ip=ip)

        return dst_f32

    @cute.jit
    def quant_sfd_row(
        self,
        tile_idx,
        tiled_copy_r2s,
        src,
        pvscale,
        norm_const,
        rcp_limit,
        tRSrD,
    ) -> None:
        # Get absolute max across a vector and Compute SFD
        tCompute = cute.make_rmem_tensor(src.shape, self.acc_dtype)
        tCompute.store(src)
        tTR_rAcc_frg = cute.logical_divide(tCompute, cute.make_layout(self.sf_vec_size))
        acc_frg = tTR_rAcc_frg.load()
        abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
        abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

        avg_fp32 = (
            abs_acc_frg[None, 0].reduce(
                cute.ReductionOp.MAX,
                cutlass.Float32(0.0),
                0,  # Use 0.0 as init for abs values
            )
            * rcp_limit
            * norm_const
        )
        #
        # Manually store pvscale to avoid spilling
        #
        if tile_idx == 0:
            pvscale[None, None, 0][0] = avg_fp32
        elif tile_idx == 1:
            pvscale[None, None, 1][0] = avg_fp32
        elif tile_idx == 2:
            pvscale[None, None, 2][0] = avg_fp32
        elif tile_idx == 3:
            pvscale[None, None, 3][0] = avg_fp32

        #
        # Compute quantized output values and convert to D type
        #
        qpvscale_up = self.cvt_f32_to_f8_to_f32(avg_fp32, self.sf_dtype)

        fp32_max = cutlass.Float32(3.40282346638528859812e38)
        acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
        acc_scale = fmin(acc_scale, fp32_max, nan=True)
        vec = tTR_rAcc_frg[None, 0]
        if cutlass.const_expr(self.vectorized_f32):
            for ei in cutlass.range_constexpr(0, self.sf_vec_size, 2):
                (
                    vec[ei],
                    vec[ei + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (vec[ei], vec[ei + 1]),
                    (acc_scale, acc_scale),
                    rnd="rn",
                    ftz=False,
                )
        else:
            for ei in cutlass.range_constexpr(self.sf_vec_size):
                vec[ei] = vec[ei] * acc_scale

        acc_vec = tiled_copy_r2s.retile(tCompute).load()
        if cutlass.const_expr(not self.use_fp8_ptx_cvt):
            tRSrD.store(acc_vec.to(self.d_dtype))
        else:
            tRSrD_i32 = cute.recast_tensor(tRSrD, cutlass.Int32)
            for ei in cutlass.range_constexpr(0, self.sf_vec_size, 4):
                fp32x4 = cute.make_rmem_tensor(4, cutlass.Float32)
                fp32x4[0] = acc_vec[ei + 0]
                fp32x4[1] = acc_vec[ei + 1]
                fp32x4[2] = acc_vec[ei + 2]
                fp32x4[3] = acc_vec[ei + 3]
                fp8x4_i32 = self.cvt_f32x4_to_f8x4_pack_i32(fp32x4, self.d_dtype)
                tRSrD_i32[ei // 4] = cutlass.Int32(fp8x4_i32)

    @cute.jit
    def quant_sfd_col(
        self,
        tile_idx,
        tiled_copy_r2s,
        src,
        pvscale,
        norm_const,
        rcp_limit,
        tRSrD,
    ):
        # Get absolute max across a vector and Compute SFD
        tCompute = cute.make_rmem_tensor(src.shape, self.acc_dtype)
        tCompute.store(src)
        tTR_rAcc_frg = cute.logical_divide(tCompute, cute.make_layout(self.sf_vec_size))
        acc_frg = tTR_rAcc_frg.load()
        abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
        acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

        tmp_fp32 = cutlass.Float32(0.0)
        fp32_max = cutlass.Float32(3.40282346638528859812e38)
        tidx, _, _ = cute.arch.thread_idx()

        for vi in cutlass.range_constexpr(0, acc_frg.shape[0], 4):
            max_value0 = cutlass.Float32(
                cute.arch.warp_redux_sync(
                    value=acc_frg[vi, 0],
                    kind="fmax",
                    mask_and_clamp=0xFFFFFFFF,
                    nan=True,
                )
            )
            max_value1 = cutlass.Float32(
                cute.arch.warp_redux_sync(
                    value=acc_frg[vi + 1, 0],
                    kind="fmax",
                    mask_and_clamp=0xFFFFFFFF,
                    nan=True,
                )
            )
            max_value2 = cutlass.Float32(
                cute.arch.warp_redux_sync(
                    value=acc_frg[vi + 2, 0],
                    kind="fmax",
                    mask_and_clamp=0xFFFFFFFF,
                    nan=True,
                )
            )
            max_value3 = cutlass.Float32(
                cute.arch.warp_redux_sync(
                    value=acc_frg[vi + 3, 0],
                    kind="fmax",
                    mask_and_clamp=0xFFFFFFFF,
                    nan=True,
                )
            )

            scale = rcp_limit * norm_const
            max_value0, max_value1 = cute.arch.mul_packed_f32x2(
                (max_value0, max_value1),
                (scale, scale),
                rnd="rn",
                ftz=False,
            )
            max_value2, max_value3 = cute.arch.mul_packed_f32x2(
                (max_value2, max_value3),
                (scale, scale),
                rnd="rn",
                ftz=False,
            )

            if tidx % 32 == vi:
                tmp_fp32 = max_value0
            if tidx % 32 == vi + 1:
                tmp_fp32 = max_value1
            if tidx % 32 == vi + 2:
                tmp_fp32 = max_value2
            if tidx % 32 == vi + 3:
                tmp_fp32 = max_value3

            max_value_tensor = cute.make_rmem_tensor(4, cutlass.Float32)
            max_value_tensor[0] = max_value0
            max_value_tensor[1] = max_value1
            max_value_tensor[2] = max_value2
            max_value_tensor[3] = max_value3

            if cutlass.const_expr(not self.use_fp8_ptx_cvt):
                max_value_vec_f8 = max_value_tensor.load().to(self.sf_dtype)
            else:
                max_value_vec_f8 = cute.make_rmem_tensor(4, self.sf_dtype)
                self.cvt_f32x4_to_f8x4(max_value_tensor, max_value_vec_f8)
                max_value_vec_f8 = max_value_vec_f8.load()
            max_value_vec_f32_chunked = max_value_vec_f8.to(cutlass.Float32)
            max_value0 = max_value_vec_f32_chunked[0]
            max_value1 = max_value_vec_f32_chunked[1]
            max_value2 = max_value_vec_f32_chunked[2]
            max_value3 = max_value_vec_f32_chunked[3]

            max_value_rcp0 = cute.arch.rcp_approx(max_value0)
            max_value_rcp1 = cute.arch.rcp_approx(max_value1)
            max_value_rcp2 = cute.arch.rcp_approx(max_value2)
            max_value_rcp3 = cute.arch.rcp_approx(max_value3)

            max_value_rcp0 = fmin(max_value_rcp0, fp32_max, nan=True)
            max_value_rcp1 = fmin(max_value_rcp1, fp32_max, nan=True)
            max_value_rcp2 = fmin(max_value_rcp2, fp32_max, nan=True)
            max_value_rcp3 = fmin(max_value_rcp3, fp32_max, nan=True)

            acc_scale_col0, acc_scale_col1 = cute.arch.mul_packed_f32x2(
                (norm_const, norm_const),
                (max_value_rcp0, max_value_rcp1),
                rnd="rn",
                ftz=False,
            )
            acc_scale_col2, acc_scale_col3 = cute.arch.mul_packed_f32x2(
                (norm_const, norm_const),
                (max_value_rcp2, max_value_rcp3),
                rnd="rn",
                ftz=False,
            )

            tTR_rAcc_frg[vi], tTR_rAcc_frg[vi + 1] = cute.arch.mul_packed_f32x2(
                (tTR_rAcc_frg[vi], tTR_rAcc_frg[vi + 1]),
                (acc_scale_col0, acc_scale_col1),
                rnd="rn",
                ftz=False,
            )
            tTR_rAcc_frg[vi + 2], tTR_rAcc_frg[vi + 3] = cute.arch.mul_packed_f32x2(
                (tTR_rAcc_frg[vi + 2], tTR_rAcc_frg[vi + 3]),
                (acc_scale_col2, acc_scale_col3),
                rnd="rn",
                ftz=False,
            )

            # if tile_idx == 0:
            #     pvscale[None, None, 0][0] = tmp_fp32
            # elif tile_idx == 1:
            #     pvscale[None, None, 1][0] = tmp_fp32
            # elif tile_idx == 2:
            #     pvscale[None, None, 2][0] = tmp_fp32
            # elif tile_idx == 3:
            #     pvscale[None, None, 3][0] = tmp_fp32
            pvscale[None, None, tile_idx][0] = tmp_fp32

        acc_vec = tiled_copy_r2s.retile(tCompute).load()
        if cutlass.const_expr(not self.use_fp8_ptx_cvt):
            tRSrD.store(acc_vec.to(self.d_dtype))
        else:
            tRSrD_i32 = cute.recast_tensor(tRSrD, cutlass.Int32)
            for ei in cutlass.range_constexpr(0, self.sf_vec_size, 4):
                fp32x4 = cute.make_rmem_tensor(4, cutlass.Float32)
                fp32x4[0] = acc_vec[ei + 0]
                fp32x4[1] = acc_vec[ei + 1]
                fp32x4[2] = acc_vec[ei + 2]
                fp32x4[3] = acc_vec[ei + 3]
                fp8x4_i32 = self.cvt_f32x4_to_f8x4_pack_i32(fp32x4, self.d_dtype)
                tRSrD_i32[ei // 4] = cutlass.Int32(fp8x4_i32)

    @cute.jit
    def stg_256(self, ptr, vec8_f32, *, loc=None, ip=None):
        """
        Store 8xf32 (256b) to global memory with L1::no_allocate.
        ptr: pointer (byte addressable)
        vec8_f32: vector<8xf32> to store
        """
        dst = ptr.ir_value(loc=loc, ip=ip) if hasattr(ptr, "ir_value") else ptr
        src = vec8_f32.ir_value(loc=loc, ip=ip) if hasattr(vec8_f32, "ir_value") else vec8_f32
        dummy = llvm.inline_asm(
            T.i32(),
            [
                dst,
                vector.extract(src, [], [0], loc=loc, ip=ip),
                vector.extract(src, [], [1], loc=loc, ip=ip),
                vector.extract(src, [], [2], loc=loc, ip=ip),
                vector.extract(src, [], [3], loc=loc, ip=ip),
                vector.extract(src, [], [4], loc=loc, ip=ip),
                vector.extract(src, [], [5], loc=loc, ip=ip),
                vector.extract(src, [], [6], loc=loc, ip=ip),
                vector.extract(src, [], [7], loc=loc, ip=ip),
            ],
            "st.global.L1::no_allocate.v8.f32 [$1], {$2, $3, $4, $5, $6, $7, $8, $9}; mov.u32 $0, 0;",
            "=r,l,f,f,f,f,f,f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def store_global_memory_256b(self, dst: cute.Tensor, src: cute.Tensor):
        vec_shape = cute.make_layout(8)
        dst_f32 = cute.flatten(cute.recast_tensor(dst, cutlass.Float32))
        src_f32 = cute.flatten(cute.recast_tensor(src, cutlass.Float32))
        dst_vf32x8 = cute.logical_divide(dst_f32, vec_shape)
        src_vf32x8 = cute.logical_divide(src_f32, vec_shape)
        for ei in cutlass.range_constexpr(dst_vf32x8.shape[1]):
            self.stg_256(dst_vf32x8[None, ei].iterator.llvm_ptr, src_vf32x8[None, ei].load())

    @cute.jit
    def tile_info_to_mn_idx(
        self,
        tile_info: cute.Tensor,
    ):
        m_idx = tile_info[0] * cute.size(self.cta_tile_shape_mnk[0])
        n_idx = tile_info[1] * cute.size(self.cta_tile_shape_mnk[1])
        return m_idx, n_idx

    @cute.jit
    def create_and_partition_new_SFDCol(
        self,
        tile_info: cute.Tensor,
        mSFDCol_mnl: cute.Tensor,
    ):
        m_idx, n_idx = self.tile_info_to_mn_idx(tile_info)
        cumsum_tokens = tile_info[5]
        tokens_this_group = tile_info[4]
        n_total = cute.size(mSFDCol_mnl.shape[1])

        sf_tile_idx_begin = cumsum_tokens // cute.size(mSFDCol_mnl.shape[0][0])
        mSFDCol_mnl_new_ptr = mSFDCol_mnl[(None, sf_tile_idx_begin), None, 0].iterator

        sfd_col_quant_layout = cute.tile_to_shape(
            blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
            (tokens_this_group, n_total, mSFDCol_mnl.shape[2]),
            (1, 2, 3),
        )
        regPerSubtile = 4
        sfd_tile = (
            cute.make_layout(128),
            cute.make_layout(32 * regPerSubtile),
        )
        mSFDCol_mnl_new = cute.make_tensor(mSFDCol_mnl_new_ptr, sfd_col_quant_layout)
        gSFDCol_mnl_new = cute.local_tile(mSFDCol_mnl_new, sfd_tile, (None, None, None))

        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((1,), order=(0,))
        copy_atom_sfd_col_quant = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gSFDCol_mnl_new.element_type,
            num_bits_per_copy=8,
        )
        tiled_copy_sfd_col_quant = cute.make_tiled_copy_tv(copy_atom_sfd_col_quant, thr_layout, val_layout)
        tidx = cute.arch.thread_idx()[0]
        thr_copy_sfd_col_quant = tiled_copy_sfd_col_quant.get_slice(tidx)
        tCgSFDCol_mnl = thr_copy_sfd_col_quant.partition_D(cute.filter_zeros(gSFDCol_mnl_new))
        tCgSFDCol_mnl = cute.filter_zeros(tCgSFDCol_mnl)
        return tCgSFDCol_mnl

    # GPU device kernel
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
        smem = utils.SmemAllocator()
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
        tmem = utils.TmemAllocator(
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
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

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
            tile_sched = StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)

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
            tile_sched = StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

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
            tile_sched = StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acd_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

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
            tiled_copy_s2r, tRS_rC1, tRS_rC2, tRS_sC = self.epilog_smem_copy_and_partition_load(tiled_copy_t2r, tTR_rC1, tTR_rC2, epi_tidx, sC)

            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

            tTR_rD1 = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tTR_rD2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD1, tRS_rD2, tRS_sD = self.epilog_smem_copy_and_partition_store(tiled_copy_t2r, tTR_rD1, tTR_rD2, epi_tidx, sD)
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
                ) = self.epilog_smem_copy_and_partition_store(tiled_copy_t2r, tTR_rD1_col, tTR_rD2_col, epi_tidx, sD_col)
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
            tile_sched = StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Load C pipeline
            c_pipeline_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_c_stage)

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

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

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
                mProb = prob[mPosition, 0, 0]
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
                            real_subtile_idx_next = self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - subtile_idx - 1
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
                            if sfd_row_idx_mn[1] * 32 * regPerSubtile < cute.size(cute.shape(mSFDRow_mnl.layout, mode=[1])):
                                cute.autovec_copy(tCrSFDRow, tCgSFDRow)
                            if sfd_col_idx_mn[1] * 32 * regPerSubtile < cute.size(cute.shape(mSFDCol_mnl.layout, mode=[1])):
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
                    # write dProb result to global memory
                    _ = atomic_add_float32(
                        ptr=dprob[(mPosition, None, None)].iterator.llvm_ptr,
                        value=dProbVal,
                    )

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
            tile_sched = StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
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

            c_pipeline_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_c_stage)
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

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor D (register source)
            - tRS_sD: The partitioned tensor D (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_copy_and_partition_load(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tTR_rC1: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory load, then use it to partition register array (destination) and shared memory (source).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_s2r, tSR_rC, tSR_sC) where:
            - tiled_copy_s2r: The tiled copy operation for smem to register copy(s2r)
            - tSR_rC: The partitioned tensor C (register destination)
            - tSR_sC: The partitioned tensor C (smem source)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, PIPE_C)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_D(sC)
        # (S2R, S2R_M, S2R_N)
        tSR_rC = tiled_copy_s2r.retile(tTR_rC)
        tSR_rC1 = tiled_copy_s2r.retile(tTR_rC1)
        return tiled_copy_s2r, tSR_rC, tSR_rC1, tSR_sC

    def epilog_smem_copy_and_partition_store(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rD1: cute.Tensor,
        tTR_rD2: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD1: The partitioned accumulator tensor
        :type tTR_rD1: cute.Tensor
        :param tTR_rD2: The partitioned accumulator tensor
        :type tTR_rD2: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor D (register source)
            - tRS_sD: The partitioned tensor D (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = None
        if cutlass.const_expr(sD is not None):
            tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD1 = tiled_copy_r2s.retile(tTR_rD1)
        tRS_rD2 = tiled_copy_r2s.retile(tTR_rD2)
        return tiled_copy_r2s, tRS_rD1, tRS_rD2, tRS_sD

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_d, bSG_sD, bSG_gD) where:
                - tma_atom_d: The TMA copy atom
                - bSG_sD: The partitioned shared memory tensor D
                - bSG_gD: The partitioned global tensor D
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_d = atom
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_d,
            0,
            cute.make_layout(1),
            sD_for_tma_partition,
            gD_for_tma_partition,
        )
        return bSG_sD, bSG_gD

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        d_dtype: Type[cutlass.Numeric],
        d_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
        store_d_directly: bool,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/D operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param d_layout: Layout of operand D.
        :type d_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, D stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # num_acc_stage = 1

        # Default C/D stages
        num_c_stage = 4 if a_dtype.width == 8 else (4 if store_d_directly else 2)
        num_d_stage = 2 if a_dtype.width == 8 else (0 if store_d_directly else 2)

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and D
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            d_dtype,
            d_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # Mbar bytes
        mbar_helpers_bytes = 1024
        # Sinfo bytes
        sinfo_bytes = 4 * 4 * num_tile_stage
        # C/D bytes
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        d_bytes = d_bytes_per_stage * num_d_stage
        if d_dtype == cutlass.Float8E5M2 or d_dtype == cutlass.Float8E4M3FN:
            d_bytes = d_bytes * 2
        # AMAX bytes
        amax_bytes = BlockScaledContiguousGroupedGemmKernel.get_amax_smem_size() if d_dtype == cutlass.BFloat16 else 0
        # Epilogue bytes
        epi_bytes = c_bytes + d_bytes + amax_bytes

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial D stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage

        # Refine epilogue stages:
        ##num_d_stage += (
        ##    num_smem_capacity
        ##    - occupancy * ab_bytes_per_stage * num_ab_stage
        ##    - occupancy * (mbar_helpers_bytes + epi_bytes)
        ##) // (occupancy * d_bytes_per_stage)

        total_bytes = occupancy * (ab_bytes_per_stage * num_ab_stage + epi_bytes + sinfo_bytes + mbar_helpers_bytes)

        ## cute.printf(
        ##     f"num_acc_stage: {num_acc_stage}, num_ab_stage: {num_ab_stage}, num_c_stage: {num_c_stage}, num_d_stage: {num_d_stage}, num_tile_stage: {num_tile_stage}, total_bytes: {total_bytes}"
        ## )
        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        output_shape: cute.Shape,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor D.

        :param output_shape: The shape (M, N, L) of the output tensor.
        :type output_shape: cute.Shape
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr
        :param raster_along_m: Rasterization order. True for M-major (col-major),
            False for N-major (row-major). When False (N-major), M coordinates
            are guaranteed to be monotonically non-decreasing across consecutive
            tiles processed by the same CTA, which enables incremental expert lookup.
        :type raster_along_m: bool

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        d_layout = cute.make_layout(output_shape)
        d_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gd = cute.zipped_divide(d_layout, tiler=d_shape)
        num_ctas_mnl = cute.slice_(gd.shape, (0, (None, None, None)))
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            raster_along_m=raster_along_m,
        )
        grid = StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def get_amax_smem_size():
        # Note: 4 is hardcoded for num_epilog_warps
        return 4 * cute.size_in_bytes(cutlass.Float32, cute.make_layout((1,)))
