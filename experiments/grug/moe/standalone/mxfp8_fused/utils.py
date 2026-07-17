# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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

"""
Shared utilities for grouped GEMM kernels.

This module contains the tile scheduler classes and helper functions used by both
the forward (grouped_gemm_swiglu) and backward (grouped_gemm_dswiglu) kernels.
"""

from typing import Tuple, Union

from cutlass.cutlass_dsl import (
    Boolean,
    Integer,
    Int32,
    min,
    extract_mlir_values,
    new_from_mlir_values,
    dsl_user_op,
    const_expr,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind
from cutlass.cutlass_dsl import T
from cutlass.cute.typing import Float32, Int32
import cutlass.cute as cute
import cutlass
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineUserType,
    make_pipeline_state,
)

# nvvm.atomicrmw signature differs ACROSS WHEEL BUILDS of nvidia-cutlass-dsl at
# the same version: the 4.5.x x86_64 wheel requires the result type as the
# first positional argument, while the aarch64 cu13 wheel (and 4.6+) infers it.
# Detect once and dispatch.
import inspect as _inspect

_ATOMICRMW_TAKES_RES = "res" in _inspect.signature(nvvm.atomicrmw).parameters


def atomicrmw_compat(res, *, op, ptr, a, loc=None, ip=None):
    """Wheel-portable nvvm.atomicrmw (see note above)."""
    if _ATOMICRMW_TAKES_RES:
        return nvvm.atomicrmw(res, op=op, ptr=ptr, a=a, loc=loc, ip=ip)
    return nvvm.atomicrmw(op=op, ptr=ptr, a=a, loc=loc, ip=ip)


##############################################################################
# Helper functions
##############################################################################


@dsl_user_op
def atomic_add_i32(
    ptr,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Perform an atomic add on an int32 value in global memory."""
    old_value = atomicrmw_compat(
        T.i32(),  # res (ignored by wheels whose atomicrmw infers it)
        op=AtomicOpKind.ADD,
        ptr=ptr,
        a=value.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return Int32(old_value)


@dsl_user_op
def store_i32_to_peer_cluster_smem_async(
    smem_ptr,
    value: Int32,
    mbar_ptr,
    cta_rank_in_cluster,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one int32 to a peer CTA via st.async.shared::cluster."""
    smem_addr = llvm.ptrtoint(T.i32(), smem_ptr.llvm_ptr, loc=loc, ip=ip)
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            smem_addr,
            value.ir_value(loc=loc, ip=ip),
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_addr;
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_addr, $0, $3;
            mapa.shared::cluster.u32 remote_mbar, $2, $3;
            st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [remote_addr], $1, [remote_mbar];
        }}""",
        constraints="r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_on_peer(
    mbar_ptr,
    tx_count: Int32,
    cta_rank_in_cluster,
    *,
    loc=None,
    ip=None,
) -> None:
    """Arrive+expect_tx on a peer CTA mbarrier via inline PTX."""
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
            tx_count.ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_mbar, $0, $1;
            mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote_mbar], $2;
        }}""",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def fmin(
    a: Union[float, Float32],
    b: Union[float, Float32],
    *,
    nan: bool = True,
    loc=None,
    ip=None,
) -> Float32:
    """Compute the minimum of two float32 values with NaN handling.

    :param a: First operand
    :param b: Second operand
    :param nan: If True, propagate NaN values
    :return: Minimum value
    """
    if nan:
        ptx_instr = "min.NaN.f32 $0, $1, $2;"
    else:
        ptx_instr = "min.f32 $0, $1, $2;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            ptx_instr,
            "=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def fmax(a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    ptx_instr = f"max.NaN.f32 $0, $1, $2;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def atomic_max_float32(
    ptr,
    value: Float32,
    *,
    positive_only: bool = True,
    loc=None,
    ip=None,
) -> Float32:
    """Perform atomic max operation on a float32 value in global memory.

    This implementation works correctly for non-negative values (>= 0) using direct bitcast.

    :param ptr: Pointer to the memory location
    :param value: The float32 value to compare and potentially store (should be >= 0)
    :return: The old value at the memory location
    """
    value_int = llvm.bitcast(T.i32(), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

    old_value_int = atomicrmw_compat(
        T.i32(),  # res (ignored by wheels whose atomicrmw infers it)
        op=cutlass._mlir.dialects.nvvm.AtomicOpKind.MAX,
        ptr=ptr,
        a=value_int,
        loc=loc,
        ip=ip,
    )

    return Float32(llvm.bitcast(T.f32(), old_value_int, loc=loc, ip=ip))


def atomic_add_float32(
    ptr,
    value: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Perform atomic add operation on a float32 value in global memory.

    :param ptr: Pointer to the memory location
    :param value: The float32 value to add
    :return: The old value at the memory location
    """
    old_value = atomicrmw_compat(
        T.f32(),  # res (ignored by wheels whose atomicrmw infers it)
        op=cutlass._mlir.dialects.nvvm.AtomicOpKind.FADD,
        ptr=ptr,
        a=value.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )

    return Float32(old_value)


def sigmoid_f32(a: Union[float, Float32], fastmath: bool = False) -> Union[float, Float32]:
    """Compute the sigmoid function: 1 / (1 + exp(-a)).

    :param a: Input value
    :param fastmath: Whether to use fast math approximations
    :return: Sigmoid of input
    """
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(a: Union[float, Float32], fastmath: bool = False) -> Union[float, Float32]:
    """Compute the SiLU (Swish) activation: a * sigmoid(a).

    :param a: Input value
    :param fastmath: Whether to use fast math approximations
    :return: SiLU of input
    """
    return a * sigmoid_f32(a, fastmath=fastmath)


def silu_f32_scaled(
    a: Union[float, Float32],
    alpha: Union[float, Float32] = 1.702,
    fastmath: bool = False,
) -> Union[float, Float32]:
    """Compute the scaled SiLU ``a * sigmoid(alpha * a)`` of the input.

    :param a: Input value.
    :param alpha: Pre-sigmoid scaling factor. Defaults to the GeGLU/GPT-OSS
        constant ``1.702`` for backwards compatibility with the previous
        ``silu_f32_geglu_scaled`` helper.
    :param fastmath: Whether to use fast math approximations.
    """
    return a * sigmoid_f32(a * alpha, fastmath=fastmath)


def silu_f32_geglu_scaled(a: Union[float, Float32], fastmath: bool = False) -> Union[float, Float32]:
    """Backwards-compatible wrapper for :func:`silu_f32_scaled` with ``alpha=1.702``."""
    return silu_f32_scaled(a, alpha=1.702, fastmath=fastmath)


@cute.jit
def search_expert_idx_full(
    padded_offsets_in_lanes: cute.Tensor,
    tile_m_start: cutlass.Int32,
    num_offsets_per_thread: cutlass.Constexpr,
) -> cutlass.Int32:
    """
    Full search algorithm for finding expert_idx. O(E/32) complexity.

    Searches from the beginning of padded_offsets to find the first expert
    whose end offset is greater than tile_m_start.

    :param padded_offsets_in_lanes: Register tensor holding this thread's portion of padded_offsets.
        Layout: padded_offsets_in_lanes[i] corresponds to padded_offsets[lane_id + i * 32]
    :type padded_offsets_in_lanes: cute.Tensor
    :param tile_m_start: Starting M position of the current tile
    :type tile_m_start: cutlass.Int32
    :param num_offsets_per_thread: Number of offsets each thread holds (ceil(expert_cnt/32))
    :type num_offsets_per_thread: cutlass.Constexpr

    :return: The expert index for this tile
    :rtype: cutlass.Int32
    """
    expert_idx = cutlass.Int32(-1)
    round_idx = cutlass.Int32(0)
    not_found = cutlass.Boolean(True)

    while not_found and round_idx < num_offsets_per_thread:
        # Each thread compares its offset with tile_m_start
        is_greater = padded_offsets_in_lanes[round_idx] > tile_m_start

        # Collect comparison results from all threads
        mask = cute.arch.vote_ballot_sync(is_greater)

        # Check if any thread found offset > tile_m_start
        any_found = cute.arch.vote_any_sync(is_greater)
        if any_found:
            # ffs using popc: find position of least significant 1
            lowbit = mask & (-mask)
            first_lane = cute.arch.popc(lowbit - cutlass.Int32(1))
            expert_idx = round_idx * 32 + first_lane
            not_found = cutlass.Boolean(False)  # Terminate loop

        round_idx = round_idx + 1

    return expert_idx


@cute.jit
def search_expert_idx_incremental(
    padded_offsets_in_lanes: cute.Tensor,
    tile_m_start: cutlass.Int32,
    num_offsets_per_thread: cutlass.Constexpr,
    prev_expert_idx: cutlass.Int32,
    prev_expert_end: cutlass.Int32,
) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """
    Incremental search algorithm for finding expert_idx. O(1) average complexity.

    Exploits the monotonicity property: when using N-major rasterization,
    M coordinates are monotonically non-decreasing, so expert_idx >= prev_expert_idx.

    Fast path: If tile_m_start < prev_expert_end, we're still in the same expert.
    Slow path: Search forward from prev_expert_idx + 1.

    :param padded_offsets_in_lanes: Register tensor holding this thread's portion of padded_offsets.
    :type padded_offsets_in_lanes: cute.Tensor
    :param tile_m_start: Starting M position of the current tile
    :type tile_m_start: cutlass.Int32
    :param num_offsets_per_thread: Number of offsets each thread holds
    :type num_offsets_per_thread: cutlass.Constexpr
    :param prev_expert_idx: Expert index from the previous tile (-1 for first tile)
    :type prev_expert_idx: cutlass.Int32
    :param prev_expert_end: End offset of the previous expert (0 for first tile)
    :type prev_expert_end: cutlass.Int32

    :return: Tuple of (expert_idx, new_prev_expert_idx, new_prev_expert_end)
    :rtype: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
    """
    expert_idx = cutlass.Int32(-1)
    new_prev_expert_idx = prev_expert_idx
    new_prev_expert_end = prev_expert_end

    # Fast path: check if still in the same expert
    if tile_m_start < prev_expert_end:
        expert_idx = prev_expert_idx
    else:
        # Slow path: search forward from prev_expert_idx + 1
        start_idx = prev_expert_idx + 1
        start_round = start_idx // 32
        start_lane_offset = start_idx % 32

        round_idx = start_round
        not_found = cutlass.Boolean(True)

        while not_found and round_idx < num_offsets_per_thread:
            is_greater = padded_offsets_in_lanes[round_idx] > tile_m_start
            mask = cute.arch.vote_ballot_sync(is_greater)

            # For the first round, clear bits before start_lane_offset
            if round_idx == start_round:
                clear_mask = (cutlass.Int32(1) << start_lane_offset) - 1
                mask = mask & (~clear_mask)

            if mask != 0:
                lowbit = mask & (-mask)
                first_lane = cute.arch.popc(lowbit - cutlass.Int32(1))
                expert_idx = round_idx * 32 + first_lane
                not_found = cutlass.Boolean(False)

            round_idx = round_idx + 1

        # Update cached values for next iteration
        new_prev_expert_idx = expert_idx
        # Get new_prev_expert_end via shuffle from the lane holding expert_idx
        expert_round = expert_idx // 32
        expert_lane = expert_idx % 32
        new_prev_expert_end = cute.arch.shuffle_sync(
            padded_offsets_in_lanes[expert_round],
            offset=expert_lane,
            mask=0xFFFFFFFF,
            mask_and_clamp=31,
        )

    return expert_idx, new_prev_expert_idx, new_prev_expert_end


##############################################################################
# Static persistent tile scheduler
##############################################################################


class WorkTileInfo:
    """A class to represent information about a work tile.

    :ivar tile_idx: The index of the tile.
    :type tile_idx: cute.Coord
    :ivar is_valid_tile: Whether the tile is valid.
    :type is_valid_tile: Boolean
    """

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.tile_idx)
        values.extend(extract_mlir_values(self.is_valid_tile))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 4
        new_tile_idx = new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> Boolean:
        """Check latest tile returned by the scheduler is valid or not.

        Any scheduling requests after all tasks completed will return an invalid tile.

        :return: The validity of the tile.
        :rtype: Boolean
        """
        return self._is_valid_tile

    @property
    def tile_idx(self) -> cute.Coord:
        """Get the index of the tile.

        :return: The index of the tile.
        :rtype: cute.Coord
        """
        return self._tile_idx


class PersistentTileSchedulerParams:
    """A class to represent parameters for a persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    :ivar cluster_shape_mn: Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
    :type cluster_shape_mn: tuple
    :ivar problem_layout_ncluster_mnl: Layout of the problem in terms of
        number of clusters in (m, n, l) dimensions.
    :type problem_layout_ncluster_mnl: cute.Layout
    """

    @dsl_user_op
    def __init__(
        self,
        problem_shape_ntile_mnl: cute.Shape,
        cluster_shape_mnk: cute.Shape,
        raster_along_m: bool = True,
        swizzle_size: int = 1,
        *,
        loc=None,
        ip=None,
    ):
        """Initializes the PersistentTileSchedulerParams with the given parameters.

        :param problem_shape_ntile_mnl: The shape of the problem in terms of
            number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
        :type problem_shape_ntile_mnl: cute.Shape
        :param cluster_shape_mnk: The shape of the cluster in (m, n) dimensions.
        :type cluster_shape_mnk: cute.Shape
        :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
        :type swizzle_size: int
        :param raster_along_m: Rasterization order of clusters. Only used when swizzle_size > 1.
            True means along M, false means along N.
        :type raster_along_m: bool

        :raises ValueError: If cluster_shape_k is not 1.
        """

        if cluster_shape_mnk[2] != 1:
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")
        if swizzle_size < 1:
            raise ValueError(f"expect swizzle_size >= 1, but get {swizzle_size}")

        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        # cluster_shape_mnk is kept for reconstruction
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self.swizzle_size = swizzle_size
        self._raster_along_m = raster_along_m
        self._loc = loc

        # By default, we follow m major (col-major) raster order, so make a col-major layout
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

        # Apply swizzle if swizzle_size > 1
        if swizzle_size > 1:
            problem_shape_ncluster_mnl = cute.round_up(
                self.problem_layout_ncluster_mnl.shape,
                (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),
            )

            if raster_along_m:
                self.problem_layout_ncluster_mnl = cute.make_layout(
                    (
                        (swizzle_size, problem_shape_ncluster_mnl[0] // swizzle_size),
                        problem_shape_ncluster_mnl[1],
                        problem_shape_ncluster_mnl[2],
                    ),
                    stride=(
                        (1, swizzle_size * problem_shape_ncluster_mnl[1]),
                        swizzle_size,
                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                    ),
                    loc=loc,
                    ip=ip,
                )
            else:
                self.problem_layout_ncluster_mnl = cute.make_layout(
                    (
                        problem_shape_ncluster_mnl[0],
                        (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),
                        problem_shape_ncluster_mnl[2],
                    ),
                    stride=(
                        swizzle_size,
                        (1, swizzle_size * problem_shape_ncluster_mnl[0]),
                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                    ),
                    loc=loc,
                    ip=ip,
                )

        # Create FastDivmod divisors (only when swizzle_size == 1 for correctness)
        # FastDivmod assumes simple col-major layout, incompatible with swizzled layouts
        if swizzle_size == 1:
            problem_layout_size = cute.size(self.problem_layout_ncluster_mnl, loc=loc, ip=ip)
            cluster_count_m = self.problem_layout_ncluster_mnl.shape[0]
            cluster_count_n = self.problem_layout_ncluster_mnl.shape[1]

            # batch_fdd: Used to map linear_idx to work_unit_id (handles persistent scheduling)
            self.batch_fdd = cute.fast_divmod_create_divisor(problem_layout_size, loc=loc, ip=ip)

            # cluster_shape_m_fdd: Used to decode work_unit_id to cluster coordinates
            self.cluster_shape_m_fdd = cute.fast_divmod_create_divisor(cluster_count_m, loc=loc, ip=ip)

            # cluster_shape_n_fdd: Used for the second level decomposition
            self.cluster_shape_n_fdd = cute.fast_divmod_create_divisor(cluster_count_n, loc=loc, ip=ip)
        else:
            # FastDivmod not applicable with swizzling, set to None
            self.batch_fdd = None
            self.cluster_shape_m_fdd = None
            self.cluster_shape_n_fdd = None

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.problem_shape_ntile_mnl,
            self._cluster_shape_mnk,
            self._raster_along_m,
            self.swizzle_size,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))

        # Add FastDivmod divisors to MLIR values for Host->Device transfer
        # Only add non-None values to avoid MLIR type errors
        fastdivmod_values = []
        fastdivmod_indices = []  # Track which FastDivmod objects are present

        for i, (fdd_name, fdd_obj) in enumerate(
            [
                ("batch_fdd", self.batch_fdd),
                ("cluster_shape_m_fdd", self.cluster_shape_m_fdd),
                ("cluster_shape_n_fdd", self.cluster_shape_n_fdd),
            ]
        ):
            if fdd_obj is not None:
                # Extract MLIR values from FastDivmodDivisor objects
                fdd_values = extract_mlir_values(fdd_obj)
                fastdivmod_values.extend(fdd_values)
                fastdivmod_indices.append(i)

        values += fastdivmod_values
        self._values_pos.append(len(fastdivmod_indices))  # Store count of FastDivmod objects, not values
        self._fastdivmod_indices = fastdivmod_indices  # Store for reconstruction

        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        values_copy = list(values)  # Make a copy to avoid modifying original

        # Reconstruct original objects from MLIR values
        for obj, n_items in zip(
            [
                self.problem_shape_ntile_mnl,
                self._cluster_shape_mnk,
                self._raster_along_m,
                self.swizzle_size,
            ],
            self._values_pos[:-1],  # Exclude FastDivmod count
        ):
            obj_list.append(new_from_mlir_values(obj, values_copy[:n_items]))
            values_copy = values_copy[n_items:]

        # Create new params object by calling __init__ with reconstructed values
        # This properly recreates layouts and other derived attributes in the device context
        new_params = PersistentTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

        # Restore FastDivmod divisors from remaining values
        fdd_names = ["batch_fdd", "cluster_shape_m_fdd", "cluster_shape_n_fdd"]

        if hasattr(self, "_fastdivmod_indices") and len(self._fastdivmod_indices) > 0:
            # Override the FastDivmod divisors created by __init__ with reconstructed ones
            for j, original_index in enumerate(self._fastdivmod_indices):
                fdd_name = fdd_names[original_index]
                # Get the original FastDivmodDivisor object
                original_fdd = getattr(self, fdd_name)
                if original_fdd is not None and j < len(values_copy):
                    # Each FastDivmodDivisor has 1 MLIR value
                    reconstructed_fdd = new_from_mlir_values(original_fdd, [values_copy[j]])
                    setattr(new_params, fdd_name, reconstructed_fdd)

        return new_params

    @dsl_user_op
    def get_grid_shape(self, max_active_clusters: Int32, *, loc=None, ip=None) -> Tuple[Integer, Integer, Integer]:
        """Computes the grid shape based on the maximum active clusters allowed.

        :param max_active_clusters: The maximum number of active clusters that
            can run in one wave.
        :type max_active_clusters: Int32

        :return: A tuple containing the grid shape in (m, n, persistent_clusters).
            - m: self.cluster_shape_m.
            - n: self.cluster_shape_n.
            - persistent_clusters: Number of persistent clusters that can run.
        """

        # Total ctas in problem size
        num_ctas_mnl = tuple(cute.size(x) * y for x, y in zip(self.problem_layout_ncluster_mnl.shape, self.cluster_shape_mn)) + (
            self.problem_layout_ncluster_mnl.shape[2],
        )

        num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)

        num_ctas_per_cluster = cute.size(self.cluster_shape_mn, loc=loc, ip=ip)
        # Total ctas that can run in one wave
        num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster

        num_persistent_ctas = min(num_ctas_in_problem, num_ctas_per_wave)
        num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster

        return (*self.cluster_shape_mn, num_persistent_clusters)


class StaticPersistentTileScheduler:
    """A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

    :ivar params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
    :type params: PersistentTileSchedulerParams
    :ivar num_persistent_clusters: Number of persistent clusters that can be launched
    :type num_persistent_clusters: Int32
    :ivar cta_id_in_cluster: ID of the CTA within its cluster
    :type cta_id_in_cluster: cute.Coord
    :ivar _num_tiles_executed: Counter for executed tiles
    :type _num_tiles_executed: Int32
    :ivar _current_work_linear_idx: Current cluster index
    :type _current_work_linear_idx: Int32
    """

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        """Initializes the StaticPersistentTileScheduler with the given parameters.

        :param params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
        :type params: PersistentTileSchedulerParams
        :param num_persistent_clusters: Number of persistent clusters that can be launched.
        :type num_persistent_clusters: Int32
        :param current_work_linear_idx: Current cluster index.
        :type current_work_linear_idx: Int32
        :param cta_id_in_cluster: ID of the CTA within its cluster.
        :type cta_id_in_cluster: cute.Coord
        :param num_tiles_executed: Counter for executed tiles.
        :type num_tiles_executed: Int32
        """
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))

        # CRITICAL: Also extract FastDivmod divisors from params
        values.extend(extract_mlir_values(self.params))

        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "StaticPersistentTileScheduler":
        assert len(values) >= 6
        new_num_persistent_clusters = new_from_mlir_values(self.num_persistent_clusters, [values[0]])
        new_current_work_linear_idx = new_from_mlir_values(self._current_work_linear_idx, [values[1]])
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])

        # Reconstruct params with FastDivmod divisors
        params_values = values[6:]  # Remaining values are from params
        new_params = new_from_mlir_values(self.params, params_values)

        return StaticPersistentTileScheduler(
            new_params,  # Use reconstructed params with FastDivmod divisors
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ):
        """Initialize the static persistent tile scheduler.

        :param params: Parameters for the persistent tile scheduler.
        :type params: PersistentTileSchedulerParams
        :param block_idx: The 3d block index in the format (bidx, bidy, bidz).
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param grid_dim: The 3d grid dimensions for kernel launch.
        :type grid_dim: Tuple[Integer, Integer, Integer]

        :return: A StaticPersistentTileScheduler object.
        :rtype: StaticPersistentTileScheduler
        """

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc, ip=ip)

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidz)

        # CTA id in the cluster
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return StaticPersistentTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: PersistentTileSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        """Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        :param params: Parameters for grid shape calculation.
        :type params: PersistentTileSchedulerParams
        :param max_active_clusters: Maximum active clusters allowed.
        :type max_active_clusters: Int32

        :return: The calculated 3d grid shape.
        :rtype: Tuple[Integer, Integer, Integer]
        """

        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    # private method
    def _get_current_work_for_linear_idx(self, current_work_linear_idx: Int32, *, loc=None, ip=None) -> WorkTileInfo:
        """Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

        :param current_work_linear_idx: The linear index of the current work.
        :type current_work_linear_idx: Int32

        :return: An object containing information about the current tile coordinates
            and validity status.
        :rtype: WorkTileInfo
        """

        is_valid = current_work_linear_idx < cute.size(self.params.problem_layout_ncluster_mnl, loc=loc, ip=ip)

        # Choose coordinate calculation method based on swizzle configuration
        if self.params.swizzle_size == 1:
            # Use FastDivmod optimization for non-swizzled layouts
            cur_cluster_coord = self._get_cluster_work_idx_with_fastdivmod(current_work_linear_idx, loc=loc, ip=ip)
        else:
            # Use get_flat_coord for swizzled layouts (FastDivmod doesn't support them)
            cur_cluster_coord = self.params.problem_layout_ncluster_mnl.get_flat_coord(current_work_linear_idx, loc=loc, ip=ip)

        # cur_tile_coord is a tuple of i32 values
        cur_tile_coord = tuple(
            Int32(x) * Int32(z) + Int32(y)
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,
                (*self.params.cluster_shape_mn, Int32(1)),
            )
        )

        return WorkTileInfo(cur_tile_coord, is_valid)

    def _get_cluster_work_idx_with_fastdivmod(self, current_work_linear_idx: Int32, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        """FastDivmod optimized CLUSTER coordinate calculation.

        CRITICAL: This should mimic problem_layout_ncluster_mnl.get_hier_coord()
        which returns CLUSTER coordinates, not tile coordinates!

        :param current_work_linear_idx: Linear index in the work space
        :type current_work_linear_idx: Int32
        :return: Cluster coordinates (m, n, l) or None if FastDivmod not available
        :rtype: Tuple[Int32, Int32, Int32] or None
        """

        # Step 1: Handle persistent scheduling - map linear_idx to work_unit_id
        work_iteration, work_unit_id = divmod(current_work_linear_idx, self.params.batch_fdd)

        # Step 2: Decode work_unit_id using FastDivmod objects
        # The layout structure is: problem_layout_ncluster_mnl has shape (cluster_count_m, cluster_count_n, batch_count)
        # work_unit_id needs to be decomposed into (batch_l, cluster_n, cluster_m) in little-endian order

        if self.params._raster_along_m:
            # First, get cluster_m using cluster_shape_m_fdd
            cluster_n_batch, cluster_m = divmod(work_unit_id, self.params.cluster_shape_m_fdd)

            # Then decode cluster_n_batch to get cluster_n and batch_l using FastDivmod
            batch_l, cluster_n = divmod(cluster_n_batch, self.params.cluster_shape_n_fdd)
        else:
            # First, get cluster_n using cluster_shape_n_fdd
            cluster_m_batch, cluster_n = divmod(work_unit_id, self.params.cluster_shape_n_fdd)

            # Then decode cluster_m_batch to get cluster_m and batch_l using FastDivmod
            batch_l, cluster_m = divmod(cluster_m_batch, self.params.cluster_shape_m_fdd)

        return (cluster_m, cluster_n, batch_l)

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._current_work_linear_idx += Int32(advance_count) * Int32(self.num_persistent_clusters)
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


class _DynamicSchedState:
    """Runtime-only state required by DynamicPersistentTileScheduler."""

    def __init__(
        self,
        counter_ptr,
        broadcast_ptr,
        is_leader_cta: Boolean,
        producer_state,
        consumer_state,
    ):
        self.counter_ptr = counter_ptr
        self.broadcast_ptr = broadcast_ptr
        self.is_leader_cta = is_leader_cta
        self.producer_state = producer_state
        self.consumer_state = consumer_state

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.counter_ptr))
        values.extend(extract_mlir_values(self.broadcast_ptr))
        values.extend(extract_mlir_values(self.is_leader_cta))
        values.extend(extract_mlir_values(self.producer_state))
        values.extend(extract_mlir_values(self.consumer_state))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "_DynamicSchedState":
        idx = 0

        counter_len = len(extract_mlir_values(self.counter_ptr))
        new_counter_ptr = new_from_mlir_values(self.counter_ptr, values[idx : idx + counter_len])
        idx += counter_len

        broadcast_len = len(extract_mlir_values(self.broadcast_ptr))
        new_broadcast_ptr = new_from_mlir_values(self.broadcast_ptr, values[idx : idx + broadcast_len])
        idx += broadcast_len

        is_leader_len = len(extract_mlir_values(self.is_leader_cta))
        new_is_leader_cta = new_from_mlir_values(self.is_leader_cta, values[idx : idx + is_leader_len])
        idx += is_leader_len

        producer_len = len(extract_mlir_values(self.producer_state))
        new_producer_state = new_from_mlir_values(self.producer_state, values[idx : idx + producer_len])
        idx += producer_len

        consumer_len = len(extract_mlir_values(self.consumer_state))
        new_consumer_state = new_from_mlir_values(self.consumer_state, values[idx : idx + consumer_len])

        return _DynamicSchedState(
            new_counter_ptr,
            new_broadcast_ptr,
            new_is_leader_cta,
            new_producer_state,
            new_consumer_state,
        )


class DynamicPersistentTileScheduler(StaticPersistentTileScheduler):
    """Cluster-granular dynamic persistent scheduler using a global atomic counter."""

    @staticmethod
    def make_storage_struct():
        """Return the shared-memory storage required by dynamic scheduling."""

        @cute.struct
        class DynamicSchedulerStorage:
            cluster_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            sClusterIdx: cute.struct.MemRange[cutlass.Int32, 1]

        return DynamicSchedulerStorage

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        dynamic_state: _DynamicSchedState,
        sched_storage=None,
        cluster_pipeline=None,
    ):
        super().__init__(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )
        self._dynamic_state = dynamic_state
        self._sched_storage = sched_storage
        self.cluster_pipeline = cluster_pipeline

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self._dynamic_state))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DynamicPersistentTileScheduler":
        assert len(values) >= 6
        new_num_persistent_clusters = new_from_mlir_values(self.num_persistent_clusters, [values[0]])
        new_current_work_linear_idx = new_from_mlir_values(self._current_work_linear_idx, [values[1]])
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])

        params_len = len(extract_mlir_values(self.params))
        params_start = 6
        params_end = params_start + params_len
        new_params = new_from_mlir_values(self.params, values[params_start:params_end])
        new_dynamic_state = new_from_mlir_values(self._dynamic_state, values[params_end:])

        return DynamicPersistentTileScheduler(
            new_params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            new_dynamic_state,
            sched_storage=self._sched_storage,
            cluster_pipeline=self.cluster_pipeline,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        counter_ptr,
        sched_storage,
        *,
        loc=None,
        ip=None,
    ) -> "DynamicPersistentTileScheduler":
        """Initialize the dynamic persistent tile scheduler."""

        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc, ip=ip)

        bidx, bidy, _ = block_idx

        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        current_work_linear_idx = Int32(-1)
        num_tiles_executed = Int32(0)

        dynamic_state = _DynamicSchedState(
            counter_ptr=counter_ptr,
            broadcast_ptr=sched_storage.sClusterIdx.data_ptr(),
            is_leader_cta=Boolean((cta_id_in_cluster[0] + cta_id_in_cluster[1] + cta_id_in_cluster[2]) == Int32(0)),
            producer_state=make_pipeline_state(PipelineUserType.Producer, 1, loc=loc, ip=ip),
            consumer_state=make_pipeline_state(PipelineUserType.Consumer, 1, loc=loc, ip=ip),
        )

        return DynamicPersistentTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            dynamic_state,
            sched_storage=sched_storage,
            cluster_pipeline=None,
        )

    def internal_init(self):
        """Create the internal cluster broadcast pipeline before warp specialization."""
        cluster_size = self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1]
        self.cluster_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=self._sched_storage.cluster_mbar.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 32 * cluster_size),
            defer_sync=True,
        )

    @dsl_user_op
    @cute.jit
    def _fetch_next_cluster_idx(self, *, loc=None, ip=None) -> Int32:
        """Fetch the next cluster linear index and broadcast it within the cluster."""
        ds = self._dynamic_state
        broadcast_tensor = cute.make_tensor(ds.broadcast_ptr, cute.make_layout((1,)))
        cluster_idx = Int32(0)

        if ds.is_leader_cta:
            self.cluster_pipeline.producer_acquire(ds.producer_state)
            full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(ds.producer_state.index, loc=loc, ip=ip)
            tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
            lane_idx = tidx % Int32(32)
            cluster_size = self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1]

            atomic_idx = Int32(0)
            if lane_idx == 0:
                atomic_idx = atomic_add_i32(
                    ds.counter_ptr.llvm_ptr,
                    Int32(1),
                    loc=loc,
                    ip=ip,
                )
            atomic_idx = cute.arch.shuffle_sync(
                atomic_idx,
                offset=0,
                mask=0xFFFFFFFF,
                mask_and_clamp=31,
            )

            if lane_idx < Int32(cluster_size):
                store_i32_to_peer_cluster_smem_async(
                    ds.broadcast_ptr,
                    atomic_idx,
                    full_barrier_ptr,
                    lane_idx,
                    loc=loc,
                    ip=ip,
                )

            self._cluster_producer_commit(loc=loc, ip=ip)

        ds.producer_state.advance()
        self.cluster_pipeline.consumer_wait(ds.consumer_state)
        cluster_idx = broadcast_tensor[0]
        return cluster_idx

    @dsl_user_op
    @cute.jit
    def _release_cluster_idx(self, *, loc=None, ip=None) -> None:
        """Release the cluster broadcast slot after consumption."""
        ds = self._dynamic_state
        self._cluster_consumer_release(loc=loc, ip=ip)
        ds.consumer_state.advance()

    @dsl_user_op
    @cute.jit
    def _cluster_producer_commit(self, *, loc=None, ip=None) -> None:
        """Fan out a single cluster index to the full barrier."""
        ds = self._dynamic_state
        tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
        lane_idx = tidx % Int32(32)
        cluster_size = self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1]
        if ds.is_leader_cta and lane_idx < Int32(cluster_size):
            mbarrier_arrive_expect_tx_on_peer(
                self.cluster_pipeline.sync_object_full.get_barrier(ds.producer_state.index, loc=loc, ip=ip),
                Int32(4),
                lane_idx,
                loc=loc,
                ip=ip,
            )

    @dsl_user_op
    @cute.jit
    def _cluster_consumer_release(self, *, loc=None, ip=None) -> None:
        """Release one consumed cluster index from the broadcast pipeline."""
        ds = self._dynamic_state
        self.cluster_pipeline.sync_object_empty.arrive(ds.consumer_state.index, Int32(0))

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.cluster_pipeline is None):
            raise RuntimeError("DynamicPersistentTileScheduler.internal_init() must be called before use.")
        self._current_work_linear_idx = self._fetch_next_cluster_idx(loc=loc, ip=ip)
        work_tile = self.get_current_work(loc=loc, ip=ip)
        self._release_cluster_idx(loc=loc, ip=ip)
        return work_tile

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        if const_expr(advance_count != 1):
            raise ValueError("DynamicPersistentTileScheduler only supports advance_count == 1")
        self._current_work_linear_idx = self._fetch_next_cluster_idx(loc=loc, ip=ip)
        self._release_cluster_idx(loc=loc, ip=ip)
        self._num_tiles_executed += Int32(1)


class StaticPersistentRuntimeTileScheduler(StaticPersistentTileScheduler):
    """A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.

    This scheduler will always launch all the SMs and the scheduler will generate
    the real tile info for each SM.

    :ivar params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
    :type params: PersistentTileSchedulerParams
    :ivar num_persistent_clusters: Number of persistent clusters that can be launched
    :type num_persistent_clusters: Int32
    :ivar cta_id_in_cluster: ID of the CTA within its cluster
    :type cta_id_in_cluster: cute.Coord
    :ivar _num_tiles_executed: Counter for executed tiles
    :type _num_tiles_executed: Int32
    :ivar _current_work_linear_idx: Current cluster index
    :type _current_work_linear_idx: Int32
    """

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        inner_mode: int = 1,
    ):
        """Initializes the StaticPersistentRuntimeTileScheduler with the given parameters.

        :param params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
        :type params: PersistentTileSchedulerParams
        :param num_persistent_clusters: Number of persistent clusters that can be launched.
        :type num_persistent_clusters: Int32
        :param current_work_linear_idx: Current cluster index.
        :type current_work_linear_idx: Int32
        :param cta_id_in_cluster: ID of the CTA within its cluster.
        :type cta_id_in_cluster: cute.Coord
        :param num_tiles_executed: Counter for executed tiles.
        :type num_tiles_executed: Int32
        :param inner_mode: The inner mode along which the linear index will be decomposed first.
        :type inner_mode: int
        """
        super().__init__(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )
        if inner_mode not in [0, 1]:
            raise ValueError(f"inner_mode must be 0(for M mode) or 1(for N mode), but got {inner_mode}")
        self.inner_mode = inner_mode

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "StaticPersistentRuntimeTileScheduler":
        assert len(values) >= 6
        new_num_persistent_clusters = new_from_mlir_values(self.num_persistent_clusters, [values[0]])
        new_current_work_linear_idx = new_from_mlir_values(self._current_work_linear_idx, [values[1]])
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])

        # Reconstruct params with FastDivmod divisors (same as parent class)
        params_values = values[6:]  # Remaining values are from params
        new_params = new_from_mlir_values(self.params, params_values)

        return StaticPersistentRuntimeTileScheduler(
            new_params,  # Use reconstructed params with FastDivmod divisors
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            self.inner_mode,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        inner_mode: int = 1,
        *,
        loc=None,
        ip=None,
    ):
        """Initialize the static persistent tile scheduler.

        :param params: Parameters for the persistent tile scheduler.
        :type params: PersistentTileSchedulerParams
        :param block_idx: The 3d block index in the format (bidx, bidy, bidz).
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param grid_dim: The 3d grid dimensions for kernel launch.
        :type grid_dim: Tuple[Integer, Integer, Integer]
        :param inner_mode: The inner mode along which the linear index will be decomposed first.
        :type inner_mode: int

        :return: A StaticPersistentRuntimeTileScheduler object.
        :rtype: StaticPersistentRuntimeTileScheduler
        """

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc, ip=ip)

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidz)

        # CTA id in the cluster
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return StaticPersistentRuntimeTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            inner_mode,
        )

    # private method
    def _get_current_work_for_linear_idx(self, current_work_linear_idx: Int32, *, loc=None, ip=None) -> WorkTileInfo:
        """Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

        :param current_work_linear_idx: The linear index of the current work.
        :type current_work_linear_idx: Int32

        :return: An object containing information about the current tile coordinates
            and validity status.
        :rtype: WorkTileInfo
        """
        ntile_shape = self.params.problem_layout_ncluster_mnl.shape
        int_max = 2147483647
        if const_expr(self.inner_mode == 1):
            ntile_layout = cute.make_layout((int_max, ntile_shape[1]), stride=(ntile_shape[1], 1))
        else:
            ntile_layout = cute.make_layout((ntile_shape[0], int_max), stride=(1, ntile_shape[0]))
        cluster_tile_coord_mn = ntile_layout.get_hier_coord(current_work_linear_idx)
        cur_tile_coord = (
            cluster_tile_coord_mn[0],
            cluster_tile_coord_mn[1],
            Int32(0),
        )

        # it is determined by kernel implementation
        is_valid = True

        return WorkTileInfo(cur_tile_coord, is_valid)
