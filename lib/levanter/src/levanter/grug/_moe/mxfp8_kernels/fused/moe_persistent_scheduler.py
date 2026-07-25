# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# ruff: noqa

"""
MoE Persistent Tile Scheduler

A specialized tile scheduler for MoE (Mixture of Experts) grouped GEMM operations.
This scheduler handles tile iteration across all experts, producing MoEWorkTileInfo
(expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt) for each tile.

Scenarios:
- 2Dx3D (Forward/DGrad): A(tokens_sum, hidden) x B(experts, intermediate, hidden) -> C(tokens_sum, intermediate)
- 2Dx2D (WGrad): A(intermediate, tokens_sum) x B(hidden, tokens_sum) -> C(experts, intermediate, hidden)

Key design principle:
- Scheduler is ONLY responsible for tile iteration (tensor-agnostic, TMA-agnostic)
- Domain conversion (fake tensor -> real expert tensor) is handled by MoESchedExtension
- TMA descriptor management is handled by OnlineTensormapDescCreator
- The kernel orchestrates all three components
"""

from typing import List, Tuple, Literal, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Integer,
    T,
    extract_mlir_values,
    new_from_mlir_values,
    const_expr,
    dsl_user_op,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm, llvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind
from .utils import atomicrmw_compat
from cutlass._mlir.dialects import cute as _cute_ir
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineUserType,
    make_pipeline_state,
)

# =============================================================================
# Helpers
# =============================================================================


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


# =============================================================================
# Work Tile Info
# =============================================================================


class MoEWorkTileInfo:
    """
    Work tile information for MoE scheduler.

    Contains CTA-level tile information for executor warps:
    - expert_idx: Which expert (-1 means invalid/done)
    - tile_m_idx: CTA tile index along GEMM M dimension
    - tile_n_idx: CTA tile index along GEMM N dimension
    - k_tile_cnt: Number of CTA tiles along K dimension

    Note: These are CTA-level indices, not cluster-level.
    tile_l_idx is always 0 for MoE, executor can hardcode it.

    For 2Dx3D (Forward):
        M = tokens_i (dynamic), N = intermediate (fixed), K = hidden (fixed)

    For 2Dx2D (Backward):
        M = intermediate (fixed), N = hidden (fixed), K = tokens_i (dynamic)
    """

    def __init__(
        self,
        expert_idx: Int32,  # -1 means invalid tile
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        k_tile_cnt: Int32,
    ):
        self.expert_idx = expert_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.k_tile_cnt = k_tile_cnt

    @property
    def is_valid_tile(self) -> Boolean:
        """Check if this is a valid work tile (expert_idx >= 0)."""
        return self.expert_idx >= Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.expert_idx)
        values.extend(extract_mlir_values(self.tile_m_idx))
        values.extend(extract_mlir_values(self.tile_n_idx))
        values.extend(extract_mlir_values(self.k_tile_cnt))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEWorkTileInfo":
        assert len(values) == 4
        return MoEWorkTileInfo(
            expert_idx=new_from_mlir_values(self.expert_idx, [values[0]]),
            tile_m_idx=new_from_mlir_values(self.tile_m_idx, [values[1]]),
            tile_n_idx=new_from_mlir_values(self.tile_n_idx, [values[2]]),
            k_tile_cnt=new_from_mlir_values(self.k_tile_cnt, [values[3]]),
        )

    def to_rmem_tensor(self):
        """Pack work tile info fields into an rmem tensor of shape (4,) for vectorized smem copy."""
        rmem = cute.make_rmem_tensor((4,), Int32)
        rmem[0] = self.expert_idx
        rmem[1] = self.tile_m_idx
        rmem[2] = self.tile_n_idx
        rmem[3] = self.k_tile_cnt
        return rmem

    @staticmethod
    def from_rmem_tensor(rmem) -> "MoEWorkTileInfo":
        """Unpack work tile info from an rmem tensor of shape (4,)."""
        return MoEWorkTileInfo(
            expert_idx=rmem[0],
            tile_m_idx=rmem[1],
            tile_n_idx=rmem[2],
            k_tile_cnt=rmem[3],
        )


# =============================================================================
# Scheduler Parameters
# =============================================================================


class MoESchedulerParams:
    """
    Parameters for MoE tile scheduler.

    Uses unified semantics for both scenarios:
    - expert_shape: (expert_cnt, intermediate, hidden)

    For 2Dx3D: GEMM is (M=tokens_i, N=intermediate, K=hidden) per expert
    For 2Dx2D: GEMM is (M=hidden, N=intermediate, K=tokens_i) per expert

    Tile hierarchy:
    - cta_tile_shape_mnk: Single CTA tile shape (tile_m, tile_n, tile_k)
    - cluster_shape_mn: CTAs per cluster (cluster_m, cluster_n)
    - cluster_tile_shape_mn: Cluster tile shape = cta_tile_shape * cluster_shape

    This class is used both on host (for grid shape calculation) and on device
    (stored in scheduler). Codegen-time constants (scenario, cta_tile_shape_mnk,
    cluster_shape_mn) are NOT serialized to MLIR values.
    """

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        expert_shape: Tuple[int | Int32, int | Int32, int | Int32],  # (expert_cnt, intermediate, hidden)
        cta_tile_shape_mnk: Tuple[int, int, int],  # (tile_m, tile_n, tile_k)
        cluster_shape_mn: Tuple[int, int],  # (cluster_m, cluster_n)
        use_dynamic_sched: bool = False,
    ):
        self.scenario = scenario
        e, i, h = expert_shape
        self.expert_cnt = e if isinstance(e, Int32) else Int32(e)
        self.intermediate = i if isinstance(i, Int32) else Int32(i)
        self.hidden = h if isinstance(h, Int32) else Int32(h)
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn
        if scenario == "2Dx2D":
            use_dynamic_sched = True
        self.use_dynamic_sched = use_dynamic_sched

    @property
    def cluster_tile_m(self) -> int:
        """Cluster tile size along M = cta_tile_m * cluster_m."""
        return self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]

    @property
    def cluster_tile_n(self) -> int:
        """Cluster tile size along N = cta_tile_n * cluster_n."""
        return self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1]

    @property
    def cta_tile_k(self) -> int:
        """CTA tile size along K (same as cluster since cluster_k = 1)."""
        return self.cta_tile_shape_mnk[2]

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Only serialize runtime values, not codegen-time constants."""
        values = []
        values.extend(extract_mlir_values(self.expert_cnt))
        values.extend(extract_mlir_values(self.intermediate))
        values.extend(extract_mlir_values(self.hidden))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoESchedulerParams":
        assert len(values) == 3
        return MoESchedulerParams(
            scenario=self.scenario,
            expert_shape=(
                new_from_mlir_values(self.expert_cnt, [values[0]]),
                new_from_mlir_values(self.intermediate, [values[1]]),
                new_from_mlir_values(self.hidden, [values[2]]),
            ),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )

    @staticmethod
    def get_grid_shape(
        params: "MoESchedulerParams",
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        """
        Compute grid shape for kernel launch.

        2Dx3D uses persistent clusters. 2Dx2D uses a full CTA grid with
        cluster-level cancellation to rebalance uneven expert K lengths.
        """
        if params.scenario == "2Dx2D":
            cm, cn = params.cluster_shape_mn
            tm, tn = params.cta_tile_shape_mnk[0], params.cta_tile_shape_mnk[1]
            grid_m = ((params.hidden + tm * cm - 1) // (tm * cm)) * cm
            grid_n = ((params.intermediate + tn * cn - 1) // (tn * cn)) * cn
            return (grid_m, grid_n, params.expert_cnt)
        return (
            params.cluster_shape_mn[0],
            params.cluster_shape_mn[1],
            max_active_clusters,
        )


# =============================================================================
# Dynamic Scheduling State
# =============================================================================


class _DynamicSchedState:
    """Runtime state for dynamic tile scheduling. All fields are MLIR-serializable."""

    def __init__(self, counter_ptr, broadcast_ptr, is_leader_cta, producer_state, consumer_state):
        self.counter_ptr = counter_ptr  # Pointer: gmem atomic counter
        self.broadcast_ptr = broadcast_ptr  # Pointer: smem broadcast base
        self.is_leader_cta = is_leader_cta  # Boolean
        self.producer_state = producer_state  # PipelineState
        self.consumer_state = consumer_state  # PipelineState

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.counter_ptr))
        values.extend(extract_mlir_values(self.broadcast_ptr))
        values.extend(extract_mlir_values(self.is_leader_cta))
        values.extend(extract_mlir_values(self.producer_state))
        values.extend(extract_mlir_values(self.consumer_state))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "_DynamicSchedState":
        idx = 0
        counter_len = len(extract_mlir_values(self.counter_ptr))
        new_counter = new_from_mlir_values(self.counter_ptr, values[idx : idx + counter_len])
        idx += counter_len

        broadcast_len = len(extract_mlir_values(self.broadcast_ptr))
        new_broadcast = new_from_mlir_values(self.broadcast_ptr, values[idx : idx + broadcast_len])
        idx += broadcast_len

        new_is_leader = new_from_mlir_values(self.is_leader_cta, [values[idx]])
        idx += 1

        prod_len = len(extract_mlir_values(self.producer_state))
        new_prod = new_from_mlir_values(self.producer_state, values[idx : idx + prod_len])
        idx += prod_len

        cons_len = len(extract_mlir_values(self.consumer_state))
        new_cons = new_from_mlir_values(self.consumer_state, values[idx : idx + cons_len])
        idx += cons_len

        return _DynamicSchedState(new_counter, new_broadcast, new_is_leader, new_prod, new_cons)


# =============================================================================
# Scheduler (Device-side)
# =============================================================================


class MoEPersistentTileScheduler:
    """
    Persistent tile scheduler specialized for MoE grouped GEMM.

    This scheduler is ONLY responsible for tile iteration. It does NOT know
    about tensor types, TMA descriptors, or domain conversion. Those concerns
    are handled by MoESchedExtension and OnlineTensormapDescCreator respectively.

    Supports two scheduling modes:
    - Static (default): Deterministic strided scheduling (linear_idx = bidz + i * stride)
    - Dynamic: Global atomic counter + cluster-level pipeline broadcast for load balancing

    Architecture:
    - Scheduler warp: Holds scheduler instance, iterates tiles, broadcasts work_tile_info
    - Executor warps: Read work_tile_info from smem, use MoESchedExtension for
      domain conversion and TMA desc selection

    The scheduler handles:
    - 2Dx3D: Dynamic M per expert (from offs), fixed N (intermediate) and K (hidden)
    - 2Dx2D: Fixed M (intermediate) and N (hidden), dynamic K per expert (reduction axis)

    Usage (Scheduler warp):
        scheduler = MoEPersistentTileScheduler.create(params, offs, block_idx, grid_dim)
        work_tile_info = scheduler.initial_work_tile_info()
        # Broadcast work_tile_info to smem...

        while work_tile_info.is_valid_tile:
            # ... do work ...
            work_tile_info = scheduler.advance_to_next_work()
            # Broadcast work_tile_info to smem...

    Usage (Executor warps - via MoESchedExtension):
        # Read work_tile_info from smem...
        real_a, desc_a = ext.get_gmem_tensor("a", tma_tensor_a, offs, work_tile_info)
        real_b, desc_b = ext.get_gmem_tensor("b", tma_tensor_b, offs, work_tile_info)
        real_c, desc_c = ext.get_gmem_tensor("c", tma_tensor_c, offs, work_tile_info)
    """

    # =========================================================================
    # Static helpers for kernel-side resource allocation
    # =========================================================================

    @staticmethod
    def make_storage_struct(
        num_tile_stages: int,
        use_dynamic_sched: bool,
    ):
        """Return a cute.struct type aggregating all scheduler-owned shared memory.

        The kernel should embed this struct in its SharedStorage definition.
        Fields are zero-sized when the corresponding feature is disabled.

        :param num_tile_stages: Number of tile info pipeline stages
        :param use_dynamic_sched: Whether dynamic scheduling is enabled
        :return: A cute.struct type
        """
        num_cluster_mbar = 2 if use_dynamic_sched else 0
        num_cluster_broadcast = 4 if use_dynamic_sched else 0

        @cute.struct
        class SchedulerStorage:
            tile_info_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * num_tile_stages],
                16,
            ]
            cluster_mbar: cute.struct.MemRange[cutlass.Int64, num_cluster_mbar]
            cluster_broadcast: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, num_cluster_broadcast],
                16,
            ]

        return SchedulerStorage

    def internal_init(self):
        """Create the internal cluster broadcast pipeline (dynamic mode only).

        Must be called before warp specialization so that warp 0 participates
        in mbarrier init.  No-op for static mode.
        """
        if const_expr(self.use_dynamic_sched):
            cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
            if const_expr(self.scenario == "2Dx2D"):
                self.cluster_pipeline = pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=self._sched_storage.cluster_mbar.data_ptr(),
                    num_stages=1,
                    producer_group=CooperativeGroup(Agent.Thread),
                    consumer_group=CooperativeGroup(Agent.Thread, 32 * cluster_size),
                    tx_count=16,
                    cta_layout_vmnk=cute.make_layout((*self.cluster_shape_mn, 1, 1)),
                    defer_sync=True,
                )
            else:
                self.cluster_pipeline = pipeline.PipelineAsync.create(
                    barrier_storage=self._sched_storage.cluster_mbar.data_ptr(),
                    num_stages=1,
                    producer_group=CooperativeGroup(Agent.Thread, 1),
                    consumer_group=CooperativeGroup(Agent.Thread, 32 * cluster_size),
                    defer_sync=True,
                )

    # =========================================================================
    # Instance
    # =========================================================================

    def __init__(
        self,
        # Params (contains scenario, expert_cnt, intermediate, hidden, tile/cluster shapes)
        params: MoESchedulerParams,
        # Runtime tensor for scheduling
        offs: cute.Tensor,  # (experts,) cumsum of token counts
        # Scheduling state
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        # Expert tracking state (for O(1) advance within same expert)
        current_expert_idx: Int32,
        expert_tile_start: Int32,  # cumsum of tiles before current expert
        expert_tile_end: Int32,  # cumsum of tiles including current expert
        # Dynamic scheduling (None for static mode)
        dynamic_state: Optional[_DynamicSchedState] = None,
        sched_storage=None,
        cluster_pipeline=None,
    ):
        self.params = params
        self.offs = offs
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        # Expert tracking
        self.current_expert_idx = current_expert_idx
        self.expert_tile_start = expert_tile_start
        self.expert_tile_end = expert_tile_end
        # Dynamic scheduling
        # _dynamic_state is MLIR-serialized; sched_storage and cluster_pipeline are Python objects
        self._dynamic_state = dynamic_state
        self._sched_storage = sched_storage
        self.cluster_pipeline = cluster_pipeline

    # =========================================================================
    # Convenience accessors for params
    # All have no-op setters so that cute_dsl copy_members (setattr) works.
    # =========================================================================

    @property
    def scenario(self) -> Literal["2Dx3D", "2Dx2D"]:
        return self.params.scenario

    @scenario.setter
    def scenario(self, value):
        pass

    @property
    def expert_cnt(self) -> Int32:
        return self.params.expert_cnt

    @expert_cnt.setter
    def expert_cnt(self, value):
        pass

    @property
    def intermediate(self) -> Int32:
        return self.params.intermediate

    @intermediate.setter
    def intermediate(self, value):
        pass

    @property
    def hidden(self) -> Int32:
        return self.params.hidden

    @hidden.setter
    def hidden(self, value):
        pass

    @property
    def cta_tile_shape_mnk(self) -> Tuple[int, int, int]:
        return self.params.cta_tile_shape_mnk

    @cta_tile_shape_mnk.setter
    def cta_tile_shape_mnk(self, value):
        pass

    @property
    def cluster_shape_mn(self) -> Tuple[int, int]:
        return self.params.cluster_shape_mn

    @cluster_shape_mn.setter
    def cluster_shape_mn(self, value):
        pass

    @property
    def cluster_tile_m(self) -> int:
        return self.params.cluster_tile_m

    @cluster_tile_m.setter
    def cluster_tile_m(self, value):
        pass

    @property
    def cluster_tile_n(self) -> int:
        return self.params.cluster_tile_n

    @cluster_tile_n.setter
    def cluster_tile_n(self, value):
        pass

    @property
    def cta_tile_k(self) -> int:
        return self.params.cta_tile_k

    @cta_tile_k.setter
    def cta_tile_k(self, value):
        pass

    @property
    def use_dynamic_sched(self) -> bool:
        return self.params.use_dynamic_sched

    @use_dynamic_sched.setter
    def use_dynamic_sched(self, value):
        pass

    # =========================================================================
    # MLIR value serialization (for SSA value passing in device code)
    # =========================================================================

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        # Params (only runtime values are extracted)
        values.extend(extract_mlir_values(self.params))
        # Runtime tensor for scheduling
        values.extend(extract_mlir_values(self.offs))
        # Scheduling state
        values.extend(extract_mlir_values(self.num_persistent_clusters))
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        # Expert tracking state
        values.extend(extract_mlir_values(self.current_expert_idx))
        values.extend(extract_mlir_values(self.expert_tile_start))
        values.extend(extract_mlir_values(self.expert_tile_end))
        # Dynamic scheduling state
        if self.params.use_dynamic_sched:
            values.extend(extract_mlir_values(self._dynamic_state))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEPersistentTileScheduler":
        idx = 0

        # Params (3 values: expert_cnt, intermediate, hidden)
        new_params = new_from_mlir_values(self.params, values[idx : idx + 3])
        idx += 3

        # Runtime tensor for scheduling (variable size)
        offs_len = len(extract_mlir_values(self.offs))
        new_offs = new_from_mlir_values(self.offs, values[idx : idx + offs_len])
        idx += offs_len

        # Scheduling state
        new_num_persistent_clusters = new_from_mlir_values(self.num_persistent_clusters, [values[idx]])
        idx += 1
        new_current_work_linear_idx = new_from_mlir_values(self._current_work_linear_idx, [values[idx]])
        idx += 1

        # cta_id_in_cluster (3 values for Coord)
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[idx : idx + 3])
        idx += 3

        # Expert tracking state
        new_current_expert_idx = new_from_mlir_values(self.current_expert_idx, [values[idx]])
        idx += 1
        new_expert_tile_start = new_from_mlir_values(self.expert_tile_start, [values[idx]])
        idx += 1
        new_expert_tile_end = new_from_mlir_values(self.expert_tile_end, [values[idx]])
        idx += 1

        # Dynamic scheduling state
        new_dynamic_state = None
        if self.params.use_dynamic_sched:
            ds_len = len(extract_mlir_values(self._dynamic_state))
            new_dynamic_state = new_from_mlir_values(self._dynamic_state, values[idx : idx + ds_len])
            idx += ds_len

        return MoEPersistentTileScheduler(
            params=new_params,
            offs=new_offs,
            num_persistent_clusters=new_num_persistent_clusters,
            current_work_linear_idx=new_current_work_linear_idx,
            cta_id_in_cluster=new_cta_id_in_cluster,
            current_expert_idx=new_current_expert_idx,
            expert_tile_start=new_expert_tile_start,
            expert_tile_end=new_expert_tile_end,
            dynamic_state=new_dynamic_state,
            sched_storage=self._sched_storage,
            cluster_pipeline=self.cluster_pipeline,
        )

    # =========================================================================
    # Factory method
    # =========================================================================

    @staticmethod
    @dsl_user_op
    def create(
        params: MoESchedulerParams,
        offs: cute.Tensor,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        # Dynamic scheduling resources (all None for static mode)
        counter_ptr: Optional[cute.Pointer] = None,
        sched_storage=None,
        *,
        loc=None,
        ip=None,
    ) -> "MoEPersistentTileScheduler":
        """
        Create a MoE persistent tile scheduler.

        :param params: Scheduler parameters (from host)
        :param offs: Cumsum tensor of token counts per expert, shape (experts,)
        :param block_idx: CUDA block index
        :param grid_dim: CUDA grid dimensions
        :param counter_ptr: Pointer to gmem atomic counter (dynamic mode only)
        :param sched_storage: SchedulerStorage instance from SharedStorage (dynamic mode only)
        """
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc, ip=ip)

        bidx, bidy, bidz = block_idx

        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )

        # Initialize expert tracking to "before expert 0"
        current_expert_idx = Int32(0)
        expert_tile_start = Int32(0)
        expert_tile_end = Int32(0)

        # Dynamic scheduling setup
        dynamic_state = None
        if const_expr(params.use_dynamic_sched):
            current_work_linear_idx = Int32(-1)
            is_leader_cta = (cta_id_in_cluster[0] + cta_id_in_cluster[1] + cta_id_in_cluster[2]) == Int32(0)
            if const_expr(params.scenario == "2Dx2D"):
                producer_state_type = PipelineUserType.ProducerConsumer
            else:
                producer_state_type = PipelineUserType.Producer
            broadcast_ptr = sched_storage.cluster_broadcast.data_ptr()
            dynamic_state = _DynamicSchedState(
                counter_ptr=counter_ptr if counter_ptr is not None else broadcast_ptr,
                broadcast_ptr=broadcast_ptr,
                is_leader_cta=is_leader_cta,
                producer_state=make_pipeline_state(producer_state_type, 1, loc=loc, ip=ip),
                consumer_state=make_pipeline_state(PipelineUserType.Consumer, 1, loc=loc, ip=ip),
            )
        else:
            current_work_linear_idx = Int32(bidz)

        # cluster_pipeline is not created here; caller must invoke
        # internal_init() before warp specialization.
        return MoEPersistentTileScheduler(
            params=params,
            offs=offs,
            num_persistent_clusters=num_persistent_clusters,
            current_work_linear_idx=current_work_linear_idx,
            cta_id_in_cluster=cta_id_in_cluster,
            current_expert_idx=current_expert_idx,
            expert_tile_start=expert_tile_start,
            expert_tile_end=expert_tile_end,
            dynamic_state=dynamic_state,
            sched_storage=sched_storage,
            cluster_pipeline=None,
        )

    # =========================================================================
    # Tile iteration methods
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        """Get the initial work tile info."""
        if const_expr(self.scenario == "2Dx2D"):
            return self._initial_work_from_block_idx(loc=loc, ip=ip)
        if const_expr(self.use_dynamic_sched):
            if const_expr(not hasattr(self, "cluster_pipeline")):
                raise RuntimeError("Make sure sched.internal_init() is called at the barrier init place before used.")
            self._current_work_linear_idx = self._fetch_next_cluster_idx(loc=loc, ip=ip)
            work_tile_info = self._get_work_tile_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)
            self._release_cluster_idx(loc=loc, ip=ip)
            return work_tile_info
        return self._get_work_tile_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        """Advance to the next work tile and return its info."""
        if const_expr(self.scenario == "2Dx2D"):
            return self._fetch_next_clc_work(loc=loc, ip=ip)
        if const_expr(self.use_dynamic_sched):
            self._current_work_linear_idx = self._fetch_next_cluster_idx(loc=loc, ip=ip)
            work_tile_info = self._get_work_tile_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)
            self._release_cluster_idx(loc=loc, ip=ip)
            return work_tile_info
        else:
            self._current_work_linear_idx += self.num_persistent_clusters
        return self._get_work_tile_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def _initial_work_from_block_idx(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        bidx, bidy, bidz = cute.arch.block_idx(loc=loc, ip=ip)
        expert_idx = bidz
        k_tile_cnt = self._compute_k_tile_cnt(expert_idx, loc=loc, ip=ip)
        return MoEWorkTileInfo(
            expert_idx=expert_idx,
            tile_m_idx=bidx,
            tile_n_idx=bidy,
            k_tile_cnt=k_tile_cnt,
        )

    @dsl_user_op
    @cute.jit
    def _fetch_next_clc_work(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        ds = self._dynamic_state

        if ds.is_leader_cta:
            self.cluster_pipeline.producer_acquire(ds.producer_state)
            mbar_ptr = self.cluster_pipeline.producer_get_barrier(ds.producer_state)
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(mbar_ptr, ds.broadcast_ptr, loc=loc, ip=ip)
        ds.producer_state.advance()

        self.cluster_pipeline.consumer_wait(ds.consumer_state)
        m_idx, n_idx, l_idx, is_valid = cute.arch.clc_response(ds.broadcast_ptr, loc=loc, ip=ip)
        cute.arch.fence_acq_rel_cta()
        self.cluster_pipeline.consumer_release(ds.consumer_state)
        ds.consumer_state.advance()

        work_tile_info = MoEWorkTileInfo(
            expert_idx=Int32(-1),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )
        if is_valid:
            expert_idx = l_idx
            tile_m_idx = m_idx + self.cta_id_in_cluster[0]
            tile_n_idx = n_idx + self.cta_id_in_cluster[1]
            k_tile_cnt = self._compute_k_tile_cnt(expert_idx, loc=loc, ip=ip)
            work_tile_info = MoEWorkTileInfo(
                expert_idx=expert_idx,
                tile_m_idx=tile_m_idx,
                tile_n_idx=tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
        return work_tile_info

    @dsl_user_op
    @cute.jit
    def _fetch_next_cluster_idx(self, *, loc=None, ip=None) -> Int32:
        """
        Fetch the next cluster_linear_idx using atomic counter + cluster pipeline broadcast.

        Leader CTA: producer_acquire -> atomicAdd -> store to smem -> fence -> custom producer_commit
        All CTAs:   consumer_wait -> read from leader's smem (DSMEM)
        """
        assert self.use_dynamic_sched
        ds = self._dynamic_state
        broadcast_tensor = cute.make_tensor(ds.broadcast_ptr, cute.make_layout((1,)))
        cluster_idx = Int32(0)

        # --- Producer: leader acquires and broadcasts ---
        if ds.is_leader_cta:
            self.cluster_pipeline.producer_acquire(ds.producer_state)
            full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(ds.producer_state.index, loc=loc, ip=ip)
            tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
            lane_idx = tidx % Int32(32)
            cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]

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

        # --- Consumer: all CTAs read the broadcast value ---
        self.cluster_pipeline.consumer_wait(ds.consumer_state)
        cluster_idx = broadcast_tensor[0]

        return cluster_idx

    @dsl_user_op
    @cute.jit
    def _release_cluster_idx(self, *, loc=None, ip=None) -> None:
        """Release the cluster broadcast slot after the fetched value has been consumed."""
        ds = self._dynamic_state
        self._cluster_consumer_release(loc=loc, ip=ip)
        ds.consumer_state.advance()

    @dsl_user_op
    @cute.jit
    def _cluster_producer_commit(self, *, loc=None, ip=None) -> None:
        """Fan out one arrive per CTA to the full barrier."""
        ds = self._dynamic_state
        tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
        lane_idx = tidx % Int32(32)
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
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
        """Gather a full warp of arrives per CTA to leader's empty barrier."""
        ds = self._dynamic_state
        self.cluster_pipeline.sync_object_empty.arrive(ds.consumer_state.index, Int32(0))

    @dsl_user_op
    @cute.jit
    def _get_work_tile_for_linear_idx(self, cluster_linear_idx: Int32, *, loc=None, ip=None) -> MoEWorkTileInfo:
        """
        Convert a linear cluster index to MoEWorkTileInfo.

        Uses cached expert tracking state for O(1) fast path when staying
        within the same expert. Advances expert state when needed.

        Returns an invalid tile (expert_idx = -1) if cluster_linear_idx is out of range.
        """
        # Ensure expert tracking is initialized and up-to-date
        self._advance_expert_to_contain(cluster_linear_idx, loc=loc, ip=ip)

        # Check if valid (still within expert range after advancing)
        is_valid = self.current_expert_idx < self.expert_cnt

        work_tile_info = MoEWorkTileInfo(
            expert_idx=Int32(-1),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )

        if is_valid:
            # Compute local cluster tile indices within current expert
            local_idx = cluster_linear_idx - self.expert_tile_start
            cluster_tile_m_idx, cluster_tile_n_idx = self._decompose_local_idx(local_idx, self.current_expert_idx, loc=loc, ip=ip)

            # Convert cluster tile indices to CTA tile indices
            # cta_tile_idx = cluster_tile_idx * cluster_shape + cta_id_in_cluster
            cta_tile_m_idx = cluster_tile_m_idx * self.cluster_shape_mn[0] + self.cta_id_in_cluster[0]
            cta_tile_n_idx = cluster_tile_n_idx * self.cluster_shape_mn[1] + self.cta_id_in_cluster[1]

            # Compute k_tile_cnt
            k_tile_cnt = self._compute_k_tile_cnt(self.current_expert_idx, loc=loc, ip=ip)

            work_tile_info = MoEWorkTileInfo(
                expert_idx=self.current_expert_idx,
                tile_m_idx=cta_tile_m_idx,
                tile_n_idx=cta_tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
            # with cute.arch.elect_one():
            #     cute.printf("[%d, %d, %d, %d]", self.current_expert_idx, cta_tile_m_idx, cta_tile_n_idx, k_tile_cnt)
        return work_tile_info

    @dsl_user_op
    @cute.jit
    def _advance_expert_to_contain(
        self,
        cluster_linear_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        """
        Advance expert tracking state until current expert contains cluster_linear_idx,
        or we run out of experts.

        Fast path: If already in correct expert, no work needed.
        """
        # Initialize expert_tile_end if this is the first call (expert_tile_end == 0)
        if self.expert_tile_end == Int32(0):
            tiles_for_expert_0 = self._compute_tiles_for_expert(Int32(0), loc=loc, ip=ip)
            self.expert_tile_end = tiles_for_expert_0

        # Advance until cluster_linear_idx < expert_tile_end or no more experts
        while cluster_linear_idx >= self.expert_tile_end and self.current_expert_idx < self.expert_cnt:
            self.current_expert_idx = self.current_expert_idx + 1
            self.expert_tile_start = self.expert_tile_end

            if self.current_expert_idx < self.expert_cnt:
                tiles_for_expert = self._compute_tiles_for_expert(self.current_expert_idx, loc=loc, ip=ip)
                self.expert_tile_end = self.expert_tile_end + tiles_for_expert

    @dsl_user_op
    @cute.jit
    def _compute_tiles_for_expert(
        self,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Int32:
        """Compute total cluster tiles for a given expert."""
        if const_expr(self.scenario == "2Dx2D"):
            # Fixed M=hidden, N=intermediate
            cluster_tile_m_cnt = (self.hidden + self.cluster_tile_m - 1) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n - 1) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt
        else:  # 2Dx3D
            # Variable M (tokens), fixed N
            tokens_i = self.offs[expert_idx]
            if expert_idx > 0:
                tokens_i = tokens_i - self.offs[expert_idx - 1]
            cluster_tile_m_cnt = (tokens_i + self.cluster_tile_m - 1) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n - 1) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt

    @dsl_user_op
    @cute.jit
    def _decompose_local_idx(
        self,
        local_idx: Int32,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32]:
        """
        Decompose local cluster tile index within expert to (cluster_tile_m_idx, cluster_tile_n_idx).

        Uses "short side first" strategy: the shorter dimension changes faster.
        This maximizes overlap between adjacent clusters for better L2 cache utilization.

        For example, if m_cnt=2, n_cnt=8:
        - N is longer, so M changes faster: local_idx = n_idx * m_cnt + m_idx
        - Linearization order: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2), ...
        """
        # Get tile counts for M and N
        cluster_tile_m_cnt, cluster_tile_n_cnt = self._get_cluster_tile_counts(expert_idx, loc=loc, ip=ip)
        cluster_tile_m_idx = -1
        cluster_tile_n_idx = -1

        # Short side first: shorter dimension changes faster
        # If m_cnt <= n_cnt: m is shorter, m changes faster
        #   local_idx = n_idx * m_cnt + m_idx
        # If n_cnt < m_cnt: n is shorter, n changes faster
        #   local_idx = m_idx * n_cnt + n_idx
        if cluster_tile_m_cnt <= cluster_tile_n_cnt:
            # M is shorter or equal, M changes faster
            cluster_tile_m_idx = local_idx % cluster_tile_m_cnt
            cluster_tile_n_idx = local_idx // cluster_tile_m_cnt
        else:
            # N is shorter, N changes faster
            cluster_tile_n_idx = local_idx % cluster_tile_n_cnt
            cluster_tile_m_idx = local_idx // cluster_tile_n_cnt

        return (cluster_tile_m_idx, cluster_tile_n_idx)

    @dsl_user_op
    @cute.jit
    def _get_cluster_tile_counts(
        self,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32]:
        """Get (cluster_tile_m_cnt, cluster_tile_n_cnt) for a given expert."""
        if const_expr(self.scenario == "2Dx2D"):
            # Fixed M=hidden, N=intermediate
            cluster_tile_m_cnt = (self.hidden + self.cluster_tile_m - 1) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n - 1) // self.cluster_tile_n
        else:  # 2Dx3D
            # Variable M (tokens), fixed N
            tokens_i = self.offs[expert_idx]
            if expert_idx > 0:
                tokens_i = tokens_i - self.offs[expert_idx - 1]
            cluster_tile_m_cnt = (tokens_i + self.cluster_tile_m - 1) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n - 1) // self.cluster_tile_n
        return (cluster_tile_m_cnt, cluster_tile_n_cnt)

    @dsl_user_op
    @cute.jit
    def _compute_k_tile_cnt(
        self,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Int32:
        """
        Compute the number of K tiles for this expert.

        2Dx3D: K = hidden (fixed) -> k_tile_cnt = ceil(hidden / cta_tile_k)
        2Dx2D: K = tokens_i (variable) -> k_tile_cnt = ceil(tokens_i / cta_tile_k)
        """
        if const_expr(self.scenario == "2Dx3D"):
            # K is hidden (fixed)
            return (self.hidden + self.cta_tile_k - 1) // self.cta_tile_k
        else:  # 2Dx2D
            # K is tokens_i (variable per expert)
            tokens_i = self.offs[expert_idx]
            if expert_idx > cutlass.Int32(0):
                tokens_i = tokens_i - self.offs[expert_idx - 1]
            return (tokens_i + self.cta_tile_k - 1) // self.cta_tile_k
