# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent MoK-style communication and expert compute stages."""

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas.mosaic_gpu import primitives as plgpu_primitives
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import vector
from jaxtyping import Array, Float, Int

from levanter.grug._moe.mok_dispatch import _thread_index_mgpu
from levanter.grug._moe.mok_expert_gemm import GroupInfo, TuningConfig, do_matmul

_DISPATCH_ROWS = 128
_DISPATCH_COLS = 512
_SWIGLU_ROWS = 128
_SWIGLU_COLS = 128
_SWIGLU_PIPE_DEPTH = 3
_SWIGLU_BWD_PIPE_DEPTH = 2
_COMBINE_ROWS = 16
_COMBINE_COLS = 1024
_COMBINE_PIPE_DEPTH = 7
_CLUSTER_SIZE = 2
_MACROBATCH_SIZE = 131072
_TMA_FLOAT_ALIGNMENT = 4
_SWIGLU_THREAD_LAYOUT = plgpu.Layout.TILED(
    plgpu.Tiling(((128, 4), (32, 4))),
    warp_dims=(-4,),
    lane_dims=(-2,),
    vector_dim=-1,
)


def _barrier_all(marker: Array, *, axis_name: str) -> Array:
    def kernel(marker_ref, marker_out):
        barrier = pl.get_global(plgpu.SemaphoreType.REGULAR((1,)))
        plgpu.semaphore_signal_multicast(barrier.at[0], collective_axes=axis_name)
        pl.semaphore_wait(barrier.at[0], value=lax.axis_size(axis_name), decrement=False)
        marker_out[()] = marker_ref[()]

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((), marker.dtype),
        grid=(1,),
        grid_names=("block",),
        num_threads=1,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        kernel_name="mok_barrier_all",
    )(marker)


def backward_epilogue(
    d_x_shared: Float[Array, "T H"],
    d_x_routed: Float[Array, "S H"],
    *,
    topk: int,
    axis_name: str,
) -> Float[Array, "T H"]:
    """Sums shared and routed input gradients using MoK's backward epilogue."""
    num_tokens, hidden = d_x_shared.shape
    if d_x_routed.shape != (num_tokens * topk, hidden):
        raise ValueError(f"{d_x_routed.shape=} must equal {(num_tokens * topk, hidden)}")
    block_cols = min(_COMBINE_COLS, hidden)
    if hidden % block_cols:
        raise ValueError(f"{hidden=} must be divisible by {block_cols=}")
    half_cols = block_cols // 2
    vector_size = min(4, half_cols // 128)
    if vector_size < 2 or half_cols % (128 * vector_size):
        raise ValueError(f"{block_cols=} cannot be split across MoK's 256 epilogue threads")
    tma_chunks = block_cols // 256
    input_bytes = (topk + 1) * block_cols * d_x_shared.dtype.itemsize

    marker = _barrier_all(d_x_shared[0, 0], axis_name=axis_name)

    def kernel(d_x_shared_ref, d_x_routed_ref, _marker_ref, d_x_out, token_vecs, inputs_arrived):
        token = lax.axis_index("token")
        col_block = lax.axis_index("col")
        wg = lax.axis_index("wg")

        @pl.when(wg == 0)
        def _load():
            for chunk in range(tma_chunks):
                chunk_slice = pl.ds(col_block * block_cols + chunk * 256, 256)
                plgpu.copy_gmem_to_smem(
                    d_x_shared_ref.at[token, chunk_slice],
                    token_vecs.at[0, pl.ds(chunk * 256, 256)],
                    inputs_arrived,
                    arrive=chunk == 0,
                    expect_bytes=input_bytes if chunk == 0 else None,
                )
                for route in range(topk):
                    plgpu.copy_gmem_to_smem(
                        d_x_routed_ref.at[token * topk + route, chunk_slice],
                        token_vecs.at[route + 1, pl.ds(chunk * 256, 256)],
                        inputs_arrived,
                        arrive=False,
                    )

        plgpu.barrier_wait(inputs_arrived)
        wg_cols = pl.ds(wg * half_cols, half_cols)
        layout = plgpu.Layout.WG_STRIDED((half_cols,), vec_size=vector_size)
        accumulator = plgpu.load(token_vecs.at[0, wg_cols], layout=layout).astype(jnp.float32)
        for route in range(topk):
            accumulator += plgpu.load(token_vecs.at[route + 1, wg_cols], layout=layout).astype(jnp.float32)
        token_vecs.at[0, wg_cols][...] = accumulator.astype(d_x_shared.dtype)
        plgpu.commit_smem()
        plgpu_primitives.cta_barrier()

        @pl.when(wg == 0)
        def _store():
            for chunk in range(tma_chunks):
                chunk_slice = pl.ds(col_block * block_cols + chunk * 256, 256)
                plgpu.copy_smem_to_gmem(
                    token_vecs.at[0, pl.ds(chunk * 256, 256)],
                    d_x_out.at[token, chunk_slice],
                    commit_group=chunk == tma_chunks - 1,
                )

        plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(d_x_shared.shape, d_x_shared.dtype),
        scratch_types=(
            plgpu.SMEM((topk + 1, block_cols), d_x_shared.dtype),
            plgpu.Barrier(num_arrivals=1),
        ),
        grid=(num_tokens, hidden // block_cols),
        grid_names=("token", "col"),
        num_threads=2,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
        kernel_name="mok_backward_epilogue",
    )(d_x_shared, d_x_routed, marker)


@plgpu.inline_mgpu(
    arg_types=(_SWIGLU_THREAD_LAYOUT.reduce(1),),
    return_type=plgpu.ShapeDtypeStruct((), jnp.float32, layout=plgpu.Layout.WG_SPLAT),
)
def _swiglu_thread_scalar_mgpu(_, value):
    scalar = vector.extract(
        value.registers.item(),
        dynamic_position=[],
        static_position=ir.DenseI64ArrayAttr.get([0]),
    )
    return mgpu.FragmentedArray.splat(scalar, ())


@plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.Layout.WG_SPLAT))
def _atomic_add_scalar_mgpu(_, ref, value):
    register = vector.broadcast(ir.VectorType.get((1,), value.mlir_dtype), value.registers.item())
    value._store_register_atomic(mgpu.utils.memref_ptr(ref), register, "add", is_smem=False)


def _dispatch_task(
    x_ref,
    peer_rank_ref,
    peer_token_ref,
    out_ref,
    staging_ref,
    owner_ref,
    token_ref,
    schedule_barrier,
    gather_barrier,
    dispatch_ready,
    task,
    phase,
    wg,
    *,
    topk: int,
    col_blocks: int,
    minibatch_size: int,
    run_gather: bool,
    run_store: bool,
    source_is_slot: bool = False,
    row_scales_ref=None,
):
    row_block = task // col_blocks
    col_block = lax.rem(task, col_blocks)
    base = row_block * _DISPATCH_ROWS

    @pl.when(wg == 0)
    def _load_wg():
        plgpu.copy_gmem_to_smem(peer_rank_ref.at[pl.ds(base, _DISPATCH_ROWS)], owner_ref, schedule_barrier)
        plgpu.copy_gmem_to_smem(peer_token_ref.at[pl.ds(base, _DISPATCH_ROWS)], token_ref, schedule_barrier)
        plgpu.barrier_wait(schedule_barrier)

    if run_gather:
        row = lax.rem(_thread_index_mgpu(), _DISPATCH_ROWS)
        worker = wg == 0
        owner = lax.select(worker, owner_ref[row], jnp.int32(0))
        token = lax.select(worker, token_ref[row], jnp.int32(0))
        valid = worker & (owner >= 0)
        source = plgpu.remote_ref(x_ref, jnp.maximum(owner, 0))
        source_row = jnp.maximum(token, 0) if source_is_slot else jnp.maximum(token, 0) // topk
        plgpu.copy_gmem_to_smem(
            source.at[
                pl.ds(source_row, 1),
                pl.ds(col_block * _DISPATCH_COLS, _DISPATCH_COLS),
            ],
            staging_ref,
            gather_barrier,
            predicate=valid,
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
            thread_parallel=True,
        )
        plgpu_primitives.barrier_wait_parity(gather_barrier, phase)

    if row_scales_ref is not None:

        @pl.when(wg == 0)
        def _scale_rows():
            values = plgpu.load(staging_ref, layout=plgpu.Layout.WGMMA, optimized=False).astype(jnp.float32)
            scales = plgpu.load(
                row_scales_ref.at[pl.ds(base, _DISPATCH_ROWS), 0],
                layout=plgpu.Layout.WGMMA.reduce(1),
                optimized=False,
            )
            scales = lax.broadcast_in_dim(scales, values.shape, (0,))
            scales = plgpu.layout_cast(scales, plgpu.Layout.WGMMA)
            staging_ref[...] = (values * scales).astype(jnp.bfloat16)
            plgpu.commit_smem()

        plgpu_primitives.cta_barrier()

    @pl.when(wg == 0)
    def _store_wg():
        if run_store:
            row = _thread_index_mgpu()
            valid = owner_ref[row] >= 0
            plgpu.copy_smem_to_gmem(
                staging_ref,
                out_ref.at[pl.ds(base + row, 1), pl.ds(col_block * _DISPATCH_COLS, _DISPATCH_COLS)],
                predicate=valid,
                thread_parallel=True,
            )
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        pl.semaphore_signal(dispatch_ready.at[base // minibatch_size])

    plgpu_primitives.cta_barrier()


def _preload_router_weights_task(
    router_weights_ref,
    peer_rank_ref,
    peer_token_ref,
    router_weights_routed_out,
    owner_ref,
    token_ref,
    row_scales_ref,
    schedule_barrier,
    gather_barrier,
    task,
    phase,
    wg,
):
    base = task * _DISPATCH_ROWS

    @pl.when(wg == 0)
    def _load_wg():
        plgpu.copy_gmem_to_smem(peer_rank_ref.at[pl.ds(base, _DISPATCH_ROWS)], owner_ref, schedule_barrier)
        plgpu.copy_gmem_to_smem(peer_token_ref.at[pl.ds(base, _DISPATCH_ROWS)], token_ref, schedule_barrier)
        plgpu.barrier_wait(schedule_barrier)

    row = lax.rem(_thread_index_mgpu(), _DISPATCH_ROWS)
    worker = wg == 0
    owner = lax.select(worker, owner_ref[row], jnp.int32(0))
    token = lax.select(worker, token_ref[row], jnp.int32(0))
    valid = worker & (owner >= 0)
    source = plgpu.remote_ref(router_weights_ref, jnp.maximum(owner, 0))
    plgpu.copy_gmem_to_smem(
        source.at[pl.ds(jnp.maximum(token, 0), 1), :],
        row_scales_ref,
        gather_barrier,
        predicate=valid,
        oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        thread_parallel=True,
    )
    plgpu_primitives.barrier_wait_parity(gather_barrier, phase)

    @pl.when(wg == 0)
    def _store_wg():
        plgpu.copy_smem_to_gmem(
            row_scales_ref,
            router_weights_routed_out.at[pl.ds(base, _DISPATCH_ROWS), pl.ds(0, 4)],
        )
        plgpu.wait_smem_to_gmem(0)

    plgpu_primitives.cta_barrier()


def dispatch_gate_up(
    x: Float[Array, "T H"],
    w_gate: Float[Array, "E H I"],
    w_up: Float[Array, "E H I"],
    peer_rank: Int[Array, "C"],
    peer_token_idx: Int[Array, "C"],
    num_routed_tokens: Int[Array, ""],
    tokens_per_expert: Int[Array, "E"],
    *,
    axis_name: str,
    topk: int,
    num_comm_sms: int = 24,
    minibatch_size: int = 4096,
    gemm_config: TuningConfig,
    w_down: Float[Array, "E I H"] | None = None,
    shared: tuple[Float[Array, "H I"], Float[Array, "H I"], Float[Array, "I H"]] | None = None,
    run_compute: bool = True,
    run_up: bool = True,
    run_dispatch_gather: bool = True,
    run_dispatch_store: bool = True,
) -> tuple[
    Float[Array, "C H"],
    Float[Array, "C I"],
    Float[Array, "C I"],
    Float[Array, "C I"],
    Float[Array, "C H"],
    Float[Array, "S H"],
    Float[Array, "T I"],
    Float[Array, "T I"],
    Float[Array, "T I"],
    Float[Array, "T H"],
]:
    """Overlaps MoK dispatch with the routed gate and up GEMMs."""
    capacity = peer_rank.shape[0]
    hidden = x.shape[1]
    intermediate = w_gate.shape[2]
    num_slots = x.shape[0] * topk
    run_down = w_down is not None
    run_shared = shared is not None
    down_weights = w_down if w_down is not None else w_gate
    shared_weights = shared if shared is not None else (w_gate[0], w_up[0], down_weights[0])
    if hidden % _DISPATCH_COLS:
        raise ValueError(f"{hidden=} must be divisible by {_DISPATCH_COLS}")
    if capacity % _DISPATCH_ROWS:
        raise ValueError(f"{capacity=} must be divisible by {_DISPATCH_ROWS}")
    if run_compute and not gemm_config.collective:
        raise ValueError("persistent MoK compute uses cluster-2 GEMMs")
    if run_down and not (run_compute and run_up):
        raise ValueError("routed down requires gate and up compute")
    if run_shared and not run_down:
        raise ValueError("shared expert requires the full routed compute chain")
    if minibatch_size <= 0 or minibatch_size % (_DISPATCH_ROWS * _CLUSTER_SIZE):
        raise ValueError(f"{minibatch_size=} must be positive and divisible by 256")

    tile_m = gemm_config.tile_m
    tile_n = gemm_config.tile_n
    tile_k = gemm_config.tile_k
    max_concurrent_steps = gemm_config.max_concurrent_steps
    epilogue_tile_n = gemm_config.epilogue_tile_n
    if tile_m != _DISPATCH_ROWS:
        raise ValueError(f"{tile_m=} must equal MoK dispatch rows {_DISPATCH_ROWS}")
    if hidden % tile_k or intermediate % (tile_n * _CLUSTER_SIZE):
        raise ValueError(f"GEMM tiles do not divide H={hidden}, I={intermediate}")

    num_sms = jax.local_devices()[0].core_count
    if num_sms % _CLUSTER_SIZE:
        raise ValueError(f"{num_sms=} must be divisible by MoK cluster size {_CLUSTER_SIZE}")
    max_comm_sms = num_sms - _CLUSTER_SIZE if run_compute else num_sms
    if not 0 < num_comm_sms <= max_comm_sms:
        raise ValueError(f"{num_comm_sms=} must be between 1 and {max_comm_sms}")
    if num_comm_sms % _CLUSTER_SIZE:
        raise ValueError(f"{num_comm_sms=} must be divisible by cluster size {_CLUSTER_SIZE}")
    num_clusters = num_sms // _CLUSTER_SIZE
    num_comm_clusters = num_comm_sms // _CLUSTER_SIZE
    col_blocks = hidden // _DISPATCH_COLS
    logical_tile_m = tile_m * _CLUSTER_SIZE
    logical_tile_n = tile_n * _CLUSTER_SIZE
    n_iters = intermediate // logical_tile_n
    minibatch_row_blocks = minibatch_size // logical_tile_m
    shared_row_blocks = x.shape[0] // logical_tile_m
    if run_shared and x.shape[0] % logical_tile_m:
        raise ValueError(f"shared token rows {x.shape[0]} must be divisible by {logical_tile_m}")
    max_minibatches = (capacity + minibatch_size - 1) // minibatch_size
    gate_up_tasks = minibatch_row_blocks * n_iters
    swiglu_col_blocks = intermediate // _SWIGLU_COLS
    swiglu_tiles = (minibatch_size // _SWIGLU_ROWS) * swiglu_col_blocks
    swiglu_tasks = (swiglu_tiles + _CLUSTER_SIZE * _SWIGLU_PIPE_DEPTH - 1) // (_CLUSTER_SIZE * _SWIGLU_PIPE_DEPTH)
    down_n_iters = hidden // logical_tile_n
    down_tasks = minibatch_row_blocks * down_n_iters
    shared_gate_up_tasks = shared_row_blocks * n_iters if run_shared else 0
    shared_swiglu_tiles = (x.shape[0] // _SWIGLU_ROWS) * swiglu_col_blocks
    shared_swiglu_tasks = (
        (shared_swiglu_tiles + _CLUSTER_SIZE * _SWIGLU_PIPE_DEPTH - 1) // (_CLUSTER_SIZE * _SWIGLU_PIPE_DEPTH)
        if run_shared
        else 0
    )
    shared_down_tasks = shared_row_blocks * down_n_iters if run_shared else 0
    shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks
    combine_block_cols = min(_COMBINE_COLS, hidden)
    if hidden % combine_block_cols:
        raise ValueError(f"{hidden=} must be divisible by {combine_block_cols=}")
    combine_col_blocks = hidden // combine_block_cols
    combine_pipeline_depth = min(_COMBINE_PIPE_DEPTH, combine_col_blocks)
    minibatch_tasks = (2 if run_up else 1) * gate_up_tasks
    if run_down:
        minibatch_tasks += swiglu_tasks + down_tasks
    logical_compute_clusters = shared_tasks + max_minibatches * minibatch_tasks
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(x.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
    comm_storage = (
        plgpu.SMEM((_DISPATCH_ROWS, _DISPATCH_COLS), x.dtype),
        plgpu.SMEM((_DISPATCH_ROWS,), jnp.int32),
        plgpu.SMEM((_DISPATCH_ROWS,), jnp.int32),
    )
    storage_groups = [comm_storage]
    if run_compute:
        storage_groups.append(
            (
                plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), x.dtype, transforms=transforms),
                plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), x.dtype, transforms=transforms),
                plgpu.SMEM((tile_m, epilogue_tile_n), x.dtype),
            )
        )
    if run_down:
        storage_groups.append(
            (
                plgpu.SMEM(
                    (_SWIGLU_PIPE_DEPTH, _SWIGLU_ROWS, _SWIGLU_COLS),
                    x.dtype,
                    transforms=transforms,
                ),
                plgpu.SMEM(
                    (_SWIGLU_PIPE_DEPTH, _SWIGLU_ROWS, _SWIGLU_COLS),
                    x.dtype,
                    transforms=transforms,
                ),
                plgpu.SMEM((_SWIGLU_ROWS, _SWIGLU_COLS), x.dtype, transforms=transforms),
            )
        )
        storage_groups.append(
            (
                plgpu.SMEM(
                    (combine_pipeline_depth, _COMBINE_ROWS, combine_block_cols),
                    x.dtype,
                ),
            )
        )
    scratch_types = {
        "stage_storage": plgpu.RefUnion(*storage_groups),
        "schedule_barrier": plgpu.Barrier(num_arrivals=2),
        "gather_barrier": plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
    }
    compute_scratch_names = (
        "a_tma_barrier",
        "b_tma_barrier",
        "store_done_barrier",
        "mma_done_barrier",
        "consumed_barrier",
        "acc_tmem",
    )
    if run_compute:
        scratch_types.update(
            a_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            b_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            store_done_barrier=plgpu.ClusterBarrier(
                collective_axes=("cta",),
                num_arrivals=1,
                num_barriers=1,
                orders_tensor_core=True,
                leader_tracked=True,
            ),
            mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
            consumed_barrier=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=max_concurrent_steps,
                orders_tensor_core=True,
            ),
        )
    if run_down:
        scratch_types["swiglu_barrier"] = plgpu.Barrier(
            num_arrivals=2,
            num_barriers=_SWIGLU_PIPE_DEPTH,
        )
        scratch_types["stage_transition_barrier"] = plgpu.ClusterBarrier(
            collective_axes=("cta",),
            num_arrivals=2,
        )
        scratch_types["combine_barrier"] = plgpu.Barrier(
            num_arrivals=1,
            num_barriers=combine_pipeline_depth,
            orders_tensor_core=True,
        )
    scratch_types["acc_tmem"] = plgpu.TMEM((tile_m, tile_n * 2), jnp.float32, collective=True)

    def kernel(
        x_ref,
        gate_ref,
        up_ref,
        down_ref,
        peer_rank_ref,
        peer_token_ref,
        num_routed_ref,
        groups_ref,
        shared_gate_ref,
        shared_up_ref,
        shared_down_ref,
        x_out,
        gate_out,
        up_out,
        hidden_out,
        y_out,
        combine_out,
        shared_gate_out,
        shared_up_out,
        shared_hidden_out,
        shared_y_out,
        **scratch,
    ):
        sm = lax.axis_index("sm") * _CLUSTER_SIZE + lax.axis_index("cta")
        wg = lax.axis_index("wg")
        plgpu.set_max_registers(248, action="increase")
        rendezvous = pl.get_global(plgpu.SemaphoreType.REGULAR((num_comm_sms,)))
        combine_done = pl.get_global(plgpu.SemaphoreType.REGULAR((num_comm_sms,)))
        dispatch_ready = pl.get_global(plgpu.SemaphoreType.REGULAR((max_minibatches,)))
        gate_up_ready = pl.get_global(
            plgpu.SemaphoreType.REGULAR((shared_gate_up_tasks + (capacity // logical_tile_m) * n_iters,))
        )
        hidden_ready = pl.get_global(plgpu.SemaphoreType.REGULAR((shared_row_blocks + capacity // logical_tile_m,)))
        y_ready = pl.get_global(plgpu.SemaphoreType.REGULAR((max_minibatches,)))
        staging_ref, owner_ref, token_ref = scratch["stage_storage"][0]
        if run_compute:
            a_smem, b_smem, acc_smem = scratch["stage_storage"][1]
            compute_scratch = {name: scratch[name] for name in compute_scratch_names}
            compute_scratch.update(a_smem=a_smem, b_smem=b_smem, acc_smem=acc_smem)
        else:
            compute_scratch = None
        if run_down:
            gate_smem, up_smem, hidden_smem = scratch["stage_storage"][2]
            (combine_smem,) = scratch["stage_storage"][3]

        def combine_task(task, phase):
            active_row_blocks = lax.div(num_routed_ref[()] + _COMBINE_ROWS - 1, _COMBINE_ROWS)
            num_tiles = active_row_blocks * combine_col_blocks
            first_tile = task * combine_pipeline_depth
            last_tile = jnp.minimum(first_tile + combine_pipeline_depth, num_tiles) - 1
            first_minibatch = (first_tile // combine_col_blocks * _COMBINE_ROWS) // minibatch_size
            last_minibatch = (last_tile // combine_col_blocks * _COMBINE_ROWS) // minibatch_size

            def required_down_arrivals(global_minibatch_idx):
                first_row = global_minibatch_idx * minibatch_size
                rows = jnp.maximum(
                    jnp.int32(0),
                    jnp.minimum(minibatch_size, num_routed_ref[()] - first_row),
                )
                return lax.div(rows + logical_tile_m - 1, logical_tile_m) * down_n_iters * _CLUSTER_SIZE

            @pl.when(wg == 0)
            def _wait_for_down():
                pl.semaphore_wait(
                    y_ready.at[first_minibatch],
                    value=required_down_arrivals(first_minibatch),
                    decrement=False,
                )

                @pl.when(last_minibatch != first_minibatch)
                def _wait_for_last_minibatch():
                    pl.semaphore_wait(
                        y_ready.at[last_minibatch],
                        value=required_down_arrivals(last_minibatch),
                        decrement=False,
                    )

            row = _thread_index_mgpu()
            safe_row = jnp.minimum(row, _COMBINE_ROWS - 1)
            for stage in range(combine_pipeline_depth):
                tile = first_tile + stage
                row_block = tile // combine_col_blocks
                col_block = lax.rem(tile, combine_col_blocks)

                @pl.when(tile < num_tiles)
                def _load_stage():
                    base = row_block * _COMBINE_ROWS
                    owner = peer_rank_ref[base + safe_row]
                    valid = (row < _COMBINE_ROWS) & (owner >= 0)
                    plgpu.copy_gmem_to_smem(
                        y_out.at[
                            pl.ds(base + row, 1),
                            pl.ds(col_block * combine_block_cols, combine_block_cols),
                        ],
                        combine_smem.at[stage],
                        scratch["combine_barrier"].at[stage],
                        predicate=valid,
                        oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
                        thread_parallel=True,
                    )

            for stage in range(combine_pipeline_depth):
                tile = first_tile + stage
                row_block = tile // combine_col_blocks
                col_block = lax.rem(tile, combine_col_blocks)

                @pl.when(tile < num_tiles)
                def _store_stage():
                    plgpu_primitives.barrier_wait_parity(
                        scratch["combine_barrier"].at[stage],
                        phase,
                    )
                    plgpu.commit_smem()
                    base = row_block * _COMBINE_ROWS
                    owner = peer_rank_ref[base + safe_row]
                    valid = (row < _COMBINE_ROWS) & (owner >= 0)
                    destination = plgpu.remote_ref(combine_out, jnp.maximum(owner, 0))
                    destination_row = jnp.where(valid, peer_token_ref[base + safe_row], num_slots)
                    plgpu.copy_smem_to_gmem(
                        combine_smem.at[stage],
                        destination.at[
                            pl.ds(destination_row, 1),
                            pl.ds(col_block * combine_block_cols, combine_block_cols),
                        ],
                        predicate=valid,
                        thread_parallel=True,
                    )

            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            plgpu_primitives.cta_barrier()

        @pl.when(sm < num_comm_sms)
        def _communication():
            @pl.when(wg == 0)
            def _rendezvous_wg():
                plgpu.semaphore_signal_multicast(rendezvous.at[sm], collective_axes=axis_name)
                pl.semaphore_wait(rendezvous.at[sm], value=lax.axis_size(axis_name), decrement=False)

            num_tasks = lax.div(num_routed_ref[()] + _DISPATCH_ROWS - 1, _DISPATCH_ROWS) * col_blocks
            num_steps = lax.div(num_tasks + num_comm_sms - 1 - sm, num_comm_sms)

            def run_task(step, _):
                task = sm + step * num_comm_sms
                _dispatch_task(
                    x_ref,
                    peer_rank_ref,
                    peer_token_ref,
                    x_out,
                    staging_ref,
                    owner_ref,
                    token_ref,
                    scratch["schedule_barrier"],
                    scratch["gather_barrier"],
                    dispatch_ready,
                    task,
                    lax.rem(step, 2) == 1,
                    wg,
                    topk=topk,
                    col_blocks=col_blocks,
                    minibatch_size=minibatch_size,
                    run_gather=run_dispatch_gather,
                    run_store=run_dispatch_store,
                )

            lax.fori_loop(0, num_steps, run_task, None)

            if run_down:
                num_combine_tiles = lax.div(num_routed_ref[()] + _COMBINE_ROWS - 1, _COMBINE_ROWS) * combine_col_blocks
                num_combine_tasks = lax.div(
                    num_combine_tiles + combine_pipeline_depth - 1,
                    combine_pipeline_depth,
                )
                num_combine_steps = lax.div(
                    num_combine_tasks + num_comm_sms - 1 - sm,
                    num_comm_sms,
                )

                def run_combine(step, _):
                    combine_task(sm + step * num_comm_sms, lax.rem(step, 2) == 1)

                lax.fori_loop(0, num_combine_steps, run_combine, None)

                @pl.when(wg == 0)
                def _combine_rendezvous():
                    plgpu.semaphore_signal_multicast(combine_done.at[sm], collective_axes=axis_name)
                    pl.semaphore_wait(combine_done.at[sm], value=lax.axis_size(axis_name), decrement=False)

        def routed_task(global_minibatch_idx, task, task_n_iters):
            group_sizes = [groups_ref[i] for i in range(tokens_per_expert.shape[0])]
            minibatch_first_block = global_minibatch_idx * minibatch_row_blocks
            minibatch_end_block = minibatch_first_block + minibatch_row_blocks
            remaining = task
            group_block_offset = jnp.int32(0)
            block = n_index = jnp.int32(0)
            valid = task < jnp.int32(0)

            for group_size in group_sizes:
                group_blocks = group_size // logical_tile_m
                first_block = jnp.maximum(minibatch_first_block, group_block_offset)
                end_block = jnp.minimum(minibatch_end_block, group_block_offset + group_blocks)
                row_blocks = jnp.maximum(jnp.int32(0), end_block - first_block)
                group_tasks = row_blocks * task_n_iters
                this_group = (~valid) & (remaining < group_tasks)
                row, col = plgpu.planar_snake(
                    remaining,
                    (jnp.maximum(jnp.int32(1), row_blocks), task_n_iters),
                    minor_dim=1,
                    tile_width=8,
                )
                block = lax.select(this_group, first_block + row, block)
                n_index = lax.select(this_group, col, n_index)
                remaining = lax.select(valid | this_group, remaining, remaining - group_tasks)
                valid = valid | this_group
                group_block_offset += group_blocks

            return group_sizes, block, n_index, valid

        def run_swiglu(
            first_tile_idx,
            tile_end,
            gate_gmem,
            up_gmem,
            hidden_gmem,
            gate_ready_base,
            hidden_ready_base,
        ):
            for stage in range(_SWIGLU_PIPE_DEPTH):
                tile_idx = first_tile_idx + stage
                valid_tile = tile_idx < tile_end
                row = tile_idx // swiglu_col_blocks
                col = lax.rem(tile_idx, swiglu_col_blocks)

                @pl.when(valid_tile)
                def _valid_tile():
                    @pl.when(wg == 0)
                    def _load():
                        parent_task = (row // 2) * n_iters + col // 2
                        pl.semaphore_wait(
                            gate_up_ready.at[gate_ready_base + parent_task],
                            value=2 * _CLUSTER_SIZE,
                            decrement=False,
                        )
                        row_slice = pl.ds(row * _SWIGLU_ROWS, _SWIGLU_ROWS)
                        col_slice = pl.ds(col * _SWIGLU_COLS, _SWIGLU_COLS)
                        plgpu.copy_gmem_to_smem(
                            gate_gmem.at[row_slice, col_slice],
                            gate_smem.at[stage],
                            scratch["swiglu_barrier"].at[stage],
                        )
                        plgpu.copy_gmem_to_smem(
                            up_gmem.at[row_slice, col_slice],
                            up_smem.at[stage],
                            scratch["swiglu_barrier"].at[stage],
                        )

                    plgpu.barrier_wait(scratch["swiglu_barrier"].at[stage])
                    wg_row = wg * (_SWIGLU_ROWS // 2)
                    wg_rows = pl.ds(wg_row, _SWIGLU_ROWS // 2)
                    gate = plgpu.load(
                        gate_smem.at[stage, wg_rows, :],
                        layout=plgpu.Layout.WGMMA,
                        optimized=False,
                    ).astype(jnp.float32)
                    up = plgpu.load(
                        up_smem.at[stage, wg_rows, :],
                        layout=plgpu.Layout.WGMMA,
                        optimized=False,
                    ).astype(jnp.float32)
                    hidden_smem.at[wg_rows, :][...] = (gate / (jnp.exp(-gate) + 1.0) * up).astype(x.dtype)
                    plgpu.commit_smem()
                    plgpu_primitives.cta_barrier()

                    @pl.when(wg == 0)
                    def _store():
                        plgpu.copy_smem_to_gmem(
                            hidden_smem,
                            hidden_gmem.at[
                                pl.ds(row * _SWIGLU_ROWS, _SWIGLU_ROWS),
                                pl.ds(col * _SWIGLU_COLS, _SWIGLU_COLS),
                            ],
                        )
                        plgpu.wait_smem_to_gmem(0)
                        pl.semaphore_signal(hidden_ready.at[hidden_ready_base + row // 2])

                    plgpu_primitives.cta_barrier()

        if run_compute:
            cluster = lax.axis_index("sm")

            @pl.when(cluster >= num_comm_clusters)
            def _compute():
                @plgpu.dynamic_scheduling_loop(
                    ("sm",),
                    thread_axis="wg",
                    cluster_axes=("cta",),
                    init_carry=(jnp.int32(0), jnp.int32(0)),
                )
                def _run_task(loop_info, carry):
                    gemm_step, previous_swiglu = carry
                    compute_task = loop_info.index[0] - num_comm_clusters
                    shared_gate_end = shared_gate_up_tasks
                    shared_up_end = 2 * shared_gate_up_tasks
                    shared_swiglu_end = shared_up_end + shared_swiglu_tasks
                    is_shared_gate = compute_task < shared_gate_end
                    is_shared_up = (shared_gate_end <= compute_task) & (compute_task < shared_up_end)
                    is_shared_swiglu = (shared_up_end <= compute_task) & (compute_task < shared_swiglu_end)
                    is_shared_down = (shared_swiglu_end <= compute_task) & (compute_task < shared_tasks)

                    routed_compute_task = jnp.maximum(jnp.int32(0), compute_task - shared_tasks)
                    global_minibatch_idx = routed_compute_task // minibatch_tasks
                    minibatch_task = lax.rem(routed_compute_task, minibatch_tasks)
                    gate_up_end = (2 if run_up else 1) * gate_up_tasks
                    swiglu_end = gate_up_end + (swiglu_tasks if run_down else 0)
                    is_routed_swiglu = (
                        (compute_task >= shared_tasks)
                        & (gate_up_end <= minibatch_task)
                        & (minibatch_task < swiglu_end)
                    )
                    is_swiglu = is_shared_swiglu | is_routed_swiglu

                    @pl.when((previous_swiglu != 0) & (~is_swiglu))
                    def _sync_after_swiglu():
                        plgpu.barrier_arrive(scratch["stage_transition_barrier"])
                        plgpu.barrier_wait(scratch["stage_transition_barrier"])

                    is_gate_up = (compute_task >= shared_tasks) & (minibatch_task < gate_up_end)
                    gate_up_task = lax.rem(minibatch_task, gate_up_tasks)
                    output_stage = minibatch_task // gate_up_tasks
                    group_sizes, block, n_index, gate_up_valid = routed_task(
                        global_minibatch_idx,
                        gate_up_task,
                        n_iters,
                    )
                    num_rows = sum(group_sizes, start=jnp.int32(0))
                    gate_up_valid = gate_up_valid & (block < lax.div(num_rows, logical_tile_m))

                    @pl.when(is_gate_up & gate_up_valid)
                    def _gate_up_task():
                        group_info = GroupInfo.from_block(group_sizes, logical_tile_m, block)

                        @pl.when(wg == 0)
                        def _wait_for_dispatch():
                            minibatch_first_row = global_minibatch_idx * minibatch_size
                            minibatch_rows = jnp.maximum(
                                jnp.int32(0),
                                jnp.minimum(minibatch_size, num_routed_ref[()] - minibatch_first_row),
                            )
                            required_count = lax.div(minibatch_rows + _DISPATCH_ROWS - 1, _DISPATCH_ROWS) * col_blocks
                            pl.semaphore_wait(
                                dispatch_ready.at[global_minibatch_idx],
                                value=required_count,
                                decrement=False,
                            )

                        do_matmul(
                            x_out,
                            gate_ref.at[group_info.group_id],
                            gate_out,
                            grid_indices=(group_info.block, n_index, lax.axis_index("cta")),
                            wg_axis="wg",
                            collective_axes=("cta",),
                            local_index=gemm_step,
                            previous_total_k_iters=lax.select(gemm_step > 0, max_concurrent_steps, 0),
                            config=gemm_config,
                            group_info=group_info,
                            alternate_b_gmem=up_ref.at[group_info.group_id] if run_up else None,
                            alternate_out_gmem=up_out if run_up else None,
                            output_stage=output_stage,
                            output_ready=gate_up_ready if run_down else None,
                            output_ready_index=shared_gate_up_tasks + group_info.block * n_iters + n_index,
                            **compute_scratch,
                        )

                    shared_gemm_valid = compute_task < jnp.int32(0)
                    if run_shared:
                        shared_gate_up_task = lax.select(
                            is_shared_gate,
                            compute_task,
                            compute_task - shared_gate_up_tasks,
                        )
                        shared_block, shared_n_index = plgpu.planar_snake(
                            shared_gate_up_task,
                            (shared_row_blocks, n_iters),
                            minor_dim=1,
                            tile_width=8,
                        )

                        @pl.when(is_shared_gate | is_shared_up)
                        def _shared_gate_up_task():
                            group_info = GroupInfo.from_block(
                                [jnp.int32(x.shape[0])],
                                logical_tile_m,
                                shared_block,
                            )
                            do_matmul(
                                x_ref,
                                shared_gate_ref,
                                shared_gate_out,
                                grid_indices=(shared_block, shared_n_index, lax.axis_index("cta")),
                                wg_axis="wg",
                                collective_axes=("cta",),
                                local_index=gemm_step,
                                previous_total_k_iters=lax.select(gemm_step > 0, max_concurrent_steps, 0),
                                config=gemm_config,
                                group_info=group_info,
                                alternate_b_gmem=shared_up_ref,
                                alternate_out_gmem=shared_up_out,
                                output_stage=is_shared_up.astype(jnp.int32),
                                output_ready=gate_up_ready,
                                output_ready_index=shared_block * n_iters + shared_n_index,
                                **compute_scratch,
                            )

                        @pl.when(is_shared_swiglu)
                        def _shared_swiglu_task():
                            task = compute_task - shared_up_end
                            first_tile_idx = (
                                task * _CLUSTER_SIZE * _SWIGLU_PIPE_DEPTH + lax.axis_index("cta") * _SWIGLU_PIPE_DEPTH
                            )
                            run_swiglu(
                                first_tile_idx,
                                jnp.int32(shared_swiglu_tiles),
                                shared_gate_out,
                                shared_up_out,
                                shared_hidden_out,
                                jnp.int32(0),
                                jnp.int32(0),
                            )

                        shared_down_task = jnp.maximum(jnp.int32(0), compute_task - shared_swiglu_end)
                        shared_down_block, shared_down_n_index = plgpu.planar_snake(
                            shared_down_task,
                            (shared_row_blocks, down_n_iters),
                            minor_dim=1,
                            tile_width=8,
                        )

                        @pl.when(is_shared_down)
                        def _shared_down_task():
                            group_info = GroupInfo.from_block(
                                [jnp.int32(x.shape[0])],
                                logical_tile_m,
                                shared_down_block,
                            )

                            @pl.when(wg == 0)
                            def _wait_for_shared_swiglu():
                                pl.semaphore_wait(
                                    hidden_ready.at[shared_down_block],
                                    value=(logical_tile_m // _SWIGLU_ROWS) * swiglu_col_blocks,
                                    decrement=False,
                                )

                            do_matmul(
                                shared_hidden_out,
                                shared_down_ref,
                                shared_y_out,
                                grid_indices=(
                                    shared_down_block,
                                    shared_down_n_index,
                                    lax.axis_index("cta"),
                                ),
                                wg_axis="wg",
                                collective_axes=("cta",),
                                local_index=gemm_step,
                                previous_total_k_iters=lax.select(gemm_step > 0, max_concurrent_steps, 0),
                                config=gemm_config,
                                group_info=group_info,
                                **compute_scratch,
                            )

                        shared_gemm_valid = is_shared_gate | is_shared_up | is_shared_down

                    is_down = compute_task < jnp.int32(0)
                    down_valid = compute_task < jnp.int32(0)
                    if run_down:
                        swiglu_task = minibatch_task - gate_up_end

                        @pl.when(is_routed_swiglu)
                        def _swiglu_task():
                            tiles_per_minibatch = (minibatch_size // _SWIGLU_ROWS) * swiglu_col_blocks
                            first_tile_idx = (
                                global_minibatch_idx * tiles_per_minibatch
                                + swiglu_task * _CLUSTER_SIZE * _SWIGLU_PIPE_DEPTH
                                + lax.axis_index("cta") * _SWIGLU_PIPE_DEPTH
                            )
                            tile_end = jnp.minimum(
                                lax.div(num_routed_ref[()], _SWIGLU_ROWS) * swiglu_col_blocks,
                                (global_minibatch_idx + 1) * tiles_per_minibatch,
                            )
                            run_swiglu(
                                first_tile_idx,
                                tile_end,
                                gate_out,
                                up_out,
                                hidden_out,
                                jnp.int32(shared_gate_up_tasks),
                                jnp.int32(shared_row_blocks),
                            )

                        is_down = (compute_task >= shared_tasks) & (minibatch_task >= swiglu_end)
                        down_task = jnp.maximum(jnp.int32(0), minibatch_task - swiglu_end)
                        down_group_sizes, down_block, down_n_index, down_valid = routed_task(
                            global_minibatch_idx,
                            down_task,
                            down_n_iters,
                        )
                        down_valid = down_valid & (down_block < lax.div(num_rows, logical_tile_m))

                        @pl.when(is_down & down_valid)
                        def _down_task():
                            group_info = GroupInfo.from_block(down_group_sizes, logical_tile_m, down_block)

                            @pl.when(wg == 0)
                            def _wait_for_swiglu():
                                pl.semaphore_wait(
                                    hidden_ready.at[shared_row_blocks + group_info.block],
                                    value=(logical_tile_m // _SWIGLU_ROWS) * swiglu_col_blocks,
                                    decrement=False,
                                )

                            do_matmul(
                                hidden_out,
                                down_ref.at[group_info.group_id],
                                y_out,
                                grid_indices=(
                                    group_info.block,
                                    down_n_index,
                                    lax.axis_index("cta"),
                                ),
                                wg_axis="wg",
                                collective_axes=("cta",),
                                local_index=gemm_step,
                                previous_total_k_iters=lax.select(gemm_step > 0, max_concurrent_steps, 0),
                                config=gemm_config,
                                group_info=group_info,
                                output_ready=y_ready,
                                output_ready_index=global_minibatch_idx,
                                **compute_scratch,
                            )

                    gemm_valid = shared_gemm_valid | (is_gate_up & gate_up_valid) | (is_down & down_valid)
                    return gemm_step + gemm_valid.astype(jnp.int32), is_swiglu.astype(jnp.int32)

    return plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct((capacity, hidden), x.dtype),
            jax.ShapeDtypeStruct((capacity, intermediate), x.dtype),
            jax.ShapeDtypeStruct((capacity, intermediate), x.dtype),
            jax.ShapeDtypeStruct((capacity, intermediate), x.dtype),
            jax.ShapeDtypeStruct((capacity, hidden), x.dtype),
            jax.ShapeDtypeStruct((num_slots + 1, hidden), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], intermediate), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], intermediate), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], intermediate), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], hidden), x.dtype),
        ),
        scratch_types=scratch_types,
        grid=(num_comm_clusters + logical_compute_clusters if run_compute else num_clusters,),
        grid_names=("sm",),
        cluster=(_CLUSTER_SIZE,),
        cluster_names=("cta",),
        num_threads=2,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        kernel_name="mok_dispatch_gate_up",
    )(
        x,
        w_gate,
        w_up,
        down_weights,
        peer_rank,
        peer_token_idx,
        num_routed_tokens,
        tokens_per_expert,
        *shared_weights,
    )


def dispatch_mlp_swiglu_combine_backward(
    router_weights: Float[Array, "S four"],
    d_out: Float[Array, "T H"],
    x: Float[Array, "T H"],
    x_routed: Float[Array, "C H"],
    gate_routed: Float[Array, "C I"],
    up_routed: Float[Array, "C I"],
    hidden_routed: Float[Array, "C I"],
    gate_shared: Float[Array, "T I"],
    up_shared: Float[Array, "T I"],
    hidden_shared: Float[Array, "T I"],
    w_gate: Float[Array, "E H I"],
    w_up: Float[Array, "E H I"],
    w_down: Float[Array, "E I H"],
    shared: tuple[Float[Array, "H I"], Float[Array, "H I"], Float[Array, "I H"]],
    peer_rank: Int[Array, "C"],
    peer_token_idx: Int[Array, "C"],
    num_routed_tokens: Int[Array, ""],
    tokens_per_expert: Int[Array, "E"],
    *,
    axis_name: str,
    topk: int,
    num_comm_sms: int = 28,
    minibatch_size: int = 4096,
    gemm_config: TuningConfig,
):
    """Runs MoK's active BF16 backward task graph for one routed macrobatch."""
    capacity, hidden = x_routed.shape
    intermediate = gate_routed.shape[1]
    num_slots = x.shape[0] * topk
    d_router_alignment = _DISPATCH_ROWS * 2
    d_router_slots_size = (num_slots + d_router_alignment) // d_router_alignment * d_router_alignment
    if router_weights.shape != (num_slots, 4):
        raise ValueError(f"{router_weights.shape=} must equal {(num_slots, 4)}")
    if d_out.shape != x.shape:
        raise ValueError(f"{d_out.shape=} must equal {x.shape=}")
    if capacity > _MACROBATCH_SIZE:
        raise ValueError("MoK backward replay is required when routed capacity exceeds one macrobatch")
    if not gemm_config.collective:
        raise ValueError("persistent MoK compute uses cluster-2 GEMMs")
    if minibatch_size <= 0 or minibatch_size % (_DISPATCH_ROWS * _CLUSTER_SIZE):
        raise ValueError(f"{minibatch_size=} must be positive and divisible by 256")

    tile_m = gemm_config.tile_m
    tile_n = gemm_config.tile_n
    tile_k = gemm_config.tile_k
    max_concurrent_steps = gemm_config.max_concurrent_steps
    epilogue_tile_n = gemm_config.epilogue_tile_n
    logical_tile_m = tile_m * _CLUSTER_SIZE
    logical_tile_n = tile_n * _CLUSTER_SIZE
    if tile_m != _DISPATCH_ROWS:
        raise ValueError(f"{tile_m=} must equal MoK dispatch rows {_DISPATCH_ROWS}")
    if hidden % logical_tile_n or intermediate % logical_tile_n:
        raise ValueError(f"GEMM tiles do not divide H={hidden}, I={intermediate}")
    if hidden % tile_k or intermediate % tile_k or capacity % tile_k or x.shape[0] % tile_k:
        raise ValueError("MoK backward operands must be divisible by the BF16 K tile")

    num_sms = jax.local_devices()[0].core_count
    if num_sms % _CLUSTER_SIZE:
        raise ValueError(f"{num_sms=} must be divisible by MoK cluster size {_CLUSTER_SIZE}")
    if not 0 < num_comm_sms <= num_sms - _CLUSTER_SIZE:
        raise ValueError(f"{num_comm_sms=} must leave at least one compute cluster")
    if num_comm_sms % _CLUSTER_SIZE:
        raise ValueError(f"{num_comm_sms=} must be divisible by cluster size {_CLUSTER_SIZE}")

    num_comm_clusters = num_comm_sms // _CLUSTER_SIZE
    max_minibatches = (capacity + minibatch_size - 1) // minibatch_size
    shared_row_blocks = x.shape[0] // logical_tile_m
    routed_row_blocks = capacity // logical_tile_m
    intermediate_n_iters = intermediate // logical_tile_n
    hidden_n_iters = hidden // logical_tile_n
    shared_dgrad_down_tasks = shared_row_blocks * intermediate_n_iters
    shared_swiglu_tiles = (x.shape[0] // _SWIGLU_ROWS) * (intermediate // _SWIGLU_COLS)
    shared_swiglu_tasks = (shared_swiglu_tiles + _CLUSTER_SIZE * _SWIGLU_BWD_PIPE_DEPTH - 1) // (
        _CLUSTER_SIZE * _SWIGLU_BWD_PIPE_DEPTH
    )
    shared_dgrad_gate_up_tasks = shared_row_blocks * hidden_n_iters
    shared_wgrad_tasks = (hidden // logical_tile_m) * intermediate_n_iters
    shared_tasks = shared_dgrad_down_tasks + shared_swiglu_tasks + shared_dgrad_gate_up_tasks + 3 * shared_wgrad_tasks
    minibatch_row_blocks = minibatch_size // logical_tile_m
    minibatch_routed_dgrad_down_tasks = minibatch_row_blocks * intermediate_n_iters
    minibatch_routed_swiglu_tiles = (minibatch_size // _SWIGLU_ROWS) * (intermediate // _SWIGLU_COLS)
    minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + _CLUSTER_SIZE * _SWIGLU_BWD_PIPE_DEPTH - 1) // (
        _CLUSTER_SIZE * _SWIGLU_BWD_PIPE_DEPTH
    )
    minibatch_routed_dgrad_gate_up_tasks = minibatch_row_blocks * hidden_n_iters
    minibatch_routed_bwd_tasks = (
        minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_dgrad_gate_up_tasks
    )
    routed_dgrad_down_tasks = max_minibatches * minibatch_routed_dgrad_down_tasks
    routed_bwd_tasks = max_minibatches * minibatch_routed_bwd_tasks
    wgrad_matrix_tasks = w_gate.shape[0] * (hidden // logical_tile_m) * intermediate_n_iters
    logical_compute_clusters = shared_tasks + routed_bwd_tasks + 3 * wgrad_matrix_tasks
    dispatch_col_blocks = hidden // _DISPATCH_COLS
    combine_block_cols = min(_COMBINE_COLS, hidden)
    if hidden % _DISPATCH_COLS or hidden % combine_block_cols:
        raise ValueError(f"communication tiles do not divide {hidden=}")
    combine_col_blocks = hidden // combine_block_cols
    combine_pipeline_depth = min(_COMBINE_PIPE_DEPTH, combine_col_blocks)
    swiglu_col_blocks = intermediate // _SWIGLU_COLS
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(x.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
    scratch_types = {
        "stage_storage": plgpu.RefUnion(
            (
                plgpu.SMEM((_DISPATCH_ROWS, _DISPATCH_COLS), x.dtype),
                plgpu.SMEM((_DISPATCH_ROWS,), jnp.int32),
                plgpu.SMEM((_DISPATCH_ROWS,), jnp.int32),
                plgpu.SMEM((_DISPATCH_ROWS, 4), jnp.float32),
            ),
            (
                plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), x.dtype, transforms=transforms),
                plgpu.SMEM((max_concurrent_steps, tile_n, tile_k), x.dtype, transforms=transforms),
                plgpu.SMEM((tile_m, epilogue_tile_n), x.dtype),
            ),
            (
                plgpu.SMEM((max_concurrent_steps, tile_k, tile_m), x.dtype, transforms=transforms),
                plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), x.dtype, transforms=transforms),
                plgpu.SMEM((tile_m, epilogue_tile_n), x.dtype),
            ),
            (
                plgpu.SMEM(
                    (_SWIGLU_BWD_PIPE_DEPTH, _SWIGLU_ROWS, _SWIGLU_COLS),
                    x.dtype,
                ),
                plgpu.SMEM(
                    (_SWIGLU_BWD_PIPE_DEPTH, _SWIGLU_ROWS, _SWIGLU_COLS),
                    x.dtype,
                ),
                plgpu.SMEM(
                    (_SWIGLU_BWD_PIPE_DEPTH, _SWIGLU_ROWS, _SWIGLU_COLS),
                    x.dtype,
                ),
                plgpu.SMEM((_SWIGLU_BWD_PIPE_DEPTH, _SWIGLU_ROWS), jnp.float32),
            ),
            (plgpu.SMEM((combine_pipeline_depth, _COMBINE_ROWS, combine_block_cols), x.dtype),),
        ),
        "schedule_barrier": plgpu.Barrier(num_arrivals=2),
        "gather_barrier": plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
        "combine_barrier": plgpu.Barrier(
            num_arrivals=1,
            num_barriers=combine_pipeline_depth,
            orders_tensor_core=True,
        ),
        "swiglu_barrier": plgpu.Barrier(num_arrivals=3, num_barriers=_SWIGLU_BWD_PIPE_DEPTH),
        "a_tma_barrier": plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
        "b_tma_barrier": plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
        "store_done_barrier": plgpu.ClusterBarrier(
            collective_axes=("cta",),
            num_arrivals=1,
            num_barriers=1,
            orders_tensor_core=True,
            leader_tracked=True,
        ),
        "mma_done_barrier": plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
        "consumed_barrier": plgpu.Barrier(
            num_arrivals=1,
            num_barriers=max_concurrent_steps,
            orders_tensor_core=True,
        ),
        "acc_tmem": plgpu.TMEM((tile_m, tile_n * 2), jnp.float32, collective=True),
    }
    compute_scratch_names = (
        "a_tma_barrier",
        "b_tma_barrier",
        "store_done_barrier",
        "mma_done_barrier",
        "consumed_barrier",
        "acc_tmem",
    )

    def kernel(
        router_weights_ref,
        d_out_ref,
        x_ref,
        x_routed_ref,
        gate_routed_ref,
        up_routed_ref,
        hidden_routed_ref,
        gate_shared_ref,
        up_shared_ref,
        hidden_shared_ref,
        gate_ref,
        up_ref,
        down_ref,
        shared_gate_ref,
        shared_up_ref,
        shared_down_ref,
        peer_rank_ref,
        peer_token_ref,
        num_routed_ref,
        groups_ref,
        d_y_routed_out,
        d_hidden_routed_out,
        d_gate_routed_out,
        d_up_routed_out,
        d_x_routed_out,
        d_x_slots_out,
        d_hidden_shared_out,
        d_gate_shared_out,
        d_up_shared_out,
        d_x_shared_out,
        d_w_gate_out,
        d_w_up_out,
        d_w_down_out,
        d_w_shared_gate_out,
        d_w_shared_up_out,
        d_w_shared_down_out,
        router_weights_routed_out,
        d_router_weight_partials_out,
        d_router_slots_out,
        **scratch,
    ):
        sm = lax.axis_index("sm") * _CLUSTER_SIZE + lax.axis_index("cta")
        wg = lax.axis_index("wg")
        plgpu.set_max_registers(248, action="increase")
        rendezvous = pl.get_global(plgpu.SemaphoreType.REGULAR((num_comm_sms,)))
        combine_done = pl.get_global(plgpu.SemaphoreType.REGULAR((num_comm_sms,)))
        d_router_zero_done = pl.get_global(plgpu.SemaphoreType.REGULAR)
        d_router_zero_ready = pl.get_global(plgpu.SemaphoreType.REGULAR)
        router_weights_ready = pl.get_global(plgpu.SemaphoreType.REGULAR)
        d_y_ready = pl.get_global(plgpu.SemaphoreType.REGULAR((max_minibatches,)))
        d_hidden_ready = pl.get_global(
            plgpu.SemaphoreType.REGULAR((shared_dgrad_down_tasks + routed_dgrad_down_tasks,))
        )
        d_gate_up_ready = pl.get_global(plgpu.SemaphoreType.REGULAR((shared_row_blocks + routed_row_blocks,)))
        d_x_ready = pl.get_global(plgpu.SemaphoreType.REGULAR((max_minibatches,)))
        staging_ref, owner_ref, token_ref, row_scales_ref = scratch["stage_storage"][0]
        a_smem, b_smem, acc_smem = scratch["stage_storage"][1]
        wgrad_a_smem, wgrad_b_smem, wgrad_acc_smem = scratch["stage_storage"][2]
        d_hidden_smem, gate_smem, up_smem, router_smem = scratch["stage_storage"][3]
        (combine_smem,) = scratch["stage_storage"][4]
        compute_scratch = {name: scratch[name] for name in compute_scratch_names}
        compute_scratch.update(a_smem=a_smem, b_smem=b_smem, acc_smem=acc_smem)
        compute_scratch.update(b_smem_transposed=True)
        wgrad_compute_scratch = {name: scratch[name] for name in compute_scratch_names}
        wgrad_compute_scratch.update(
            a_smem=wgrad_a_smem,
            b_smem=wgrad_b_smem,
            acc_smem=wgrad_acc_smem,
            a_smem_transposed=True,
        )

        @pl.when(sm < num_comm_sms)
        def _communication():
            @pl.when(wg == 0)
            def _rendezvous_wg():
                plgpu.semaphore_signal_multicast(rendezvous.at[sm], collective_axes=axis_name)
                pl.semaphore_wait(rendezvous.at[sm], value=lax.axis_size(axis_name), decrement=False)

            @pl.when((sm == 0) & (wg == 0))
            def _zero_d_router():
                d_router_slots_out[...] = jnp.zeros_like(d_router_slots_out)
                plgpu.semaphore_signal_multicast(d_router_zero_done, collective_axes=axis_name)
                pl.semaphore_wait(d_router_zero_done, value=lax.axis_size(axis_name), decrement=False)
                pl.semaphore_signal(d_router_zero_ready)

            @pl.when(wg == 0)
            def _wait_for_d_router_zero():
                pl.semaphore_wait(d_router_zero_ready, value=1, decrement=False)

            plgpu_primitives.cta_barrier()

            num_router_weight_tasks = capacity // _DISPATCH_ROWS
            num_router_weight_steps = lax.div(num_router_weight_tasks + num_comm_sms - 1 - sm, num_comm_sms)

            def preload_router_weights(step, _):
                _preload_router_weights_task(
                    router_weights_ref,
                    peer_rank_ref,
                    peer_token_ref,
                    router_weights_routed_out,
                    owner_ref,
                    token_ref,
                    row_scales_ref,
                    scratch["schedule_barrier"],
                    scratch["gather_barrier"],
                    sm + step * num_comm_sms,
                    lax.rem(step, 2) == 1,
                    wg,
                )

            lax.fori_loop(0, num_router_weight_steps, preload_router_weights, None)

            plgpu_primitives.cta_barrier()

            @pl.when(wg == 0)
            def _router_weights_rendezvous():
                pl.semaphore_signal(router_weights_ready)
                pl.semaphore_wait(router_weights_ready, value=num_comm_sms, decrement=False)

            plgpu_primitives.cta_barrier()

            num_dispatch_tasks = lax.div(num_routed_ref[()] + _DISPATCH_ROWS - 1, _DISPATCH_ROWS) * dispatch_col_blocks
            num_dispatch_steps = lax.div(num_dispatch_tasks + num_comm_sms - 1 - sm, num_comm_sms)

            def run_dispatch(step, _):
                _dispatch_task(
                    d_out_ref,
                    peer_rank_ref,
                    peer_token_ref,
                    d_y_routed_out,
                    staging_ref,
                    owner_ref,
                    token_ref,
                    scratch["schedule_barrier"],
                    scratch["gather_barrier"],
                    d_y_ready,
                    sm + step * num_comm_sms,
                    lax.rem(num_router_weight_steps + step, 2) == 1,
                    wg,
                    topk=topk,
                    col_blocks=dispatch_col_blocks,
                    minibatch_size=minibatch_size,
                    run_gather=True,
                    run_store=True,
                    row_scales_ref=router_weights_routed_out,
                )

            lax.fori_loop(0, num_dispatch_steps, run_dispatch, None)

            active_row_blocks = lax.div(num_routed_ref[()] + _COMBINE_ROWS - 1, _COMBINE_ROWS)
            num_combine_tiles = active_row_blocks * combine_col_blocks
            num_combine_tasks = lax.div(
                num_combine_tiles + combine_pipeline_depth - 1,
                combine_pipeline_depth,
            )
            num_combine_steps = lax.div(num_combine_tasks + num_comm_sms - 1 - sm, num_comm_sms)

            def run_combine(step, _):
                task = sm + step * num_comm_sms
                first_tile = task * combine_pipeline_depth
                last_tile = jnp.minimum(first_tile + combine_pipeline_depth, num_combine_tiles) - 1
                first_minibatch = (first_tile // combine_col_blocks * _COMBINE_ROWS) // minibatch_size
                last_minibatch = (last_tile // combine_col_blocks * _COMBINE_ROWS) // minibatch_size

                def required_d_x_arrivals(global_minibatch_idx):
                    first_row = global_minibatch_idx * minibatch_size
                    rows = jnp.maximum(
                        jnp.int32(0),
                        jnp.minimum(minibatch_size, num_routed_ref[()] - first_row),
                    )
                    return lax.div(rows + logical_tile_m - 1, logical_tile_m) * hidden_n_iters * _CLUSTER_SIZE

                @pl.when(wg == 0)
                def _wait_for_d_x():
                    pl.semaphore_wait(
                        d_x_ready.at[first_minibatch],
                        value=required_d_x_arrivals(first_minibatch),
                        decrement=False,
                    )

                    @pl.when(last_minibatch != first_minibatch)
                    def _wait_for_last_minibatch():
                        pl.semaphore_wait(
                            d_x_ready.at[last_minibatch],
                            value=required_d_x_arrivals(last_minibatch),
                            decrement=False,
                        )

                row = _thread_index_mgpu()
                safe_row = jnp.minimum(row, _COMBINE_ROWS - 1)
                for stage in range(combine_pipeline_depth):
                    tile = first_tile + stage
                    row_block = tile // combine_col_blocks
                    col_block = lax.rem(tile, combine_col_blocks)

                    @pl.when(tile < num_combine_tiles)
                    def _load_stage():
                        base = row_block * _COMBINE_ROWS
                        owner = peer_rank_ref[base + safe_row]
                        valid = (row < _COMBINE_ROWS) & (owner >= 0)
                        plgpu.copy_gmem_to_smem(
                            d_x_routed_out.at[
                                pl.ds(base + row, 1),
                                pl.ds(col_block * combine_block_cols, combine_block_cols),
                            ],
                            combine_smem.at[stage],
                            scratch["combine_barrier"].at[stage],
                            predicate=valid,
                            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
                            thread_parallel=True,
                        )

                for stage in range(combine_pipeline_depth):
                    tile = first_tile + stage
                    row_block = tile // combine_col_blocks
                    col_block = lax.rem(tile, combine_col_blocks)

                    @pl.when(tile < num_combine_tiles)
                    def _store_stage():
                        plgpu_primitives.barrier_wait_parity(
                            scratch["combine_barrier"].at[stage],
                            lax.rem(step, 2) == 1,
                        )
                        plgpu.commit_smem()
                        base = row_block * _COMBINE_ROWS
                        owner = peer_rank_ref[base + safe_row]
                        valid = (row < _COMBINE_ROWS) & (owner >= 0)
                        destination = plgpu.remote_ref(d_x_slots_out, jnp.maximum(owner, 0))
                        destination_row = jnp.where(valid, peer_token_ref[base + safe_row], num_slots)
                        plgpu.copy_smem_to_gmem(
                            combine_smem.at[stage],
                            destination.at[
                                pl.ds(destination_row, 1),
                                pl.ds(col_block * combine_block_cols, combine_block_cols),
                            ],
                            predicate=valid,
                            thread_parallel=True,
                        )

                        @pl.when(valid & (col_block == 0))
                        def _store_d_router():
                            d_router = jnp.float32(0)
                            for col in range(swiglu_col_blocks):
                                d_router += d_router_weight_partials_out[col, base + safe_row]
                            destination = plgpu.remote_ref(d_router_slots_out, jnp.maximum(owner, 0))
                            _atomic_add_scalar_mgpu(destination.at[destination_row], d_router)

                plgpu.wait_smem_to_gmem(0)
                plgpu_primitives.cta_barrier()

            lax.fori_loop(0, num_combine_steps, run_combine, None)

            @pl.when(wg == 0)
            def _combine_rendezvous():
                plgpu.semaphore_signal_multicast(combine_done.at[sm], collective_axes=axis_name)
                pl.semaphore_wait(combine_done.at[sm], value=lax.axis_size(axis_name), decrement=False)

        def routed_task(global_minibatch_idx, task, task_n_iters):
            group_sizes = [groups_ref[i] for i in range(tokens_per_expert.shape[0])]
            minibatch_first_block = global_minibatch_idx * minibatch_row_blocks
            minibatch_end_block = minibatch_first_block + minibatch_row_blocks
            remaining = task
            group_block_offset = jnp.int32(0)
            block = n_index = jnp.int32(0)
            valid = task < jnp.int32(0)
            for group_size in group_sizes:
                group_blocks = group_size // logical_tile_m
                first_block = jnp.maximum(minibatch_first_block, group_block_offset)
                end_block = jnp.minimum(minibatch_end_block, group_block_offset + group_blocks)
                row_blocks = jnp.maximum(jnp.int32(0), end_block - first_block)
                group_tasks = row_blocks * task_n_iters
                this_group = (~valid) & (remaining < group_tasks)
                row, col = plgpu.planar_snake(
                    remaining,
                    (jnp.maximum(jnp.int32(1), row_blocks), task_n_iters),
                    minor_dim=1,
                    tile_width=8,
                )
                block = lax.select(this_group, first_block + row, block)
                n_index = lax.select(this_group, col, n_index)
                remaining = lax.select(valid | this_group, remaining, remaining - group_tasks)
                valid = valid | this_group
                group_block_offset += group_blocks
            return group_sizes, block, n_index, valid

        def wgrad_task(task, m_iters, n_iters):
            group_sizes = [groups_ref[i] for i in range(tokens_per_expert.shape[0])]
            tasks_per_expert = m_iters * n_iters
            expert = task // tasks_per_expert
            matrix_task = lax.rem(task, tasks_per_expert)
            m_index, n_index = plgpu.planar_snake(
                matrix_task,
                (m_iters, n_iters),
                minor_dim=1,
                tile_width=8,
            )
            group_start = group_end = jnp.int32(0)
            offset = jnp.int32(0)
            for i, group_size in enumerate(group_sizes):
                is_expert = expert == i
                group_start = lax.select(is_expert, offset, group_start)
                offset += group_size
                group_end = lax.select(is_expert, offset, group_end)
            return expert, m_index, n_index, group_start, group_end

        def run_swiglu_backward(
            first_tile_idx,
            tile_end,
            d_hidden_gmem,
            gate_gmem,
            up_gmem,
            d_gate_gmem,
            d_up_gmem,
            d_hidden_ready_base,
            d_gate_up_ready_base,
            router_weights_gmem=None,
            d_router_weight_partials_gmem=None,
        ):
            for stage in range(_SWIGLU_BWD_PIPE_DEPTH):
                tile_idx = first_tile_idx + stage
                valid_tile = tile_idx < tile_end
                row = tile_idx // swiglu_col_blocks
                col = lax.rem(tile_idx, swiglu_col_blocks)

                @pl.when(valid_tile)
                def _valid_tile():
                    @pl.when(wg == 0)
                    def _load():
                        parent_task = (row // 2) * intermediate_n_iters + col // 2
                        pl.semaphore_wait(
                            d_hidden_ready.at[d_hidden_ready_base + parent_task],
                            value=_CLUSTER_SIZE,
                            decrement=False,
                        )
                        row_slice = pl.ds(row * _SWIGLU_ROWS, _SWIGLU_ROWS)
                        col_slice = pl.ds(col * _SWIGLU_COLS, _SWIGLU_COLS)
                        plgpu.copy_gmem_to_smem(
                            d_hidden_gmem.at[row_slice, col_slice],
                            d_hidden_smem.at[stage],
                            scratch["swiglu_barrier"].at[stage],
                        )
                        plgpu.copy_gmem_to_smem(
                            gate_gmem.at[row_slice, col_slice],
                            gate_smem.at[stage],
                            scratch["swiglu_barrier"].at[stage],
                        )
                        plgpu.copy_gmem_to_smem(
                            up_gmem.at[row_slice, col_slice],
                            up_smem.at[stage],
                            scratch["swiglu_barrier"].at[stage],
                        )

                    plgpu.barrier_wait(scratch["swiglu_barrier"].at[stage])
                    if router_weights_gmem is not None:
                        thread = _thread_index_mgpu()
                        tile_row = lax.rem(thread, _SWIGLU_ROWS)
                        tile_col_half = thread // _SWIGLU_ROWS
                        routed_row = row * _SWIGLU_ROWS
                        router_weight = router_weights_gmem[routed_row + tile_row, 0]
                        inverse_weight = jnp.where(router_weight > 0, 1.0 / router_weight, 0.0)
                        d_router_partial = None
                        for i in range(_SWIGLU_COLS // 8):
                            cols = pl.ds(tile_col_half * (_SWIGLU_COLS // 2) + i * 4, 4)
                            gate_values = plgpu.load(
                                gate_smem.at[stage, :, cols],
                                layout=_SWIGLU_THREAD_LAYOUT,
                                optimized=False,
                            ).astype(jnp.float32)
                            up_values = plgpu.load(
                                up_smem.at[stage, :, cols],
                                layout=_SWIGLU_THREAD_LAYOUT,
                                optimized=False,
                            ).astype(jnp.float32)
                            d_hidden_values = plgpu.load(
                                d_hidden_smem.at[stage, :, cols],
                                layout=_SWIGLU_THREAD_LAYOUT,
                                optimized=False,
                            ).astype(jnp.float32)
                            sigmoid = 1.0 / (jnp.exp(-gate_values) + 1.0)
                            silu = gate_values * sigmoid
                            d_silu = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))
                            partial = jnp.sum(
                                d_hidden_values * inverse_weight * silu * up_values,
                                axis=1,
                            )
                            d_router_partial = partial if d_router_partial is None else d_router_partial + partial
                            gate_smem.at[stage, :, cols][...] = (d_hidden_values * up_values * d_silu).astype(x.dtype)
                            up_smem.at[stage, :, cols][...] = (d_hidden_values * silu).astype(x.dtype)
                        d_router_partial = _swiglu_thread_scalar_mgpu(d_router_partial)

                        @pl.when(tile_col_half == 1)
                        def _store_router_partial():
                            router_smem[stage, tile_row] = d_router_partial

                        plgpu.commit_smem()
                        plgpu_primitives.cta_barrier()

                        @pl.when(tile_col_half == 0)
                        def _store_router_gradient():
                            d_router_weight_partials_gmem[col, routed_row + tile_row] = (
                                d_router_partial + router_smem[stage, tile_row]
                            )

                        plgpu_primitives.cta_barrier()
                    else:
                        wg_row = wg * (_SWIGLU_ROWS // 2)
                        wg_rows = pl.ds(wg_row, _SWIGLU_ROWS // 2)
                        gate = plgpu.load(
                            gate_smem.at[stage, wg_rows, :],
                            layout=plgpu.Layout.WGMMA,
                            optimized=False,
                        ).astype(jnp.float32)
                        up = plgpu.load(
                            up_smem.at[stage, wg_rows, :],
                            layout=plgpu.Layout.WGMMA,
                            optimized=False,
                        ).astype(jnp.float32)
                        d_hidden = plgpu.load(
                            d_hidden_smem.at[stage, wg_rows, :],
                            layout=plgpu.Layout.WGMMA,
                            optimized=False,
                        ).astype(jnp.float32)
                        sigmoid = 1.0 / (jnp.exp(-gate) + 1.0)
                        silu = gate * sigmoid
                        d_silu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
                        gate_smem.at[stage, wg_rows, :][...] = (d_hidden * up * d_silu).astype(x.dtype)
                        up_smem.at[stage, wg_rows, :][...] = (d_hidden * silu).astype(x.dtype)
                        plgpu.commit_smem()
                        plgpu_primitives.cta_barrier()

                    @pl.when(wg == 0)
                    def _store():
                        plgpu.copy_smem_to_gmem(
                            gate_smem.at[stage],
                            d_gate_gmem.at[
                                pl.ds(row * _SWIGLU_ROWS, _SWIGLU_ROWS),
                                pl.ds(col * _SWIGLU_COLS, _SWIGLU_COLS),
                            ],
                        )
                        plgpu.copy_smem_to_gmem(
                            up_smem.at[stage],
                            d_up_gmem.at[
                                pl.ds(row * _SWIGLU_ROWS, _SWIGLU_ROWS),
                                pl.ds(col * _SWIGLU_COLS, _SWIGLU_COLS),
                            ],
                        )
                        plgpu.wait_smem_to_gmem(0)
                        pl.semaphore_signal(d_gate_up_ready.at[d_gate_up_ready_base + row // 2])

                    plgpu_primitives.cta_barrier()

        cluster = lax.axis_index("sm")

        @pl.when(cluster >= num_comm_clusters)
        def _compute():
            @plgpu.dynamic_scheduling_loop(
                ("sm",),
                thread_axis="wg",
                cluster_axes=("cta",),
                init_carry=(jnp.int32(0), jnp.int32(0), jnp.int32(0)),
            )
            def _run_task(loop_info, carry):
                gemm_step, previous_total_k_iters, previous_swiglu = carry
                compute_task = loop_info.index[0] - num_comm_clusters
                shared_dgrad_down_end = shared_dgrad_down_tasks
                shared_swiglu_end = shared_dgrad_down_end + shared_swiglu_tasks
                shared_dgrad_gate_up_end = shared_swiglu_end + shared_dgrad_gate_up_tasks
                shared_wgrad_down_end = shared_dgrad_gate_up_end + shared_wgrad_tasks
                shared_wgrad_gate_end = shared_wgrad_down_end + shared_wgrad_tasks
                is_shared_dgrad_down = compute_task < shared_dgrad_down_end
                is_shared_swiglu = (shared_dgrad_down_end <= compute_task) & (compute_task < shared_swiglu_end)
                is_shared_dgrad_gate_up = (shared_swiglu_end <= compute_task) & (
                    compute_task < shared_dgrad_gate_up_end
                )
                is_shared_wgrad_down = (shared_dgrad_gate_up_end <= compute_task) & (
                    compute_task < shared_wgrad_down_end
                )
                is_shared_wgrad_gate = (shared_wgrad_down_end <= compute_task) & (compute_task < shared_wgrad_gate_end)
                is_shared_wgrad_up = (shared_wgrad_gate_end <= compute_task) & (compute_task < shared_tasks)

                routed_task_idx = jnp.maximum(jnp.int32(0), compute_task - shared_tasks)
                global_minibatch_idx = routed_task_idx // minibatch_routed_bwd_tasks
                minibatch_task = lax.rem(routed_task_idx, minibatch_routed_bwd_tasks)
                routed_swiglu_start = minibatch_routed_dgrad_down_tasks
                routed_dgrad_gate_up_start = routed_swiglu_start + minibatch_routed_swiglu_tasks
                routed_wgrad_start = routed_bwd_tasks
                is_routed_dgrad_down = (
                    (compute_task >= shared_tasks)
                    & (routed_task_idx < routed_bwd_tasks)
                    & (minibatch_task < routed_swiglu_start)
                )
                is_routed_swiglu = (
                    (compute_task >= shared_tasks)
                    & (routed_task_idx < routed_bwd_tasks)
                    & (routed_swiglu_start <= minibatch_task)
                    & (minibatch_task < routed_dgrad_gate_up_start)
                )
                is_routed_dgrad_gate_up = (
                    (compute_task >= shared_tasks)
                    & (routed_task_idx < routed_bwd_tasks)
                    & (routed_dgrad_gate_up_start <= minibatch_task)
                )
                routed_wgrad_task = jnp.maximum(jnp.int32(0), routed_task_idx - routed_wgrad_start)
                is_routed_wgrad_down = (
                    (compute_task >= shared_tasks)
                    & (routed_wgrad_start <= routed_task_idx)
                    & (routed_wgrad_task < wgrad_matrix_tasks)
                )
                is_routed_wgrad_gate = (wgrad_matrix_tasks <= routed_wgrad_task) & (
                    routed_wgrad_task < 2 * wgrad_matrix_tasks
                )
                is_routed_wgrad_up = (2 * wgrad_matrix_tasks <= routed_wgrad_task) & (
                    routed_wgrad_task < 3 * wgrad_matrix_tasks
                )
                is_swiglu = is_shared_swiglu | is_routed_swiglu

                @pl.when((previous_swiglu != 0) & (~is_swiglu))
                def _sync_after_swiglu():
                    plgpu_primitives.cluster_barrier()

                shared_down_block, shared_down_n = plgpu.planar_snake(
                    jnp.maximum(jnp.int32(0), compute_task),
                    (shared_row_blocks, intermediate_n_iters),
                    minor_dim=1,
                    tile_width=8,
                )

                @pl.when(is_shared_dgrad_down)
                def _shared_dgrad_down():
                    group_info = GroupInfo.from_block(
                        [jnp.int32(x.shape[0])],
                        logical_tile_m,
                        shared_down_block,
                    )
                    do_matmul(
                        d_out_ref,
                        shared_down_ref,
                        d_hidden_shared_out,
                        grid_indices=(shared_down_block, shared_down_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_b=True,
                        output_ready=d_hidden_ready,
                        output_ready_index=shared_down_block * intermediate_n_iters + shared_down_n,
                        **compute_scratch,
                    )

                @pl.when(is_shared_swiglu)
                def _shared_swiglu():
                    task = compute_task - shared_dgrad_down_end
                    first_tile_idx = (
                        task * _CLUSTER_SIZE * _SWIGLU_BWD_PIPE_DEPTH + lax.axis_index("cta") * _SWIGLU_BWD_PIPE_DEPTH
                    )
                    run_swiglu_backward(
                        first_tile_idx,
                        jnp.int32(shared_swiglu_tiles),
                        d_hidden_shared_out,
                        gate_shared_ref,
                        up_shared_ref,
                        d_gate_shared_out,
                        d_up_shared_out,
                        jnp.int32(0),
                        jnp.int32(0),
                    )

                shared_gate_up_task = jnp.maximum(jnp.int32(0), compute_task - shared_swiglu_end)
                shared_gate_up_block, shared_gate_up_n = plgpu.planar_snake(
                    shared_gate_up_task,
                    (shared_row_blocks, hidden_n_iters),
                    minor_dim=1,
                    tile_width=8,
                )

                @pl.when(is_shared_dgrad_gate_up)
                def _shared_dgrad_gate_up():
                    @pl.when(wg == 0)
                    def _wait_for_shared_swiglu():
                        pl.semaphore_wait(
                            d_gate_up_ready.at[shared_gate_up_block],
                            value=2 * swiglu_col_blocks,
                            decrement=False,
                        )

                    group_info = GroupInfo.from_block(
                        [jnp.int32(x.shape[0])],
                        logical_tile_m,
                        shared_gate_up_block,
                    )
                    do_matmul(
                        d_gate_shared_out,
                        shared_gate_ref,
                        d_x_shared_out,
                        grid_indices=(shared_gate_up_block, shared_gate_up_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_b=True,
                        second_a_gmem=d_up_shared_out,
                        second_b_gmem=shared_up_ref,
                        **compute_scratch,
                    )

                shared_wgrad_task = jnp.maximum(jnp.int32(0), compute_task - shared_dgrad_gate_up_end)
                shared_wgrad_matrix_task = lax.rem(shared_wgrad_task, shared_wgrad_tasks)
                shared_wgrad_gate_m, shared_wgrad_gate_n = plgpu.planar_snake(
                    shared_wgrad_matrix_task,
                    (hidden // logical_tile_m, intermediate_n_iters),
                    minor_dim=1,
                    tile_width=8,
                )
                shared_wgrad_down_m, shared_wgrad_down_n = plgpu.planar_snake(
                    shared_wgrad_matrix_task,
                    (intermediate // logical_tile_m, hidden_n_iters),
                    minor_dim=1,
                    tile_width=8,
                )

                @pl.when(is_shared_wgrad_down)
                def _shared_wgrad_down():
                    group_info = GroupInfo.from_block(
                        [jnp.int32(intermediate)],
                        logical_tile_m,
                        shared_wgrad_down_m,
                    )
                    do_matmul(
                        hidden_shared_ref,
                        d_out_ref,
                        d_w_shared_down_out,
                        grid_indices=(shared_wgrad_down_m, shared_wgrad_down_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_a=True,
                        k_end=x.shape[0],
                        **wgrad_compute_scratch,
                    )

                @pl.when(is_shared_wgrad_gate | is_shared_wgrad_up)
                def _shared_wgrad_gate_up():
                    for row_block in range(shared_row_blocks):

                        @pl.when(wg == 0)
                        def _wait_for_shared_swiglu():
                            pl.semaphore_wait(
                                d_gate_up_ready.at[row_block],
                                value=2 * swiglu_col_blocks,
                                decrement=False,
                            )

                    group_info = GroupInfo.from_block(
                        [jnp.int32(hidden)],
                        logical_tile_m,
                        shared_wgrad_gate_m,
                    )
                    do_matmul(
                        x_ref,
                        d_gate_shared_out,
                        d_w_shared_gate_out,
                        grid_indices=(shared_wgrad_gate_m, shared_wgrad_gate_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_a=True,
                        alternate_b_gmem=d_up_shared_out,
                        alternate_out_gmem=d_w_shared_up_out,
                        output_stage=is_shared_wgrad_up.astype(jnp.int32),
                        k_end=x.shape[0],
                        **wgrad_compute_scratch,
                    )

                routed_down_task = jnp.maximum(jnp.int32(0), minibatch_task)
                routed_group_sizes, routed_down_block, routed_down_n, routed_down_valid = routed_task(
                    global_minibatch_idx,
                    routed_down_task,
                    intermediate_n_iters,
                )
                routed_num_rows = sum(routed_group_sizes, start=jnp.int32(0))
                routed_down_valid = routed_down_valid & (routed_down_block < lax.div(routed_num_rows, logical_tile_m))

                @pl.when(is_routed_dgrad_down & routed_down_valid)
                def _routed_dgrad_down():
                    @pl.when(wg == 0)
                    def _wait_for_reverse_combine():
                        first_row = global_minibatch_idx * minibatch_size
                        rows = jnp.maximum(
                            jnp.int32(0),
                            jnp.minimum(minibatch_size, num_routed_ref[()] - first_row),
                        )
                        required = lax.div(rows + _DISPATCH_ROWS - 1, _DISPATCH_ROWS)
                        pl.semaphore_wait(
                            d_y_ready.at[global_minibatch_idx],
                            value=required * dispatch_col_blocks,
                            decrement=False,
                        )

                    group_info = GroupInfo.from_block(routed_group_sizes, logical_tile_m, routed_down_block)
                    do_matmul(
                        d_y_routed_out,
                        down_ref.at[group_info.group_id],
                        d_hidden_routed_out,
                        grid_indices=(routed_down_block, routed_down_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_b=True,
                        output_ready=d_hidden_ready,
                        output_ready_index=(
                            shared_dgrad_down_tasks + routed_down_block * intermediate_n_iters + routed_down_n
                        ),
                        **compute_scratch,
                    )

                @pl.when(is_routed_swiglu)
                def _routed_swiglu():
                    task = minibatch_task - routed_swiglu_start
                    first_tile_idx = (
                        global_minibatch_idx * minibatch_routed_swiglu_tiles
                        + task * _CLUSTER_SIZE * _SWIGLU_BWD_PIPE_DEPTH
                        + lax.axis_index("cta") * _SWIGLU_BWD_PIPE_DEPTH
                    )
                    tile_end = jnp.minimum(
                        lax.div(num_routed_ref[()], _SWIGLU_ROWS) * swiglu_col_blocks,
                        (global_minibatch_idx + 1) * minibatch_routed_swiglu_tiles,
                    )
                    run_swiglu_backward(
                        first_tile_idx,
                        tile_end,
                        d_hidden_routed_out,
                        gate_routed_ref,
                        up_routed_ref,
                        d_gate_routed_out,
                        d_up_routed_out,
                        jnp.int32(shared_dgrad_down_tasks),
                        jnp.int32(shared_row_blocks),
                        router_weights_routed_out,
                        d_router_weight_partials_out,
                    )

                routed_gate_up_task = jnp.maximum(
                    jnp.int32(0),
                    minibatch_task - routed_dgrad_gate_up_start,
                )
                routed_gate_group_sizes, routed_gate_block, routed_gate_n, routed_gate_valid = routed_task(
                    global_minibatch_idx,
                    routed_gate_up_task,
                    hidden_n_iters,
                )
                routed_gate_num_rows = sum(routed_gate_group_sizes, start=jnp.int32(0))
                routed_gate_valid = routed_gate_valid & (
                    routed_gate_block < lax.div(routed_gate_num_rows, logical_tile_m)
                )

                @pl.when(is_routed_dgrad_gate_up & routed_gate_valid)
                def _routed_dgrad_gate_up():
                    @pl.when(wg == 0)
                    def _wait_for_routed_swiglu():
                        pl.semaphore_wait(
                            d_gate_up_ready.at[shared_row_blocks + routed_gate_block],
                            value=2 * swiglu_col_blocks,
                            decrement=False,
                        )

                    group_info = GroupInfo.from_block(routed_gate_group_sizes, logical_tile_m, routed_gate_block)
                    do_matmul(
                        d_gate_routed_out,
                        gate_ref.at[group_info.group_id],
                        d_x_routed_out,
                        grid_indices=(routed_gate_block, routed_gate_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_b=True,
                        second_a_gmem=d_up_routed_out,
                        second_b_gmem=up_ref.at[group_info.group_id],
                        output_ready=d_x_ready,
                        output_ready_index=global_minibatch_idx,
                        **compute_scratch,
                    )

                routed_matrix_task = lax.rem(routed_wgrad_task, wgrad_matrix_tasks)
                routed_wgrad_gate_task = lax.rem(routed_matrix_task, wgrad_matrix_tasks)
                expert, gate_m, gate_n, group_start, group_end = wgrad_task(
                    routed_wgrad_gate_task,
                    hidden // logical_tile_m,
                    intermediate_n_iters,
                )
                _, down_m, down_n, _, _ = wgrad_task(
                    routed_wgrad_gate_task,
                    intermediate // logical_tile_m,
                    hidden_n_iters,
                )
                routed_wgrad_valid = group_end > group_start

                @pl.when(is_routed_wgrad_down & routed_wgrad_valid)
                def _routed_wgrad_down():
                    @pl.when(wg == 0)
                    def _wait_for_reverse_combine():
                        for minibatch in range(max_minibatches):
                            first_row = minibatch * minibatch_size
                            rows = jnp.maximum(
                                jnp.int32(0),
                                jnp.minimum(minibatch_size, num_routed_ref[()] - first_row),
                            )

                            @pl.when((group_start < first_row + rows) & (first_row < group_end))
                            def _wait_for_minibatch():
                                required = lax.div(rows + _DISPATCH_ROWS - 1, _DISPATCH_ROWS)
                                pl.semaphore_wait(
                                    d_y_ready.at[minibatch],
                                    value=required * dispatch_col_blocks,
                                    decrement=False,
                                )

                    group_info = GroupInfo.from_block(
                        [jnp.int32(intermediate)],
                        logical_tile_m,
                        down_m,
                    )
                    do_matmul(
                        hidden_routed_ref,
                        d_y_routed_out,
                        d_w_down_out.at[expert],
                        grid_indices=(down_m, down_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_a=True,
                        k_start=group_start,
                        k_end=group_end,
                        **wgrad_compute_scratch,
                    )

                @pl.when((is_routed_wgrad_gate | is_routed_wgrad_up) & routed_wgrad_valid)
                def _routed_wgrad_gate_up():
                    active_row_blocks = lax.div(num_routed_ref[()] + logical_tile_m - 1, logical_tile_m)
                    for row_block in range(routed_row_blocks):

                        @pl.when((wg == 0) & (row_block < active_row_blocks))
                        def _wait_for_routed_swiglu():
                            pl.semaphore_wait(
                                d_gate_up_ready.at[shared_row_blocks + row_block],
                                value=2 * swiglu_col_blocks,
                                decrement=False,
                            )

                    group_info = GroupInfo.from_block(
                        [jnp.int32(hidden)],
                        logical_tile_m,
                        gate_m,
                    )
                    do_matmul(
                        x_routed_ref,
                        d_gate_routed_out,
                        d_w_gate_out.at[expert],
                        grid_indices=(gate_m, gate_n, lax.axis_index("cta")),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=gemm_step,
                        previous_total_k_iters=previous_total_k_iters,
                        config=gemm_config,
                        group_info=group_info,
                        transpose_a=True,
                        alternate_b_gmem=d_up_routed_out,
                        alternate_out_gmem=d_w_up_out.at[expert],
                        output_stage=is_routed_wgrad_up.astype(jnp.int32),
                        k_start=group_start,
                        k_end=group_end,
                        **wgrad_compute_scratch,
                    )

                shared_gemm = (
                    is_shared_dgrad_down
                    | is_shared_dgrad_gate_up
                    | is_shared_wgrad_down
                    | is_shared_wgrad_gate
                    | is_shared_wgrad_up
                )
                routed_gemm = (
                    (is_routed_dgrad_down & routed_down_valid)
                    | (is_routed_dgrad_gate_up & routed_gate_valid)
                    | (is_routed_wgrad_down & routed_wgrad_valid)
                    | (is_routed_wgrad_gate & routed_wgrad_valid)
                    | (is_routed_wgrad_up & routed_wgrad_valid)
                )
                gemm = shared_gemm | routed_gemm
                total_k_iters = lax.select(
                    is_shared_dgrad_down | is_routed_dgrad_down,
                    jnp.int32(hidden // tile_k),
                    lax.select(
                        is_shared_dgrad_gate_up | is_routed_dgrad_gate_up,
                        jnp.int32(2 * intermediate // tile_k),
                        lax.select(
                            is_shared_wgrad_down | is_shared_wgrad_gate | is_shared_wgrad_up,
                            jnp.int32(x.shape[0] // tile_k),
                            lax.div(group_end - group_start, tile_k),
                        ),
                    ),
                )
                return (
                    gemm_step + gemm.astype(jnp.int32),
                    lax.select(gemm, jnp.minimum(total_k_iters, max_concurrent_steps), previous_total_k_iters),
                    is_swiglu.astype(jnp.int32),
                )

            plgpu_primitives.cluster_barrier()

    outputs = plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct((capacity, hidden), x.dtype),
            jax.ShapeDtypeStruct((capacity, intermediate), x.dtype),
            jax.ShapeDtypeStruct((capacity, intermediate), x.dtype),
            jax.ShapeDtypeStruct((capacity, intermediate), x.dtype),
            jax.ShapeDtypeStruct((capacity, hidden), x.dtype),
            jax.ShapeDtypeStruct((num_slots + 1, hidden), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], intermediate), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], intermediate), x.dtype),
            jax.ShapeDtypeStruct((x.shape[0], intermediate), x.dtype),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(w_gate.shape, x.dtype),
            jax.ShapeDtypeStruct(w_up.shape, x.dtype),
            jax.ShapeDtypeStruct(w_down.shape, x.dtype),
            jax.ShapeDtypeStruct(shared[0].shape, x.dtype),
            jax.ShapeDtypeStruct(shared[1].shape, x.dtype),
            jax.ShapeDtypeStruct(shared[2].shape, x.dtype),
            jax.ShapeDtypeStruct((capacity, _TMA_FLOAT_ALIGNMENT), jnp.float32),
            jax.ShapeDtypeStruct((swiglu_col_blocks, capacity), jnp.float32),
            jax.ShapeDtypeStruct((d_router_slots_size,), jnp.float32),
        ),
        scratch_types=scratch_types,
        grid=(num_comm_clusters + logical_compute_clusters,),
        grid_names=("sm",),
        cluster=(_CLUSTER_SIZE,),
        cluster_names=("cta",),
        num_threads=2,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane,
            reduction_scratch_bytes=0,
        ),
        kernel_name="mok_dispatch_mlp_swiglu_combine_backward",
    )(
        router_weights,
        d_out,
        x,
        x_routed,
        gate_routed,
        up_routed,
        hidden_routed,
        gate_shared,
        up_shared,
        hidden_shared,
        w_gate,
        w_up,
        w_down,
        *shared,
        peer_rank,
        peer_token_idx,
        num_routed_tokens,
        tokens_per_expert,
    )
    return *outputs[:-1], outputs[-1][: num_slots + 1]
