# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused dispatch + gate/up GEMM for the marin EP mgpu path (M7b).

One persistent Warpgroup-semantics kernel runs the dispatch transport and the
gate/up grouped GEMM together: a dedicated transport warpgroup streams this
shard's kept rows into every owner's pool (the `put_segments` segment plan) and
signals per-destination-expert arrival semaphores; the two GEMM warpgroups run
the Blackwell ragged-dot schedule, gating each output tile on its expert group's
rows having fully arrived. The GEMM therefore starts while remote rows are
still in flight instead of waiting for the whole dispatch to land.

The kernel dual-writes: it returns the transported pool (`x_pool`) alongside
the gate/up product, so the backward is exactly `brd_expert_mlp`'s plus one
transpose put for the dispatch cotangent. Rows beyond the covered prefix of the
pool are UNINITIALIZED, matching `put_segments`; every consumer is bounded by
`group_sizes` offsets or segment plans, so they are never read.

Constraints inherited from the two fused halves: Warpgroup semantics means TMA
peer ids must be host-recomputable, so the destination loop is a static Python
unroll over `dev_id + 1 + k`; the cluster axis must not be named "x"
(`do_matmul` hardcodes "x" internally when told otherwise — we pass "cta"); and
only CTA 0 of each 2-CTA cluster transports, with arrival counts halved to
match.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import blackwell_ragged_dot_mgpu as brd
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu

from levanter.grug._moe.brd_expert_mlp import _CONFIGS, _N_ALIGN, ROW_ALIGN, _cfg, _grouped
from levanter.grug._moe.cudnn_wgrad_cute import cudnn_grouped_wgrad
from levanter.grug._moe.marin_ep_transport import _OUT_KWARG, LANE, SegmentPlan, put_with_transpose

# Rows per transport SMEM stage and resident lane chunk. SMEM budget at the
# hero shape (hidden 6144 = 24 lanes): chunk * rows * LANE * 2B = 24.6 KiB on
# top of the GEMM's double buffers, which forces max_concurrent_steps below the
# standalone kernel's 6.
TRANSPORT_STAGE_ROWS = 8
TRANSPORT_LANE_CHUNK = 6
_FUSED_MCS = 4


def _fused_config(k: int, n_padded: int) -> "brd.TuningConfig":
    base = _CONFIGS.get((k, n_padded))
    if base is None:
        return _cfg(128, 12, 0, _FUSED_MCS)
    return _cfg(base.tile_n, base.grid_tile_width, base.grid_minor_dim.value, _FUSED_MCS)


def _grid_size() -> int:
    return jax.local_devices()[0].core_count // 2


def expected_arrival_signals(
    accepted: jax.Array,
    shard_id: jax.Array,
    *,
    local_experts: int,
    grid_size: int,
    stage_rows: int = TRANSPORT_STAGE_ROWS,
) -> jax.Array:
    """Per-local-expert arrival-signal totals the fused kernel waits for.

    A sender moves a segment as SM-strided full stages plus one tail owner;
    each participating SM signals once, so a segment of `rows` rows yields
    `min(rows // stage_rows, grid_size) + (rows % stage_rows > 0)` signals.
    Senders with zero rows never signal. `accepted` is the globally known
    [S, E] acceptance matrix, so every shard computes its own totals.
    """
    my_cols = shard_id * local_experts + jnp.arange(local_experts, dtype=jnp.int32)
    rows = accepted[:, my_cols].astype(jnp.int32)  # [S, El]
    num_full = rows // stage_rows
    tail = rows - num_full * stage_rows
    signals = jnp.minimum(num_full, grid_size) + (tail > 0).astype(jnp.int32)
    return jnp.sum(signals, axis=0).astype(jnp.int32)


def _fused_dispatch_gemm(
    src: jax.Array,
    plan: SegmentPlan,
    weights_padded: jax.Array,
    group_sizes: jax.Array,
    expected: jax.Array,
    *,
    out_rows: int,
    axis_name: str,
    num_devices: int,
    local_experts: int,
):
    """Returns (x_pool, gu): the transported pool and the arrival-gated GEMM."""
    hidden = src.shape[1]
    if hidden % LANE:
        raise ValueError(f"hidden={hidden} must be divisible by {LANE}")
    lanes = hidden // LANE
    lane_chunk = TRANSPORT_LANE_CHUNK
    while lanes % lane_chunk:
        lane_chunk -= 1
    num_groups, k2, n = weights_padded.shape
    if (num_groups, k2) != (local_experts, hidden):
        raise ValueError(f"weights {weights_padded.shape} vs experts {local_experts}, hidden {hidden}")
    dtype = weights_padded.dtype
    config = _fused_config(hidden, n)
    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    eff_tile_m, eff_tile_n = 2 * tile_m, 2 * tile_n  # collective 2-CTA tiles
    if out_rows % eff_tile_m or n % eff_tile_n or hidden % tile_k:
        raise ValueError(
            f"shape ({out_rows},{hidden},{n}) not divisible by tiles ({eff_tile_m},{tile_k},{eff_tile_n})"
        )
    m_iters = out_rows // eff_tile_m
    n_iters = n // eff_tile_n
    max_concurrent_steps = config.max_concurrent_steps
    epilogue_tile_n = config.epilogue_tile_n
    entries_per_dest = plan.src_lo.shape[0] // num_devices

    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))

    def kernel(x_ref, src_lo_ref, dst_lo_ref, rows_ref, gs_ref, expected_ref, b_gmem, pool_gmem, out_gmem):
        arrivals = pl.get_global(plgpu.SemaphoreType.REGULAR((local_experts,)))
        wg = lax.axis_index("wg")
        dev_id = lax.axis_index(axis_name)
        sm_id = lax.axis_index("sm")
        num_sms = lax.axis_size("sm")
        cluster_idx = lax.axis_index("cta")
        linear_grid = (m_iters + local_experts - 1) * n_iters
        group_sizes_regs = [gs_ref[i] for i in range(local_experts)]

        @functools.partial(
            pl.run_scoped,
            a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms),
            b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms),
            acc_smem=plgpu.SMEM((tile_m, epilogue_tile_n), dtype),
            a_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            b_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            store_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
            mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
            consumed_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps, orders_tensor_core=True),
            acc_tmem=plgpu.TMEM((tile_m, tile_n * 2), jnp.float32, collective=True),
            t_smem=plgpu.SMEM((lane_chunk, TRANSPORT_STAGE_ROWS, LANE), src.dtype),
            t_barrier=plgpu.Barrier(num_arrivals=lane_chunk),
            collective_axes=("wg",),
        )
        def _scoped(**refs):
            t_smem = refs.pop("t_smem")
            t_barrier = refs.pop("t_barrier")

            # Only CTA 0 of each 2-CTA cluster transports; otherwise both
            # cluster members duplicate sends and double the arrival signals.
            @pl.when((wg == 2) & (cluster_idx == 0))
            def _transport():
                def copy_rows(src_lo, dst_ref, dst_lo, rows_shape):
                    # TMA boxes cap at 256 elements per dimension, so rows move
                    # in LANE-wide column chunks against the 2D refs; a chunk of
                    # lanes stages together behind one multi-arrival barrier.
                    for chunk_lo in range(0, lanes, lane_chunk):
                        for i in range(lane_chunk):
                            cols = pl.ds((chunk_lo + i) * LANE, LANE)
                            stage = t_smem.at[i, pl.ds(0, rows_shape)]
                            plgpu.copy_gmem_to_smem(x_ref.at[pl.ds(src_lo, rows_shape), cols], stage, t_barrier)
                        plgpu.barrier_wait(t_barrier)
                        for i in range(lane_chunk):
                            cols = pl.ds((chunk_lo + i) * LANE, LANE)
                            plgpu.copy_smem_to_gmem(
                                t_smem.at[i, pl.ds(0, rows_shape)], dst_ref.at[pl.ds(dst_lo, rows_shape), cols]
                            )
                        plgpu.wait_smem_to_gmem(0, wait_read_only=False)

                def signal(j, dest):
                    @pl.when(dest != dev_id)
                    def _remote():
                        pl.semaphore_signal(arrivals.at[j], device_id={axis_name: dest})

                    @pl.when(dest == dev_id)
                    def _local():
                        pl.semaphore_signal(arrivals.at[j])

                @pl.loop(0, entries_per_dest)
                def _expert_loop(j):
                    # Static Python unroll: Warpgroup-semantics TMA descriptors
                    # are built on the host per (ref, peer), so the peer id must
                    # be host-recomputable — `device_id() + constant` qualifies,
                    # a loop induction variable does not.
                    for k in range(num_devices):
                        dest = lax.rem(dev_id + jnp.int32(1 + k), jnp.int32(num_devices))
                        dst_ref = plgpu.remote_ref(pool_gmem, {axis_name: dest})
                        entry = k * entries_per_dest + j
                        start = src_lo_ref[entry]
                        rows = rows_ref[entry]
                        offset = dst_lo_ref[entry]
                        num_full = lax.div(rows, jnp.int32(TRANSPORT_STAGE_ROWS))
                        tail = rows - num_full * TRANSPORT_STAGE_ROWS

                        # The nested bodies run immediately at trace time, but
                        # bind the unrolled loop's values explicitly anyway.
                        @pl.loop(sm_id, num_full, step=num_sms)
                        def _tile(t, start=start, offset=offset, dst_ref=dst_ref):
                            lo = t * TRANSPORT_STAGE_ROWS
                            copy_rows(start + lo, dst_ref, offset + lo, TRANSPORT_STAGE_ROWS)

                        does_tail = (tail > 0) & (lax.rem(entry, num_sms) == sm_id)

                        @pl.when(does_tail)
                        def _tail(start=start, offset=offset, dst_ref=dst_ref, num_full=num_full, tail=tail):
                            @pl.loop(0, tail)
                            def _row(r):
                                copy_rows(
                                    start + num_full * TRANSPORT_STAGE_ROWS + r,
                                    dst_ref,
                                    offset + num_full * TRANSPORT_STAGE_ROWS + r,
                                    1,
                                )

                        my_full = jnp.maximum(lax.div(num_full - sm_id + num_sms - 1, num_sms), 0)

                        @pl.when(my_full > 0)
                        def _sig_full(j=j, dest=dest):
                            signal(j, dest)

                        @pl.when(does_tail)
                        def _sig_tail(j=j, dest=dest):
                            signal(j, dest)

            @pl.when(wg < 2)
            def _gemm():
                @plgpu.nd_loop(grid=(linear_grid,), collective_axes="sm")
                def mn_loop(loop_info: plgpu.NDLoopInfo):
                    (linear_idx,) = loop_info.index
                    local_index = loop_info.local_index
                    m_index, n_index = plgpu.planar_snake(
                        linear_idx,
                        (m_iters + local_experts - 1, n_iters),
                        config.grid_minor_dim,
                        config.grid_tile_width,
                    )
                    group_info = ragged_dot_mgpu.GroupInfo.create(group_sizes_regs, eff_tile_m, m_index)
                    # Gate the tile on its group's rows having fully arrived.
                    pl.semaphore_wait(
                        arrivals.at[group_info.group_id], value=expected_ref[group_info.group_id], decrement=False
                    )
                    brd.do_matmul(
                        pool_gmem,
                        b_gmem.at[group_info.group_id],
                        out_gmem,
                        grid_indices=(group_info.block, n_index, cluster_idx),
                        wg_axis="wg",
                        collective_axes=("cta",),
                        local_index=local_index,
                        config=config,
                        group_info=group_info,
                        **refs,
                    )

    compiler_params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)
    pool, out = plgpu.kernel(
        kernel,
        compiler_params=compiler_params,
        kernel_name="fused_dispatch_gemm",
        grid=(_grid_size(),),
        grid_names=("sm",),
        num_threads=3,
        thread_name="wg",
        cluster_names=("cta",),
        cluster=(2,),
        **{
            _OUT_KWARG: [
                jax.ShapeDtypeStruct((out_rows, hidden), src.dtype),
                jax.ShapeDtypeStruct((out_rows, n), dtype),
            ]
        },
    )(src, plan.src_lo, plan.dst_lo, plan.rows, group_sizes, expected, weights_padded)
    return pool, out


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12))
def fused_dispatch_expert_mlp(
    sorted_x,
    moe_w13,
    moe_w2,
    dispatch_plan,
    combine_plan,
    group_sizes,
    expected,
    send_rows,
    pool_recv_rows,
    pool_rows: int,
    assignments_per_shard: int,
    axis_name: str,
    ep_size: int,
):
    """out_pool = down(swiglu(dispatch(sorted_x) @ w13)) with dispatch+w13 fused.

    `sorted_x` is this shard's compacted kept rows (expert-major); the result
    is the expert-output pool in region layout, ready for the combine put.
    """
    y, _res = _fused_fwd(
        sorted_x,
        moe_w13,
        moe_w2,
        dispatch_plan,
        combine_plan,
        group_sizes,
        expected,
        send_rows,
        pool_recv_rows,
        pool_rows,
        assignments_per_shard,
        axis_name,
        ep_size,
    )
    return y


def _fused_fwd(
    sorted_x,
    moe_w13,
    moe_w2,
    dispatch_plan,
    combine_plan,
    group_sizes,
    expected,
    send_rows,
    pool_recv_rows,
    pool_rows: int,
    assignments_per_shard: int,
    axis_name: str,
    ep_size: int,
):
    local_experts = moe_w13.shape[0]
    moe_dim = moe_w2.shape[1]
    padded_rows = _round_up(pool_rows, ROW_ALIGN)
    n = moe_w13.shape[2]
    n_pad = -n % _N_ALIGN
    w13_padded = jnp.pad(moe_w13, ((0, 0), (0, 0), (0, n_pad))) if n_pad else moe_w13
    x_pool, gu_padded = _fused_dispatch_gemm(
        sorted_x,
        dispatch_plan,
        w13_padded,
        group_sizes,
        expected,
        out_rows=padded_rows,
        axis_name=axis_name,
        num_devices=ep_size,
        local_experts=local_experts,
    )
    gu = gu_padded[:, :n] if n_pad else gu_padded
    gate, up = gu[:, :moe_dim], gu[:, moe_dim:]
    act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(sorted_x.dtype)
    y = _grouped(act, moe_w2, group_sizes)
    res = (
        sorted_x,
        x_pool,
        moe_w13,
        moe_w2,
        gu,
        act,
        group_sizes,
        expected,
        dispatch_plan,
        combine_plan,
        send_rows,
        pool_recv_rows,
    )
    return y[:pool_rows], res


def _fused_bwd(pool_rows, assignments_per_shard, axis_name, ep_size, res, dy):
    (
        sorted_x,
        x_pool,
        moe_w13,
        moe_w2,
        gu,
        act,
        group_sizes,
        expected,
        dispatch_plan,
        combine_plan,
        send_rows,
        pool_recv_rows,
    ) = res
    moe_dim = moe_w2.shape[1]
    padded_rows = x_pool.shape[0]
    dy = dy.astype(x_pool.dtype)
    if padded_rows != pool_rows:
        dy = jnp.pad(dy, ((0, padded_rows - pool_rows), (0, 0)))
    dact = _grouped(dy, moe_w2.transpose(0, 2, 1), group_sizes)
    dw2 = cudnn_grouped_wgrad(act, dy, group_sizes)
    gate = gu[:, :moe_dim].astype(jnp.float32)
    up = gu[:, moe_dim:].astype(jnp.float32)
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    dact_f = dact.astype(jnp.float32)
    dgate = dact_f * up * (sg + silu * (1.0 - sg))
    dup = dact_f * silu
    d_gu = jnp.concatenate([dgate, dup], axis=1).astype(x_pool.dtype)
    dx_pool = _grouped(d_gu, moe_w13.transpose(0, 2, 1), group_sizes)
    dw13 = cudnn_grouped_wgrad(x_pool, d_gu, group_sizes)
    # The dispatch cotangent is the transpose transfer: pool-frame rows back to
    # the sender's compacted buffer, rows beyond `send_rows` zeroed.
    dsorted_x = put_with_transpose(
        dx_pool[:pool_rows],
        combine_plan,
        dispatch_plan,
        send_rows,
        pool_recv_rows,
        assignments_per_shard,
        pool_rows,
        axis_name,
        ep_size,
    ).astype(sorted_x.dtype)

    def float0(a):
        return np.zeros(jnp.shape(a), dtype=jax.dtypes.float0)

    return (
        dsorted_x,
        dw13,
        dw2,
        jax.tree.map(float0, dispatch_plan),
        jax.tree.map(float0, combine_plan),
        float0(group_sizes),
        float0(expected),
        float0(send_rows),
        float0(pool_recv_rows),
    )


fused_dispatch_expert_mlp.defvjp(_fused_fwd, _fused_bwd)
