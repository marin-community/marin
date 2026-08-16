# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M7b spike: arrival-gated Blackwell grouped GEMM fused with the dispatch puts.

One persistent kernel per device, three warpgroups: wg0/wg1 are the upstream
``blackwell_ragged_dot_mgpu.do_matmul`` compute/store pair iterating the tile
grid; wg2 runs the ``put_segments`` transport loop in expert-major order,
writing peers' pools and signaling per-expert arrival semaphores. Each (m, n)
tile waits for its group's expected signal count before ``do_matmul`` touches
the pool, so remote experts' GEMM tiles start as soon as their rows land while
later experts are still in flight — no end-of-transport barrier and no second
launch.

Gate: outputs bit-match a put_segments -> ragged_dot_kernel two-launch
baseline, and the fused wall time beats it.

Single process, >= 2 local GPUs (one GB200 tray):
  uv run python experiments/marin_ep/bench/spike_fused_gemm.py
"""

import functools
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu, ragged_dot_mgpu
from jax.experimental.pallas.ops.gpu import blackwell_ragged_dot_mgpu as brd
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.marin_ep_transport import LANE, dispatch_segments

from experiments.marin_ep.planref import execute_plans

_OUT_KWARG = "out_type" if "out_type" in inspect.signature(plgpu.kernel).parameters else "out_shape"

HIDDEN = 2560  # k; must divide LANE for the transport view and tile_k for the GEMM
N_OUT = 1536  # n per expert (weight columns)
LOCAL_EXPERTS = 3
MAX_SEG_ROWS = 2000
TRANSPORT_STAGE_ROWS = 4  # rows per transport SMEM stage (k * rows * 4B each)

CONFIG = brd.TuningConfig(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    grid_tile_width=12,
    grid_minor_dim=blackwell_matmul_mgpu.MatmulDimension(0),
    max_concurrent_steps=4,  # small to leave SMEM for the transport stage
    collective=True,
)


def fused_dispatch_gemm(
    src: jax.Array,
    plan,
    weights: jax.Array,
    group_sizes: jax.Array,
    expected: jax.Array,
    *,
    out_rows: int,
    axis_name: str,
    num_devices: int,
    local_experts: int,
):
    """Returns (pool, out): transport-written pool and the gated grouped GEMM."""
    hidden = src.shape[1]
    if hidden % LANE:
        raise ValueError(f"hidden={hidden} must divide {LANE}")
    lanes = hidden // LANE
    num_groups, k2, n = weights.shape
    if (num_groups, k2) != (local_experts, hidden):
        raise ValueError(f"weights {weights.shape} vs experts {local_experts}, hidden {hidden}")
    dtype = weights.dtype
    tile_m, tile_n, tile_k = CONFIG.tile_m, CONFIG.tile_n, CONFIG.tile_k
    block_tile_m, block_tile_n = tile_m, tile_n
    eff_tile_m, eff_tile_n = 2 * tile_m, 2 * tile_n  # collective
    if out_rows % eff_tile_m or n % eff_tile_n or hidden % tile_k:
        raise ValueError(f"shape ({out_rows},{hidden},{n}) not divisible by tiles")
    m_iters = out_rows // eff_tile_m
    n_iters = n // eff_tile_n
    max_concurrent_steps = CONFIG.max_concurrent_steps
    epilogue_tile_n = CONFIG.epilogue_tile_n
    entries_per_dest = plan.src_lo.shape[0] // num_devices
    src_view = src.reshape(src.shape[0], lanes, LANE)

    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(x_ref, dest_ids_ref, src_lo_ref, dst_lo_ref, rows_ref, gs_ref, expected_ref, b_gmem, pool_gmem, out_gmem):
        arrivals = pl.get_global(plgpu.SemaphoreType.REGULAR((local_experts,)))
        wg = lax.axis_index("wg")
        dev_id = lax.axis_index(axis_name)
        sm_id = lax.axis_index("sm")
        num_sms = lax.axis_size("sm")
        cluster_idx = lax.axis_index("x")
        linear_grid = (m_iters + local_experts - 1) * n_iters
        group_sizes_regs = [gs_ref[i] for i in range(local_experts)]
        # The pool the GEMM reads is viewed [rows, lanes, LANE] for the puts.
        pool_view = pool_gmem.reshape(out_rows, lanes, LANE)

        @functools.partial(
            pl.run_scoped,
            a_smem=plgpu.SMEM((max_concurrent_steps, block_tile_m, tile_k), dtype, transforms=transforms),
            b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, block_tile_n), dtype, transforms=transforms),
            acc_smem=plgpu.SMEM((block_tile_m, epilogue_tile_n), dtype),
            a_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            b_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            store_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
            mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
            consumed_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps, orders_tensor_core=True),
            acc_tmem=plgpu.TMEM((block_tile_m, tile_n * 2), jnp.float32, collective=True),
            t_smem=plgpu.SMEM((TRANSPORT_STAGE_ROWS, lanes, LANE), src.dtype),
            t_barrier=plgpu.Barrier(num_arrivals=1),
            collective_axes=("wg",),
        )
        def _scoped(**refs):
            t_smem = refs.pop("t_smem")
            t_barrier = refs.pop("t_barrier")

            @pl.when(wg == 2)
            def _transport():
                def copy_rows(src_lo, dst_ref, dst_lo, rows_shape):
                    stage = t_smem.at[pl.ds(0, rows_shape)]
                    plgpu.copy_gmem_to_smem(x_ref.at[pl.ds(src_lo, rows_shape)], stage, t_barrier)
                    plgpu.barrier_wait(t_barrier)
                    plgpu.copy_smem_to_gmem(stage, dst_ref.at[pl.ds(dst_lo, rows_shape)])
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
                    @pl.loop(0, num_devices)
                    def _dest_loop(k):
                        dest = dest_ids_ref[k]
                        dst_ref = plgpu.remote_ref(pool_view, {axis_name: dest})
                        entry = k * entries_per_dest + j
                        start = src_lo_ref[entry]
                        rows = rows_ref[entry]
                        offset = dst_lo_ref[entry]
                        num_full = lax.div(rows, jnp.int32(TRANSPORT_STAGE_ROWS))
                        tail = rows - num_full * TRANSPORT_STAGE_ROWS

                        @pl.loop(sm_id, num_full, step=num_sms)
                        def _tile(t):
                            lo = t * TRANSPORT_STAGE_ROWS
                            copy_rows(start + lo, dst_ref, offset + lo, TRANSPORT_STAGE_ROWS)

                        does_tail = (tail > 0) & (lax.rem(entry, num_sms) == sm_id)

                        @pl.when(does_tail)
                        def _tail():
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
                        def _sig_full():
                            signal(j, dest)

                        @pl.when(does_tail)
                        def _sig_tail():
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
                        CONFIG.grid_minor_dim,
                        CONFIG.grid_tile_width,
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
                        collective_axes=("x",),
                        local_index=local_index,
                        config=CONFIG,
                        group_info=group_info,
                        **refs,
                    )

    num_sms = jax.local_devices()[0].core_count
    compiler_params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)
    pool, out = plgpu.kernel(
        kernel,
        compiler_params=compiler_params,
        kernel_name="fused_dispatch_gemm",
        grid=(num_sms // 2,),
        grid_names=("sm",),
        num_threads=3,
        thread_name="wg",
        cluster_names=("x",),
        cluster=(2,),
        **{
            _OUT_KWARG: [
                jax.ShapeDtypeStruct((out_rows, hidden), src.dtype),
                jax.ShapeDtypeStruct((out_rows, n), dtype),
            ]
        },
    )(src_view, plan.dest_ids, plan.src_lo, plan.dst_lo, plan.rows, group_sizes, expected, weights)
    return pool, out


def main() -> None:
    devices = jax.local_device_count()
    assert devices >= 2
    num_experts = devices * LOCAL_EXPERTS

    rng = np.random.default_rng(seed=9)
    accepted = rng.integers(0, MAX_SEG_ROWS, size=(devices, num_experts)).astype(np.int32)
    kept = accepted.sum(axis=0)
    kept_by_owner = kept.reshape(devices, LOCAL_EXPERTS)
    region = (np.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(num_experts).astype(np.int32)
    pool_rows_raw = int(kept_by_owner.sum(axis=1).max())
    pool_rows = (pool_rows_raw + 255) // 256 * 256
    send_rows = int(accepted.sum(axis=1).max())

    sends = []
    for d in range(devices):
        cnt = int(accepted[d].sum())
        buf = np.zeros((send_rows, HIDDEN), np.float32)
        buf[:cnt] = (d * 7 + np.arange(cnt)[:, None] % 13 + np.arange(HIDDEN)[None, :] % 5) * 0.01
        sends.append(buf.astype(np.float32))

    mesh = Mesh(np.asarray(jax.devices()), ("x",), axis_types=(AxisType.Explicit,))
    accepted_j = jnp.asarray(accepted)
    region_j = jnp.asarray(region)
    w_all = (0.05 * rng.standard_normal((devices, LOCAL_EXPERTS, HIDDEN, N_OUT))).astype(jnp.bfloat16)

    num_sms = jax.local_devices()[0].core_count

    def build(shard_id):
        plan = dispatch_segments(accepted_j, region_j, shard_id, local_experts=LOCAL_EXPERTS)
        my_bank = lax.dynamic_slice_in_dim(jnp.asarray(kept_by_owner), shard_id, 1, 0)[0].astype(jnp.int32)
        my_cols = lax.dynamic_slice_in_dim(accepted_j, shard_id * LOCAL_EXPERTS, LOCAL_EXPERTS, 1)
        num_full = my_cols // TRANSPORT_STAGE_ROWS
        expected = (
            (jnp.minimum(num_full, num_sms) + ((my_cols - num_full * TRANSPORT_STAGE_ROWS) > 0).astype(jnp.int32))
            .sum(axis=0)
            .astype(jnp.int32)
        )
        return plan, my_bank, expected

    def fused_fn(src, w_local):
        shard_id = lax.axis_index("x")
        plan, my_bank, expected = build(shard_id)
        pool, out = fused_dispatch_gemm(
            src.astype(jnp.bfloat16),
            plan,
            w_local[0],
            my_bank,
            expected,
            out_rows=pool_rows,
            axis_name="x",
            num_devices=devices,
            local_experts=LOCAL_EXPERTS,
        )
        return out

    spec = P("x", None)
    wspec = P("x", None, None, None)
    run_fused = jax.jit(shard_map(fused_fn, mesh=mesh, in_specs=(spec, wspec), out_specs=spec, check_vma=False))
    src_global = jax.device_put(np.concatenate(sends, axis=0), NamedSharding(mesh, spec))
    w_global = jax.device_put(np.asarray(w_all), NamedSharding(mesh, wspec))

    out = jax.block_until_ready(run_fused(src_global, w_global))

    # Reference: plan-executed pools -> per-group numpy GEMM.
    plans = [dispatch_segments(accepted_j, region_j, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(devices)]
    pools = execute_plans(plans, [s.astype(np.float32) for s in sends], pool_rows)
    out_np = np.asarray(out, np.float32).reshape(devices, pool_rows, N_OUT)
    for d in range(devices):
        offs = np.cumsum(kept_by_owner[d]) - kept_by_owner[d]
        for g in range(LOCAL_EXPERTS):
            lo, cnt = int(offs[g]), int(kept_by_owner[d, g])
            a = pools[d][lo : lo + cnt].astype(np.float32)
            a = jnp.asarray(a, jnp.bfloat16).astype(np.float32)
            ref = a @ np.asarray(w_all[d, g], np.float32)
            np.testing.assert_allclose(out_np[d, lo : lo + cnt], ref, rtol=5e-2, atol=0.1)
    print(f"FUSED GEMM CORRECT ({devices} devices, pool {pool_rows}x{HIDDEN} -> {N_OUT})", flush=True)


if __name__ == "__main__":
    main()
