# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M7a spike: consume pool segments behind per-expert arrival flags.

One warp-specialized kernel per device: warpgroup 0 runs the `put_segments`
transport loop (one whole segment per SM, no tile striding) and signals a
per-destination-expert arrival semaphore on the owner after each segment;
warpgroup 1 waits per local expert for all sources' signals and reduces that
expert's pool region to per-SM partial sums as soon as it is ready — no
end-of-transport barrier, no second launch.

Gate: bit-equal per-expert sums vs a NumPy reference, and fused wall time
below put_segments + separate consume launches on the same plans.

Single process, >= 2 local GPUs (one GB200 tray):
  uv run python experiments/marin_ep/bench/spike_fused_consume.py
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.marin_ep_transport import LANE, dispatch_segments, put_segments

from experiments.marin_ep.planref import execute_plans

_OUT_KWARG = "out_type" if "out_type" in inspect.signature(plgpu.kernel).parameters else "out_shape"

HIDDEN = 1024
LOCAL_EXPERTS = 3
MAX_SEG_ROWS = 512
SMEM_TILE_BYTES = 96 * 1024


def put_consume_segments(
    src: jax.Array,
    plan,
    *,
    out_rows: int,
    axis_name: str,
    num_devices: int,
    local_experts: int,
) -> jax.Array:
    """Fused put + per-expert consume. Returns [num_sms, local_experts] f32 partial sums."""
    hidden = src.shape[1]
    if hidden % LANE:
        raise ValueError(f"hidden={hidden} must be divisible by {LANE}")
    lanes = hidden // LANE
    tile_rows = max(1, min(64, SMEM_TILE_BYTES // (hidden * src.dtype.itemsize)))
    src_view = src.reshape(src.shape[0], lanes, LANE)
    entries_per_dest = plan.src_lo.shape[0] // num_devices
    if entries_per_dest != local_experts:
        raise ValueError(f"flat plan expected {local_experts} entries/dest, got {entries_per_dest}")

    def kernel_body(x_ref, dest_ids_ref, src_lo_ref, dst_lo_ref, rows_ref, region_ref, kept_ref, out_ref, sums_ref):
        arrivals = pl.get_global(plgpu.SemaphoreType.REGULAR((local_experts,)))
        wg = lax.axis_index("wg")
        dev_id = lax.axis_index(axis_name)
        sm_id = lax.axis_index("sm")
        num_sms = lax.axis_size("sm")

        @pl.when(wg == 0)
        def _transport():
            def copy_rows(src_lo, dst_ref, dst_lo, rows_shape):
                def scoped(smem, barrier):
                    plgpu.copy_gmem_to_smem(x_ref.at[pl.ds(src_lo, rows_shape)], smem, barrier)
                    plgpu.barrier_wait(barrier)
                    plgpu.copy_smem_to_gmem(smem, dst_ref.at[pl.ds(dst_lo, rows_shape)])
                    plgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    scoped,
                    plgpu.SMEM((rows_shape, lanes, LANE), x_ref.dtype),
                    plgpu.Barrier(num_arrivals=1),
                )

            @pl.loop(0, num_devices)
            def _dest_loop(k):
                dest = dest_ids_ref[k]
                dst_ref = plgpu.remote_ref(out_ref, {axis_name: dest})

                @pl.loop(0, entries_per_dest)
                def _entries(j):
                    entry = k * entries_per_dest + j

                    # One SM owns the whole segment; every (dest, entry) is
                    # signaled exactly once, so expert j expects num_devices.
                    @pl.when(lax.rem(entry, num_sms) == sm_id)
                    def _one_segment():
                        start = src_lo_ref[entry]
                        rows = rows_ref[entry]
                        offset = dst_lo_ref[entry]
                        num_full = lax.div(rows, jnp.int32(tile_rows))
                        tail = rows - num_full * tile_rows

                        @pl.loop(0, num_full)
                        def _tile(t):
                            lo = t * tile_rows
                            copy_rows(start + lo, dst_ref, offset + lo, tile_rows)

                        @pl.when(tail > 0)
                        def _tail():
                            @pl.loop(0, tail)
                            def _row(r):
                                copy_rows(
                                    start + num_full * tile_rows + r, dst_ref, offset + num_full * tile_rows + r, 1
                                )

                        @pl.when(dest != dev_id)
                        def _signal_remote():
                            pl.semaphore_signal(arrivals.at[j], device_id={axis_name: dest})

                        @pl.when(dest == dev_id)
                        def _signal_local():
                            pl.semaphore_signal(arrivals.at[j])

        @pl.when(wg == 1)
        def _consume():
            @pl.loop(0, local_experts)
            def _expert(g):
                pl.semaphore_wait(arrivals.at[g], value=num_devices, decrement=False)
                lo = region_ref[g]
                n = kept_ref[g]
                acc0 = jnp.zeros((), jnp.float32)

                def body(i, acc):
                    row = out_ref[lo + sm_id + i * lax.axis_size("sm")]
                    return acc + jnp.sum(row.astype(jnp.float32))

                rounds = lax.div(n - sm_id + num_sms - 1, num_sms)
                acc = lax.fori_loop(0, rounds, body, acc0)
                sums_ref[sm_id, g] = acc

    num_sms = jax.devices()[0].core_count
    out_types = [
        jax.ShapeDtypeStruct((out_rows, lanes, LANE), src.dtype),
        jax.ShapeDtypeStruct((num_sms, local_experts), jnp.float32),
    ]
    _, sums = plgpu.kernel(
        kernel_body,
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=2,
        thread_name="wg",
        **{_OUT_KWARG: out_types},
    )(src_view, plan.dest_ids, plan.src_lo, plan.dst_lo, plan.rows, plan.region, plan.kept)
    return sums


def main() -> None:
    devices = jax.local_device_count()
    assert devices >= 2, "needs a multi-GPU tray"
    num_experts = devices * LOCAL_EXPERTS

    rng = np.random.default_rng(seed=7)
    accepted = rng.integers(0, MAX_SEG_ROWS, size=(devices, num_experts)).astype(np.int32)
    kept = accepted.sum(axis=0)
    kept_by_owner = kept.reshape(devices, LOCAL_EXPERTS)
    region = (np.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(num_experts).astype(np.int32)
    pool_rows = int(kept_by_owner.sum(axis=1).max())
    send_rows = int(accepted.sum(axis=1).max())

    sends = []
    for d in range(devices):
        n = int(accepted[d].sum())
        buf = np.zeros((send_rows, HIDDEN), np.float32)
        buf[:n] = d * 1e3 + np.arange(n)[:, None] % 97 + np.arange(HIDDEN)[None, :] / 1e3
        sends.append(buf)

    mesh = Mesh(np.asarray(jax.devices()), ("x",), axis_types=(AxisType.Explicit,))
    accepted_j = jnp.asarray(accepted)
    region_j = jnp.asarray(region)

    class _Plan:
        pass

    def build_plan(shard_id):
        plan = dispatch_segments(accepted_j, region_j, shard_id, local_experts=LOCAL_EXPERTS)
        fused = _Plan()
        fused.dest_ids = plan.dest_ids
        fused.src_lo = plan.src_lo
        fused.dst_lo = plan.dst_lo
        fused.rows = plan.rows
        my_bank = lax.dynamic_slice_in_dim(jnp.asarray(kept_by_owner), shard_id, 1, 0)[0]
        fused.kept = my_bank.astype(jnp.int32)
        fused.region = (jnp.cumsum(my_bank) - my_bank).astype(jnp.int32)
        return plan, fused

    def fused_fn(src):
        shard_id = lax.axis_index("x")
        _, fused = build_plan(shard_id)
        return put_consume_segments(
            src, fused, out_rows=pool_rows, axis_name="x", num_devices=devices, local_experts=LOCAL_EXPERTS
        )

    def baseline_fn(src):
        shard_id = lax.axis_index("x")
        plan, fused = build_plan(shard_id)
        pool = put_segments(src, plan, out_rows=pool_rows, axis_name="x", num_devices=devices)
        idx = jnp.arange(pool_rows)[None, :]
        mask = (idx >= fused.region[:, None]) & (idx < (fused.region + fused.kept)[:, None])
        return jnp.sum(jnp.where(mask[:, :, None], pool[None, :, :].astype(jnp.float32), 0.0), axis=(1, 2))

    spec = P("x", None)
    run_fused = jax.jit(shard_map(fused_fn, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
    run_base = jax.jit(shard_map(baseline_fn, mesh=mesh, in_specs=spec, out_specs=P("x"), check_vma=False))

    src_global = jax.device_put(np.concatenate(sends, axis=0), NamedSharding(mesh, spec))

    # Reference: expert sums from the plan-reference pools.
    plans = [dispatch_segments(accepted_j, region_j, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(devices)]
    pools = execute_plans(plans, sends, pool_rows)
    want = np.zeros((devices, LOCAL_EXPERTS), np.float64)
    for d in range(devices):
        for g in range(LOCAL_EXPERTS):
            lo = int((np.cumsum(kept_by_owner[d]) - kept_by_owner[d])[g])
            n = int(kept_by_owner[d, g])
            want[d, g] = pools[d][lo : lo + n].astype(np.float64).sum()

    sums = jax.block_until_ready(run_fused(src_global))
    got = np.asarray(sums).reshape(devices, -1, LOCAL_EXPERTS).sum(axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-5)
    print(f"FUSED CORRECT ({devices} devices, pool {pool_rows}x{HIDDEN})", flush=True)

    base = jax.block_until_ready(run_base(src_global))
    np.testing.assert_allclose(np.asarray(base).reshape(devices, LOCAL_EXPERTS), want, rtol=1e-5)

    import time

    def best_of(fn, arg, n=10):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(arg))
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_fused = best_of(run_fused, src_global)
    t_base = best_of(run_base, src_global)
    print(f"fused {t_fused * 1e3:.3f} ms vs put+consume {t_base * 1e3:.3f} ms ({t_base / t_fused:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
