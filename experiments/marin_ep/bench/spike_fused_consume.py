# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M7a spike: consume pool segments behind per-expert arrival flags.

One warp-specialized kernel per device: warpgroup 0 runs the `put_segments`
transport loop in EXPERT-MAJOR order (every destination's expert-j segments
before any expert-j+1 traffic) with tile striding across SMs, signaling unit
arrival semaphores on the owner (one per participating SM per entry, plus
one per tail — the mosaic backend only supports constant increments);
warpgroup 1 waits per local expert for the host-computed expected count and
runs a GEMM-weight consumer (REPEAT rotated read passes) on that region as
soon as it is ready — no end-of-transport barrier, no second launch.

Measured 2026-08-16 (1 GB200 tray, 4 devices, pool 14594x2560): correctness
exact; fused 0.953 ms vs put+consume 0.981 ms vs put-only 0.578 ms — only
~7% of the consumer hides because the memory-bound reader contends with
transport TMA for bandwidth. Verdict: flag-gating mechanism and expert-major
head starts validated; the perf case needs the compute-bound MMA consumer
(M7b), whose tensor-core work does not fight the copy engines for HBM.

Single process, >= 2 local GPUs (one GB200 tray):
  uv run python experiments/marin_ep/bench/spike_fused_consume.py
"""

import inspect
import time

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
_SCRATCH_KWARG = "scratch_types" if "scratch_types" in inspect.signature(plgpu.kernel).parameters else "scratch_shapes"

HIDDEN = 2560
LOCAL_EXPERTS = 3
MAX_SEG_ROWS = 2000
SMEM_TILE_BYTES = 96 * 1024
# GEMM-weight stand-in: the consumer reads every region row REPEAT times
# (rotated so loads cannot be hoisted), emulating compute that dominates
# transport the way the expert GEMM does.
REPEAT = 8


def _consume_expert(out_ref, lo, n, sm_id, num_sms):
    """REPEAT rotated read passes over region [lo, lo+n); per-SM partial sum."""
    acc0 = jnp.zeros((), jnp.float32)
    n_safe = jnp.maximum(n, 1)

    def rep(r, acc_r):
        def body(i, acc):
            idx = lax.rem(sm_id + i * num_sms + r, n_safe)
            row = out_ref[lo + idx]
            return acc + jnp.sum(row.astype(jnp.float32))

        rounds = lax.div(n - sm_id + num_sms - 1, num_sms)
        return acc_r + lax.fori_loop(0, rounds, body, jnp.zeros((), jnp.float32))

    return lax.fori_loop(0, REPEAT, rep, acc0)


def consume_segments(pool, region, kept, *, local_experts):
    """Standalone consumer launch (the baseline's second kernel)."""

    def kernel_body(out_ref, region_ref, kept_ref, sums_ref):
        sm_id = lax.axis_index("sm")
        num_sms = lax.axis_size("sm")

        @pl.loop(0, local_experts)
        def _expert(g):
            sums_ref[sm_id, g] = _consume_expert(out_ref, region_ref[g], kept_ref[g], sm_id, num_sms)

    num_sms = jax.devices()[0].core_count
    return plgpu.kernel(
        kernel_body,
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=1,
        thread_name="wg",
        **{_OUT_KWARG: jax.ShapeDtypeStruct((num_sms, local_experts), jnp.float32)},
    )(pool, region, kept)


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

    def kernel_body(
        x_ref,
        dest_ids_ref,
        src_lo_ref,
        dst_lo_ref,
        rows_ref,
        region_ref,
        kept_ref,
        expected_ref,
        out_ref,
        sums_ref,
        smem,
        barrier,
    ):
        arrivals = pl.get_global(plgpu.SemaphoreType.REGULAR((local_experts,)))
        wg = lax.axis_index("wg")
        dev_id = lax.axis_index(axis_name)
        sm_id = lax.axis_index("sm")
        num_sms = lax.axis_size("sm")

        @pl.when(wg == 0)
        def _transport():
            def copy_rows(src_lo, dst_ref, dst_lo, rows_shape):
                stage = smem.at[pl.ds(0, rows_shape)]
                plgpu.copy_gmem_to_smem(x_ref.at[pl.ds(src_lo, rows_shape)], stage, barrier)
                plgpu.barrier_wait(barrier)
                plgpu.copy_smem_to_gmem(stage, dst_ref.at[pl.ds(dst_lo, rows_shape)])
                plgpu.wait_smem_to_gmem(0, wait_read_only=False)

            def signal(j, dest):
                # Unit signals only: the mosaic backend requires a constant
                # increment. Expert j on the owner is ready once
                # `expected_ref[j]` unit signals arrive (one per participating
                # SM per entry, plus one for a tail).
                @pl.when(dest != dev_id)
                def _signal_remote():
                    pl.semaphore_signal(arrivals.at[j], device_id={axis_name: dest})

                @pl.when(dest == dev_id)
                def _signal_local():
                    pl.semaphore_signal(arrivals.at[j])

            # Expert-major send order: all destinations' expert-j segments
            # complete before any expert-(j+1) traffic, so owners' early
            # experts become consumable while later experts are in flight.
            @pl.loop(0, entries_per_dest)
            def _expert_loop(j):
                @pl.loop(0, num_devices)
                def _dest_loop(k):
                    dest = dest_ids_ref[k]
                    dst_ref = plgpu.remote_ref(out_ref, {axis_name: dest})
                    entry = k * entries_per_dest + j
                    start = src_lo_ref[entry]
                    rows = rows_ref[entry]
                    offset = dst_lo_ref[entry]
                    num_full = lax.div(rows, jnp.int32(tile_rows))
                    tail = rows - num_full * tile_rows

                    # Full tiles strided across SMs, exactly like put_segments.
                    @pl.loop(sm_id, num_full, step=num_sms)
                    def _tile(t):
                        lo = t * tile_rows
                        copy_rows(start + lo, dst_ref, offset + lo, tile_rows)

                    # Tails handled by one SM per entry; the shifted full tile
                    # re-copies old rows, so only `tail` NEW rows count.
                    does_tail = (tail > 0) & (lax.rem(entry, num_sms) == sm_id)

                    @pl.when(does_tail)
                    def _tail():
                        @pl.when(rows >= tile_rows)
                        def _shifted():
                            lo = rows - tile_rows
                            copy_rows(start + lo, dst_ref, offset + lo, tile_rows)

                        @pl.when(rows < tile_rows)
                        def _tiny():
                            @pl.loop(0, tail)
                            def _row(r):
                                copy_rows(start + r, dst_ref, offset + r, 1)

                    # Unit signals: one per participating SM per entry (SM s
                    # runs full tiles iff s < num_full), plus one for the tail.
                    my_full = jnp.maximum(lax.div(num_full - sm_id + num_sms - 1, num_sms), 0)

                    @pl.when(my_full > 0)
                    def _signal_full():
                        signal(j, dest)

                    @pl.when(does_tail)
                    def _signal_tail():
                        signal(j, dest)

        @pl.when(wg == 1)
        def _consume():
            @pl.loop(0, local_experts)
            def _expert(g):
                pl.semaphore_wait(arrivals.at[g], value=expected_ref[g], decrement=False)
                sums_ref[sm_id, g] = _consume_expert(out_ref, region_ref[g], kept_ref[g], sm_id, num_sms)

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
        **{
            _OUT_KWARG: out_types,
            _SCRATCH_KWARG: [
                plgpu.SMEM((tile_rows, lanes, LANE), src.dtype),
                plgpu.Barrier(num_arrivals=1),
            ],
        },
    )(src_view, plan.dest_ids, plan.src_lo, plan.dst_lo, plan.rows, plan.region, plan.kept, plan.expected)
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
        # Expected unit signals per local expert: one per participating SM per
        # source entry (SM s runs full tiles iff s < num_full) plus one per
        # nonzero tail. Must mirror the kernel's tile_rows/num_sms exactly.
        tile_rows = max(1, min(64, SMEM_TILE_BYTES // (HIDDEN * 4)))
        num_sms = jax.devices()[0].core_count
        my_cols = lax.dynamic_slice_in_dim(accepted_j, shard_id * LOCAL_EXPERTS, LOCAL_EXPERTS, 1)
        num_full = my_cols // tile_rows
        fused.expected = (
            (jnp.minimum(num_full, num_sms) + ((my_cols - num_full * tile_rows) > 0).astype(jnp.int32))
            .sum(axis=0)
            .astype(jnp.int32)
        )
        return plan, fused

    def fused_fn(src):
        shard_id = lax.axis_index("x")
        _, fused = build_plan(shard_id)
        return put_consume_segments(
            src, fused, out_rows=pool_rows, axis_name="x", num_devices=devices, local_experts=LOCAL_EXPERTS
        )

    def put_only_fn(src):
        shard_id = lax.axis_index("x")
        plan, _ = build_plan(shard_id)
        return put_segments(src, plan, out_rows=pool_rows, axis_name="x", num_devices=devices)

    def baseline_fn(src):
        shard_id = lax.axis_index("x")
        plan, fused = build_plan(shard_id)
        pool = put_segments(src, plan, out_rows=pool_rows, axis_name="x", num_devices=devices)
        return consume_segments(pool, fused.region, fused.kept, local_experts=LOCAL_EXPERTS)

    spec = P("x", None)
    run_fused = jax.jit(shard_map(fused_fn, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
    run_base = jax.jit(shard_map(baseline_fn, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))

    src_global = jax.device_put(np.concatenate(sends, axis=0), NamedSharding(mesh, spec))

    # Reference: expert sums from the plan-reference pools.
    plans = [dispatch_segments(accepted_j, region_j, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(devices)]
    pools = execute_plans(plans, sends, pool_rows)
    want = np.zeros((devices, LOCAL_EXPERTS), np.float64)
    for d in range(devices):
        for g in range(LOCAL_EXPERTS):
            lo = int((np.cumsum(kept_by_owner[d]) - kept_by_owner[d])[g])
            n = int(kept_by_owner[d, g])
            want[d, g] = REPEAT * pools[d][lo : lo + n].astype(np.float64).sum()

    sums = jax.block_until_ready(run_fused(src_global))
    got = np.asarray(sums).reshape(devices, -1, LOCAL_EXPERTS).sum(axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-5)
    print(f"FUSED CORRECT ({devices} devices, pool {pool_rows}x{HIDDEN})", flush=True)

    base = jax.block_until_ready(run_base(src_global))
    np.testing.assert_allclose(np.asarray(base).reshape(devices, -1, LOCAL_EXPERTS).sum(axis=1), want, rtol=1e-5)

    def best_of(fn, arg, n=10):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(arg))
            ts.append(time.perf_counter() - t0)
        return min(ts)

    run_put = jax.jit(shard_map(put_only_fn, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
    jax.block_until_ready(run_put(src_global))
    t_fused = best_of(run_fused, src_global)
    t_base = best_of(run_base, src_global)
    t_put = best_of(run_put, src_global)
    hidden = 1.0 - (t_fused - t_put) / max(t_base - t_put, 1e-9)
    print(
        f"put-only {t_put * 1e3:.3f} ms | fused {t_fused * 1e3:.3f} ms | put+consume {t_base * 1e3:.3f} ms"
        f" | consume hidden {100 * hidden:.0f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
