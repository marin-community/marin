# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M5 transport spike: Pallas Mosaic-GPU remote puts + arrival semaphores.

Measures the substrate the fused Marin EP kernel is built on: device-initiated
writes into peer GPUs' symmetric memory (`plgpu.remote_ref` +
`copy_smem_to_gmem`) with `pl.semaphore_signal(device_id=...)` arrival flags —
the `put` primitive of `experiments/marin_ep/SPEC.md` Section 3. Modeled on
jax's `pallas/ops/gpu/collective_matmul_mgpu.py` all-gather example.

The benchmark is a rotated all-to-all: every device pushes one `[rows, H]`
block to every peer's receive pool (peer `p` gets the block at slot
`my_dev_id`), in the rotated order the L1 simulator showed is required
(source `d` starts at peer `d+1`). Outputs: correctness vs `all_gather`, and
per-device egress bandwidth to calibrate the L1 model's `link_efficiency`
and `message_latency`.

Run on GB200 (NOT runnable on CPU jaxlib; unvalidated until the M5 session):
  - jax[cuda13] env (`uv sync --extra gpu`), which brings nvidia-nvshmem-cu13.
  - ONE PROCESS PER GPU (nvshmem path requires local_device_count == 1).
  - XLA_FLAGS must include --xla_gpu_experimental_enable_nvshmem.
  - single-node smoke: 1 process with all local GPUs also works (XLA
    collective-metadata path, no nvshmem needed).

  uv run python experiments/marin_ep/bench/spike_transport_mgpu.py
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import PartitionSpec as P

TILE_ROWS = 128


def all_to_all_put(x: jax.Array, axis_name: str, *, num_devices: int) -> jax.Array:
    """Push my `[S, rows, H]` source blocks into peers' pools; return my pool.

    Peer `p` receives my `x[p]` at pool slot `my_dev_id`. Rotated send order
    (start at `dev_id + 1`). Local block is copied locally. One SM per tile
    row-chunk via the grid; each SM signals each peer once, so receivers wait
    for `(S - 1) * num_sms` arrivals.
    """
    _, rows, hidden = x.shape
    if rows % TILE_ROWS:
        raise ValueError(f"rows={rows} must be divisible by {TILE_ROWS}")
    num_sms = jax.devices()[0].core_count
    tiles_total = rows // TILE_ROWS

    def kernel_body(x_ref, pool_ref, done_ref):
        received_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        dev_id = lax.axis_index(axis_name)
        sm_id = lax.axis_index("sm")

        def send_block(peer, dst_pool_ref):
            # Tiles are strided across SMs; each SM pushes its share of rows.
            @pl.loop(0, tiles_total)
            def _tile(tile_idx):
                @pl.when(lax.rem(tile_idx, jnp.int32(num_sms)) == sm_id)
                def _():
                    row_slice = pl.ds(tile_idx * TILE_ROWS, TILE_ROWS)

                    def scoped(smem, barrier):
                        plgpu.copy_gmem_to_smem(x_ref.at[peer, row_slice], smem, barrier)
                        plgpu.barrier_wait(barrier)
                        plgpu.copy_smem_to_gmem(smem, dst_pool_ref.at[dev_id, row_slice])
                        plgpu.wait_smem_to_gmem(0, wait_read_only=False)

                    pl.run_scoped(
                        scoped,
                        plgpu.SMEM((TILE_ROWS, hidden), x_ref.dtype),
                        plgpu.Barrier(num_arrivals=1),
                    )

        # Rotated order: peer dev_id+1 first (L1 sim: convoy avoidance).
        @pl.loop(1, num_devices)
        def _peer_loop(step):
            peer = lax.rem(dev_id + step, jnp.int32(num_devices))
            send_block(peer, plgpu.remote_ref(pool_ref, peer))
            pl.semaphore_signal(received_sem, device_id=peer)

        # Local block: plain copy into my own pool.
        send_block(dev_id, pool_ref)

        # Wait until every peer's every SM has signaled me.
        pl.semaphore_wait(received_sem, value=(num_devices - 1) * num_sms, decrement=False)
        done_ref[0] = jnp.int32(1)

    pool, _ = plgpu.kernel(
        kernel_body,
        out_type=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # receive pool
            jax.ShapeDtypeStruct((1,), jnp.int32),  # forces the wait into the dataflow
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=1,
        thread_name="wg",
    )(x)
    return pool


def main() -> None:
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))
    jax.set_mesh(mesh)

    rows_per_block, hidden = 8192, 3072  # ~50 MB bf16 per block, hero routed width
    x = jax.random.normal(jax.random.key(0), (num_devices, num_devices * rows_per_block, hidden), jnp.bfloat16).reshape(
        num_devices * num_devices, rows_per_block, hidden
    )
    x = jax.sharding.reshard(x, P("x", None, None))

    bench = jax.jit(
        jax.shard_map(
            functools.partial(all_to_all_put, axis_name="x", num_devices=num_devices),
            out_specs=P("x", None, None),
            check_vma=False,
        )
    )

    pool = bench(x)
    reference = jax.jit(
        jax.shard_map(
            # Peer p's pool slot d holds source d's block destined to p.
            lambda x_: lax.all_gather(x_, "x", axis=0, tiled=False)[:, lax.axis_index("x")],
            out_specs=P("x", None, None),
            check_vma=False,
        )
    )(x)
    np.testing.assert_array_equal(np.asarray(pool), np.asarray(reference))
    print("correctness: pool matches all_gather reference")

    _, kernels_ms = profiler.measure(bench, aggregate=False)(x)
    assert kernels_ms is not None
    time_ms = min(t for _, t in kernels_ms)
    egress_bytes = (num_devices - 1) * rows_per_block * hidden * 2
    print(
        f"devices={num_devices} block={rows_per_block}x{hidden} bf16: "
        f"{time_ms:.3f} ms, egress {egress_bytes / time_ms * 1e3 / 1e9:.1f} GB/s/device "
        f"(NVLink5 peak 900 GB/s)"
    )


if __name__ == "__main__":
    main()
