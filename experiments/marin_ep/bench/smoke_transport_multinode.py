# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-process (1 GPU/process) smoke of `put_segments` over nvshmem.

The single-tray validation ran one process with 4 local devices, where XLA
serves symmetric memory through its collective-metadata path. Real hero
jobs span nodes, which requires the nvshmem path: one process per GPU and
`--xla_gpu_experimental_enable_nvshmem` in XLA_FLAGS. This script is run
once per GPU with:

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=<N> MARIN_EP_PROC_ID=<i>
  CUDA_VISIBLE_DEVICES=<local gpu> uv run python .../smoke_transport_multinode.py

Every process computes the same NumPy plan reference (fixed seed) and
asserts its own pool slice, then measures rotated all-to-all put bandwidth.
"""

import os
import time

import jax
import numpy as np

REQUIRED_FLAG = "--xla_gpu_experimental_enable_nvshmem"
if REQUIRED_FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " " + REQUIRED_FLAG

jax.distributed.initialize(
    coordinator_address=os.environ["MARIN_EP_COORD"],
    num_processes=int(os.environ["MARIN_EP_NUM_PROCS"]),
    process_id=int(os.environ["MARIN_EP_PROC_ID"]),
)

import jax.numpy as jnp  # noqa: E402
from jax import lax, shard_map  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from levanter.grug._moe.marin_ep_transport import dispatch_segments, put_segments  # noqa: E402

HIDDEN = 1024
LOCAL_EXPERTS = 3
MAX_SEG_ROWS = 2000


def main() -> None:
    proc = jax.process_index()
    devices = jax.device_count()
    assert jax.local_device_count() == 1, "nvshmem path needs one process per GPU"
    num_experts = devices * LOCAL_EXPERTS

    rng = np.random.default_rng(seed=5)
    accepted = rng.integers(0, MAX_SEG_ROWS, size=(devices, num_experts)).astype(np.int32)
    accepted[0, 1] = 0
    kept = accepted.sum(axis=0)
    kept_by_owner = kept.reshape(devices, LOCAL_EXPERTS)
    region = (np.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(num_experts).astype(np.int32)
    pool_rows = int(kept_by_owner.sum(axis=1).max())
    send_rows = int(accepted.sum(axis=1).max())

    sends = []
    for d in range(devices):
        n = int(accepted[d].sum())
        buf = np.zeros((send_rows, HIDDEN), np.float32)
        buf[:n] = d * 1e6 + np.arange(n)[:, None] + np.arange(HIDDEN)[None, :] / 1e3
        sends.append(buf)

    mesh = Mesh(np.asarray(jax.devices()), ("x",), axis_types=(AxisType.Explicit,))
    accepted_j = jnp.asarray(accepted)
    region_j = jnp.asarray(region)

    def local_fn(src):
        shard_id = lax.axis_index("x")
        plan = dispatch_segments(accepted_j, region_j, shard_id, local_experts=LOCAL_EXPERTS)
        return put_segments(src, plan, out_rows=pool_rows, axis_name="x", num_devices=devices)

    run = jax.jit(shard_map(local_fn, mesh=mesh, in_specs=P("x", None), out_specs=P("x", None), check_vma=False))
    src_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P("x", None)),
        sends[proc],
        (devices * send_rows, HIDDEN),
    )

    # Reference: every process executes all plans in NumPy (same seed).
    want = [np.zeros((pool_rows, HIDDEN), np.float32) for _ in range(devices)]
    for device in range(devices):
        plan = dispatch_segments(accepted_j, region_j, jnp.int32(device), local_experts=LOCAL_EXPERTS)
        dest_ids, entry_start = np.asarray(plan.dest_ids), np.asarray(plan.entry_start)
        src_lo, dst_lo, rows = (np.asarray(a) for a in (plan.src_lo, plan.dst_lo, plan.rows))
        for k, dest in enumerate(dest_ids):
            for e in range(entry_start[k], entry_start[k + 1]):
                want[dest][dst_lo[e] : dst_lo[e] + rows[e]] = sends[device][src_lo[e] : src_lo[e] + rows[e]]

    for attempt in range(2):
        pool = run(src_global)
        local = np.asarray(pool.addressable_shards[0].data)
        covered = int(kept_by_owner[proc].sum())
        np.testing.assert_array_equal(local[:covered], want[proc][:covered], err_msg=f"attempt {attempt} proc {proc}")
    print(f"[proc {proc}] CORRECT ({devices} devices, pool {pool_rows}x{HIDDEN})", flush=True)

    # Bandwidth: time the same collective at larger uniform segments.
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        jax.block_until_ready(run(src_global))
        times.append(time.perf_counter() - t0)
    best = min(times)
    egress = (
        (accepted[proc].sum() - accepted[proc, proc * LOCAL_EXPERTS : (proc + 1) * LOCAL_EXPERTS].sum()) * HIDDEN * 4
    )
    print(f"[proc {proc}] best {best * 1e3:.3f} ms, egress ~{egress / best / 1e9:.1f} GB/s", flush=True)


if __name__ == "__main__":
    main()
