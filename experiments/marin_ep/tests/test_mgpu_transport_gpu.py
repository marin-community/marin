# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`put_segments` kernel vs the NumPy plan interpreter, on real GPUs.

Isolates transport-kernel bugs from backend wiring: random per-(source,
expert) segment counts (zeros included), compacted-region layout, dispatch
direction. The jitted collective runs twice — global semaphores live in
per-launch zero-initialized GMEM scratch, and a stale semaphore would make
the second call return before peers finish writing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.marin_ep_transport import dispatch_segments, put_segments

HIDDEN = 512
LOCAL_EXPERTS = 3
MAX_SEG_ROWS = 200


def _plan_reference(plans, sources, out_rows):
    """What the kernel should copy: execute every device's plan in NumPy."""
    outputs = [np.zeros((out_rows, HIDDEN), np.float32) for _ in range(len(plans))]
    for device, plan in enumerate(plans):
        dest_ids, entry_start = np.asarray(plan.dest_ids), np.asarray(plan.entry_start)
        src_lo, dst_lo, rows = (np.asarray(a) for a in (plan.src_lo, plan.dst_lo, plan.rows))
        for k, dest in enumerate(dest_ids):
            for e in range(entry_start[k], entry_start[k + 1]):
                outputs[dest][dst_lo[e] : dst_lo[e] + rows[e]] = sources[device][src_lo[e] : src_lo[e] + rows[e]]
    return outputs


def test_put_segments_matches_plan_reference_and_survives_relaunch():
    if jax.default_backend() != "gpu":
        pytest.skip("needs real GPUs")
    devices = len(jax.devices())
    if devices < 2:
        pytest.skip("needs >= 2 GPUs")
    num_experts = devices * LOCAL_EXPERTS

    rng = np.random.default_rng(seed=5)
    accepted = rng.integers(0, MAX_SEG_ROWS, size=(devices, num_experts)).astype(np.int32)
    accepted[0, 1] = 0  # zero-row segments must be handled
    kept = accepted.sum(axis=0)
    kept_by_owner = kept.reshape(devices, LOCAL_EXPERTS)
    region = (np.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(num_experts).astype(np.int32)
    pool_rows = int(kept_by_owner.sum(axis=1).max())
    send_rows = int(accepted.sum(axis=1).max())

    # Row value encodes (device, row) so misplaced copies are identifiable.
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
    src_global = jax.device_put(jnp.asarray(np.concatenate(sends, axis=0)), NamedSharding(mesh, P("x", None)))

    plans = [dispatch_segments(accepted_j, region_j, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(devices)]
    want = _plan_reference(plans, sends, pool_rows)

    for attempt in range(2):  # relaunch: semaphores must reset between calls
        got = np.asarray(run(src_global)).reshape(devices, pool_rows, HIDDEN)
        for d in range(devices):
            covered = int(kept_by_owner[d].sum())
            np.testing.assert_array_equal(got[d, :covered], want[d][:covered], err_msg=f"attempt {attempt} device {d}")


def test_put_segments_full_tile_path():
    """Segment sizes above the SMEM tile height exercise the strided full-tile
    loop, not just the row-by-row tail."""
    if jax.default_backend() != "gpu":
        pytest.skip("needs real GPUs")
    devices = len(jax.devices())
    if devices < 2:
        pytest.skip("needs >= 2 GPUs")
    num_experts = devices  # one expert per device, one big segment per pair

    rng = np.random.default_rng(seed=7)
    accepted = rng.integers(300, 1000, size=(devices, num_experts)).astype(np.int32)
    kept = accepted.sum(axis=0)
    # One expert per owner: each owner's pool starts at 0.
    region = np.zeros(num_experts, np.int32)
    pool_rows = int(kept.max())
    send_rows = int(accepted.sum(axis=1).max())

    sends = []
    for d in range(devices):
        n = int(accepted[d].sum())
        buf = np.zeros((send_rows, HIDDEN), np.float32)
        buf[:n] = rng.standard_normal((n, HIDDEN))
        sends.append(buf)

    mesh = Mesh(np.asarray(jax.devices()), ("x",), axis_types=(AxisType.Explicit,))
    accepted_j = jnp.asarray(accepted)
    region_j = jnp.asarray(region)

    def local_fn(src):
        shard_id = lax.axis_index("x")
        plan = dispatch_segments(accepted_j, region_j, shard_id, local_experts=1)
        return put_segments(src, plan, out_rows=pool_rows, axis_name="x", num_devices=devices)

    run = jax.jit(shard_map(local_fn, mesh=mesh, in_specs=P("x", None), out_specs=P("x", None), check_vma=False))
    src_global = jax.device_put(jnp.asarray(np.concatenate(sends, axis=0)), NamedSharding(mesh, P("x", None)))
    got = np.asarray(run(src_global)).reshape(devices, pool_rows, HIDDEN)

    plans = [dispatch_segments(accepted_j, region_j, jnp.int32(d), local_experts=1) for d in range(devices)]
    want = _plan_reference(plans, sends, pool_rows)
    for d in range(devices):
        covered = int(kept[d])
        np.testing.assert_array_equal(got[d, :covered], want[d][:covered], err_msg=f"device {d}")
