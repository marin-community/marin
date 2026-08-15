# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M6 draft: fused dispatch/combine transport on Pallas Mosaic-GPU.

Replaces the two `ragged_all_to_all` calls of `xla_backend` with
device-initiated remote puts + arrival semaphores (SPEC Sections 3 F3/F5),
while routing metadata (counts, waterfilling, offsets) and expert GEMMs
stay in XLA. This is the intermediate M6 milestone: fused transport, stock
GEMMs; the persistent megakernel (M7) fuses the GEMMs behind the same
arrival flags.

Both directions are one generic kernel, `put_segments`: scatter contiguous
row segments of a local buffer into peers' buffers at given offsets, in
rotated destination order, signaling each destination's arrival semaphore
after its group of segments. The direction-specific part is pure metadata
(`dispatch_segments` / `combine_segments`), built with jnp inside the
shard_map body from the accepted-count matrix — CPU-testable against the
oracle keep rule.

Constructs were validated on GB200 in MEP-005
(`bench/spike_transport_mgpu.py`): `plgpu.remote_ref` puts,
`pl.semaphore_signal(device_id=...)`, `collective_axes="wg"` scoping, the
256-elements-per-dim async-copy limit (hence the `[rows, H/256, 256]`
payload view), and jax 0.10/0.11 `out_shape`/`out_type` compat.

Gradients: dispatch and combine are exact transposes of each other — the
cotangent of "scatter my segments to peers" is "peers scatter the
corresponding cotangent segments back", which is the same kernel run with
the opposite-direction plan. `put_with_transpose` packages that as a
`custom_vjp` so the backend differentiates end-to-end.

The kernel's output buffer is uninitialized memory (unlike
`ragged_all_to_all`, which fills from a zeroed operand), so rows not
covered by any segment are garbage; `put_with_transpose` zeroes rows at or
beyond `valid_rows` (with the compacted-region layout, coverage is the
contiguous prefix `[0, valid_rows)`).

Metadata builders are exact and tested on CPU
(`tests/test_mgpu_transport_metadata.py`); the kernel itself needs GPUs.
"""

import functools
import inspect
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxtyping import Array, Float, Int

_OUT_KWARG = "out_type" if "out_type" in inspect.signature(plgpu.kernel).parameters else "out_shape"

LANE = 256  # async-copy limit per dimension
# Per-copy SMEM staging budget; the GB200 carveout is ~228 KB/SM, and the
# tile height adapts so `tile_rows * hidden * itemsize` stays under this.
SMEM_TILE_BYTES = 160 * 1024
MAX_TILE_ROWS = 64


@dataclass(frozen=True)
class SegmentPlan:
    """Rotated per-destination segment layout for `put_segments`.

    `dest_ids[k]` is the k-th destination (rotated: self last);
    entries `entry_start[k] : entry_start[k+1]` belong to it. Entry `e`
    copies `src[src_lo[e] : + rows[e]]` to the destination buffer at
    `dst_lo[e]`. All arrays are device-local traced values (built inside
    shard_map).
    """

    dest_ids: Int[Array, " S"]
    entry_start: Int[Array, " S1"]
    src_lo: Int[Array, " N"]
    dst_lo: Int[Array, " N"]
    rows: Int[Array, " N"]


jax.tree_util.register_dataclass(SegmentPlan, [f.name for f in fields(SegmentPlan)], [])


def _rotated(dev_id: jax.Array, num_devices: int) -> jax.Array:
    """Destination order dev_id+1, ..., dev_id (self last)."""
    return jnp.mod(dev_id + 1 + jnp.arange(num_devices, dtype=jnp.int32), num_devices)


def dispatch_segments(
    accepted: Int[Array, "S E"],
    region: Int[Array, " E"],
    shard_id: jax.Array,
    *,
    local_experts: int,
) -> SegmentPlan:
    """SPEC F3 metadata: my compacted kept rows -> owners' pool regions.

    My send buffer is expert-major (compacted kept rows), so entry `g` is
    `[my_seg_start[g], +accepted[me, g])` landing at
    `region[g] + base_accepted[me, g]` in owner `g // El`'s pool. Entries
    must be per-expert: other sources' rows interleave between my
    consecutive experts' blocks at the destination.
    """
    num_devices = accepted.shape[0]
    my_rows = accepted[shard_id]  # [E]
    my_seg_start = jnp.cumsum(my_rows) - my_rows
    base_accepted = (jnp.cumsum(accepted, axis=0) - accepted)[shard_id]
    dst_lo = region + base_accepted
    dest_order = _rotated(shard_id, num_devices)
    # Entries grouped by destination in rotated order: experts of owner o.
    entry_expert = (dest_order[:, None] * local_experts + jnp.arange(local_experts)[None, :]).reshape(-1)
    return SegmentPlan(
        dest_ids=dest_order,
        entry_start=jnp.arange(num_devices + 1, dtype=jnp.int32) * local_experts,
        src_lo=my_seg_start[entry_expert].astype(jnp.int32),
        dst_lo=dst_lo[entry_expert].astype(jnp.int32),
        rows=my_rows[entry_expert].astype(jnp.int32),
    )


def combine_segments(
    accepted: Int[Array, "S E"],
    region: Int[Array, " E"],
    shard_id: jax.Array,
    *,
    local_experts: int,
) -> SegmentPlan:
    """SPEC F5 metadata: my expert-output pool rows -> sources' return buffers.

    For my expert `g` and source `d`, rows
    `pool[region[g] + base_accepted[d, g] : +accepted[d, g]]` return to
    `d`'s compacted buffer at `d`'s own segment start for `g`
    (`sum_{g' < g} accepted[d, g']`). Grouped by destination source in
    rotated order, `El` entries per destination.
    """
    num_devices = accepted.shape[0]
    base_accepted = jnp.cumsum(accepted, axis=0) - accepted  # [S, E]
    dest_seg_start = jnp.cumsum(accepted, axis=1) - accepted  # [S, E] per-source compacted prefix
    my_bank = shard_id * local_experts + jnp.arange(local_experts, dtype=jnp.int32)  # [El]
    dest_order = _rotated(shard_id, num_devices)  # [S]
    src_lo = region[my_bank][None, :] + base_accepted[dest_order[:, None], my_bank[None, :]]
    dst_lo = dest_seg_start[dest_order[:, None], my_bank[None, :]]
    rows = accepted[dest_order[:, None], my_bank[None, :]]
    return SegmentPlan(
        dest_ids=dest_order,
        entry_start=jnp.arange(num_devices + 1, dtype=jnp.int32) * local_experts,
        src_lo=src_lo.reshape(-1).astype(jnp.int32),
        dst_lo=dst_lo.reshape(-1).astype(jnp.int32),
        rows=rows.reshape(-1).astype(jnp.int32),
    )


def put_segments(
    src: Float[Array, "N H"],
    plan: SegmentPlan,
    *,
    out_rows: int,
    axis_name: str,
    num_devices: int,
) -> jax.Array:
    """Scatter `plan` segments of `src` into peers' output buffers.

    Every device runs this collectively; each returns its own receive
    buffer `[out_rows, H]`. Rows not covered by any segment are
    UNINITIALIZED — callers mask them (see `put_with_transpose`). Wait
    condition: every peer signals once per SM after finishing all segments
    destined to me.
    """
    hidden = src.shape[1]
    if hidden % LANE:
        raise ValueError(f"hidden={hidden} must be divisible by {LANE}")
    lanes = hidden // LANE
    tile_rows = max(1, min(MAX_TILE_ROWS, SMEM_TILE_BYTES // (hidden * src.dtype.itemsize)))
    src_view = src.reshape(src.shape[0], lanes, LANE)
    entries_per_dest = plan.src_lo.shape[0] // num_devices

    def kernel_body(x_ref, dest_ids_ref, src_lo_ref, dst_lo_ref, rows_ref, out_ref, done_ref):
        received_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        dev_id = lax.axis_index(axis_name)
        sm_id = lax.axis_index("sm")
        num_sms = lax.axis_size("sm")

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
                collective_axes="wg",
            )

        def send_entry(entry, dst_ref):
            start = src_lo_ref[entry]
            rows = rows_ref[entry]
            offset = dst_lo_ref[entry]
            num_full = lax.div(rows, jnp.int32(tile_rows))
            tail = rows - num_full * tile_rows

            # Full tiles strided across SMs; dynamic bound, so idle
            # iterations are never issued.
            @pl.loop(sm_id, num_full, step=num_sms)
            def _tile(tile_idx):
                lo = tile_idx * tile_rows
                copy_rows(start + lo, dst_ref, offset + lo, tile_rows)

            # TMA boxes are static. Segments with >= tile_rows rows finish
            # with one shifted full tile over the last tile_rows rows
            # (overlap re-copies identical data, which is harmless); smaller
            # segments go row-by-row. Tails are spread over SMs by entry.
            @pl.when((tail > 0) & (lax.rem(entry, num_sms) == sm_id))
            def _tail():
                @pl.when(rows >= tile_rows)
                def _shifted():
                    lo = rows - tile_rows
                    copy_rows(start + lo, dst_ref, offset + lo, tile_rows)

                @pl.when(rows < tile_rows)
                def _tiny():
                    @pl.loop(0, tail)
                    def _row(row_idx):
                        copy_rows(start + row_idx, dst_ref, offset + row_idx, 1)

        @pl.loop(0, num_devices)
        def _dest_loop(k):
            dest = dest_ids_ref[k]
            is_remote = dest != dev_id
            # Dict device ids: coordinates along other mesh axes default to
            # the caller's own, so this works on multi-axis meshes.
            dst_ref = plgpu.remote_ref(out_ref, {axis_name: dest})

            @pl.loop(0, entries_per_dest)
            def _entries(j):
                send_entry(k * entries_per_dest + j, dst_ref)

            @pl.when(is_remote)
            def _signal():
                pl.semaphore_signal(received_sem, device_id={axis_name: dest})

        pl.semaphore_wait(received_sem, value=(num_devices - 1) * num_sms, decrement=False)
        done_ref[0] = jnp.int32(1)

    num_sms = jax.devices()[0].core_count
    out_types = [
        jax.ShapeDtypeStruct((out_rows, lanes, LANE), src.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    ]
    out, _ = plgpu.kernel(
        kernel_body,
        grid=(num_sms,),
        grid_names=("sm",),
        num_threads=1,
        thread_name="wg",
        **{_OUT_KWARG: out_types},
    )(src_view, plan.dest_ids, plan.src_lo, plan.dst_lo, plan.rows)
    return out.reshape(out_rows, hidden)


def _zero_invalid_rows(buf: jax.Array, valid_rows: jax.Array) -> jax.Array:
    positions = jnp.arange(buf.shape[0], dtype=jnp.int32)
    return jnp.where((positions < valid_rows)[:, None], buf, 0)


def _int_cotangents(tree):
    return jax.tree.map(lambda a: np.zeros(jnp.shape(a), dtype=jax.dtypes.float0), tree)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def put_with_transpose(
    x: Float[Array, "N H"],
    fwd_plan: SegmentPlan,
    bwd_plan: SegmentPlan,
    fwd_valid: Int[Array, ""],
    bwd_valid: Int[Array, ""],
    fwd_rows: int,
    bwd_rows: int,
    axis_name: str,
    num_devices: int,
) -> jax.Array:
    """Differentiable `put_segments`: `bwd_plan` is the transpose transfer.

    Dispatch and combine plans built from the same `accepted`/`region` are
    each other's transposes (entry (d, g) copies the same row range in the
    opposite direction), so the cotangent of one put is the other put
    applied to the output cotangent. `*_valid` are the covered-row counts
    of the respective receive buffers; rows beyond them are zeroed.
    """
    out = put_segments(x, fwd_plan, out_rows=fwd_rows, axis_name=axis_name, num_devices=num_devices)
    return _zero_invalid_rows(out, fwd_valid)


def _put_with_transpose_fwd(x, fwd_plan, bwd_plan, fwd_valid, bwd_valid, fwd_rows, bwd_rows, axis_name, num_devices):
    out = put_with_transpose(x, fwd_plan, bwd_plan, fwd_valid, bwd_valid, fwd_rows, bwd_rows, axis_name, num_devices)
    return out, (fwd_plan, bwd_plan, fwd_valid, bwd_valid)


def _put_with_transpose_bwd(fwd_rows, bwd_rows, axis_name, num_devices, residuals, dy):
    fwd_plan, bwd_plan, fwd_valid, bwd_valid = residuals
    dx = put_segments(dy, bwd_plan, out_rows=bwd_rows, axis_name=axis_name, num_devices=num_devices)
    dx = _zero_invalid_rows(dx, bwd_valid)
    return (dx, _int_cotangents(fwd_plan), _int_cotangents(bwd_plan), *_int_cotangents((fwd_valid, bwd_valid)))


put_with_transpose.defvjp(_put_with_transpose_fwd, _put_with_transpose_bwd)
