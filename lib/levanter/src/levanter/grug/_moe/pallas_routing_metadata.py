# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable counting sorts for ``ring_filtered`` sender slots and receiver expert rows.

Each sort counts items per block, scans bucket counts, then ranks and scatters items. In-block ranks pack four
16-bit counters in each int64. The wrappers operate on shard-local arrays.
"""

import functools
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Bool, Float, Int

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

MoeRoutingMetadata = Literal["xla", "pallas"]
"""How ``ring_filtered`` builds its routing index maps: XLA fusions ("xla") or this module's kernels ("pallas")."""

# Items per program and warps (best of a sweep at one June layer-microbatch on MI355X).
_SEND_BLOCK = 256
_EXPERT_BLOCK = 512
_NUM_WARPS = 4
# Packed in-block counts: four 16-bit counters per int64 word, so a block holds at most 2**16 - 1 items.
_FIELDS = 4
_FIELD_BITS = 16


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def resolve_interpret(interpret: bool | None) -> bool:
    """``interpret=None``: compile with Triton on GPU and use the Pallas interpreter on CPU; other backends raise."""
    if interpret is not None:
        return interpret
    backend = jax.default_backend()
    if backend not in ("gpu", "cpu"):
        raise ValueError(
            f"the Pallas-Triton routing kernels need a GPU (or the CPU interpreter), got backend {backend!r}"
        )
    return backend == "cpu"


def _cost(inputs, outputs, *, compares: int) -> pl.CostEstimate:
    """Integer kernels: ``compares`` bucket tests, no transcendentals, and their operands read and written once."""
    return with_io_bytes_accessed(
        pl.CostEstimate(flops=compares, transcendentals=0, bytes_accessed=0),
        kernel_inputs_specs=inputs,
        kernel_outputs_specs=outputs,
    )


def _scalar(ref) -> jax.Array:
    """Load a one-element int32 operand as a scalar."""
    return jnp.sum(plgpu.load(ref.at[pl.ds(0, 1)]), dtype=jnp.int32)


def _exclusive_cumsum(x: jax.Array) -> jax.Array:
    return jnp.cumsum(x, axis=0, dtype=x.dtype) - x


def _field_counts(word: jax.Array, field: jax.Array) -> jax.Array:
    """Unpack the 16-bit counter ``field`` (0-3) of int64 words as int32."""
    shift = (field * _FIELD_BITS).astype(jnp.int64)
    return (jnp.right_shift(word, shift) & (2**_FIELD_BITS - 1)).astype(jnp.int32)


def _field_bit(field: int | jax.Array) -> jax.Array:
    return jnp.left_shift(jnp.int64(1), jnp.asarray(field * _FIELD_BITS, jnp.int64))


def _scan_kernel(counts_ref, *refs, blocks: int, offsets: bool):
    """One program per bucket: the bucket's first rank in every block of items (``start``).

    With ``offsets`` (receiver) the ranks run across buckets in bucket order, capacity is allocated in that order and
    program 0 also writes the capped group sizes, the drops and the accepted total. Without (sender) every bucket ranks
    from zero and each program writes its bucket's filled rows, ``min(count, capacity)``.
    """
    if offsets:
        cap_ref, start_ref, group_sizes_ref, dropped_ref, accepted_total_ref = refs
    else:
        cap_ref, start_ref, filled_ref = refs
    bucket = pl.program_id(0)
    table_rows, buckets = counts_ref.shape
    rows = jnp.arange(table_rows, dtype=jnp.int32)
    # Rows past `blocks` were never written: outputs start uninitialized.
    counts = jnp.where((rows < blocks)[:, None], plgpu.load(counts_ref), 0)
    column = plgpu.load(counts_ref.at[pl.ds(0, table_rows), bucket], mask=rows < blocks, other=0)
    total = jnp.sum(counts, axis=0, dtype=jnp.int32)
    mine = jnp.arange(buckets, dtype=jnp.int32) == bucket
    cap = _scalar(cap_ref)
    if not offsets:
        plgpu.store(start_ref.at[pl.ds(0, table_rows), bucket], _exclusive_cumsum(column))
        plgpu.store(filled_ref.at[pl.ds(0, buckets)], jnp.minimum(total, cap), mask=mine)
        return
    starts = _exclusive_cumsum(total)
    plgpu.store(
        start_ref.at[pl.ds(0, table_rows), bucket],
        jnp.sum(jnp.where(mine, starts, 0), dtype=jnp.int32) + _exclusive_cumsum(column),
    )
    accepted = jnp.minimum(total, jnp.maximum(cap - starts, 0))
    accepted_total = jnp.sum(accepted, keepdims=True, dtype=jnp.int32)
    first = bucket == 0
    plgpu.store(group_sizes_ref.at[pl.ds(0, buckets)], accepted, mask=jnp.full((buckets,), first))
    plgpu.store(
        dropped_ref.at[pl.ds(0, 1)],
        jnp.sum(total, keepdims=True, dtype=jnp.int32) - accepted_total,
        mask=jnp.full((1,), first),
    )
    plgpu.store(accepted_total_ref.at[pl.ds(0, 1)], accepted_total, mask=jnp.full((1,), first))


# ----------------------------------------------------------------------------------------------------------------------
# Sender: tokens -> send slots per destination shard.


def _send_hits(selected_ref, valid_ref, row0, *, block: int, tokens: int, local_experts: int, destinations: int):
    """Per destination, a [block] bool: whether each token of the block is valid and has an expert there."""
    topk = selected_ref.shape[1]
    rows = row0 + jnp.arange(block, dtype=jnp.int32)
    in_range = rows < tokens
    valid = (plgpu.load(valid_ref.at[pl.ds(row0, block)], mask=in_range, other=0) != 0) & in_range
    dests = [
        plgpu.load(selected_ref.at[pl.ds(row0, block), jnp.int32(k)], mask=in_range, other=0) // local_experts
        for k in range(topk)
    ]
    hits = []
    for d in range(destinations):
        hit = dests[0] == d
        for dest in dests[1:]:
            hit = hit | (dest == d)
        hits.append(hit & valid)
    return hits, dests, rows, in_range


def _send_count_kernel(selected_ref, valid_ref, counts_ref, *, block, tokens, local_experts, destinations):
    pid = pl.program_id(0)
    hits, _, _, _ = _send_hits(
        selected_ref,
        valid_ref,
        pid * block,
        block=block,
        tokens=tokens,
        local_experts=local_experts,
        destinations=destinations,
    )
    dest_ids = jnp.arange(destinations, dtype=jnp.int32)
    counts = jnp.zeros((destinations,), jnp.int32)
    for d, hit in enumerate(hits):
        counts = jnp.where(dest_ids == d, jnp.sum(hit.astype(jnp.int32), dtype=jnp.int32), counts)
    plgpu.store(counts_ref.at[pid, pl.ds(0, destinations)], counts)


def _send_slot_kernel(
    selected_ref,
    valid_ref,
    cap_ref,
    start_ref,
    filled_ref,
    slot_ref,
    source_ref,
    kept_ref,
    *,
    block,
    tokens,
    local_experts,
    destinations,
    pair_capacity,
    init_chunks,
):
    pid = pl.program_id(0)
    send_rows = destinations * pair_capacity
    hits, dests, rows, in_range = _send_hits(
        selected_ref,
        valid_ref,
        pid * block,
        block=block,
        tokens=tokens,
        local_experts=local_experts,
        destinations=destinations,
    )
    cap = _scalar(cap_ref)
    start = plgpu.load(start_ref.at[pid, pl.ds(0, destinations)])
    dest_ids = jnp.arange(destinations, dtype=jnp.int32)
    keeps = []
    for w in range(pl.cdiv(destinations, _FIELDS)):
        word = jnp.zeros((block,), jnp.int64)
        for d in range(w * _FIELDS, min(destinations, (w + 1) * _FIELDS)):
            word = word | jnp.where(hits[d], _field_bit(d - w * _FIELDS), jnp.int64(0))
        word = jnp.cumsum(word, dtype=jnp.int64)
        for d in range(w * _FIELDS, min(destinations, (w + 1) * _FIELDS)):
            rank = (
                jnp.sum(jnp.where(dest_ids == d, start, 0), dtype=jnp.int32)
                + _field_counts(word, jnp.int32(d - w * _FIELDS))
                - 1
            )
            keep = hits[d] & (rank < cap)
            slot = jnp.where(keep, d * pair_capacity + rank, send_rows)
            plgpu.store(slot_ref.at[pl.ds(pid * block, block), jnp.int32(d)], slot, mask=in_range)
            # The inverse map: each kept (token, destination) writes its token id to its send row. Masked scatter
            # lanes point past the end: the interpreter writes them back with their old value, which must not hit a
            # written row.
            plgpu.store(source_ref.at[slot], rows, mask=keep)
            keeps.append(keep)
    # Send rows no token fills keep the `tokens` sentinel: row d * C + c is filled iff c < min(count[d], cap).
    filled = plgpu.load(filled_ref.at[pl.ds(0, destinations)])
    for j in range(init_chunks):
        row = (pid * init_chunks + j) * block + jnp.arange(block, dtype=jnp.int32)
        dest = row // pair_capacity
        filled_d = jnp.sum(jnp.where(dest[:, None] == dest_ids[None, :], filled[None, :], 0), axis=1, dtype=jnp.int32)
        empty = (row < send_rows) & (row - dest * pair_capacity >= filled_d)
        plgpu.store(source_ref.at[jnp.where(empty, row, send_rows)], jnp.full((block,), tokens, jnp.int32), mask=empty)
    kept = jnp.zeros((block,), jnp.int32)
    for dest in dests:
        for d in range(destinations):
            kept = kept + jnp.where((dest == d) & keeps[d], 1, 0)
    plgpu.store(kept_ref.at[pl.ds(pid, 1)], jnp.sum(kept, keepdims=True, dtype=jnp.int32))


class SendSlots(NamedTuple):
    """Sender-side routing maps of ``ring_filtered``.

    ``slot[t, d]``: token ``t``'s row in the flat [destinations * C] send buffer, or ``destinations * C`` when it sends
    nothing to ``d``. ``source[r]``: the token in send row ``r``, or ``tokens``. ``kept_assignments``: valid
    assignments whose destination row was kept (the sender's drops are the valid assignments minus this).
    """

    slot: Int[Array, "T D"]
    source: Int[Array, "R"]
    kept_assignments: Int[Array, ""]


def send_slots(
    selected: Int[Array, "T K"],
    token_valid: Bool[Array, "T"],
    logical_pair_capacity: Int[Array, ""],
    *,
    local_experts: int,
    destinations: int,
    pair_capacity: int,
    interpret: bool | None = None,
) -> SendSlots:
    """Rank each token per destination shard among the valid tokens that hit it, keep ranks below the capacity.

    Equals ``ep_ring_filtered``'s XLA dispatch maps: ``hit[t, d]`` is whether a valid token has an expert on ``d``, its
    rank is the number of earlier hitting tokens, and kept tokens fill ``d``'s rows ``[d * C, d * C + rank]``.
    """
    tokens = selected.shape[0]
    if not _is_pow2(destinations):
        raise ValueError(f"send_slots needs a power-of-two destination count, got {destinations}")
    interpret = resolve_interpret(interpret)
    block = min(_SEND_BLOCK, _pow2(tokens))
    blocks = pl.cdiv(tokens, block)
    table_rows = _pow2(blocks)
    send_rows = destinations * pair_capacity
    init_chunks = pl.cdiv(pl.cdiv(send_rows, blocks), block)
    selected = jax.lax.stop_gradient(selected).astype(jnp.int32)
    valid = token_valid.astype(jnp.int32)
    cap = jnp.reshape(logical_pair_capacity, (1,)).astype(jnp.int32)
    params = dict(block=block, tokens=tokens, local_experts=local_experts, destinations=destinations)
    compiler_params = plgpu.CompilerParams(num_warps=_NUM_WARPS, num_stages=1)
    # The packed counters are int64, whatever the caller's x64 setting.
    with jax.enable_x64(True):
        counts_shape = jax.ShapeDtypeStruct((table_rows, destinations), jnp.int32)
        counts = pl.pallas_call(
            functools.partial(_send_count_kernel, **params),
            grid=(blocks,),
            in_specs=[pl.no_block_spec, pl.no_block_spec],
            out_specs=pl.no_block_spec,
            out_shape=counts_shape,
            compiler_params=compiler_params,
            cost_estimate=_cost((selected, valid), counts_shape, compares=selected.size * destinations),
            interpret=interpret,
            name="grug_send_slot_counts",
        )(selected, valid)
        scan_shapes = [counts_shape, jax.ShapeDtypeStruct((destinations,), jnp.int32)]
        start, filled = pl.pallas_call(
            functools.partial(_scan_kernel, blocks=blocks, offsets=False),
            grid=(destinations,),
            in_specs=[pl.no_block_spec] * 2,
            out_specs=[pl.no_block_spec] * 2,
            out_shape=scan_shapes,
            compiler_params=compiler_params,
            cost_estimate=_cost((counts, cap), scan_shapes, compares=destinations * counts.size),
            interpret=interpret,
            name="grug_send_slot_scan",
        )(counts, cap)
        slot_shapes = [
            jax.ShapeDtypeStruct((tokens, destinations), jnp.int32),
            jax.ShapeDtypeStruct((send_rows,), jnp.int32),
            jax.ShapeDtypeStruct((blocks,), jnp.int32),
        ]
        slot, source, kept = pl.pallas_call(
            functools.partial(_send_slot_kernel, **params, pair_capacity=pair_capacity, init_chunks=init_chunks),
            grid=(blocks,),
            in_specs=[pl.no_block_spec] * 5,
            out_specs=[pl.no_block_spec] * 3,
            out_shape=slot_shapes,
            compiler_params=compiler_params,
            cost_estimate=_cost(
                (selected, valid, cap, start, filled), slot_shapes, compares=selected.size * destinations
            ),
            interpret=interpret,
            name="grug_send_slots",
        )(selected, valid, cap, start, filled)
    return SendSlots(slot, source, jnp.sum(kept, dtype=jnp.int32))


# ----------------------------------------------------------------------------------------------------------------------
# Receiver: received assignments -> expert rows.


def _received_ids(meta_ref, start, *, block: int, assignments: int, topk: int, pair_capacity: int):
    """Local expert ids of assignments ``[start, start + block)``, read from the received metadata; -1 if none.

    The metadata is the flattened f32 [destinations, C + 1, 2K] all-to-all buffer: assignment ``(d * C + c) * K + k``
    has its id at ``[d, c, K + k]``. Ids are small integers or -1, exact in float32, so truncation equals rounding.
    """
    a = start + jnp.arange(block, dtype=jnp.int32)
    in_range = a < assignments
    row = a // topk
    k = a - row * topk
    d = row // pair_capacity
    offset = (row + d) * (2 * topk) + topk + k
    ids = plgpu.load(meta_ref.at[jnp.where(in_range, offset, 0)], mask=in_range, other=-1.0)
    return ids.astype(jnp.int32), a, in_range


def _expert_count_kernel(meta_ref, counts_ref, *, block, assignments, topk, pair_capacity, groups):
    pid = pl.program_id(0)
    ids, _, _ = _received_ids(
        meta_ref, pid * block, block=block, assignments=assignments, topk=topk, pair_capacity=pair_capacity
    )
    counts = jnp.sum(
        (ids[:, None] == jnp.arange(groups, dtype=jnp.int32)[None, :]).astype(jnp.int32), axis=0, dtype=jnp.int32
    )
    plgpu.store(counts_ref.at[pid, pl.ds(0, groups)], counts)


def _expert_row_kernel(
    meta_ref,
    start_ref,
    accepted_total_ref,
    picked_ref,
    position_ref,
    *,
    block,
    assignments,
    topk,
    pair_capacity,
    groups,
    physical_capacity,
):
    pid = pl.program_id(0)
    ids, a, in_range = _received_ids(
        meta_ref, pid * block, block=block, assignments=assignments, topk=topk, pair_capacity=pair_capacity
    )
    assigned = ids >= 0
    expert = jnp.where(assigned, ids, 0)
    word_of, field = expert // _FIELDS, expert % _FIELDS
    rank = jnp.zeros((block,), jnp.int32)
    for w in range(pl.cdiv(groups, _FIELDS)):
        here = assigned & (word_of == w)
        word = jnp.cumsum(jnp.where(here, _field_bit(field), jnp.int64(0)), dtype=jnp.int64)
        rank = jnp.where(here, _field_counts(word, field), rank)
    slot = plgpu.load(start_ref.at[pid, expert]) + rank - 1
    # Slots fill experts in order and capacity is a prefix of that order, so the accepted slots are those below the
    # accepted total.
    accepted_total = _scalar(accepted_total_ref)
    ok = assigned & (slot < accepted_total)
    plgpu.store(position_ref.at[pl.ds(pid * block, block)], jnp.where(ok, slot, physical_capacity), mask=in_range)
    # Masked scatter lanes point past the end (see _send_slot_kernel).
    plgpu.store(picked_ref.at[jnp.where(ok, slot, physical_capacity)], a, mask=ok)
    # Rows past the accepted total are padding: they read the `assignments` sentinel. Every row is below
    # blocks * block because physical_capacity <= assignments.
    row = pid * block + jnp.arange(block, dtype=jnp.int32)
    pad = (row >= accepted_total) & (row < physical_capacity)
    plgpu.store(
        picked_ref.at[jnp.where(pad, row, physical_capacity)], jnp.full((block,), assignments, jnp.int32), mask=pad
    )


class ExpertRows(NamedTuple):
    """Receiver-side routing maps of ``ring_filtered``.

    ``group_sizes``: accepted rows per local expert. ``dropped``: assignments past the capacity. ``picked[s]``: the
    flat received assignment in expert row ``s``, or ``assignments`` for padding rows. ``position[a]``: the expert row
    of assignment ``a``, or ``physical_capacity`` when it has none.
    """

    group_sizes: Int[Array, "G"]
    dropped: Int[Array, ""]
    picked: Int[Array, "P"]
    position: Int[Array, "A"]


def expert_rows(
    received_metadata: Float[Array, "D C1 K2"],
    logical_capacity: Int[Array, ""],
    *,
    local_experts: int,
    physical_capacity: int,
    interpret: bool | None = None,
) -> ExpertRows:
    """Sort the received assignments by local expert, stable, and keep the first ``logical_capacity`` rows.

    ``received_metadata`` is ``ring_filtered``'s received f32 [destinations, C + 1, 2K] buffer (combine weights in
    columns ``[0, K)``, local expert ids or -1 in ``[K, 2K)``, a trailer row ``C``). Equals
    ``ep_ring._select_local_assignments`` followed by ``ring_filtered``'s sentinel maps.
    """
    destinations, rows_plus_trailer, two_k = received_metadata.shape
    topk, pair_capacity = two_k // 2, rows_plus_trailer - 1
    assignments = destinations * pair_capacity * topk
    if not _is_pow2(local_experts):
        raise ValueError(f"expert_rows needs a power-of-two local expert count, got {local_experts}")
    if physical_capacity > assignments:
        raise ValueError(f"physical_capacity={physical_capacity} exceeds the {assignments} received assignments")
    interpret = resolve_interpret(interpret)
    block = min(_EXPERT_BLOCK, _pow2(assignments))
    blocks = pl.cdiv(assignments, block)
    table_rows = _pow2(blocks)
    meta = jax.lax.stop_gradient(received_metadata).astype(jnp.float32).reshape(-1)
    cap = jnp.reshape(logical_capacity, (1,)).astype(jnp.int32)
    params = dict(block=block, assignments=assignments, topk=topk, pair_capacity=pair_capacity, groups=local_experts)
    compiler_params = plgpu.CompilerParams(num_warps=_NUM_WARPS, num_stages=1)
    with jax.enable_x64(True):
        counts_shape = jax.ShapeDtypeStruct((table_rows, local_experts), jnp.int32)
        counts = pl.pallas_call(
            functools.partial(_expert_count_kernel, **params),
            grid=(blocks,),
            in_specs=[pl.no_block_spec],
            out_specs=pl.no_block_spec,
            out_shape=counts_shape,
            compiler_params=compiler_params,
            cost_estimate=_cost(meta, counts_shape, compares=assignments * local_experts),
            interpret=interpret,
            name="grug_expert_row_counts",
        )(meta)
        scan_shapes = [
            counts_shape,
            jax.ShapeDtypeStruct((local_experts,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ]
        start, group_sizes, dropped, accepted_total = pl.pallas_call(
            functools.partial(_scan_kernel, blocks=blocks, offsets=True),
            grid=(local_experts,),
            in_specs=[pl.no_block_spec] * 2,
            out_specs=[pl.no_block_spec] * 4,
            out_shape=scan_shapes,
            compiler_params=compiler_params,
            cost_estimate=_cost((counts, cap), scan_shapes, compares=local_experts * counts.size),
            interpret=interpret,
            name="grug_expert_row_scan",
        )(counts, cap)
        row_shapes = [
            jax.ShapeDtypeStruct((physical_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((assignments,), jnp.int32),
        ]
        picked, position = pl.pallas_call(
            functools.partial(_expert_row_kernel, **params, physical_capacity=physical_capacity),
            grid=(blocks,),
            in_specs=[pl.no_block_spec] * 3,
            out_specs=[pl.no_block_spec] * 2,
            out_shape=row_shapes,
            compiler_params=compiler_params,
            cost_estimate=_cost((meta, start, accepted_total), row_shapes, compares=assignments * local_experts),
            interpret=interpret,
            name="grug_expert_rows",
        )(meta, start, accepted_total)
    return ExpertRows(group_sizes, dropped[0], picked, position)
