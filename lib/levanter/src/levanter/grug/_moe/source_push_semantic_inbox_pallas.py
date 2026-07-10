# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic-plan adapter for the stable source-push inbox W13 kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe import source_push_inbox
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_META_FIELDS,
    SourcePushSemanticPlan,
    SourcePushSemanticQueueMetadata,
    _exclusive_cumsum_jax,
)


SEMANTIC_INBOX_PACK_HIDDEN_BLOCK = 256
SEMANTIC_INBOX_PACK_METADATA_ELEMENTS = 128


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticInboxLayout:
    """Fixed-capacity source-padded expert row layout shared by forward/backward."""

    expert_base: Int[Array, "Dst E"]
    src_base_by_expert: Int[Array, "Dst S E"]
    rounded_rows_per_expert: Int[Array, "Dst E"]
    transport_rows_by_src_dst_expert: Int[Array, "S Dst E"]
    valid: Bool[Array, "Dst E C"]
    overflow_rows: Int[Array, ""]
    rows_per_expert_capacity: int = field(metadata={"static": True})


class SourcePushSemanticInboxKernelInputs(NamedTuple):
    """Block-aligned inputs for the stable packed source-push inbox kernel."""

    packed_x: Float[Array, "S DstOrd Q M H"]
    send_meta: Int[Array, "S DstOrd Q F"]
    recv_meta: Int[Array, "Dst SrcOrd Q F"]
    layout: SourcePushSemanticInboxLayout


class SourcePushSemanticInboxMetadata(NamedTuple):
    """Small queue metadata used by source-local payload packing and W13."""

    token_ids: Int[Array, "S DstOrd Q M"]
    valid_mask: Bool[Array, "S DstOrd Q M"]
    send_meta: Int[Array, "S DstOrd Q F"]
    recv_meta: Int[Array, "Dst SrcOrd Q F"]
    layout: SourcePushSemanticInboxLayout


class SourcePushSemanticInboxW13Result(NamedTuple):
    """Source-padded destination expert-major W13 outputs and reusable layout."""

    z: Float[Array, "Dst E C twoI"]
    h: Float[Array, "Dst E C I"]
    valid: Bool[Array, "Dst E C"]
    layout: SourcePushSemanticInboxLayout
    packed_x: Float[Array, "S DstOrd Q M H"]
    recv_x: Float[Array, "Dst S Slots M H"]


def source_push_semantic_inbox_layout_jax(
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    *,
    rows_per_expert_capacity: int,
) -> SourcePushSemanticInboxLayout:
    """Build the canonical source-padded expert layout without queue payloads.

    ``src_base_by_expert[dst, src, expert]`` is the source's row base within
    the destination expert's fixed-capacity slice. All values are computed
    with JAX operations so callers can keep layout construction inside JIT.
    """

    source_count, destination_count, experts_per_rank = plan.xcounts.shape
    row_block = queue.return_row_block
    rounded_counts = ((plan.xcounts + row_block - 1) // row_block) * row_block
    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    rounded_rows_per_expert = jnp.sum(rounded_counts, axis=0, dtype=jnp.int32)
    padded_expert_base = jnp.broadcast_to(
        jnp.arange(experts_per_rank, dtype=jnp.int32)[None, :] * rows_per_expert_capacity,
        (destination_count, experts_per_rank),
    )
    padded_src_base_by_expert = jnp.transpose(_exclusive_cumsum_jax(rounded_counts, axis=0), (1, 0, 2))

    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None]
    actual_dst = (source_index + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(queue.local_expert, 0)
    source_padded_row_start = (
        padded_src_base_by_expert.at[actual_dst, source_index, safe_expert].get() + queue.local_row_start
    )
    transport_valid_rows = jnp.clip(
        rows_per_expert_capacity - source_padded_row_start,
        0,
        queue.valid_rows,
    )

    entry_expert = jax.nn.one_hot(safe_expert, experts_per_rank, dtype=jnp.int32)
    transport_rows_by_dst_ordinal = jnp.sum(
        entry_expert * transport_valid_rows[..., None],
        axis=2,
        dtype=jnp.int32,
    )
    actual_destination = jnp.arange(destination_count, dtype=jnp.int32)[None, :]
    source_for_destination = jnp.arange(source_count, dtype=jnp.int32)[:, None]
    destination_ordinal = (actual_destination - source_for_destination) % destination_count
    transport_rows_by_src_dst_expert = transport_rows_by_dst_ordinal.at[
        source_for_destination,
        destination_ordinal,
    ].get()

    expert_row = jnp.arange(rows_per_expert_capacity, dtype=jnp.int32)[None, None, :, None]
    source_row_start = jnp.transpose(padded_src_base_by_expert, (0, 2, 1))[:, :, None, :]
    source_valid_rows = jnp.transpose(transport_rows_by_src_dst_expert, (1, 2, 0))[:, :, None, :]
    valid = jnp.any(
        (expert_row >= source_row_start) & (expert_row < source_row_start + source_valid_rows),
        axis=3,
    )
    overflow_rows = jnp.sum(
        jnp.maximum(rounded_rows_per_expert - rows_per_expert_capacity, 0),
        dtype=jnp.int32,
    )
    return SourcePushSemanticInboxLayout(
        expert_base=padded_expert_base,
        src_base_by_expert=padded_src_base_by_expert,
        rounded_rows_per_expert=rounded_rows_per_expert,
        transport_rows_by_src_dst_expert=transport_rows_by_src_dst_expert,
        valid=valid,
        overflow_rows=overflow_rows,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )


def source_push_semantic_inbox_metadata_jax(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    *,
    rows_per_expert_capacity: int,
) -> SourcePushSemanticInboxMetadata:
    """Build block-aligned queue indices and layout without gathering payloads."""

    _validate_semantic_inbox_metadata(x, plan, queue)
    source_count, destination_count, _experts_per_rank = plan.xcounts.shape
    row_block = queue.return_row_block
    layout = source_push_semantic_inbox_layout_jax(
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None]
    actual_dst = (source_index + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(queue.local_expert, 0)
    source_padded_row_start = (
        layout.src_base_by_expert.at[actual_dst, source_index, safe_expert].get() + queue.local_row_start
    )
    transport_valid_rows = jnp.clip(
        rows_per_expert_capacity - source_padded_row_start,
        0,
        queue.valid_rows,
    )
    padded_row_start = layout.expert_base.at[actual_dst, safe_expert].get() + source_padded_row_start
    send_source = jnp.broadcast_to(source_index, queue.local_expert.shape)
    entry_valid = transport_valid_rows > 0
    send_meta = jnp.stack(
        (
            jnp.where(entry_valid, send_source, 0),
            queue.local_expert,
            jnp.where(entry_valid, padded_row_start, 0),
            transport_valid_rows,
        ),
        axis=-1,
    ).astype(jnp.int32)

    dst_index = jnp.arange(destination_count, dtype=jnp.int32)[:, None]
    recv_src_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    recv_source = (dst_index + recv_src_ordinal) % source_count
    recv_dst_ordinal = (-recv_src_ordinal) % destination_count
    recv_meta = send_meta.at[recv_source, recv_dst_ordinal].get()

    queue_row = jnp.arange(row_block, dtype=jnp.int32)[None, None, None, :]
    pair_row = (
        plan.pair_expert_base.at[source_index, actual_dst, safe_expert].get()[..., None]
        + queue.local_row_start[..., None]
        + queue_row
    )
    safe_pair_row = jnp.minimum(pair_row, plan.assignment_ids.shape[-1] - 1)
    token_id = plan.token_ids.at[source_index[..., None], actual_dst[..., None], safe_pair_row].get()
    row_valid = queue_row < transport_valid_rows[..., None]
    row_valid &= queue.local_expert[..., None] >= 0
    row_valid &= plan.valid_mask.at[source_index[..., None], actual_dst[..., None], safe_pair_row].get()
    token_id = jnp.where(row_valid, token_id, 0).astype(jnp.int32)

    assert token_id.shape == (source_count, destination_count, queue.entries_per_dst, row_block)
    assert send_meta.shape[-1] == SOURCE_PUSH_META_FIELDS
    return SourcePushSemanticInboxMetadata(
        token_ids=token_id,
        valid_mask=row_valid,
        send_meta=send_meta,
        recv_meta=recv_meta,
        layout=layout,
    )


def source_push_semantic_inbox_kernel_inputs_jax(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    *,
    rows_per_expert_capacity: int,
) -> SourcePushSemanticInboxKernelInputs:
    """Reference JAX queue pack used by CPU/interpret correctness paths."""

    metadata = source_push_semantic_inbox_metadata_jax(
        x,
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    source_index = jnp.arange(x.shape[0], dtype=jnp.int32)[:, None, None, None]
    packed_x = x.at[source_index, metadata.token_ids].get()
    packed_x = jnp.where(metadata.valid_mask[..., None], packed_x, jnp.zeros((), dtype=x.dtype))
    return SourcePushSemanticInboxKernelInputs(
        packed_x=packed_x,
        send_meta=metadata.send_meta,
        recv_meta=metadata.recv_meta,
        layout=metadata.layout,
    )


def source_push_semantic_inbox_pack_pallas_mgpu(
    x: Float[Array, "S T H"],
    token_ids: Int[Array, "S DstOrd Q M"],
    valid_mask: Bool[Array, "S DstOrd Q M"],
    *,
    mesh: Mesh | AbstractMesh | None = None,
    interpret: bool = False,
) -> Float[Array, "S DstOrd Q M H"]:
    """Pack semantic queue payloads with a source-local rectangular copy kernel."""

    _validate_semantic_inbox_pack_request(x, token_ids, valid_mask, interpret=interpret)
    if interpret and mesh is None:
        hidden_block = _semantic_inbox_pack_interpret_hidden_block(x.shape[-1])
        return jnp.stack(
            [
                _source_push_semantic_inbox_pack_local_pallas_call(
                    x[source],
                    token_ids[source],
                    valid_mask[source],
                    hidden_block=hidden_block,
                    interpret=True,
                )
                for source in range(x.shape[0])
            ],
            axis=0,
        )

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic inbox packing requires a GPU backend")
    if mesh is None:
        mesh = jax.sharding.get_abstract_mesh()
        if mesh.empty:
            raise ValueError("mesh is required for sharded semantic inbox packing")
    mesh_source_count = mesh.shape[source_push_inbox.AXIS]
    if mesh_source_count != x.shape[0]:
        raise ValueError(
            f"semantic inbox pack requires one source per {source_push_inbox.AXIS} rank; "
            f"got {x.shape[0]} sources and mesh size {mesh_source_count}"
        )

    hidden_block = (
        _semantic_inbox_pack_interpret_hidden_block(x.shape[-1])
        if interpret
        else _semantic_inbox_pack_hidden_block(x.shape[-1])
    )

    def local_fn(
        x_local: Float[Array, "1 T H"],
        token_ids_local: Int[Array, "1 DstOrd Q M"],
        valid_mask_local: Bool[Array, "1 DstOrd Q M"],
    ) -> Float[Array, "1 DstOrd Q M H"]:
        packed_local = _source_push_semantic_inbox_pack_local_pallas_call(
            x_local[0],
            token_ids_local[0],
            valid_mask_local[0],
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return packed_local[None, ...]

    x = jax.sharding.reshard(
        x,
        NamedSharding(mesh, P(source_push_inbox.AXIS, None, None)),
    )
    token_ids = jax.sharding.reshard(
        token_ids,
        NamedSharding(mesh, P(source_push_inbox.AXIS, None, None, None)),
    )
    valid_mask = jax.sharding.reshard(
        valid_mask,
        NamedSharding(mesh, P(source_push_inbox.AXIS, None, None, None)),
    )
    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(source_push_inbox.AXIS, None, None),
            P(source_push_inbox.AXIS, None, None, None),
            P(source_push_inbox.AXIS, None, None, None),
        ),
        out_specs=P(source_push_inbox.AXIS, None, None, None, None),
        check_vma=False,
    )(x, token_ids, valid_mask)


def _source_push_semantic_inbox_pack_local_pallas_call(
    x: Float[Array, "T H"],
    token_ids: Int[Array, "DstOrd Q M"],
    valid_mask: Bool[Array, "DstOrd Q M"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "DstOrd Q M H"]:
    destination_count, entries_per_dst, row_block = token_ids.shape
    metadata_entry_block = _semantic_inbox_pack_metadata_entry_block(row_block, interpret=interpret)
    if entries_per_dst % metadata_entry_block:
        raise ValueError(
            "semantic inbox pack metadata groups must divide entries_per_dst; "
            f"got entries_per_dst={entries_per_dst}, metadata_entry_block={metadata_entry_block}"
        )
    if not interpret and metadata_entry_block * row_block % SEMANTIC_INBOX_PACK_METADATA_ELEMENTS:
        raise ValueError(
            "semantic inbox pack metadata tile must contain a multiple of 128 elements; "
            f"got {metadata_entry_block * row_block}"
        )

    output_shape = jax.ShapeDtypeStruct((*token_ids.shape, x.shape[-1]), x.dtype)
    valid_i = valid_mask.astype(jnp.int32)
    gmem = pl.BlockSpec(memory_space=mgpu.GMEM)
    kernel = _make_source_push_semantic_inbox_pack_kernel(
        row_block=row_block,
        metadata_entry_block=metadata_entry_block,
        hidden_block=hidden_block,
        output_dtype=x.dtype,
    )
    return pl.pallas_call(
        kernel,
        in_specs=(gmem, gmem, gmem),
        out_specs=gmem,
        out_shape=output_shape,
        grid=(destination_count, entries_per_dst // metadata_entry_block, x.shape[-1] // hidden_block),
        interpret=interpret,
        name="source_push_semantic_inbox_pack_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_source_push_semantic_inbox_pack_cost_estimate(
            x,
            token_ids,
            valid_i,
            output_shape,
            hidden_block=hidden_block,
        ),
    )(x, token_ids, valid_i)


def _make_source_push_semantic_inbox_pack_kernel(
    *,
    row_block: int,
    metadata_entry_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        x_ref: Float[pl.Ref, "T H"],
        token_ids_ref: Int[pl.Ref, "DstOrd Q M"],
        valid_mask_ref: Int[pl.Ref, "DstOrd Q M"],
        packed_ref: Float[pl.Ref, "DstOrd Q M H"],
    ) -> None:
        destination = pl.program_id(0)
        entry_group = pl.program_id(1)
        hidden_start = pl.program_id(2) * hidden_block
        entry_group_start = entry_group * metadata_entry_block
        zero = jnp.zeros((hidden_block,), dtype=output_dtype)

        @pl.loop(0, metadata_entry_block)
        def _entry_loop(metadata_entry) -> None:
            entry = entry_group_start + metadata_entry

            @pl.loop(0, row_block)
            def _row_loop(row) -> None:
                valid = valid_mask_ref[destination, entry, row] != 0
                safe_token = jnp.maximum(token_ids_ref[destination, entry, row], 0)
                x_tile = x_ref[safe_token, pl.ds(hidden_start, hidden_block)]
                packed_ref[destination, entry, row, pl.ds(hidden_start, hidden_block)] = jnp.where(
                    valid,
                    x_tile.astype(output_dtype),
                    zero,
                )

    return kernel


def _source_push_semantic_inbox_pack_reference(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
) -> Array:
    packed = x.at[jnp.maximum(token_ids, 0)].get()
    return jnp.where(valid_mask[..., None], packed, jnp.zeros((), dtype=x.dtype))


def _source_push_semantic_inbox_pack_cost_estimate(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
    output_shape: jax.ShapeDtypeStruct,
    *,
    hidden_block: int,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(token_ids.shape, token_ids.dtype),
        jax.ShapeDtypeStruct(valid_mask.shape, valid_mask.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_semantic_inbox_pack_reference, *input_specs)
    payload_bytes = math.prod(output_shape.shape) * jnp.dtype(output_shape.dtype).itemsize
    metadata_bytes = (
        math.prod(token_ids.shape) * jnp.dtype(token_ids.dtype).itemsize
        + math.prod(valid_mask.shape) * jnp.dtype(valid_mask.dtype).itemsize
    )
    hidden_tiles = x.shape[-1] // hidden_block
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        # Every rectangular output element is gathered from x and written once.
        # Token/valid metadata is reloaded by each independently scheduled H tile.
        bytes_accessed=2 * payload_bytes + hidden_tiles * metadata_bytes,
        remote_bytes_transferred=0,
    )


def _semantic_inbox_pack_hidden_block(hidden_dim: int) -> int:
    if hidden_dim % SEMANTIC_INBOX_PACK_HIDDEN_BLOCK == 0:
        return SEMANTIC_INBOX_PACK_HIDDEN_BLOCK
    if hidden_dim % 128 == 0:
        return 128
    raise ValueError(f"semantic inbox Pallas pack hidden dim must be divisible by 128, got {hidden_dim}")


def _semantic_inbox_pack_interpret_hidden_block(hidden_dim: int) -> int:
    """Choose a bounded interpreter tile that exactly divides the hidden dimension."""

    return math.gcd(hidden_dim, SEMANTIC_INBOX_PACK_HIDDEN_BLOCK)


def _semantic_inbox_pack_metadata_entry_block(row_block: int, *, interpret: bool) -> int:
    """Group B64 entries exactly as production while keeping tiny interpret tests cheap."""

    if interpret and row_block < 64:
        return 1
    return math.ceil(SEMANTIC_INBOX_PACK_METADATA_ELEMENTS / row_block)


def source_push_semantic_inbox_w13_pallas_mgpu(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    *,
    config: source_push_inbox.PushInboxConfig,
    mesh: Mesh | None = None,
    interpret: bool = False,
) -> SourcePushSemanticInboxW13Result:
    """Run stable packed source-push transport/W13 for semantic metadata.

    Production execution reuses ``source_push_inbox._sharded_w13_h_kernel``.
    The kernel writes source-padded preactivations so every WGMMA store remains
    block aligned. Experts occupy equal-capacity contiguous slices, allowing a
    direct reshape of the flat output to expert-major ``z`` followed by only
    pointwise SwiGLU and validity masking.

    Production packs queue payloads with a source-local Pallas copy custom call
    before invoking the reused W13 kernel. ``packed_x`` remains the complete
    source queue residual for W13 backward. The reused kernel's ``recv_x``
    output contains only the bounded rolling inbox slots and is not a complete
    route checkpoint when entries exceed slots.

    ``interpret=True`` uses the Pallas interpreter for queue packing and a JAX
    simulation for transport/W13 because the reused W13 kernel contains Mosaic
    GPU communication and WGMMA operations.
    """

    _validate_semantic_inbox_request(x, w_gate_up, plan, queue, config)
    rows_per_expert_capacity = config.hidden_rows_per_rank // config.experts_per_rank
    metadata = source_push_semantic_inbox_metadata_jax(
        x,
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    if interpret:
        packed_x = source_push_semantic_inbox_pack_pallas_mgpu(
            x,
            metadata.token_ids,
            metadata.valid_mask,
            interpret=True,
        )
        kernel_inputs = SourcePushSemanticInboxKernelInputs(
            packed_x=packed_x,
            send_meta=metadata.send_meta,
            recv_meta=metadata.recv_meta,
            layout=metadata.layout,
        )
        recv_x, padded_z = _source_push_semantic_inbox_padded_w13_reference_jax(
            kernel_inputs,
            w_gate_up,
            queue,
            padded_rows=config.hidden_rows_per_rank,
            inbox_slots=config.inbox_slots,
        )
    else:
        if mesh is None:
            raise ValueError("mesh is required for source-push inbox MGPU execution")
        packed_x = source_push_semantic_inbox_pack_pallas_mgpu(
            x,
            metadata.token_ids,
            metadata.valid_mask,
            mesh=mesh,
        )
        kernel = source_push_inbox._sharded_w13_h_kernel(mesh, config, use_exact_expert_major=False)
        source_sharding = NamedSharding(mesh, P(source_push_inbox.AXIS, None, None, None, None))
        source_metadata_sharding = NamedSharding(mesh, P(source_push_inbox.AXIS, None, None, None))
        destination_metadata_2d_sharding = NamedSharding(mesh, P(source_push_inbox.AXIS, None))
        destination_metadata_3d_sharding = NamedSharding(mesh, P(source_push_inbox.AXIS, None, None))
        destination_weights_sharding = NamedSharding(mesh, P(source_push_inbox.AXIS, None, None, None))
        recv_x, padded_z = kernel(
            jax.sharding.reshard(packed_x, source_sharding),
            jax.sharding.reshard(metadata.send_meta, source_metadata_sharding),
            jax.sharding.reshard(metadata.recv_meta, source_metadata_sharding),
            jax.sharding.reshard(metadata.layout.expert_base, destination_metadata_2d_sharding),
            jax.sharding.reshard(metadata.layout.src_base_by_expert, destination_metadata_3d_sharding),
            jax.sharding.reshard(w_gate_up, destination_weights_sharding),
        )

    z = padded_z.reshape(
        plan.assignment_ids.shape[1],
        plan.xcounts.shape[-1],
        rows_per_expert_capacity,
        padded_z.shape[-1],
    )
    valid = metadata.layout.valid
    z = jnp.where(valid[..., None], z, jnp.zeros((), dtype=z.dtype))
    intermediate_dim = z.shape[-1] // 2
    gate = z[..., :intermediate_dim].astype(jnp.float32)
    up = z[..., intermediate_dim:].astype(jnp.float32)
    h = jax.nn.silu(gate) * up
    h = jnp.where(valid[..., None], h, jnp.zeros((), dtype=h.dtype))
    return SourcePushSemanticInboxW13Result(
        z=z,
        h=h,
        valid=valid,
        layout=metadata.layout,
        packed_x=packed_x,
        recv_x=recv_x,
    )


def _source_push_semantic_inbox_padded_w13_reference_jax(
    kernel_inputs: SourcePushSemanticInboxKernelInputs,
    w_gate_up: Float[Array, "Dst E H twoI"],
    queue: SourcePushSemanticQueueMetadata,
    *,
    padded_rows: int,
    inbox_slots: int,
) -> tuple[Float[Array, "Dst S Slots M H"], Float[Array, "Dst rows twoI"]]:
    source_count, destination_count, entries_per_dst, row_block, _hidden_dim = kernel_inputs.packed_x.shape
    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None]
    actual_dst = (source_index + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(queue.local_expert, 0)
    queue_weights = w_gate_up.at[actual_dst, safe_expert].get()
    queue_z = jnp.einsum(
        "sdqmh,sdqho->sdqmo",
        kernel_inputs.packed_x.astype(jnp.float32),
        queue_weights.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    queue_row = jnp.arange(row_block, dtype=jnp.int32)[None, None, None, :]
    flat_row = kernel_inputs.send_meta[..., 2, None] + queue_row
    row_valid = queue_row < kernel_inputs.send_meta[..., 3, None]
    queue_z = jnp.where(row_valid[..., None], queue_z, jnp.zeros((), dtype=queue_z.dtype))
    padded_z = jnp.zeros((destination_count, padded_rows, w_gate_up.shape[-1]), dtype=queue_z.dtype)
    scatter_row = jnp.where(row_valid, flat_row, padded_rows)
    padded_z = padded_z.at[
        jnp.broadcast_to(actual_dst[..., None], scatter_row.shape),
        scatter_row,
    ].set(queue_z, mode="drop")

    recv_x = jnp.zeros(
        (destination_count, source_count, inbox_slots, row_block, kernel_inputs.packed_x.shape[-1]),
        dtype=kernel_inputs.packed_x.dtype,
    )
    source = jnp.arange(source_count, dtype=jnp.int32)[:, None]
    actual_dst_2d = jnp.broadcast_to(actual_dst[..., 0], (source_count, destination_count))
    source_2d = jnp.broadcast_to(source, actual_dst_2d.shape)
    for entry in range(entries_per_dst):
        entry_valid = kernel_inputs.send_meta[..., entry, 3] > 0
        previous = recv_x.at[actual_dst_2d, source_2d, entry % inbox_slots].get()
        entry_x = jnp.where(entry_valid[..., None, None], kernel_inputs.packed_x[..., entry, :, :], previous)
        recv_x = recv_x.at[actual_dst_2d, source_2d, entry % inbox_slots].set(entry_x)
    return recv_x, padded_z


def _validate_semantic_inbox_metadata(
    x: Array,
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    source_count, destination_count, _experts_per_rank = plan.xcounts.shape
    if source_count != destination_count:
        raise ValueError(f"semantic source/destination dims must match, got {plan.xcounts.shape}")
    if x.shape[:2] != (source_count, plan.tokens_per_source):
        raise ValueError(
            f"x leading shape {x.shape[:2]} must match semantic sources/tokens "
            f"{(source_count, plan.tokens_per_source)}"
        )
    expected_queue_shape = (source_count, destination_count, queue.entries_per_dst)
    for name, value in (
        ("local_expert", queue.local_expert),
        ("local_row_start", queue.local_row_start),
        ("valid_rows", queue.valid_rows),
    ):
        if value.shape != expected_queue_shape:
            raise ValueError(f"queue {name} shape {value.shape} must be {expected_queue_shape}")
    if queue.return_row_block <= 0:
        raise ValueError(f"queue return_row_block must be positive, got {queue.return_row_block}")


def _validate_semantic_inbox_pack_request(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
    *,
    interpret: bool,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if token_ids.ndim != 4:
        raise ValueError(f"token_ids must have shape [source, destination, entry, row], got {token_ids.shape}")
    if token_ids.shape != valid_mask.shape:
        raise ValueError(f"token_ids shape {token_ids.shape} must match valid_mask shape {valid_mask.shape}")
    if token_ids.shape[0] != x.shape[0]:
        raise ValueError(f"token_ids source dim {token_ids.shape[0]} must match x source dim {x.shape[0]}")
    if not jnp.issubdtype(token_ids.dtype, jnp.integer):
        raise ValueError(f"token_ids must have an integer dtype, got {token_ids.dtype}")
    if valid_mask.dtype != jnp.bool_:
        raise ValueError(f"valid_mask must have boolean dtype, got {valid_mask.dtype}")
    if x.shape[-1] <= 0:
        raise ValueError(f"x hidden dim must be positive, got {x.shape[-1]}")
    if not interpret:
        _semantic_inbox_pack_hidden_block(x.shape[-1])


def _validate_semantic_inbox_request(
    x: Array,
    w_gate_up: Array,
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    config: source_push_inbox.PushInboxConfig,
) -> None:
    _validate_semantic_inbox_metadata(x, plan, queue)
    config.validate()
    source_count, destination_count, experts_per_rank = plan.xcounts.shape
    expected_weight_shape = (destination_count, experts_per_rank, x.shape[-1])
    if w_gate_up.ndim != 4 or w_gate_up.shape[:3] != expected_weight_shape:
        raise ValueError(
            f"w_gate_up leading shape {w_gate_up.shape[:3]} must be {expected_weight_shape}, got {w_gate_up.shape}"
        )
    if w_gate_up.shape[-1] % 2:
        raise ValueError(f"w_gate_up output dim must be even, got {w_gate_up.shape[-1]}")
    expected_config = (
        source_count,
        queue.entries_per_dst,
        queue.return_row_block,
        x.shape[-1],
        w_gate_up.shape[-1] // 2,
        experts_per_rank,
    )
    actual_config = (
        config.ep_size,
        config.entries_per_rank,
        config.block_m,
        config.hidden_dim,
        config.intermediate_dim,
        config.experts_per_rank,
    )
    if actual_config != expected_config:
        raise ValueError(
            "inbox config (ep_size, entries_per_rank, block_m, hidden_dim, intermediate_dim, "
            f"experts_per_rank) must be {expected_config}, got {actual_config}"
        )
    if config.hidden_rows_per_rank % experts_per_rank:
        raise ValueError(
            "flat inbox output rows must divide evenly across experts; "
            f"got hidden_rows_per_rank={config.hidden_rows_per_rank}, experts_per_rank={experts_per_rank}"
        )
    rows_per_expert_capacity = config.hidden_rows_per_rank // experts_per_rank
    if rows_per_expert_capacity % config.block_m:
        raise ValueError(
            "fixed expert row capacity must be block_m aligned; "
            f"got rows_per_expert_capacity={rows_per_expert_capacity}, block_m={config.block_m}"
        )
    if x.dtype != jnp.bfloat16 or w_gate_up.dtype != jnp.bfloat16:
        raise ValueError(
            f"stable source-push inbox W13 requires bfloat16 x and weights, got {x.dtype} and {w_gate_up.dtype}"
        )
