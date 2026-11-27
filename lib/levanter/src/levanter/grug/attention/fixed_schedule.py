# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-schedule TPU attention over replicated KV pages."""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from haliax.partitioning import _get_mesh
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel import DEFAULT_MASK_VALUE
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class FixedScheduleSpec:
    """Static generated-fixture contract for the shared forward."""

    max_num_seqs: int = 4
    max_kv_len: int = 256
    num_physical_pages: int = 8
    pages_per_seq: int = 2
    page_size: int = 128
    num_q_heads: int = 2
    num_kv_heads: int = 1
    head_dim: int = 128
    num_queries_per_block: int = 128
    num_kv_pages_per_block: int = 1
    sm_scale: float = 1.0 / math.sqrt(128.0)
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        if self.max_kv_len != self.pages_per_seq * self.page_size:
            raise ValueError("max_kv_len must equal pages_per_seq * page_size")
        if self.max_num_seqs < 1 or self.pages_per_seq < 1 or self.num_physical_pages < self.pages_per_seq:
            raise ValueError("Fixed attention requires nonempty sequence slots with addressable KV pages")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.head_dim != 128 or self.page_size != 128:
            raise ValueError("the bounded TPUv4 prototype requires head_dim=page_size=128")
        if self.num_queries_per_block != 128 or self.num_kv_pages_per_block != 1:
            raise ValueError("the bounded prototype fixes query blocks to 128 and KV blocks to 1 page")
        if self.sliding_window is not None and not 0 < self.sliding_window <= self.max_kv_len:
            raise ValueError("sliding_window must be within the fixed KV capacity")


def _page_map_contract(
    page_indices: jax.Array,
    spec: FixedScheduleSpec,
    *,
    require_unique: bool,
    num_sequences: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    expected_sequences = spec.max_num_seqs if num_sequences is None else num_sequences
    if page_indices.shape != (expected_sequences, spec.pages_per_seq):
        raise ValueError(f"page_indices must have the fixed [{expected_sequences}, {spec.pages_per_seq}] shape")
    if page_indices.dtype != jnp.int32:
        raise ValueError("page_indices must be int32")

    in_range = jnp.all((page_indices >= 0) & (page_indices < spec.num_physical_pages))
    valid = in_range
    if require_unique:
        ordered = jnp.sort(page_indices.reshape(-1))
        valid = valid & jnp.all(ordered[1:] != ordered[:-1])
    safe = jnp.clip(page_indices, 0, spec.num_physical_pages - 1)
    return safe, valid


def materialize_contiguous_kv(
    keys: jax.Array,
    values: jax.Array,
    page_indices: jax.Array,
    *,
    spec: FixedScheduleSpec = FixedScheduleSpec(),
) -> jax.Array:
    """Pack contiguous K/V into the exact serving page layout.

    Invalid or duplicate adapter write maps are sanitized to page zero before
    any dynamic update and poison the returned cache with BF16 NaNs. A valid
    map must be unique and in range because this adapter writes each logical
    page exactly once. Serving attention itself permits shared physical pages.
    """
    expected = (spec.max_num_seqs, spec.max_kv_len, spec.num_kv_heads, spec.head_dim)
    if keys.shape != expected or values.shape != expected:
        raise ValueError(
            f"keys and values must have the fixed [{spec.max_num_seqs}, {spec.max_kv_len}, "
            f"{spec.num_kv_heads}, {spec.head_dim}] shape"
        )
    if keys.dtype not in (jnp.bfloat16, jnp.float32) or values.dtype != keys.dtype:
        raise ValueError("contiguous keys and values must be BF16 or FP32 with matching dtypes")

    safe_pages, map_valid = _page_map_contract(page_indices, spec, require_unique=True)
    finite = jnp.all(jnp.isfinite(keys)) & jnp.all(jnp.isfinite(values))
    safe_keys = jnp.where(jnp.isfinite(keys), keys, 0).astype(jnp.bfloat16)
    safe_values = jnp.where(jnp.isfinite(values), values, 0).astype(jnp.bfloat16)
    logical = jnp.stack((safe_keys, safe_values), axis=3)
    # Gather complete logical rows before combining page positions and K/V heads.
    # In an Explicit expert mesh, the reshape cannot infer this transfer.
    mesh = _get_mesh()
    if mesh is not None and not mesh.empty:
        logical = reshard(logical, NamedSharding(mesh, P()))
    logical = logical.reshape(-1, spec.page_size, 2 * spec.num_kv_heads, spec.head_dim)
    pages = jnp.zeros(
        (spec.num_physical_pages, spec.page_size, 2 * spec.num_kv_heads, spec.head_dim),
        dtype=jnp.bfloat16,
    )
    if mesh is not None and not mesh.empty:
        pages = reshard(pages, NamedSharding(mesh, P()))
    pages = pages.at[safe_pages.reshape(-1)].set(logical)
    return jnp.where(map_valid & finite, pages, jnp.full_like(pages, jnp.nan))


def _local_fixed_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    query_lengths: jax.Array,
    query_positions: jax.Array,
    query_valid: jax.Array,
    *,
    spec: FixedScheduleSpec,
) -> jax.Array:
    """Run one complete sequence slot inside the explicit data shard."""
    q = q[0]
    query_lengths = query_lengths.reshape((1,))
    query_positions = query_positions[0]
    query_valid = query_valid[0]

    safe_pages, pages_in_range = _page_map_contract(page_indices, spec, require_unique=False, num_sequences=1)
    pages_finite = jnp.all(jnp.isfinite(kv_pages))
    safe_kv_pages = jnp.where(jnp.isfinite(kv_pages), kv_pages, 0).astype(jnp.bfloat16)
    lengths_in_range = (query_lengths >= 0) & (query_lengths <= spec.max_kv_len)
    kv_length_valid = kv_lens == spec.max_kv_len
    safe_length = jnp.clip(query_lengths, 0, spec.max_kv_len)
    safe_kv_lens = jnp.where(kv_length_valid, kv_lens, spec.max_kv_len)
    # The ragged kernel prefetches KV before its no-work predicate. Keep zero
    # as the logical length, but give that kernel one sanitized dummy query;
    # query_valid and valid_metadata still erase it or poison invalid input.
    execution_length = jnp.maximum(safe_length, 1)
    cu_q_lens = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), execution_length))

    row = jnp.arange(spec.max_kv_len, dtype=jnp.int32)
    in_span = row < safe_length[0]
    expected_positions = spec.max_kv_len - safe_length[0] + row
    positions_valid = jnp.all(~query_valid | (in_span & (query_positions == expected_positions)))
    valid_metadata = (
        jnp.all(lengths_in_range)
        & jnp.all(kv_length_valid)
        & pages_in_range
        & pages_finite
        & jnp.all(jnp.isfinite(q))
        & positions_valid
    )
    safe_q = jnp.where(jnp.isfinite(q), q, 0).astype(jnp.float32)

    # Four KV heads fit in v4 VMEM. Launch the tail with its own head width
    # so a five-head model runs one full tile and one single-head tile.
    kv_heads_per_call = 4
    q_per_kv = spec.num_q_heads // spec.num_kv_heads

    def attend(query, pages):
        # Keep DEFAULT dot precision independent of the caller's context.
        with jax.default_matmul_precision("default"):
            return ragged_paged_attention(
                query,
                pages,
                safe_kv_lens,
                safe_pages,
                cu_q_lens,
                jnp.asarray([1], dtype=jnp.int32),
                sm_scale=spec.sm_scale,
                sliding_window=spec.sliding_window,
                soft_cap=None,
                mask_value=DEFAULT_MASK_VALUE,
                k_scale=None,
                v_scale=None,
                num_kv_pages_per_block=spec.num_kv_pages_per_block,
                num_queries_per_block=spec.num_queries_per_block,
                vmem_limit_bytes=None,
            )

    outputs = []
    for start in range(0, spec.num_kv_heads, kv_heads_per_call):
        stop = min(start + kv_heads_per_call, spec.num_kv_heads)
        num_heads = stop - start
        query = safe_q[:, start * q_per_kv : stop * q_per_kv]
        pages = safe_kv_pages[:, :, 2 * start : 2 * stop]
        padding = (1 << (num_heads - 1).bit_length()) - num_heads
        if padding:
            query = jnp.pad(query, ((0, 0), (0, padding * q_per_kv), (0, 0)))
            pages = jnp.pad(pages, ((0, 0), (0, 0), (0, 2 * padding), (0, 0)))
        outputs.append(attend(query, pages)[:, : num_heads * q_per_kv])
    output = jnp.concatenate(outputs, axis=1)
    output = jnp.where(query_valid[:, None, None], output, 0.0)
    output = jnp.where(valid_metadata, output, jnp.nan)
    return output[None, ...]


def fixed_attention_row_axis(num_slots: int) -> str:
    """Return the explicit mesh axis that holds one logical sequence per device."""
    mesh = _get_mesh()
    if mesh is not None and not mesh.empty and mesh.size == num_slots:
        for axis in ("data", "expert"):
            if (
                mesh.shape.get(axis) == num_slots
                and mesh.axis_types[mesh.axis_names.index(axis)] == jax.sharding.AxisType.Explicit
            ):
                return axis
    raise ValueError("Fixed attention requires one explicit data or expert axis spanning the sequence slots")


def fixed_schedule_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    query_lengths: jax.Array,
    query_positions: jax.Array,
    query_valid: jax.Array,
    *,
    spec: FixedScheduleSpec = FixedScheduleSpec(),
) -> jax.Array:
    """Run one sequence slot per device using the same KV reduction schedule."""
    if q.shape != (spec.max_num_seqs, spec.max_kv_len, spec.num_q_heads, spec.head_dim):
        raise ValueError("Query shape must match the fixed attention spec")
    if q.dtype != jnp.float32:
        raise ValueError("q must be float32 at the explicit mixed-precision boundary")
    if kv_pages.shape != (
        spec.num_physical_pages,
        spec.page_size,
        2 * spec.num_kv_heads,
        spec.head_dim,
    ):
        raise ValueError("KV page shape must match the fixed attention spec")
    if kv_pages.dtype != jnp.bfloat16:
        raise ValueError("kv_pages must be BF16 at the cache boundary")
    if kv_lens.shape != (spec.max_num_seqs,) or query_lengths.shape != (spec.max_num_seqs,):
        raise ValueError("Lengths must have one value per sequence slot")
    if query_positions.shape != (spec.max_num_seqs, spec.max_kv_len) or query_valid.shape != query_positions.shape:
        raise ValueError("Query metadata must match the sequence slot and position dimensions")
    if (
        kv_lens.dtype != jnp.int32
        or query_lengths.dtype != jnp.int32
        or query_positions.dtype != jnp.int32
        or query_valid.dtype != jnp.bool_
    ):
        raise ValueError("lengths and positions must be int32 and query_valid must be bool")
    if page_indices.shape != (spec.max_num_seqs, spec.pages_per_seq) or page_indices.dtype != jnp.int32:
        raise ValueError("Page indices must be int32 and match the fixed attention spec")

    axis = fixed_attention_row_axis(spec.max_num_seqs)
    mesh = _get_mesh()
    in_specs = (P(axis, None, None, None), P(), P(axis), P(axis, None), P(axis), P(axis, None), P(axis, None))
    shard = jax.shard_map(
        lambda q_, pages_, lens_, indices_, lengths_, positions_, valid_: _local_fixed_attention(
            q_, pages_, lens_, indices_, lengths_, positions_, valid_, spec=spec
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis, None, None, None),
        check_vma=False,
    )
    args = (q, kv_pages, kv_lens, page_indices, query_lengths, query_positions, query_valid)
    args = tuple(jax.sharding.reshard(arg, NamedSharding(mesh, partition)) for arg, partition in zip(args, in_specs))
    return shard(*args)
