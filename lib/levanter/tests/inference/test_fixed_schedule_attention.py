# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel import DEFAULT_MASK_VALUE
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention.fixed_schedule import (
    FixedScheduleSpec,
    fixed_schedule_attention,
    materialize_contiguous_kv,
)


def _require_four_tpu_devices() -> None:
    if jax.default_backend() != "tpu" or len(jax.devices()) < 4:
        pytest.skip("fixed shared attention requires four TPU devices")


def _fixture(spec: FixedScheduleSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    shape = (spec.max_num_seqs, spec.max_kv_len, 1, spec.head_dim)
    keys = jax.random.normal(jax.random.PRNGKey(17), shape, dtype=jnp.float32) * 0.125
    values = jax.random.normal(jax.random.PRNGKey(19), shape, dtype=jnp.float32) * 0.25
    page_indices = jnp.asarray(
        [[3, 1]] if spec.max_num_seqs == 1 else [[0, 1], [2, 3], [4, 5], [6, 7]],
        dtype=jnp.int32,
    )
    return keys, values, page_indices


def _dense_oracle(
    q_row: jax.Array, pages: jax.Array, page_indices: jax.Array, position: int, window: int | None
) -> jax.Array:
    kv = pages[page_indices].reshape(256, 2, 128)
    keys, values = kv[:, 0], kv[:, 1]
    scores = jnp.einsum("hd,kd->hk", q_row, keys, preferred_element_type=jnp.float32)
    scores = scores * (1.0 / jnp.sqrt(jnp.asarray(128.0, jnp.float32)))
    key_positions = jnp.arange(256)
    allowed = key_positions <= position
    if window is not None:
        allowed &= key_positions > position - window
    scores = jnp.where(allowed[None, :], scores, DEFAULT_MASK_VALUE)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hk,kd->hd", weights, values, preferred_element_type=jnp.float32)


@pytest.mark.parametrize("position", [0, 127, 128])
@pytest.mark.parametrize("window", [None, 128])
def test_fixed_schedule_public_forward_matches_oracle_and_decode(position: int, window: int | None) -> None:
    _require_four_tpu_devices()
    spec = FixedScheduleSpec(sliding_window=window)
    keys, values, page_indices = _fixture(spec)
    pages = materialize_contiguous_kv(keys, values, page_indices, spec=spec)
    query = jax.random.normal(jax.random.PRNGKey(23), (4, 2, 128), dtype=jnp.float32) * 0.125
    q = jnp.broadcast_to(query[:, None, :, :], (4, 256, 2, 128))
    full_positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (4, 256))
    full_valid = jnp.ones((4, 256), dtype=jnp.bool_)
    decode_positions = jnp.full((4, 256), -1, dtype=jnp.int32).at[:, 0].set(position)
    decode_valid = jnp.zeros((4, 256), dtype=jnp.bool_).at[:, 0].set(True)
    with jax.set_mesh(Mesh(jax.devices()[:4], ("data",))):
        full = fixed_schedule_attention(
            q,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            jnp.full((4,), 256, jnp.int32),
            full_positions,
            full_valid,
            spec=spec,
        )
        decode = fixed_schedule_attention(
            q,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            jnp.full((4,), 256 - position, jnp.int32),
            decode_positions,
            decode_valid,
            spec=spec,
        )

    assert jnp.all(jnp.isfinite(full))
    assert full.sharding.spec == P("data", None, None, None)
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(full[:, position], jnp.uint32),
        jax.lax.bitcast_convert_type(decode[:, 0], jnp.uint32),
    )
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(decode[:, 1:], jnp.uint32),
        jax.lax.bitcast_convert_type(jnp.zeros_like(decode[:, 1:]), jnp.uint32),
    )
    oracle = _dense_oracle(q[0, position], pages, page_indices[0], position, window)
    assert jnp.allclose(full[0, position], oracle, atol=5e-3, rtol=5e-3)

    permuted_indices = jnp.asarray([[1, 0], [3, 2], [5, 4], [7, 6]], dtype=jnp.int32)
    permuted_pages = materialize_contiguous_kv(keys, values, permuted_indices, spec=spec)
    with jax.set_mesh(Mesh(jax.devices()[:4], ("data",))):
        permuted = fixed_schedule_attention(
            q,
            permuted_pages,
            jnp.full((4,), 256, jnp.int32),
            permuted_indices,
            jnp.full((4,), 256, jnp.int32),
            full_positions,
            full_valid,
            spec=spec,
        )
    assert jnp.array_equal(full, permuted)


def test_fixed_schedule_public_forward_isolates_data_sharded_neighbor_changes() -> None:
    _require_four_tpu_devices()
    spec = FixedScheduleSpec()
    keys, values, page_indices = _fixture(spec)
    pages = materialize_contiguous_kv(keys, values, page_indices, spec=spec)
    query = jax.random.normal(jax.random.PRNGKey(29), (4, 2, 128), dtype=jnp.float32) * 0.125
    q = jnp.broadcast_to(query[:, None, :, :], (4, 256, 2, 128))
    positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (4, 256))
    valid = jnp.ones((4, 256), dtype=jnp.bool_)
    with jax.set_mesh(Mesh(jax.devices()[:4], ("data",))):
        before = fixed_schedule_attention(
            q,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            jnp.full((4,), 256, jnp.int32),
            positions,
            valid,
            spec=spec,
        )
        changed = q.at[1].set(7.0)
        changed_positions = positions.at[1].set(-1).at[1, :128].set(jnp.arange(128, 256, dtype=jnp.int32))
        changed_valid = valid.at[1].set(False).at[1, :128].set(True)
        after = fixed_schedule_attention(
            changed,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            jnp.asarray([256, 128, 256, 256], dtype=jnp.int32),
            changed_positions,
            changed_valid,
            spec=spec,
        )
    assert jnp.array_equal(before[0], after[0])
    assert not jnp.array_equal(before[1], after[1])
    assert before.sharding.spec == P("data", None, None, None)


def test_fixed_schedule_public_forward_normalizes_compound_explicit_input_shardings() -> None:
    _require_four_tpu_devices()
    spec = FixedScheduleSpec()
    keys, values, page_indices = _fixture(spec)
    pages = materialize_contiguous_kv(keys, values, page_indices, spec=spec)
    query = jax.random.normal(jax.random.PRNGKey(41), (4, 2, 128), dtype=jnp.float32) * 0.125
    q = jnp.broadcast_to(query[:, None, :, :], (4, 256, 2, 128))
    kv_lens = jnp.full((4,), 256, dtype=jnp.int32)
    query_lengths = jnp.full((4,), 256, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (4, 256))
    valid = jnp.ones((4, 256), dtype=jnp.bool_)

    devices = np.asarray(jax.devices()[:4]).reshape(1, 4, 1, 1)
    mesh = Mesh(
        devices,
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    canonical_specs = (
        P("data", None, None, None),
        P(),
        P("data"),
        P("data", None),
        P("data"),
        P("data", None),
        P("data", None),
    )
    compound_specs = (
        P(("replica_dcn", "data", "expert"), None, "model", None),
        P(),
        P(("replica_dcn", "data", "expert")),
        P(("replica_dcn", "data", "expert"), None),
        P(("replica_dcn", "data", "expert")),
        P(("replica_dcn", "data", "expert"), None),
        P(("replica_dcn", "data", "expert"), None),
    )
    args = (q, pages, kv_lens, page_indices, query_lengths, positions, valid)
    canonical_args = tuple(
        jax.device_put(arg, NamedSharding(mesh, partition)) for arg, partition in zip(args, canonical_specs)
    )
    compound_args = tuple(
        jax.device_put(arg, NamedSharding(mesh, partition)) for arg, partition in zip(args, compound_specs)
    )

    with jax.set_mesh(mesh):
        canonical = fixed_schedule_attention(*canonical_args, spec=spec)
        compound = fixed_schedule_attention(*compound_args, spec=spec)

    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(canonical, jnp.uint32),
        jax.lax.bitcast_convert_type(compound, jnp.uint32),
    )
    assert compound.sharding.spec == P("data", None, None, None)


def test_fixed_schedule_public_forward_empty_neighbor_preserves_target() -> None:
    _require_four_tpu_devices()
    spec = FixedScheduleSpec()
    keys, values, page_indices = _fixture(spec)
    pages = materialize_contiguous_kv(keys, values, page_indices, spec=spec)
    query = jax.random.normal(jax.random.PRNGKey(37), (4, 2, 128), dtype=jnp.float32) * 0.125
    q = jnp.broadcast_to(query[:, None, :, :], (4, 256, 2, 128))
    positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (4, 256))
    valid = jnp.ones((4, 256), dtype=jnp.bool_)
    empty_positions = positions.at[1].set(-1)
    empty_valid = valid.at[1].set(False)
    with jax.set_mesh(Mesh(jax.devices()[:4], ("data",))):
        baseline = fixed_schedule_attention(
            q,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            jnp.full((4,), 256, jnp.int32),
            positions,
            valid,
            spec=spec,
        )
        empty_neighbor = fixed_schedule_attention(
            q,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            jnp.asarray([256, 0, 256, 256], dtype=jnp.int32),
            empty_positions,
            empty_valid,
            spec=spec,
        )
    assert jnp.all(jnp.isfinite(empty_neighbor))
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(baseline[0], jnp.uint32),
        jax.lax.bitcast_convert_type(empty_neighbor[0], jnp.uint32),
    )
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(empty_neighbor[1], jnp.uint32),
        jax.lax.bitcast_convert_type(jnp.zeros_like(empty_neighbor[1]), jnp.uint32),
    )


@pytest.mark.parametrize("invalid_map", [[[3, 3]], [[4, 1]]])
def test_materialize_contiguous_kv_poison_invalid_adapter_map(invalid_map: list[list[int]]) -> None:
    spec = FixedScheduleSpec(max_num_seqs=1, num_physical_pages=4)
    keys = jnp.zeros((1, 256, 1, 128), dtype=jnp.float32)
    pages = materialize_contiguous_kv(keys, keys, jnp.asarray(invalid_map, dtype=jnp.int32), spec=spec)
    assert jnp.all(jnp.isnan(pages))


def test_materialize_contiguous_kv_valid_adapter_map_writes_and_preserves_untouched_pages() -> None:
    spec = FixedScheduleSpec(max_num_seqs=1, num_physical_pages=4)
    keys, values, _ = _fixture(spec)
    placement = jnp.asarray([[3, 1]], dtype=jnp.int32)
    pages = materialize_contiguous_kv(keys, values, placement, spec=spec)
    keys_bf16 = keys.astype(jnp.bfloat16)[:, :, 0, :]
    values_bf16 = values.astype(jnp.bfloat16)[:, :, 0, :]
    assert jnp.array_equal(pages[3, :, 0, :], keys_bf16[0, :128])
    assert jnp.array_equal(pages[3, :, 1, :], values_bf16[0, :128])
    assert jnp.array_equal(pages[1, :, 0, :], keys_bf16[0, 128:])
    assert jnp.array_equal(pages[1, :, 1, :], values_bf16[0, 128:])
    assert jnp.array_equal(pages[0], jnp.zeros_like(pages[0]))
    assert jnp.array_equal(pages[2], jnp.zeros_like(pages[2]))


def test_fixed_schedule_public_forward_poison_invalid_metadata() -> None:
    _require_four_tpu_devices()
    spec = FixedScheduleSpec()
    keys, values, page_indices = _fixture(spec)
    pages = materialize_contiguous_kv(keys, values, page_indices, spec=spec)
    query = jax.random.normal(jax.random.PRNGKey(31), (4, 2, 128), dtype=jnp.float32) * 0.125
    q = jnp.broadcast_to(query[:, None, :, :], (4, 256, 2, 128))
    positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (4, 256))
    valid = jnp.ones((4, 256), dtype=jnp.bool_)
    bad_lengths = jnp.asarray([257, 256, 256, 256], dtype=jnp.int32)
    with jax.set_mesh(Mesh(jax.devices()[:4], ("data",))):
        output = fixed_schedule_attention(
            q,
            pages,
            jnp.full((4,), 256, jnp.int32),
            page_indices,
            bad_lengths,
            positions,
            valid,
            spec=spec,
        )
    assert jnp.all(jnp.isnan(output[0]))
