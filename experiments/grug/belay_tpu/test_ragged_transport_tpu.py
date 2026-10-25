# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check lossless transport permutations without training or optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.ragged_transport_tpu import (
    compact_segments,
    gather_rows,
    pack_scalar_peers,
    restore_segments,
    unpack_scalar_peers,
)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("width", [1, 16, 256])
@pytest.mark.parametrize("case", ["gaps", "empty", "full"])
def test_segment_transport_preserves_values_and_unwritten_rows(dtype, width, case):
    rng = np.random.RandomState(101)
    rows = 131
    if case == "empty":
        offsets, sizes = np.array([7, 2, 0], np.int32), np.zeros(3, np.int32)
    elif case == "full":
        offsets, sizes = np.array([80, 0, 29, 29], np.int32), np.array([51, 29, 0, 51], np.int32)
    else:
        offsets, sizes = np.array([100, 7, 2, 7, 32], np.int32), np.array([20, 0, 5, 11, 37], np.int32)
    source = jnp.asarray(rng.normal(size=(rows, width)), dtype)
    initial = jnp.full(source.shape, -7, dtype)
    expected_packed = np.zeros(source.shape, np.float32)
    expected_restored = np.full(source.shape, -7, np.float32)
    source_host = np.asarray(source, np.float32)
    position = 0
    for offset, size in zip(offsets, sizes, strict=True):
        expected_packed[position : position + size] = source_host[offset : offset + size]
        expected_restored[offset : offset + size] = source_host[offset : offset + size]
        position += size
    packed = jax.jit(compact_segments)(source, offsets, sizes)
    restored = jax.jit(restore_segments)(packed, initial, offsets, sizes)
    np.testing.assert_array_equal(np.asarray(packed, np.float32), expected_packed)
    np.testing.assert_array_equal(np.asarray(restored, np.float32), expected_restored)


def test_bf16_row_gather_preserves_every_bit_pattern():
    bits = jnp.arange(65536, dtype=jnp.uint16).reshape(4096, 16)
    values = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    indices = jnp.asarray(np.random.RandomState(102).permutation(4096))
    result = jax.jit(gather_rows)(values, indices)
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(result, jnp.uint16), bits[indices])


@pytest.mark.parametrize("sizes", [(1, 127, 0, 129), (0, 0, 0, 0), (0, 257, 0, 0), (128, 1, 127, 1)])
def test_scalar_peer_padding_preserves_float32_bits(sizes):
    bits = np.random.RandomState(103).randint(0, 2**32, size=(267, 1), dtype=np.uint32)
    values = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.float32)
    packed, blocks = jax.jit(pack_scalar_peers)(values, jnp.asarray(sizes, jnp.int32))
    expected = np.zeros(packed.shape, np.uint32)
    input_start, block_start = 0, 0
    for size in sizes:
        count = (size + 127) // 128
        expected[block_start : block_start + count].reshape(-1)[:size] = bits[input_start : input_start + size, 0]
        input_start += size
        block_start += count
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(packed, jnp.uint32), expected)
    np.testing.assert_array_equal(blocks, [(size + 127) // 128 for size in sizes])
    restored = jax.jit(lambda x, s: unpack_scalar_peers(x, s, rows=267))(packed, jnp.asarray(sizes, jnp.int32))
    expected_restored = np.zeros_like(bits)
    expected_restored[: sum(sizes)] = bits[: sum(sizes)]
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(restored, jnp.uint32), expected_restored)
