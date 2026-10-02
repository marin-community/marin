# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.inference.utils import INVALID
from levanter.kernels.pallas.short_conv import short_conv_reference
from levanter.layers.paged_short_conv import ShortConvSequenceCache, paged_short_conv


class PackedBatch(NamedTuple):
    inputs: hax.NamedArray
    info: PageBatchInfo
    positions: hax.NamedArray


def _packed_batch(chunks, page_rows, starts, page_size, slot_ids) -> PackedBatch:
    lengths = [len(chunk) for chunk in chunks]
    positions = np.concatenate(
        [np.arange(start, start + length) for start, length in zip(starts, lengths, strict=True)]
    )
    destinations = np.concatenate(
        [
            np.asarray(pages)[np.arange(start, start + length) // page_size] * page_size
            + np.arange(start, start + length) % page_size
            for pages, start, length in zip(page_rows, starts, lengths, strict=True)
        ]
    )
    # Nonzero padded activations must neither leak into output nor update history.
    values = jnp.concatenate([*chunks, jnp.full((2, chunks[0].shape[1]), 42, dtype=chunks[0].dtype)])
    info = PageBatchInfo(
        slot_ids=hax.named(jnp.asarray(slot_ids), "seq"),
        page_indices=hax.named(jnp.asarray(page_rows), ("seq", "page")),
        seq_lens=hax.named(jnp.asarray(starts) + jnp.asarray(lengths), "seq"),
        cu_q_lens=hax.named(jnp.asarray([0, *np.cumsum(lengths), INVALID], dtype=jnp.int32), "seq"),
        num_seqs=jnp.asarray(len(chunks), dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray([*destinations, INVALID, INVALID], dtype=jnp.int32), "position"),
        page_size=page_size,
    )
    return PackedBatch(
        hax.named(values, ("position", "channel")),
        info,
        hax.named(jnp.asarray([*positions, 0, 0], dtype=jnp.int32), "position"),
    )


@pytest.mark.parametrize("page_size,kernel_size", [(4, 2), (4, 4), (3, 7)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_paged_short_conv_chunked_interleaved_requests_match_full_sequence(page_size, kernel_size, dtype):
    rng = np.random.default_rng(72)
    a = jnp.asarray(rng.normal(size=(8, 8)), dtype)
    b = jnp.asarray(rng.normal(size=(7, 8)), dtype)
    weight = jnp.asarray(rng.normal(size=(kernel_size, 8)), dtype)
    pages_a, pages_b = [3, 0, 7], [4, 6, 1]
    cache = ShortConvSequenceCache.init(
        PageTableSpec(10, page_size, max_seqs=2), hax.Axis("channel", 8), kernel_size, dtype
    )
    actual_a, actual_b = [], []
    # Reversing request order exercises batch row versus persistent slot identity.
    for chunks, rows, starts, a_first in [
        ([a[:3], b[:2]], [pages_a, pages_b], [0, 0], True),
        ([b[2:5], a[3:7]], [pages_b, pages_a], [2, 3], False),
        ([a[7:], b[5:]], [pages_a, pages_b], [7, 5], True),
    ]:
        inputs, info, positions = _packed_batch(chunks, rows, starts, page_size, [0, 1] if a_first else [1, 0])
        result, cache = jax.jit(paged_short_conv)(weight, inputs, cache, info, positions)
        first, second = result.array[: len(chunks[0])], result.array[len(chunks[0]) : -2]
        actual_a.append(first if a_first else second)
        actual_b.append(second if a_first else first)
        np.testing.assert_array_equal(result.array[-2:], 0)
    for full_input, chunks in [(a, actual_a), (b, actual_b)]:
        expected = jax.jit(short_conv_reference)(weight, full_input[None, ...])[0]
        np.testing.assert_allclose(
            jnp.concatenate(chunks).astype(jnp.float32), expected.astype(jnp.float32), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("prefix_length", [2, 4])
def test_paged_short_conv_clone_diverges_without_corrupting_parent(prefix_length):
    weight = jnp.asarray([[1.0, 0.5], [0.3, 0.2], [-0.7, 0.4]])
    prefix = jnp.arange(1, 2 * prefix_length + 1, dtype=jnp.float32).reshape(prefix_length, 2)
    parent_tail = jnp.asarray([[5.0, 6.0], [7.0, 8.0]])
    child_tail = jnp.asarray([[9.0, 10.0], [-2.0, 1.0]])
    cache = ShortConvSequenceCache.init(PageTableSpec(4, 4, max_seqs=2), hax.Axis("channel", 2), 3, jnp.float32)
    inputs, info, positions = _packed_batch([prefix], [[0, 2]], [0], 4, [0])
    _, cache = jax.jit(paged_short_conv)(weight, inputs, cache, info, positions)
    cache = cache.copy_sequence(0, 1)
    inputs, info, positions = _packed_batch(
        [parent_tail, child_tail], [[0, 2], [1, 3]], [prefix_length] * 2, 4, [0, 1]
    )
    output, cache = jax.jit(paged_short_conv)(weight, inputs, cache, info, positions)
    for actual, tail in [(output.array[:2], parent_tail), (output.array[2:4], child_tail)]:
        expected = short_conv_reference(weight, jnp.concatenate([prefix, tail])[None, ...])[0, prefix_length:]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    inputs, info, positions = _packed_batch([child_tail], [[0, 2]], [0], 4, [0])
    fresh_expected = short_conv_reference(weight, child_tail[None, ...])[0]
    reused_output, _ = jax.jit(paged_short_conv)(weight, inputs, cache, info, positions)
    reset_output, _ = jax.jit(paged_short_conv)(weight, inputs, cache.reset(), info, positions)
    np.testing.assert_allclose(reused_output.array[:2], fresh_expected)
    np.testing.assert_allclose(reset_output.array[:2], fresh_expected)
