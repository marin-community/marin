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
from levanter.kernels.triton.short_conv import short_conv_reference
from levanter.layers.paged_short_conv import ShortConvSequenceCache, paged_short_conv


_PADDING_TOKENS = 2
_PAGE_SIZE = 4


class PackedBatch(NamedTuple):
    inputs: hax.NamedArray
    info: PageBatchInfo
    positions: hax.NamedArray


def _packed_batch(chunks, starts, slot_ids) -> PackedBatch:
    lengths = [len(chunk) for chunk in chunks]
    positions = np.concatenate(
        [np.arange(start, start + length) for start, length in zip(starts, lengths, strict=True)]
    )
    # Nonzero padded activations must neither leak into output nor update history.
    values = jnp.concatenate([*chunks, jnp.full((_PADDING_TOKENS, chunks[0].shape[1]), 42, dtype=chunks[0].dtype)])
    info = PageBatchInfo(
        slot_ids=hax.named(jnp.asarray([*slot_ids, INVALID]), "seq"),
        # ShortConv uses request slots, so the KV page addresses can be invalid.
        page_indices=hax.full({"seq": len(chunks) + 1, "page": 1}, INVALID, dtype=jnp.int32),
        seq_lens=hax.named(jnp.asarray([*(np.asarray(starts) + lengths), INVALID]), "seq"),
        cu_q_lens=hax.named(jnp.asarray([0, *np.cumsum(lengths), INVALID], dtype=jnp.int32), "seq"),
        num_seqs=jnp.asarray(len(chunks), dtype=jnp.int32),
        new_token_dests=hax.full({"position": values.shape[0]}, INVALID, dtype=jnp.int32),
        page_size=_PAGE_SIZE,
    )
    return PackedBatch(
        hax.named(values, ("position", "channel")),
        info,
        hax.named(jnp.asarray(np.pad(positions, (0, _PADDING_TOKENS)), dtype=jnp.int32), "position"),
    )


@pytest.mark.parametrize("kernel_size", [2, 4, 7])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_paged_short_conv_chunked_interleaved_requests_match_full_sequence(kernel_size, dtype):
    rng = np.random.default_rng(72)
    a = jnp.asarray(rng.normal(size=(8, 8)), dtype)
    b = jnp.asarray(rng.normal(size=(7, 8)), dtype)
    weight = jnp.asarray(rng.normal(size=(kernel_size, 8)), dtype)
    cache = ShortConvSequenceCache.init(
        PageTableSpec(4, _PAGE_SIZE, max_seqs=2), hax.Axis("channel", 8), kernel_size, dtype
    )
    actual_a, actual_b = [], []
    # Reversing request order exercises batch row versus persistent slot identity.
    for chunks, starts, a_first in [
        ([a[:3], b[:2]], [0, 0], True),
        ([b[2:5], a[3:7]], [2, 3], False),
        ([a[7:], b[5:]], [7, 5], True),
    ]:
        inputs, info, positions = _packed_batch(chunks, starts, [0, 1] if a_first else [1, 0])
        result, cache = jax.jit(paged_short_conv)(weight, inputs, cache, info, positions)
        first, second = result.array[: len(chunks[0])], result.array[len(chunks[0]) : -_PADDING_TOKENS]
        actual_a.append(first if a_first else second)
        actual_b.append(second if a_first else first)
        np.testing.assert_array_equal(result.array[-_PADDING_TOKENS:], 0)
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
    cache = ShortConvSequenceCache.init(
        PageTableSpec(4, _PAGE_SIZE, max_seqs=2), hax.Axis("channel", 2), 3, jnp.float32
    )
    conv = jax.jit(paged_short_conv)
    inputs, info, positions = _packed_batch([prefix], [0], [0])
    _, cache = conv(weight, inputs, cache, info, positions)
    cache = cache.copy_sequence(0, 1)
    inputs, info, positions = _packed_batch([parent_tail, child_tail], [prefix_length] * 2, [0, 1])
    output, _ = conv(weight, inputs, cache, info, positions)
    for actual, tail in [(output.array[:2], parent_tail), (output.array[2:4], child_tail)]:
        expected = short_conv_reference(weight, jnp.concatenate([prefix, tail])[None, ...])[0, prefix_length:]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_paged_short_conv_reset_discards_history():
    weight = jnp.array([[1.0], [2.0], [3.0]])
    cache = ShortConvSequenceCache.init(
        PageTableSpec(4, _PAGE_SIZE, max_seqs=1), hax.Axis("channel", 1), 3, jnp.float32
    )
    conv = jax.jit(paged_short_conv)
    prefix = jnp.array([[2.0], [5.0]])
    inputs, info, positions = _packed_batch([prefix], [0], [0])
    _, cache = conv(weight, inputs, cache, info, positions)
    tail = jnp.array([[7.0], [11.0]])
    inputs, info, positions = _packed_batch([tail], [len(prefix)], [0])
    output, _ = conv(weight, inputs, cache.reset(), info, positions)
    np.testing.assert_array_equal(output.array[: len(tail)], short_conv_reference(weight, tail[None])[0])


@pytest.mark.tpu
def test_paged_short_conv_mixed_prefill_clears_only_reused_slot():
    channels = hax.Axis("channel", 128)
    weight = jnp.broadcast_to(jnp.array([1, 2, 3, 4], dtype=jnp.bfloat16)[:, None], (4, channels.size))
    cache = ShortConvSequenceCache.init(PageTableSpec(4, _PAGE_SIZE, max_seqs=2), channels, 4, jnp.bfloat16)
    previous = jnp.full((4, channels.size), 19, dtype=jnp.bfloat16)
    live = jnp.broadcast_to(jnp.arange(1, 7, dtype=jnp.bfloat16)[:, None], (6, channels.size))
    fresh = jnp.broadcast_to(jnp.arange(7, 10, dtype=jnp.bfloat16)[:, None], (3, channels.size))
    conv = jax.jit(paged_short_conv)

    inputs, info, pos = _packed_batch([previous, live[:4]], [0, 0], [0, 1])
    _, cache = conv(weight, inputs, cache, info, pos)
    # Slot 0 gets a new one-token prompt while slot 1 decodes from its old prefix.
    inputs, info, pos = _packed_batch([live[4:5], fresh[:1]], [4, 0], [1, 0])
    output, cache = conv(weight, inputs, cache, info, pos)
    live_expected = short_conv_reference(weight, live[None])[0]
    fresh_expected = short_conv_reference(weight, fresh[None])[0]
    np.testing.assert_array_equal(output.array[:2], jnp.stack([live_expected[4], fresh_expected[0]]))

    inputs, info, pos = _packed_batch([fresh[1:], live[5:]], [1, 5], [0, 1])
    output, _ = conv(weight, inputs, cache, info, pos)
    np.testing.assert_array_equal(output.array[:3], jnp.concatenate([fresh_expected[1:], live_expected[5:]]))
