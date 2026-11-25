# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Causal short convolution with request-owned history for incremental Hero inference."""

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp

from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.kv_cache import PageCache


class ShortConvSequenceCache(PageCache):
    """Keep the last kernel_size - 1 inputs in each persistent request slot.

    Updates require contiguous positions per sequence. New requests start at zero,
    clearing recycled slots; sequence cloning copies the current processed prefix.
    Earlier-prefix replay must recompute history rather than reuse a KV page alone.
    """

    history: hax.NamedArray

    @staticmethod
    def init(spec: PageTableSpec, channels: hax.Axis, kernel_size: int, dtype) -> "ShortConvSequenceCache":
        if kernel_size < 2:
            raise ValueError("Cached history is only needed for a convolution with at least two taps")
        return ShortConvSequenceCache(
            history=hax.zeros(
                {"seq": spec.max_seqs, "history": kernel_size - 1, channels.name: channels.size}, dtype=dtype
            ),
        )

    def copy_page(self, src_page: int, dst_page: int) -> "ShortConvSequenceCache":
        return self

    def copy_sequence(self, src_slot: int, dst_slot: int) -> "ShortConvSequenceCache":
        return dataclasses.replace(self, history=self.history.at["seq", dst_slot].set(self.history["seq", src_slot]))

    def reset(self) -> "ShortConvSequenceCache":
        return dataclasses.replace(self, history=hax.zeros_like(self.history))


def paged_short_conv(
    weight: jax.Array,
    inputs: hax.NamedArray,
    cache: ShortConvSequenceCache,
    batch_info: PageBatchInfo,
    positions: hax.NamedArray,
) -> tuple[hax.NamedArray, ShortConvSequenceCache]:
    """Convolve contiguous chunks packed by request and advance their slot histories."""
    x = inputs.array
    history = cache.history.array
    assert x.ndim == 2 and weight.shape[1] == x.shape[1]
    num_tokens, channels = x.shape
    num_slots, ring_size, _ = history.shape
    assert ring_size == weight.shape[0] - 1
    if num_tokens == 0:
        return inputs, cache

    slot_ids = batch_info.slot_ids.array
    starts = batch_info.cu_q_lens.array[: slot_ids.size]
    ends = batch_info.cu_q_lens.array[1 : slot_ids.size + 1]
    active = (jnp.arange(slot_ids.size) < batch_info.num_seqs) & (starts < ends)
    slots = jnp.where(active, slot_ids, 0)
    token_index = jnp.arange(num_tokens)
    cached_x = x.astype(history.dtype)
    input_offset = num_slots * ring_size
    zero_index = input_offset + num_tokens
    source = jnp.concatenate((history.reshape(input_offset, channels), cached_x, jnp.zeros_like(cached_x[:1])))
    first_pos = positions.array[jnp.clip(starts, 0, num_tokens - 1)]
    ring_index = jnp.arange(ring_size)

    output = x * weight[0]
    # Only the first `lag` tokens of each chunk need a source from the slot history.
    # Scatter their indices, so the channel data can be gathered in parallel for every tap.
    for lag in range(1, weight.shape[0]):
        offsets = ring_index[:lag]
        boundary = starts[:, None] + offsets
        previous = first_pos[:, None] + offsets - lag
        history_index = slots[:, None] * ring_size + previous % ring_size
        boundary_index = jnp.where(active[:, None] & (boundary < ends[:, None]), boundary, num_tokens)
        source_index = (
            (input_offset + token_index - lag)
            .at[boundary_index]
            .set(jnp.where(previous >= 0, history_index, zero_index), mode="drop")
        )
        output = output + weight[lag] * source[source_index]
    output = jnp.where((token_index < batch_info.num_new_tokens)[:, None], output, 0)

    # Each ring entry gets the latest token at that position modulo the ring size.
    # Entries preceding the chunk retain their old value unless the slot is fresh.
    last_token = jnp.clip(ends - 1, 0, num_tokens - 1)
    last_pos = positions.array[last_token]
    distance = (last_pos[:, None] - ring_index) % ring_size
    tail_index = last_token[:, None] - distance
    previous_index = slots[:, None] * ring_size + ring_index
    tail_source = jnp.where(tail_index >= starts[:, None], input_offset + tail_index, previous_index)
    tail_source = jnp.where(last_pos[:, None] >= distance, tail_source, zero_index)
    history = history.at[jnp.where(active, slots, num_slots)].set(source[tail_source], mode="drop")
    return hax.named(output, inputs.axes), dataclasses.replace(cache, history=hax.named(history, cache.history.axes))
