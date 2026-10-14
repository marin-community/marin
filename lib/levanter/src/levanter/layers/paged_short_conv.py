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

    Updates require increasing positions per sequence. New requests start at zero,
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
    """Apply depthwise convolution to packed tokens using persistent request slots."""
    assert inputs.ndim == 2 and weight.shape[1] == inputs.array.shape[1]
    assert cache.history.array.shape[1] == weight.shape[0] - 1
    token_index = jnp.arange(inputs.array.shape[0])
    starts = batch_info.cu_q_lens.array[:-1]
    active = jnp.arange(starts.shape[0]) < batch_info.num_seqs
    sequence = jnp.sum((token_index[:, None] >= starts[None, :]) & active[None, :], axis=-1) - 1
    slots = batch_info.slot_ids.array[jnp.maximum(sequence, 0)]
    lags = jnp.arange(1, weight.shape[0])
    ring_size = cache.history.array.shape[1]
    num_slots = cache.history.array.shape[0]

    def step(history, token):
        x, position, slot, valid = token
        row = history[jnp.clip(slot, 0, num_slots - 1)]
        row = jnp.where(position == 0, jnp.zeros_like(row), row)
        previous = position - lags
        prior_values = row[previous % ring_size]
        prior_values = jnp.where((previous >= 0)[:, None], prior_values, 0)
        output = x * weight[0]
        # Preserve the reference convolution's lag-ordered, dtype-rounded accumulation.
        for lag in range(1, weight.shape[0]):
            output = output + weight[lag] * prior_values[lag - 1]
        row = row.at[position % ring_size].set(x)
        # Padding cannot clear or update an active request's history.
        history = history.at[jnp.where(valid, slot, num_slots)].set(row, mode="drop")
        return history, jnp.where(valid, output, 0)

    history, output = jax.lax.scan(
        step,
        cache.history.array,
        (inputs.array, positions.array, slots, token_index < batch_info.num_new_tokens),
    )
    return hax.named(output, inputs.axes), dataclasses.replace(cache, history=hax.named(history, cache.history.axes))
