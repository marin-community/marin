# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Causal short convolution with page-local history for incremental Hero inference."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp

from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.kv_cache import PageCache


class ShortConvPageCache(PageCache):
    """Keep only the last kernel_size - 1 inputs of each physical page.

    Page-local rings preserve sequence isolation and page cloning without storing
    every historical activation. Updates must follow increasing token positions
    within each sequence, as provided by the inference engine's packed batches.
    """

    history: hax.NamedArray
    page_size: int = eqx.field(static=True)

    @staticmethod
    def init(spec: PageTableSpec, channels: hax.Axis, kernel_size: int, dtype) -> "ShortConvPageCache":
        if kernel_size < 2:
            raise ValueError("Paged history is only needed for a convolution with at least two taps")
        ring_size = min(spec.page_size, kernel_size - 1)
        return ShortConvPageCache(
            history=hax.zeros(
                {"page": spec.num_pages, "history": ring_size, channels.name: channels.size}, dtype=dtype
            ),
            page_size=spec.page_size,
        )

    def copy_page(self, src_page: int, dst_page: int) -> "ShortConvPageCache":
        return dataclasses.replace(self, history=self.history.at["page", dst_page].set(self.history["page", src_page]))

    def reset(self) -> "ShortConvPageCache":
        return dataclasses.replace(self, history=hax.zeros_like(self.history))


def paged_short_conv(
    weight: jax.Array,
    inputs: hax.NamedArray,
    cache: ShortConvPageCache,
    batch_info: PageBatchInfo,
    positions: hax.NamedArray,
) -> tuple[hax.NamedArray, ShortConvPageCache]:
    """Apply depthwise convolution to packed incremental tokens, then update history.

    Args:
        weight: Convolution taps, shaped [kernel_size, channels], lag zero first.
        inputs: Packed activations with axes [position, channels].
        cache: Page-local rings from prior prefill/decode calls.
        batch_info: The same page allocation used for attention KV updates.
        positions: Absolute token positions within each request.
    """
    assert batch_info.page_size == cache.page_size
    assert inputs.ndim == 2 and weight.shape[1] == inputs.array.shape[1]
    assert cache.history.array.shape[1] == min(cache.page_size, weight.shape[0] - 1)
    token_index = jnp.arange(inputs.array.shape[0])
    # Padded cumulative lengths are unspecified; exclude them from the inverse map.
    starts = batch_info.cu_q_lens.array[:-1]
    active = jnp.arange(starts.shape[0]) < batch_info.num_seqs
    sequence = jnp.sum((token_index[:, None] >= starts[None, :]) & active[None, :], axis=-1) - 1
    page_rows = batch_info.page_indices.array[jnp.maximum(sequence, 0)]
    destinations = batch_info.new_token_dests.array
    lags = jnp.arange(1, weight.shape[0])
    ring_size = cache.history.array.shape[1]
    num_pages = cache.history.array.shape[0]

    def step(history, token):
        x, position, pages, destination, valid = token
        previous = position - lags
        page_offsets = jnp.maximum(previous, 0) // cache.page_size
        prior_pages = pages[page_offsets]
        prior_slots = (jnp.maximum(previous, 0) % cache.page_size) % ring_size
        prior_values = history[jnp.clip(prior_pages, 0, num_pages - 1), prior_slots]
        prior_valid = (previous >= 0) & (prior_pages >= 0) & (prior_pages < num_pages)
        prior_values = jnp.where(prior_valid[:, None], prior_values, 0)
        output = x * weight[0]
        # Match the reference convolution's lag-ordered, dtype-rounded accumulation.
        for lag in range(1, weight.shape[0]):
            output = output + weight[lag] * prior_values[lag - 1]
        # A dropped scatter leaves padding unable to overwrite a valid page's ring.
        page = jnp.where(valid, destination // cache.page_size, num_pages)
        slot = (destination % cache.page_size) % ring_size
        history = history.at[page, slot].set(x, mode="drop")
        return history, jnp.where(valid, output, 0)

    history, output = jax.lax.scan(
        step,
        cache.history.array,
        (inputs.array, positions.array, page_rows, destinations, token_index < batch_info.num_new_tokens),
    )
    return hax.named(output, inputs.axes), dataclasses.replace(cache, history=hax.named(history, cache.history.axes))
