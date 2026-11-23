# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Map full-context logical pages into a smaller sliding-window KV pool."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp

from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.page_table import PageBatchInfo
from levanter.inference.utils import INVALID


class WindowPageMap(eqx.Module):
    """Share one compact page map across the model's sliding-window layers."""

    logical_to_window: jax.Array
    window_to_logical: jax.Array
    next_slot: jax.Array
    terminal_pins: jax.Array
    window: int = eqx.field(static=True)

    @staticmethod
    def init(logical_pages: int, window_pages: int, window: int) -> "WindowPageMap":
        return WindowPageMap(
            jnp.full(logical_pages, INVALID, jnp.int32),
            jnp.full(window_pages, INVALID, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.zeros(logical_pages, jnp.bool_),
            window,
        )

    def reset(self) -> "WindowPageMap":
        return self.init(self.logical_to_window.size, self.window_to_logical.size, self.window)

    def remap(self, sequences: SequenceTable, batch: PageBatchInfo) -> tuple["WindowPageMap", PageBatchInfo]:
        """Allocate window pages for this batch's writes and translate its metadata."""
        destinations = batch.new_token_dests.array
        valid_destinations = jnp.arange(destinations.size) < batch.num_new_tokens
        write_pages = jnp.where(valid_destinations, destinations // batch.page_size, INVALID)
        mapped = self.logical_to_window.at[write_pages].get(mode="fill", fill_value=INVALID)
        missing = valid_destinations & (mapped == INVALID)
        updated = jax.lax.cond(
            jnp.any(missing),
            lambda: self._allocate(sequences, batch, jnp.where(missing, write_pages, INVALID)),
            lambda: self,
        )
        translated_pages = updated.logical_to_window.at[batch.page_indices.array].get(mode="fill", fill_value=INVALID)
        translations = updated.logical_to_window.at[write_pages].get(mode="fill", fill_value=INVALID)
        translated_destinations = jnp.where(
            valid_destinations, translations * batch.page_size + destinations % batch.page_size, INVALID
        )
        return updated, dataclasses.replace(
            batch,
            page_indices=hax.named(translated_pages, batch.page_indices.axes),
            new_token_dests=hax.named(translated_destinations, batch.new_token_dests.axes),
        )

    def _allocate(self, sequences: SequenceTable, batch: PageBatchInfo, missing_pages: jax.Array) -> "WindowPageMap":
        global_pages = self.logical_to_window.size
        compact_pages = self.window_to_logical.size
        page_size = batch.page_size
        columns = jnp.arange(batch.page_indices.array.shape[1])[None, :]
        active_lens = sequences.seq_lens.array[:, None]
        live_start = jnp.maximum(active_lens - self.window, 0) // page_size
        live_keep = sequences.used_mask.array[:, None] & (columns >= live_start)
        live_pages = jnp.where(live_keep, sequences.page_indices.array, INVALID)

        batch_lens = batch.seq_lens.array[:, None]
        queries = jnp.diff(batch.cu_q_lens.array)[:, None]
        batch_start = jnp.maximum(batch_lens - queries - self.window + 1, 0) // page_size
        batch_keep = (jnp.arange(batch_lens.shape[0])[:, None] < batch.num_seqs) & (columns >= batch_start)
        batch_pages = jnp.where(batch_keep, batch.page_indices.array, INVALID)

        pinned = self.terminal_pins
        pinned = pinned.at[live_pages].set(True, mode="drop")
        pinned = pinned.at[batch_pages].set(True, mode="drop")

        missing = jnp.zeros(global_pages, jnp.bool_).at[missing_pages].set(True, mode="drop")
        rank = jnp.arange(compact_pages, dtype=jnp.int32)
        ordered_slots = (rank + self.next_slot) % compact_pages
        owners = self.window_to_logical[ordered_slots]
        available = ~pinned.at[owners].get(mode="fill", fill_value=False)
        count = jnp.sum(missing, dtype=jnp.int32)
        count = eqx.error_if(count, count > jnp.sum(available), "Out of sliding-window KV pages")
        new_owners = jnp.nonzero(missing, size=compact_pages, fill_value=INVALID)[0]
        free_ranks = jnp.nonzero(available, size=compact_pages, fill_value=0)[0]
        selected = jnp.where(rank < count, ordered_slots[free_ranks], INVALID)

        evicted = self.window_to_logical.at[selected].get(mode="fill", fill_value=INVALID)
        placements = self.logical_to_window.at[evicted].set(INVALID, mode="drop")
        placements = placements.at[new_owners].set(selected, mode="drop")
        reverse = self.window_to_logical.at[selected].set(new_owners, mode="drop")
        next_slot = (selected[count - 1] + 1) % compact_pages
        return dataclasses.replace(self, logical_to_window=placements, window_to_logical=reverse, next_slot=next_slot)
