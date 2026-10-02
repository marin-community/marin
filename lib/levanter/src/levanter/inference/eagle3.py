# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Learned EAGLE proposals over explicitly allocated draft-cache pages."""

from typing import NamedTuple

import equinox as eqx

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray
from jax.sharding import PartitionSpec as P, reshard

from levanter.inference.page_table import PageBatchInfo
from levanter.inference.speculative import SpeculativeTargetOutput
from levanter.layers.kv_cache import KvPageCache
from levanter.models.eagle3 import Eagle3Draft


class Eagle3State(eqx.Module):
    """Resident draft cache and target residual seed for the next proposal round."""

    cache: KvPageCache
    target_auxiliary: jax.Array

    def reset(self):
        return Eagle3State(self.cache.reset(), jnp.zeros_like(self.target_auxiliary))


def prefill_eagle3(
    model: Eagle3Draft,
    state: Eagle3State,
    tokens: NamedArray,
    target_auxiliary: jax.Array,
    target_info: PageBatchInfo,
) -> Eagle3State:
    """Prefill shifted draft inputs and retain the final residual for each request."""
    sequence_capacity = target_info.seq_lens.size
    active = jnp.arange(sequence_capacity) < target_info.num_seqs
    lengths = jnp.where(active, jnp.diff(target_info.cu_q_lens.array) - 1, 0)
    offsets = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(lengths)])
    sequence_rows = jnp.repeat(jnp.arange(sequence_capacity), lengths, total_repeat_length=tokens.size)
    positions = jnp.arange(tokens.size) - offsets[sequence_rows]
    source_rows = target_info.cu_q_lens.array[sequence_rows] + positions
    valid = jnp.arange(tokens.size) < offsets[-1]
    source_rows = jnp.clip(source_rows, 0, tokens.size - 1)
    shifted_tokens = tokens.array.at[jnp.minimum(source_rows + 1, tokens.size - 1)].get(out_sharding=P(None))
    residuals = target_auxiliary.at[source_rows].get(out_sharding=P(None, None))
    pages = target_info.page_indices.array.at[sequence_rows, positions // target_info.page_size].get(
        out_sharding=P(None)
    )
    destinations = jnp.where(valid, pages * target_info.page_size + positions % target_info.page_size, -1)
    info = PageBatchInfo(
        slot_ids=target_info.slot_ids,
        page_indices=target_info.page_indices,
        seq_lens=hax.named(lengths, target_info.seq_lens.axes),
        cu_q_lens=hax.named(offsets, target_info.cu_q_lens.axes),
        num_seqs=target_info.num_seqs,
        new_token_dests=hax.named(destinations, tokens.axes),
        page_size=target_info.page_size,
    )
    output = model.decode(
        hax.named(jnp.where(valid, shifted_tokens, 0), tokens.axes),
        reshard(model.project_target_states(residuals), P(None, None)),
        state.cache,
        info,
        hax.named(jnp.where(valid, positions, 0), tokens.axes),
    )
    last_rows = jnp.maximum(target_info.cu_q_lens.array[1:] - 1, 0)
    seeds = target_auxiliary.at[last_rows].get(out_sharding=P(None, None))
    slots = jnp.where(active, target_info.slot_ids.array, state.target_auxiliary.shape[0])
    auxiliary = state.target_auxiliary.at[slots].set(seeds, mode="drop")
    return Eagle3State(output.cache, auxiliary)


class Eagle3Proposals(NamedTuple):
    token_ids: jax.Array
    tentative_cache: KvPageCache


class _DraftStep(NamedTuple):
    token_ids: NamedArray
    hidden_states: jax.Array
    cache: KvPageCache


def propose_eagle3(
    model: Eagle3Draft,
    pending_tokens: NamedArray,
    target_auxiliary: jax.Array,
    cache: KvPageCache,
    batch_info: PageBatchInfo,
    positions: NamedArray,
    *,
    num_draft_tokens: int,
    draft_lengths: jax.Array,
) -> Eagle3Proposals:
    """Generate bounded greedy proposals using each request's remaining budget.

    Inputs contain one pending token per active sequence followed by padding.
    Positions describe target residuals, one token behind the input IDs. Pages
    must cover each sequence's draft_lengths, which cannot exceed num_draft_tokens.
    Finished proposal rows stop writing KV; their padding IDs are -1. The returned
    cache is tentative and requires target-residual reconciliation before reuse.
    """
    if num_draft_tokens < 1:
        raise ValueError("EAGLE proposal count must be positive")
    capacity = pending_tokens.size
    sequences = batch_info.seq_lens.size
    if sequences > capacity or draft_lengths.shape != (sequences,):
        raise ValueError("EAGLE capacity and proposal lengths must cover each sequence")

    def step(state: _DraftStep, index):
        active = (jnp.arange(sequences) < batch_info.num_seqs) & (index < draft_lengths)
        count = jnp.sum(active).astype(jnp.int32)
        selected = jnp.nonzero(active, size=capacity, fill_value=0)[0]
        valid_tokens = jnp.arange(capacity) < count
        valid_sequences = jnp.arange(sequences) < count
        selected_sequences = selected[:sequences]
        token_ids = state.token_ids.array.at[selected].get(out_sharding=P(None))
        hidden = state.hidden_states.at[selected].get(out_sharding=P(None, None))
        step_positions = positions.array.at[selected].get(out_sharding=P(None)) + index
        step_positions = jnp.where(valid_tokens, step_positions, 0)
        pages = batch_info.page_indices.array.at[selected, step_positions // batch_info.page_size].get(
            out_sharding=P(None)
        )
        destinations = jnp.where(
            valid_tokens, pages * batch_info.page_size + step_positions % batch_info.page_size, -1
        )
        page_indices = batch_info.page_indices.array.at[selected_sequences].get(out_sharding=P(None, None))
        seq_lens = batch_info.seq_lens.array.at[selected_sequences].get(out_sharding=P(None)) + index
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.where(valid_sequences, batch_info.slot_ids.array[selected_sequences], -1), "seq"),
            page_indices=hax.named(
                jnp.where(valid_sequences[:, None], page_indices, -1), batch_info.page_indices.axes
            ),
            seq_lens=hax.named(jnp.where(valid_sequences, seq_lens, 0), "seq"),
            cu_q_lens=hax.named(jnp.minimum(jnp.arange(sequences + 1), count), "seq"),
            num_seqs=count,
            new_token_dests=hax.named(destinations, pending_tokens.axes),
            page_size=batch_info.page_size,
        )
        result = model.decode(
            hax.named(jnp.where(valid_tokens, token_ids, 0), pending_tokens.axes),
            hidden,
            state.cache,
            info,
            hax.named(step_positions, positions.axes),
        )
        proposed = model.greedy_tokens(result.output)
        destinations = jnp.where(valid_tokens, selected, capacity)
        next_tokens = state.token_ids.array.at[destinations].set(proposed, mode="drop")
        next_hidden = state.hidden_states.at[destinations].set(
            reshard(result.output.hidden_states, P(None, None)), mode="drop"
        )
        output_destinations = jnp.where(valid_tokens, selected, sequences)
        output = jnp.full((sequences,), -1, jnp.int32).at[output_destinations].set(proposed, mode="drop")
        return _DraftStep(hax.named(next_tokens, pending_tokens.axes), next_hidden, result.cache), output

    initial = _DraftStep(pending_tokens, reshard(model.project_target_states(target_auxiliary), P(None, None)), cache)
    final, proposals = jax.lax.scan(step, initial, jnp.arange(num_draft_tokens))
    return Eagle3Proposals(proposals.T, final.cache)


class Eagle3Reconciliation(NamedTuple):
    cache: KvPageCache
    committed_seq_lens: NamedArray
    pending_tokens: jax.Array
    target_auxiliary: jax.Array


def reconcile_eagle3(
    model: Eagle3Draft,
    initial_pending_tokens: NamedArray,
    initial_target_auxiliary: jax.Array,
    tentative_cache: KvPageCache,
    initial_batch_info: PageBatchInfo,
    verification: SpeculativeTargetOutput,
    *,
    token_capacity: int,
) -> Eagle3Reconciliation:
    """Replace predicted draft-cache rows with the verified target residual prefix.

    Replay the original target-grounded row plus accepted outputs preceding the
    final pending token. Only verified lengths become visible; rejected or cancelled
    suffixes retain no logical ownership. No pages are allocated or reclaimed.
    Returned pending IDs and residuals seed the next proposal round; cancelled
    rows contain -1 IDs and zero residuals and must be removed by the scheduler.
    """
    sequences, width = verification.tokens.token_ids.shape
    if token_capacity < sequences * width:
        raise ValueError("Reconciliation token capacity must cover every verified block")
    active = jnp.arange(sequences) < initial_batch_info.num_seqs
    lengths = jnp.where(active, verification.tokens.lengths, 0)
    prefix_lengths = initial_batch_info.seq_lens.array - jnp.diff(initial_batch_info.cu_q_lens.array)
    offsets = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(lengths)])
    sequence_rows = jnp.repeat(jnp.arange(sequences), lengths, total_repeat_length=token_capacity)
    token_rows = jnp.arange(token_capacity) - offsets[sequence_rows]
    valid = jnp.arange(token_capacity) < offsets[-1]
    token_rows = jnp.clip(token_rows, 0, width - 1)
    tokens = jnp.concatenate(
        [initial_pending_tokens.array[:sequences, None], verification.tokens.token_ids[:, :-1]], axis=1
    )
    residuals = jnp.concatenate(
        [initial_target_auxiliary[:sequences, None], verification.auxiliary_states[:, :-1]], axis=1
    )
    packed_tokens = tokens.at[sequence_rows, token_rows].get(out_sharding=P(None))
    packed_residuals = residuals.at[sequence_rows, token_rows].get(out_sharding=P(None, None))
    positions = prefix_lengths[sequence_rows] + token_rows
    pages = initial_batch_info.page_indices.array.at[sequence_rows, positions // initial_batch_info.page_size].get(
        out_sharding=P(None)
    )
    destinations = jnp.where(
        valid, pages * initial_batch_info.page_size + positions % initial_batch_info.page_size, -1
    )
    info = PageBatchInfo(
        slot_ids=initial_batch_info.slot_ids,
        page_indices=initial_batch_info.page_indices,
        seq_lens=hax.named(prefix_lengths + lengths, initial_batch_info.seq_lens.axes),
        cu_q_lens=hax.named(offsets, initial_batch_info.cu_q_lens.axes),
        num_seqs=initial_batch_info.num_seqs,
        new_token_dests=hax.named(destinations, "position"),
        page_size=initial_batch_info.page_size,
    )
    output = model.decode(
        hax.named(jnp.where(valid, packed_tokens, 0), "position"),
        reshard(model.project_target_states(packed_residuals), P(None, None)),
        tentative_cache,
        info,
        hax.named(jnp.where(valid, positions, 0), "position"),
    )
    last = jnp.maximum(lengths - 1, 0)
    pending = verification.tokens.token_ids.at[jnp.arange(sequences), last].get(out_sharding=P(None))
    auxiliary = verification.auxiliary_states.at[jnp.arange(sequences), last].get(out_sharding=P(None, None))
    return Eagle3Reconciliation(
        output.cache,
        info.seq_lens,
        jnp.where(lengths > 0, pending, -1),
        jnp.where(lengths[:, None] > 0, auxiliary, 0),
    )
