# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Learned EAGLE proposals over explicitly allocated draft-cache pages."""

from typing import NamedTuple

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray
from jax.sharding import PartitionSpec as P, reshard

from levanter.inference.page_table import PageBatchInfo
from levanter.layers.kv_cache import KvPageCache
from levanter.models.eagle3 import Eagle3Draft


class Eagle3Proposals(NamedTuple):
    token_ids: jax.Array
    tentative_cache: KvPageCache


class _DraftStep(NamedTuple):
    token_ids: NamedArray
    hidden_states: jax.Array
    cache: KvPageCache
    batch_info: PageBatchInfo
    positions: NamedArray


def propose_eagle3(
    model: Eagle3Draft,
    pending_tokens: NamedArray,
    target_auxiliary: jax.Array,
    cache: KvPageCache,
    batch_info: PageBatchInfo,
    positions: NamedArray,
    *,
    num_draft_tokens: int,
) -> Eagle3Proposals:
    """Generate K greedy proposals from target residuals and a pending target token.

    The packed batch must have one query per active sequence, followed by padding.
    Its pages must already cover all K draft steps. Draft positions refer to the
    target residual positions: the input token is shifted one token ahead of that
    residual. This API does not allocate pages or commit speculative cache entries.

    Only the first written row uses a real target residual. After verification,
    overwrite later rows with accepted target residuals before continuing; generated
    draft states are not interchangeable with target residuals. Cancelled requests
    discard all tentative rows. This function does not reconcile or commit that cache.
    """
    if num_draft_tokens < 1:
        raise ValueError("EAGLE proposal count must be positive")
    capacity = pending_tokens.size
    sequence_capacity = batch_info.seq_lens.size
    if sequence_capacity > capacity:
        raise ValueError("EAGLE token capacity must cover every sequence row")
    token_rows = jnp.arange(capacity)
    valid_tokens = token_rows < batch_info.num_seqs
    valid_sequences = jnp.arange(sequence_capacity) < batch_info.num_seqs

    def step(state: _DraftStep, _):
        result = model.decode(state.token_ids, state.hidden_states, state.cache, state.batch_info, state.positions)
        proposed = model.greedy_tokens(result.output)
        next_positions = hax.named(jnp.where(valid_tokens, state.positions.array + 1, 0), positions.axes)
        page_rows = jnp.minimum(token_rows, sequence_capacity - 1)
        page_columns = next_positions.array // batch_info.page_size
        pages = batch_info.page_indices.array.at[page_rows, page_columns].get(out_sharding=P(None))
        destinations = jnp.where(
            valid_tokens, pages * batch_info.page_size + next_positions.array % batch_info.page_size, -1
        )
        next_info = PageBatchInfo(
            slot_ids=batch_info.slot_ids,
            page_indices=batch_info.page_indices,
            seq_lens=hax.named(state.batch_info.seq_lens.array + valid_sequences, batch_info.seq_lens.axes),
            cu_q_lens=batch_info.cu_q_lens,
            num_seqs=batch_info.num_seqs,
            new_token_dests=hax.named(destinations, batch_info.new_token_dests.axes),
            page_size=batch_info.page_size,
        )
        next_state = _DraftStep(
            hax.named(proposed, pending_tokens.axes),
            reshard(result.output.hidden_states, P(None, None)),
            result.cache,
            next_info,
            next_positions,
        )
        output = jnp.where(valid_sequences, proposed[:sequence_capacity], -1)
        return next_state, output

    initial = _DraftStep(
        pending_tokens,
        reshard(model.project_target_states(target_auxiliary), P(None, None)),
        cache,
        batch_info,
        positions,
    )
    final, proposals = jax.lax.scan(step, initial, xs=None, length=num_draft_tokens)
    return Eagle3Proposals(proposals.T, final.cache)
