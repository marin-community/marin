# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in target verification for deterministic EAGLE-style draft proposals."""

from typing import NamedTuple

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray
from jax.sharding import PartitionSpec as P

from levanter.inference.page_table import PageBatchInfo
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.layers.sampler import LogprobsMode
from levanter.models.snowball import SnowballLMHeadModel
from levanter.utils.jax_utils import logsumexp_last_axis


class VerifiedTokens(NamedTuple):
    token_ids: jax.Array
    logprobs: jax.Array
    lengths: jax.Array
    accepted_draft_tokens: jax.Array


class SpeculativeTargetOutput(NamedTuple):
    tokens: VerifiedTokens
    cache: ListCache[KvPageCache]
    committed_seq_lens: NamedArray
    auxiliary_states: jax.Array


def verify_deterministic_proposals(
    target_logits: jax.Array,
    draft_token_ids: jax.Array,
    draft_lengths: jax.Array,
    temperature: jax.Array,
    remaining_tokens: jax.Array,
    cancelled: jax.Array,
    *,
    key: jax.Array,
    logprobs_mode: LogprobsMode,
    stop_token_ids: tuple[int, ...] = (),
) -> VerifiedTokens:
    """Accept greedy draft proposals against the target distribution.

    Target logits have shape [sequence, K+1, vocabulary], drafts [sequence, K].
    Greedy target requests accept matching argmax tokens. Stochastic requests use
    q=one-hot: accept with p(draft), otherwise sample p excluding that draft ID.
    An all-accepted proposal emits a target bonus token. Scores always describe
    the target distribution, never the acceptance or recovery distribution.

    Draft lengths must be in [0, K], proposal IDs within the vocabulary, and
    temperatures nonnegative. This is the EAGLE greedy-proposer contract, with
    top-p=1 and no logits processors. Probabilistic draft distributions are not accepted by this API.
    """
    batch, width, vocab = target_logits.shape
    if draft_token_ids.shape != (batch, width - 1):
        raise ValueError("Target verification requires K draft IDs and K+1 target rows")
    if any(value.shape != (batch,) for value in (draft_lengths, temperature, remaining_tokens, cancelled)):
        raise ValueError("Speculative request metadata must have one entry per sequence")
    raw = target_logits.astype(jnp.float32)
    scaled = raw / jnp.where(temperature > 0, temperature, 1)[:, None, None]
    shifted = scaled - jnp.max(scaled, axis=-1, keepdims=True)
    target_logprobs = shifted - logsumexp_last_axis(shifted)[..., None]
    positions = jnp.arange(width)[None, :]
    draft_positions = positions[:, :-1]
    draft_valid = draft_positions < draft_lengths[:, None]
    safe_drafts = jnp.clip(draft_token_ids, 0, vocab - 1)
    draft_scores = jnp.take_along_axis(target_logprobs[:, :-1], safe_drafts[..., None], axis=-1)[..., 0]
    accept_key, recovery_key, bonus_key = jax.random.split(key, 3)
    uniforms = jax.random.uniform(accept_key, draft_token_ids.shape)
    stochastic_accept = uniforms <= jax.lax.exp(draft_scores, accuracy=jax.lax.AccuracyMode.HIGHEST)
    greedy_ids = jnp.argmax(raw, axis=-1)
    accepted = jnp.where(temperature[:, None] == 0, greedy_ids[:, :-1] == draft_token_ids, stochastic_accept)
    accepted &= (draft_token_ids >= 0) & (draft_token_ids < vocab)
    first_reject = jnp.min(jnp.where(draft_valid & ~accepted, draft_positions, width - 1), axis=-1)
    accepted_count = jnp.minimum(first_reject, draft_lengths)

    recovery_logits = jnp.where(jnp.arange(vocab) == safe_drafts[..., None], -jnp.inf, scaled[:, :-1])
    recovery = jax.random.categorical(recovery_key, recovery_logits, axis=-1).astype(jnp.int32)
    bonus = jax.random.categorical(bonus_key, scaled, axis=-1).astype(jnp.int32)
    recovery = jnp.concatenate([recovery, bonus[:, -1:]], axis=1)
    recovery = jnp.where(positions < draft_lengths[:, None], recovery, bonus)
    recovery = jnp.where(temperature[:, None] == 0, greedy_ids, recovery)
    padded_drafts = jnp.concatenate([draft_token_ids, jnp.zeros((batch, 1), jnp.int32)], axis=1)
    output = jnp.where(positions < accepted_count[:, None], padded_drafts, recovery)
    lengths = jnp.minimum(accepted_count + 1, jnp.maximum(remaining_tokens, 0))
    if stop_token_ids:
        is_stop = jnp.any(output[..., None] == jnp.asarray(stop_token_ids), axis=-1)
        first_stop = jnp.min(jnp.where(is_stop & (positions < lengths[:, None]), positions + 1, width), axis=-1)
        lengths = jnp.minimum(lengths, first_stop)
    lengths = jnp.where(cancelled, 0, lengths)
    if logprobs_mode == "raw_logprobs":
        shifted_raw = raw - jnp.max(raw, axis=-1, keepdims=True)
        reported = shifted_raw - logsumexp_last_axis(shifted_raw)[..., None]
    elif logprobs_mode == "processed_logprobs":
        reported = target_logprobs
    else:
        raise ValueError(f"Unsupported speculative reporting mode: {logprobs_mode}")
    scores = jnp.take_along_axis(reported, output[..., None], axis=-1)[..., 0]
    valid = positions < lengths[:, None]
    return VerifiedTokens(
        jnp.where(valid, output, -1),
        jnp.where(valid, scores, 0),
        lengths,
        jnp.minimum(accepted_count, lengths),
    )


def verify_snowball_proposals(
    model: SnowballLMHeadModel,
    input_ids: NamedArray,
    cache: ListCache[KvPageCache],
    batch_info: PageBatchInfo,
    positions: NamedArray,
    temperature: jax.Array,
    remaining_tokens: jax.Array,
    cancelled: jax.Array,
    *,
    max_draft_tokens: int,
    auxiliary_layers: tuple[int, ...],
    keys: jax.Array,
    logprobs_mode: LogprobsMode,
    stop_token_ids: tuple[int, ...] = (),
) -> SpeculativeTargetOutput:
    """Verify packed pending-token-plus-draft blocks with the real paged target.

    Each active sequence contributes its pending token followed by at most K
    proposals. PageBatchInfo describes tentative lengths after writing that block.
    The returned committed lengths exclude rejected proposals and cancelled rows;
    the next decode must use these lengths. Each sequence has its own PRNG key,
    so reordering or cancelling peers does not change its draws. Rejected KV slots remain physically
    present but are outside the visible prefix and overwritten on continuation.
    The recovered/bonus output is pending: its KV is written by the next call.

    Auxiliary states [sequence, K+1, features] describe each input position that
    predicts an emitted token, not the emitted token itself. Padding, rejected
    suffixes and cancelled rows are zero. The caller owns page allocation,
    reclamation, and draft execution.
    """
    if max_draft_tokens < 1:
        raise ValueError("Target verification requires at least one proposal slot")
    logits, cache, auxiliary = model.decode_with_auxiliary_states(
        input_ids,
        cache,
        batch_info,
        positions,
        layers=auxiliary_layers,
    )
    starts = batch_info.cu_q_lens.array[:-1]
    query_lengths = jnp.diff(batch_info.cu_q_lens.array)
    active = jnp.arange(starts.shape[0]) < batch_info.num_seqs
    indices = starts[:, None] + jnp.arange(max_draft_tokens + 1)[None, :]
    indices = jnp.minimum(indices, input_ids.size - 1)
    # The sampler owns replicated vocabulary rows, matching the ordinary engine.
    rows = logits.array.at[indices].get(out_sharding=P(None, None, None))
    proposals = input_ids.array.at[indices[:, 1:]].get(out_sharding=P(None, None))

    def verify_row(logits, proposals, length, temp, remaining, cancelled, key):
        result = verify_deterministic_proposals(
            logits[None],
            proposals[None],
            length[None],
            temp[None],
            remaining[None],
            cancelled[None],
            key=key,
            logprobs_mode=logprobs_mode,
            stop_token_ids=stop_token_ids,
        )
        return jax.tree.map(lambda x: x[0], result)

    tokens = jax.vmap(verify_row)(
        rows,
        proposals,
        jnp.maximum(query_lengths - 1, 0),
        temperature,
        remaining_tokens,
        cancelled | ~active,
        keys,
    )
    committed = batch_info.seq_lens.array - query_lengths + tokens.lengths
    features = auxiliary.array.at[indices].get(out_sharding=P(None, None, None))
    features = jnp.where(jnp.arange(max_draft_tokens + 1)[None, :, None] < tokens.lengths[:, None, None], features, 0)
    return SpeculativeTargetOutput(tokens, cache, hax.named(committed, batch_info.seq_lens.axes), features)
