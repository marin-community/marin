# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from jax.sharding import PartitionSpec as P, reshard

from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.inference.speculative import verify_deterministic_proposals, verify_snowball_proposals
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel


def _target():
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=24,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
        sliding_window=2,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )
    model = SnowballLMHeadModel.init(Axis("vocab", 32), config, key=jax.random.key(71))
    blocks = tuple(
        eqx.tree_at(
            lambda b: b.attn.attn_gate,
            block,
            jax.random.normal(jax.random.key(i), block.attn.attn_gate.shape),
        )
        for i, block in enumerate(model.transformer.blocks)
    )
    return eqx.tree_at(lambda m: m.transformer.blocks, model, blocks)


class _PackedInputs(NamedTuple):
    tokens: hax.NamedArray
    batch_info: PageBatchInfo
    positions: hax.NamedArray


def _packed(chunks, starts):
    capacity = 16 * jax.device_count()
    # Separate requests span noncontiguous pages, including proposals across a page boundary.
    pages = np.arange(12, dtype=np.int32).reshape(4, 3).T
    tokens = np.zeros(capacity, dtype=np.int32)
    positions = np.zeros(capacity, dtype=np.int32)
    dests = np.full(capacity, -1, dtype=np.int32)
    lengths = np.array([len(chunk) for chunk in chunks], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    for i, (chunk, start) in enumerate(zip(chunks, starts, strict=True)):
        section = slice(offsets[i], offsets[i + 1])
        pos = np.arange(start, start + len(chunk))
        tokens[section], positions[section] = chunk, pos
        dests[section] = pages[i, pos // 2] * 2 + pos % 2
    info = PageBatchInfo(
        slot_ids=hax.named(jnp.arange(3, dtype=jnp.int32), "seq"),
        page_indices=hax.named(jnp.asarray(pages), ("seq", "page")),
        seq_lens=hax.named(jnp.asarray(starts) + lengths, "seq"),
        cu_q_lens=hax.named(jnp.asarray(offsets), "seq"),
        num_seqs=jnp.asarray(3, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray(dests), "position"),
        page_size=2,
    )
    return _PackedInputs(
        hax.named(jnp.asarray(tokens), "position"), info, hax.named(jnp.asarray(positions), "position")
    )


def _logprobs(logits):
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=-1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))


@jax.default_matmul_precision("highest")
def test_paged_target_verification_matches_sequential_decode_and_discards_rejected_cache():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = _target()
        cache = model.initial_cache(PageTableSpec(12, 2), dtype=jnp.float32)
        decode = eqx.filter_jit(lambda m, c, args: m.decode(args.tokens, c, args.batch_info, args.positions))
        _, cache = decode(model, cache, _packed([[2, 7], [4, 6], [1, 3]], [0, 0, 0]))
        # The oracle runs ordinary one-token decoding, independently of verification.
        sequential_cache = cache
        pending = np.array([5, 8, 11])
        original_pending = pending.copy()
        oracle_ids, oracle_scores, oracle_logits = [], [], []
        for step in range(5):
            logits, sequential_cache = decode(model, sequential_cache, _packed(pending[:, None], [2 + step] * 3))
            values = np.asarray(logits.array)[:3]
            pending = values.argmax(axis=-1)
            oracle_logits.append(values)
            oracle_ids.append(pending)
            oracle_scores.append(_logprobs(values)[np.arange(3), pending])
        expected_ids = np.stack(oracle_ids, axis=1)
        expected_scores = np.stack(oracle_scores, axis=1)
        proposals = expected_ids[:, :3].copy()
        proposals[1, 0] = (proposals[1, 0] + 1) % model.Vocab.size
        proposals[2, 1] = (proposals[2, 1] + 1) % model.Vocab.size
        args = _packed(np.concatenate([original_pending[:, None], proposals], axis=1), [2, 2, 2])
        verify = eqx.filter_jit(
            lambda m, c, a: verify_snowball_proposals(
                m,
                a.tokens,
                c,
                a.batch_info,
                a.positions,
                jnp.zeros(3),
                jnp.full(3, 10),
                jnp.zeros(3, dtype=bool),
                max_draft_tokens=3,
                auxiliary_layers=(0, 1, 2),
                key=jax.random.key(9),
                logprobs_mode="raw_logprobs",
            )
        )
        result = verify(model, cache, args)
        np.testing.assert_array_equal(result.tokens.lengths, [4, 1, 2])
        np.testing.assert_array_equal(result.tokens.accepted_draft_tokens, [3, 0, 1])
        np.testing.assert_array_equal(result.committed_seq_lens.array, [6, 3, 4])
        for i, length in enumerate([4, 1, 2]):
            np.testing.assert_array_equal(result.tokens.token_ids[i, :length], expected_ids[i, :length])
            np.testing.assert_allclose(result.tokens.logprobs[i, :length], expected_scores[i, :length], atol=1e-5)
            np.testing.assert_array_equal(result.auxiliary_states[i, length:], 0)
        # Rejected slots contain real, incorrect KV. Visible lengths must hide them on continuation.
        continuation = [[expected_ids[i, length - 1]] for i, length in enumerate([4, 1, 2])]
        logits, _ = decode(model, result.cache, _packed(continuation, [6, 3, 4]))
        expected = np.stack([oracle_logits[length][i] for i, length in enumerate([4, 1, 2])])
        np.testing.assert_allclose(np.asarray(logits.array)[:3], expected, rtol=1e-4, atol=1e-4)

        # A cancelled request retains its old prefix; peers commit only their budget.
        limited = eqx.filter_jit(
            lambda m, c, a: verify_snowball_proposals(
                m,
                a.tokens,
                c,
                a.batch_info,
                a.positions,
                jnp.zeros(3),
                jnp.array([2, 10, 1]),
                jnp.array([False, True, False]),
                max_draft_tokens=3,
                auxiliary_layers=(0, 1, 2),
                key=jax.random.key(9),
                logprobs_mode="raw_logprobs",
            )
        )(model, cache, args)
        np.testing.assert_array_equal(limited.tokens.lengths, [2, 0, 1])
        np.testing.assert_array_equal(limited.committed_seq_lens.array, [4, 2, 3])
        np.testing.assert_array_equal(limited.auxiliary_states[1], 0)
        retry = [[expected_ids[0, 1]], [original_pending[1]], [expected_ids[2, 0]]]
        logits, _ = decode(model, limited.cache, _packed(retry, [4, 2, 3]))
        expected = np.stack([oracle_logits[2][0], oracle_logits[0][1], oracle_logits[1][2]])
        np.testing.assert_allclose(np.asarray(logits.array)[:3], expected, rtol=1e-4, atol=1e-4)


@jax.default_matmul_precision("highest")
def test_auxiliary_boundaries_match_full_forward_and_preserve_default_decode():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = _target()
        cache = model.initial_cache(PageTableSpec(12, 2), dtype=jnp.float32)
        chunks = [[2, 7, 5], [4, 6], [1]]
        args = _packed(chunks, [0, 0, 0])
        ordinary = eqx.filter_jit(lambda m, c: m.decode(args.tokens, c, args.batch_info, args.positions))(model, cache)
        observed = eqx.filter_jit(
            lambda m, c: m.decode_with_auxiliary_states(
                args.tokens,
                c,
                args.batch_info,
                args.positions,
                layers=(2, 0, 1, 1),
            )
        )(model, cache)
        np.testing.assert_allclose(observed.logits.array, ordinary[0].array, rtol=1e-5, atol=1e-5)
        for actual, expected in zip(observed.cache, ordinary[1], strict=True):
            np.testing.assert_allclose(actual.kv_pages.array, expected.kv_pages.array, rtol=1e-5, atol=1e-5)

        @eqx.filter_jit
        def full_states(m, tokens):
            tokens = reshard(tokens, P(("replica_dcn", "data", "expert"), None))
            hidden = m.transformer.token_embed.at[tokens].get(
                out_sharding=P(("replica_dcn", "data", "expert"), None, None)
            )
            hidden = m.transformer.embed_gated_norm(m.transformer.embed_norm(hidden))
            states = [hidden]
            for i, block in enumerate(m.transformer.blocks):
                hidden = block(
                    hidden,
                    AttentionMask(is_causal=True, sliding_window=2),
                    AttentionMask.causal(),
                    i == len(m.transformer.blocks) - 1,
                )
                states.append(hidden)
            return jnp.concatenate(states, axis=-1)

        expected = []
        for chunk in chunks:
            tokens = jnp.broadcast_to(jnp.asarray(chunk), (jax.device_count(), len(chunk)))
            expected.append(np.asarray(full_states(model, tokens))[0])
        np.testing.assert_allclose(
            np.asarray(observed.auxiliary_states.array)[:6], np.concatenate(expected), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_array_equal(np.asarray(observed.auxiliary_states.array)[6:], 0)


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_stochastic_verification_preserves_target_distribution_and_behavior_logprobs(mode):
    # A fixed greedy draft chooses 0. Rejection must sample *outside* 0, while
    # reported behavior scores still include its mass in the target normalizer.
    count = 32768
    logits = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 2.0]), (count, 2, 3))
    result = jax.jit(
        lambda: verify_deterministic_proposals(
            logits,
            jnp.zeros((count, 1), dtype=jnp.int32),
            jnp.ones(count, dtype=jnp.int32),
            jnp.full(count, 2.0),
            jnp.ones(count, dtype=jnp.int32),
            jnp.zeros(count, dtype=bool),
            key=jax.random.key(42),
            logprobs_mode=mode,
        )
    )()
    output = np.asarray(result.token_ids[:, 0])
    probabilities = np.exp(_logprobs(np.array([0.0, 0.5, 1.0])))
    frequencies = np.bincount(output, minlength=3) / count
    # Six binomial standard deviations; fixed seed, actual categorical outcome.
    np.testing.assert_allclose(frequencies, probabilities, atol=6 * np.sqrt(0.25 / count), rtol=0)
    expected_scores = _logprobs(np.array([0.0, 1.0, 2.0]) / (2 if mode == "processed_logprobs" else 1))
    np.testing.assert_allclose(result.logprobs[:, 0], expected_scores[output], atol=1e-6)


def test_verification_true_lengths_stop_budget_and_cancel_boundaries():
    logits = jnp.broadcast_to(jnp.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0], [0.0, 0.0, 0.0, 9.0]]), (5, 3, 4))
    result = jax.jit(
        lambda: verify_deterministic_proposals(
            logits,
            jnp.array([[1, 2]] * 5),
            jnp.array([2, 0, 1, 2, 2]),
            jnp.zeros(5),
            jnp.array([9, 9, 9, 1, 9]),
            jnp.array([False, False, False, False, True]),
            key=jax.random.key(2),
            logprobs_mode="raw_logprobs",
            stop_token_ids=(2,),
        )
    )()
    np.testing.assert_array_equal(result.token_ids, [[1, 2, -1], [1, -1, -1], [1, 2, -1], [1, -1, -1], [-1, -1, -1]])
    np.testing.assert_array_equal(result.lengths, [2, 1, 2, 1, 0])
    np.testing.assert_array_equal(result.accepted_draft_tokens, [2, 0, 1, 1, 0])
    assert np.all(np.asarray(result.logprobs)[np.asarray(result.token_ids) == -1] == 0)
