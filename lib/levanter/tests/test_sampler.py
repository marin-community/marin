# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from levanter.layers.sampler import Sampler
from levanter.models.loss import cross_entropy_loss_and_log_normalizers


def test_sampler_top_p_keeps_only_the_nucleus_head():
    vocab = hax.Axis("vocab", 4)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.array([6.0, 5.0, 1.0, -1.0], dtype=jnp.float32), (vocab,))

    token, log_prob = sampler(
        logits,
        jnp.array(1.0, dtype=jnp.float32),
        top_ps=jnp.array(0.5, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )

    assert int(token.array) == 0
    assert float(log_prob.array) == pytest.approx(0.0)


def test_sampler_top_p_keeps_cutoff_crossing_token():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.log(jnp.array([0.4, 0.35, 0.25], dtype=jnp.float32)), (vocab,))

    masked_logits = sampler._apply_top_p(logits, jnp.array(0.6, dtype=jnp.float32))

    assert jnp.isfinite(masked_logits.array[:2]).all()
    assert jnp.isneginf(masked_logits.array[2])


def test_sampler_top_p_does_not_overshoot_exact_threshold():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.log(jnp.array([0.4, 0.35, 0.25], dtype=jnp.float32)), (vocab,))

    masked_logits = sampler._apply_top_p(logits, jnp.array(0.4, dtype=jnp.float32))

    assert jnp.isfinite(masked_logits.array[0])
    assert jnp.isneginf(masked_logits.array[1:]).all()


@pytest.mark.parametrize("cutoffs", [(1.0, 1.0), (1.0, 0.5)])
def test_sampling_positions_preserves_token_draws_and_logprobs(cutoffs):
    position, vocab = hax.Axis("position", 2), hax.Axis("vocab", 4)
    logits = hax.named(jnp.array([[0.5, 1.0, 0.0, -1.0], [6.0, 5.0, 1.0, -1.0]]), (position, vocab))
    temperatures = hax.named(jnp.array([0.0, 1.0]), position)
    top_ps = hax.named(jnp.asarray(cutoffs), position)
    keys = jax.random.split(jax.random.key(17), position.size)
    sampler = Sampler(vocab)
    expected = jax.jit(hax.vmap(sampler, position))(logits, temperatures, top_ps=top_ps, key=keys)
    actual = jax.jit(sampler.sample_positions)(logits, temperatures, top_ps=top_ps, key=keys)
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array.array, expected_array.array)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_full_distribution_sampling_keeps_tail_probabilities(temperature):
    batch, vocab = hax.Axis("batch", 2), hax.Axis("vocab", 4)
    # The tail lies below the top-p cutoff tolerance. RL recomputes logprobs
    # from the entire temperature-scaled vocabulary, including this tail.
    logits = hax.named(jnp.array([[0.0, -12.0, -13.0, -14.0]] * 2) * temperature, (batch, vocab))
    tokens, log_probs = jax.jit(
        lambda x: Sampler(vocab)(x, temperature, top_ps=hax.named(jnp.array([1.0, 0.5]), batch), key=jax.random.key(0))
    )(logits)
    expected = jax.nn.log_softmax(logits.array[0] / temperature)[tokens.array[0]]
    np.testing.assert_allclose(log_probs.array[0], expected, rtol=1e-6, atol=1e-7)
    # Another sequence in the same batch still uses nucleus sampling.
    assert float(log_probs.array[1]) == pytest.approx(0.0)


def test_sampler_and_policy_loss_share_logprobs_across_batch_layouts():
    vocab = hax.Axis("vocab", 23)
    row = jax.random.normal(jax.random.key(19), (23,))
    for rows in (1, 4, 129):
        batch = hax.Axis("batch", rows)
        logits = hax.named(jnp.broadcast_to(row, (rows, 23)), (batch, vocab))
        tokens, sampled = jax.jit(lambda x: Sampler(vocab)(x, 0.0, key=jax.random.key(0)))(logits)
        temperatures = hax.named(jnp.zeros(rows), batch)
        keys = jax.random.split(jax.random.key(0), rows)
        mapped_tokens, mapped_logprobs = jax.jit(hax.vmap(Sampler(vocab), batch))(logits, temperatures, key=keys)
        np.testing.assert_array_equal(mapped_tokens.array, tokens.array)
        np.testing.assert_array_equal(
            np.asarray(mapped_logprobs.array).view(np.uint32), np.asarray(sampled.array).view(np.uint32)
        )
        target = hax.nn.one_hot(tokens[batch, 0], vocab)
        learner_loss, _ = jax.jit(lambda x: cross_entropy_loss_and_log_normalizers(x, vocab, target))(
            hax.named(row, vocab)
        )
        np.testing.assert_array_equal(np.asarray(sampled.array), np.full(rows, -float(learner_loss.array), np.float32))
