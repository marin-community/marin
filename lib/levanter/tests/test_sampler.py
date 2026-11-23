# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from levanter.layers.sampler import LogprobMode, Sampler
from levanter.models.loss import CrossEntropyNormalization, next_token_loss


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


def test_top_p_one_preserves_full_distribution_in_a_mixed_batch():
    batch, vocab = hax.Axis("batch", 2), hax.Axis("vocab", 4)
    scores = jnp.array([0.0, -10.0, -12.0, -13.0])
    logits = hax.named(jnp.broadcast_to(scores, (batch.size, vocab.size)), (batch, vocab))
    top_ps = hax.named(jnp.array([1.0, 0.5]), batch)

    tokens, logprobs = jax.jit(Sampler(vocab))(logits, 1.0, top_ps=top_ps, key=jax.random.key(0))

    np.testing.assert_allclose(logprobs.array[0], jax.nn.log_softmax(scores)[tokens.array[0]], rtol=1e-6, atol=1e-7)
    assert int(tokens.array[1]) == 0
    assert float(logprobs.array[1]) == 0.0


def test_policy_scores_match_unwarped_next_token_loss():
    position, vocab = hax.Axis("position", 3), hax.Axis("vocab", 4)
    logits = hax.named(
        jnp.array([[1.5, 0.0, -0.5, -1.0], [0.0, 1.75, 1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]),
        (position, vocab),
    )
    kwargs = dict(top_ps=0.2, key=jax.random.PRNGKey(23))
    behavior_tokens, behavior_scores = Sampler(vocab)(logits, 0.5, **kwargs)
    policy_tokens, policy_scores = Sampler(vocab, logprob_mode=LogprobMode.POLICY)(logits, 0.5, **kwargs)
    tokens = hax.named(jnp.array([0, policy_tokens.array[0], policy_tokens.array[1]]), position)
    losses = next_token_loss(
        position, vocab, logits, tokens, reduction=None, normalization=CrossEntropyNormalization.BATCH_INVARIANT
    )

    np.testing.assert_array_equal(policy_tokens.array, behavior_tokens.array)
    np.testing.assert_array_equal(behavior_scores.array[:2], jnp.zeros(2))
    np.testing.assert_array_equal(policy_scores.array[:2], -losses.array[:2])
    assert jnp.all(policy_scores.array[:2] < 0)
