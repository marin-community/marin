# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.analysis.entropy import entropy_from_logits, top2_gap_from_logits


def _random_logits(Batch, Vocab, seed):
    data = jax.random.normal(jax.random.PRNGKey(seed), (Batch.size, Vocab.size)) * 3.0
    return hax.named(data, (Batch, Vocab))


def test_top2_gap_matches_sorted_reference():
    Batch = hax.Axis("batch", 8)
    Vocab = hax.Axis("vocab", 17)
    logits = _random_logits(Batch, Vocab, seed=0)

    gap = top2_gap_from_logits(logits, axis=Vocab).array

    sorted_logits = np.sort(np.asarray(logits.array), axis=-1)
    expected = sorted_logits[:, -1] - sorted_logits[:, -2]

    np.testing.assert_allclose(np.asarray(gap), expected, atol=1e-5)
    # The gap between the two largest logits is always non-negative.
    assert np.all(np.asarray(gap) >= 0.0)


def test_top2_gap_zero_on_tie():
    Batch = hax.Axis("batch", 1)
    Vocab = hax.Axis("vocab", 4)
    logits = hax.named(jnp.array([[5.0, 5.0, 3.0, 2.0]]), (Batch, Vocab))

    gap = top2_gap_from_logits(logits, axis=Vocab).array
    np.testing.assert_allclose(np.asarray(gap), [0.0], atol=1e-6)


def test_entropy_matches_softmax_reference():
    Batch = hax.Axis("batch", 8)
    Vocab = hax.Axis("vocab", 17)
    logits = _random_logits(Batch, Vocab, seed=1)

    entropy = entropy_from_logits(logits, axis=Vocab).array

    raw = np.asarray(logits.array)
    log_probs = raw - jax.scipy.special.logsumexp(raw, axis=-1, keepdims=True)
    probs = np.exp(log_probs)
    expected = -np.sum(probs * log_probs, axis=-1)

    np.testing.assert_allclose(np.asarray(entropy), expected, atol=1e-5)
