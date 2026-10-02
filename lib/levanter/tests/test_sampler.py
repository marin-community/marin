# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from levanter.layers.sampler import Sampler


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


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
@pytest.mark.parametrize("top_p", [0.6, 1.0])
def test_rollout_candidates_match_reporting_distribution_without_changing_sampling(mode, top_p):
    vocab = hax.Axis("vocab", 4)
    logits = hax.named(jnp.asarray(np.log([0.4, 0.3, 0.2, 0.1]), dtype=jnp.float32), vocab)
    sampler = Sampler(vocab, logprobs_mode=mode, max_logprobs=3)
    tokens, chosen, ids, scores = jax.jit(sampler.sample_with_candidates)(
        logits, 0.5, top_ps=top_p, key=jax.random.key(2)
    )
    baseline, _ = Sampler(vocab)(logits, 0.5, top_ps=top_p, key=jax.random.key(2))
    assert int(tokens.array) == int(baseline.array)
    # At temperature 1/2 the probabilities are [16,9,4,1]/30. A 0.6
    # nucleus retains the first two and renormalizes them to [16,9]/25.
    probabilities = np.array([0.4, 0.3, 0.2, 0.1])
    if mode == "processed_logprobs":
        probabilities = np.array([16, 9, 4, 1]) / 30 if top_p == 1 else np.array([16, 9, 0, 0]) / 25
    with np.errstate(divide="ignore"):
        expected = np.log(probabilities)
    np.testing.assert_array_equal(ids.array, [0, 1, 2])
    np.testing.assert_allclose(scores.array, expected[:3], atol=1e-6)
    np.testing.assert_allclose(chosen.array, expected[int(tokens.array)], atol=1e-6)


def test_disabled_candidate_capture_has_no_candidate_outputs_or_top_k_work():
    vocab = hax.Axis("vocab", 4)
    logits = hax.named(jnp.array([3.0, 2.0, 1.0, 0.0]), vocab)
    sampler = Sampler(vocab)
    call = lambda x: sampler.sample_with_candidates(x, 0.0, key=jax.random.key(0))
    tokens, logprobs, ids, scores = jax.jit(call)(logits)
    assert int(tokens.array) == 0
    assert float(logprobs.array) == pytest.approx(-math.log(sum(math.exp(-i) for i in range(4))), abs=1e-6)
    assert ids is scores is None
    # Capture is a static resource option: disabling it removes the top-K
    # primitive and candidate result arrays from the compiled computation.
    assert "top_k" not in str(jax.make_jaxpr(call)(logits))
    assert len(jax.tree.leaves(jax.eval_shape(call, logits))) == 2
