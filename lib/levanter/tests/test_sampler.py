# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

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


@pytest.mark.tpu
def test_policy_scores_match_unwarped_next_token_loss():
    position, vocab = hax.Axis("position", 3), hax.Axis("vocab", 4)
    logits = hax.named(
        jnp.array([[9.0, -9.0, -9.0, -9.0], [0.0, 1.75, 1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]),
        (position, vocab),
    )
    temperatures = hax.full(position, 0.5)
    kwargs = dict(top_ps=hax.full(position, 0.2), key=jax.random.split(jax.random.key(23), position.size))
    behavior = hax.vmap(Sampler(vocab), position)
    policy = hax.vmap(Sampler(vocab, logprob_mode=LogprobMode.POLICY), position)
    behavior_tokens, behavior_scores = jax.jit(behavior)(logits, temperatures, **kwargs)
    policy_tokens, policy_scores = jax.jit(policy)(logits, temperatures, **kwargs)
    tokens = hax.named(jnp.array([0, policy_tokens.array[0], policy_tokens.array[1]]), position)

    @jax.jit
    def learner_loss(logits, tokens):
        return next_token_loss(
            position, vocab, logits, tokens, reduction=None, normalization=CrossEntropyNormalization.BATCH_INVARIANT
        )

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        losses = learner_loss(logits, tokens)

    np.testing.assert_array_equal(policy_tokens.array, behavior_tokens.array)
    np.testing.assert_array_equal(behavior_scores.array[:2], jnp.zeros(2))
    np.testing.assert_array_equal(policy_scores.array[:2], -losses.array[:2])
    assert jnp.all(policy_scores.array[:2] < 0)


@pytest.mark.tpu
@pytest.mark.parametrize("z_loss_weight", [0.0, 0.01])
def test_batch_invariant_next_token_loss_preserves_small_target_gradient(z_loss_weight):
    position, vocab = hax.Axis("position", 4), hax.Axis("vocab", 3)
    values = np.array([[9, -9, -9], [109, 91, 91], [9, -np.inf, -np.inf], [0, 0, 0]], dtype=np.float32)
    tokens = hax.named(jnp.zeros(position.size, dtype=jnp.int32), position)

    def objective(logits):
        losses = next_token_loss(
            position,
            vocab,
            hax.named(logits, (position, vocab)),
            tokens,
            reduction=None,
            logsumexp_weight=z_loss_weight,
            normalization=CrossEntropyNormalization.BATCH_INVARIANT,
        ).array
        return losses.sum(), losses

    (_, losses), gradients = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.asarray(values))
    fp64 = values[:-1].astype(np.float64)
    tails = np.exp(fp64[:, 1:] - fp64[:, :1])
    tail_mass = tails.sum(axis=-1)
    normalizers = fp64[:, 0] + np.log1p(tail_mass)
    expected_losses = np.log1p(tail_mass) + z_loss_weight * normalizers**2
    expected_gradients = np.column_stack((-tail_mass, tails)) / (1 + tail_mass[:, None])
    probabilities = np.column_stack((np.ones_like(tail_mass), tails)) / (1 + tail_mass[:, None])
    expected_gradients += 2 * z_loss_weight * normalizers[:, None] * probabilities
    np.testing.assert_allclose(losses[:-1], expected_losses, rtol=1e-5, atol=0)
    np.testing.assert_allclose(gradients[:-1], expected_gradients, rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    "second_cutoff,num_samples",
    [
        pytest.param(1.0, 2, id="full-distribution"),
        pytest.param(0.5, 2, id="mixed-cutoffs"),
        pytest.param(0.5, 1, id="stale-padding-cutoff"),
    ],
)
def test_sampling_positions_matches_seeded_distributions(second_cutoff, num_samples):
    position, vocab = hax.Axis("position", 2), hax.Axis("vocab", 4)
    logits = jnp.array([[0.25, 1.0, -0.5, -2.0], [6.0, 5.0, 1.0, -1.0]])
    temperatures = jnp.array([0.5, 1.0])
    keys = jax.random.split(jax.random.key(17), position.size)
    sampler = Sampler(vocab)

    tokens, logprobs = jax.jit(sampler.sample_positions)(
        hax.named(logits, (position, vocab)),
        hax.named(temperatures, position),
        top_ps=hax.named(jnp.array([1.0, second_cutoff]), position),
        num_samples=jnp.int32(num_samples),
        key=keys,
    )

    distributions = logits / temperatures[:, None]
    if num_samples == 2 and second_cutoff == 0.5:
        distributions = distributions.at[1, 1:].set(-jnp.inf)
    expected_tokens = jax.vmap(jax.random.categorical)(keys, distributions)
    expected_logprobs = jax.nn.log_softmax(distributions)[jnp.arange(position.size), expected_tokens]
    np.testing.assert_array_equal(tokens.array, expected_tokens)
    np.testing.assert_allclose(logprobs.array, expected_logprobs, rtol=1e-6, atol=1e-7)
