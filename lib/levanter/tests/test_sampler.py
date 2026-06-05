# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

import haliax as hax
from levanter.layers.sampler import Sampler, SamplerTopKMode


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


def test_sampler_greedy_ignores_top_p_and_reports_model_logprob():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32), (vocab,))

    token, log_prob = sampler(
        logits,
        jnp.array(0.0, dtype=jnp.float32),
        top_ps=jnp.array(0.1, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )

    assert int(token.array) == 0
    assert float(log_prob.array) == pytest.approx(float(jax.nn.log_softmax(logits.array)[0]))


def test_sampler_greedy_fast_path_jits():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)

    @jax.jit
    def sample(logits_array):
        logits = hax.named(logits_array, (vocab,))
        return sampler(logits, jnp.array(0.0, dtype=jnp.float32), key=jax.random.PRNGKey(0))

    token, log_prob = sample(jnp.array([1.0, 3.0, 2.0], dtype=jnp.float32))

    assert int(token.array) == 1
    assert float(log_prob.array) == pytest.approx(float(jax.nn.log_softmax(jnp.array([1.0, 3.0, 2.0]))[1]))


def test_sampler_mixed_temperature_keeps_greedy_slots():
    batch = hax.Axis("batch", 2)
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(
        jnp.array(
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
        (batch, vocab),
    )
    temperatures = hax.named(jnp.array([0.0, 1.0], dtype=jnp.float32), (batch,))

    tokens, log_probs = sampler(logits, temperatures, key=jax.random.PRNGKey(0))

    assert int(tokens.array[0]) == 0
    assert float(log_probs.array[0]) == pytest.approx(float(jax.nn.log_softmax(logits.array[0])[0]))


def test_sampler_full_distribution_sampling_handles_degenerate_rows():
    batch = hax.Axis("batch", 2)
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(
        jnp.array(
            [
                [0.0, -jnp.inf, -jnp.inf],
                [-jnp.inf, 0.0, -jnp.inf],
            ],
            dtype=jnp.float32,
        ),
        (batch, vocab),
    )

    tokens, log_probs = sampler(
        logits,
        hax.named(jnp.ones((batch.size,), dtype=jnp.float32), (batch,)),
        top_ps=jnp.array(1.0, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )

    assert tokens.array.tolist() == [0, 1]
    assert log_probs.array.tolist() == pytest.approx([0.0, 0.0])


def test_sampler_sampling_path_unshards_vocab_axis():
    batch = hax.Axis("batch", 2)
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.ones((batch.size, vocab.size), dtype=jnp.float32), (batch, vocab))
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))

    with hax.set_mesh(mesh), hax.axis_mapping({"vocab": "model"}):
        constrained = sampler._with_unsharded_vocab(logits)

    assert constrained.array.sharding.spec == PartitionSpec(None, None)


def test_sampler_top_k_unshards_only_candidate_axis():
    batch = hax.Axis("batch", 2)
    vocab = hax.Axis("vocab", 8)
    sampler = Sampler(vocab)
    logits = hax.named(
        jnp.arange(batch.size * vocab.size, dtype=jnp.float32).reshape(batch.size, vocab.size), (batch, vocab)
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))

    with hax.set_mesh(mesh), hax.axis_mapping({"vocab": "model"}):
        candidate_logits, candidate_ids = sampler._apply_optional_top_k(
            logits,
            hax.named(jnp.array([2, 2], dtype=jnp.int32), (batch,)),
            top_k_limit=2,
        )
        candidate_logits = sampler._with_unsharded_vocab(candidate_logits)
        assert candidate_ids is not None
        candidate_ids = sampler._with_unsharded_vocab(candidate_ids)

    assert candidate_logits.resolve_axis("vocab").size == 2
    assert candidate_ids.resolve_axis("vocab").size == 2
    assert candidate_logits.array.sharding.spec == PartitionSpec(None, None)
    assert candidate_ids.array.sharding.spec == PartitionSpec(None, None)
    np.testing.assert_array_equal(np.asarray(candidate_ids.array), np.array([[7, 6], [7, 6]], dtype=np.int32))


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


def test_sampler_top_p_one_keeps_full_distribution():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.array([1.0, 3.0, 2.0], dtype=jnp.float32), (vocab,))

    filtered = sampler._apply_optional_top_p(logits, jnp.array(1.0, dtype=jnp.float32))

    assert filtered.array.tolist() == logits.array.tolist()


def test_sampler_top_k_samples_only_from_candidate_set():
    vocab = hax.Axis("vocab", 4)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.array([-100.0, 3.0, -100.0, 2.0], dtype=jnp.float32), (vocab,))

    for seed in range(8):
        token, log_prob = sampler(
            logits,
            jnp.array(1.0, dtype=jnp.float32),
            top_ks=jnp.array(2, dtype=jnp.int32),
            top_k_limit=2,
            key=jax.random.PRNGKey(seed),
        )
        assert int(token.array) in {1, 3}
        assert float(log_prob.array) <= 0.0


def test_sampler_top_k_limit_masks_smaller_requested_k():
    vocab = hax.Axis("vocab", 4)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.array([0.0, 5.0, 4.0, 3.0], dtype=jnp.float32), (vocab,))

    token, log_prob = sampler(
        logits,
        jnp.array(1.0, dtype=jnp.float32),
        top_ks=jnp.array(1, dtype=jnp.int32),
        top_k_limit=3,
        key=jax.random.PRNGKey(0),
    )

    assert int(token.array) == 1
    assert float(log_prob.array) == pytest.approx(0.0)


def test_sampler_threshold_mask_top_k_keeps_full_vocab_axis():
    vocab = hax.Axis("vocab", 4)
    sampler = Sampler(vocab, top_k_mode=SamplerTopKMode.THRESHOLD_MASK)
    logits = hax.named(jnp.array([0.0, 5.0, 4.0, 3.0], dtype=jnp.float32), (vocab,))

    filtered, token_ids = sampler._apply_optional_top_k(logits, jnp.array(2, dtype=jnp.int32), top_k_limit=2)

    assert token_ids is None
    assert filtered.resolve_axis("vocab").size == 4
    np.testing.assert_array_equal(np.isfinite(np.asarray(filtered.array)), np.array([False, True, True, False]))


def test_sampler_threshold_mask_top_k_handles_per_row_k():
    batch = hax.Axis("batch", 2)
    vocab = hax.Axis("vocab", 5)
    sampler = Sampler(vocab, top_k_mode=SamplerTopKMode.THRESHOLD_MASK)
    logits = hax.named(
        jnp.array(
            [
                [0.0, 4.0, 2.0, 1.0, 3.0],
                [5.0, 1.0, 4.0, 2.0, 3.0],
            ],
            dtype=jnp.float32,
        ),
        (batch, vocab),
    )

    filtered, token_ids = sampler._apply_optional_top_k(
        logits,
        hax.named(jnp.array([1, 3], dtype=jnp.int32), (batch,)),
        top_k_limit=3,
    )

    assert token_ids is None
    np.testing.assert_array_equal(
        np.isfinite(np.asarray(filtered.array)),
        np.array(
            [
                [False, True, False, False, False],
                [True, False, True, False, True],
            ]
        ),
    )


def test_sampler_top_k_path_does_not_trace_full_vocab_greedy_logsumexp():
    vocab = hax.Axis("vocab", 8)
    sampler = Sampler(vocab)

    def sample(logits_array):
        logits = hax.named(logits_array, (vocab,))
        token, log_prob = sampler(
            logits,
            jnp.array(1.0, dtype=jnp.float32),
            top_ps=1.0,
            top_ks=jnp.array(2, dtype=jnp.int32),
            top_k_limit=2,
            key=jax.random.PRNGKey(0),
        )
        return token.array, log_prob.array

    hlo = str(jax.jit(sample).lower(jnp.arange(vocab.size, dtype=jnp.float32)).compiler_ir(dialect="stablehlo"))

    exponential_lines = [line for line in hlo.splitlines() if "stablehlo.exponential" in line]
    assert "chlo.top_k" in hlo
    assert any("tensor<2xf32>" in line for line in exponential_lines)
    assert all("tensor<8xf32>" not in line for line in exponential_lines)
    assert "call @argsort" not in hlo


def test_sampler_threshold_mask_top_k_path_avoids_candidate_top_k():
    vocab = hax.Axis("vocab", 8)
    sampler = Sampler(vocab, top_k_mode=SamplerTopKMode.THRESHOLD_MASK)

    def sample(logits_array):
        logits = hax.named(logits_array, (vocab,))
        token, log_prob = sampler(
            logits,
            jnp.array(1.0, dtype=jnp.float32),
            top_ks=jnp.array(2, dtype=jnp.int32),
            top_k_limit=2,
            key=jax.random.PRNGKey(0),
        )
        return token.array, log_prob.array

    hlo = str(jax.jit(sample).lower(jnp.arange(vocab.size, dtype=jnp.float32)).compiler_ir(dialect="stablehlo"))

    assert "chlo.top_k" not in hlo
    assert "tensor<8xf32>" in hlo
