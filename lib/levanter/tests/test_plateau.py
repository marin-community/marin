# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.layers.attention_mask import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.loss import next_token_loss_weight
from levanter.plateau import PlateauDroppingConfig, PlateauHistory, PlateauState, plateau_mask, plateau_training_loss

Batch = hax.Axis("batch", 2)
Pos = hax.Axis("position", 6)
CONFIG = PlateauDroppingConfig()


def _named(values, dtype=jnp.float32):
    return hax.named(jnp.asarray(values, dtype=dtype), (Batch, Pos))


def _visit(state, input_ids, loss_weights, token_loss):
    keys, host = state.lookup(input_ids, loss_weights)
    history = PlateauHistory(_named(host.previous), _named(host.last), _named(host.visits, jnp.int32))
    dropped = np.asarray(plateau_mask(CONFIG, history, _named(token_loss)).array) & (loss_weights > 0)
    state.record(keys, loss_weights, token_loss)
    return dropped


def test_constant_token_is_dropped_at_third_visit_and_returns_when_it_moves():
    input_ids = np.arange(12, dtype=np.int32).reshape(2, 6)
    loss_weights = np.array([[0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0]], dtype=np.float32)
    state = PlateauState()
    # Row 0: position 1 constant, position 2 keeps falling, position 3 learned (below the floor),
    # position 4 constant then jumps. Row 1 is constant everywhere.
    visits = [
        [[9, 2.0, 2.0, 0.040, 1.0, 9], [9, 9, 1.0, 1.0, 1.0, 9]],
        [[9, 2.0, 1.5, 0.038, 1.0, 9], [9, 9, 1.0, 1.0, 1.0, 9]],
        [[9, 2.0, 1.1, 0.037, 1.0, 9], [9, 9, 1.0, 1.0, 1.0, 9]],
        [[9, 2.0, 0.8, 0.036, 2.0, 9], [9, 9, 1.0, 1.0, 1.0, 9]],
    ]
    masks = [_visit(state, input_ids, loss_weights, np.asarray(losses, np.float32)) for losses in visits]
    assert not masks[0].any() and not masks[1].any()
    np.testing.assert_array_equal(masks[2], [[0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0]])
    np.testing.assert_array_equal(masks[3], [[0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0]])


def test_rows_with_the_same_tokens_and_different_masks_keep_separate_histories():
    input_ids = np.ones((2, 6), dtype=np.int32)
    loss_weights = np.array([[0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=np.float32)
    state = PlateauState()
    keys, _ = state.lookup(input_ids, loss_weights)
    state.record(keys, loss_weights, np.ones((2, 6), np.float32))
    assert len(set(keys)) == 2
    restored = PlateauState.loads(state.dumps())
    for key in keys:
        np.testing.assert_array_equal(restored.rows[key].positions, state.rows[key].positions)
        np.testing.assert_array_equal(restored.rows[key].last, state.rows[key].last)
        assert restored.rows[key].visits == 1


class _LogitsModel:
    """A stand-in LM whose activations are fixed logits and whose head is the identity."""

    def __init__(self, logits, Vocab):
        self.logits = logits
        self.Vocab = Vocab
        self.Embed = Vocab.alias("embed")
        self.Pos = Pos

    def activations(self, tokens, attn_mask, *, key=None):
        return self.logits.rename({self.Vocab: self.Embed})

    def get_lm_head(self):
        return hax.named(jnp.eye(self.Vocab.size), (self.Vocab, self.Embed))


def test_loss_is_the_mean_over_kept_tokens_and_dropped_tokens_get_no_gradient():
    Vocab = hax.Axis("vocab", 7)
    tokens = jnp.arange(Batch.size * Pos.size).reshape(Batch.size, Pos.size) % Vocab.size
    weights = np.array([[1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]], dtype=np.float32)
    example = LmExample(
        tokens=hax.named(tokens, (Batch, Pos)),
        loss_weight=_named(weights),
        attn_mask=AttentionMask.causal(),
    )
    logits = jax.random.normal(jax.random.PRNGKey(0), (Batch.size, Pos.size, Vocab.size))
    targets = jnp.roll(tokens, -1, axis=1)
    reference_losses = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)[..., 0]
    # Every token has two earlier visits at its current loss, except (0, 1) and (1, 2), which moved.
    previous = np.array(reference_losses)
    previous[0, 1] += 1.0
    previous[1, 2] += 1.0
    history = PlateauHistory(_named(previous), _named(reference_losses), _named(np.full((2, 6), 2), jnp.int32))
    supervised = np.asarray(next_token_loss_weight(Pos, _named(weights)).array)
    kept = np.zeros_like(supervised)
    kept[0, 1] = kept[1, 2] = 1

    def actual(values):
        model = _LogitsModel(hax.named(values, (Batch, Pos, Vocab)), Vocab)
        with jax.default_matmul_precision("float32"):
            loss, metrics = plateau_training_loss(model, example, history, config=CONFIG)
        return loss, metrics

    def reference(values):
        losses = -jnp.take_along_axis(jax.nn.log_softmax(values), targets[..., None], axis=-1)[..., 0]
        return jnp.sum(losses * kept) / kept.sum()

    (loss, metrics), gradient = jax.value_and_grad(actual, has_aux=True)(logits)
    expected_loss, expected_gradient = jax.value_and_grad(reference)(logits)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-6)
    assert float(metrics["plateau_kept_tokens"].value()) == 2
    assert float(metrics["plateau_dropped_tokens"].value()) == supervised.sum() - 2
    assert float(metrics["supervised_tokens"].value()) == supervised.sum()
    np.testing.assert_allclose(
        float(metrics["loss_sum"].value()), float(jnp.sum(reference_losses * supervised)), rtol=1e-5
    )

    with jax.default_matmul_precision("float32"):
        plain, plain_metrics = plateau_training_loss(
            _LogitsModel(hax.named(logits, (Batch, Pos, Vocab)), Vocab), example, config=CONFIG
        )
    np.testing.assert_allclose(plain, jnp.sum(reference_losses * supervised) / supervised.sum(), rtol=1e-5)
    assert float(plain_metrics["plateau_dropped_tokens"].value()) == 0
