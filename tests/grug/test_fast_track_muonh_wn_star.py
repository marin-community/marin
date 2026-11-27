# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonh_wn_star`` (WN*, on ``muonh_nonsi_anneal``): the matrices start at sqrt(8) x their init, the
matrix LR holds at its peak for the first 20% of the post-warmup run and keeps nonsi's mean, and the head logit
multiplier ``c`` follows the closed-form law that the model's ``head_logit_mult`` applies to the logits."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.optimizer import (
    LATE_ANNEAL_FRAC,
    GrugMoeMuonHConfig,
    _front_heavy_matrix_lr,
    _power_decay_schedule,
)

_SEQ_LEN = 4096
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "output_proj": (_D, _V),
    "stacked_blocks": {
        "stacked": {
            "attn": {"w_q": (2, _D, _D), "w_o": (2, _D, _D)},
            "mlp": {"router": (2, _D, 4), "w_up": (2, 4, _D, _D)},
            "norm": {"weight": (2, _D)},
        }
    },
}
_W_Q = ("stacked_blocks", "stacked", "attn", "w_q")
_SCALED = (_W_Q, ("stacked_blocks", "stacked", "attn", "w_o"), ("stacked_blocks", "stacked", "mlp", "w_up"))
_KEPT = (
    ("token_embed",),
    ("output_proj",),
    ("stacked_blocks", "stacked", "mlp", "router"),
    ("stacked_blocks", "stacked", "norm", "weight"),
)


def _params(seed=0):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _recipe(num_train_steps, batch_size, hidden_dim, **overrides):
    """The s7n ladder rung's optimizer: the heuristic's LRs, a cosine Adam schedule and MuonH decay power 0.7."""
    config = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps, batch_size=batch_size, hidden_dim=hidden_dim, seq_len=_SEQ_LEN
    )
    return dataclasses.replace(config, lr_schedule="cosine", muonh_decay_power=0.7, **overrides)


@pytest.mark.parametrize(
    ("hidden_dim", "num_train_steps", "batch_size", "expected"),
    [(512, 10767, 128, 1.696), (768, 30389, 128, 1.440), (1024, 39210, 256, 1.366), (1280, 20023, 1024, 1.498)],
)
def test_head_logit_mult_matches_the_ladder_values(hidden_dim, num_train_steps, batch_size, expected):
    config = _recipe(num_train_steps, batch_size, hidden_dim, muonh_nonsi_anneal=True, muonh_wn_star=True)
    mult = config.head_logit_mult(
        num_train_steps, hidden_dim=hidden_dim, vocab_size=16384, initializer_std=0.5 / math.sqrt(hidden_dim)
    )
    assert mult == pytest.approx(expected, abs=1e-3)
    off = dataclasses.replace(config, muonh_wn_star=False)
    assert off.head_logit_mult(num_train_steps, hidden_dim=hidden_dim, vocab_size=16384, initializer_std=0.02) == 0.0


def test_matrix_lr_holds_its_peak_then_keeps_the_nonsi_mean():
    num_train_steps = 10767
    config = _recipe(num_train_steps, 128, 512, muonh_nonsi_anneal=True, muonh_wn_star=True)
    warmup = round(0.01 * num_train_steps)
    late = round(LATE_ANNEAL_FRAC * num_train_steps)
    peak, floor = config.learning_rate, config.min_lr_ratio
    lr_schedule = _power_decay_schedule(peak, peak * floor, warmup, num_train_steps, 0.7)

    schedule, q = _front_heavy_matrix_lr(lr_schedule, num_train_steps, warmup, late, peak, floor)
    steps = np.arange(num_train_steps)
    got = np.asarray(jax.vmap(schedule)(jnp.arange(num_train_steps)), np.float64)
    base = np.asarray(jax.vmap(lr_schedule)(jnp.arange(num_train_steps)), np.float64)
    nonsi = np.where(steps >= late, base * base / float(lr_schedule(late)), base)

    assert q == pytest.approx(0.357, abs=2e-3)
    np.testing.assert_allclose(got[:warmup], base[:warmup], rtol=1e-6)
    held = (steps >= warmup) & (steps < warmup + 0.2 * (num_train_steps - warmup))
    np.testing.assert_allclose(got[held], peak, rtol=1e-6)
    assert np.mean(got[warmup:]) == pytest.approx(np.mean(nonsi[warmup:]), rel=1e-4)
    assert np.mean(got[warmup:]) / peak == pytest.approx(0.4336, abs=1e-3)
    assert np.all(np.diff(got[warmup:]) <= 1e-9)


def test_frozen_matrix_steps_at_the_front_heavy_lr():
    num_train_steps, step = 50, 8
    on = GrugMoeMuonHConfig(learning_rate=0.1, warmup=0.0, muonh_nonsi_anneal=True, muonh_wn_star=True)
    params, grads = _params(1), _params(2)

    def update_at(config):
        optimizer = config.build(num_train_steps=num_train_steps)
        state = optimizer.init(params)
        zeros = jax.tree.map(jnp.zeros_like, params)
        for _ in range(step):
            _, state = optimizer.update(zeros, state, params)
        updates, _ = optimizer.update(grads, state, params)
        return updates

    # Step 8 of 50 is inside the hold, so the frozen q projection steps at the peak, above nonsi's cosine LR.
    plain = GrugMoeMuonHConfig(learning_rate=0.1, adam_lr=on.adam_lr, warmup=0.0, lr_schedule="constant")
    np.testing.assert_allclose(_leaf(update_at(on), _W_Q), _leaf(update_at(plain), _W_Q), rtol=1e-4, atol=1e-6)
    nonsi = dataclasses.replace(on, muonh_wn_star=False)
    assert np.linalg.norm(_leaf(update_at(nonsi), _W_Q)) < np.linalg.norm(_leaf(update_at(on), _W_Q))


def test_matrices_start_at_sqrt8_their_init_and_train():
    config = GrugMoeMuonHConfig(learning_rate=0.1, warmup=0.0, muonh_nonsi_anneal=True, muonh_wn_star=True)
    params = _params()
    scaled = config.scale_initial_params(params)
    for path in _SCALED:
        np.testing.assert_allclose(_leaf(scaled, path), math.sqrt(8) * _leaf(params, path), rtol=1e-6)
    for path in _KEPT:
        np.testing.assert_array_equal(_leaf(scaled, path), _leaf(params, path))

    optimizer = config.build(num_train_steps=20)
    state = optimizer.init(scaled)
    updates, _ = optimizer.update(jax.tree.map(lambda p: 0.01 * p, scaled), state, scaled)
    assert all(np.all(np.isfinite(np.asarray(u))) for u in jax.tree.leaves(updates))


def test_needs_muonh_nonsi_anneal():
    with pytest.raises(ValueError, match="muonh_wn_star"):
        GrugMoeMuonHConfig(muonh_wn_star=True)


def test_head_logit_mult_scales_the_logits():
    mult = 1.7
    config = dataclasses.replace(tiny_config(), head_logit_mult=mult)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, head_logit_mult=0.0), key=jax.random.key(0))
        plain = eqx.tree_at(lambda m: m.output_proj, plain, model.output_proj * mult)
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        weight = jnp.ones(tokens.shape, jnp.float32)

        def loss_fn(m):
            return m.next_token_loss(tokens, weight, logsumexp_weight=1e-4)

        loss = eqx.filter_jit(loss_fn)(model)
        unscaled = eqx.filter_jit(loss_fn)(plain)
        no_mult = eqx.filter_jit(loss_fn)(
            Transformer.init(dataclasses.replace(config, head_logit_mult=0.0), key=jax.random.key(0))
        )

    np.testing.assert_allclose(float(loss), float(unscaled), rtol=1e-5)
    assert abs(float(loss) - float(no_mult)) > 1e-4
