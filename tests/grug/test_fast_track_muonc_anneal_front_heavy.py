# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_front_heavy``: under ``muonc_anneal_decay`` the matrix LR holds at its peak for the first
20% of the post-warmup run, decays as ``1 - x^q``, takes the late anneal ``b(t)^2 / b(t0)`` on that shape, and keeps
the post-warmup mean of ``muonc_anneal_decay``'s matrix LR."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.optimizer import (
    LATE_ANNEAL_FRAC,
    GrugMoeMuonHConfig,
    _muonc_front_heavy_matrix_lr,
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
_MUONC = {"muonc_anneal_decay": 0.2, "muonc_anneal_adamc": 12.5, "adamc_adam_decay": 0.3}


def _params(seed=0):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def test_matrix_lr_holds_its_peak_then_keeps_the_muonc_mean():
    num_train_steps = 3030
    config = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps, batch_size=128, hidden_dim=512, seq_len=_SEQ_LEN
    )
    config = dataclasses.replace(config, lr_schedule="cosine", muonh_decay_power=0.7)
    warmup = round(0.01 * num_train_steps)
    late = round(LATE_ANNEAL_FRAC * num_train_steps)
    peak, floor = config.learning_rate, config.min_lr_ratio
    lr_schedule = _power_decay_schedule(peak, peak * floor, warmup, num_train_steps, 0.7)
    lr_at_late = float(lr_schedule(late))

    def muonc_matrix_lr(step):
        lr = lr_schedule(step)
        return jnp.where(step >= late, lr * lr / lr_at_late, lr)

    schedule, q = _muonc_front_heavy_matrix_lr(muonc_matrix_lr, lr_schedule, num_train_steps, warmup, late, peak, floor)
    steps = np.arange(num_train_steps)
    got = np.asarray(jax.vmap(schedule)(jnp.arange(num_train_steps)), np.float64)
    base = np.asarray(jax.vmap(lr_schedule)(jnp.arange(num_train_steps)), np.float64)
    muonc = np.where(steps >= late, base * base / lr_at_late, base)

    assert q == pytest.approx(0.356, abs=1e-3)
    np.testing.assert_allclose(got[:warmup], base[:warmup], rtol=1e-6)
    held = (steps >= warmup) & (steps < warmup + 0.2 * (num_train_steps - warmup))
    np.testing.assert_allclose(got[held], peak, rtol=1e-6)
    assert np.mean(got[warmup:]) == pytest.approx(np.mean(muonc[warmup:]), rel=1e-4)
    assert np.mean(got[warmup:]) / peak == pytest.approx(0.4337, abs=1e-3)
    assert np.all(np.diff(got[warmup:]) <= 1e-9)

    progress = np.clip((steps - warmup) / (num_train_steps - warmup), 0.0, 1.0)
    x = np.clip((progress - 0.2) / 0.8, 0.0, 1.0)
    shape = np.where(progress < 0.2, 1.0, floor + (1.0 - floor) * (1.0 - x**q))
    after = steps >= late
    np.testing.assert_allclose(got[after], peak * shape[after] ** 2 / shape[late], rtol=1e-4)


def test_matrix_steps_at_the_peak_inside_the_hold():
    num_train_steps, step = 50, 8
    on = GrugMoeMuonHConfig(learning_rate=0.1, warmup=0.0, muonc_anneal_front_heavy=True, **_MUONC)
    params, grads = _params(1), _params(2)

    def update_at(config):
        optimizer = config.build(num_train_steps=num_train_steps)
        state = optimizer.init(params)
        zeros = jax.tree.map(jnp.zeros_like, params)
        for _ in range(step):
            _, state = optimizer.update(zeros, state, params)
        updates, _ = optimizer.update(grads, state, params)
        return updates

    # Step 8 of 50 is inside the hold, so the matrix steps (and decays) at the peak, as under a constant schedule.
    constant = dataclasses.replace(on, muonc_anneal_front_heavy=False, lr_schedule="constant")
    np.testing.assert_allclose(_leaf(update_at(on), _W_Q), _leaf(update_at(constant), _W_Q), rtol=1e-4, atol=1e-6)
    off = dataclasses.replace(on, muonc_anneal_front_heavy=False)
    assert np.linalg.norm(_leaf(update_at(off), _W_Q)) < np.linalg.norm(_leaf(update_at(on), _W_Q))


def test_needs_muonc_anneal_decay():
    with pytest.raises(ValueError, match="muonc_anneal_front_heavy"):
        GrugMoeMuonHConfig(muonc_anneal_front_heavy=True)
