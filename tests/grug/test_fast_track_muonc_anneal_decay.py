# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_decay``: MuonC matrices decay by ``lr_m lambda_t`` with the late-annealed matrix LR, the
head takes AdamW at ``lr_max s(t)^2`` and the Adam groups decay only their rank >= 2 leaves."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import LATE_ANNEAL_FRAC, MUONC_ANNEAL_ADAMW_DECAY, GrugMoeMuonHConfig

_LAMBDA = 0.2
_NUM_TRAIN_STEPS = 20
# A 3-D layer stack and a 2-D matrix in ``muonh``, the output head, and a router stack and 1-D gain in ``adam``.
_SHAPES = {
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, 8, 8)}, "mlp": {"router": (2, 8, 4)}}},
    "gated": {"w_up": (4, 8)},
    "output_proj": (8, 16),
    "final_norm": {"weight": (8,)},
}
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_W_UP = ("gated", "w_up")
_HEAD = ("output_proj",)
_ROUTER = ("stacked_blocks", "stacked", "mlp", "router")
_GAIN = ("final_norm", "weight")


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _update_at(config, params, grads, step):
    """The update for ``grads`` at ``step``, after ``step`` zero-gradient updates (which leave every moment at zero)."""
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    for _ in range(step):
        _, state = optimizer.update(zeros, state, params)
    updates, _ = optimizer.update(grads, state, params)
    return updates


def test_zero_gradient_updates_are_the_scheduled_decays():
    config = GrugMoeMuonHConfig(learning_rate=0.1, adam_lr=0.01, warmup=0.0, muonc_anneal_decay=_LAMBDA)
    schedule = config.lr_scheduler(_NUM_TRAIN_STEPS)
    adam_schedule = config.lr_scheduler(_NUM_TRAIN_STEPS, override_lr=config.adam_lr)
    late_anneal_start = round(LATE_ANNEAL_FRAC * _NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(0))
    zeros = jax.tree.map(jnp.zeros_like, params)

    for step in (late_anneal_start - 4, late_anneal_start + 2):
        lr = float(schedule(step))
        matrix_lr = lr * lr / float(schedule(late_anneal_start)) if step >= late_anneal_start else lr
        updates = _update_at(config, params, zeros, step)
        # W <- W - lr_m (O + lambda_t W) with O = 0: a pure decay, lambda_t = lambda lr_m / lr_max.
        matrix_decay = matrix_lr * _LAMBDA * matrix_lr / config.learning_rate
        for path in (_W_O, _W_UP):
            np.testing.assert_allclose(_leaf(updates, path), -matrix_decay * _leaf(params, path), rtol=1e-4)
        head_lr = lr * lr / config.learning_rate
        np.testing.assert_allclose(
            _leaf(updates, _HEAD), -head_lr * MUONC_ANNEAL_ADAMW_DECAY * _leaf(params, _HEAD), rtol=1e-4
        )
        # The router's annealed gate_router_weight_decay is replaced by the fixed AdamW decay; 1-D gains are not decayed.
        adam_lr = float(adam_schedule(step))
        np.testing.assert_allclose(
            _leaf(updates, _ROUTER), -adam_lr * MUONC_ANNEAL_ADAMW_DECAY * _leaf(params, _ROUTER), rtol=1e-4
        )
        np.testing.assert_array_equal(_leaf(updates, _GAIN), 0.0)


def test_muon_step_does_not_scale_with_the_matrix_norm():
    config = GrugMoeMuonHConfig(learning_rate=0.1, warmup=0.0, muonc_anneal_decay=_LAMBDA)
    params = _random_tree(jax.random.key(1))
    grads = _random_tree(jax.random.key(2))
    step = 3
    lr = float(config.lr_scheduler(_NUM_TRAIN_STEPS)(step))
    decay = lr * _LAMBDA * lr / config.learning_rate

    def muon_part(scale):
        scaled = jax.tree.map(lambda p: scale * p, params)
        updates = _update_at(config, scaled, grads, step)
        return {path: _leaf(updates, path) + decay * _leaf(scaled, path) for path in (_W_O, _W_UP)}

    small, large = muon_part(1.0), muon_part(10.0)
    for path in (_W_O, _W_UP):
        np.testing.assert_allclose(small[path], large[path], rtol=1e-4, atol=1e-6)
        assert np.linalg.norm(small[path]) > 0.1 * lr
