# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_adamc``: under ``muonc_anneal_decay`` the head takes AdamC at ``lr_h = lr_max s(t)^2``
with decay ``muonc_anneal_adamc lr_h^2``, and the router weights and ``token_embed`` take
``(adam_lr_t^2 / adam_lr) adamc_adam_decay`` in place of AdamW; the matrices keep X0247's MuonC."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import MUONC_ANNEAL_ADAMW_DECAY, GrugMoeMuonHConfig

_LAMBDA = 0.15
_HEAD_DECAY_MULT = 12.5
_ADAM_DECAY = 0.3
_NUM_TRAIN_STEPS = 20
_SHAPES = {
    "token_embed": (16, 8),
    "stacked_blocks": {
        "stacked": {"attn": {"w_o": (2, 8, 8), "attn_gate": (2, 8, 4)}, "mlp": {"router": (2, 8, 4)}},
    },
    "output_proj": (8, 16),
}
_EMBED = ("token_embed",)
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_ATTN_GATE = ("stacked_blocks", "stacked", "attn", "attn_gate")
_ROUTER = ("stacked_blocks", "stacked", "mlp", "router")
_HEAD = ("output_proj",)


def _config():
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=_LAMBDA,
        muonc_anneal_adamc=_HEAD_DECAY_MULT,
        adamc_adam_decay=_ADAM_DECAY,
    )


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


def test_zero_gradient_updates_are_the_adamc_decays():
    config = _config()
    step = 15
    lr = float(config.lr_scheduler(_NUM_TRAIN_STEPS)(step))
    adam_lr = float(config.lr_scheduler(_NUM_TRAIN_STEPS, override_lr=config.adam_lr)(step))
    params = _random_tree(jax.random.key(0))
    updates = _update_at(config, params, jax.tree.map(jnp.zeros_like, params), step)

    head_lr = lr * lr / config.learning_rate
    np.testing.assert_allclose(_leaf(updates, _HEAD), -_HEAD_DECAY_MULT * head_lr**2 * _leaf(params, _HEAD), rtol=1e-4)
    adamc_decay = adam_lr**2 / config.adam_lr * _ADAM_DECAY
    for path in (_ROUTER, _EMBED):
        np.testing.assert_allclose(_leaf(updates, path), -adamc_decay * _leaf(params, path), rtol=1e-4)
    # Other rank >= 2 Adam leaves keep X0247's AdamW decay.
    np.testing.assert_allclose(
        _leaf(updates, _ATTN_GATE), -adam_lr * MUONC_ANNEAL_ADAMW_DECAY * _leaf(params, _ATTN_GATE), rtol=1e-4
    )


def test_head_step_has_norm_lr_times_initial_norm():
    config = _config()
    step = 3
    lr = float(config.lr_scheduler(_NUM_TRAIN_STEPS)(step))
    head_lr = lr * lr / config.learning_rate
    params = _random_tree(jax.random.key(1))
    updates = _update_at(config, params, _random_tree(jax.random.key(2)), step)

    head = _leaf(params, _HEAD)
    step_part = _leaf(updates, _HEAD) + _HEAD_DECAY_MULT * head_lr**2 * head
    np.testing.assert_allclose(np.linalg.norm(step_part), head_lr * np.linalg.norm(head), rtol=1e-4)
