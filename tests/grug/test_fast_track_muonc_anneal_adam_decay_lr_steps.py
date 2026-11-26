# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_adam_decay_lr_steps``: the ``adamc_adam`` ``lambda`` and the Adam groups' AdamW decay
scale by ``lr_steps / (adam_lr N)``, so their total decay ``lambda adam_lr N`` is the same at every rung; the head's
decay is unchanged."""

import dataclasses

import jax
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import MUONC_ANNEAL_ADAMW_DECAY, GrugMoeMuonHConfig

_ADAMC_DECAY = 0.3
_SHAPES = {
    "token_embed": (64, 32),
    "output_proj": (32, 64),
    "stacked_blocks": {
        "stacked": {
            "mlp": {"router": (2, 32, 8)},
            "attn": {"attn_gate": (2, 32, 4)},
        }
    },
}
_ROUTER = ("stacked_blocks", "stacked", "mlp", "router")
_ATTN_GATE = ("stacked_blocks", "stacked", "attn", "attn_gate")


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _updates(config, total, step, params):
    optimizer = config.build(total)
    state = optimizer.init(params)
    # Same gradients at the same parameters, so every config holds the same Adam moments; only the decays differ.
    for key in jax.random.split(jax.random.key(1), step + 1):
        updates, state = optimizer.update(_random_tree(key), state, params)
    return updates


def _base():
    return GrugMoeMuonHConfig(
        learning_rate=0.05,
        adam_lr=0.01,
        warmup=0.0,
        lr_schedule="linear",
        min_lr_ratio=0.1,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=12.5,
        adamc_adam_decay=_ADAMC_DECAY,
    )


@pytest.mark.parametrize("lr_steps_mult", [0.5, 2.0])
def test_adam_side_decays_scale_to_the_reference_lr_steps(lr_steps_mult):
    total, step = 100, 50
    base = _base()
    scale = lr_steps_mult
    on = dataclasses.replace(base, muonc_anneal_adam_decay_lr_steps=lr_steps_mult * base.adam_lr * total)
    params = _random_tree(jax.random.key(0))
    off_updates = _updates(base, total, step, params)
    on_updates = _updates(on, total, step, params)

    adam_lr = float(base.lr_scheduler(total, override_lr=base.adam_lr)(step))
    assert adam_lr < base.adam_lr
    adamc_extra = adam_lr**2 / base.adam_lr * _ADAMC_DECAY * (scale - 1)
    for path in (_ROUTER, ("token_embed",)):
        np.testing.assert_allclose(
            _leaf(on_updates, path) - _leaf(off_updates, path), -adamc_extra * _leaf(params, path), atol=1e-6
        )
    adamw_extra = adam_lr * MUONC_ANNEAL_ADAMW_DECAY * (scale - 1)
    np.testing.assert_allclose(
        _leaf(on_updates, _ATTN_GATE) - _leaf(off_updates, _ATTN_GATE),
        -adamw_extra * _leaf(params, _ATTN_GATE),
        atol=1e-6,
    )
    np.testing.assert_array_equal(_leaf(on_updates, ("output_proj",)), _leaf(off_updates, ("output_proj",)))


def test_reference_rung_keeps_its_decays():
    total, step = 100, 20
    base = _base()
    on = dataclasses.replace(base, muonc_anneal_adam_decay_lr_steps=base.adam_lr * total)
    params = _random_tree(jax.random.key(0))
    off_updates = _updates(base, total, step, params)
    on_updates = _updates(on, total, step, params)
    for path in (_ROUTER, ("token_embed",), _ATTN_GATE, ("output_proj",)):
        np.testing.assert_allclose(_leaf(on_updates, path), _leaf(off_updates, path), rtol=1e-6, atol=1e-9)
