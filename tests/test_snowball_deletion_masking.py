# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Deletion Tuning on the Levanter Snowball model.

The paper's three invariants run against the production model class at toy
scale: the hard mask equals ``W * 1[G > 0]``, training only the gate logits
leaves the frozen weights untouched, and the regularizer rides the aux-loss
hook during training only.
"""

import math

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.models.snowball_deletion import (
    DeletionTuningConfig,
    DeletionTuningScope,
    alpha_ramp,
    deletion_regularizer,
    deletion_trainable_filter,
    wanda_delete_threshold,
)


@pytest.fixture(scope="module")
def mesh():
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)


def _tiny_config(*, shared_dim: int = 4) -> SnowballConfig:
    return SnowballConfig(
        deletion_tuning=DeletionTuningConfig(scope=DeletionTuningScope.ROUTED_GATE, lr=1e-2, calibration_sequences=8),
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=shared_dim,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
        moe_implementation="scatter",
    )


def _init_model(config: SnowballConfig, mesh, key_seed: int = 0):
    with jax.set_mesh(mesh):
        vocab = Axis("vocab", config.vocab_size)
        return SnowballLMHeadModel.init(vocab, config, key=jax.random.PRNGKey(key_seed))


def _tokens(seq_len: int = 8, rows: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    array = rng.integers(1, 15, size=(rows, seq_len)).astype(np.int32)
    return hax.named(array, ("batch", "position"))


def _install_logits(model, seed: int = 3):
    rng = np.random.default_rng(seed)
    logits_per_layer = []
    for block in model.transformer.blocks:
        layer_shape = block.mlp.expert_mlp.w_gate.shape
        logits = rng.normal(0.05, 0.02, size=layer_shape).astype(np.float32)
        logits[0] -= 0.2  # force some negative logits in every layer
        logits_per_layer.append(logits)
    spec = jax.typeof(model.transformer.blocks[0].mlp.expert_mlp.w_gate).sharding.spec

    def where(m):
        return tuple(block.mlp.gate_logits for block in m.transformer.blocks)

    updated = eqx.tree_at(
        where,
        model,
        tuple(jax.sharding.reshard(jnp.asarray(logits), spec) for logits in logits_per_layer),
        is_leaf=lambda x: x is None,
    )
    updated = eqx.tree_at(lambda m: m.transformer.deletion_alpha, updated, jnp.asarray(1.0), is_leaf=lambda x: x is None)
    updated = eqx.tree_at(lambda m: m.transformer.deletion_hard, updated, jnp.asarray(0.0), is_leaf=lambda x: x is None)
    return updated


def test_soft_forward_matches_baked_mask(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(), mesh))
    tokens = _tokens()
    with jax.set_mesh(mesh):
        through_mask = model.activations(tokens, key=jax.random.PRNGKey(0))
        assert isinstance(through_mask, tuple)  # training mode returns the aux regularizer
        hidden = through_mask[0]
        # Bake the soft-masked weights into a copy with the logits removed.
        alpha = model.transformer.deletion_alpha
        baked = model
        for i, block in enumerate(model.transformer.blocks):
            soft = jax.nn.sigmoid(alpha * block.mlp.gate_logits)
            w = jax.lax.stop_gradient(block.mlp.expert_mlp.w_gate) * soft
            spec = jax.typeof(block.mlp.expert_mlp.w_gate).sharding.spec
            baked = eqx.tree_at(
                lambda m, i=i: m.transformer.blocks[i].mlp.expert_mlp.w_gate,
                baked,
                jax.sharding.reshard(w, spec),
            )
        for i in range(len(model.transformer.blocks)):
            baked = eqx.tree_at(
                lambda m, i=i: m.transformer.blocks[i].mlp.gate_logits,
                baked,
                None,
                is_leaf=lambda x: x is None,
            )
        baked = eqx.tree_at(lambda m: m.transformer.deletion_alpha, baked, None, is_leaf=lambda x: x is None)
        baked_hidden = baked.activations(tokens)
        np.testing.assert_allclose(np.asarray(hidden.array), np.asarray(baked_hidden.array), rtol=1e-4, atol=1e-5)


def test_hard_mask_equals_zeroed_weights(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(), mesh))
    tokens = _tokens()
    with jax.set_mesh(mesh):
        hard = eqx.tree_at(lambda m: m.transformer.deletion_hard, model, jnp.asarray(1.0))
        hard_hidden = hard.activations(tokens)
        zeroed = model
        for i, block in enumerate(model.transformer.blocks):
            keep = (block.mlp.gate_logits > 0).astype(np.float32)
            w = np.asarray(block.mlp.expert_mlp.w_gate) * np.asarray(keep)
            spec = jax.typeof(block.mlp.expert_mlp.w_gate).sharding.spec
            zeroed = eqx.tree_at(
                lambda m, i=i: m.transformer.blocks[i].mlp.expert_mlp.w_gate,
                zeroed,
                jax.sharding.reshard(jnp.asarray(w), spec),
            )
        for i in range(len(model.transformer.blocks)):
            zeroed = eqx.tree_at(
                lambda m, i=i: m.transformer.blocks[i].mlp.gate_logits,
                zeroed,
                None,
                is_leaf=lambda x: x is None,
            )
        zeroed = eqx.tree_at(lambda m: m.transformer.deletion_alpha, zeroed, None, is_leaf=lambda x: x is None)
        zeroed_hidden = zeroed.activations(tokens)
        np.testing.assert_allclose(np.asarray(hard_hidden.array), np.asarray(zeroed_hidden.array), rtol=1e-4, atol=1e-5)


def test_regularizer_value_and_train_only_aux(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(shared_dim=4), mesh))
    with jax.set_mesh(mesh):
        lam = model._config.deletion_tuning.lam
        reg = deletion_regularizer(model.transformer, lam)
        logits = np.asarray(model.transformer.blocks[0].mlp.gate_logits)
        alpha = float(model.transformer.deletion_alpha)
        soft = 1.0 / (1.0 + np.exp(-alpha * logits))
        w_gate = np.asarray(model.transformer.blocks[0].mlp.expert_mlp.w_gate)
        expected_gate = -lam * float(np.sum((w_gate * soft) ** 2, dtype=np.float64))
        # Two layers, both negative contributions: the total is below layer 0's term alone.
        assert float(reg) < expected_gate
        tokens = _tokens()
        train_out = model.activations(tokens, key=jax.random.PRNGKey(0))
        eval_out = model.activations(tokens)
        assert isinstance(train_out, tuple), "training mode must return the aux regularizer"
        assert not isinstance(eval_out, tuple), "eval (key=None) must not add the regularizer"


def test_trainable_filter_marks_only_logits(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(), mesh))
    f = deletion_trainable_filter(model)
    trainable, frozen = eqx.partition(model, f)
    flat_trainable = [np.asarray(leaf).size for leaf in jax.tree.leaves(trainable) if eqx.is_array(leaf)]
    assert flat_trainable, "the filter must mark the logits trainable"
    frozen_flat, _frozen_treedef = jax.tree_util.tree_flatten_with_path(frozen)
    for path, _leaf in frozen_flat:
        key = getattr(path[-1], "name", None) or getattr(path[-1], "key", None)
        assert not (isinstance(key, str) and key.endswith("_logits"))


def test_threshold_rule_matches_core(mesh):

    rng = np.random.default_rng(5)
    scores = rng.random((4, 8, 8)).astype(np.float32)
    eta = 0.25
    samples = scores.reshape(-1)
    threshold = wanda_delete_threshold(samples, eta)
    threshold_delete = scores <= threshold
    core_delete = scores <= np.sort(samples)[math.floor(eta * samples.size) - 1]
    ties = int(np.sum(scores == threshold))
    assert abs(int(core_delete.sum()) - int(threshold_delete.sum())) <= ties


def test_alpha_ramp_matches_core():
    for step, total in [(0, 10), (5, 10), (10, 10)]:
        np.testing.assert_allclose(alpha_ramp(step, total, 1e2, 1e3), core_alpha_ramp(step, total, 1e2, 1e3))
