# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Deletion Tuning on the Snowball SFT runtime's tiny MoE.

The paper's three invariants run against the production model class at toy
scale: the hard mask equals ``W * 1[G > 0]``, training only the gate logits
leaves the frozen weights bit-identical, and the exported checkpoint plus mask
artifact rebuild the hard-masked model.
"""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.sharding import compact_grug_mesh
from marin.rl.deletion_tuning import alpha_ramp, hard_keep_mask
from scipy.stats import norm

from experiments.post_training.long_loop.sft_runtime.deletion_masking import (
    DeletionTrainConfig,
    DeletionTuningScope,
    DeletionTuningSettings,
    _tree_replace,
    alpha_ramp_jax,
    apply_deletion_init,
    apply_hard_mask,
    capture_expert_input_norms,
    deletion_logit_filter,
    deletion_mask_state_dict,
    deletion_regularizer,
    deletion_sparsity,
    hard_keep_tensors,
    wanda_delete_threshold,
    wanda_score_samples,
    with_deletion_eval_mode,
)
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig, Transformer
from experiments.post_training.long_loop.sft_runtime.train import GrugTrainState, _make_train_step


def _tiny_config(*, shared_dim: int = 0) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=shared_dim,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
        moe_implementation="scatter",
    )


@pytest.fixture(scope="module")
def mesh():
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)


def _init_model(config: GrugModelConfig, mesh: Mesh, key_seed: int = 0) -> Transformer:
    with jax.set_mesh(mesh):
        return Transformer.init(config, key=jax.random.PRNGKey(key_seed))


def _batch(seq_len: int = 8, rows: int = 2, seed: int = 1) -> GrugLmExample:
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(rng.integers(1, 15, size=(rows, seq_len)), dtype=jnp.int32)
    return GrugLmExample(tokens=tokens, loss_weight=jnp.ones((rows, seq_len), dtype=jnp.float32))


def _install_logits(model: Transformer, seed: int = 3) -> Transformer:
    """Install deterministic stacked gate logits with a fixed sign pattern."""
    mlp = model.stacked_blocks.stacked.mlp
    shape = mlp.expert_mlp.w_gate.shape  # [L, E, D, I]
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.05, 0.02, size=shape).astype(np.float32)
    logits[0, 0] -= 0.2  # force some negative logits in every layer
    logits[1, 1] -= 0.2
    spec = jax.typeof(mlp.expert_mlp.w_gate).sharding.spec
    installed = _tree_replace(
        lambda m: m.stacked_blocks.stacked.mlp.gate_logits,
        model,
        jax.sharding.reshard(jnp.asarray(logits), spec),
    )
    installed = _tree_replace(lambda m: m.deletion_alpha, installed, jnp.asarray(1.0))
    installed = _tree_replace(lambda m: m.deletion_hard, installed, jnp.asarray(0.0))
    return installed


def _soft_forward(model: Transformer, batch: GrugLmExample):
    logits = model.logits(batch.tokens, mask=batch.attn_mask)
    return logits


def _unmasked_copy(model: Transformer) -> Transformer:
    """A model with masked weights baked in and the logits removed."""
    spec = jax.typeof(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate).sharding.spec
    keep = np.asarray(model.stacked_blocks.stacked.mlp.gate_logits > 0)
    w_gate = np.asarray(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate)
    baked = _tree_replace(
        lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_gate,
        model,
        jax.sharding.reshard(jnp.asarray(w_gate * keep.astype(np.float32)), spec),
    )
    stripped = _tree_replace(lambda m: m.stacked_blocks.stacked.mlp.gate_logits, baked, None)
    return _tree_replace(lambda m: m.deletion_alpha, stripped, None)


def test_soft_forward_matches_baked_mask(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(), mesh))
    batch = _batch()
    with jax.set_mesh(mesh):
        alpha = model.deletion_alpha
        soft = jax.nn.sigmoid(alpha * model.stacked_blocks.stacked.mlp.gate_logits)
        baked_model = _unmasked_copy(model)
        w_spec = jax.typeof(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate).sharding.spec
        soft_masked = np.asarray(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate) * np.asarray(soft)
        baked_model = _tree_replace(
            lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_gate,
            baked_model,
            jax.sharding.reshard(jnp.asarray(soft_masked), w_spec),
        )
        through_mask = _soft_forward(model, batch)
        through_baked = _soft_forward(baked_model, batch)
    np.testing.assert_allclose(np.asarray(through_mask), np.asarray(through_baked), rtol=1e-5, atol=1e-6)


def test_hard_mask_equals_zeroed_weights(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(), mesh))
    batch = _batch()
    with jax.set_mesh(mesh):
        hard_model = with_deletion_eval_mode(model)
        zeroed = _unmasked_copy(model)
        keep = hard_keep_mask(np.asarray(model.stacked_blocks.stacked.mlp.gate_logits))
        w_gate = np.asarray(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate)
        w_spec = jax.typeof(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate).sharding.spec
        zeroed = _tree_replace(
            lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_gate,
            zeroed,
            jax.sharding.reshard(jnp.asarray(w_gate * keep.astype(np.float32)), w_spec),
        )
        through_hard = _soft_forward(hard_model, batch)
        through_zeroed = _soft_forward(zeroed, batch)
    np.testing.assert_allclose(np.asarray(through_hard), np.asarray(through_zeroed), rtol=1e-5, atol=1e-6)


def test_train_step_moves_only_gate_logits(mesh):
    config = _tiny_config()
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(config, mesh))
    settings = DeletionTuningSettings(scope=DeletionTuningScope.ROUTED_GATE, lr=1e-2, total_steps=4)
    step_config = DeletionTrainConfig(settings=settings, filter_tree=deletion_logit_filter(model), total_steps=4)
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    optimizer = optax.adam(settings.lr)
    train_step = _make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None, deletion=step_config)
    state = dataclasses.replace(
        _empty_state(model), opt_state=optimizer.init(eqx.partition(model, step_config.filter_tree)[0])
    )
    batch = _batch()
    with jax.set_mesh(mesh):
        frozen_before = {name: np.asarray(leaf) for name, leaf in _frozen_leaves(model, step_config.filter_tree).items()}
        logits_before = np.asarray(model.stacked_blocks.stacked.mlp.gate_logits)
        state, _metrics, _watch = train_step(state, batch)
        trained = state.params
        frozen_after = _frozen_leaves(trained, step_config.filter_tree)
        for name, before in frozen_before.items():
            np.testing.assert_array_equal(np.asarray(frozen_after[name]), before, err_msg=f"frozen weight {name} moved")
        logits_after = np.asarray(trained.stacked_blocks.stacked.mlp.gate_logits)
        assert not np.array_equal(logits_after, logits_before), "gate logits did not train"
        expected_alpha = alpha_ramp_jax(jnp.asarray(1), 4, settings.alpha_start, settings.alpha_end)
        np.testing.assert_allclose(np.asarray(trained.deletion_alpha), np.asarray(expected_alpha), rtol=1e-6)


def _empty_state(model: Transformer):
    return GrugTrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=model,
        master_params=None,
        opt_state=None,
        ema_params=None,
        pending_qb_betas=jnp.zeros((model.config.num_layers, model.config.num_experts)),
    )


def _frozen_leaves(model: Transformer, filter_tree):
    _trainable, frozen = eqx.partition(model, filter_tree)
    flat, _treedef = jax.tree_util.tree_flatten_with_path(frozen)
    # The alpha ramp and hard/soft mode are schedule inputs, not frozen weights.
    schedule_fields = {"deletion_alpha", "deletion_hard"}
    return {
        jax.tree_util.keystr(path): leaf for path, leaf in flat if getattr(path[-1], "name", None) not in schedule_fields
    }


def test_wanda_capture_and_init_delete_fraction(mesh):
    config = _tiny_config()
    model = _init_model(config, mesh)
    batches = iter([_batch(seed=2), _batch(seed=3)])
    with jax.set_mesh(mesh):
        routed_norms, shared_norms = capture_expert_input_norms(model, batches, max_sequences=4)
        assert routed_norms.shape == (config.num_layers, config.num_experts, config.hidden_dim)
        settings = DeletionTuningSettings(scope=DeletionTuningScope.ROUTED_GATE, eta=0.25)
        samples = wanda_score_samples(model, settings, routed_norms, shared_norms, stride=1)
        threshold = wanda_delete_threshold(samples, settings.eta)
        tuned = apply_deletion_init(
            model, settings, routed_norms, shared_norms, threshold=threshold, key=jax.random.PRNGKey(settings.seed)
        )
        logits = np.asarray(tuned.stacked_blocks.stacked.mlp.gate_logits)
        assert logits.shape == (config.num_layers, config.num_experts, config.hidden_dim, config.intermediate_dim)
        deleted = float(np.mean(logits < 0))
        # eta = 0.25 of the scores enter the delete group and cross zero with
        # probability Phi(3); the keep group with Phi(-3).
        expected = settings.eta * norm.cdf(3) + (1 - settings.eta) * norm.cdf(-3)
        assert deleted == pytest.approx(expected, abs=0.03)
        # Same seed and threshold reproduce the same logits.
        again = apply_deletion_init(
            model, settings, routed_norms, shared_norms, threshold=threshold, key=jax.random.PRNGKey(settings.seed)
        )
        np.testing.assert_array_equal(np.asarray(again.stacked_blocks.stacked.mlp.gate_logits), logits)
        # The threshold rule matches the core module's group assignment up to
        # ties at the boundary value: a value threshold cannot break ties by
        # position, so membership may differ only on tied scores.
        mlp = model.stacked_blocks.stacked.mlp
        w_gate = np.asarray(mlp.expert_mlp.w_gate)
        broadcast = routed_norms.astype(np.float32).reshape((*routed_norms.shape, 1))
        scores = np.abs(w_gate.astype(np.float32)) * broadcast
        threshold_delete = scores <= threshold
        core_delete = scores <= np.sort(scores.reshape(-1))[math.floor(settings.eta * scores.size) - 1]
        ties = int(np.sum(scores == threshold))
        assert abs(int(core_delete.sum()) - int(threshold_delete.sum())) <= ties


def test_regularizer_value_and_sparsity(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(shared_dim=4), mesh))
    with jax.set_mesh(mesh):
        lam = 1e-5
        reg = deletion_regularizer(model, lam)
        logits = np.asarray(model.stacked_blocks.stacked.mlp.gate_logits)
        alpha = float(model.deletion_alpha)
        soft = 1.0 / (1.0 + np.exp(-alpha * logits))
        w_gate = np.asarray(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate)
        expected = -lam * float(np.sum((w_gate * soft) ** 2, dtype=np.float64))
        assert float(reg) == pytest.approx(expected, rel=1e-4)
        sparsity = float(deletion_sparsity(model))
        assert sparsity == pytest.approx(float(np.mean(logits <= 0)), rel=1e-6)


def test_export_hard_mask_zeroes_and_mask_state_dict_round_trips(mesh):
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_config(shared_dim=4), mesh))
    with jax.set_mesh(mesh):
        keep_before = hard_keep_tensors(model, model.config.num_experts)
        masked = apply_hard_mask(model)
        w_gate = np.asarray(masked.stacked_blocks.stacked.mlp.expert_mlp.w_gate)
        logits = np.asarray(model.stacked_blocks.stacked.mlp.gate_logits)
        expected_gate = np.asarray(model.stacked_blocks.stacked.mlp.expert_mlp.w_gate) * (logits > 0)
        np.testing.assert_array_equal(w_gate, expected_gate)

        masks = deletion_mask_state_dict(masked)
        packed = np.asarray(masks["deletion_mask/experts.gate_proj.weight"])
        unpacked = np.unpackbits(packed.reshape(-1), bitorder="big").reshape((logits.shape[0], logits.shape[1], -1))[
            ..., : logits.shape[2] * logits.shape[3]
        ]
        unpacked = unpacked.reshape(logits.shape).astype(bool)
        np.testing.assert_array_equal(unpacked, logits > 0)
        for keep in keep_before.values():
            assert keep.dtype == np.bool_


def test_alpha_ramp_jax_matches_core():
    for step, total in [(0, 10), (5, 10), (10, 10), (99, 10)]:
        np.testing.assert_allclose(
            float(alpha_ramp_jax(jnp.asarray(step, dtype=jnp.int32), total, 1e2, 1e3)),
            alpha_ramp(step, total, 1e2, 1e3),
        )
