# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom

from levanter.data.text import DpoExample
from levanter.layers.attention import AttentionMask
from levanter.lora import (
    LoraConfig,
    LoraLinear,
    lora_trainable_params_filter,
    loraize,
    unwrap_lora_modules,
)
from levanter.main.train_dpo import _logp_sum, dpo_loss_from_logps
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.lm_model import LmExample
from levanter.utils.tree_utils import inference_mode


Embed = hax.Axis("embed", 32)
Vocab = hax.Axis("vocab", 64)
Pos = hax.Axis("position", 16)


def _make_gpt2_model(key):
    config = Gpt2Config(
        hidden_dim=Embed.size,
        num_heads=2,
        num_layers=2,
        max_seq_len=Pos.size,
    )
    return Gpt2LMHeadModel.init(Vocab, config=config, key=key)


def _make_dpo_example(key):
    k1, k2 = jrandom.split(key)
    chosen_tokens = hax.random.randint(k1, Pos, 0, Vocab.size)
    rejected_tokens = hax.random.randint(k2, Pos, 0, Vocab.size)
    chosen = LmExample.causal(chosen_tokens)
    rejected = LmExample.causal(rejected_tokens)
    return DpoExample(chosen=chosen, rejected=rejected)


def test_unwrap_lora_modules_returns_base_model():
    """Loraize a model, unwrap, verify forward pass matches original base model output."""
    key = jrandom.PRNGKey(0)
    model_key, lora_key, input_key = jrandom.split(key, 3)

    model = _make_gpt2_model(model_key)
    model = inference_mode(model, True)

    loraized = loraize(model, LoraConfig(r=4), key=lora_key)
    unwrapped = unwrap_lora_modules(loraized)

    input_tokens = hax.random.randint(input_key, Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()

    base_out = model(input_tokens, attn_mask=attn_mask)
    unwrapped_out = unwrapped(input_tokens, attn_mask=attn_mask)

    assert jnp.allclose(
        base_out.array, unwrapped_out.array, atol=1e-5
    ), "Unwrapped model output should match the original base model"


def test_unwrap_lora_modules_no_lora_linear_nodes():
    """Verify no LoraLinear instances remain after unwrap."""
    key = jrandom.PRNGKey(1)
    model_key, lora_key = jrandom.split(key)

    model = _make_gpt2_model(model_key)
    loraized = loraize(model, LoraConfig(r=4), key=lora_key)

    # Verify LoraLinear exists in the loraized model
    lora_leaves = jax.tree_util.tree_leaves(loraized, is_leaf=lambda x: isinstance(x, LoraLinear))
    has_lora = any(isinstance(leaf, LoraLinear) for leaf in lora_leaves)
    assert has_lora, "Loraized model should contain LoraLinear nodes"

    unwrapped = unwrap_lora_modules(loraized)

    # Check that no LoraLinear remains
    unwrapped_leaves = jax.tree_util.tree_leaves(unwrapped, is_leaf=lambda x: isinstance(x, LoraLinear))
    remaining_lora = [leaf for leaf in unwrapped_leaves if isinstance(leaf, LoraLinear)]
    assert len(remaining_lora) == 0, f"Found {len(remaining_lora)} LoraLinear nodes after unwrap"


def test_lora_dpo_loss_computes_correctly():
    """Build model + LoRA, compute DPO loss, verify finite and metrics present."""
    key = jrandom.PRNGKey(2)
    model_key, lora_key, example_key, loss_key = jrandom.split(key, 4)

    model = _make_gpt2_model(model_key)
    loraized = loraize(model, LoraConfig(r=4), key=lora_key)
    example = _make_dpo_example(example_key)

    reference_model = unwrap_lora_modules(loraized)
    reference_model = inference_mode(reference_model, True)

    key_chosen, key_rejected = jrandom.split(loss_key)

    logp_pi_chosen = _logp_sum(loraized, example.chosen, key=key_chosen)
    logp_pi_rejected = _logp_sum(loraized, example.rejected, key=key_rejected)

    logp_ref_chosen = jax.lax.stop_gradient(_logp_sum(reference_model, example.chosen, key=key_chosen))
    logp_ref_rejected = jax.lax.stop_gradient(_logp_sum(reference_model, example.rejected, key=key_rejected))

    delta_pi = logp_pi_chosen - logp_pi_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected

    beta = 0.1
    loss, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=beta)

    assert jnp.isfinite(loss), f"DPO loss should be finite, got {loss}"
    assert "dpo_loss" in metrics
    assert "dpo_margin_policy" in metrics
    assert "dpo_margin_ref" in metrics
    assert "dpo_accuracy" in metrics


def test_lora_dpo_trainable_filter_only_lora_params():
    """Verify lora_trainable_params_filter only marks LoRA as trainable."""

    key = jrandom.PRNGKey(3)
    model_key, lora_key = jrandom.split(key)

    model = _make_gpt2_model(model_key)
    loraized = loraize(model, LoraConfig(r=4), key=lora_key)

    trainable_filter = lora_trainable_params_filter(loraized)

    # Use is_lora_param as is_leaf so LowRankLinear nodes become True leaves
    filter_leaves = jax.tree_util.tree_leaves(trainable_filter, is_leaf=lambda x: x is True or x is False)

    trainable_count = sum(1 for f in filter_leaves if f is True)
    non_trainable_count = sum(1 for f in filter_leaves if f is False)

    assert trainable_count > 0, "Should have some trainable (LoRA) parameters"
    assert non_trainable_count > 0, "Should have some frozen (base) parameters"
    assert (
        non_trainable_count > trainable_count
    ), f"Non-trainable ({non_trainable_count}) should outnumber trainable ({trainable_count})"


def test_lora_dpo_gradient_only_flows_to_lora_params():
    """Compute gradients of DPO loss, verify only LoRA params receive nonzero gradients."""

    key = jrandom.PRNGKey(4)
    model_key, lora_key, example_key, loss_key = jrandom.split(key, 4)

    model = _make_gpt2_model(model_key)
    loraized = loraize(model, LoraConfig(r=4), key=lora_key)
    example = _make_dpo_example(example_key)

    def compute_loss(model):
        reference_model = unwrap_lora_modules(model)
        reference_model = inference_mode(reference_model, True)

        k_c, k_r = jrandom.split(loss_key)
        logp_pi_chosen = _logp_sum(model, example.chosen, key=k_c)
        logp_pi_rejected = _logp_sum(model, example.rejected, key=k_r)
        logp_ref_chosen = jax.lax.stop_gradient(_logp_sum(reference_model, example.chosen, key=k_c))
        logp_ref_rejected = jax.lax.stop_gradient(_logp_sum(reference_model, example.rejected, key=k_r))
        delta_pi = logp_pi_chosen - logp_pi_rejected
        delta_ref = logp_ref_chosen - logp_ref_rejected
        loss, _ = dpo_loss_from_logps(delta_pi, delta_ref, beta=0.1)
        return loss

    # eqx.filter_grad differentiates w.r.t. all inexact arrays and returns
    # the same structure with gradients (or None/zero for non-diff leaves)
    grads = eqx.filter_grad(compute_loss)(loraized)

    # The gradient tree has the same structure as the model.
    # LoRA params live inside LoraLinear.lora (a LowRankLinear).
    # Check that some LoRA gradient arrays are nonzero.
    all_grad_leaves = jax.tree_util.tree_leaves(grads)
    # Gradients may be NamedArrays, raw jax arrays, or None
    array_grads = [g for g in all_grad_leaves if g is not None and not isinstance(g, bool) and hasattr(g, "shape")]
    assert len(array_grads) > 0, (
        f"Should have gradient arrays, got {len(all_grad_leaves)} leaves, "
        f"types: {set(type(g).__name__ for g in all_grad_leaves)}"
    )

    has_nonzero = any(jnp.any(jnp.asarray(g.array if isinstance(g, hax.NamedArray) else g) != 0) for g in array_grads)
    assert has_nonzero, "At least one gradient should be nonzero (from LoRA params)"


def test_unwrap_lora_modules_with_stacked():
    """Verify correct handling of Stacked (scan) layers with extra batch dims."""
    In = hax.Axis("In", 10)
    Mid = hax.Axis("Mid", 20)
    Layers = hax.Axis("Layers", 3)

    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

        @staticmethod
        def init(*, key):
            k1, k2 = jax.random.split(key)
            first = hnn.Linear.init(In, Mid, key=k1, out_first=True)
            second = hnn.Linear.init(Mid, In, key=k2, out_first=True)
            return Module(first, second)

    k0 = jrandom.PRNGKey(5)
    module: hnn.Stacked[Module] = hnn.Stacked.init(Layers, Module)(key=jrandom.split(k0, Layers.size))

    loraized = loraize(module, LoraConfig(r=4, target_modules=["first"]), key=k0)
    assert isinstance(loraized.stacked.first, LoraLinear)

    unwrapped = unwrap_lora_modules(loraized)
    assert isinstance(unwrapped.stacked.first, hnn.Linear)
    assert not isinstance(unwrapped.stacked.first, LoraLinear)

    # Verify the unwrapped stacked module produces the same output as the original
    input_vec = hax.random.normal(k0, (In,))
    original_out = module.fold(input_vec)
    unwrapped_out = unwrapped.fold(input_vec)
    assert jnp.allclose(
        original_out.array, unwrapped_out.array, atol=1e-5
    ), "Unwrapped stacked model should match the original"


def test_zero_init_b_makes_lora_identity():
    """With zero_init_b=True, loraized model should produce identical output to the base model."""
    key = jrandom.PRNGKey(6)
    model_key, lora_key, input_key = jrandom.split(key, 3)

    model = _make_gpt2_model(model_key)
    model = inference_mode(model, True)

    loraized = loraize(model, LoraConfig(r=4, zero_init_b=True), key=lora_key)

    input_tokens = hax.random.randint(input_key, Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()

    base_out = model(input_tokens, attn_mask=attn_mask)
    lora_out = loraized(input_tokens, attn_mask=attn_mask)

    assert jnp.allclose(
        base_out.array, lora_out.array, atol=1e-5
    ), "With zero_init_b=True, loraized model should produce identical output to base model"


def test_zero_init_b_dpo_loss_starts_near_log2():
    """With zero_init_b, initial DPO loss should be near ln(2) ≈ 0.693 (policy = reference)."""
    key = jrandom.PRNGKey(7)
    model_key, lora_key, example_key, loss_key = jrandom.split(key, 4)

    model = _make_gpt2_model(model_key)
    loraized = loraize(model, LoraConfig(r=4, zero_init_b=True), key=lora_key)
    example = _make_dpo_example(example_key)

    reference_model = unwrap_lora_modules(loraized)
    reference_model = inference_mode(reference_model, True)

    key_chosen, key_rejected = jrandom.split(loss_key)

    logp_pi_chosen = _logp_sum(loraized, example.chosen, key=key_chosen)
    logp_pi_rejected = _logp_sum(loraized, example.rejected, key=key_rejected)

    logp_ref_chosen = jax.lax.stop_gradient(_logp_sum(reference_model, example.chosen, key=key_chosen))
    logp_ref_rejected = jax.lax.stop_gradient(_logp_sum(reference_model, example.rejected, key=key_rejected))

    delta_pi = logp_pi_chosen - logp_pi_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected

    loss, _ = dpo_loss_from_logps(delta_pi, delta_ref, beta=0.1)

    # With zero_init_b, policy = reference, so delta_pi = delta_ref,
    # and DPO loss = -log(sigmoid(0)) = log(2) ≈ 0.693
    assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"
    assert (
        jnp.abs(loss - jnp.log(2.0)) < 0.01
    ), f"With zero_init_b, initial DPO loss should be ~ln(2)=0.693, got {float(loss):.4f}"
