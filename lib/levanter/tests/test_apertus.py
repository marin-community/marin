# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from jax import random

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.apertus import ApertusConfig, ApertusLMHeadModel, XIELUActivation
from levanter.utils.activation import ActivationFunctionEnum
from test_utils import skip_if_module_missing, skip_if_no_torch, use_test_mesh


pytestmark = skip_if_module_missing("transformers.models.apertus.modeling_apertus")


@skip_if_no_torch
@pytest.mark.parametrize("alpha_p_init", [0.8, 1.0, 0.5])
@pytest.mark.parametrize("alpha_n_init", [0.8, 1.0, 0.5])
@pytest.mark.parametrize("beta", [0.5, 0.3])
def test_xielu_vs_hf(alpha_p_init, alpha_n_init, beta):
    """Test that Levanter XIELUActivation matches HuggingFace implementation."""
    import torch
    from transformers.activations import XIELUActivation as HfXIELUActivation

    # Create both activations with same init parameters
    hf_xielu = HfXIELUActivation(
        alpha_p_init=alpha_p_init,
        alpha_n_init=alpha_n_init,
        beta=beta,
        eps=-1e-6,
        dtype=torch.float32,
    )
    hf_xielu.eval()

    lev_xielu = XIELUActivation.init(
        alpha_p_init=alpha_p_init,
        alpha_n_init=alpha_n_init,
        beta=beta,
        eps=-1e-6,
    )

    # Test on various input ranges (positive, negative, near zero)
    test_inputs = [
        np.linspace(-5.0, 5.0, 100).astype(np.float32),
        np.linspace(-0.1, 0.1, 50).astype(np.float32),  # near zero
        np.array([-1e-7, 0.0, 1e-7], dtype=np.float32),  # edge cases around zero
        np.random.randn(64, 128).astype(np.float32),  # 2D input
    ]

    for test_input in test_inputs:
        # HuggingFace forward
        torch_input = torch.from_numpy(test_input)
        with torch.no_grad():
            torch_out = hf_xielu(torch_input).numpy()

        # Levanter forward
        Batch = hax.Axis("batch", test_input.shape[0])
        if test_input.ndim == 1:
            jax_input = hax.named(test_input, Batch)
        else:
            Features = hax.Axis("features", test_input.shape[1])
            jax_input = hax.named(test_input, (Batch, Features))

        jax_out = lev_xielu(jax_input).array

        np.testing.assert_allclose(
            torch_out,
            jax_out,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"XIELUActivation mismatch for input shape {test_input.shape}",
        )


def _apertus_rope_scaling(max_position_embeddings: int) -> dict:
    original_max_position_embeddings = max_position_embeddings // 8
    return {
        "rope_type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": original_max_position_embeddings,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }


def _make_hf_apertus_config(vocab_size: int, num_kv_heads: int):
    from transformers.models.apertus.configuration_apertus import ApertusConfig as HfApertusConfig
    import torch

    max_position_embeddings = 256
    hidden_size = 64
    intermediate_size = 336  # 21504 / 4096 * 64 = 336

    config = HfApertusConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=num_kv_heads,
        hidden_act="xielu",
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_theta=12000000.0,
        rope_scaling=_apertus_rope_scaling(max_position_embeddings),
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        dtype=torch.float32,
    )
    return config


def _make_levanter_apertus_config(scan_layers: bool, num_kv_heads: int) -> ApertusConfig:
    max_position_embeddings = 256
    rope = Llama3RotaryEmbeddingsConfig(
        theta=12000000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=max_position_embeddings // 8,
    )
    return ApertusConfig(
        max_seq_len=max_position_embeddings,
        hidden_dim=64,
        intermediate_dim=336,
        num_layers=4,
        num_heads=8,
        num_kv_heads=num_kv_heads,
        activation_function=ActivationFunctionEnum.xielu,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        use_bias=False,
        rope=rope,
        gradient_checkpointing=False,
        scan_layers=scan_layers,
        tokenizer="swiss-ai/Apertus-8B-2509",
    )


@skip_if_no_torch
@pytest.mark.parametrize("scan_layers", [True, False])
@pytest.mark.parametrize("num_kv_heads", [2])
def test_apertus_roundtrip(scan_layers, num_kv_heads):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.apertus.modeling_apertus import ApertusForCausalLM

    Vocab = hax.Axis("vocab", 4096)
    hf_config = _make_hf_apertus_config(Vocab.size, num_kv_heads)
    lev_config = _make_levanter_apertus_config(scan_layers, num_kv_heads)
    input_ids = hax.random.randint(random.PRNGKey(0), lev_config.max_Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)
    torch_model = ApertusForCausalLM(hf_config)
    torch_model.eval()
    torch_out = torch_model(input_ids=input_torch).logits[0].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_path)

        model = converter.load_pretrained(
            ApertusLMHeadModel,
            ref=model_path,
            resize_vocab_to_match_tokenizer=False,
        )

        @hax.named_jit
        def compute(model, input_ids):
            return model(input_ids, attn_mask=attn_mask)

        jax_out = compute(model, input_ids).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"

        abs_diff = np.abs(torch_out - jax_out.astype(np.float32))
        max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"\nMaximum difference at {max_diff_idx}: {abs_diff[max_diff_idx]}")
        print(f"HF value: {torch_out[max_diff_idx]}, JAX value: {jax_out[max_diff_idx]}")

        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)

        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()
        torch_out2 = torch_model2(input_ids=input_torch).logits[0].detach().cpu().numpy()
        np.testing.assert_allclose(torch_out2, jax_out, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_xielu_parameters_loaded_from_checkpoint():
    """Test that xIELU learnable parameters (alpha_p, alpha_n) are correctly loaded from HF checkpoint.

    This test modifies the xIELU parameters to non-default values before saving,
    then verifies the loaded Levanter model has the same parameter values.
    This catches bugs where xIELU parameters might be re-initialized instead of loaded.
    """
    import torch
    from transformers.models.apertus.modeling_apertus import ApertusForCausalLM

    Vocab = hax.Axis("vocab", 4096)
    num_kv_heads = 2
    hf_config = _make_hf_apertus_config(Vocab.size, num_kv_heads)
    lev_config = _make_levanter_apertus_config(scan_layers=True, num_kv_heads=num_kv_heads)

    torch.random.manual_seed(42)
    torch_model = ApertusForCausalLM(hf_config)

    # Modify xIELU parameters to distinctive non-default values
    # Use values that are clearly different from defaults (0.8, 0.8, 0.5)
    modified_alpha_p = 1.5
    modified_alpha_n = 2.0

    with torch.no_grad():
        for layer in torch_model.model.layers:
            # Set alpha_p and alpha_n to known values
            layer.mlp.act_fn.alpha_p.fill_(modified_alpha_p)
            layer.mlp.act_fn.alpha_n.fill_(modified_alpha_n)

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model_modified_xielu"
        torch_model.save_pretrained(model_path)
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_path)

        model = converter.load_pretrained(
            ApertusLMHeadModel,
            ref=model_path,
            resize_vocab_to_match_tokenizer=False,
        )

        # Verify xIELU parameters were loaded correctly for each layer
        for i, layer in enumerate(model.transformer.layers.unstacked()):
            lev_alpha_p = float(layer.mlp.act_fn.alpha_p.array.item())
            lev_alpha_n = float(layer.mlp.act_fn.alpha_n.array.item())

            np.testing.assert_allclose(
                lev_alpha_p,
                modified_alpha_p,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Layer {i}: alpha_p mismatch. Expected {modified_alpha_p}, got {lev_alpha_p}",
            )
            np.testing.assert_allclose(
                lev_alpha_n,
                modified_alpha_n,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Layer {i}: alpha_n mismatch. Expected {modified_alpha_n}, got {lev_alpha_n}",
            )

        # Also verify output parity with modified parameters
        input_ids = hax.random.randint(random.PRNGKey(0), lev_config.max_Pos, 0, Vocab.size)
        attn_mask = AttentionMask.causal()
        input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int32).unsqueeze(0)

        torch_model.eval()
        torch_out = torch_model(input_ids=input_torch).logits[0].detach().cpu().numpy()

        @hax.named_jit
        def compute(model, input_ids):
            return model(input_ids, attn_mask=attn_mask)

        jax_out = compute(model, input_ids).array

        np.testing.assert_allclose(
            torch_out,
            jax_out,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Output mismatch with modified xIELU parameters",
        )
