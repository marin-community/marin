# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import chex
import equinox as eqx
import numpy as np
import pytest
from jax import random

import haliax as hax
import haliax.nn as hnn

from levanter.layers.attention import AttentionMask
from levanter.models.olmo import Olmo2Attention, Olmo2Config, Olmo2DecoderLayer, Olmo2LMHeadModel
from levanter.utils.jax_utils import parameter_count
from test_utils import skip_if_no_torch


def _get_olmo2_config(use_flash=False, num_kv_heads=4, seq_len=128) -> Olmo2Config:
    return Olmo2Config(
        seq_len=seq_len,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=4,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
    )


def _get_random_inputs(config: Olmo2Config, override_Pos=None):
    Embed = config.Embed
    if override_Pos is not None:
        Pos = override_Pos
    else:
        Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = AttentionMask.causal()

    return x, mask


@skip_if_no_torch
def test_olmo2_config():
    # Check we can create a config
    config = _get_olmo2_config()

    # Check that model axes are properly set
    assert config.Pos.size == 128
    assert config.Embed.size == 16
    assert config.Heads.size == 4
    assert config.KVHeads.size == 4
    assert config.Layers.size == 4
    assert config.Mlp.size == 32
    assert config.HeadSize.size == 4

    # Check HF config conversion
    hf_config = config.to_hf_config(vocab_size=100352)
    assert hf_config.hidden_size == 16
    assert hf_config.intermediate_size == 32
    assert hf_config.max_position_embeddings == 128
    assert hf_config.num_attention_heads == 4
    assert hf_config.num_key_value_heads == 4

    # Convert back and check fields
    config2 = Olmo2Config.from_hf_config(hf_config)
    assert config2.hidden_dim == 16
    assert config2.intermediate_dim == 32
    assert config2.seq_len == 128
    assert config2.num_heads == 4
    assert config2.num_kv_heads == 4


@skip_if_no_torch
def test_olmo2_rms_norm():
    import torch
    from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm as HFOlmo2RMSNorm

    config = _get_olmo2_config()
    ln = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_weight=config.use_layer_norm_weight)
    hf_ln = HFOlmo2RMSNorm(config.Embed.size, eps=config.layer_norm_epsilon)

    x, _ = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))

    out = ln(x)
    hf_out = hf_ln(x_torch)

    assert np.isclose(
        hf_out.detach().cpu().numpy(), np.array(out.array), rtol=1e-6, atol=1e-6
    ).all(), f"{hf_out} != {out}"


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_mlp(num_kv_heads):
    config = _get_olmo2_config(num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)

    # Direct reference to Olmo2MLP instead of going through model_type
    from levanter.models.olmo import Olmo2MLP

    mlp = Olmo2MLP.init(config.Embed, config.Mlp, config.activation_function, key=key, use_bias=config.use_bias)

    x, _ = _get_random_inputs(config)
    out = mlp(x)

    # Check output has correct shape
    assert out.array.shape == x.array.shape
    assert out.axes == x.axes


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_attention_vs_hf(use_flash, num_kv_heads):
    import torch
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention as HFOlmo2Attention
    from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding as HFOlmo2RotaryEmbedding

    config = _get_olmo2_config(use_flash=use_flash, num_kv_heads=num_kv_heads)

    attention = Olmo2Attention.init(config=config, key=random.PRNGKey(0))  # type: ignore

    state = hax.state_dict.to_torch_compatible_state_dict(attention)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = config.to_hf_config(32000)

    hf_rotary_emb = HFOlmo2RotaryEmbedding(config=hf_config)
    hf_attention = HFOlmo2Attention(hf_config, layer_idx=0)
    hf_attention.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(config.Pos, config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    position_ids = torch.arange(config.Pos.size).unsqueeze(0)  # [1, seq_len]
    cos, sin = hf_rotary_emb(x_torch, position_ids)  # Pass x_torch instead of zeros tensor
    hf_out = hf_attention(
        x_torch, position_ids=position_ids, attention_mask=mask_torch, position_embeddings=(cos, sin)
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), out.array, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_decoder_layer_vs_hf(num_kv_heads):
    import torch
    from transformers.models.olmo2.modeling_olmo2 import Olmo2DecoderLayer as HFOlmo2DecoderLayer
    from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding as HFOlmo2RotaryEmbedding

    olmo2_config = _get_olmo2_config(num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)
    olmo2_decoder_layer = Olmo2DecoderLayer.init(config=olmo2_config, key=key)

    state = hax.state_dict.to_torch_compatible_state_dict(olmo2_decoder_layer)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = olmo2_config.to_hf_config(32000)
    hf_decoder_layer = HFOlmo2DecoderLayer(hf_config, layer_idx=0)
    hf_decoder_layer.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(olmo2_config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(olmo2_config.Pos, olmo2_config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e10

    position_ids = torch.arange(olmo2_config.Pos.size).unsqueeze(0)
    hf_rotary_emb = HFOlmo2RotaryEmbedding(config=hf_config)
    cos, sin = hf_rotary_emb(x_torch, position_ids)

    out = olmo2_decoder_layer(x, mask)
    hf_out = hf_decoder_layer(
        x_torch, attention_mask=mask_torch, position_ids=position_ids, position_embeddings=(cos, sin)
    )

    # Handle the case where HF returns separate batch elements vs single tensor
    if isinstance(hf_out, torch.Tensor):
        hf_array = hf_out.detach().cpu().numpy()
    else:
        hf_stacked = torch.stack(hf_out)
        hf_array = hf_stacked.detach().cpu().numpy()

    chex.assert_trees_all_close(hf_array, out.array, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_attention(use_flash, num_kv_heads):
    config = _get_olmo2_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)

    attention = Olmo2Attention.init(config=config, key=key)

    x, mask = _get_random_inputs(config)
    out = attention(x, mask)

    # Check output has correct shape
    assert out.array.shape == x.array.shape
    assert out.axes == x.axes


@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_decoder_layer(use_flash, num_kv_heads):
    config = _get_olmo2_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)

    layer = Olmo2DecoderLayer.init(config=config, key=key)

    x, mask = _get_random_inputs(config)
    out = layer(x, mask)

    # Check output has correct shape
    assert out.array.shape == x.array.shape
    assert out.axes == x.axes


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_lm_head_model(num_kv_heads):
    config = _get_olmo2_config(num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    olmo2_model = Olmo2LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))
    out = olmo2_model(input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo2_lm_head_model_bwd(use_flash, num_kv_heads):
    config = _get_olmo2_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    olmo2_model = Olmo2LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))

    def f(olmo2_model, input_ids, mask):
        out = olmo2_model(input_ids, mask)
        return hax.sum(out).scalar()

    _, grads = eqx.filter_value_and_grad(f)(olmo2_model, input_ids, mask)

    # Check that we can compute gradients
    assert grads is not None


@skip_if_no_torch
@pytest.mark.parametrize("scan_layers", [True, False])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_olmo2_roundtrip(scan_layers, num_kv_heads):
    import torch
    from transformers import AutoModelForCausalLM, Olmo2ForCausalLM

    converter = Olmo2Config().hf_checkpoint_converter()

    config = Olmo2Config(
        seq_len=128,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        num_layers=4,
        gradient_checkpointing=False,
        scan_layers=scan_layers,
    )
    # olmo2 gets mad if vocab size < 100278
    Vocab = hax.Axis("vocab", 150000)
    hf_config = config.to_hf_config(Vocab.size)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    # Create HF model with our config
    torch_model = Olmo2ForCausalLM(hf_config)
    torch_model.eval()

    # Forward pass through HF model
    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF model
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)

        # Check saved files

        # Load into our model
        model = converter.load_pretrained(Olmo2LMHeadModel, ref=model_path, resize_vocab_to_match_tokenizer=False)

        # Forward pass through our model
        @hax.named_jit
        def compute(model, input):
            model_output = model(input, attn_mask=attn_mask)
            return model_output

        jax_out = compute(model, input).array

        # Check shapes match
        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"

        # For more detail on significant differences:
        abs_diff = np.abs(torch_out - jax_out.astype(np.float32))
        max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"\nMaximum difference at {max_diff_idx}: {abs_diff[max_diff_idx]}")
        print(f"HF value: {torch_out[max_diff_idx]}, JAX value: {jax_out[max_diff_idx]}")

        # Check outputs are close
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        # Save our model
        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)

        # Load saved model into HF
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        # Check forward pass still works
        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        np.testing.assert_allclose(torch_out2, jax_out, rtol=1e-5, atol=1e-5)


def test_olmo2_param_counts_dont_change_with_seqlen():
    model = Olmo2LMHeadModel.init(hax.Axis("v", 2048), _get_olmo2_config(seq_len=128), key=random.PRNGKey(0))
    model2 = Olmo2LMHeadModel.init(hax.Axis("v", 2048), _get_olmo2_config(seq_len=256), key=random.PRNGKey(0))
    assert parameter_count(model) == parameter_count(model2)


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_olmo2_state_dict_consistency(num_kv_heads):
    from transformers import Olmo2ForCausalLM

    config = Olmo2Config(
        seq_len=128,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=4,
        num_layers=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
        use_bias=False,
        scan_layers=True,
    )
    Vocab = hax.Axis("vocab", 1000)
    model = Olmo2LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))
    hf_config = config.to_hf_config(Vocab.size)
    hf_model = Olmo2ForCausalLM(hf_config)
    levanter_state_dict = hax.state_dict.to_torch_compatible_state_dict(model)
    assert set(hf_model.state_dict().keys()) == set(levanter_state_dict.keys())


@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_olmo2_seq_len_doesnt_change_predictions(num_kv_heads):
    config = Olmo2Config(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
    )
    Vocab = hax.Axis("vocab", 1000)

    # Make input and attn_mask
    input_256 = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    input_128 = input_256[config.Pos, :128]
    attn_mask = AttentionMask.causal()

    model = Olmo2LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))

    @hax.named_jit
    def compute(model, input):
        model_output = model(input, attn_mask=attn_mask)
        return model_output

    jax_out_1 = compute(model, input_128)
    jax_out_2 = compute(model, input_256)[config.Pos, :128]

    assert np.allclose(jax_out_1.array, jax_out_2.array, rtol=1e-6, atol=1e-6)
