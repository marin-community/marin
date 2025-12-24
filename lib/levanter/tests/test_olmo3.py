# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
Test suite for OLMo 3 model implementation.

OLMo 3 key differences from OLMo 2:
- Mixed attention strategy: 3 out of 4 layers use sliding window attention,
  every 4th layer uses full attention
- `layer_types` parameter specifies attention pattern per layer
- `sliding_window` parameter (default 4096 tokens)
"""

import tempfile

import chex
import numpy as np
import pytest
from jax import random

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.utils.jax_utils import parameter_count
from test_utils import skip_if_module_missing, skip_if_no_torch, use_test_mesh

pytestmark = skip_if_module_missing("transformers")


def _get_olmo3_config(use_flash=False, num_kv_heads=4, seq_len=128, sliding_window=64):
    """Create a minimal OLMo3 config for testing.

    Args:
        use_flash: Whether to use flash attention
        num_kv_heads: Number of key-value heads (for GQA)
        seq_len: Maximum sequence length
        sliding_window: Size of sliding window for sliding attention layers
    """
    # Import here to avoid import errors before implementation exists
    from levanter.models.olmo3 import Olmo3Config

    return Olmo3Config(
        max_seq_len=seq_len,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=4,  # 4 layers to test the 3:1 sliding:full pattern
        num_heads=4,
        num_kv_heads=num_kv_heads,
        sliding_window=sliding_window,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
    )


def _get_random_inputs(config, override_Pos=None):
    """Generate random inputs for testing."""
    Embed = config.Embed
    if override_Pos is not None:
        Pos = override_Pos
    else:
        Pos = config.max_Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = AttentionMask.causal()

    return x, mask


def _get_olmo3_attention(config, layer_idx: int, key):
    return config.init_attention(layer_idx, key=key)


def test_olmo3_config_hf_roundtrip():
    """Test HF config conversion roundtrip."""
    config = _get_olmo3_config()

    # Convert to HF config
    hf_config = config.to_hf_config(vocab_size=50304)
    assert hf_config.hidden_size == 16
    assert hf_config.intermediate_size == 32
    assert hf_config.max_position_embeddings == 128
    assert hf_config.num_attention_heads == 4
    assert hf_config.num_key_value_heads == 4

    # Convert back and check fields
    from levanter.models.olmo3 import Olmo3Config

    config2 = Olmo3Config.from_hf_config(hf_config)
    assert config2.hidden_dim == 16
    assert config2.intermediate_dim == 32
    assert config2.max_seq_len == 128
    assert config2.num_heads == 4
    assert config2.num_kv_heads == 4


def test_olmo3_layer_types():
    """Test that layer types follow the 3:1 sliding:full pattern by default."""
    config = _get_olmo3_config()

    # With 4 layers, expect: sliding, sliding, sliding, full (indices 0,1,2,3)
    # Default pattern: layers 0,1,2 are sliding, layer 3 is full
    layer_types = config.get_layer_types()

    assert len(layer_types) == 4
    # Every 4th layer (0-indexed: 3, 7, 11...) should be full attention
    assert layer_types[0] == "sliding_attention"
    assert layer_types[1] == "sliding_attention"
    assert layer_types[2] == "sliding_attention"
    assert layer_types[3] == "full_attention"


def test_olmo3_layer_types_8_layers():
    """Test layer types with 8 layers."""
    from levanter.models.olmo3 import Olmo3Config

    config = Olmo3Config(
        max_seq_len=128,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=8,
        num_heads=4,
        num_kv_heads=4,
    )

    layer_types = config.get_layer_types()
    assert len(layer_types) == 8

    # Pattern: sliding, sliding, sliding, full, sliding, sliding, sliding, full
    expected = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert layer_types == expected


def test_olmo3_custom_layer_types():
    """Test that custom layer_types override the default pattern."""
    from levanter.models.olmo3 import Olmo3Config, Olmo3LMHeadModel

    # Custom pattern: alternating full/sliding
    custom_pattern = [
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]

    config = Olmo3Config(
        max_seq_len=128,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        sliding_window=32,
        layer_types=custom_pattern,
    )

    # Verify get_layer_types returns custom pattern
    assert list(config.get_layer_types()) == custom_pattern

    # Build model and verify each layer has correct attention type
    Vocab = hax.Axis("vocab", 1000)
    model = Olmo3LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))

    for i, layer in enumerate(model.transformer.layers):
        expected = custom_pattern[i]
        if expected == "sliding_attention":
            assert layer.self_attn.config.sliding_window == config.sliding_window
        else:
            assert layer.self_attn.config.sliding_window is None

    # Verify model runs successfully
    Batch = hax.Axis("batch", 2)
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, config.max_Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    out = model(input_ids, mask)
    assert out.array.shape == (Batch.size, config.max_Pos.size, Vocab.size)


def test_olmo3_sliding_window_config():
    """Test sliding window configuration."""
    config = _get_olmo3_config(sliding_window=64)
    assert config.sliding_window == 64

    hf_config = config.to_hf_config(vocab_size=50304)
    assert hf_config.sliding_window == 64


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo3_attention_vs_hf(use_flash, num_kv_heads):
    """Test attention matches HuggingFace implementation."""
    import torch
    from transformers.models.olmo3.modeling_olmo3 import Olmo3Attention as HFOlmo3Attention
    from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding as HFOlmo3RotaryEmbedding

    config = _get_olmo3_config(use_flash=use_flash, num_kv_heads=num_kv_heads)

    # Use layer_idx=3 for full attention comparison
    attention = _get_olmo3_attention(config, layer_idx=3, key=random.PRNGKey(0))

    state = hax.state_dict.to_torch_compatible_state_dict(attention)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = config.to_hf_config(32000)

    hf_rotary_emb = HFOlmo3RotaryEmbedding(config=hf_config)
    hf_attention = HFOlmo3Attention(hf_config, layer_idx=3)
    hf_attention.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(config.max_Pos, config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    position_ids = torch.arange(config.max_Pos.size).unsqueeze(0)
    cos, sin = hf_rotary_emb(x_torch, position_ids)
    hf_out = hf_attention(
        x_torch, position_ids=position_ids, attention_mask=mask_torch, position_embeddings=(cos, sin)
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), out.array, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("layer_idx", [0, 3])  # 0 = sliding, 3 = full
def test_olmo3_attention_layer_type_detection(layer_idx):
    """Test that attention correctly detects its layer type."""
    config = _get_olmo3_config()
    attention = _get_olmo3_attention(config, layer_idx=layer_idx, key=random.PRNGKey(0))

    expected_sliding = (layer_idx + 1) % 4 != 0
    if expected_sliding:
        assert attention.config.sliding_window == config.sliding_window
    else:
        assert attention.config.sliding_window is None


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("layer_idx", [0, 3])  # Test both sliding and full attention layers
def test_olmo3_decoder_layer_vs_hf(num_kv_heads, layer_idx):
    """Test decoder layer matches HuggingFace implementation."""
    import torch
    from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer as HFOlmo3DecoderLayer
    from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding as HFOlmo3RotaryEmbedding

    from levanter.models.olmo3 import Olmo3DecoderLayer

    olmo3_config = _get_olmo3_config(num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)
    olmo3_decoder_layer = Olmo3DecoderLayer.init(config=olmo3_config, layer_idx=layer_idx, key=key)

    state = hax.state_dict.to_torch_compatible_state_dict(olmo3_decoder_layer)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = olmo3_config.to_hf_config(32000)
    hf_decoder_layer = HFOlmo3DecoderLayer(hf_config, layer_idx=layer_idx)
    hf_decoder_layer.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(olmo3_config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]

    # For sliding attention layers, apply sliding window to the mask for HF
    # (Levanter applies it internally in Attention)
    layer_types = olmo3_config.get_layer_types()
    mask_for_hf = mask
    if layer_types[layer_idx] == "sliding_attention":
        mask_for_hf = mask.with_sliding_window(olmo3_config.sliding_window)

    explicit_mask = torch.from_numpy(
        np.array(mask_for_hf.materialize(olmo3_config.max_Pos, olmo3_config.KeyPos).array)
    )
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e10

    position_ids = torch.arange(olmo3_config.max_Pos.size).unsqueeze(0)
    hf_rotary_emb = HFOlmo3RotaryEmbedding(config=hf_config)
    cos, sin = hf_rotary_emb(x_torch, position_ids)

    out = olmo3_decoder_layer(x, mask)
    hf_out = hf_decoder_layer(
        x_torch, attention_mask=mask_torch, position_ids=position_ids, position_embeddings=(cos, sin)
    )

    if isinstance(hf_out, torch.Tensor):
        hf_array = hf_out.detach().cpu().numpy()
    else:
        hf_stacked = torch.stack(hf_out)
        hf_array = hf_stacked.detach().cpu().numpy()

    chex.assert_trees_all_close(hf_array, out.array, rtol=1e-5, atol=1e-5)


def test_olmo3_sliding_vs_full_attention_differ():
    """Test that sliding and full attention produce different outputs."""
    # Use seq_len much larger than sliding_window
    config = _get_olmo3_config(sliding_window=16, seq_len=64)

    # Same initialization for fair comparison
    key = random.PRNGKey(42)

    # Create sliding attention (layer 0)
    sliding_attention = _get_olmo3_attention(config, layer_idx=0, key=key)

    # Create full attention (layer 3)
    full_attention = _get_olmo3_attention(config, layer_idx=3, key=key)

    x, mask = _get_random_inputs(config)

    sliding_out = sliding_attention(x, mask)
    full_out = full_attention(x, mask)

    # Outputs should differ due to different attention patterns
    # (unless seq_len <= sliding_window, in which case they'd be the same)
    assert not np.allclose(sliding_out.array, full_out.array, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
@pytest.mark.parametrize("scan_layers", [True, False])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_olmo3_roundtrip(scan_layers, num_kv_heads):
    """Test save/load roundtrip with HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, Olmo3ForCausalLM

    from levanter.models.olmo3 import Olmo3Config, Olmo3LMHeadModel

    converter = Olmo3Config().hf_checkpoint_converter()

    config = Olmo3Config(
        max_seq_len=128,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        num_layers=4,
        sliding_window=64,
        gradient_checkpointing=False,
        scan_layers=scan_layers,
    )
    Vocab = hax.Axis("vocab", 150000)
    hf_config = config.to_hf_config(Vocab.size)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.max_Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    # Create HF model with our config
    torch_model = Olmo3ForCausalLM(hf_config)
    torch_model.eval()

    # Forward pass through HF model
    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        # Save HF model
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)

        # Load into our model
        model = converter.load_pretrained(Olmo3LMHeadModel, ref=model_path, resize_vocab_to_match_tokenizer=False)

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


def test_olmo3_param_counts_dont_change_with_seqlen():
    """Test that parameter counts are independent of sequence length."""
    from levanter.models.olmo3 import Olmo3LMHeadModel

    model = Olmo3LMHeadModel.init(hax.Axis("v", 2048), _get_olmo3_config(seq_len=128), key=random.PRNGKey(0))
    model2 = Olmo3LMHeadModel.init(hax.Axis("v", 2048), _get_olmo3_config(seq_len=256), key=random.PRNGKey(0))
    assert parameter_count(model) == parameter_count(model2)


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_olmo3_state_dict_consistency(num_kv_heads):
    """Test state dict keys match HuggingFace."""
    from transformers import Olmo3ForCausalLM

    from levanter.models.olmo3 import Olmo3Config, Olmo3LMHeadModel

    config = Olmo3Config(
        max_seq_len=128,
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
    model = Olmo3LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))
    hf_config = config.to_hf_config(Vocab.size)
    hf_model = Olmo3ForCausalLM(hf_config)
    levanter_state_dict = hax.state_dict.to_torch_compatible_state_dict(model)
    assert set(hf_model.state_dict().keys()) == set(levanter_state_dict.keys())


@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_olmo3_seq_len_doesnt_change_predictions(num_kv_heads):
    """Test that predictions are consistent across sequence lengths."""
    from levanter.models.olmo3 import Olmo3Config, Olmo3LMHeadModel

    config = Olmo3Config(
        max_seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
    )
    Vocab = hax.Axis("vocab", 1000)

    # Make input and attn_mask
    input_256 = hax.random.randint(random.PRNGKey(0), config.max_Pos, 0, Vocab.size)
    input_128 = input_256[config.max_Pos, :128]
    attn_mask = AttentionMask.causal()

    model = Olmo3LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))

    @hax.named_jit
    def compute(model, input):
        model_output = model(input, attn_mask=attn_mask)
        return model_output

    jax_out_1 = compute(model, input_128)
    jax_out_2 = compute(model, input_256)[config.max_Pos, :128]

    assert np.allclose(jax_out_1.array, jax_out_2.array, rtol=1e-6, atol=1e-6)


def test_olmo3_all_layers_have_correct_attention_type():
    """Test that all layers in a model have the expected attention type."""
    from levanter.models.olmo3 import Olmo3Config, Olmo3LMHeadModel

    config = Olmo3Config(
        max_seq_len=128,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=8,
        num_heads=4,
        num_kv_heads=4,
    )
    Vocab = hax.Axis("vocab", 1000)
    model = Olmo3LMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))

    expected_types = config.get_layer_types()

    # Access transformer layers and verify attention types
    for i, layer in enumerate(model.transformer.layers):
        expected = expected_types[i]
        if expected == "sliding_attention":
            assert layer.self_attn.config.sliding_window == config.sliding_window
        else:
            assert layer.self_attn.config.sliding_window is None
