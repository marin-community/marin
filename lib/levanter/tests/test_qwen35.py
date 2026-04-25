# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from haliax import Axis

from levanter.layers.rotary import PartialRotaryEmbeddingsConfig
from levanter.models.qwen35 import Qwen35Config, Qwen35LMHeadModel


jax.config.update("jax_default_matmul_precision", "float32")


def _small_config(**overrides):
    defaults = dict(
        max_seq_len=128,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        full_attention_interval=4,
        vocab_size=256,
        gradient_checkpointing=False,
    )
    defaults.update(overrides)
    return Qwen35Config(**defaults)


def test_config_layer_types():
    config = _small_config(num_layers=8, full_attention_interval=4)
    types = config.get_layer_types()
    assert len(types) == 8
    # full_attention at positions 3, 7 (every 4th, 0-indexed: (i+1) % 4 == 0)
    for i, lt in enumerate(types):
        if (i + 1) % 4 == 0:
            assert lt == "full_attention", f"Layer {i} should be full_attention"
        else:
            assert lt == "linear_attention", f"Layer {i} should be linear_attention"


def test_config_explicit_layer_types():
    custom = ("linear_attention", "full_attention", "linear_attention", "full_attention")
    config = _small_config(layer_types=custom)
    assert tuple(config.get_layer_types()) == custom


def test_config_roundtrip():
    """from_hf_config -> to_hf_config preserves key fields."""
    config = _small_config()
    hf_config = config.to_hf_config(vocab_size=config.vocab_size)
    config2 = Qwen35Config.from_hf_config(hf_config)

    assert config2.hidden_dim == config.hidden_dim
    assert config2.num_layers == config.num_layers
    assert config2.num_heads == config.num_heads
    assert config2.num_kv_heads == config.num_kv_heads
    assert config2.head_dim == config.head_dim
    assert config2.linear_num_key_heads == config.linear_num_key_heads
    assert config2.linear_num_value_heads == config.linear_num_value_heads
    assert config2.linear_key_head_dim == config.linear_key_head_dim
    assert config2.linear_value_head_dim == config.linear_value_head_dim
    assert config2.tie_word_embeddings == config.tie_word_embeddings
    assert config2.vocab_size == config.vocab_size
    assert list(config2.get_layer_types()) == list(config.get_layer_types())


def test_partial_rope_only_rotates_first_fraction():
    """Partial RoPE should only modify the first partial_rotary_factor * head_dim dims."""
    HeadSize = Axis("head_size", 32)
    partial_factor = 0.25
    rotary_dim = int(HeadSize.size * partial_factor)  # 8

    rope_config = PartialRotaryEmbeddingsConfig(theta=10_000_000.0, partial_rotary_factor=partial_factor)
    rope = rope_config.build(HeadSize)

    Pos = Axis("position", 4)
    pos_ids = hax.arange(Pos)

    # Input: all ones
    q = hax.ones((Pos, HeadSize))
    q_rot = rope(q, pos_ids)

    # First rotary_dim dims should be modified, rest unchanged
    q_rot_rest = q_rot[HeadSize, rotary_dim:]
    q_orig_rest = q[HeadSize, rotary_dim:]

    # The pass-through dims should be identical
    np.testing.assert_allclose(q_rot_rest.array, q_orig_rest.array, atol=1e-6)

    # Position 0 should have identity rotation (cos=1, sin=0)
    q_pos0_rot = q_rot[Pos, 0]
    q_pos0_orig = q[Pos, 0]
    np.testing.assert_allclose(q_pos0_rot.array, q_pos0_orig.array, atol=1e-5)

    # Non-zero positions should differ in the rotary dims
    q_pos1_rot = q_rot[Pos, 1][HeadSize, :rotary_dim]
    q_pos1_orig = q[Pos, 1][HeadSize, :rotary_dim]
    assert not np.allclose(q_pos1_rot.array, q_pos1_orig.array, atol=1e-3)


def test_model_forward_pass():
    """Model init + forward produces correct output shape."""
    config = _small_config()
    Vocab = Axis("vocab", config.vocab_size)
    model = Qwen35LMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 2)
    Pos = Axis("position", 16)
    input_ids = hax.named(jnp.zeros((2, 16), dtype=jnp.int32), (Batch, Pos))
    logits = model(input_ids, key=None)

    assert logits.axes == (Batch, Pos, Vocab)


def test_model_gradient_flow():
    """Gradients flow through both attention and GDN layers."""
    import equinox as eqx

    config = _small_config()
    Vocab = Axis("vocab", config.vocab_size)
    model = Qwen35LMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(42))

    Batch = Axis("batch", 1)
    Pos = Axis("position", 8)
    input_ids = hax.named(jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32), (Batch, Pos))

    @eqx.filter_value_and_grad
    def loss_fn(model):
        logits = model(input_ids, key=None)
        return hax.mean(logits).scalar()

    loss, grads = loss_fn(model)
    assert jnp.isfinite(loss)


def test_state_dict_roundtrip():
    """to/from state dict roundtrip preserves model outputs."""
    from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict

    config = _small_config()
    Vocab = Axis("vocab", config.vocab_size)
    model = Qwen35LMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))

    # Use the torch-compatible roundtrip (flatten→save→load→unflatten)
    # which is the same path HFCheckpointConverter uses.
    sd = to_torch_compatible_state_dict(model)
    model2 = from_torch_compatible_state_dict(model, sd, unflatten=True)

    Batch = Axis("batch", 1)
    Pos = Axis("position", 8)
    input_ids = hax.named(jnp.ones((1, 8), dtype=jnp.int32), (Batch, Pos))
    logits1 = model(input_ids, key=None)
    logits2 = model2(input_ids, key=None)

    np.testing.assert_allclose(logits1.array, logits2.array, atol=1e-5)


def test_state_dict_keys_match_hf_format():
    """State dict keys should match HF checkpoint naming convention."""
    config = _small_config()
    Vocab = Axis("vocab", config.vocab_size)
    model = Qwen35LMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))

    sd = model.to_state_dict()
    keys = sorted(sd.keys())

    # Check GDN layer keys (layer 0)
    gdn_keys = [k for k in keys if "layers.0.linear_attn" in k]
    gdn_suffixes = {k.split("linear_attn.")[1] for k in gdn_keys}
    assert "in_proj_qkv.weight" in gdn_suffixes
    assert "in_proj_z.weight" in gdn_suffixes
    assert "in_proj_a.weight" in gdn_suffixes
    assert "in_proj_b.weight" in gdn_suffixes
    assert "conv1d.weight" in gdn_suffixes
    assert "norm.weight" in gdn_suffixes
    assert "out_proj.weight" in gdn_suffixes
    assert "A_log" in gdn_suffixes
    assert "dt_bias" in gdn_suffixes

    # Check attention layer keys (layer 3)
    attn_keys = [k for k in keys if "layers.3.self_attn" in k]
    attn_suffixes = {k.split("self_attn.")[1] for k in attn_keys}
    assert "q_proj.weight" in attn_suffixes
    assert "k_proj.weight" in attn_suffixes
    assert "v_proj.weight" in attn_suffixes
    assert "o_proj.weight" in attn_suffixes
    assert "q_norm.weight" in attn_suffixes
    assert "k_norm.weight" in attn_suffixes

    # Check embeddings and norm
    assert "language_model.embed_tokens.weight" in keys
    assert "language_model.norm.weight" in keys


@pytest.mark.slow
def test_load_hf_checkpoint():
    """Load Qwen3.5-0.8B from HuggingFace and run a forward pass."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download("Qwen/Qwen3.5-0.8B", "config.json")
    except Exception:
        pytest.skip("Qwen3.5-0.8B not accessible")

    model = Qwen35LMHeadModel.load_from_hf_checkpoint("Qwen/Qwen3.5-0.8B")

    Batch = Axis("batch", 1)
    Pos = Axis("position", 8)
    input_ids = hax.named(jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32), (Batch, Pos))
    logits = model(input_ids, key=None)

    # Sanity checks on logits
    assert logits.axes[0].name == "batch"
    assert logits.axes[1].name == "position"
    assert logits.axes[2].size == 248320
    assert jnp.all(jnp.isfinite(logits.array))
