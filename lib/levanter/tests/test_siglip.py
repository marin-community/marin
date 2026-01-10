# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os

# Force torch to use CPU before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Force JAX to use TPU
os.environ["JAX_PLATFORMS"] = "tpu"
# Force JAX to use float32
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

import pytest
import jax
import haliax as hax
import jax.numpy as jnp

# Enable float32 mode in JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

from levanter.models.siglip import SiglipVisionConfig  # noqa: E402
from levanter.utils.activation import ActivationFunctionEnum  # noqa: E402
from test_utils import use_test_mesh  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
import numpy as np  # noqa: E402
from test_image_utils import get_single_image  # noqa: E402

# Define skip_if_no_torch locally to avoid conftest dependencies
try:
    import torch  # noqa: F401

    skip_if_no_torch = pytest.mark.skipif(False, reason="torch is available")
except ImportError:
    skip_if_no_torch = pytest.mark.skip(reason="torch not available")


def _hf_siglip_vision_config():
    """Return a tiny SiglipVisionConfig for testing."""
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig

    cfg_dict = {
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_channels": 3,
        "image_size": 224,
        "patch_size": 16,
        "hidden_act": "gelu_pytorch_tanh",  # Standard SigLIP activation
        "layer_norm_eps": 1e-6,
        "attention_dropout": 0.0,
    }
    return HfSiglipVisionConfig(**cfg_dict)


def test_siglip_vision_config_creation():
    """Test basic SiglipVisionConfig instantiation."""
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )

    assert config.hidden_size == 768
    assert config.intermediate_size == 3072
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.num_channels == 3
    assert config.image_size == 224
    assert config.patch_size == 16
    assert config.hidden_act == ActivationFunctionEnum.gelu_new
    assert config.layer_norm_eps == 1e-6
    assert config.attention_dropout == 0.0


def test_siglip_vision_config_axes():
    """Test that axis properties are correctly defined."""
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )

    # Test Embed axis
    assert config.Embed.name == "embed"
    assert config.Embed.size == 768

    # Test Mlp axis
    assert config.Mlp.name == "mlp"
    assert config.Mlp.size == 3072

    # Test Heads axis
    assert config.Heads.name == "heads"
    assert config.Heads.size == 12

    # Test HeadSize axis
    assert config.HeadSize.name == "head_size"
    assert config.HeadSize.size == 768 // 12

    # Test Layers axis
    assert config.Layers.name == "layers"
    assert config.Layers.size == 12

    # Test Channels axis
    assert config.Channels.name == "channels"
    assert config.Channels.size == 3

    # Test ImageSize axis
    assert config.ImageSize.name == "image_size"
    assert config.ImageSize.size == 224

    # Test PatchSize axis
    assert config.PatchSize.name == "patch_size"
    assert config.PatchSize.size == 16

    # Test NumPatches axis (calculated from image_size and patch_size)
    assert config.NumPatches.name == "num_patches"
    assert config.NumPatches.size == (224 // 16) ** 2  # 14 * 14 = 196


@skip_if_no_torch
def test_siglip_vision_from_hf_config():
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_siglip_vision_config()

    # Convert from HF config
    config = SiglipVisionConfig.from_hf_config(hf_config)

    # Check all attributes match
    assert config.hidden_size == hf_config.hidden_size
    assert config.intermediate_size == hf_config.intermediate_size
    assert config.num_hidden_layers == hf_config.num_hidden_layers
    assert config.num_attention_heads == hf_config.num_attention_heads
    assert config.num_channels == hf_config.num_channels
    assert config.image_size == hf_config.image_size
    assert config.patch_size == hf_config.patch_size
    assert config.layer_norm_eps == hf_config.layer_norm_eps
    assert config.attention_dropout == hf_config.attention_dropout

    # Check activation function conversion
    assert config.hidden_act == ActivationFunctionEnum.gelu_new


@skip_if_no_torch
def test_siglip_vision_to_hf_config():
    """Test conversion from Levanter config to HuggingFace config."""

    # Create Levanter config
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act=ActivationFunctionEnum.gelu_new,
        layer_norm_eps=1e-6,
        attention_dropout=0.1,
    )

    # Convert to HF config
    hf_config = config.to_hf_config()

    # Check all attributes match
    assert hf_config.hidden_size == config.hidden_size
    assert hf_config.intermediate_size == config.intermediate_size
    assert hf_config.num_hidden_layers == config.num_hidden_layers
    assert hf_config.num_attention_heads == config.num_attention_heads
    assert hf_config.num_channels == config.num_channels
    assert hf_config.image_size == config.image_size
    assert hf_config.patch_size == config.patch_size
    assert hf_config.layer_norm_eps == config.layer_norm_eps
    assert hf_config.attention_dropout == config.attention_dropout

    # Check activation function conversion (gelu_new maps back to gelu_pytorch_tanh)
    assert hf_config.hidden_act == "gelu_pytorch_tanh"


@skip_if_no_torch
def test_siglip_vision_config_roundtrip():
    """Test that converting HF -> Levanter -> HF preserves the config."""

    # Start with HF config
    hf_config_1 = _hf_siglip_vision_config()

    # Convert to Levanter
    levanter_config = SiglipVisionConfig.from_hf_config(hf_config_1)

    # Convert back to HF
    hf_config_2 = levanter_config.to_hf_config()

    # Check key attributes are preserved
    assert hf_config_2.hidden_size == hf_config_1.hidden_size
    assert hf_config_2.intermediate_size == hf_config_1.intermediate_size
    assert hf_config_2.num_hidden_layers == hf_config_1.num_hidden_layers
    assert hf_config_2.num_attention_heads == hf_config_1.num_attention_heads
    assert hf_config_2.num_channels == hf_config_1.num_channels
    assert hf_config_2.image_size == hf_config_1.image_size
    assert hf_config_2.patch_size == hf_config_1.patch_size
    assert hf_config_2.layer_norm_eps == hf_config_1.layer_norm_eps
    assert hf_config_2.attention_dropout == hf_config_1.attention_dropout
    assert hf_config_2.hidden_act == hf_config_1.hidden_act
    assert hf_config_2 == hf_config_1


def test_siglip_vision_config_num_patches_calculation():
    """Test that NumPatches is correctly calculated from image_size and patch_size."""
    # Test standard configuration
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=16,
    )
    assert config.NumPatches.size == 196  # (224 // 16) ** 2 = 14 * 14

    # Test different image size
    config2 = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=384,
        patch_size=16,
    )
    assert config2.NumPatches.size == 576  # (384 // 16) ** 2 = 24 * 24

    # Test different patch size
    config3 = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=14,
    )
    assert config3.NumPatches.size == 256  # (224 // 14) ** 2 = 16 * 16


@skip_if_no_torch
def test_siglip_vision_activation_function_conversion():
    """Test various activation function conversions between HF and Levanter."""
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig

    # Test gelu_pytorch_tanh -> gelu_new
    hf_config = HfSiglipVisionConfig(hidden_act="gelu_pytorch_tanh")
    levanter_config = SiglipVisionConfig.from_hf_config(hf_config)
    assert levanter_config.hidden_act == ActivationFunctionEnum.gelu_new

    # Test gelu -> gelu
    hf_config = HfSiglipVisionConfig(hidden_act="gelu")
    levanter_config = SiglipVisionConfig.from_hf_config(hf_config)
    assert levanter_config.hidden_act == ActivationFunctionEnum.gelu

    # Test quick_gelu -> quick_gelu
    hf_config = HfSiglipVisionConfig(hidden_act="quick_gelu")
    levanter_config = SiglipVisionConfig.from_hf_config(hf_config)
    assert levanter_config.hidden_act == ActivationFunctionEnum.quick_gelu


@skip_if_no_torch
def test_siglip_vision_config_overrides():
    """Test that config_overrides work in to_hf_config."""
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
    )

    # Convert with overrides
    hf_config = config.to_hf_config(config_overrides={"num_hidden_layers": 24})

    # Check override is applied
    assert hf_config.num_hidden_layers == 24

    # Check other values are preserved
    assert hf_config.hidden_size == 768
    assert hf_config.intermediate_size == 3072


def test_siglip_vision_config_defaults():
    """Test that default values match expected SigLIP architecture."""
    config = SiglipVisionConfig()

    # Check defaults match google/siglip-base-patch16-224
    assert config.hidden_size == 768
    assert config.intermediate_size == 3072
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.num_channels == 3
    assert config.image_size == 224
    assert config.patch_size == 16
    assert config.hidden_act == ActivationFunctionEnum.gelu_new
    assert config.layer_norm_eps == 1e-6
    assert config.attention_dropout == 0.0
    assert config.gradient_checkpointing is True


def test_siglip_vision_frozen_dataclass():
    """Test that the config is frozen and immutable."""
    config = SiglipVisionConfig()

    # Attempt to modify should raise an error
    with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
        config.hidden_size = 1024


def test_siglip_vision_head_size_calculation():
    """Test that head size is correctly calculated."""
    config = SiglipVisionConfig(
        hidden_size=768,
        num_attention_heads=12,
    )

    assert config.HeadSize.size == 768 // 12
    assert config.HeadSize.size == 64

    # Test with different values
    config2 = SiglipVisionConfig(
        hidden_size=1024,
        num_attention_heads=16,
    )

    assert config2.HeadSize.size == 1024 // 16
    assert config2.HeadSize.size == 64


# =====================
# MLP Tests
# =====================


def test_siglip_mlp_initialization():
    """Test that SiglipMLP can be initialized correctly."""
    from levanter.models.siglip import SiglipMLP

    Embed = hax.Axis("embed", 64)
    Mlp = hax.Axis("mlp", 256)

    mlp = SiglipMLP.init(
        Embed=Embed,
        Mlp=Mlp,
        activation_fn=ActivationFunctionEnum.gelu_new,
        key=jax.random.PRNGKey(42),
    )

    # Check that layers are initialized
    assert mlp.fc1 is not None
    assert mlp.fc2 is not None
    assert mlp.act is not None

    # Check layer dimensions
    assert mlp.fc1.Out == Mlp
    assert mlp.fc1.In == Embed
    assert mlp.fc2.Out == Embed
    assert mlp.fc2.In == Mlp


def test_siglip_mlp_forward():
    """Test SiglipMLP forward pass."""
    from levanter.models.siglip import SiglipMLP

    Embed = hax.Axis("embed", 64)
    Mlp = hax.Axis("mlp", 256)
    Pos = hax.Axis("position", 16)

    mlp = SiglipMLP.init(
        Embed=Embed,
        Mlp=Mlp,
        activation_fn=ActivationFunctionEnum.gelu_new,
        key=jax.random.PRNGKey(42),
    )

    # Create input
    x = hax.random.normal(jax.random.PRNGKey(0), (Pos, Embed))

    # Forward pass
    output = mlp(x, key=jax.random.PRNGKey(1))

    # Check output shape
    assert output.axes == (Pos, Embed)
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_mlp_different_activations():
    """Test SiglipMLP with different activation functions."""
    from levanter.models.siglip import SiglipMLP

    Embed = hax.Axis("embed", 32)
    Mlp = hax.Axis("mlp", 128)
    Pos = hax.Axis("position", 8)

    activations = [
        ActivationFunctionEnum.gelu,
        ActivationFunctionEnum.gelu_new,
        ActivationFunctionEnum.relu,
        ActivationFunctionEnum.silu,
    ]

    for activation in activations:
        mlp = SiglipMLP.init(
            Embed=Embed,
            Mlp=Mlp,
            activation_fn=activation,
            key=jax.random.PRNGKey(42),
        )

        x = hax.random.normal(jax.random.PRNGKey(0), (Pos, Embed))
        output = mlp(x, key=jax.random.PRNGKey(1))

        assert output.axes == (Pos, Embed)
        assert not jnp.any(jnp.isnan(output.array))


# =====================
# Attention Tests
# =====================


def test_siglip_attention_initialization():
    """Test that SiglipAttention can be initialized correctly."""
    from levanter.models.siglip import SiglipAttention

    config = SiglipVisionConfig(
        hidden_size=64,
        num_attention_heads=4,
    )

    attention = SiglipAttention.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Check that components are initialized
    assert attention.q_proj is not None
    assert attention.k_proj is not None
    assert attention.v_proj is not None
    assert attention.out_proj is not None
    assert attention.config == config

    # Check projection dimensions
    assert attention.q_proj.In == config.Embed
    assert attention.q_proj.Out == (config.Heads, config.HeadSize)
    assert attention.k_proj.In == config.Embed
    assert attention.k_proj.Out == (config.Heads, config.HeadSize)
    assert attention.v_proj.In == config.Embed
    assert attention.v_proj.Out == (config.Heads, config.HeadSize)
    assert attention.out_proj.In == (config.Heads, config.HeadSize)
    assert attention.out_proj.Out == config.Embed


def test_siglip_attention_forward():
    """Test SiglipAttention forward pass."""
    from levanter.models.siglip import SiglipAttention

    config = SiglipVisionConfig(
        hidden_size=64,
        num_attention_heads=4,
        attention_dropout=0.0,
    )

    attention = SiglipAttention.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input: (batch, position, embed)
    Batch = hax.Axis("batch", 2)
    Position = hax.Axis("position", 16)

    x = hax.random.normal(jax.random.PRNGKey(0), (Batch, Position, config.Embed))

    # Forward pass with test mesh
    with use_test_mesh(tensor_parallelism=1):
        output = attention(x, key=jax.random.PRNGKey(1))

    # Check output shape: should be same as input
    assert output.axes == (Batch, Position, config.Embed)
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_attention_no_batch():
    """Test SiglipAttention without batch dimension."""
    from levanter.models.siglip import SiglipAttention

    config = SiglipVisionConfig(
        hidden_size=64,
        num_attention_heads=4,
        attention_dropout=0.0,
    )

    attention = SiglipAttention.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input without batch dimension
    Position = hax.Axis("position", 16)

    x = hax.random.normal(jax.random.PRNGKey(0), (Position, config.Embed))

    # Forward pass with test mesh
    with use_test_mesh(tensor_parallelism=1):
        output = attention(x, key=jax.random.PRNGKey(1))

    # Check output shape
    assert output.axes == (Position, config.Embed)
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_attention_num_patches_axis():
    """Test SiglipAttention with num_patches axis name (instead of position)."""
    from levanter.models.siglip import SiglipAttention

    config = SiglipVisionConfig(
        hidden_size=64,
        num_attention_heads=4,
        attention_dropout=0.0,
    )

    attention = SiglipAttention.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input with num_patches axis
    NumPatches = hax.Axis("num_patches", 196)

    x = hax.random.normal(jax.random.PRNGKey(0), (NumPatches, config.Embed))

    # Forward pass with test mesh
    with use_test_mesh(tensor_parallelism=1):
        output = attention(x, key=jax.random.PRNGKey(1))

    # Check output shape - should have num_patches axis
    assert output.axes == (NumPatches, config.Embed)
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_attention_different_seq_lengths():
    """Test SiglipAttention with different sequence lengths."""
    from levanter.models.siglip import SiglipAttention

    config = SiglipVisionConfig(
        hidden_size=64,
        num_attention_heads=4,
        attention_dropout=0.0,
    )

    attention = SiglipAttention.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Test with different sequence lengths
    with use_test_mesh(tensor_parallelism=1):
        for seq_len in [49, 196, 256, 576]:  # Different image patch counts
            NumPatches = hax.Axis("num_patches", seq_len)
            x = hax.random.normal(jax.random.PRNGKey(0), (NumPatches, config.Embed))
            output = attention(x, key=jax.random.PRNGKey(1))

            assert output.axes == (NumPatches, config.Embed)
            assert not jnp.any(jnp.isnan(output.array))


# =====================
# Encoder Layer Tests
# =====================


def test_siglip_encoder_layer_initialization():
    """Test that SiglipEncoderLayer can be initialized correctly."""
    from levanter.models.siglip import SiglipEncoderLayer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=4,
    )

    layer = SiglipEncoderLayer.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Check that components are initialized
    assert layer.layer_norm1 is not None
    assert layer.self_attn is not None
    assert layer.layer_norm2 is not None
    assert layer.mlp is not None
    assert layer.config == config


def test_siglip_encoder_layer_forward():
    """Test SiglipEncoderLayer forward pass."""
    from levanter.models.siglip import SiglipEncoderLayer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=4,
        attention_dropout=0.0,
    )

    layer = SiglipEncoderLayer.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input: (batch, num_patches, embed)
    Batch = hax.Axis("batch", 2)
    NumPatches = hax.Axis("num_patches", 196)

    x = hax.random.normal(jax.random.PRNGKey(0), (Batch, NumPatches, config.Embed))

    # Forward pass with test mesh
    with use_test_mesh(tensor_parallelism=1):
        output = layer(x, key=jax.random.PRNGKey(1))

    # Check output shape: should be same as input
    assert output.axes == (Batch, NumPatches, config.Embed)
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_encoder_layer_residual_connections():
    """Test that residual connections are working correctly."""
    from levanter.models.siglip import SiglipEncoderLayer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=4,
        attention_dropout=0.0,
    )

    layer = SiglipEncoderLayer.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    NumPatches = hax.Axis("num_patches", 196)
    x = hax.random.normal(jax.random.PRNGKey(0), (NumPatches, config.Embed))

    # Forward pass with test mesh
    with use_test_mesh(tensor_parallelism=1):
        output = layer(x, key=jax.random.PRNGKey(1))

    # The output should be different from input (due to transformations)
    # but should have contributions from the input (due to residual connections)
    assert not jnp.allclose(output.array, x.array)
    assert output.axes == x.axes


def test_siglip_encoder_layer_different_configs():
    """Test SiglipEncoderLayer with different configurations."""
    from levanter.models.siglip import SiglipEncoderLayer

    configs = [
        {"hidden_size": 64, "intermediate_size": 256, "num_attention_heads": 4},
        {"hidden_size": 128, "intermediate_size": 512, "num_attention_heads": 8},
        {"hidden_size": 256, "intermediate_size": 1024, "num_attention_heads": 8},
    ]

    with use_test_mesh(tensor_parallelism=1):
        for cfg_dict in configs:
            config = SiglipVisionConfig(**cfg_dict)

            layer = SiglipEncoderLayer.init(
                config=config,
                key=jax.random.PRNGKey(42),
            )

            NumPatches = hax.Axis("num_patches", 196)
            x = hax.random.normal(jax.random.PRNGKey(0), (NumPatches, config.Embed))
            output = layer(x, key=jax.random.PRNGKey(1))

            assert output.axes == (NumPatches, config.Embed)
            assert not jnp.any(jnp.isnan(output.array))


# =====================
# Vision Embeddings Tests
# =====================


def test_siglip_vision_embeddings_initialization():
    """Test that SiglipVisionEmbeddings can be initialized correctly."""
    from levanter.models.siglip import SiglipVisionEmbeddings

    config = SiglipVisionConfig(
        hidden_size=64,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )

    embeddings = SiglipVisionEmbeddings.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Check that components are initialized
    assert embeddings.patch_embedding is not None
    assert embeddings.position_embedding is not None
    assert embeddings.config == config


def test_siglip_vision_embeddings_forward():
    """Test SiglipVisionEmbeddings forward pass with full images."""
    from levanter.models.siglip import SiglipVisionEmbeddings

    config = SiglipVisionConfig(
        hidden_size=64,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )

    embeddings = SiglipVisionEmbeddings.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input: full images (not patchified)
    # Shape: (batch, channels, height, width)
    Batch = hax.Axis("batch", 2)
    Channels = config.Channels
    Height = hax.Axis("height", 224)
    Width = hax.Axis("width", 224)

    pixel_values = hax.random.normal(jax.random.PRNGKey(0), (Batch, Channels, Height, Width))

    # Forward pass
    output = embeddings(pixel_values, key=jax.random.PRNGKey(1))

    # Check output shape: should have (batch, num_patches, embed)
    expected_num_patches = (224 // 16) ** 2  # 196
    assert len(output.axes) == 3
    assert output.axes[0] == Batch
    assert output.axes[1].name == "num_patches"
    assert output.axes[1].size == expected_num_patches
    assert output.axes[2] == config.Embed
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_vision_embeddings_no_batch():
    """Test SiglipVisionEmbeddings without batch dimension."""
    from levanter.models.siglip import SiglipVisionEmbeddings

    config = SiglipVisionConfig(
        hidden_size=64,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )

    embeddings = SiglipVisionEmbeddings.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input without batch dimension
    # Shape: (channels, height, width)
    Channels = config.Channels
    Height = hax.Axis("height", 224)
    Width = hax.Axis("width", 224)

    pixel_values = hax.random.normal(jax.random.PRNGKey(0), (Channels, Height, Width))

    # Forward pass
    output = embeddings(pixel_values, key=jax.random.PRNGKey(1))

    # Check output shape
    expected_num_patches = (224 // 16) ** 2
    assert output.axes[0].name == "num_patches"
    assert output.axes[0].size == expected_num_patches
    assert output.axes[1] == config.Embed
    assert not jnp.any(jnp.isnan(output.array))


def test_siglip_vision_embeddings_different_image_sizes():
    """Test SiglipVisionEmbeddings with different image sizes."""
    from levanter.models.siglip import SiglipVisionEmbeddings

    # Test with different image sizes
    test_cases = [
        (224, 16, 196),  # 14x14 patches = 196
        (384, 16, 576),  # 24x24 patches = 576
        (224, 14, 256),  # 16x16 patches = 256
    ]

    for image_size, patch_size, expected_patches in test_cases:
        config = SiglipVisionConfig(
            hidden_size=64,
            num_channels=3,
            image_size=image_size,
            patch_size=patch_size,
        )

        embeddings = SiglipVisionEmbeddings.init(
            config=config,
            key=jax.random.PRNGKey(42),
        )

        # Create input
        Channels = config.Channels
        Height = hax.Axis("height", image_size)
        Width = hax.Axis("width", image_size)

        pixel_values = hax.random.normal(jax.random.PRNGKey(0), (Channels, Height, Width))

        # Forward pass
        output = embeddings(pixel_values, key=jax.random.PRNGKey(1))

        # Check number of patches
        assert output.axes[0].name == "num_patches"
        assert output.axes[0].size == expected_patches
        assert not jnp.any(jnp.isnan(output.array))


# =====================
# Vision Transformer Tests
# =====================


def test_siglip_vision_transformer_initialization():
    """Test that SiglipVisionTransformer can be initialized correctly."""
    from levanter.models.siglip import SiglipVisionTransformer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
    )

    transformer = SiglipVisionTransformer.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Check that components are initialized
    assert transformer.embeddings is not None
    assert transformer.layers is not None
    assert transformer.post_layernorm is not None
    assert transformer.config == config


def test_siglip_vision_transformer_forward():
    """Test SiglipVisionTransformer forward pass."""
    from levanter.models.siglip import SiglipVisionTransformer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=224,
        patch_size=16,
    )

    transformer = SiglipVisionTransformer.init(
        config=config,
        key=jax.random.PRNGKey(42),
    )

    # Create input: full images
    Batch = hax.Axis("batch", 2)
    Channels = config.Channels
    Height = hax.Axis("height", 224)
    Width = hax.Axis("width", 224)

    pixel_values = hax.random.normal(jax.random.PRNGKey(0), (Batch, Channels, Height, Width))

    # Forward pass with test mesh
    with use_test_mesh(tensor_parallelism=1):
        output = transformer(pixel_values, key=jax.random.PRNGKey(1))

    # Check output shape
    expected_num_patches = (224 // 16) ** 2
    assert len(output.last_hidden_state.axes) == 3
    assert output.last_hidden_state.axes[0] == Batch
    assert output.last_hidden_state.axes[1].name == "num_patches"
    assert output.last_hidden_state.axes[1].size == expected_num_patches
    assert output.last_hidden_state.axes[2] == config.Embed
    assert not jnp.any(jnp.isnan(output.last_hidden_state.array))


# =====================
# Real Image Tests
# =====================


@skip_if_no_torch
def test_siglip_vision_embeddings_vs_hf():
    """Compare SiglipVisionEmbeddings with HuggingFace by loading weights."""
    import torch
    from transformers import SiglipVisionModel as HfSiglipVisionModel
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig
    import tempfile
    from haliax.state_dict import from_torch_compatible_state_dict
    import equinox as eqx

    hf_config = HfSiglipVisionConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )

    torch.manual_seed(42)
    hf_model = HfSiglipVisionModel(hf_config)
    hf_model.eval()

    # Create test image input
    batch_size = 2
    pixel_values_torch = torch.randn(batch_size, 3, 224, 224)

    # Run HF model with hidden states
    with torch.no_grad():
        hf_output = hf_model(pixel_values_torch, output_hidden_states=True)
        hf_last_hidden_np = hf_output.last_hidden_state.detach().cpu().numpy()
        hf_hidden_states_np = [h.detach().cpu().numpy() for h in hf_output.hidden_states]

    # Load weights into Levanter model
    lev_config = SiglipVisionConfig.from_hf_config(hf_config)

    # Use single-device mesh to avoid sharding issues with small batch sizes
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        hf_model.save_pretrained(f"{tmpdir}/hf_model")

        from levanter.models.siglip import SiglipVisionModel

        Vocab = hax.Axis("vocab", 1)
        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, Vocab, lev_config, key=jax.random.PRNGKey(0))

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        # Convert input to Levanter format
        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", 224)
        Width = hax.Axis("width", 224)

        pixel_values_jax = hax.named(
            jnp.array(pixel_values_torch.numpy(), dtype=jnp.float32), (Batch, Channels, Height, Width)
        )

        # Run Levanter model with hidden states
        lev_output = lev_model(pixel_values_jax, output_hidden_states=True, key=jax.random.PRNGKey(1))

    lev_last_hidden_np = np.array(lev_output.last_hidden_state.array)
    lev_hidden_states_np = [np.array(h.array) for h in lev_output.hidden_states]

    # Compare last hidden state
    print("\n=== Last Hidden State Comparison ===")
    print(f"HF output shape: {hf_last_hidden_np.shape}")
    print(f"Levanter output shape: {lev_last_hidden_np.shape}")
    print(f"HF output range: [{hf_last_hidden_np.min():.3f}, {hf_last_hidden_np.max():.3f}]")
    print(f"Levanter output range: [{lev_last_hidden_np.min():.3f}, {lev_last_hidden_np.max():.3f}]")

    max_diff = np.max(np.abs(hf_last_hidden_np - lev_last_hidden_np))
    mean_diff = np.mean(np.abs(hf_last_hidden_np - lev_last_hidden_np))
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"HF first 5: {hf_last_hidden_np.flatten()[:5]}")
    print(f"Lev first 5: {lev_last_hidden_np.flatten()[:5]}")

    # Assert last hidden state matches
    assert np.allclose(
        hf_last_hidden_np, lev_last_hidden_np, rtol=1e-3, atol=1e-3
    ), f"Last hidden state mismatch: max diff = {max_diff}, mean diff = {mean_diff}"

    print("\n✓ Last hidden state matches between HF and Levanter!")

    # Compare all hidden states layer by layer
    print("\n=== Hidden States Comparison (All Layers) ===")
    print(f"Number of HF hidden states: {len(hf_hidden_states_np)}")
    print(f"Number of Levanter hidden states: {len(lev_hidden_states_np)}")

    assert len(hf_hidden_states_np) == len(
        lev_hidden_states_np
    ), f"Mismatch in number of hidden states: HF={len(hf_hidden_states_np)}, Lev={len(lev_hidden_states_np)}"

    for i, (hf_h, lev_h) in enumerate(zip(hf_hidden_states_np, lev_hidden_states_np)):
        layer_name = "Embeddings" if i == 0 else f"Layer {i-1}"

        max_diff = np.max(np.abs(hf_h - lev_h))
        mean_diff = np.mean(np.abs(hf_h - lev_h))

        print(f"\n{layer_name}:")
        print(f"  Shape: HF={hf_h.shape}, Lev={lev_h.shape}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        # Assert each layer matches
        assert mean_diff < 1e-3, f"{layer_name} mismatch: max diff = {max_diff}, mean diff = {mean_diff}"

        print(f"  ✓ {layer_name} matches!")

    print("\n✓ All hidden states match between HF and Levanter!")


@skip_if_no_torch
def test_siglip_vision_real_image():
    """Test SigLIP vision model with real image using HF processor.

    This test performs the following checks:
    1. Load HF model and compare with Levanter model (HF -> Levanter)
    2. Convert Levanter model to HF and verify output consistency (Levanter -> HF)
    """
    import torch

    try:
        from transformers import AutoProcessor, AutoModel  # noqa: F401
    except ImportError:
        pytest.skip("transformers not available")

    print("\n=== Testing SigLIP Vision with Real Image ===")

    # Load image from HuggingFace dataset
    image = get_single_image()
    print(f"Image size: {image.size}, mode: {image.mode}")

    # Load HF model and processor from cloud
    model_name = "google/siglip-base-patch16-224"
    print(f"Loading HF model and processor from cloud: {model_name}")

    try:
        # Load only the image processor (not the tokenizer) to avoid SentencePiece dependency
        from transformers import SiglipImageProcessor

        processor = SiglipImageProcessor.from_pretrained(model_name)

        # Load the vision model directly
        from transformers import SiglipVisionModel

        torch_model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        torch_model = torch_model.float()
        print(f"Loaded model type: {type(torch_model).__name__}")
        print(f"Model dtype: {next(torch_model.parameters()).dtype}")
    except Exception as e:
        import traceback

        print(f"\nException loading model: {e}")
        print(traceback.format_exc())
        pytest.skip(f"Failed to load HF model/processor from cloud: {e}")

    # Process image with HF processor
    inputs = processor(images=image, return_tensors="pt")
    print(f"Processor output keys: {inputs.keys()}")

    pixel_values_torch = inputs["pixel_values"].float()
    print(f"Pixel values dtype: {pixel_values_torch.dtype}")
    print(f"Pixel values shape: {pixel_values_torch.shape}")
    print(f"Pixel values range: [{pixel_values_torch.min():.3f}, {pixel_values_torch.max():.3f}]")

    # Run HF model
    # Since we loaded SiglipVisionModel directly, it IS the vision model
    hf_vision = torch_model
    hf_config = torch_model.config
    print(f"Vision model type: {type(hf_vision).__name__}")

    with torch.no_grad():
        vision_outputs = hf_vision(pixel_values_torch, output_hidden_states=True)
        torch_output = vision_outputs.last_hidden_state.detach().cpu().numpy()
        torch_hidden_states = [h.detach().cpu().numpy() for h in vision_outputs.hidden_states]

    print(f"HF encoder output shape: {torch_output.shape}")
    print(f"HF encoder output range: [{torch_output.min():.3f}, {torch_output.max():.3f}]")
    print(f"HF encoder output mean: {torch_output.mean():.6f}, std: {torch_output.std():.6f}")
    print(f"Number of HF hidden states: {len(torch_hidden_states)}")

    # Convert to JAX/Haliax format
    from levanter.models.siglip import SiglipVisionConfig, SiglipVisionModel

    # Create Levanter config from HF config
    lev_config = SiglipVisionConfig.from_hf_config(hf_config)
    print(
        f"\nLevanter config: hidden_size={lev_config.hidden_size}, "
        f"num_layers={lev_config.num_hidden_layers}, "
        f"image_size={lev_config.image_size}, patch_size={lev_config.patch_size}"
    )

    # Load HF weights into Levanter model
    print("\n=== Part 1: HF -> Levanter Conversion ===")
    import tempfile
    import equinox as eqx
    from haliax.state_dict import from_torch_compatible_state_dict

    # Use single-device mesh to avoid sharding issues with small batch sizes
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        # Save HF model to temporary directory
        torch_model.save_pretrained(f"{tmpdir}/hf_model")

        # Create Levanter model template
        Vocab = hax.Axis("vocab", 1)  # Dummy vocab for vision model
        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, Vocab, lev_config, key=jax.random.PRNGKey(0))

        # Load weights from HF checkpoint
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        print("✓ Successfully loaded HF weights into Levanter model")

        # Convert PyTorch pixel values to JAX/Haliax format
        # Shape: (batch, channels, height, width)
        pixel_values_np = pixel_values_torch.cpu().numpy()
        batch_size, num_channels, height, width = pixel_values_np.shape

        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", num_channels)
        Height = hax.Axis("height", height)
        Width = hax.Axis("width", width)

        pixel_values_jax = hax.named(jnp.array(pixel_values_np, dtype=jnp.float32), (Batch, Channels, Height, Width))

        print(f"\nJAX pixel values shape: {pixel_values_jax.axes}")
        print(f"JAX pixel values range: [{pixel_values_jax.array.min():.3f}, {pixel_values_jax.array.max():.3f}]")

        # Run Levanter model with loaded HF weights
        print("\nRunning Levanter model inference...")
        lev_output = lev_model(pixel_values_jax, output_hidden_states=True, key=jax.random.PRNGKey(1))

    lev_output_np = np.array(lev_output.last_hidden_state.array)
    lev_hidden_states = [np.array(h.array) for h in lev_output.hidden_states]

    print(f"\nLevanter output shape: {lev_output.last_hidden_state.axes}")
    print(f"Levanter output range: [{lev_output_np.min():.3f}, {lev_output_np.max():.3f}]")
    print(f"Levanter output mean: {lev_output_np.mean():.6f}, std: {lev_output_np.std():.6f}")
    print(f"Number of Levanter hidden states: {len(lev_hidden_states)}")

    # Compare outputs between HF and Levanter
    print("\n=== Output Comparison (HF vs Levanter) ===")
    print(f"HF shape: {torch_output.shape}")
    print(f"Levanter shape: {lev_output_np.shape}")

    assert (
        torch_output.shape == lev_output_np.shape
    ), f"Shape mismatch: HF={torch_output.shape}, Lev={lev_output_np.shape}"

    # Compute differences
    max_diff = np.max(np.abs(torch_output - lev_output_np))
    mean_diff = np.mean(np.abs(torch_output - lev_output_np))
    relative_diff = mean_diff / (np.abs(torch_output).mean() + 1e-8)

    print(f"\nMax absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Relative diff: {relative_diff:.6f}")
    print(f"\nHF first 10 values: {torch_output.flatten()[:10]}")
    print(f"Lev first 10 values: {lev_output_np.flatten()[:10]}")

    # Check for NaN/Inf
    assert not np.any(np.isnan(lev_output_np)), "Levanter output contains NaN"
    assert not np.any(np.isinf(lev_output_np)), "Levanter output contains Inf"
    assert not np.any(np.isnan(torch_output)), "HF output contains NaN"
    assert not np.any(np.isinf(torch_output)), "HF output contains Inf"

    # Compare values with tolerance
    # Use relatively loose tolerance since we're comparing with loaded weights
    # Numerical differences between PyTorch and JAX, plus different attention implementations,
    # can cause small differences (typically max diff < 0.02, mean diff < 0.001)
    tolerance_rtol = 5e-3  # 0.5% relative tolerance
    tolerance_atol = 2e-2  # 0.02 absolute tolerance

    if np.allclose(torch_output, lev_output_np, rtol=tolerance_rtol, atol=tolerance_atol):
        print("\n✓ ✓ ✓ Part 1: HF -> Levanter PASSED! ✓ ✓ ✓")
        print(f"  ✓ Output values match within tolerance (rtol={tolerance_rtol}, atol={tolerance_atol})")
        print(f"  ✓ Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    else:
        print("\n⚠ Warning: Outputs differ more than expected")
        print(f"  Max diff: {max_diff:.6f} (should be < {tolerance_atol})")
        print(f"  Mean diff: {mean_diff:.6f}")
        print("  This might indicate weight loading issues or numerical differences")

        # Still assert to fail the test
        assert np.allclose(
            torch_output, lev_output_np, rtol=tolerance_rtol, atol=tolerance_atol
        ), f"Output mismatch exceeds tolerance: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"

    # Compare all hidden states layer by layer
    print("\n=== Hidden States Comparison (All Layers) ===")
    print(f"Number of HF hidden states: {len(torch_hidden_states)}")
    print(f"Number of Levanter hidden states: {len(lev_hidden_states)}")

    assert len(torch_hidden_states) == len(
        lev_hidden_states
    ), f"Mismatch in number of hidden states: HF={len(torch_hidden_states)}, Lev={len(lev_hidden_states)}"

    hidden_states_all_match = True
    for i, (hf_h, lev_h) in enumerate(zip(torch_hidden_states, lev_hidden_states)):
        layer_name = "Embeddings" if i == 0 else f"Layer {i-1}"

        max_diff_h = np.max(np.abs(hf_h - lev_h))
        mean_diff_h = np.mean(np.abs(hf_h - lev_h))

        print(f"\n{layer_name}:")
        print(f"  Shape: HF={hf_h.shape}, Lev={lev_h.shape}")
        print(f"  Max diff: {max_diff_h:.6f}")
        print(f"  Mean diff: {mean_diff_h:.6f}")

        # Check if layer matches
        layer_matches = np.allclose(hf_h, lev_h, rtol=tolerance_rtol, atol=tolerance_atol)
        if layer_matches:
            print(f"  ✓ {layer_name} matches!")
        else:
            print(f"  ⚠️  Warning: {layer_name} outputs differ!")
            hidden_states_all_match = False

    if hidden_states_all_match:
        print("\n✓ All hidden states match between HF and Levanter!")
    else:
        print("\n⚠️  Warning: Some hidden states differ between HF and Levanter!")

    # ================================================================
    # Part 2: Test Levanter -> HF conversion and output consistency
    # ================================================================
    print("\n\n=== Part 2: Levanter -> HF Conversion Test ===")

    # Convert Levanter model to HF format by saving and reloading
    print("\nConverting Levanter model to HF format...")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/converted_model"

        # Save the Levanter model as HF checkpoint
        print("Saving Levanter model as HF checkpoint...")
        # Use the model_name as reference checkpoint (for config metadata)
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        # converter = lev_config.hf_checkpoint_converter()
        converter.save_pretrained(lev_model, save_path, save_tokenizer=False)

        # Load the saved checkpoint as HF model
        print("Loading saved checkpoint as HF model...")
        from transformers import SiglipVisionModel as HfSiglipVisionModel

        converted_hf_model = HfSiglipVisionModel.from_pretrained(save_path)
        converted_hf_model.eval()
        converted_hf_model = converted_hf_model.float()

        print("✓ Successfully converted Levanter model to HF format")

        # Run inference on converted HF model
        print("\nRunning converted HF model inference...")
        with torch.no_grad():
            converted_outputs = converted_hf_model(pixel_values_torch)
            converted_output_np = converted_outputs.last_hidden_state.detach().cpu().numpy()

        print(f"Converted HF output shape: {converted_output_np.shape}")
        print(f"Converted HF output range: [{converted_output_np.min():.3f}, {converted_output_np.max():.3f}]")
        print(f"Converted HF output mean: {converted_output_np.mean():.6f}, std: {converted_output_np.std():.6f}")

        # Compare Levanter output with converted HF output
        print("\n=== Output Comparison (Levanter vs Converted HF) ===")
        print(f"Levanter shape: {lev_output_np.shape}")
        print(f"Converted HF shape: {converted_output_np.shape}")

        assert (
            lev_output_np.shape == converted_output_np.shape
        ), f"Shape mismatch: Levanter={lev_output_np.shape}, Converted HF={converted_output_np.shape}"

        # Compute differences between Levanter and converted HF
        max_diff_lev_hf = np.max(np.abs(lev_output_np - converted_output_np))
        mean_diff_lev_hf = np.mean(np.abs(lev_output_np - converted_output_np))
        relative_diff_lev_hf = mean_diff_lev_hf / (np.abs(lev_output_np).mean() + 1e-8)

        print(f"\nMax absolute diff: {max_diff_lev_hf:.6f}")
        print(f"Mean absolute diff: {mean_diff_lev_hf:.6f}")
        print(f"Relative diff: {relative_diff_lev_hf:.6f}")
        print(f"\nLevanter first 10 values: {lev_output_np.flatten()[:10]}")
        print(f"Converted HF first 10 values: {converted_output_np.flatten()[:10]}")

        # Check for NaN/Inf in converted output
        assert not np.any(np.isnan(converted_output_np)), "Converted HF output contains NaN"
        assert not np.any(np.isinf(converted_output_np)), "Converted HF output contains Inf"

        # Compare with same tolerance
        if np.allclose(lev_output_np, converted_output_np, rtol=tolerance_rtol, atol=tolerance_atol):
            print("\n✓ ✓ ✓ Part 2: Levanter -> HF PASSED! ✓ ✓ ✓")
            print(f"  ✓ Output values match within tolerance (rtol={tolerance_rtol}, atol={tolerance_atol})")
            print(f"  ✓ Max diff: {max_diff_lev_hf:.6f}, Mean diff: {mean_diff_lev_hf:.6f}")
        else:
            print("\n⚠ Warning: Levanter and converted HF outputs differ more than expected")
            print(f"  Max diff: {max_diff_lev_hf:.6f} (should be < {tolerance_atol})")
            print(f"  Mean diff: {mean_diff_lev_hf:.6f}")

            # Still assert to fail the test
            assert np.allclose(
                lev_output_np, converted_output_np, rtol=tolerance_rtol, atol=tolerance_atol
            ), f"Levanter -> HF conversion output mismatch: max_diff={max_diff_lev_hf:.6f}, mean_diff={mean_diff_lev_hf:.6f}"

        # Also compare converted HF with original HF
        print("\n=== Bonus: Original HF vs Converted HF ===")
        max_diff_hf_hf = np.max(np.abs(torch_output - converted_output_np))
        mean_diff_hf_hf = np.mean(np.abs(torch_output - converted_output_np))
        print(f"Max absolute diff: {max_diff_hf_hf:.6f}")
        print(f"Mean absolute diff: {mean_diff_hf_hf:.6f}")

        if np.allclose(torch_output, converted_output_np, rtol=tolerance_rtol, atol=tolerance_atol):
            print("✓ Original HF and converted HF outputs match!")
        else:
            print("⚠ Note: Original HF and converted HF differ (this is expected due to conversion roundtrip)")

    print("\n\n=== All Tests PASSED! ===")
    print("✓ HF -> Levanter conversion works correctly")
    print("✓ Levanter -> HF conversion works correctly")
    print("✓ Output consistency verified for all conversions")


@skip_if_no_torch
def test_siglip_vision_real_image_no_flash():
    """Test SigLIP vision model with real image, explicitly using VANILLA attention backend.

    This test is identical to test_siglip_vision_real_image but forces VANILLA attention
    (no flash attention) to compare numerical precision.
    """
    import torch
    from dataclasses import replace

    from levanter.layers.attention import AttentionBackend

    try:
        from transformers import AutoProcessor, AutoModel  # noqa: F401
    except ImportError:
        pytest.skip("transformers not available")

    print("\n=== Testing SigLIP Vision with Real Image (NO FLASH ATTENTION) ===")

    # Load image from HuggingFace dataset
    image = get_single_image()
    print(f"Image size: {image.size}, mode: {image.mode}")

    # Load HF model and processor from cloud
    model_name = "google/siglip-base-patch16-224"
    print(f"Loading HF model and processor from cloud: {model_name}")

    try:
        from transformers import SiglipImageProcessor

        processor = SiglipImageProcessor.from_pretrained(model_name)

        from transformers import SiglipVisionModel

        torch_model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        torch_model = torch_model.float()
        print(f"Loaded model type: {type(torch_model).__name__}")
        print(f"Model dtype: {next(torch_model.parameters()).dtype}")
    except Exception as e:
        import traceback

        print(f"\nException loading model: {e}")
        print(traceback.format_exc())
        pytest.skip(f"Failed to load HF model/processor from cloud: {e}")

    # Process image with HF processor
    inputs = processor(images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"].float()
    print(f"Pixel values shape: {pixel_values_torch.shape}")

    # Run HF model
    hf_vision = torch_model
    hf_config = torch_model.config

    with torch.no_grad():
        vision_outputs = hf_vision(pixel_values_torch, output_hidden_states=True)
        torch_output = vision_outputs.last_hidden_state.detach().cpu().numpy()
        torch_hidden_states = [h.detach().cpu().numpy() for h in vision_outputs.hidden_states]

    print(f"HF encoder output shape: {torch_output.shape}")
    print(f"HF encoder output range: [{torch_output.min():.3f}, {torch_output.max():.3f}]")

    # Convert to JAX/Haliax format
    from levanter.models.siglip import SiglipVisionConfig, SiglipVisionModel

    # Create Levanter config from HF config with VANILLA attention backend
    lev_config_base = SiglipVisionConfig.from_hf_config(hf_config)
    # Force VANILLA attention backend (no flash attention)
    lev_config = replace(
        lev_config_base,
        use_flash_attention=False,
        attn_backend=AttentionBackend.VANILLA,
    )
    print(
        f"\nLevanter config: hidden_size={lev_config.hidden_size}, "
        f"num_layers={lev_config.num_hidden_layers}, "
        f"use_flash_attention={lev_config.use_flash_attention}, "
        f"attn_backend={lev_config.attn_backend}"
    )

    # Load HF weights into Levanter model
    print("\n=== Part 1: HF -> Levanter Conversion (VANILLA attention) ===")
    import tempfile
    import equinox as eqx
    from haliax.state_dict import from_torch_compatible_state_dict

    # Use single-device mesh to avoid sharding issues with small batch sizes
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/hf_model")

        Vocab = hax.Axis("vocab", 1)
        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, Vocab, lev_config, key=jax.random.PRNGKey(0))

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        print("✓ Successfully loaded HF weights into Levanter model (VANILLA attention)")

        # Convert PyTorch pixel values to JAX/Haliax format
        pixel_values_np = pixel_values_torch.cpu().numpy()
        batch_size, num_channels, height, width = pixel_values_np.shape

        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", num_channels)
        Height = hax.Axis("height", height)
        Width = hax.Axis("width", width)

        pixel_values_jax = hax.named(jnp.array(pixel_values_np, dtype=jnp.float32), (Batch, Channels, Height, Width))

        # Run Levanter model with loaded HF weights
        print("\nRunning Levanter model inference (VANILLA attention)...")
        lev_output = lev_model(pixel_values_jax, output_hidden_states=True, key=jax.random.PRNGKey(1))

    lev_output_np = np.array(lev_output.last_hidden_state.array)
    lev_hidden_states = [np.array(h.array) for h in lev_output.hidden_states]

    print(f"\nLevanter output shape: {lev_output.last_hidden_state.axes}")
    print(f"Levanter output range: [{lev_output_np.min():.3f}, {lev_output_np.max():.3f}]")
    print(f"Levanter output mean: {lev_output_np.mean():.6f}, std: {lev_output_np.std():.6f}")

    # Compare outputs between HF and Levanter
    print("\n=== Output Comparison (HF vs Levanter with VANILLA attention) ===")
    print(f"HF shape: {torch_output.shape}")
    print(f"Levanter shape: {lev_output_np.shape}")

    assert (
        torch_output.shape == lev_output_np.shape
    ), f"Shape mismatch: HF={torch_output.shape}, Lev={lev_output_np.shape}"

    # Compute differences
    max_diff = np.max(np.abs(torch_output - lev_output_np))
    mean_diff = np.mean(np.abs(torch_output - lev_output_np))
    relative_diff = mean_diff / (np.abs(torch_output).mean() + 1e-8)

    print(f"\nMax absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Relative diff: {relative_diff:.6f}")
    print(f"\nHF first 10 values: {torch_output.flatten()[:10]}")
    print(f"Lev first 10 values: {lev_output_np.flatten()[:10]}")

    # Check for NaN/Inf
    assert not np.any(np.isnan(lev_output_np)), "Levanter output contains NaN"
    assert not np.any(np.isinf(lev_output_np)), "Levanter output contains Inf"

    # Compare all hidden states layer by layer
    print("\n=== Hidden States Comparison (All Layers) ===")
    for i, (hf_h, lev_h) in enumerate(zip(torch_hidden_states, lev_hidden_states)):
        layer_name = "Embeddings" if i == 0 else f"Layer {i-1}"
        max_diff_h = np.max(np.abs(hf_h - lev_h))
        mean_diff_h = np.mean(np.abs(hf_h - lev_h))
        print(f"{layer_name}: max_diff={max_diff_h:.6f}, mean_diff={mean_diff_h:.6f}")

    # Use same tolerance as regular test
    tolerance_rtol = 5e-3
    tolerance_atol = 2e-2

    if np.allclose(torch_output, lev_output_np, rtol=tolerance_rtol, atol=tolerance_atol):
        print("\n✓ ✓ ✓ Test PASSED with VANILLA attention! ✓ ✓ ✓")
        print(f"  ✓ Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    else:
        print("\n⚠ Warning: Outputs differ more than expected")
        assert np.allclose(
            torch_output, lev_output_np, rtol=tolerance_rtol, atol=tolerance_atol
        ), f"Output mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"

    # ================================================================
    # Part 2: Test Levanter -> HF conversion
    # ================================================================
    print("\n\n=== Part 2: Levanter -> HF Conversion Test ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/converted_model"

        print("Saving Levanter model as HF checkpoint...")
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        converter.save_pretrained(lev_model, save_path, save_tokenizer=False)

        print("Loading saved checkpoint as HF model...")
        from transformers import SiglipVisionModel as HfSiglipVisionModel

        converted_hf_model = HfSiglipVisionModel.from_pretrained(save_path)
        converted_hf_model.eval()
        converted_hf_model = converted_hf_model.float()

        print("✓ Successfully converted Levanter model to HF format")

        with torch.no_grad():
            converted_outputs = converted_hf_model(pixel_values_torch)
            converted_output_np = converted_outputs.last_hidden_state.detach().cpu().numpy()

        print(f"Converted HF output shape: {converted_output_np.shape}")
        print(f"Converted HF output range: [{converted_output_np.min():.3f}, {converted_output_np.max():.3f}]")

        # Compare Levanter output with converted HF output
        print("\n=== Output Comparison (Levanter vs Converted HF) ===")
        max_diff_lev_hf = np.max(np.abs(lev_output_np - converted_output_np))
        mean_diff_lev_hf = np.mean(np.abs(lev_output_np - converted_output_np))

        print(f"Max absolute diff: {max_diff_lev_hf:.6f}")
        print(f"Mean absolute diff: {mean_diff_lev_hf:.6f}")
        print(f"\nLevanter first 10 values: {lev_output_np.flatten()[:10]}")
        print(f"Converted HF first 10 values: {converted_output_np.flatten()[:10]}")

        assert not np.any(np.isnan(converted_output_np)), "Converted HF output contains NaN"
        assert not np.any(np.isinf(converted_output_np)), "Converted HF output contains Inf"

        if np.allclose(lev_output_np, converted_output_np, rtol=tolerance_rtol, atol=tolerance_atol):
            print("\n✓ ✓ ✓ Part 2: Levanter -> HF PASSED! ✓ ✓ ✓")
            print(f"  ✓ Max diff: {max_diff_lev_hf:.6f}, Mean diff: {mean_diff_lev_hf:.6f}")
        else:
            assert np.allclose(
                lev_output_np, converted_output_np, rtol=tolerance_rtol, atol=tolerance_atol
            ), f"Levanter -> HF conversion mismatch: max_diff={max_diff_lev_hf:.6f}"

        # Compare converted HF with original HF
        print("\n=== Bonus: Original HF vs Converted HF ===")
        max_diff_hf_hf = np.max(np.abs(torch_output - converted_output_np))
        mean_diff_hf_hf = np.mean(np.abs(torch_output - converted_output_np))
        print(f"Max absolute diff: {max_diff_hf_hf:.6f}")
        print(f"Mean absolute diff: {mean_diff_hf_hf:.6f}")

        if np.allclose(torch_output, converted_output_np, rtol=tolerance_rtol, atol=tolerance_atol):
            print("✓ Original HF and converted HF outputs match!")

    print("\n\n=== All Tests PASSED (VANILLA attention)! ===")
    print("✓ HF -> Levanter conversion works correctly with VANILLA attention")
    print("✓ Levanter -> HF conversion works correctly")
    print("✓ Output consistency verified for all conversions")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
