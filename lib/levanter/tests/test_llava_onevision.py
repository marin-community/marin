# Test file for LLaVA OneVision model
# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib.util
import os
import sys
import tempfile
import time

# Force torch to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Force JAX to use TPU (with CPU fallback)
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
# Set PJRT device to TPU
if "PJRT_DEVICE" not in os.environ:
    os.environ["PJRT_DEVICE"] = "TPU"
# Set coordinator address for TPU initialization (if not already set)
if "COORDINATOR_ADDRESS" not in os.environ and "JAX_COORDINATOR_ADDRESS" not in os.environ:
    # Try to detect local IP for single-host TPU setup
    import socket

    try:
        # Get non-localhost IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        # Set coordinator address and port (JAX default uses 8471)
        os.environ["JAX_COORDINATOR_ADDRESS"] = f"{local_ip}:8471"
    except Exception:
        # If IP detection fails, use localhost
        os.environ["JAX_COORDINATOR_ADDRESS"] = "127.0.0.1:8471"
# Force JAX to use float32
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random

# Enable float32 mode in JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402

from levanter.models.llava_onevision import (  # noqa: E402
    LlavaOnevisionConfig,
    LlavaOnevisionMultimodalProjector,
    LlavaOnevisionModel,
    VLMRequest,
    LlavaInferenceEngine,
)
from levanter.models.qwen import QwenConfig  # noqa: E402
from levanter.models.siglip2 import Siglip2VisionConfig  # noqa: E402
from levanter.models.siglip import SiglipVisionConfig  # noqa: E402
from levanter.layers.attention import AttentionBackend  # noqa: E402
from levanter.utils.activation import ActivationFunctionEnum  # noqa: E402
from levanter.inference.engine import InferenceEngineConfig  # noqa: E402
from levanter.inference.jit_scheduler import SeqDecodingParams  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig, DEFAULT_DP_AXES  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from transformers.models.llava_onevision.modeling_llava_onevision import (  # noqa: E402
    image_size_to_num_patches as hf_image_size_to_num_patches,
)

# Import test utils for mesh context
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import use_test_mesh  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402

# Define skip_if_no_torch locally to avoid conftest dependencies
if importlib.util.find_spec("torch") is not None:
    skip_if_no_torch = pytest.mark.skipif(False, reason="torch is available")
else:
    skip_if_no_torch = pytest.mark.skip(reason="torch not available")

# Import shared helper functions from test_image_utils
from test_image_utils import (  # noqa: E402
    create_grid_mask,
    pad_pixel_values,
    prepare_test_data_single,
    DEFAULT_GRID_PINPOINTS,
    compare_logits_by_region,
    create_lev_jax_tensors,
)
from test_image_utils import get_single_image, get_multi_images  # noqa: E402
import jax.tree_util as jtu  # noqa: E402


def _to_float32(x):
    """Convert JAX arrays to float32 for numerical consistency in tests."""
    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.float32)
    return x


def _tiny_vision_config():
    """Return a tiny SiglipVisionConfig for testing."""
    return SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=128,
        patch_size=16,
    )


def _tiny_text_config():
    """Return a tiny QwenConfig for testing."""
    return QwenConfig(
        max_seq_len=256,
        hidden_dim=128,
        intermediate_dim=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
    )


def _tiny_llava_onevision_config():
    """Return a tiny LlavaOnevisionConfig for testing."""
    return LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
        image_token_index=151646,
        video_token_index=151647,
    )


@skip_if_no_torch
def _hf_llava_onevision_config():
    """Return a HuggingFace LlavaOnevisionConfig for testing."""
    from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig
    from transformers import Qwen2Config as HfQwen2Config

    vision_config = HfSiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        patch_size=16,
        image_size=128,
    )

    text_config = HfQwen2Config(
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        vocab_size=151936,
        no_bias=True,
    )

    return HfLlavaOnevisionConfig(
        vision_config=vision_config.to_dict(),
        text_config=text_config.to_dict(),
        image_token_index=151646,
        video_token_index=151647,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        vision_aspect_ratio="anyres_max_9",
        image_grid_pinpoints=[[128, 128]],
        multimodal_projector_bias=True,
    )


# =====================
# Config Creation Tests
# =====================


def test_llava_onevision_config_creation():
    """Test basic LlavaOnevisionConfig instantiation."""
    vision_config = Siglip2VisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=26,
        num_attention_heads=16,
        num_channels=3,
        num_patches=256,
        patch_size=14,
    )

    text_config = QwenConfig(
        max_seq_len=2048,
        hidden_dim=3584,
        intermediate_dim=18944,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
    )

    config = LlavaOnevisionConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=151646,
        video_token_index=151647,
    )

    # Verify basic attributes
    assert config.vision_config.hidden_size == 1152
    assert config.text_config.hidden_dim == 3584
    assert config.image_token_index == 151646
    assert config.video_token_index == 151647
    assert config.projector_hidden_act == ActivationFunctionEnum.gelu
    assert config.vision_feature_select_strategy == "full"
    assert config.vision_feature_layer == -1
    assert config.vision_aspect_ratio == "anyres_max_9"
    assert config.multimodal_projector_bias is True
    assert config.gradient_checkpointing is True


def test_llava_onevision_config_axes():
    """Test that axis properties are correctly defined."""
    config = _tiny_llava_onevision_config()

    # Test VisionEmbed axis
    assert config.VisionEmbed.name == "vision_embed"
    assert config.VisionEmbed.size == 64

    # Test TextEmbed axis
    assert config.TextEmbed.name == "embed"
    assert config.TextEmbed.size == 128

    # Test Embed axis (same as TextEmbed)
    assert config.Embed.name == "embed"
    assert config.Embed.size == 128

    # Test Pos axis
    assert config.Pos.name == "position"
    assert config.Pos.size == 256

    # Test max_Pos axis
    assert config.max_Pos.name == "position"
    assert config.max_Pos.size == 256

    # Test KeyPos axis
    assert config.KeyPos.name == "key_position"
    assert config.KeyPos.size == 256


def test_llava_onevision_config_default_image_grid_pinpoints():
    """Test that default image_grid_pinpoints is set correctly."""
    config = _tiny_llava_onevision_config()

    # Should have 36 pinpoints (6x6 grid)
    assert config.image_grid_pinpoints is not None
    assert len(config.image_grid_pinpoints) == 9

    # Check first and last pinpoints
    assert config.image_grid_pinpoints[0] == [384, 384]
    assert config.image_grid_pinpoints[-1] == [1152, 1152]

    # Check some intermediate pinpoints
    assert [768, 1152] in config.image_grid_pinpoints
    assert [768, 768] in config.image_grid_pinpoints


def test_llava_onevision_config_custom_image_grid_pinpoints():
    """Test that custom image_grid_pinpoints is preserved."""
    custom_pinpoints = [[224, 224], [448, 448], [672, 672]]

    config = LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
        image_grid_pinpoints=custom_pinpoints,
    )

    assert config.image_grid_pinpoints == custom_pinpoints


def test_llava_onevision_config_vision_feature_strategy_validation():
    """Test that invalid vision_feature_select_strategy raises an error."""
    with pytest.raises(ValueError, match="vision_feature_select_strategy must be"):
        LlavaOnevisionConfig(
            vision_config=_tiny_vision_config(),
            text_config=_tiny_text_config(),
            vision_feature_select_strategy="invalid_strategy",
        )


def test_llava_onevision_config_vision_feature_strategy_valid():
    """Test that valid vision_feature_select_strategy values work."""
    for strategy in ["default", "full"]:
        config = LlavaOnevisionConfig(
            vision_config=_tiny_vision_config(),
            text_config=_tiny_text_config(),
            vision_feature_select_strategy=strategy,
        )
        assert config.vision_feature_select_strategy == strategy


def test_llava_onevision_config_frozen_dataclass():
    """Test that the config is frozen and immutable."""
    config = _tiny_llava_onevision_config()

    # Attempt to modify should raise an error
    with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
        config.image_token_index = 99999


def test_llava_onevision_config_model_type():
    """Test that model_type property returns correct class."""
    config = _tiny_llava_onevision_config()
    assert config.model_type == LlavaOnevisionModel


# =====================
# HF Config Conversion Tests
# =====================


@skip_if_no_torch
def test_llava_onevision_from_hf_config():
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_llava_onevision_config()

    # Convert from HF config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Check all attributes match
    assert config.image_token_index == hf_config.image_token_index
    assert config.video_token_index == hf_config.video_token_index
    assert config.vision_feature_select_strategy == hf_config.vision_feature_select_strategy
    assert config.vision_feature_layer == hf_config.vision_feature_layer
    assert config.vision_aspect_ratio == hf_config.vision_aspect_ratio
    assert config.multimodal_projector_bias == hf_config.multimodal_projector_bias

    # Check vision config conversion
    assert config.vision_config.hidden_size == 64
    assert config.vision_config.intermediate_size == 256
    assert config.vision_config.num_hidden_layers == 2
    assert config.vision_config.num_attention_heads == 4

    # Check text config conversion
    assert config.text_config.hidden_dim == 128
    assert config.text_config.intermediate_dim == 512
    assert config.text_config.num_layers == 2
    assert config.text_config.num_heads == 4


@skip_if_no_torch
def test_llava_onevision_to_hf_config():
    """Test conversion from Levanter config to HuggingFace config."""
    config = _tiny_llava_onevision_config()

    # Convert to HF config
    hf_config = config.to_hf_config(vocab_size=151936)

    # Check all attributes match
    assert hf_config.image_token_index == config.image_token_index
    assert hf_config.video_token_index == config.video_token_index
    assert hf_config.vision_feature_select_strategy == config.vision_feature_select_strategy
    assert hf_config.vision_feature_layer == config.vision_feature_layer
    assert hf_config.vision_aspect_ratio == config.vision_aspect_ratio
    assert hf_config.multimodal_projector_bias == config.multimodal_projector_bias

    # Check projector activation function
    assert hf_config.projector_hidden_act == "gelu"


@skip_if_no_torch
def test_llava_onevision_config_roundtrip():
    """Test that converting HF -> Levanter -> HF preserves the config."""
    # Start with HF config
    hf_config_orig = _hf_llava_onevision_config()

    # Convert to Levanter
    levanter_config = LlavaOnevisionConfig.from_hf_config(hf_config_orig)

    # Convert back to HF
    hf_config_roundtrip = levanter_config.to_hf_config(vocab_size=151936)

    # Check key attributes match
    assert hf_config_roundtrip.image_token_index == hf_config_orig.image_token_index
    assert hf_config_roundtrip.video_token_index == hf_config_orig.video_token_index
    assert hf_config_roundtrip.projector_hidden_act == hf_config_orig.projector_hidden_act
    assert hf_config_roundtrip.vision_feature_select_strategy == hf_config_orig.vision_feature_select_strategy
    assert hf_config_roundtrip.vision_feature_layer == hf_config_orig.vision_feature_layer
    assert hf_config_roundtrip.vision_aspect_ratio == hf_config_orig.vision_aspect_ratio
    assert hf_config_roundtrip.multimodal_projector_bias == hf_config_orig.multimodal_projector_bias


@skip_if_no_torch
def test_llava_onevision_config_roundtrip_levanter_to_hf_to_levanter():
    """Test that converting Levanter -> HF -> Levanter preserves the config."""
    # Start with Levanter config
    levanter_config_orig = _tiny_llava_onevision_config()

    # Convert to HF
    hf_config = levanter_config_orig.to_hf_config(vocab_size=1000)

    # Convert back to Levanter
    levanter_config_roundtrip = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Check key attributes match
    assert levanter_config_roundtrip.image_token_index == levanter_config_orig.image_token_index
    assert levanter_config_roundtrip.video_token_index == levanter_config_orig.video_token_index
    assert levanter_config_roundtrip.projector_hidden_act == levanter_config_orig.projector_hidden_act
    assert (
        levanter_config_roundtrip.vision_feature_select_strategy == levanter_config_orig.vision_feature_select_strategy
    )
    assert levanter_config_roundtrip.vision_feature_layer == levanter_config_orig.vision_feature_layer
    assert levanter_config_roundtrip.vision_aspect_ratio == levanter_config_orig.vision_aspect_ratio
    assert levanter_config_roundtrip.multimodal_projector_bias == levanter_config_orig.multimodal_projector_bias

    # Check vision config
    assert levanter_config_roundtrip.vision_config.hidden_size == levanter_config_orig.vision_config.hidden_size
    assert (
        levanter_config_roundtrip.vision_config.num_hidden_layers
        == levanter_config_orig.vision_config.num_hidden_layers
    )
    assert (
        levanter_config_roundtrip.vision_config.num_attention_heads
        == levanter_config_orig.vision_config.num_attention_heads
    )

    # Check text config
    assert levanter_config_roundtrip.text_config.hidden_dim == levanter_config_orig.text_config.hidden_dim
    assert levanter_config_roundtrip.text_config.num_layers == levanter_config_orig.text_config.num_layers
    assert levanter_config_roundtrip.text_config.num_heads == levanter_config_orig.text_config.num_heads
    assert levanter_config_roundtrip.text_config.num_kv_heads == levanter_config_orig.text_config.num_kv_heads


@skip_if_no_torch
def test_llava_onevision_config_roundtrip_comprehensive():
    """Test comprehensive config roundtrip with various settings."""
    from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig
    from transformers import Siglip2VisionConfig as HfSiglip2VisionConfig
    from transformers import Qwen2Config as HfQwen2Config

    # Test with different configurations
    test_configs = [
        # Config 1: Default settings
        {
            "vision": {"hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4},
            "text": {
                "hidden_size": 128,
                "intermediate_size": 512,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            },
            "projector_hidden_act": "gelu",
            "vision_feature_select_strategy": "full",
            "vision_feature_layer": -1,
        },
        # Config 2: Alternative activation and strategy
        {
            "vision": {"hidden_size": 128, "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 8},
            "text": {
                "hidden_size": 256,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
            },
            "projector_hidden_act": "silu",
            "vision_feature_select_strategy": "default",
            "vision_feature_layer": -1,
        },
    ]

    for i, cfg in enumerate(test_configs):
        # Create HF config
        vision_config = HfSiglip2VisionConfig(**cfg["vision"])
        text_config = HfQwen2Config(**cfg["text"], vocab_size=1000)
        hf_config_orig = HfLlavaOnevisionConfig(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            projector_hidden_act=cfg["projector_hidden_act"],
            vision_feature_select_strategy=cfg["vision_feature_select_strategy"],
            vision_feature_layer=cfg["vision_feature_layer"],
        )

        # HF -> Levanter -> HF roundtrip
        levanter_config = LlavaOnevisionConfig.from_hf_config(hf_config_orig)
        hf_config_roundtrip = levanter_config.to_hf_config(vocab_size=1000)

        # Verify key fields
        assert (
            hf_config_roundtrip.projector_hidden_act == hf_config_orig.projector_hidden_act
        ), f"Config {i}: projector_hidden_act mismatch"
        assert (
            hf_config_roundtrip.vision_feature_select_strategy == hf_config_orig.vision_feature_select_strategy
        ), f"Config {i}: vision_feature_select_strategy mismatch"
        assert (
            hf_config_roundtrip.vision_feature_layer == hf_config_orig.vision_feature_layer
        ), f"Config {i}: vision_feature_layer mismatch"
        assert (
            hf_config_roundtrip.vision_config.hidden_size == hf_config_orig.vision_config.hidden_size
        ), f"Config {i}: vision hidden_size mismatch"
        assert (
            hf_config_roundtrip.text_config.hidden_size == hf_config_orig.text_config.hidden_size
        ), f"Config {i}: text hidden_size mismatch"


@skip_if_no_torch
def test_llava_onevision_activation_function_mapping():
    """Test that various activation functions are correctly mapped."""
    from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig
    from transformers import Siglip2VisionConfig as HfSiglip2VisionConfig
    from transformers import Qwen2Config as HfQwen2Config

    vision_config = HfSiglip2VisionConfig(hidden_size=64, num_attention_heads=4)
    text_config = HfQwen2Config(hidden_size=128, num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)

    activation_mappings = [
        ("gelu", ActivationFunctionEnum.gelu),
        ("gelu_new", ActivationFunctionEnum.gelu_new),
        ("relu", ActivationFunctionEnum.relu),
        ("silu", ActivationFunctionEnum.silu),
    ]

    for hf_act_name, expected_enum in activation_mappings:
        hf_config = HfLlavaOnevisionConfig(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            projector_hidden_act=hf_act_name,
        )

        levanter_config = LlavaOnevisionConfig.from_hf_config(hf_config)
        assert (
            levanter_config.projector_hidden_act == expected_enum
        ), f"Failed for {hf_act_name}: expected {expected_enum}, got {levanter_config.projector_hidden_act}"


@skip_if_no_torch
def test_llava_onevision_config_overrides():
    """Test that config overrides work correctly in to_hf_config."""
    config = _tiny_llava_onevision_config()

    # Convert to HF config with overrides
    overrides = {
        "architectures": ["LlavaOnevisionForConditionalGeneration"],
        "model_type": "llava_onevision",
    }
    hf_config = config.to_hf_config(vocab_size=151936, config_overrides=overrides)

    # Check that overrides were applied
    assert hf_config.architectures == ["LlavaOnevisionForConditionalGeneration"]
    assert hf_config.model_type == "llava_onevision"

    # Other values should remain the same
    assert hf_config.image_token_index == config.image_token_index
    assert hf_config.video_token_index == config.video_token_index


@skip_if_no_torch
def test_llava_onevision_from_hf_pretrained():
    """Test loading LLaVA OneVision config from HuggingFace pretrained."""
    from transformers import AutoConfig

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    print(f"Loading HF config from: {model_name}")

    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Convert to Levanter config
        config = LlavaOnevisionConfig.from_hf_config(hf_config)

        # Verify config
        assert config.image_token_index == hf_config.image_token_index
        assert config.video_token_index == hf_config.video_token_index
        assert config.vision_feature_select_strategy == hf_config.vision_feature_select_strategy

        print(f"✓ Loaded config from HF: {model_name}")
        print(f"  Vision hidden size: {config.vision_config.hidden_size}")
        print(f"  Text hidden dim: {config.text_config.hidden_dim}")
        print(f"  Image token index: {config.image_token_index}")
        print(f"  Video token index: {config.video_token_index}")

    except Exception as e:
        pytest.skip(f"Could not load from HF (requires internet): {e}")


# =====================
# Multimodal Projector Tests
# =====================


def test_llava_onevision_projector_initialization():
    """Test that LlavaOnevisionMultimodalProjector can be initialized correctly."""
    config = _tiny_llava_onevision_config()

    projector = LlavaOnevisionMultimodalProjector.init(
        config=config,
        key=random.PRNGKey(42),
    )

    # Check that layers are initialized
    assert projector.linear_1 is not None
    assert projector.linear_2 is not None
    assert projector.act is not None
    assert projector.config == config

    # Check layer dimensions
    # linear_1: vision_embed -> projector_hidden
    assert projector.linear_1.In == config.VisionEmbed
    assert projector.linear_1.Out.name == "projector_hidden"
    assert projector.linear_1.Out.size == config.text_config.hidden_dim

    # linear_2: projector_hidden -> text_embed
    assert projector.linear_2.In.name == "projector_hidden"
    assert projector.linear_2.In.size == config.text_config.hidden_dim
    assert projector.linear_2.Out == config.TextEmbed


def test_llava_onevision_projector_forward():
    """Test LlavaOnevisionMultimodalProjector forward pass."""
    config = _tiny_llava_onevision_config()

    projector = LlavaOnevisionMultimodalProjector.init(
        config=config,
        key=random.PRNGKey(42),
    )

    # Create input: (batch, num_patches, vision_embed)
    Batch = Axis("batch", 2)
    NumPatches = Axis("num_patches", 64)

    x = hax.random.normal(random.PRNGKey(0), (Batch, NumPatches, config.VisionEmbed))

    # Forward pass
    output = projector(x, key=random.PRNGKey(1))

    # Check output shape: should project from VisionEmbed to TextEmbed
    assert output.axes == (Batch, NumPatches, config.TextEmbed)
    assert not jnp.any(jnp.isnan(output.array))


def test_llava_onevision_projector_different_activations():
    """Test LlavaOnevisionMultimodalProjector with different activation functions."""
    activations = [
        ActivationFunctionEnum.gelu,
        ActivationFunctionEnum.gelu_new,
        ActivationFunctionEnum.relu,
        ActivationFunctionEnum.silu,
    ]

    for activation in activations:
        vision_config = _tiny_vision_config()
        text_config = _tiny_text_config()

        config = LlavaOnevisionConfig(
            vision_config=vision_config,
            text_config=text_config,
            projector_hidden_act=activation,
        )

        projector = LlavaOnevisionMultimodalProjector.init(
            config=config,
            key=random.PRNGKey(42),
        )

        Batch = Axis("batch", 2)
        NumPatches = Axis("num_patches", 16)

        x = hax.random.normal(random.PRNGKey(0), (Batch, NumPatches, config.VisionEmbed))
        output = projector(x, key=random.PRNGKey(1))

        assert output.axes == (Batch, NumPatches, config.TextEmbed)
        assert not jnp.any(jnp.isnan(output.array))


def test_llava_onevision_projector_no_bias():
    """Test LlavaOnevisionMultimodalProjector without bias."""
    config = LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
        multimodal_projector_bias=False,
    )

    projector = LlavaOnevisionMultimodalProjector.init(
        config=config,
        key=random.PRNGKey(42),
    )

    # Check that bias is None
    assert projector.linear_1.bias is None
    assert projector.linear_2.bias is None

    # Forward pass should still work
    Batch = Axis("batch", 2)
    NumPatches = Axis("num_patches", 16)

    x = hax.random.normal(random.PRNGKey(0), (Batch, NumPatches, config.VisionEmbed))
    output = projector(x, key=random.PRNGKey(1))

    assert output.axes == (Batch, NumPatches, config.TextEmbed)
    assert not jnp.any(jnp.isnan(output.array))


# =====================
# Full Model Tests
# =====================


def test_llava_onevision_model_initialization():
    """Test that LlavaOnevisionModel can be initialized correctly."""
    config = _tiny_llava_onevision_config()
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    # Check that components are initialized
    assert model.vision_tower is not None
    assert model.multi_modal_projector is not None
    assert model.language_model is not None
    assert model.config == config


def test_llava_onevision_model_text_only_forward():
    """Test LlavaOnevisionModel forward pass with text only (no images)."""
    config = _tiny_llava_onevision_config()
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    # Create text-only input
    Batch = Axis("batch", 2)
    SeqLen = Axis("position", 32)

    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, SeqLen), 0, 1000)

    # Forward pass without images
    output = model(input_ids, pixel_values=None, key=random.PRNGKey(1))

    # Check output shape
    assert Batch in output.axes
    assert SeqLen in output.axes
    assert Vocab in output.axes
    assert not jnp.any(jnp.isnan(output.array))


def test_llava_onevision_model_different_configs():
    """Test LlavaOnevisionModel with different configurations."""
    configs = [
        {
            "vision_hidden": 64,
            "text_hidden": 128,
            "num_layers": 2,
        },
        {
            "vision_hidden": 128,
            "text_hidden": 256,
            "num_layers": 4,
        },
    ]

    for cfg_dict in configs:
        vision_config = SiglipVisionConfig(
            hidden_size=cfg_dict["vision_hidden"],
            intermediate_size=cfg_dict["vision_hidden"] * 4,
            num_hidden_layers=cfg_dict["num_layers"],
            num_attention_heads=4,
            image_size=128,
            patch_size=16,
        )

        text_config = QwenConfig(
            hidden_dim=cfg_dict["text_hidden"],
            intermediate_dim=cfg_dict["text_hidden"] * 4,
            num_layers=cfg_dict["num_layers"],
            num_heads=4,
            num_kv_heads=2,
        )

        config = LlavaOnevisionConfig(
            vision_config=vision_config,
            text_config=text_config,
        )

        Vocab = Axis("vocab", 1000)
        model = LlavaOnevisionModel.init(
            Vocab=Vocab,
            config=config,
            key=random.PRNGKey(42),
        )

        Batch = Axis("batch", 2)
        SeqLen = Axis("position", 16)
        input_ids = hax.random.randint(random.PRNGKey(0), (Batch, SeqLen), 0, 1000)

        output = model(input_ids, key=random.PRNGKey(1))

        assert Batch in output.axes
        assert SeqLen in output.axes
        assert Vocab in output.axes
        assert not jnp.any(jnp.isnan(output.array))


# =====================
# HF Checkpoint Converter Tests
# =====================


@skip_if_no_torch
def test_llava_onevision_hf_checkpoint_converter():
    """Test that hf_checkpoint_converter returns a valid converter."""
    # Test with reference checkpoint
    config_with_ref = LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
        reference_checkpoint="llava-hf/llava-onevision-qwen2-0.5b-si-hf",
    )

    converter = config_with_ref.hf_checkpoint_converter()
    assert converter is not None


# =====================
# Axis Compatibility Tests
# =====================


def test_llava_onevision_axis_compatibility():
    """Test that vision and text axes are compatible for projection."""
    config = _tiny_llava_onevision_config()

    # VisionEmbed and TextEmbed should have different sizes for this test
    assert config.VisionEmbed.size != config.TextEmbed.size

    # Projector should be able to map between them
    projector = LlavaOnevisionMultimodalProjector.init(
        config=config,
        key=random.PRNGKey(42),
    )

    # linear_1 maps VisionEmbed -> TextEmbed
    assert projector.linear_1.In.size == config.VisionEmbed.size
    assert projector.linear_1.Out.size == config.TextEmbed.size


def test_llava_onevision_embed_axis_is_text_embed():
    """Test that Embed axis equals TextEmbed axis."""
    config = _tiny_llava_onevision_config()

    # Embed should be the same as TextEmbed
    assert config.Embed == config.TextEmbed
    assert config.Embed.name == config.TextEmbed.name
    assert config.Embed.size == config.TextEmbed.size


# =====================
# Default Values Tests
# =====================


def test_llava_onevision_default_values():
    """Test that default values match expected LLaVA OneVision defaults."""
    config = LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
    )

    # Check default values
    assert config.image_token_index == 151646
    assert config.video_token_index == 151647
    assert config.projector_hidden_act == ActivationFunctionEnum.gelu
    assert config.vision_feature_select_strategy == "full"
    assert config.vision_feature_layer == -1
    assert config.vision_aspect_ratio == "anyres_max_9"
    assert config.multimodal_projector_bias is True
    assert config.gradient_checkpointing is True
    assert config.reference_checkpoint is None
    assert config.tokenizer is None


# =====================
# Vision Feature Layer Tests
# =====================


def test_llava_onevision_vision_feature_layer_single():
    """Test config with single vision feature layer."""
    config = LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
        vision_feature_layer=-1,
    )

    assert config.vision_feature_layer == -1


def test_llava_onevision_vision_feature_layer_list():
    """Test config with multiple vision feature layers."""
    config = LlavaOnevisionConfig(
        vision_config=_tiny_vision_config(),
        text_config=_tiny_text_config(),
        vision_feature_layer=[-2, -1],
    )

    assert config.vision_feature_layer == [-2, -1]


# =====================
# Multimodal Functionality Tests
# =====================


def test_llava_onevision_get_input_embeddings():
    """Test that get_input_embeddings returns the correct embedding layer."""
    config = _tiny_llava_onevision_config()
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    # Get input embeddings
    embeddings = model.get_input_embeddings()

    # Should return the language model's token embeddings
    assert embeddings is not None
    assert embeddings is model.language_model.embeddings.token_embeddings


def test_llava_onevision_get_placeholder_mask():
    """Test placeholder mask creation for image tokens."""
    config = _tiny_llava_onevision_config()
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    # Create input with image tokens
    Batch = Axis("batch", 2)
    SeqLen = Axis("position", 16)

    # Create input_ids with some image tokens at specific positions
    input_ids_array = jnp.full((Batch.size, SeqLen.size), 100, dtype=jnp.int32)
    # Place image tokens at positions 3, 4, 5 in first batch
    input_ids_array = input_ids_array.at[0, 3:6].set(config.image_token_index)
    # Place image tokens at positions 7, 8 in second batch
    input_ids_array = input_ids_array.at[1, 7:9].set(config.image_token_index)

    input_ids = hax.named(input_ids_array, (Batch, SeqLen))

    # Create dummy image features (5 total image tokens)
    TotalPatches = Axis("total_patches", 5)
    image_features = hax.random.normal(random.PRNGKey(0), (TotalPatches, config.TextEmbed))

    # Get placeholder mask (function only takes input_ids and image_features)
    mask = model.get_placeholder_mask(input_ids, image_features)

    # Check mask shape - should be (batch, position) boolean mask
    assert Batch in mask.axes
    assert SeqLen in mask.axes
    assert len(mask.axes) == 2  # No embed dimension

    # Check that mask is True at image token positions
    mask_array = mask.array  # (batch, position)

    # First batch should have True at positions 3, 4, 5
    assert mask_array[0, 3]
    assert mask_array[0, 4]
    assert mask_array[0, 5]
    assert not mask_array[0, 0]

    # Second batch should have True at positions 7, 8
    assert mask_array[1, 7]
    assert mask_array[1, 8]
    assert not mask_array[1, 0]


def test_llava_onevision_get_placeholder_mask_count_mismatch():
    """Test that placeholder mask raises error when token count doesn't match feature count."""
    config = _tiny_llava_onevision_config()
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    Batch = Axis("batch", 1)
    SeqLen = Axis("position", 16)

    # Create input with 3 image tokens
    input_ids_array = jnp.full((Batch.size, SeqLen.size), 100, dtype=jnp.int32)
    input_ids_array = input_ids_array.at[0, 3:6].set(config.image_token_index)
    input_ids = hax.named(input_ids_array, (Batch, SeqLen))

    # Create image features with wrong count (5 instead of 3)
    TotalPatches = Axis("total_patches", 5)
    image_features = hax.random.normal(random.PRNGKey(0), (TotalPatches, config.TextEmbed))

    # Should raise ValueError for count mismatch (use validate_placeholder_mask for non-JIT validation)
    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        model.validate_placeholder_mask(input_ids, image_features)


def test_llava_onevision_multimodal_forward():
    """Test full forward pass with both text and images using fixed-shape processing."""
    # Create config with custom image_grid_pinpoints matching our test image size
    vision_config = _tiny_vision_config()
    text_config = _tiny_text_config()
    image_size = vision_config.image_size  # 128
    patch_size = vision_config.patch_size  # 16

    # Use a single grid pinpoint matching our image size to avoid anyres complexity
    config = LlavaOnevisionConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_grid_pinpoints=[[image_size, image_size]],
    )
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    # Create input with image tokens
    Batch = Axis("batch", 1)
    # SiglipVisionConfig with image_size=128 and patch_size=16 produces (128/16)^2 = 64 patches per tile
    grid_h = grid_w = image_size // patch_size  # 8
    num_patches_per_tile = grid_h * grid_w  # 64

    # Fixed-shape processing: pad to TOTAL_PATCHES
    # Use 2 actual patches (tiles), pad to 10 (max_patches=9 + 1)
    actual_patches = 2  # 1 tile (base) + 1 high-res tile
    total_patches = 10  # Fixed size: max_patches + 1

    # Calculate total image tokens:
    # Each patch produces num_patches_per_tile features
    # With fixed-shape processing, input must have tokens for ALL patches (including padding)
    # The model will mask out padding patches during processing
    num_image_tokens = total_patches * num_patches_per_tile  # 10 * 64 = 640
    SeqLen = Axis("position", 10 + num_image_tokens)  # 10 text tokens + image tokens

    # Create input_ids: regular tokens + image tokens
    input_ids_array = jnp.full((Batch.size, SeqLen.size), 100, dtype=jnp.int32)
    input_ids_array = input_ids_array.at[0, 5 : 5 + num_image_tokens].set(config.image_token_index)
    input_ids = hax.named(input_ids_array, (Batch, SeqLen))

    # Create pixel values in 5D format: (batch, TOTAL_PATCHES, channels, height, width)
    # Pad to fixed size
    NumPatches = Axis("num_patches", total_patches)
    Channels = Axis("channels", 3)
    Height = Axis("height", image_size)
    Width = Axis("width", image_size)

    # Create actual patches
    pv_array = np.random.randn(Batch.size, actual_patches, 3, image_size, image_size).astype(np.float32)
    # Pad to total_patches
    pv_padded = pad_pixel_values(pv_array[0], total_patches)  # (total_patches, C, H, W)
    pv_array_padded = np.expand_dims(pv_padded, 0)  # (batch, total_patches, C, H, W)
    pixel_values = hax.named(jnp.array(pv_array_padded), (Batch, NumPatches, Channels, Height, Width))

    # Create grid_mask: True for actual patches, False for padding
    grid_mask_array = create_grid_mask(actual_patches, total_patches)
    GridMaskAxis = Axis("num_patches", total_patches)
    grid_mask = hax.named(jnp.array(np.expand_dims(grid_mask_array, 0)), (Batch, GridMaskAxis))

    # Forward pass with images (new API)
    output = model(
        input_ids,
        pixel_values=pixel_values,
        grid_mask=grid_mask,
        key=random.PRNGKey(2),
    )

    # Check output shape
    assert Batch in output.axes
    assert SeqLen in output.axes
    assert Vocab in output.axes
    assert not jnp.any(jnp.isnan(output.array))

    # Output should be different from text-only forward pass
    output_text_only = model(input_ids, pixel_values=None, key=random.PRNGKey(2))
    assert not jnp.allclose(output.array, output_text_only.array)


def test_llava_onevision_inputs_embeds_parameter():
    """Test that inputs_embeds parameter works correctly as alternative to input_ids."""
    config = _tiny_llava_onevision_config()
    Vocab = Axis("vocab", 1000)

    model = LlavaOnevisionModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(42),
    )

    Batch = Axis("batch", 2)
    SeqLen = Axis("position", 16)

    # Create input_ids
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, SeqLen), 0, 1000)

    # Get embeddings manually
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # Forward pass using input_ids
    output1 = model(input_ids, pixel_values=None, key=random.PRNGKey(1))

    # Forward pass using inputs_embeds
    output2 = model(input_ids, pixel_values=None, inputs_embeds=inputs_embeds, key=random.PRNGKey(1))

    # Outputs should be identical when using same embeddings
    assert jnp.allclose(output1.array, output2.array, rtol=1e-5)


# =====================
# Numerical Consistency Tests (vs HuggingFace)
# =====================


@skip_if_no_torch
def test_llava_onevision_multimodal_projector_vs_hf():
    """Compare multimodal projector output with HuggingFace."""
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict

    hf_config = _hf_llava_onevision_config()
    torch.random.manual_seed(0)
    torch_model = HfLlavaOnevision(hf_config)
    torch_model.eval()

    # Get HF multimodal projector
    hf_projector = torch_model.model.multi_modal_projector

    # Create test input (vision features)
    batch_size = 2
    num_patches = 16
    vision_hidden_size = hf_config.vision_config.hidden_size

    vision_features_torch = torch.randn(batch_size, num_patches, vision_hidden_size)

    # Run HF projector
    with torch.no_grad():
        hf_output = hf_projector(vision_features_torch)
        hf_output_np = hf_output.detach().cpu().numpy()

    # Load weights into Levanter projector
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Use single-device mesh to avoid sharding issues
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        # Save a tiny dummy tokenizer locally (avoids network dependency)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel(
                    {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3},
                    unk_token="<unk>",
                )
            ),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        tokenizer.save_pretrained(f"{tmpdir}/torch_model")

        import equinox as eqx
        from jax.random import PRNGKey

        # Use the correct vocab size from the HF config
        Vocab = Axis("vocab", hf_config.text_config.vocab_size)
        model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=PRNGKey(0))

        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/torch_model")
        model = from_torch_compatible_state_dict(model_template, state_dict)

        lev_projector = model.multi_modal_projector

    # Create Levanter input
    Batch = Axis("batch", batch_size)
    NumPatches = Axis("num_patches", num_patches)
    VisionEmbed = Axis("embed", vision_hidden_size)

    vision_features = hax.named(
        jnp.array(vision_features_torch.numpy().astype(np.float32), dtype=jnp.float32),
        (Batch, NumPatches, VisionEmbed),
    )

    # Run Levanter projector
    @hax.named_jit
    def compute_projector(projector, features):
        return projector(features, key=None)

    lev_output = compute_projector(lev_projector, vision_features).array

    print("\n=== Multimodal Projector ===")
    print(f"HF output shape: {hf_output_np.shape}, Levanter output shape: {lev_output.shape}")
    max_diff = np.max(np.abs(hf_output_np - np.array(lev_output)))
    mean_diff = np.mean(np.abs(hf_output_np - np.array(lev_output)))
    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")
    print(f"HF first 5: {hf_output_np.flatten()[:5]}")
    print(f"Lev first 5: {np.array(lev_output).flatten()[:5]}")

    # Assertions
    assert np.allclose(
        hf_output_np, np.array(lev_output), rtol=1e-2, atol=1e-2
    ), f"Multimodal Projector mismatch: max diff = {max_diff}"


@skip_if_no_torch
def test_llava_onevision_full_model_vs_hf():
    """Test LLaVA OneVision full model forward pass matches HuggingFace.

    This test validates multiple forward pass scenarios:
    1. Patch embeddings (vision tower input layer)
    2. Vision features (vision tower output)
    3. Projected vision features (after multimodal projector)
    4. Text-only forward pass
    5. Multimodal forward pass (text + images)
    """
    import torch

    # Import from transformers instead of local file
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict

    # Start profiling
    total_start_time = time.perf_counter()
    step_times = {}

    hf_config = _hf_llava_onevision_config()
    torch.random.manual_seed(0)
    torch_model = HfLlavaOnevision(hf_config)
    torch_model.eval()

    # Disable image_newline to match Levanter's behavior (Levanter doesn't add newline tokens)
    # HF adds newline tokens between image rows, which changes the feature count
    torch_model.model.image_newline = None

    # Create test inputs
    batch_size = 1
    seq_len = 25  # Must be >= 5 + num_image_tokens to fit all image tokens
    image_height = hf_config.vision_config.image_size
    image_width = hf_config.vision_config.image_size

    # Create pixel values as regular image data (4D)
    num_channels = hf_config.vision_config.num_channels
    pixel_values_4d = torch.randn(batch_size, num_channels, image_height, image_width)

    # Create anyres-style 5D inputs expected by HF (batch, num_patches, channels, height, width)
    num_patches_anyres = hf_image_size_to_num_patches(
        [image_height, image_width], hf_config.image_grid_pinpoints, hf_config.vision_config.image_size
    )
    pixel_values_5d = pixel_values_4d.unsqueeze(1).expand(-1, num_patches_anyres, -1, -1, -1).contiguous()

    with torch.no_grad():
        # Compute HF image features to determine placeholder token count
        hf_image_features_list = torch_model.model.get_image_features(
            pixel_values=pixel_values_5d, image_sizes=torch.tensor([[image_height, image_width]])
        )
        hf_image_features_concat = torch.cat(hf_image_features_list, dim=0)
        num_image_tokens_full = hf_image_features_concat.shape[0]

    # Create input_ids for multimodal test
    # Use token count that matches packed image features
    seq_len = 5 + num_image_tokens_full + 5
    input_ids_torch = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    input_ids_torch[0, 5 : 5 + num_image_tokens_full] = hf_config.image_token_index

    # Load Levanter model
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    import tempfile

    # Use single-device mesh to avoid sharding issues
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        # Save a tiny dummy tokenizer locally (avoids network dependency)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel(
                    {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3},
                    unk_token="<unk>",
                )
            ),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        tokenizer.save_pretrained(f"{tmpdir}/torch_model")

        import equinox as eqx
        from jax.random import PRNGKey

        Vocab = Axis("vocab", hf_config.text_config.vocab_size)
        model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=PRNGKey(0))

        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/torch_model")
        model = from_torch_compatible_state_dict(model_template, state_dict)

    # ==========================================
    # Test 1: Patch Embeddings
    # ==========================================
    print("\n=== Test 1: Patch Embeddings ===")
    step_start = time.perf_counter()
    with torch.no_grad():
        hf_patch_embed = torch_model.model.vision_tower.vision_model.embeddings(pixel_values_4d)
        hf_patch_embed_np = hf_patch_embed.detach().cpu().numpy()

    Batch = Axis("batch", batch_size)
    Channels = Axis("channels", num_channels)
    Height = Axis("height", image_height)
    Width = Axis("width", image_width)

    pixel_values_lev = hax.named(
        jnp.array(pixel_values_4d.numpy().astype(np.float32), dtype=jnp.float32), (Batch, Channels, Height, Width)
    )

    @hax.named_jit
    def compute_patch_embed(vision_tower, pixel_values):
        return vision_tower.vision_model.embeddings(pixel_values, key=None)

    lev_patch_embed = compute_patch_embed(model.vision_tower, pixel_values_lev).array

    step_end = time.perf_counter()
    step_times["Test 1: Patch Embeddings"] = step_end - step_start
    print(f"HF patch embed shape: {hf_patch_embed_np.shape}, Levanter: {lev_patch_embed.shape}")
    max_diff = np.max(np.abs(hf_patch_embed_np - np.array(lev_patch_embed)))
    mean_diff = np.mean(np.abs(hf_patch_embed_np - np.array(lev_patch_embed)))
    print(f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print(f"⏱️  Time: {step_times['Test 1: Patch Embeddings']:.4f}s")
    assert np.allclose(
        hf_patch_embed_np, np.array(lev_patch_embed), rtol=1e-2, atol=1e-2
    ), f"Patch embedding mismatch: max diff = {max_diff}"

    # ==========================================
    # Test 2: Vision Tower Output (Vision Features)
    # ==========================================
    print("\n=== Test 2: Vision Tower Output ===")
    step_start = time.perf_counter()

    with torch.no_grad():
        # HF vision tower forward (use vision_tower, not vision_model)
        hf_vision_output = torch_model.model.vision_tower(pixel_values=pixel_values_4d, output_hidden_states=True)
        hf_vision_features = hf_vision_output.hidden_states[-1].detach().cpu().numpy()

    # Infer number of patches from HF output
    NumPatches = Axis("num_patches", hf_vision_features.shape[1])

    def compute_vision_features(vision_tower, pixel_values):
        return vision_tower(pixel_values, output_hidden_states=True, key=None)

    lev_vision_features = compute_vision_features(model.vision_tower, pixel_values_lev).hidden_states[-1].array

    step_end = time.perf_counter()
    step_times["Test 2: Vision Tower Output"] = step_end - step_start
    print(f"HF vision features shape: {hf_vision_features.shape}, Levanter: {lev_vision_features.shape}")
    max_diff = np.max(np.abs(hf_vision_features - np.array(lev_vision_features)))
    mean_diff = np.mean(np.abs(hf_vision_features - np.array(lev_vision_features)))
    print(f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print(f"⏱️  Time: {step_times['Test 2: Vision Tower Output']:.4f}s")
    assert np.allclose(
        hf_vision_features, np.array(lev_vision_features), rtol=1e-2, atol=1e-2
    ), f"Vision features mismatch: max diff = {max_diff}"

    # ==========================================
    # Test 3: Multimodal Projector Output
    # ==========================================
    print("\n=== Test 3: Multimodal Projector ===")
    step_start = time.perf_counter()
    with torch.no_grad():
        # Use vision features from Test 2
        hf_projected = (
            torch_model.model.multi_modal_projector(torch.from_numpy(hf_vision_features)).detach().cpu().numpy()
        )

    # Create named array from vision features
    VisionEmbed = Axis("embed", hf_config.vision_config.hidden_size)
    vision_features_named = hax.named(
        jnp.array(hf_vision_features, dtype=jnp.float32), (Batch, NumPatches, VisionEmbed)
    )

    @hax.named_jit
    def compute_projected(projector, features):
        return projector(features, key=None)

    lev_projected = compute_projected(model.multi_modal_projector, vision_features_named).array

    step_end = time.perf_counter()
    step_times["Test 3: Multimodal Projector"] = step_end - step_start
    print(f"HF projected shape: {hf_projected.shape}, Levanter: {lev_projected.shape}")
    max_diff = np.max(np.abs(hf_projected - np.array(lev_projected)))
    mean_diff = np.mean(np.abs(hf_projected - np.array(lev_projected)))
    print(f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print(f"⏱️  Time: {step_times['Test 3: Multimodal Projector']:.4f}s")
    assert np.allclose(
        hf_projected, np.array(lev_projected), rtol=1e-2, atol=1e-2
    ), f"Projected features mismatch: max diff = {max_diff}"

    # ==========================================
    # Test 4: Text Embeddings (simpler test than full forward)
    # ==========================================
    print("\n=== Test 4: Text Embeddings ===")
    step_start = time.perf_counter()
    # Create text-only input (no image tokens)
    text_input = torch.randint(100, 200, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        # Get embeddings from language model
        hf_text_embed = torch_model.model.language_model.get_input_embeddings()(text_input)
        hf_text_embed_np = hf_text_embed.detach().cpu().numpy()

    # Levanter text embeddings
    SeqLen = Axis("position", seq_len)
    text_input_lev = hax.named(jnp.array(text_input.numpy(), dtype=jnp.int32), (Batch, SeqLen))

    @hax.named_jit
    def compute_text_embed(lm, input_ids):
        return lm.embeddings.token_embeddings.embed(input_ids)

    lev_text_embed = compute_text_embed(model.language_model, text_input_lev).array

    step_end = time.perf_counter()
    step_times["Test 4: Text Embeddings"] = step_end - step_start
    print(f"HF text embed shape: {hf_text_embed_np.shape}, Levanter: {lev_text_embed.shape}")
    max_diff = np.max(np.abs(hf_text_embed_np - np.array(lev_text_embed)))
    mean_diff = np.mean(np.abs(hf_text_embed_np - np.array(lev_text_embed)))
    print(f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print(f"HF first 5: {hf_text_embed_np.flatten()[:5]}")
    print(f"Lev first 5: {np.array(lev_text_embed).flatten()[:5]}")
    print(f"⏱️  Time: {step_times['Test 4: Text Embeddings']:.4f}s")
    # Use looser tolerance for embeddings
    assert np.allclose(
        hf_text_embed_np, np.array(lev_text_embed), rtol=1e-2, atol=1e-2
    ), f"Text embeddings mismatch: max diff = {max_diff}"

    # ==========================================
    # Test 5: Multimodal Forward Pass Validation (End-to-End with Patchified Images)
    # ==========================================
    print("\n=== Test 5: Multimodal Forward Pass ===")
    step_start = time.perf_counter()
    # This test compares full end-to-end forward pass using anyres 5D inputs
    # num_image_tokens_full is determined by HF pack_image_features (already computed above)
    seq_len_full = 5 + num_image_tokens_full + 5  # prefix + image tokens + suffix
    input_ids_multimodal_torch = torch.randint(0, 1000, (batch_size, seq_len_full), dtype=torch.long)
    input_ids_multimodal_torch[0, 5 : 5 + num_image_tokens_full] = hf_config.image_token_index

    # HuggingFace multimodal forward pass using anyres 5D images
    image_sizes_full = torch.tensor([[image_height, image_width]] * batch_size, dtype=torch.long)

    with torch.no_grad():
        hf_output = torch_model(
            input_ids=input_ids_multimodal_torch,
            pixel_values=pixel_values_5d,
            image_sizes=image_sizes_full,
            attention_mask=torch.ones_like(input_ids_multimodal_torch),
            return_dict=True,
        )
        hf_multimodal_logits = hf_output.logits.detach().cpu().numpy()

    # Levanter multimodal forward
    # Use the same 4D format as HF
    NumPatchesAnyres = Axis("num_patches_anyres", num_patches_anyres)
    _pixel_values_lev_full = hax.named(
        jnp.array(pixel_values_5d.numpy().astype(np.float32), dtype=jnp.float32),
        (Batch, NumPatchesAnyres, Channels, Height, Width),
    )

    # Create Levanter input_ids with updated seq_len
    # Use "position" axis name as expected by Qwen transformer
    PositionFull = Axis("position", seq_len_full)
    input_ids_multimodal_lev = hax.named(
        jnp.array(input_ids_multimodal_torch.numpy(), dtype=jnp.int32), (Batch, PositionFull)
    )

    # Create grid_mask for fixed-shape processing
    actual_patches = num_patches_anyres
    total_patches = 10  # max_patches + 1
    grid_mask_np = create_grid_mask(actual_patches, total_patches)

    # Pad pixel_values to fixed size
    pv_array = pixel_values_5d.numpy().astype(np.float32)
    pv_padded = pad_pixel_values(pv_array[0], total_patches)
    pv_padded = np.expand_dims(pv_padded, 0)

    NumPatchesPadded = Axis("num_patches", total_patches)
    pixel_values_lev_padded = hax.named(
        jnp.array(pv_padded, dtype=jnp.float32),
        (Batch, NumPatchesPadded, Channels, Height, Width),
    )

    # Create grid_mask NamedArray
    GridMaskAxis = Axis("num_patches", total_patches)
    grid_mask = hax.named(
        jnp.array(np.expand_dims(grid_mask_np, 0)),
        (Batch, GridMaskAxis),
    )

    # Create unpad_indices for HF-compatible feature ordering
    # For this synthetic test with square images (128x128) and grid_pinpoints=[[128,128]],
    # the unpad_indices is identity mapping since no spatial unpadding is needed
    NumImageTokens = Axis("num_image_tokens", num_image_tokens_full)
    unpad_indices_np = np.arange(num_image_tokens_full, dtype=np.int32)
    unpad_indices = hax.named(
        jnp.array(np.expand_dims(unpad_indices_np, 0)),
        (Batch, NumImageTokens),
    )

    def compute_multimodal(model, input_ids, pixel_values, grid_mask, unpad_indices):
        # Run without JIT for consistency
        return model(
            input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            key=None,
        )

    lev_multimodal_logits = compute_multimodal(
        model, input_ids_multimodal_lev, pixel_values_lev_padded, grid_mask, unpad_indices
    ).array

    step_end = time.perf_counter()
    step_times["Test 5: Multimodal Forward Pass"] = step_end - step_start
    print(f"HF multimodal logits shape: {hf_multimodal_logits.shape}")
    print(f"Levanter multimodal logits shape: {lev_multimodal_logits.shape}")

    # Compare HF and Levanter multimodal outputs
    max_diff = np.max(np.abs(hf_multimodal_logits - np.array(lev_multimodal_logits)))
    mean_diff = np.mean(np.abs(hf_multimodal_logits - np.array(lev_multimodal_logits)))
    print(f"HF vs Levanter - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print(f"HF first 5 logits: {hf_multimodal_logits.flatten()[:5]}")
    print(f"Lev first 5 logits: {np.array(lev_multimodal_logits).flatten()[:5]}")
    print(f"⏱️  Time: {step_times['Test 5: Multimodal Forward Pass']:.4f}s")

    # Assert that outputs match within tolerance
    assert np.allclose(
        hf_multimodal_logits, np.array(lev_multimodal_logits), rtol=5e-2, atol=5e-2
    ), f"Multimodal forward pass mismatch: max diff = {max_diff}"

    # Also verify that multimodal output is different from text-only (sanity check)
    @hax.named_jit
    def compute_text_only(model, input_ids):
        return model(input_ids, pixel_values=None, key=None)

    lev_text_only_logits = compute_text_only(model, input_ids_multimodal_lev).array
    text_vs_multimodal_diff = np.abs(lev_multimodal_logits - lev_text_only_logits)
    mean_text_diff = np.mean(text_vs_multimodal_diff)
    print(f"Levanter text-only vs multimodal - Mean diff: {mean_text_diff:.2e}")

    # The outputs should be significantly different when images are included
    assert not np.allclose(
        lev_multimodal_logits, lev_text_only_logits, rtol=1e-3, atol=1e-3
    ), "Multimodal output should differ from text-only output when images are provided"

    # Print profiling summary
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print("\n=== All Tests Passed ===")
    print("✓ Patch embeddings match")
    print("✓ Vision features match")
    print("✓ Projected features match")
    print("✓ Text-only forward pass matches")
    print("✓ Multimodal forward pass produces expected behavior")
    print("\n=== Profiling Summary ===")
    for step_name, step_time in step_times.items():
        percentage = (step_time / total_time) * 100
        print(f"{step_name}: {step_time:.4f}s ({percentage:.1f}%)")
    print(f"Total time: {total_time:.4f}s")


@skip_if_no_torch
def test_llava_onevision_visual_embeddings_match():
    """Compare HF vs Levanter merged embeddings (text + visual) before LM."""
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )
    import equinox as eqx
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict

    from levanter.data.image import create_custom_processor

    print("\n=== Test: Visual Embeddings Match (Pre-LM) ===")

    image = get_single_image()

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    print(f"Loading HuggingFace model and processor: {model_name}")
    torch_model = HfLlavaOnevision.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    torch_model.model.image_newline = None  # Disable image_newline for consistency
    torch_model.eval()
    # Update image_grid_pinpoints in config
    torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    # Create two processors: HF uses unpadded, Levanter uses padded
    processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
    processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)

    text = "Describe this image briefly."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)
    # HF inputs (unpadded)
    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    # Levanter inputs (padded)
    inputs_lev = processor_lev(images=image, text=prompt, return_tensors="pt")

    print(f"HF input_ids shape: {inputs_hf['input_ids'].shape}")
    print(f"HF pixel_values shape: {inputs_hf['pixel_values'].shape}")
    print(f"Levanter input_ids shape: {inputs_lev['input_ids'].shape}")
    print(f"Levanter pixel_values shape: {inputs_lev['pixel_values'].shape}")

    with torch.no_grad():
        hf_inputs_embeds = torch_model.model.get_input_embeddings()(inputs_hf["input_ids"])
        hf_image_features_list = torch_model.model.get_image_features(
            pixel_values=inputs_hf["pixel_values"],
            image_sizes=inputs_hf["image_sizes"],
            vision_feature_layer=torch_model.config.vision_feature_layer,
            vision_feature_select_strategy=torch_model.config.vision_feature_select_strategy,
        )
        hf_image_features = torch.cat(hf_image_features_list, dim=0).to(
            hf_inputs_embeds.device, hf_inputs_embeds.dtype
        )
        hf_special_image_mask, _ = torch_model.model.get_placeholder_mask(
            inputs_hf["input_ids"], inputs_embeds=hf_inputs_embeds, image_features=hf_image_features
        )
        hf_merged_embeds = hf_inputs_embeds.masked_scatter(hf_special_image_mask, hf_image_features)

    print(f"HF merged embeds shape: {hf_merged_embeds.shape}")
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Load directly from HuggingFace instead of saving to temp directory
    # This avoids tokenizer loading issues
    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

    # Use single-device mesh to avoid sharding issues
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with use_test_mesh(mesh=single_device_mesh):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        state_dict = converter.load_state_dict(model_name)
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    # Convert model weights to float32 for consistency
    lev_model = jtu.tree_map(_to_float32, lev_model)

    batch_size = inputs_lev["input_ids"].shape[0]
    Batch = Axis("batch", batch_size)

    pixel_values_torch = inputs_lev["pixel_values"]
    if pixel_values_torch.dim() != 5:
        raise ValueError(f"Expected 5D pixel_values, got {pixel_values_torch.shape}")
    num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    height = pixel_values_torch.shape[3]
    width = pixel_values_torch.shape[4]

    # Create grid_mask for fixed-shape processing
    actual_patches = num_patches
    # Compute max_patches from image_grid_pinpoints
    patch_size = config.vision_config.image_size
    max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
    max_patches_per_dim = max_resolution // patch_size
    total_patches = max_patches_per_dim * max_patches_per_dim + 1  # +1 for base image
    grid_mask_np = create_grid_mask(actual_patches, total_patches)

    # Pad pixel_values to fixed size
    pv_np = pixel_values_torch.numpy().astype(np.float32)
    pv_padded_np = pad_pixel_values(pv_np[0], total_patches)
    pv_padded_np = np.expand_dims(pv_padded_np, 0)

    # Create Levanter tensors with padded shapes
    NumPatchesPadded = Axis("num_patches", total_patches)
    Channels = Axis("channels", channels)
    Height = Axis("height", height)
    Width = Axis("width", width)
    GridMaskAxis = Axis("grid_mask", total_patches)
    pixel_values_lev = hax.named(
        jnp.array(pv_padded_np, dtype=jnp.float32), (Batch, NumPatchesPadded, Channels, Height, Width)
    )
    grid_mask = hax.named(jnp.array(np.expand_dims(grid_mask_np, 0)), (Batch, GridMaskAxis))

    @hax.named_jit
    def compute_image_features(model, pixel_values, grid_mask):
        return model.get_image_features(
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            key=None,
        )

    image_features_result = compute_image_features(lev_model, pixel_values_lev, grid_mask)
    # Unpack result - get_image_features now returns (features, grid_mask) tuple
    if isinstance(image_features_result, tuple):
        image_features_lev, returned_grid_mask = image_features_result
    else:
        image_features_lev = image_features_result

    # Get dimensions from image features
    batch_ax, num_patches_ax, features_per_patch_ax, embed_ax = image_features_lev.axes
    lev_features_per_patch = features_per_patch_ax.size
    lev_embed = embed_ax.size

    print(f"Image features shape: {image_features_lev.shape}")
    print(f"Features per patch: {lev_features_per_patch}, Embed dim: {lev_embed}")

    # Compute unpad_indices for HF-style feature ordering
    image_sizes = inputs_hf["image_sizes"].tolist()
    num_hf_image_tokens = hf_image_features.shape[0]  # Use HF's actual feature count
    unpad_indices_np = processor_lev.compute_unpad_indices(
        image_sizes=image_sizes,
        height=patch_size,
        width=patch_size,
        max_num_features=num_hf_image_tokens,
    )
    NumImageTokens = Axis("num_image_tokens", num_hf_image_tokens)
    unpad_indices = hax.named(jnp.array(unpad_indices_np, dtype=jnp.int32), (Batch, NumImageTokens))
    print(f"unpad_indices shape: {unpad_indices.array.shape}")
    print(f"unpad_indices first 10: {unpad_indices.array[0, :10]}")
    print(f"unpad_indices last 10: {unpad_indices.array[0, -10:]}")

    # Flatten image features: (batch, num_patches, features_per_patch, embed) -> (batch, total_features, embed)
    total_image_tokens = num_patches_ax.size * features_per_patch_ax.size
    ImageTokens = Axis("image_tokens", total_image_tokens)
    image_features_flat = hax.flatten_axes(image_features_lev, (num_patches_ax, features_per_patch_ax), ImageTokens)
    print(f"Flattened image features shape: {image_features_flat.shape}")

    # Gather features in HF's unpadded order using unpad_indices
    def gather_unpadded(features, indices):
        # indices[i] = Levanter index for HF position i
        # Output[i] = features[indices[i]]
        return features[indices]

    image_features_reordered = jax.vmap(gather_unpadded)(image_features_flat.array, unpad_indices.array)
    # Shape: (batch, num_hf_image_tokens, embed)
    print(f"Reordered image features shape: {image_features_reordered.shape}")

    # ===== Compare raw image features directly =====
    hf_raw_features = hf_image_features.cpu().numpy()  # (num_hf_features, embed_dim)
    lev_raw_features = np.array(image_features_reordered[0])  # (num_hf_features, embed_dim)

    print("\n=== Raw image features comparison ===")
    print(f"HF raw features shape: {hf_raw_features.shape}")
    print(f"Levanter raw features shape: {lev_raw_features.shape}")

    # Compare base features (first 729)
    base_count = 729
    hf_base = hf_raw_features[:base_count]
    lev_base = lev_raw_features[:base_count]
    base_diff = np.mean(np.abs(hf_base - lev_base))
    base_max_diff = np.max(np.abs(hf_base - lev_base))
    print(f"Base features (first {base_count}) mean diff: {base_diff:.6e}, max diff: {base_max_diff:.6e}")

    # Compare grid features
    hf_grid = hf_raw_features[base_count:]
    lev_grid = lev_raw_features[base_count:]
    grid_diff = np.mean(np.abs(hf_grid - lev_grid))
    grid_max_diff = np.max(np.abs(hf_grid - lev_grid))
    print(f"Grid features ({hf_grid.shape[0]} tokens) mean diff: {grid_diff:.6e}, max diff: {grid_max_diff:.6e}")

    # Check first few features of each
    print("\nFirst 5 features comparison:")
    print(f"HF base[0,:5]: {hf_base[0,:5]}")
    print(f"Lev base[0,:5]: {lev_base[0,:5]}")
    print(f"HF grid[0,:5]: {hf_grid[0,:5]}")
    print(f"Lev grid[0,:5]: {lev_grid[0,:5]}")

    # Overall comparison
    overall_diff = np.mean(np.abs(hf_raw_features - lev_raw_features))
    overall_max_diff = np.max(np.abs(hf_raw_features - lev_raw_features))

    print("\n=== Overall Comparison Summary ===")
    print(f"Base features:  mean={base_diff:.6e}, max={base_max_diff:.6e}")
    print(f"Grid features:  mean={grid_diff:.6e}, max={grid_max_diff:.6e}")
    print(f"Overall:        mean={overall_diff:.6e}, max={overall_max_diff:.6e}")

    print(f"\n{'✓ PASS' if overall_diff < 1e-3 else '✗ FAIL'}: Image features match within tolerance=1e-3")
    assert overall_diff < 1e-3, f"Image features mismatch: overall_diff={overall_diff:.6e}"


@skip_if_no_torch
def test_llava_onevision_real_image_text():
    """Test with real image and text using processor with feature alignment.

    This test uses the same feature alignment approach as test_llava_onevision_visual_embeddings_match
    to properly compare logits between HF (unpadded) and Levanter (padded) models.
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )

    print("\n=== Test: Real Image and Text Input (with Feature Alignment) ===")

    # Load real image
    print("\n--- [Timing] Loading Image ---")
    start_time = time.time()
    image = get_single_image()
    print(f"Loaded image: {image.size}")
    image_load_time = time.time() - start_time
    print(f"  Time: {image_load_time:.4f} seconds")

    # Use a small pretrained model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print("\n--- [Timing] Loading HuggingFace Model ---")
    start_time = time.time()
    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.model.image_newline = None  # Disable image_newline for consistency
        torch_model.eval()
        # Update image_grid_pinpoints in config to match DEFAULT_GRID_PINPOINTS
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return
    hf_load_time = time.time() - start_time
    print(f"  Time: {hf_load_time:.4f} seconds")

    # Prepare inputs with processor using test_image_utils
    print("\n--- [Timing] Preparing Inputs with Processor ---")
    start_time = time.time()
    text = "Describe this image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    # Use prepare_test_data_single for unified data preparation
    test_pair = prepare_test_data_single(
        messages=messages,
        images=[image],
        model_name=model_name,
        add_generation_prompt=True,
    )
    processor_time = time.time() - start_time
    print(f"  Time: {processor_time:.4f} seconds")

    # Extract HF data for HF forward pass
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)
    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes).unsqueeze(0)

    inputs_hf = {
        "input_ids": hf_input_ids,
        "pixel_values": hf_pixel_values,
        "attention_mask": hf_attention_mask,
        "image_sizes": hf_image_sizes,
    }

    print(f"HF input_ids shape: {hf_input_ids.shape}")
    print(f"HF pixel_values shape: {hf_pixel_values.shape}")
    print(f"Levanter input_ids shape: {test_pair.lev.input_ids.shape}")
    print(f"Levanter pixel_values shape: {test_pair.lev.pixel_values.shape}")
    print(f"HF image_sizes: {hf_image_sizes}")

    # HuggingFace forward pass
    print("\n--- [Timing] HuggingFace Forward Pass ---")
    start_time = time.time()
    with torch.no_grad():
        hf_output = torch_model(**inputs_hf)
        hf_logits = hf_output.logits.detach().cpu().numpy()
    hf_forward_time = time.time() - start_time
    print(f"  Time: {hf_forward_time:.4f} seconds")

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF logits stats: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")
    print(f"HF first 5 logits: {hf_logits.flatten()[:5]}")

    # Convert to Levanter
    print("\n--- [Timing] Converting to Levanter ---")
    start_time = time.time()
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    config_time = time.time() - start_time
    print(f"  Config conversion time: {config_time:.4f} seconds")

    # Load directly from HuggingFace instead of saving to temp directory
    # This avoids processor.save_pretrained() issues with audio_tokenizer

    # Use model parallelism to shard vocab dimension and avoid OOM:
    # - logits tensor is seq_len * vocab_size ≈ 7000 * 152000 = 4GB per sample in fp32
    # - With model=8, vocab is sharded across 8 devices, reducing to ~0.5GB per device
    # - Set data=1 so batch_size=1 works (no data parallel sharding requirement)
    # - Note: heads=14 cannot be evenly divided by model=8, so we map heads to data (which is 1, i.e., no sharding)
    # - Also, vision_batch is mapped to model, so we need to prevent heads from also mapping to model
    #   to avoid duplicate model axis mapping (vision_batch and heads both on model axis)
    mesh_config = MeshConfig(
        axes={"model": 8, "data": 1, "replica": 1},
        compute_mapping={
            "vision_batch": ("model",),  # Shard vision patches across model axis
            "vocab": "model",  # Shard vocab dimension to reduce logits memory
            "batch": ("replica_dcn", "replica"),  # Map batch without data to avoid conflict with mlp/heads on data
        },
        shared_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding and conflict with vision_batch
            "mlp": "data",  # Map mlp to data (size 1) to avoid conflict with vision_batch on model axis
        },
        param_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding since 14 is not divisible by 8
        },
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        model_convert_time = time.time() - start_time
        print(f"  Total conversion time: {model_convert_time:.4f} seconds")

        # Use Levanter data from test_pair (already has grid_mask, padded pixel_values, unpad_indices)
        print("\n--- [Timing] Preparing Levanter Inputs ---")
        start_time = time.time()

        # Create JAX tensors with batch_size=1 (data parallel axis is 1)
        jax_tensors = create_lev_jax_tensors(test_pair.lev, batch_size=1)
        input_ids_lev_tensor = jax_tensors.input_ids
        pixel_values_lev_tensor = jax_tensors.pixel_values
        grid_mask = jax_tensors.grid_mask
        unpad_indices = jax_tensors.unpad_indices

        print(f"Levanter input_ids shape: {input_ids_lev_tensor.array.shape}")
        print(f"Levanter pixel_values shape: {pixel_values_lev_tensor.array.shape}")
        print(f"grid_mask shape: {grid_mask.array.shape}, valid patches: {test_pair.lev.grid_mask.sum()}")
        print(f"unpad_indices shape: {unpad_indices.array.shape}")
        print(f"unpad_indices first 10: {unpad_indices.array[0, :10]}")
        print(f"unpad_indices last 10: {unpad_indices.array[0, -10:]}")

        input_prep_time = time.time() - start_time
        print(f"  Time: {input_prep_time:.4f} seconds")

        print("\n--- [Timing] Levanter Forward Pass ---")

        @hax.named_jit
        def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=None,
            )

        # First call includes JIT compilation
        print("  First forward pass (includes JIT compilation)...")
        start_time = time.time()
        lev_logits_first = compute_lev(
            lev_model, input_ids_lev_tensor, pixel_values_lev_tensor, grid_mask, unpad_indices
        )
        lev_logits_first.array.block_until_ready()
        first_forward_time = time.time() - start_time
        print(f"  First forward pass time: {first_forward_time:.4f} seconds")

        # Warmup runs
        print("  Running warmup passes...")
        for i in range(3):
            _ = compute_lev(lev_model, input_ids_lev_tensor, pixel_values_lev_tensor, grid_mask, unpad_indices)
            _.array.block_until_ready()

        # Measure execution time (excluding compilation)
        print("  Measuring forward pass time (averaging over 5 runs)...")
        times = []
        for i in range(5):
            start_time = time.time()
            _ = compute_lev(lev_model, input_ids_lev_tensor, pixel_values_lev_tensor, grid_mask, unpad_indices)
            _.array.block_until_ready()
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.4f} seconds")

        avg_forward_time = sum(times) / len(times)
        min_forward_time = min(times)
        max_forward_time = max(times)
        lev_logits = lev_logits_first.array
        print(f"  Average forward pass time: {avg_forward_time:.4f} seconds")
        print(f"  Min: {min_forward_time:.4f} seconds, Max: {max_forward_time:.4f} seconds")

        print(f"Lev logits shape: {lev_logits.shape}")
        print(
            f"Lev logits stats: min={lev_logits.min():.4f}, max={lev_logits.max():.4f}, mean={lev_logits.mean():.4f}"
        )
        print(f"Lev first 5 logits: {np.array(lev_logits).flatten()[:5]}")

        # ===== Compare logits by region using unified compare_logits_by_region =====
        # Note: HF logits may have different length than Levanter (HF is unpadded, Levanter is padded)
        # compare_logits_by_region handles this by taking min(hf_len, lev_len)
        print("\n--- [Timing] Comparison by Region ---")
        start_time = time.time()

        lev_logits_np = np.array(lev_logits)
        if lev_logits_np.ndim == 3:
            lev_logits_np = lev_logits_np[0]  # Remove batch dimension

        # HF logits
        hf_logits_flat = hf_logits[0]  # (seq_len, vocab_size)

        print(f"HF logits shape: {hf_logits_flat.shape}")
        print(f"Lev logits shape: {lev_logits_np.shape}")
        # Use compare_logits_by_region for unified comparison
        # detailed=False for faster comparison (only overall diff, no per-region breakdown)
        # Pass attention_mask to exclude padding from Levanter
        image_token_id = torch_model.config.image_token_index
        comparison_result = compare_logits_by_region(
            hf_logits=hf_logits_flat,
            lev_logits=lev_logits_np,
            input_ids=test_pair.hf.input_ids,
            image_token_id=image_token_id,
            tolerance=1e-2,
            verbose=True,
            detailed=False,
            attention_mask=test_pair.lev.attention_mask,
        )

        compare_time = time.time() - start_time
        print(f"\n  Comparison time: {compare_time:.4f} seconds")

        # Print timing summary
        print("\n=== Timing Summary ===")
        print(f"Image loading:           {image_load_time:.4f} seconds")
        print(f"HF model loading:        {hf_load_time:.4f} seconds")
        print(f"Processor (input prep):  {processor_time:.4f} seconds")
        print(f"HF forward pass:         {hf_forward_time:.4f} seconds")
        print(f"Config conversion:       {config_time:.4f} seconds")
        print(f"Model conversion:        {model_convert_time:.4f} seconds")
        print(f"Levanter input prep:     {input_prep_time:.4f} seconds")
        print(f"Levanter forward (first): {first_forward_time:.4f} seconds (includes JIT)")
        print(f"Levanter forward (avg):   {avg_forward_time:.4f} seconds")
        print(f"Comparison:              {compare_time:.4f} seconds")
        total_time = (
            image_load_time
            + hf_load_time
            + processor_time
            + hf_forward_time
            + model_convert_time
            + input_prep_time
            + first_forward_time
            + compare_time
        )
        print(f"Total time:              {total_time:.4f} seconds")

        assert (
            comparison_result.passed
        ), f"Real image/text test failed: pre={comparison_result.details['pre_matches']}, image={comparison_result.details['image_matches']}, post={comparison_result.details['post_matches']}"
        print("✓ Real image and text input produces matching results!")


@skip_if_no_torch
def test_llava_onevision_real_multi_image_text():
    """Test Levanter model with multiple images, comparing HF and Levanter outputs.

    This test validates multi-image behavior where:
    - Both HF and Levanter use base patch per image (no anyres sub-patches)
    - unpad_indices is None for multi-image case
    - grid_mask marks which patches are valid (num_images base patches)
    - HF processor generates correct image tokens with padding_mode=True
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )

    print("\n=== Test: Multi-Image Real Input (Levanter only) ===")

    # Load multiple images
    print("\n--- [Timing] Loading Images ---")
    start_time = time.time()
    images = get_multi_images()  # Returns list of 2 images
    num_images = len(images)
    print(f"Loaded {num_images} images: {[img.size for img in images]}")
    image_load_time = time.time() - start_time
    print(f"  Time: {image_load_time:.4f} seconds")

    # Use a small pretrained model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print("\n--- [Timing] Loading HuggingFace Model (for weight conversion) ---")
    start_time = time.time()
    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.model.image_newline = None  # Disable image_newline for consistency
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return
    hf_load_time = time.time() - start_time
    print(f"  Time: {hf_load_time:.4f} seconds")

    # Prepare inputs with processor using test_image_utils
    print("\n--- [Timing] Preparing Inputs with Processor ---")
    start_time = time.time()
    text = "Compare these two images and describe the differences."
    # Create messages with multiple image placeholders
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]

    # Use prepare_test_data_single for unified data preparation
    # Note: multi-image requires larger max_length because processor generates tokens
    # for all anyres patches, even though model only uses base patches
    test_pair = prepare_test_data_single(
        messages=messages,
        images=images,
        model_name=model_name,
        add_generation_prompt=True,
        max_length=16384,  # Larger max_length for multi-image to avoid truncation
    )
    processor_time = time.time() - start_time
    print(f"  Time: {processor_time:.4f} seconds")

    print(f"Levanter input_ids shape: {test_pair.lev.input_ids.shape}")
    print(f"Levanter pixel_values shape: {test_pair.lev.pixel_values.shape}")
    print(f"Levanter grid_mask: {test_pair.lev.grid_mask.sum()} valid patches")

    # Verify multi-image preprocessing is correct
    assert test_pair.lev.unpad_indices is None, "Multi-image should have None unpad_indices"
    assert (
        test_pair.lev.grid_mask.sum() == num_images
    ), f"Multi-image should have {num_images} valid patches (base only)"
    print(f"✓ Multi-image preprocessing verified: {num_images} base patches, no unpad_indices")

    # Prepare HF inputs for forward pass
    # For multi-image, we need to use batch_num_images parameter
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)

    # For multi-image: pixel_values is already (num_images, patches, C, H, W) - 5D
    # DON'T unsqueeze(0) - HF model expects 5D where dim 0 is num_images
    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values)
    if hf_pixel_values.dim() == 4:
        # Single image: (patches, C, H, W) -> add batch dim
        hf_pixel_values = hf_pixel_values.unsqueeze(0)
    # Multi-image: already 5D (num_images, patches, C, H, W) - keep as is

    # image_sizes: for multi-image, keep as (num_images, 2), don't add extra dim
    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes)
    if hf_image_sizes.dim() == 1:
        # Single image: (2,) -> (1, 2)
        hf_image_sizes = hf_image_sizes.unsqueeze(0)
    # Multi-image: already (num_images, 2) - keep as is

    print(f"HF input_ids shape: {hf_input_ids.shape}")
    print(f"HF pixel_values shape: {hf_pixel_values.shape}")
    print(f"HF image_sizes shape: {hf_image_sizes.shape}")

    # HuggingFace forward pass with batch_num_images for multi-image mode
    print("\n--- [Timing] HuggingFace Forward Pass ---")
    start_time = time.time()
    with torch.no_grad():
        hf_output = torch_model(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            attention_mask=hf_attention_mask,
            image_sizes=hf_image_sizes,
            batch_num_images=torch.tensor([num_images]),  # Multi-image mode
        )
        hf_logits = hf_output.logits.detach().cpu().numpy()
    hf_forward_time = time.time() - start_time
    print(f"  Time: {hf_forward_time:.4f} seconds")

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF logits stats: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")

    # Convert to Levanter
    print("\n--- [Timing] Converting to Levanter ---")
    start_time = time.time()
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Load directly from HuggingFace
    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig

    # Use proper multi-device mesh with vision_batch sharding
    # Use batch_size=1 to avoid OOM (logits tensor is ~4GB per sample with vocab=152k)
    mesh_config = MeshConfig(
        axes={"model": 8, "data": 1, "replica": 1},
        compute_mapping={
            "vision_batch": ("model",),  # Shard vision patches across model axis
            "vocab": "model",  # Shard vocab dimension to reduce logits memory
            "batch": ("replica_dcn", "replica"),  # Map batch without data to avoid conflict with mlp/heads on data
        },
        shared_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding and conflict with vision_batch
            "mlp": "data",  # Map mlp to data (size 1) to avoid conflict with vision_batch on model axis
        },
        param_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding since 14 is not divisible by 8
        },
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        model_convert_time = time.time() - start_time
        print(f"  Total conversion time: {model_convert_time:.4f} seconds")

        # Use Levanter data from test_pair
        print("\n--- [Timing] Preparing Levanter Inputs ---")
        start_time = time.time()

        # Create JAX tensors using helper function with batch_size=1 to avoid OOM
        # The logits tensor is very large: seq_len * vocab_size ≈ 7000 * 152000 = 4GB per sample in fp32
        jax_tensors = create_lev_jax_tensors(test_pair.lev, batch_size=1)
        input_ids_lev_tensor = jax_tensors.input_ids
        pixel_values_lev_tensor = jax_tensors.pixel_values
        grid_mask = jax_tensors.grid_mask
        unpad_indices = jax_tensors.unpad_indices

        print(f"Levanter input_ids shape: {input_ids_lev_tensor.array.shape}")
        print(f"Levanter pixel_values shape: {pixel_values_lev_tensor.array.shape}")
        print(f"grid_mask shape: {grid_mask.array.shape}, valid patches: {test_pair.lev.grid_mask.sum()}")
        assert unpad_indices is None, "Multi-image should have None unpad_indices in JAX tensors"
        print("unpad_indices: None (multi-image mode, no anyres)")

        input_prep_time = time.time() - start_time
        print(f"  Time: {input_prep_time:.4f} seconds")

        print("\n--- [Timing] Levanter Forward Pass ---")

        @hax.named_jit
        def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=None,
            )

        # First call includes JIT compilation
        print("  First forward pass (includes JIT compilation)...")
        start_time = time.time()
        lev_logits_first = compute_lev(
            lev_model, input_ids_lev_tensor, pixel_values_lev_tensor, grid_mask, unpad_indices
        )
        lev_logits_first.array.block_until_ready()
        first_forward_time = time.time() - start_time
        print(f"  First forward pass time: {first_forward_time:.4f} seconds")

        lev_logits = lev_logits_first.array

        print(f"Lev logits shape: {lev_logits.shape}")
        print(
            f"Lev logits stats: min={float(lev_logits.min()):.4f}, max={float(lev_logits.max()):.4f}, mean={float(lev_logits.mean()):.4f}"
        )

        # Verify logits are not NaN/Inf
        assert not jnp.isnan(lev_logits).any(), "Logits contain NaN"
        assert not jnp.isinf(lev_logits).any(), "Logits contain Inf"

        # ===== Compare logits by region using unified compare_logits_by_region =====
        print("\n--- [Timing] Comparison by Region ---")
        start_time = time.time()

        lev_logits_np = np.array(lev_logits)
        if lev_logits_np.ndim == 3:
            lev_logits_np = lev_logits_np[0]  # Remove batch dimension

        # HF logits
        hf_logits_flat = hf_logits[0]  # (seq_len, vocab_size)

        print(f"HF logits shape: {hf_logits_flat.shape}")
        print(f"Lev logits shape: {lev_logits_np.shape}")

        # Use compare_logits_by_region for unified comparison
        image_token_id = torch_model.config.image_token_index
        comparison_result = compare_logits_by_region(
            hf_logits=hf_logits_flat,
            lev_logits=lev_logits_np,
            input_ids=test_pair.hf.input_ids,
            image_token_id=image_token_id,
            tolerance=1e-2,
            verbose=True,
            detailed=False,
            attention_mask=test_pair.lev.attention_mask,
        )

        compare_time = time.time() - start_time
        print(f"\n  Comparison time: {compare_time:.4f} seconds")

        # Print timing summary
        print("\n=== Timing Summary ===")
        print(f"Image loading:           {image_load_time:.4f} seconds")
        print(f"HF model loading:        {hf_load_time:.4f} seconds")
        print(f"Processor (input prep):  {processor_time:.4f} seconds")
        print(f"HF forward pass:         {hf_forward_time:.4f} seconds")
        print(f"Model conversion:        {model_convert_time:.4f} seconds")
        print(f"Levanter input prep:     {input_prep_time:.4f} seconds")
        print(f"Levanter forward (first): {first_forward_time:.4f} seconds (includes JIT)")
        print(f"Comparison:              {compare_time:.4f} seconds")

        assert (
            comparison_result.passed
        ), f"Multi-image test failed: pre={comparison_result.details['pre_matches']}, image={comparison_result.details['image_matches']}, post={comparison_result.details['post_matches']}"
        print("✓ Multi-image forward pass produces matching results!")


@skip_if_no_torch
@pytest.mark.skip(reason="7B model requires more memory than available on current hardware (needs ~4.6G, has ~3.7G)")
def test_llava_onevision_real_image_text_7b():
    """Test with real image and text using processor.

    Uses prepare_test_data_single and create_lev_jax_tensors from test_image_utils.
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )

    print("\n=== Test: Real Image and Text Input (7B) ===")

    # Load real image
    print("\n--- [Timing] Loading Image ---")
    start_time = time.time()
    image = get_single_image()
    print(f"Loaded image: {image.size}")
    image_load_time = time.time() - start_time
    print(f"  Time: {image_load_time:.4f} seconds")

    # Use 7B model for testing
    model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    print("\n--- [Timing] Loading HuggingFace Model ---")
    start_time = time.time()
    try:
        # Use bfloat16 for 7B model to fit in memory
        # This halves memory usage (14GB instead of 28GB)
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        torch_model.model.image_newline = None  # Disable image_newline for consistency
        torch_model.eval()
        # Update image_grid_pinpoints in config to match DEFAULT_GRID_PINPOINTS
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return
    hf_load_time = time.time() - start_time
    print(f"  Time: {hf_load_time:.4f} seconds")

    # Prepare inputs using test_image_utils
    print("\n--- [Timing] Preparing Inputs with Processor ---")
    start_time = time.time()
    text = "Describe this image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    # Use prepare_test_data_single for unified data preparation
    test_pair = prepare_test_data_single(
        messages=messages,
        images=[image],
        model_name=model_name,
        add_generation_prompt=True,
    )
    processor_time = time.time() - start_time
    print(f"  Time: {processor_time:.4f} seconds")

    # Extract HF data for HF forward pass
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)
    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes).unsqueeze(0)

    inputs_hf = {
        "input_ids": hf_input_ids,
        "pixel_values": hf_pixel_values,
        "attention_mask": hf_attention_mask,
        "image_sizes": hf_image_sizes,
    }

    print(f"HF input_ids shape: {hf_input_ids.shape}")
    print(f"HF pixel_values shape: {hf_pixel_values.shape}")
    print(f"Levanter input_ids shape: {test_pair.lev.input_ids.shape}")
    print(f"Levanter pixel_values shape: {test_pair.lev.pixel_values.shape}")
    print(f"HF image_sizes: {hf_image_sizes}")

    # HuggingFace forward pass
    print("\n--- [Timing] HuggingFace Forward Pass ---")
    start_time = time.time()
    with torch.no_grad():
        hf_output = torch_model(**inputs_hf)
        hf_logits = hf_output.logits.detach().float().cpu().numpy()
    hf_forward_time = time.time() - start_time
    print(f"  Time: {hf_forward_time:.4f} seconds")

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF logits stats: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")
    print(f"HF first 5 logits: {hf_logits.flatten()[:5]}")

    # Convert to Levanter
    print("\n--- [Timing] Converting to Levanter ---")
    start_time = time.time()
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Enable flash attention for both vision and text models for better performance
    # Use JAX_FLASH backend which works with bfloat16 (SPLASH has compatibility issues)
    vision_config_updated = dataclasses.replace(
        config.vision_config,
        use_flash_attention=True,
        attn_backend=AttentionBackend.JAX_FLASH,  # Use JAX flash for bfloat16
        gradient_checkpointing=False,  # Disable for inference performance
    )
    # Text model: use JAX_FLASH backend for bfloat16 compatibility
    text_config_updated = dataclasses.replace(
        config.text_config,
        attn_backend=AttentionBackend.JAX_FLASH,  # Use JAX flash for bfloat16
        gradient_checkpointing=False,  # Disable for inference performance
    )
    config = dataclasses.replace(
        config,
        vision_config=vision_config_updated,
        text_config=text_config_updated,
        gradient_checkpointing=False,  # Disable for inference performance
    )

    config_time = time.time() - start_time
    print(f"  Config conversion time: {config_time:.4f} seconds")

    print("\n--- [Timing] Saving and Loading Model ---")
    start_time = time.time()

    mesh_config = MeshConfig(
        axes={"model": 8, "data": 1, "replica": 1},
        compute_mapping={
            "vision_batch": ("model",),  # Shard vision patches across model axis
            "vocab": "model",  # Shard vocab dimension to reduce logits memory
            "batch": ("replica_dcn", "replica"),  # Map batch without data to avoid conflict with mlp/heads on data
        },
        shared_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding and conflict with vision_batch
            "mlp": "data",  # Map mlp to data (size 1) to avoid conflict with vision_batch on model axis
        },
        param_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding since 14 is not divisible by 8
        },
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        # Use bfloat16 for inference to halve memory (14GB instead of 28GB)
        # This is acceptable for inference where numerical precision is less critical
        compute_dtype = jnp.bfloat16

        # Load model using converter.load_pretrained() - same pattern as Qwen3 loading
        # Use parameter_axis_mapping for FSDP sharding (not compute_axis_mapping which is unsharded)
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
            resize_vocab_to_match_tokenizer=False,  # LlavaOnevisionModel doesn't have resize_vocab
        )

        model_convert_time = time.time() - start_time
        print(f"  Total conversion time: {model_convert_time:.4f} seconds")

        # Use Levanter data from test_pair (already has grid_mask, padded pixel_values, unpad_indices)
        print("\n--- [Timing] Preparing Levanter Inputs ---")
        start_time = time.time()

        # Create JAX tensors using helper function
        jax_tensors = create_lev_jax_tensors(test_pair.lev)
        input_ids_lev_tensor = jax_tensors.input_ids
        # Convert pixel_values to bfloat16 to match model dtype
        pixel_values_lev = jax_tensors.pixel_values.astype(jnp.bfloat16)
        grid_mask = jax_tensors.grid_mask
        unpad_indices = jax_tensors.unpad_indices

        print(f"Levanter input_ids shape: {input_ids_lev_tensor.array.shape}")
        print(f"Levanter pixel_values shape: {pixel_values_lev.array.shape}, dtype: {pixel_values_lev.dtype}")
        print(f"grid_mask shape: {grid_mask.array.shape}, valid patches: {test_pair.lev.grid_mask.sum()}")
        print(f"unpad_indices shape: {unpad_indices.array.shape}")

        input_prep_time = time.time() - start_time
        print(f"  Time: {input_prep_time:.4f} seconds")

        print("\n--- [Timing] Levanter Forward Pass ---")

        # Profile individual components to find bottleneck
        print("\n  --- Profiling individual components ---")

        # Create custom inference axis mapping:
        # - Include FSDP axis (embed) → data for parameter sharding (keeps params distributed)
        # - Include TP axes (mlp, heads) → model axis for tensor parallelism
        # - Exclude batch axis (since batch=1 can't be divided)
        # Use parameter_axis_mapping which has FSDP (embed→data), then remove batch
        inference_axis_mapping = dict(trainer_config.parameter_axis_mapping)
        # Remove batch mapping since batch=1 can't be sharded
        if "batch" in inference_axis_mapping:
            del inference_axis_mapping["batch"]
        print(f"  Inference axis mapping: {inference_axis_mapping}")

        # 1. Profile vision encoder + projector only
        @hax.named_jit(axis_resources=inference_axis_mapping)
        def compute_vision_only(model, pixel_values, grid_mask):
            return model.get_image_features(
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                key=None,
            )

        # 2. Profile LM only (text-only forward pass)
        @hax.named_jit(axis_resources=inference_axis_mapping)
        def compute_lm_only(model, input_ids):
            return model.language_model(input_ids, key=None)

        # 3. Full forward pass with grid mask and unpad_indices
        @hax.named_jit(axis_resources=inference_axis_mapping)
        def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=None,
            )

        # Profile vision encoder
        print("  Profiling vision encoder + projector...")

        def wait_for_vision_result(result):
            """Wait for vision result to complete, handling tuple return."""
            # get_image_features now returns (features, grid_mask) tuple
            if isinstance(result, tuple):
                features, _ = result
            else:
                features = result
            if isinstance(features, list):
                features[0].array.block_until_ready()
            else:
                features.array.block_until_ready()

        _ = compute_vision_only(lev_model, pixel_values_lev, grid_mask)  # Warmup/compile
        wait_for_vision_result(_)
        vision_times = []
        for i in range(3):
            start_time = time.time()
            result = compute_vision_only(lev_model, pixel_values_lev, grid_mask)
            wait_for_vision_result(result)
            vision_times.append(time.time() - start_time)
        avg_vision_time = sum(vision_times) / len(vision_times)
        print(f"    Vision encoder avg time: {avg_vision_time:.4f} seconds")

        # Profile LM only
        print("  Profiling LM only...")
        _ = compute_lm_only(lev_model, input_ids_lev_tensor)  # Warmup/compile
        _.array.block_until_ready()
        lm_times = []
        for i in range(3):
            start_time = time.time()
            _ = compute_lm_only(lev_model, input_ids_lev_tensor)
            _.array.block_until_ready()
            lm_times.append(time.time() - start_time)
        avg_lm_time = sum(lm_times) / len(lm_times)
        print(f"    LM only avg time: {avg_lm_time:.4f} seconds")

        print(f"    Vision + LM separate: {avg_vision_time + avg_lm_time:.4f} seconds")

        # First call includes JIT compilation
        print("\n  First forward pass (includes JIT compilation)...")
        start_time = time.time()
        lev_logits_first = compute_lev(lev_model, input_ids_lev_tensor, pixel_values_lev, grid_mask, unpad_indices)
        lev_logits_first.array.block_until_ready()
        first_forward_time = time.time() - start_time
        print(f"  First forward pass time: {first_forward_time:.4f} seconds")

        # Warmup runs
        print("  Running warmup passes...")
        for i in range(3):
            _ = compute_lev(lev_model, input_ids_lev_tensor, pixel_values_lev, grid_mask, unpad_indices)
            _.array.block_until_ready()

        # Measure execution time (excluding compilation)
        print("  Measuring forward pass time (averaging over 5 runs)...")
        times = []
        for i in range(5):
            start_time = time.time()
            _ = compute_lev(lev_model, input_ids_lev_tensor, pixel_values_lev, grid_mask, unpad_indices)
            _.array.block_until_ready()
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.4f} seconds")

        avg_forward_time = sum(times) / len(times)
        min_forward_time = min(times)
        max_forward_time = max(times)
        lev_logits = lev_logits_first.array
        print(f"  Average forward pass time: {avg_forward_time:.4f} seconds")
        print(f"  Min: {min_forward_time:.4f} seconds, Max: {max_forward_time:.4f} seconds")
        print("\n  --- Component breakdown ---")
        print(f"    Vision encoder: {avg_vision_time:.4f}s ({100*avg_vision_time/avg_forward_time:.1f}%)")
        print(f"    LM only:        {avg_lm_time:.4f}s ({100*avg_lm_time/avg_forward_time:.1f}%)")
        print(f"    Overhead:       {avg_forward_time - avg_vision_time - avg_lm_time:.4f}s")

        print(f"Lev logits shape: {lev_logits.shape}")
        print(
            f"Lev logits stats: min={lev_logits.min():.4f}, max={lev_logits.max():.4f}, mean={lev_logits.mean():.4f}"
        )
        print(f"Lev first 5 logits: {np.array(lev_logits).flatten()[:5]}")

    # ===== Compare logits by region =====
    # Use compare_logits_by_region from test_image_utils for unified comparison
    print("\n--- [Timing] Comparison by Region ---")
    start_time = time.time()

    # Prepare logits for comparison
    lev_logits_np = np.array(lev_logits)
    if lev_logits_np.ndim == 3:
        lev_logits_np = lev_logits_np[0]  # Remove batch dimension
    hf_logits_flat = hf_logits[0]  # Remove batch dimension

    print(f"HF logits shape: {hf_logits_flat.shape}")
    print(f"Lev logits shape: {lev_logits_np.shape}")

    # Use compare_logits_by_region for unified comparison
    result = compare_logits_by_region(
        hf_logits=hf_logits_flat,
        lev_logits=lev_logits_np,
        input_ids=test_pair.hf.input_ids,
        image_token_id=hf_config.image_token_index,
        tolerance=1e-2,
        verbose=True,
        detailed=True,
    )

    compare_time = time.time() - start_time
    print(f"\n  Comparison time: {compare_time:.4f} seconds")

    # Print timing summary
    print("\n=== Timing Summary ===")
    print(f"Image loading:           {image_load_time:.4f} seconds")
    print(f"HF model loading:        {hf_load_time:.4f} seconds")
    print(f"Processor (input prep):  {processor_time:.4f} seconds")
    print(f"HF forward pass:         {hf_forward_time:.4f} seconds")
    print(f"Config conversion:       {config_time:.4f} seconds")
    print(f"Model conversion:        {model_convert_time:.4f} seconds")
    print(f"Levanter input prep:     {input_prep_time:.4f} seconds")
    print(f"Levanter forward (first): {first_forward_time:.4f} seconds (includes JIT)")
    print(f"Levanter forward (avg):   {avg_forward_time:.4f} seconds")
    print(f"  - Vision encoder:      {avg_vision_time:.4f} seconds")
    print(f"  - LM only:             {avg_lm_time:.4f} seconds")
    print(f"Comparison:              {compare_time:.4f} seconds")
    total_time = (
        image_load_time
        + hf_load_time
        + processor_time
        + hf_forward_time
        + model_convert_time
        + input_prep_time
        + first_forward_time
        + compare_time
    )
    print(f"Total time:              {total_time:.4f} seconds")

    assert (
        result.passed
    ), f"Real image/text test failed: max diff = {result.overall_max_diff}, mean diff = {result.overall_mean_diff}"
    print("✓ Real image and text input produces matching results!")


@skip_if_no_torch
def test_llava_onevision_real_image_text_0_5b_batch():
    """Test with batch padding for better TPU utilization.

    TPU has 8 devices for data parallel, so batch=8 enables proper sharding.
    This test pads the input to batch=8 and compares with HF reference.
    Uses 0.5B model to fit in memory.
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )
    from levanter.data.image import create_custom_processor

    print("\n=== Test: Real Image with Batch Padding (batch=8) ===")

    # Load real image
    print("\n--- [Timing] Loading Image ---")
    start_time = time.time()
    image = get_single_image()
    print(f"Loaded image: {image.size}")
    image_load_time = time.time() - start_time
    print(f"  Time: {image_load_time:.4f} seconds")

    # Use 0.5B model for testing (fits in TPU memory)
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    # Load HF model first to get reference logits
    print("\n--- [Timing] Loading HuggingFace Model ---")
    start_time = time.time()
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.model.image_newline = None  # Disable image_newline for consistency
        torch_model.eval()
        # Update image_grid_pinpoints in config to 3x3 grid (matches anyres_max_9)
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
        # Create two processors: HF uses unpadded, Levanter uses padded
        processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
        processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return
    hf_load_time = time.time() - start_time
    print(f"  Time: {hf_load_time:.4f} seconds")

    # Prepare inputs with processor
    print("\n--- [Timing] Preparing Inputs with Processor ---")
    start_time = time.time()
    text = "Describe this image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)
    # HF inputs (unpadded)
    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    # Levanter inputs (padded)
    inputs_lev = processor_lev(
        images=image, text=prompt, return_tensors="pt", padding="max_length", max_length=8192, padding_mode=True
    )
    processor_time = time.time() - start_time
    print(f"  Time: {processor_time:.4f} seconds")

    print(f"HF input_ids shape: {inputs_hf['input_ids'].shape}")
    print(f"Lev input_ids shape: {inputs_lev['input_ids'].shape}")
    print(f"HF pixel_values shape: {inputs_hf['pixel_values'].shape}")
    print(f"Lev pixel_values shape: {inputs_lev['pixel_values'].shape}")

    # Run HF forward pass to get reference logits
    print("\n--- [Timing] HuggingFace Forward Pass ---")
    start_time = time.time()
    with torch.no_grad():
        hf_output = torch_model(**inputs_hf)
        hf_logits = hf_output.logits.detach().cpu().numpy()
    hf_forward_time = time.time() - start_time
    print(f"  Time: {hf_forward_time:.4f} seconds")
    print(f"HF logits shape: {hf_logits.shape}")

    # Get image token info for later use
    image_token_id = torch_model.config.image_token_index
    input_ids_for_mask = inputs_hf["input_ids"].numpy()[0]
    image_mask = input_ids_for_mask == image_token_id
    num_image_tokens = image_mask.sum()

    # Delete HF model to free memory
    del torch_model
    import gc

    gc.collect()

    # Convert to Levanter
    print("\n--- [Timing] Converting to Levanter ---")
    start_time = time.time()
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    config_time = time.time() - start_time
    print(f"  Config conversion time: {config_time:.4f} seconds")

    print("\n--- [Timing] Loading Model ---")
    start_time = time.time()

    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig

    # Use proper sharding with batch=8 (divisible by data axis size=8)
    # Add vision_batch to compute_mapping so it gets sharded across TPU devices
    mesh_config = MeshConfig(
        compute_mapping={
            "vision_batch": DEFAULT_DP_AXES,  # Shard vision_batch like batch
        }
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    # Use compute_axis_mapping for proper sharding across TPU devices
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        compute_dtype = jnp.float32

        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )

        model_convert_time = time.time() - start_time
        print(f"  Total conversion time: {model_convert_time:.4f} seconds")

        # Prepare Levanter inputs WITH BATCH PADDING
        print("\n--- [Timing] Preparing Levanter Inputs (with batch padding) ---")
        start_time = time.time()
        original_batch_size = inputs_lev["input_ids"].shape[0]
        seq_len = inputs_lev["input_ids"].shape[1]

        # Pad batch to 8 for proper TPU sharding (divisible by data axis size=8)
        target_batch_size = 8
        print(f"  Original batch size: {original_batch_size}, padding to: {target_batch_size}")

        Batch = Axis("batch", target_batch_size)
        Position = Axis("position", seq_len)

        # Pad input_ids by repeating the first sample
        input_ids_np = inputs_lev["input_ids"].numpy()
        input_ids_np = np.tile(input_ids_np, (target_batch_size, 1))
        input_ids_lev = hax.named(jnp.array(input_ids_np, dtype=jnp.int32), (Batch, Position))

        # Handle pixel_values
        pixel_values_torch = inputs_lev["pixel_values"]
        print(f"pixel_values_torch shape: {pixel_values_torch.shape}")

        if pixel_values_torch.dim() == 5:
            num_patches = pixel_values_torch.shape[1]
            channels = pixel_values_torch.shape[2]
            height = pixel_values_torch.shape[3]
            width = pixel_values_torch.shape[4]

            NumPatches = Axis("num_patches", num_patches)
            Channels = Axis("channels", channels)
            Height = Axis("height", height)
            Width = Axis("width", width)

            # Pad pixel_values by repeating
            pixel_values_np = pixel_values_torch.numpy().astype(np.float32)
            pixel_values_np = np.tile(pixel_values_np, (target_batch_size, 1, 1, 1, 1))

            pixel_values_lev = hax.named(
                jnp.array(pixel_values_np, dtype=jnp.float32),
                (Batch, NumPatches, Channels, Height, Width),
            )
            spatial_shapes_np = inputs_lev.get("spatial_shapes")
            if spatial_shapes_np is not None:
                spatial_shapes_np = spatial_shapes_np.numpy()
        else:
            raise ValueError(f"Pixel values shape: {pixel_values_torch.shape}")

        # Get image_sizes and pad
        image_sizes_torch = inputs_lev.get("image_sizes")
        if image_sizes_torch is None:
            raise ValueError("Processor outputs must include image_sizes")
        image_sizes_np = image_sizes_torch.numpy()
        image_sizes_np = np.tile(image_sizes_np, (target_batch_size, 1))

        # Create grid_mask for fixed-shape processing
        actual_patches = pixel_values_torch.shape[1]  # num_patches from processor
        # Compute total_patches from image_grid_pinpoints
        patch_size = config.vision_config.image_size
        max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
        max_patches_per_dim = max_resolution // patch_size
        total_patches = max_patches_per_dim * max_patches_per_dim + 1  # +1 for base image
        grid_mask_np = create_grid_mask(actual_patches, total_patches)

        # Pad pixel_values to fixed size (already tiled for batch)
        pv_np = pixel_values_torch.numpy().astype(np.float32)
        pv_padded_single = pad_pixel_values(pv_np[0], total_patches)
        pv_padded_np = np.tile(np.expand_dims(pv_padded_single, 0), (target_batch_size, 1, 1, 1, 1))

        # Create Levanter tensors with padded shapes
        NumPatchesPadded = Axis("num_patches", total_patches)
        GridMaskAxis = Axis("grid_mask", total_patches)
        pixel_values_lev = hax.named(
            jnp.array(pv_padded_np, dtype=jnp.float32),
            (Batch, NumPatchesPadded, Channels, Height, Width),
        )
        grid_mask_tiled = np.tile(np.expand_dims(grid_mask_np, 0), (target_batch_size, 1))
        grid_mask = hax.named(jnp.array(grid_mask_tiled), (Batch, GridMaskAxis))

        # Compute unpad_indices for HF-compatible feature ordering
        image_sizes = inputs_lev["image_sizes"].tolist()
        unpad_indices_np = processor_lev.compute_unpad_indices(
            image_sizes=image_sizes,
            height=patch_size,
            width=patch_size,
            max_num_features=num_image_tokens,
        )
        # Tile for batch
        unpad_indices_np = np.tile(unpad_indices_np, (target_batch_size, 1))
        NumImageTokens = Axis("num_image_tokens", num_image_tokens)
        unpad_indices = hax.named(jnp.array(unpad_indices_np, dtype=jnp.int32), (Batch, NumImageTokens))
        print(f"  unpad_indices shape: {unpad_indices.array.shape}")

        input_prep_time = time.time() - start_time
        print(f"  Time: {input_prep_time:.4f} seconds")

        print("\n--- [Timing] Levanter Forward Pass (batch=8) ---")

        # Full forward pass function with unpad_indices
        @hax.named_jit
        def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=None,
            )

        # Forward pass (includes JIT compilation)
        print("\n  Forward pass (includes JIT compilation)...")
        start_time = time.time()
        lev_logits = compute_lev(lev_model, input_ids_lev, pixel_values_lev, grid_mask, unpad_indices)
        lev_logits.array.block_until_ready()
        forward_time = time.time() - start_time
        print(f"  Forward pass time: {forward_time:.4f} seconds")
        print(f"  lev_logits shape: {lev_logits.array.shape}")

        # Compare logits with HF reference (first sample only, since all are identical)
        print("\n--- [Timing] Comparing Logits ---")
        start_time = time.time()

        # Get first sample from Levanter logits
        lev_logits_np = np.array(lev_logits.array[0])  # (seq_len, vocab_size)
        hf_logits_flat = hf_logits[0]  # (seq_len, vocab_size)

        print(f"HF logits shape: {hf_logits_flat.shape}")
        print(f"Lev logits shape: {lev_logits_np.shape}")

        # Compare by region using compare_logits_by_region
        tolerance = 1e-2
        attention_mask_np = inputs_lev["attention_mask"].numpy()[0]
        result = compare_logits_by_region(
            hf_logits=hf_logits_flat,
            lev_logits=lev_logits_np,
            input_ids=input_ids_for_mask,
            image_token_id=image_token_id,
            tolerance=tolerance,
            verbose=True,
            detailed=True,
            attention_mask=attention_mask_np,
        )

        compare_time = time.time() - start_time
        print(f"\n  Comparison time: {compare_time:.4f} seconds")

        all_ok = result.passed

        # Print timing summary
        print("\n=== Timing Summary ===")
        print(f"Image loading:           {image_load_time:.4f} seconds")
        print(f"HF model loading:        {hf_load_time:.4f} seconds")
        print(f"Processor (input prep):  {processor_time:.4f} seconds")
        print(f"HF forward pass:         {hf_forward_time:.4f} seconds")
        print(f"Config conversion:       {config_time:.4f} seconds")
        print(f"Model conversion:        {model_convert_time:.4f} seconds")
        print(f"Levanter input prep:     {input_prep_time:.4f} seconds")
        print(f"Levanter forward:        {forward_time:.4f} seconds (batch={target_batch_size})")
        print(f"Comparison:              {compare_time:.4f} seconds")

        assert (
            all_ok
        ), f"Batch test failed: pre_mean={result.pre_image_mean_diff:.6e}, img_mean={result.image_mean_diff:.6e}, post_mean={result.post_image_mean_diff:.6e}"
        print("\n✓ Batch test completed successfully!")


@pytest.mark.skip(
    reason="Skipping test_llava_onevision_generation, beacuse padded flash attention is not supported yet"
)
@skip_if_no_torch
def test_llava_onevision_generation():
    """Test generation consistency between HuggingFace and Levanter/JAX implementations.

    This test compares the generated text from both implementations using greedy decoding
    to verify that the Levanter model produces the same output as HuggingFace.
    Uses ImageTextExample for new data API compatibility.
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )
    import equinox as eqx
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict
    from levanter.data.image import create_custom_processor
    from levanter.data.image import ImageTextExample
    from haliax import NamedArray

    print("\n=== Test: Generation Consistency ===")

    # Load real image
    image = get_single_image()
    print(f"Loaded image: {image.size}")

    # Use a small pretrained model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print(f"Loading HuggingFace model and processor: {model_name}")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.model.image_newline = None  # Disable for consistency
        torch_model.eval()

        # Use 3x3 grid (matches other tests)
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS

        # Create processors (HF unpadded, Levanter padded)
        processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
        processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return

    # Prepare inputs with processor
    text = "Describe the image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)
    # HF inputs (unpadded)
    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    # Levanter inputs (padded)
    inputs_lev = processor_lev(
        images=image, text=prompt, return_tensors="pt", padding="max_length", max_length=8192, padding_mode=True
    )

    print(f"Processor output keys (HF): {inputs_hf.keys()}")
    print(f"HF input_ids shape: {inputs_hf['input_ids'].shape}")
    print(f"HF pixel_values shape: {inputs_hf['pixel_values'].shape}")
    print(f"Lev input_ids shape: {inputs_lev['input_ids'].shape}")
    print(f"Lev pixel_values shape: {inputs_lev['pixel_values'].shape}")

    # HuggingFace generation with greedy decoding
    max_new_tokens = 30
    print(f"\n--- HuggingFace Generation (max_new_tokens={max_new_tokens}) ---")
    with torch.no_grad():
        hf_output_ids = torch_model.generate(
            **inputs_hf,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=processor_hf.tokenizer.pad_token_id,
        )

    # Get only the generated tokens (excluding prompt)
    prompt_len = inputs_hf["input_ids"].shape[1]
    hf_generated_ids = hf_output_ids[0, prompt_len:].cpu().numpy()
    hf_generated_text = processor_hf.decode(hf_generated_ids, skip_special_tokens=True)
    print(f"HF generated tokens: {hf_generated_ids[:10]}...")
    print(f"HF generated text: {hf_generated_text[:200]}...")

    # Convert to Levanter
    print("\n--- Converting to Levanter ---")
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Disable flash attention for this test
    text_config_updated = dataclasses.replace(config.text_config, attn_backend="dot", flash_attention_block_size=None)
    config = dataclasses.replace(config, text_config=text_config_updated)

    # Load directly from HuggingFace instead of saving to temp directory
    # This avoids processor.save_pretrained() issues with audio_tokenizer
    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

    # Use single-device mesh to avoid sharding issues
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with use_test_mesh(mesh=single_device_mesh):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        state_dict = converter.load_state_dict(model_name)
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    # Convert model weights to float32 for consistency
    lev_model = jtu.tree_map(_to_float32, lev_model)

    # Prepare Levanter inputs using ImageTextExample
    print("\n--- Levanter Generation ---")
    batch_size = inputs_lev["input_ids"].shape[0]
    seq_len = inputs_lev["input_ids"].shape[1]

    Batch = Axis("batch", batch_size)
    Position = Axis("position", seq_len)

    # Handle pixel_values
    pixel_values_torch = inputs_lev["pixel_values"]
    if pixel_values_torch.dim() != 5:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values_torch.shape}")

    _num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    height = pixel_values_torch.shape[3]
    width = pixel_values_torch.shape[4]

    # Calculate total_patches for fixed-shape processing
    patch_size = config.vision_config.image_size
    max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
    max_patches_per_dim = max_resolution // patch_size
    total_patches = max_patches_per_dim * max_patches_per_dim + 1  # +1 for base image

    NumPatches = Axis("num_patches", total_patches)
    Channels = Axis("channels", channels)
    Height = Axis("height", height)
    Width = Axis("width", width)

    # Pad pixel_values and create grid_mask
    pv_np = inputs_lev["pixel_values"].numpy().astype(np.float32)[0]  # Remove batch dim
    pv_padded = pad_pixel_values(pv_np, total_patches)
    actual_patches = inputs_lev["pixel_values"].shape[1]
    grid_mask_np = create_grid_mask(actual_patches, total_patches)

    # Create NamedArrays (without batch dimension for ImageTextExample)
    pixel_values_named = NamedArray(pv_padded, (NumPatches, Channels, Height, Width))
    input_ids_named = NamedArray(inputs_lev["input_ids"].numpy()[0].astype(np.int32), (Position,))
    grid_mask_named = NamedArray(grid_mask_np, (NumPatches,))

    # Compute unpad_indices
    image_sizes = inputs_lev["image_sizes"].tolist()
    num_image_tokens = int((inputs_hf["input_ids"].numpy() == torch_model.config.image_token_index).sum())
    unpad_indices_np = processor_lev.compute_unpad_indices(
        image_sizes=image_sizes,
        height=patch_size,
        width=patch_size,
        max_num_features=num_image_tokens,
    )
    # compute_unpad_indices returns (1, num_tokens) or (num_tokens,), squeeze to 1D
    if unpad_indices_np.ndim == 2:
        unpad_indices_np = unpad_indices_np[0]
    NumImageTokens = Axis("num_image_tokens", num_image_tokens)
    unpad_indices_named = NamedArray(unpad_indices_np.astype(np.int32), (NumImageTokens,))

    # Create ImageTextExample
    example = ImageTextExample(
        pixel_values=pixel_values_named,
        input_ids=input_ids_named,
        loss_mask=None,
        grid_mask=grid_mask_named,
        unpad_indices=unpad_indices_named,
    )
    print("Created ImageTextExample with:")
    print(f"  pixel_values: {example.pixel_values.array.shape}")
    print(f"  input_ids: {example.input_ids.array.shape}")
    print(f"  grid_mask: {example.grid_mask.array.shape}")
    print(f"  unpad_indices: {example.unpad_indices.array.shape}")

    # Add batch dimension for model forward pass
    def add_batch(arr, Batch):
        return hax.named(jnp.expand_dims(jnp.array(arr.array), 0), (Batch,) + arr.axes)

    pixel_values_lev = add_batch(example.pixel_values, Batch)
    input_ids_lev = add_batch(example.input_ids, Batch)
    grid_mask = add_batch(example.grid_mask, Batch)
    unpad_indices = add_batch(example.unpad_indices, Batch)

    # Greedy generation loop for Levanter
    # Strategy: Compute merged embeddings once (image + text), then use LM transformer for generation.
    # This avoids recomputing image features at every step.

    # Step 1: Get merged embeddings from LlavaOnevision (image + text)
    @hax.named_jit
    def get_merged_embeddings(model, input_ids, pixel_values, grid_mask, unpad_indices):
        """Get merged embeddings with image features inserted.

        Replicates the logic from LlavaOnevisionModel.forward() to:
        1. Get image features
        2. Flatten and reorder using unpad_indices
        3. Merge with text embeddings
        """
        # Get input embeddings
        inputs_embeds = model.get_input_embeddings().embed(input_ids)

        # Get image features (without unpad_indices - that's applied after)
        image_features_result = model.get_image_features(
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            key=None,
        )

        # Unpack result - get_image_features returns (features, updated_grid_mask)
        if isinstance(image_features_result, tuple):
            image_features, _ = image_features_result
        else:
            image_features = image_features_result

        # image_features shape: (batch, num_patches, features_per_patch, embed)
        batch_ax = image_features.axes[0]
        num_patches_ax = image_features.axes[1]
        features_per_patch_ax = image_features.axes[2]
        embed_ax = image_features.axes[3]

        features_per_patch = features_per_patch_ax.size
        total_patches = num_patches_ax.size
        total_image_tokens = total_patches * features_per_patch

        # Flatten image features to (batch, total_image_tokens, embed)
        ImageTokens = Axis("image_tokens", total_image_tokens)
        image_features_flat = hax.flatten_axes(image_features, (num_patches_ax, features_per_patch_ax), ImageTokens)

        # Apply unpad_indices to reorder features to HF's spatial order
        if unpad_indices is not None:
            num_unpadded_tokens = unpad_indices.axis_size("num_image_tokens")

            def gather_unpadded(features, indices):
                # features: (total_image_tokens, embed)
                # indices: (num_unpadded_tokens,)
                return features[indices]

            image_features_reordered = jax.vmap(gather_unpadded)(image_features_flat.array, unpad_indices.array)
            UnpaddedTokens = Axis("image_tokens", num_unpadded_tokens)
            image_features_flat = hax.named(image_features_reordered, (batch_ax, UnpaddedTokens, embed_ax))

        # Get placeholder mask
        special_image_mask = model.get_placeholder_mask(input_ids, image_features_flat)

        batch_size_val = inputs_embeds.axes[0].size
        seq_len_val = inputs_embeds.axes[1].size
        embed_size = inputs_embeds.axes[2].size

        inputs_flat = inputs_embeds.array.reshape(batch_size_val * seq_len_val, embed_size)
        # Mask is now (batch, position), flatten it directly
        mask_flat = special_image_mask.array.reshape(batch_size_val * seq_len_val)

        feature_indices = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1
        feature_indices = jnp.clip(feature_indices, 0, image_features_flat.axis_size("image_tokens") - 1)

        # Flatten image features for gathering
        img_feat_flat = image_features_flat.array.reshape(-1, embed_size)
        gathered_features = img_feat_flat[feature_indices]
        inputs_flat = jnp.where(mask_flat[:, None], gathered_features, inputs_flat)

        merged_embeds = inputs_flat.reshape(batch_size_val, seq_len_val, embed_size)
        return hax.named(merged_embeds, inputs_embeds.axes)

    # Get the merged embeddings (image features + text embeddings)
    merged_embeds = get_merged_embeddings(lev_model, input_ids_lev, pixel_values_lev, grid_mask, unpad_indices)
    print(f"Merged embeddings shape: {merged_embeds.array.shape}")

    # Now use the language model's transformer directly for generation
    lm = lev_model.language_model
    from levanter.layers.attention import AttentionMask

    @hax.named_jit
    def forward_with_embeds(transformer, lm_head, embeds, TextEmbed):
        """Forward pass using embeddings directly."""
        causal_mask = AttentionMask.causal()
        activations = transformer(embeds, attn_mask=causal_mask, key=None)
        logits = hax.dot(activations, lm_head, axis=TextEmbed)
        return logits

    # Generation loop
    generated_tokens = []
    current_embeds = merged_embeds
    TextEmbed = lev_model.config.TextEmbed

    for step in range(max_new_tokens):
        # Forward pass through transformer
        logits = forward_with_embeds(lm.transformer, lm.get_lm_head(), current_embeds, TextEmbed)

        # Get logits for the last position
        logits_np = np.array(logits.array)
        last_logits = logits_np[0, -1, :]  # (vocab_size,)

        # Greedy: pick the token with highest logit
        next_token = int(np.argmax(last_logits))
        generated_tokens.append(next_token)

        # Check for EOS token
        if next_token == processor_hf.tokenizer.eos_token_id:
            print(f"  EOS token reached at step {step + 1}")
            break

        # Get embedding for the new token and append
        new_token_arr = jnp.array([[next_token]], dtype=jnp.int32)
        NewPosition = Axis("position", 1)
        new_token_named = hax.named(new_token_arr, (Batch, NewPosition))
        new_embed = lm.embeddings.embed(new_token_named)

        # Concatenate to current embeddings along position axis
        current_embeds_arr = current_embeds.array
        new_embed_arr = new_embed.array
        concat_arr = jnp.concatenate([current_embeds_arr, new_embed_arr], axis=1)

        # Create new position axis with updated size
        new_seq_len = current_embeds.axes[1].size + 1
        NewFullPosition = Axis("position", new_seq_len)
        current_embeds = hax.named(concat_arr, (Batch, NewFullPosition, current_embeds.axes[2]))

        if (step + 1) % 10 == 0:
            print(f"  Generated {step + 1} tokens...")

    lev_generated_ids = np.array(generated_tokens)
    lev_generated_text = processor_hf.decode(lev_generated_ids, skip_special_tokens=True)
    print(f"\nLev generated tokens: {lev_generated_ids[:10]}...")
    print(f"Lev generated text: {lev_generated_text[:200]}...")

    # Compare results
    print("\n--- Comparison ---")
    print(f"HF generated {len(hf_generated_ids)} tokens")
    print(f"Lev generated {len(lev_generated_ids)} tokens")

    # Compare token by token
    min_len = min(len(hf_generated_ids), len(lev_generated_ids))
    matching_tokens = sum(1 for i in range(min_len) if hf_generated_ids[i] == lev_generated_ids[i])
    match_ratio = matching_tokens / min_len if min_len > 0 else 0

    print(f"Matching tokens: {matching_tokens}/{min_len} ({match_ratio:.1%})")

    # Show first few token comparisons
    print("\nFirst 20 token comparison:")
    for i in range(min(20, min_len)):
        hf_tok = hf_generated_ids[i]
        lev_tok = lev_generated_ids[i]
        match = "✓" if hf_tok == lev_tok else "✗"
        hf_word = processor_hf.decode([hf_tok])
        lev_word = processor_hf.decode([lev_tok])
        print(f"  [{i:2d}] HF: {hf_tok:6d} ({hf_word!r:15s}) | Lev: {lev_tok:6d} ({lev_word!r:15s}) {match}")

    # Check if texts match
    texts_match = hf_generated_text.strip() == lev_generated_text.strip()
    print(f"\n{'✓ PASS' if texts_match else '✗ FAIL'}: Generated texts {'match' if texts_match else 'do not match'}")

    # For now, we check if at least 80% of tokens match (allowing for small numerical differences)
    assert match_ratio >= 0.8, f"Token match ratio too low: {match_ratio:.1%}"
    print("✓ Generation test passed!")


@skip_if_no_torch
def test_llava_onevision_generation_with_kv_cache():
    """Test generation with KV cache for efficient autoregressive decoding.

    This test uses the Qwen transformer's KV cache mechanism:
    1. First, compute merged embeddings (image + text) using LlavaOnevision
    2. Prefill the KV cache with merged embeddings
    3. Generate new tokens using cached KV states

    Uses ImageTextExample for new data API compatibility.
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )
    import equinox as eqx
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict
    from levanter.inference.page_table import PageTable, PageBatchInfo
    from levanter.data.image import create_custom_processor

    print("\n=== Test: Generation with KV Cache ===")

    # Load real image
    image = get_single_image()
    print(f"Loaded image: {image.size}")

    # Use a small pretrained model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print(f"Loading HuggingFace model and processor: {model_name}")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.eval()

        # Disable image_newline for consistency with other tests
        torch_model.model.image_newline = None
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS

        # Create processors (HF unpadded, Levanter padded)
        processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
        processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return

    # Prepare inputs with both processors
    text = "Describe the image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)
    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    inputs_lev = processor_lev(images=image, text=prompt, return_tensors="pt")

    print(f"Processor output keys (HF): {inputs_hf.keys()}")
    print(f"input_ids shape (HF): {inputs_hf['input_ids'].shape}")
    print(f"pixel_values shape (HF): {inputs_hf['pixel_values'].shape}")
    print(f"Processor output keys (Lev): {inputs_lev.keys()}")
    print(f"input_ids shape (Lev): {inputs_lev['input_ids'].shape}")
    print(f"pixel_values shape (Lev): {inputs_lev['pixel_values'].shape}")

    # HuggingFace generation with greedy decoding
    max_new_tokens = 500
    print(f"\n--- HuggingFace Generation (max_new_tokens={max_new_tokens}) ---")
    with torch.no_grad():
        hf_output_ids = torch_model.generate(
            **inputs_hf,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=processor_hf.tokenizer.pad_token_id,
        )

    # Get only the generated tokens (excluding prompt)
    prompt_len = inputs_hf["input_ids"].shape[1]
    hf_generated_ids = hf_output_ids[0, prompt_len:].cpu().numpy()
    hf_generated_text = processor_hf.decode(hf_generated_ids, skip_special_tokens=True)
    print(f"HF generated tokens: {hf_generated_ids[:10]}...")
    print(f"HF generated text: {hf_generated_text[:200]}...")

    # Convert to Levanter
    print("\n--- Converting to Levanter ---")
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Disable flash attention for this test
    text_config_updated = dataclasses.replace(config.text_config, attn_backend="dot", flash_attention_block_size=None)
    config = dataclasses.replace(config, text_config=text_config_updated)

    # Load from HuggingFace directly (avoid temp directory issues with audio_tokenizer)
    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

    # Use single-device mesh to avoid sharding issues
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with use_test_mesh(mesh=single_device_mesh):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        state_dict = converter.load_state_dict(model_name)
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    # Convert weights to float32 (model may have float16 weights)
    lev_model = jtu.tree_map(_to_float32, lev_model)

    # Prepare Levanter inputs
    print("\n--- Levanter Generation with KV Cache ---")
    batch_size = inputs_lev["input_ids"].shape[0]
    seq_len = inputs_lev["input_ids"].shape[1]

    Batch = Axis("batch", batch_size)
    Position = Axis("position", seq_len)

    input_ids_np = inputs_lev["input_ids"].numpy()

    # Handle pixel_values
    pixel_values_torch = inputs_lev["pixel_values"]
    if pixel_values_torch.dim() == 5:
        num_patches = pixel_values_torch.shape[1]
        channels = pixel_values_torch.shape[2]
        height = pixel_values_torch.shape[3]
        width = pixel_values_torch.shape[4]

        NumPatches = Axis("num_patches", num_patches)
        Channels = Axis("channels", channels)
        Height = Axis("height", height)
        Width = Axis("width", width)

        pixel_values_lev = hax.named(
            jnp.array(pixel_values_torch.numpy().astype(np.float32), dtype=jnp.float32),
            (Batch, NumPatches, Channels, Height, Width),
        )
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values_torch.shape}")

    # Create grid_mask for fixed-shape processing
    actual_patches = pixel_values_torch.shape[1]  # num_patches from processor
    # Compute total_patches from image_grid_pinpoints
    patch_size = config.vision_config.image_size
    max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
    max_patches_per_dim = max_resolution // patch_size
    total_patches = max_patches_per_dim * max_patches_per_dim + 1  # +1 for base image
    grid_mask_np = create_grid_mask(actual_patches, total_patches)

    # Pad pixel_values to fixed size
    pv_np = pixel_values_torch.numpy().astype(np.float32)
    pv_padded_np = pad_pixel_values(pv_np[0], total_patches)  # Remove batch dim, pad, then add back
    pv_padded_np = np.expand_dims(pv_padded_np, 0)  # Add batch dim back

    # Create Levanter tensors with padded shapes
    NumPatchesPadded = Axis("num_patches", total_patches)
    GridMaskAxis = Axis("grid_mask", total_patches)
    pixel_values_lev = hax.named(
        jnp.array(pv_padded_np, dtype=jnp.float32),
        (Batch, NumPatchesPadded, Channels, Height, Width),
    )
    grid_mask = hax.named(jnp.array(np.expand_dims(grid_mask_np, 0)), (Batch, GridMaskAxis))

    # Compute unpad_indices for HF-compatible spatial ordering
    image_sizes = inputs_lev["image_sizes"].tolist()
    num_image_tokens = (inputs_hf["input_ids"].numpy() == torch_model.config.image_token_index).sum()
    unpad_indices_np = processor_lev.compute_unpad_indices(
        image_sizes=image_sizes,
        height=patch_size,
        width=patch_size,
        max_num_features=num_image_tokens,
    )
    if unpad_indices_np.ndim == 2:
        unpad_indices_np = unpad_indices_np[0]  # Squeeze to 1D if needed
    NumImageTokens = Axis("num_image_tokens", num_image_tokens)
    # Add batch dimension
    unpad_indices_batched = hax.named(
        jnp.expand_dims(jnp.array(unpad_indices_np.astype(np.int32)), 0),
        (Batch, NumImageTokens),
    )
    print(f"unpad_indices shape: {unpad_indices_batched.shape}")

    # Step 1: Prepare text ids (image merging happens inside model.decode)
    input_ids_lev = hax.named(jnp.array(input_ids_np, dtype=jnp.int32), (Batch, Position))
    seq_len_val = input_ids_lev.axes[1].size

    # Step 2: Setup KV cache infrastructure
    lm = lev_model.language_model  # Used for embedding new tokens

    # Page table configuration
    # Use smaller page_size for float32 to reduce VMEM usage in Pallas kernel
    # (TPU v4 has 16MB VMEM limit, float32 uses 2x memory vs bf16)
    page_size = 16  # tokens per page (reduced from 32/64 to fit VMEM)
    max_seq_len = seq_len + max_new_tokens + 64  # total sequence length with buffer
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    max_pages = max_pages_per_seq * batch_size + 128  # total pages with buffer (increase for smaller page_size)
    max_seqs = batch_size

    # Create page table and initial KV cache
    page_table = PageTable.init(max_pages, max_seqs, page_size, max_pages_per_seq)
    spec = page_table.spec()

    # Initialize KV cache using the model's method
    kv_cache = lev_model.initial_cache(spec, dtype=jnp.float32)
    print(f"KV cache initialized with {max_pages} pages, page_size={page_size}")

    # Helper function to create PageBatchInfo for prefill/decode
    def make_batch_info_for_prefill(seq_len_val, page_table, page_size):
        """Create PageBatchInfo for prefilling a sequence."""
        # For a single sequence prefill
        num_pages_needed = (seq_len_val + page_size - 1) // page_size

        # Allocate pages (simple sequential allocation for single sequence)
        page_indices = jnp.arange(num_pages_needed, dtype=jnp.int32)
        # Pad to max_pages_per_seq
        page_indices_padded = jnp.full((1, max_pages_per_seq), -1, dtype=jnp.int32)
        page_indices_padded = page_indices_padded.at[0, :num_pages_needed].set(page_indices)

        # Token destinations: each token goes to page * page_size + slot
        token_dests = jnp.zeros(seq_len_val, dtype=jnp.int32)
        for i in range(seq_len_val):
            page_idx = i // page_size
            slot_idx = i % page_size
            token_dests = token_dests.at[i].set(page_indices[page_idx] * page_size + slot_idx)

        # Cumulative query lengths for flash attention
        cu_q_lens = jnp.array([0, seq_len_val], dtype=jnp.int32)

        return PageBatchInfo(
            slot_ids=hax.named(jnp.array([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(page_indices_padded, ("seq", "page")),
            seq_lens=hax.named(jnp.array([seq_len_val], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(cu_q_lens, "seq"),
            num_seqs=jnp.array(1, dtype=jnp.int32),
            new_token_dests=hax.named(token_dests, "position"),
            page_size=page_size,
        )

    def make_batch_info_for_decode(current_len, page_table, page_size):
        """Create PageBatchInfo for decoding a single new token."""
        num_pages_used = (current_len + page_size - 1) // page_size

        # Page indices (same as prefill, we're extending the same sequence)
        page_indices = jnp.arange(num_pages_used, dtype=jnp.int32)
        page_indices_padded = jnp.full((1, max_pages_per_seq), -1, dtype=jnp.int32)
        page_indices_padded = page_indices_padded.at[0, :num_pages_used].set(page_indices)

        # New token destination
        new_page_idx = (current_len - 1) // page_size
        new_slot_idx = (current_len - 1) % page_size
        token_dest = page_indices[new_page_idx] * page_size + new_slot_idx

        cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)

        return PageBatchInfo(
            slot_ids=hax.named(jnp.array([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(page_indices_padded, ("seq", "page")),
            seq_lens=hax.named(jnp.array([current_len], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(cu_q_lens, "seq"),
            num_seqs=jnp.array(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.array([token_dest], dtype=jnp.int32), "position"),
            page_size=page_size,
        )

    # Step 3: Prefill - process all merged embeddings at once
    prefill_seq_len = seq_len_val
    print(f"Prefilling {prefill_seq_len} tokens...")

    # Create position IDs for prefill (shape: batch x position)
    PrefillPos = Axis("position", prefill_seq_len)
    pos_ids_arr = jnp.broadcast_to(jnp.arange(prefill_seq_len, dtype=jnp.int32), (batch_size, prefill_seq_len))
    prefill_pos_ids = hax.named(pos_ids_arr, (Batch, PrefillPos))

    # Create batch info for prefill
    prefill_batch_info = make_batch_info_for_prefill(prefill_seq_len, page_table, page_size)

    # Prefill: run model.decode with on-the-fly embedding merge
    @hax.named_jit
    def prefill_step(model, kv_cache, batch_info, pos_ids, input_ids, pixel_values, grid_mask, unpad_indices):
        """Prefill step: process all embeddings and cache KV states."""
        logits, new_cache = model.decode(
            None,
            kv_cache,
            batch_info,
            pos_ids,
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            key=None,
        )
        return logits, new_cache

    prefill_logits, kv_cache = prefill_step(
        lev_model,
        kv_cache,
        prefill_batch_info,
        prefill_pos_ids,
        input_ids_lev,
        pixel_values_lev,
        grid_mask,
        unpad_indices_batched,
    )

    # Get first token from prefill logits
    prefill_logits_np = np.array(prefill_logits.array)
    last_logits = prefill_logits_np[0, -1, :]
    first_token = int(np.argmax(last_logits))

    print(f"First generated token: {first_token} ({processor_hf.decode([first_token])!r})")

    # Step 4: Autoregressive decoding with KV cache
    generated_tokens = [first_token]
    current_len = prefill_seq_len + 1

    @hax.named_jit
    def decode_step(
        model, kv_cache, token_embed, batch_info, pos_ids, input_ids, pixel_values, grid_mask, unpad_indices
    ):
        """Single decode step with cached KV states."""
        logits, new_cache = model.decode(
            token_embed,
            kv_cache,
            batch_info,
            pos_ids,
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            key=None,
        )
        return logits, new_cache

    for step in range(1, max_new_tokens):
        # Check for EOS
        if generated_tokens[-1] == processor_hf.tokenizer.eos_token_id:
            print(f"  EOS token reached at step {step}")
            break

        # Embed the new token
        new_token = generated_tokens[-1]
        new_token_arr = jnp.array([[new_token]], dtype=jnp.int32)
        DecodePos = Axis("position", 1)
        new_token_named = hax.named(new_token_arr, (Batch, DecodePos))
        new_embed = lm.embeddings.embed(new_token_named)

        # Position ID for the new token
        decode_pos_ids = hax.named(jnp.array([[current_len - 1]], dtype=jnp.int32), (Batch, DecodePos))

        # Batch info for this decode step
        decode_batch_info = make_batch_info_for_decode(current_len, page_table, page_size)

        # Run decode step
        decode_logits, kv_cache = decode_step(
            lev_model,
            kv_cache,
            new_embed,
            decode_batch_info,
            decode_pos_ids,
            input_ids_lev,
            pixel_values_lev,
            grid_mask,
            unpad_indices_batched,
        )

        # Get next token from logits
        decode_logits_np = np.array(decode_logits.array)
        next_logits = decode_logits_np[0, 0, :]  # single token output
        next_token = int(np.argmax(next_logits))

        generated_tokens.append(next_token)
        current_len += 1

        if (step + 1) % 10 == 0:
            print(f"  Generated {step + 1} tokens...")

    lev_generated_ids = np.array(generated_tokens)
    lev_generated_text = processor_hf.decode(lev_generated_ids, skip_special_tokens=True)
    print(f"\nLev generated tokens (KV cache): {lev_generated_ids[:10]}...")
    print(f"Lev generated text: {lev_generated_text[:200]}...")

    # Compare results
    print("\n--- Comparison ---")
    print(f"HF generated {len(hf_generated_ids)} tokens")
    print(f"Lev generated {len(lev_generated_ids)} tokens")

    min_len = min(len(hf_generated_ids), len(lev_generated_ids))
    matching_tokens = sum(1 for i in range(min_len) if hf_generated_ids[i] == lev_generated_ids[i])
    match_ratio = matching_tokens / min_len if min_len > 0 else 0

    print(f"Matching tokens: {matching_tokens}/{min_len} ({match_ratio:.1%})")

    # Show first few token comparisons
    print("\nFirst 20 token comparison:")
    for i in range(min(20, min_len)):
        hf_tok = hf_generated_ids[i]
        lev_tok = lev_generated_ids[i]
        match = "✓" if hf_tok == lev_tok else "✗"
        hf_word = processor_hf.decode([hf_tok])
        lev_word = processor_hf.decode([lev_tok])
        print(f"  [{i:2d}] HF: {hf_tok:6d} ({hf_word!r:15s}) | Lev: {lev_tok:6d} ({lev_word!r:15s}) {match}")

    texts_match = hf_generated_text.strip() == lev_generated_text.strip()
    print(f"\n{'✓ PASS' if texts_match else '✗ FAIL'}: Generated texts {'match' if texts_match else 'do not match'}")

    assert match_ratio >= 0.8, f"Token match ratio too low: {match_ratio:.1%}"
    print("✓ Generation with KV cache test passed!")


@skip_if_no_torch
def test_llava_onevision_generation_with_inference_engine():
    """Test generation using Levanter's built-in InferenceEngine with VLMRequest.

    This test demonstrates how to use LlavaInferenceEngine with VLMRequest:
    1. Load the LlavaOnevision model
    2. Create a LlavaInferenceEngine
    3. Create a VLMRequest with image data (pixel_values, image_sizes, input_ids)
    4. Generate text using the engine's generate API

    Uses ImageTextExample for new data API compatibility.
    """
    import torch
    from transformers import (
        LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    )
    from levanter.trainer import TrainerConfig
    from levanter.data.image import create_custom_processor

    print("\n=== Test: Generation with InferenceEngine using VLMRequest ===")

    # Load real image
    image = get_single_image()
    print(f"Loaded image: {image.size}")

    # Use a small pretrained model for testing (0.5B instead of 7B to fit in TPU VMEM)
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print(f"Loading HuggingFace config and processor: {model_name}")
    try:
        # Only load config and processor, NOT the model to save memory for Levanter loading
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Create processors (HF unpadded, Levanter padded)
        processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
        processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)

        # Comment out torch model loading to save memory - we only need config
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.eval()

        # Disable image_newline for consistency with other tests
        torch_model.model.image_newline = None
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    except Exception as e:
        print(f"Could not load config/processor: {e}")
        pytest.skip(f"Could not download model config: {model_name}")
        return

    # Prepare inputs with both processors
    text = "Describe the image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)
    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    inputs_lev = processor_lev(images=image, text=prompt, return_tensors="pt")

    print(f"Processor output keys (HF): {inputs_hf.keys()}")
    print(f"input_ids shape (HF): {inputs_hf['input_ids'].shape}")
    print(f"pixel_values shape (HF): {inputs_hf['pixel_values'].shape}")
    print(f"Processor output keys (Lev): {inputs_lev.keys()}")
    print(f"input_ids shape (Lev): {inputs_lev['input_ids'].shape}")
    print(f"pixel_values shape (Lev): {inputs_lev['pixel_values'].shape}")

    # HuggingFace generation with greedy decoding
    max_new_tokens = 500
    print(f"\n--- HuggingFace Generation (max_new_tokens={max_new_tokens}) ---")
    with torch.no_grad():
        hf_output_ids = torch_model.generate(
            **inputs_hf,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=processor_hf.tokenizer.pad_token_id,
        )

    # Get only the generated tokens (excluding prompt)
    prompt_len = inputs_hf["input_ids"].shape[1]
    hf_generated_ids = hf_output_ids[0, prompt_len:].cpu().numpy()
    hf_generated_text = processor_hf.decode(hf_generated_ids, skip_special_tokens=True)
    print(f"HF generated tokens: {hf_generated_ids[:10]}...")
    print(f"HF generated text: {hf_generated_text[:200]}...")

    # Convert to Levanter
    print("\n--- Converting to Levanter ---")
    # hf_config already loaded above (not from torch_model.config to save memory)
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Disable flash attention for this test (to match the kv_cache test)
    text_config_updated = dataclasses.replace(config.text_config, attn_backend="dot", flash_attention_block_size=None)
    config = dataclasses.replace(config, text_config=text_config_updated)

    # Prepare Levanter inputs for VLMRequest
    print("\n--- Levanter Generation with LlavaInferenceEngine ---")
    batch_size = inputs_lev["input_ids"].shape[0]
    seq_len = inputs_lev["input_ids"].shape[1]

    Batch = Axis("batch", batch_size)
    Position = Axis("position", seq_len)

    input_ids_np = inputs_lev["input_ids"].numpy()

    # Handle pixel_values
    pixel_values_torch = inputs_lev["pixel_values"]
    if pixel_values_torch.dim() == 5:
        _num_patches = pixel_values_torch.shape[1]
        channels = pixel_values_torch.shape[2]
        height = pixel_values_torch.shape[3]
        width = pixel_values_torch.shape[4]

        _NumPatches = Axis("num_patches", _num_patches)
        Channels = Axis("channels", channels)
        Height = Axis("height", height)
        Width = Axis("width", width)

        pixel_values_np = pixel_values_torch.numpy().astype(np.float32)
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values_torch.shape}")

    # Create grid_mask for fixed-shape processing
    actual_patches = pixel_values_torch.shape[1]  # num_patches from processor
    # Compute total_patches from image_grid_pinpoints
    patch_size = config.vision_config.image_size
    max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
    max_patches_per_dim = max_resolution // patch_size
    total_patches = max_patches_per_dim * max_patches_per_dim + 1  # +1 for base image
    grid_mask_np = create_grid_mask(actual_patches, total_patches)

    # Compute unpad_indices for HF-compatible spatial ordering
    image_sizes = inputs_lev["image_sizes"].tolist()
    num_image_tokens = (inputs_hf["input_ids"].numpy() == torch_model.config.image_token_index).sum()
    unpad_indices_np = processor_lev.compute_unpad_indices(
        image_sizes=image_sizes,
        height=patch_size,
        width=patch_size,
        max_num_features=num_image_tokens,
    )
    if unpad_indices_np.ndim == 2:
        unpad_indices_np = unpad_indices_np[0]  # Squeeze to 1D if needed
    NumImageTokens = Axis("num_image_tokens", num_image_tokens)
    print(f"unpad_indices shape: {unpad_indices_np.shape}")

    # Pad pixel_values to fixed size
    pv_padded_np = pad_pixel_values(pixel_values_np[0], total_patches)
    pv_padded_np = np.expand_dims(pv_padded_np, 0)

    # torch_model not loaded, no need to delete
    import gc

    gc.collect()

    # Enter mesh context for InferenceEngine and model loading
    # Use FSDP (data axis) for sharding - this allows best_effort_sharding to work properly
    # when loading safetensors. model_axis_size=1 means all devices are on the data axis.
    trainer_config = TrainerConfig()  # Default: model_axis_size=1, all devices on data axis
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        mesh = trainer_config.device_mesh
        compute_dtype = jnp.float32
        Vocab = Axis("vocab", hf_config.text_config.vocab_size)

        # Load model using converter.load_pretrained() - same pattern as Qwen3 loading
        # Use parameter_axis_mapping for FSDP sharding (not compute_axis_mapping which is unsharded)
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
            resize_vocab_to_match_tokenizer=False,  # LlavaOnevisionModel doesn't have resize_vocab
        )

        # Create Levanter tensors with padded shapes
        NumPatchesPadded = Axis("num_patches", total_patches)
        GridMaskAxis = Axis("grid_mask", total_patches)
        pixel_values_lev = hax.named(
            jnp.array(pv_padded_np, dtype=jnp.float32),
            (Batch, NumPatchesPadded, Channels, Height, Width),
        )
        grid_mask = hax.named(jnp.array(np.expand_dims(grid_mask_np, 0)), (Batch, GridMaskAxis))

        # Create unpad_indices with batch dimension
        unpad_indices_batched = hax.named(
            jnp.expand_dims(jnp.array(unpad_indices_np.astype(np.int32)), 0),
            (Batch, NumImageTokens),
        )

        input_ids_lev = hax.named(
            jnp.array(input_ids_np, dtype=jnp.int32),
            (Batch, Position),
        )

        # Configure InferenceEngine
        # Note: max_seq_len needs to account for expanded image tokens
        # A rough estimate: each image patch expands to many tokens
        estimated_max_seq_len = seq_len * 10 + max_new_tokens + 64
        # Use smaller page_size for float32 to reduce VMEM usage in Pallas kernel
        # (TPU v4 has 16MB VMEM limit, float32 uses 2x memory vs bf16)
        page_size = 16 if compute_dtype == jnp.float32 else 64
        engine_config = InferenceEngineConfig(
            max_seq_len=estimated_max_seq_len,
            page_size=page_size,
            max_seqs=1,
            max_rounds=32,
            max_stop_seqs=1,
            max_stop_tokens=4,
            max_pages=800,  # Increase max_pages to compensate for smaller page_size
            compute_dtype=compute_dtype,
        )

        # Build the LlavaInferenceEngine inside mesh context
        print("Creating LlavaInferenceEngine...")
        engine = LlavaInferenceEngine.from_model_with_config(
            model=lev_model,
            tokenizer=processor_hf.tokenizer,
            config=engine_config,
            Vocab=Vocab,
            mesh=mesh,
        )
        print(f"LlavaInferenceEngine initialized with max_seq_len={engine_config.max_seq_len}")

        # Use original input_ids as prompt tokens
        prompt_tokens = input_ids_np.flatten().tolist()
        print(f"Prompt tokens: {len(prompt_tokens)} tokens")

        # Create decoding parameters (greedy decoding with temperature=0)
        # NOTE: max_num_tokens is the total sequence length (prompt + generated tokens)
        # Set up EOS token as stop token so generation stops when HF would stop
        eos_token_id = processor_hf.tokenizer.eos_token_id
        if eos_token_id is not None:
            stop_tokens = hax.named(jnp.array([[eos_token_id]], dtype=jnp.int32), ("stop_seq", "position"))
        else:
            stop_tokens = None

        decode_params = SeqDecodingParams(
            max_num_tokens=estimated_max_seq_len,
            temperature=0.0,  # Greedy decoding
            key=random.PRNGKey(42),
            stop_tokens=stop_tokens,
        )

        # Create a VLMRequest with all image data included
        vlm_request = VLMRequest(
            prompt_tokens=prompt_tokens,
            request_id=0,
            decode_params=decode_params,
            n_generations=1,
            pixel_values=pixel_values_lev,
            input_ids=input_ids_lev,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices_batched,
        )

        # Generate using VLMRequest
        print("Starting generation with VLMRequest...")
        result = engine.generate([vlm_request])

    # Extract generated tokens
    lev_generated_ids = np.array(result.tokens[0])
    lev_generated_text = processor_hf.decode(lev_generated_ids, skip_special_tokens=True)
    print(f"\nLev generated tokens (InferenceEngine): {lev_generated_ids[:10]}...")
    print(f"Lev generated text: {lev_generated_text[:200]}...")
    print(f"Total tokens generated: {result.total_generated}")

    # Compare results
    print("\n--- Comparison ---")
    print(f"HF generated {len(hf_generated_ids)} tokens")
    print(f"Lev generated {len(lev_generated_ids)} tokens")

    min_len = min(len(hf_generated_ids), len(lev_generated_ids))
    matching_tokens = sum(1 for i in range(min_len) if hf_generated_ids[i] == lev_generated_ids[i])
    match_ratio = matching_tokens / min_len if min_len > 0 else 0

    print(f"Matching tokens: {matching_tokens}/{min_len} ({match_ratio:.1%})")

    # Show first few token comparisons
    print("\nFirst 20 token comparison:")
    for i in range(min(20, min_len)):
        hf_tok = hf_generated_ids[i]
        lev_tok = lev_generated_ids[i]
        match = "✓" if hf_tok == lev_tok else "✗"
        hf_word = processor_hf.decode([hf_tok])
        lev_word = processor_hf.decode([lev_tok])
        print(f"  [{i:2d}] HF: {hf_tok:6d} ({hf_word!r:15s}) | Lev: {lev_tok:6d} ({lev_word!r:15s}) {match}")

    texts_match = hf_generated_text.strip() == lev_generated_text.strip()
    print(f"\n{'✓ PASS' if texts_match else '✗ FAIL'}: Generated texts {'match' if texts_match else 'do not match'}")

    # Check that we generated a reasonable number of tokens (at least 50% of HF's output)
    min_expected_tokens = len(hf_generated_ids) // 2
    assert len(lev_generated_ids) >= min_expected_tokens, (
        f"Levanter generated too few tokens: {len(lev_generated_ids)} < {min_expected_tokens} "
        f"(HF generated {len(hf_generated_ids)})"
    )
    assert match_ratio >= 0.8, f"Token match ratio too low: {match_ratio:.1%}"
    print("✓ Generation with InferenceEngine test passed!")


@pytest.mark.slow
def test_llava_onevision_generation_with_inference_engine_multi():
    """Test LlavaOnevision generation with InferenceEngine using multiple images.

    This test verifies that Levanter generates the same output as HuggingFace
    for multi-image inputs, using base-only resolution (no anyres expansion).
    """
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision
    from levanter.trainer import TrainerConfig
    from test_image_utils import get_multi_images, prepare_test_data_single, create_lev_jax_tensors

    print("\n=== Test: Generation with InferenceEngine (Multi-Image) ===")

    # Use 0.5B model for testing (smaller to fit in TPU memory)
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    # Load multi-image data
    images = get_multi_images()  # Returns list of 2 PIL Images
    num_images = len(images)

    # Create prompt-only messages (no assistant response - this is for generation)
    text = "Compare these two images and describe the differences."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]

    print(f"Loaded {num_images} images for multi-image test")

    # Load HF model for comparison
    print(f"\nLoading HuggingFace model: {model_name}")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.eval()
        torch_model.model.image_newline = None  # Disable for consistency
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
        hf_config = torch_model.config
    except Exception as e:
        print(f"Could not load model: {e}")
        pytest.skip(f"Could not download model: {model_name}")
        return

    # Use prepare_test_data_single to process multi-image data
    test_pair = prepare_test_data_single(
        messages=messages,
        images=images,
        model_name=model_name,
        add_generation_prompt=True,
    )

    print(f"HF input_ids shape: {test_pair.hf.input_ids.shape}")
    print(f"HF pixel_values shape: {test_pair.hf.pixel_values.shape}")
    print(f"Lev input_ids shape: {test_pair.lev.input_ids.shape}")
    print(f"Lev pixel_values shape: {test_pair.lev.pixel_values.shape}")
    print(f"Lev grid_mask: {test_pair.lev.grid_mask.sum()} valid patches")

    # Verify multi-image preprocessing is correct
    assert test_pair.lev.unpad_indices is None, "Multi-image should have None unpad_indices"
    assert test_pair.lev.grid_mask.sum() == num_images, f"Multi-image should have {num_images} valid patches"

    # === HuggingFace Generation ===
    max_new_tokens = 200  # Allow full generation until EOS
    print(f"\n--- HuggingFace Generation (max_new_tokens={max_new_tokens}) ---")

    # Prepare HF inputs for multi-image (same as forward pass test)
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)

    # For multi-image: pixel_values is already 5D (num_images, patches, C, H, W)
    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values)
    if hf_pixel_values.dim() == 4:
        hf_pixel_values = hf_pixel_values.unsqueeze(0)

    # image_sizes: keep as (num_images, 2)
    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes)
    if hf_image_sizes.dim() == 1:
        hf_image_sizes = hf_image_sizes.unsqueeze(0)

    print(f"HF pixel_values shape: {hf_pixel_values.shape}")
    print(f"HF image_sizes shape: {hf_image_sizes.shape}")

    with torch.no_grad():
        hf_output_ids = torch_model.generate(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            attention_mask=hf_attention_mask,
            image_sizes=hf_image_sizes,
            batch_num_images=torch.tensor([num_images]),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
        )

    # Get only the generated tokens (excluding prompt)
    prompt_len = hf_input_ids.shape[1]
    hf_generated_ids = hf_output_ids[0, prompt_len:].cpu().numpy()
    hf_generated_text = torch_model.config._get_non_default_generation_parameters()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    hf_generated_text = processor.decode(hf_generated_ids, skip_special_tokens=True)
    print(f"HF generated tokens: {hf_generated_ids[:10]}...")
    print(f"HF generated text: {hf_generated_text[:200]}...")

    # === Levanter Generation ===
    print("\n--- Converting to Levanter ---")
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Enter mesh context for InferenceEngine and model loading
    trainer_config = TrainerConfig()

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        mesh = trainer_config.device_mesh
        compute_dtype = jnp.float32
        Vocab = Axis("vocab", hf_config.text_config.vocab_size)

        # Load Levanter model
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )

        # Create Levanter tensors using the data pipeline
        jax_tensors = create_lev_jax_tensors(test_pair.lev)

        print("\n--- Levanter Generation with InferenceEngine ---")
        print(f"pixel_values_lev axes: {jax_tensors.pixel_values.axes}")
        print(f"grid_mask axes: {jax_tensors.grid_mask.axes}")

        # Configure InferenceEngine
        # Use HF input_ids length since that's what we use as prompt
        prompt_len = len(test_pair.hf.input_ids)
        estimated_max_seq_len = prompt_len + max_new_tokens + 64
        page_size = 16
        engine_config = InferenceEngineConfig(
            max_seq_len=estimated_max_seq_len,
            page_size=page_size,
            max_seqs=1,
            max_rounds=32,
            max_stop_seqs=1,
            max_stop_tokens=4,
            max_pages=200,
            compute_dtype=compute_dtype,
        )

        # Build the LlavaInferenceEngine
        print("Creating LlavaInferenceEngine...")
        engine = LlavaInferenceEngine.from_model_with_config(
            model=lev_model,
            tokenizer=processor.tokenizer,
            config=engine_config,
            Vocab=Vocab,
            mesh=mesh,
        )

        # Use HF input_ids as prompt tokens (not padded Levanter ones)
        prompt_tokens = test_pair.hf.input_ids.tolist()
        print(f"Prompt tokens: {len(prompt_tokens)} tokens")

        # Create decoding parameters (greedy decoding with temperature=0)
        eos_token_id = processor.tokenizer.eos_token_id
        if eos_token_id is not None:
            stop_tokens = hax.named(jnp.array([[eos_token_id]], dtype=jnp.int32), ("stop_seq", "position"))
        else:
            stop_tokens = None

        decode_params = SeqDecodingParams(
            max_num_tokens=estimated_max_seq_len,
            temperature=0.0,  # Greedy decoding
            key=random.PRNGKey(42),
            stop_tokens=stop_tokens,
        )

        # Create VLMRequest with all data
        vlm_request = VLMRequest(
            prompt_tokens=prompt_tokens,
            request_id=0,
            decode_params=decode_params,
            n_generations=1,
            pixel_values=jax_tensors.pixel_values,
            input_ids=jax_tensors.input_ids,
            grid_mask=jax_tensors.grid_mask,
            unpad_indices=jax_tensors.unpad_indices,
        )

        # Generate using VLMRequest
        print("Starting generation with VLMRequest...")
        result = engine.generate([vlm_request])

    # Extract generated tokens
    lev_generated_ids = np.array(result.tokens[0])
    lev_generated_text = processor.decode(lev_generated_ids, skip_special_tokens=True)
    print(f"\nLev generated tokens: {lev_generated_ids[:10]}...")
    print(f"Lev generated text: {lev_generated_text[:200]}...")

    # === Compare HF and Levanter outputs ===
    print("\n--- Comparing HF and Levanter Generation ---")
    print(f"HF generated {len(hf_generated_ids)} tokens")
    print(f"Lev generated {len(lev_generated_ids)} tokens")

    # Compare token-by-token
    min_len = min(len(hf_generated_ids), len(lev_generated_ids))
    matching_tokens = sum(1 for i in range(min_len) if hf_generated_ids[i] == lev_generated_ids[i])
    match_rate = matching_tokens / min_len if min_len > 0 else 0

    print(f"First {min_len} tokens: {matching_tokens}/{min_len} match ({match_rate:.1%})")
    print(f"HF first 20 tokens:  {hf_generated_ids[:20]}")
    print(f"Lev first 20 tokens: {lev_generated_ids[:20]}")

    # Assert high match rate (greedy decoding should be deterministic)
    assert match_rate >= 0.9, f"Token match rate {match_rate:.1%} is too low, expected >= 90%"

    print("\n✓ Multi-image generation test passed!")
    print(f"  - Token match rate: {match_rate:.1%}")
    print(f"  - HF text: {hf_generated_text[:100]}...")
    print(f"  - Lev text: {lev_generated_text[:100]}...")


def test_get_image_features_vs_hf_real_single_image():
    """Compare raw image features with HF using a real single image.

    NOTE: Compares at the RAW feature level (vision tower + projector), BEFORE HF's pack_image_features().
    """
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision
    from transformers import LlavaOnevisionProcessor
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict
    import equinox as eqx

    print("\n=== Testing get_image_features vs HF with Real Single Image (raw features) ===")

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

    print(f"Loading HF model and processor: {model_name}")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        hf_config = torch_model.config
        processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}, error: {e}")
        return

    # Load a real image
    print("Loading real image...")
    image = get_single_image()
    print(f"  Loaded image: size={image.size}, mode={image.mode}")

    # Process image with HF processor
    print("Processing image with HF processor...")
    inputs = processor(text="Describe this image.", images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"]  # (1, num_patches, C, H, W)

    batch_size = pixel_values_torch.shape[0]
    num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    patch_height = pixel_values_torch.shape[3]
    patch_width = pixel_values_torch.shape[4]

    print(f"  Processed pixel_values shape: {pixel_values_torch.shape}")

    # Flatten to 4D for vision tower: (batch * num_patches, C, H, W)
    pixel_values_flat = pixel_values_torch.reshape(-1, channels, patch_height, patch_width)

    # Get HF raw features (vision tower + projector, WITHOUT pack_image_features)
    print("Running HF vision tower + projector (raw features)...")
    with torch.no_grad():
        hf_vision_outputs = torch_model.model.vision_tower(pixel_values_flat, output_hidden_states=True)

        vision_feature_layer = hf_config.vision_feature_layer
        if isinstance(vision_feature_layer, int):
            selected_hf_feature = hf_vision_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [hf_vision_outputs.hidden_states[idx] for idx in vision_feature_layer]
            selected_hf_feature = torch.cat(hs_pool, dim=-1)

        if hf_config.vision_feature_select_strategy == "default":
            selected_hf_feature = selected_hf_feature[:, 1:]

        hf_raw_features = torch_model.model.multi_modal_projector(selected_hf_feature)

    print(f"  HF raw features shape: {hf_raw_features.shape}")

    # Convert to Levanter
    print("Converting to Levanter...")
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

    # Use single-device mesh to avoid sharding issues
    mesh_config = MeshConfig(
        axes={"model": 8, "data": 1, "replica": 1},
        compute_mapping={
            "vision_batch": ("model",),  # Shard vision patches across model axis
            "vocab": "model",  # Shard vocab dimension to reduce logits memory
            "batch": ("replica_dcn", "replica"),  # Map batch without data to avoid conflict with mlp/heads on data
        },
        shared_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding and conflict with vision_batch
            "mlp": "data",  # Map mlp to data (size 1) to avoid conflict with vision_batch on model axis
        },
        param_mapping={
            "heads": "data",  # Map heads to data (size 1) to avoid sharding since 14 is not divisible by 8
        },
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        state_dict = converter.load_state_dict(model_name)
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    lev_model = jtu.tree_map(_to_float32, lev_model)

    # Create 5D input for Levanter (no padding needed - use exact patches)
    pv_np = pixel_values_torch.numpy().astype(np.float32)
    grid_mask_np = np.ones((batch_size, num_patches), dtype=bool)

    Batch = Axis("batch", batch_size)
    NumPatches = Axis("num_patches", num_patches)
    Channels = Axis("channels", channels)
    Height = Axis("height", patch_height)
    Width = Axis("width", patch_width)

    pixel_values_lev = hax.named(jnp.array(pv_np, dtype=jnp.float32), (Batch, NumPatches, Channels, Height, Width))
    grid_mask = hax.named(jnp.array(grid_mask_np), (Batch, NumPatches))

    print("Running Levanter get_image_features...")

    @hax.named_jit
    def compute_lev_single(model, pixel_values, grid_mask):
        return model.get_image_features(pixel_values=pixel_values, grid_mask=grid_mask, key=None)

    lev_result = compute_lev_single(lev_model, pixel_values_lev, grid_mask)
    lev_image_features = lev_result[0] if isinstance(lev_result, tuple) else lev_result

    # Compare results
    print("Comparing results...")
    hf_array = hf_raw_features.detach().numpy()
    lev_array = np.array(lev_image_features.array)

    # HF: (batch * num_patches, features_per_patch, embed)
    # Lev: (batch, num_patches, features_per_patch, embed)
    hf_array_reshaped = hf_array.reshape(batch_size, num_patches, -1, hf_array.shape[-1])

    print(f"  HF reshaped: {hf_array_reshaped.shape}")
    print(f"  Lev shape: {lev_array.shape}")

    assert (
        hf_array_reshaped.shape == lev_array.shape
    ), f"Shape mismatch: HF={hf_array_reshaped.shape}, Lev={lev_array.shape}"

    max_diff = np.max(np.abs(hf_array_reshaped - lev_array))
    mean_diff = np.mean(np.abs(hf_array_reshaped - lev_array))
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    assert mean_diff < 1e-3, f"Values don't match: mean diff = {mean_diff}, max diff = {max_diff}"

    print("✓ Raw image features match for real single image!")


def test_get_image_features_vs_hf_real_multi_image():
    """Compare raw image features with HF using real multiple images.

    NOTE: Compares at the RAW feature level (vision tower + projector), BEFORE HF's pack_image_features().
    """
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision
    from transformers import LlavaOnevisionProcessor

    print("\n=== Testing get_image_features vs HF with Real Multiple Images (raw features) ===")

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print(f"Loading HF model and processor: {model_name}")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        hf_config = torch_model.config
        processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}, error: {e}")
        return

    # Load a real image and create multiple copies
    print("Loading real image and creating multiple copies...")
    image = get_single_image()
    print(f"  Loaded image: size={image.size}, mode={image.mode}")
    images = [image, image, image]
    print(f"  Created {len(images)} image copies for multi-image test")

    # Process images with HF processor
    print("Processing images with HF processor...")
    inputs = processor(text="Describe these images.", images=images, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"]  # (batch, num_patches, C, H, W)

    batch_size = pixel_values_torch.shape[0]
    num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    patch_height = pixel_values_torch.shape[3]
    patch_width = pixel_values_torch.shape[4]

    print(f"  Processed pixel_values shape: {pixel_values_torch.shape}")

    # Flatten to 4D for vision tower: (batch * num_patches, C, H, W)
    pixel_values_flat = pixel_values_torch.reshape(-1, channels, patch_height, patch_width)

    # Get HF raw features (vision tower + projector, WITHOUT pack_image_features)
    print("Running HF vision tower + projector (raw features)...")
    with torch.no_grad():
        hf_vision_outputs = torch_model.model.vision_tower(pixel_values_flat, output_hidden_states=True)

        vision_feature_layer = hf_config.vision_feature_layer
        if isinstance(vision_feature_layer, int):
            selected_hf_feature = hf_vision_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [hf_vision_outputs.hidden_states[idx] for idx in vision_feature_layer]
            selected_hf_feature = torch.cat(hs_pool, dim=-1)

        if hf_config.vision_feature_select_strategy == "default":
            selected_hf_feature = selected_hf_feature[:, 1:]

        hf_raw_features = torch_model.model.multi_modal_projector(selected_hf_feature)

    print(f"  HF raw features shape: {hf_raw_features.shape}")

    # Convert to Levanter
    print("Converting to Levanter...")
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig

    # Use proper multi-device mesh with vision_batch sharding to avoid OOM
    # Pad batch to 8 for proper TPU sharding (divisible by data axis size=8)
    mesh_config = MeshConfig(
        compute_mapping={
            "vision_batch": DEFAULT_DP_AXES,  # Shard vision_batch like batch
        }
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=config,
            axis_mapping=parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        # Pad batch to 8 for proper TPU sharding
        original_batch_size = batch_size
        target_batch_size = 8
        print(f"  Padding batch from {original_batch_size} to {target_batch_size} for TPU sharding")

        # Create 5D input for Levanter with batch padding
        pv_np = pixel_values_torch.numpy().astype(np.float32)
        # Tile to reach target batch size
        pv_padded = np.tile(pv_np, (target_batch_size // original_batch_size + 1, 1, 1, 1, 1))[:target_batch_size]
        grid_mask_np = np.ones((target_batch_size, num_patches), dtype=bool)

        Batch = Axis("batch", target_batch_size)
        NumPatches = Axis("num_patches", num_patches)
        Channels = Axis("channels", channels)
        Height = Axis("height", patch_height)
        Width = Axis("width", patch_width)

        pixel_values_lev = hax.named(
            jnp.array(pv_padded, dtype=jnp.float32), (Batch, NumPatches, Channels, Height, Width)
        )
        grid_mask = hax.named(jnp.array(grid_mask_np), (Batch, NumPatches))

        print("Running Levanter get_image_features...")

        @hax.named_jit
        def compute_lev_multi(model, pixel_values, grid_mask):
            return model.get_image_features(pixel_values=pixel_values, grid_mask=grid_mask, key=None)

        lev_result = compute_lev_multi(lev_model, pixel_values_lev, grid_mask)
        lev_image_features = lev_result[0] if isinstance(lev_result, tuple) else lev_result

        # Compare results (only first original_batch_size samples)
        print("Comparing results...")
        hf_array = hf_raw_features.detach().numpy()
        lev_array = np.array(lev_image_features.array)[:original_batch_size]  # Only compare original samples

        # HF: (batch * num_patches, features_per_patch, embed)
        # Lev: (batch, num_patches, features_per_patch, embed)
        hf_array_reshaped = hf_array.reshape(original_batch_size, num_patches, -1, hf_array.shape[-1])

        print(f"  HF reshaped: {hf_array_reshaped.shape}")
        print(f"  Lev shape: {lev_array.shape}")

        assert (
            hf_array_reshaped.shape == lev_array.shape
        ), f"Shape mismatch: HF={hf_array_reshaped.shape}, Lev={lev_array.shape}"

        max_diff = np.max(np.abs(hf_array_reshaped - lev_array))
        mean_diff = np.mean(np.abs(hf_array_reshaped - lev_array))
        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

        assert mean_diff < 1e-3, f"Values don't match: mean diff = {mean_diff}, max diff = {max_diff}"

        print("✓ Raw image features match for real multiple images!")


def test_get_placeholder_mask_vs_hf():
    """Compare get_placeholder_mask with HuggingFace implementation."""
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision

    print("\n=== Testing get_placeholder_mask vs HF ===")

    # Use a small pretrained model
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    print(f"Loading HF model: {model_name}")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        hf_config = torch_model.config
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}, error: {e}")
        return

    # Create dummy inputs
    batch_size = 2
    seq_len = 20
    embed_dim = hf_config.text_config.hidden_size

    # Create input_ids with image tokens
    # Put image tokens at specific positions
    input_ids_torch = torch.randint(100, 1000, (batch_size, seq_len), dtype=torch.long)
    input_ids_torch[0, 5] = hf_config.image_token_index  # Image token in first batch
    input_ids_torch[0, 10] = hf_config.image_token_index  # Another image token
    input_ids_torch[1, 3] = hf_config.image_token_index  # Image token in second batch

    # Create inputs_embeds (random for testing)
    inputs_embeds_torch = torch.randn(batch_size, seq_len, embed_dim)

    # Create dummy image_features (3 image tokens total)
    num_image_tokens = 3
    image_features_torch = torch.randn(num_image_tokens, embed_dim)

    # HF get_placeholder_mask (use model.model)
    print("Running HF get_placeholder_mask...")
    with torch.no_grad():
        hf_image_mask, hf_video_mask = torch_model.model.get_placeholder_mask(
            input_ids=input_ids_torch, inputs_embeds=inputs_embeds_torch, image_features=image_features_torch
        )

    # Convert to Levanter
    print("Converting to Levanter...")
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # For this test, we just need to test the get_placeholder_mask logic
    # which doesn't depend on model weights, so we can initialize a fresh model
    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    lev_model = LlavaOnevisionModel.init(Vocab, config, key=random.PRNGKey(0))

    # Convert to Levanter format
    Batch = Axis("batch", batch_size)
    SeqLen = Axis("position", seq_len)
    Embed = Axis("embed", embed_dim)
    NumImageTokens = Axis("total_patches", num_image_tokens)

    input_ids_lev = hax.named(jnp.array(input_ids_torch.numpy(), dtype=jnp.int32), (Batch, SeqLen))
    image_features_lev = hax.named(jnp.array(image_features_torch.numpy()), (NumImageTokens, Embed))

    # Run Levanter get_placeholder_mask (now only takes input_ids and image_features)
    print("Running Levanter get_placeholder_mask...")
    lev_image_mask = lev_model.get_placeholder_mask(input_ids=input_ids_lev, image_features=image_features_lev)

    # Compare results
    print("Comparing results...")

    # HF returns (batch, seq_len, embed), but positions are the same across embed
    # Lev now returns (batch, seq_len) boolean mask
    hf_mask_array = hf_image_mask.detach().numpy()[:, :, 0]  # Take first embed slice
    lev_mask_array = np.array(lev_image_mask.array)

    print(f"  Shape: HF={hf_mask_array.shape}, Lev={lev_mask_array.shape}")
    assert (
        hf_mask_array.shape == lev_mask_array.shape
    ), f"Shape mismatch: HF={hf_mask_array.shape}, Lev={lev_mask_array.shape}"

    # Compare boolean values
    matches = np.all(hf_mask_array == lev_mask_array)
    print(f"  {'✓ PASS' if matches else '✗ FAIL'}: Masks match exactly")

    # Print some sample values
    print(f"  Sample HF mask values at [0, 5]: {hf_mask_array[0, 5]}")
    print(f"  Sample Lev mask values at [0, 5]: {lev_mask_array[0, 5]}")

    assert matches, "get_placeholder_mask test failed: masks don't match"
    print("✓ All get_placeholder_mask comparisons passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
