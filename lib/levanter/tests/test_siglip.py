# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.partitioning import ResourceAxis
from haliax.state_dict import from_torch_compatible_state_dict
from jax.sharding import Mesh

from levanter.models.siglip import SiglipVisionConfig, SiglipVisionModel
from levanter.utils.activation import ActivationFunctionEnum
from test_image_utils import get_single_image
from test_utils import use_test_mesh

from test_utils import skip_if_no_torch


def _hf_siglip_vision_config():
    """Return a tiny SiglipVisionConfig for testing."""
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig

    return HfSiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )


@skip_if_no_torch
def test_siglip_vision_from_hf_config():
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_siglip_vision_config()
    config = SiglipVisionConfig.from_hf_config(hf_config)

    assert config.hidden_size == hf_config.hidden_size
    assert config.intermediate_size == hf_config.intermediate_size
    assert config.num_hidden_layers == hf_config.num_hidden_layers
    assert config.num_attention_heads == hf_config.num_attention_heads
    assert config.hidden_act == ActivationFunctionEnum.gelu_new


@skip_if_no_torch
def test_siglip_vision_to_hf_config():
    """Test conversion from Levanter config to HuggingFace config."""
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_act=ActivationFunctionEnum.gelu_new,
    )

    hf_config = config.to_hf_config()

    assert hf_config.hidden_size == config.hidden_size
    assert hf_config.intermediate_size == config.intermediate_size
    assert hf_config.num_hidden_layers == config.num_hidden_layers
    assert hf_config.hidden_act == "gelu_pytorch_tanh"


@skip_if_no_torch
def test_siglip_vision_config_roundtrip():
    """Test that converting HF -> Levanter -> HF preserves the config."""
    hf_config_1 = _hf_siglip_vision_config()
    levanter_config = SiglipVisionConfig.from_hf_config(hf_config_1)
    hf_config_2 = levanter_config.to_hf_config()

    assert hf_config_2 == hf_config_1


@skip_if_no_torch
def test_siglip_vision_activation_function_conversion():
    """Test various activation function conversions between HF and Levanter."""
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig

    test_cases = [
        ("gelu_pytorch_tanh", ActivationFunctionEnum.gelu_new),
        ("gelu", ActivationFunctionEnum.gelu),
        ("quick_gelu", ActivationFunctionEnum.quick_gelu),
    ]

    for hf_act, expected_lev_act in test_cases:
        hf_config = HfSiglipVisionConfig(hidden_act=hf_act)
        levanter_config = SiglipVisionConfig.from_hf_config(hf_config)
        assert levanter_config.hidden_act == expected_lev_act


@skip_if_no_torch
def test_siglip_vision_embeddings_vs_hf():
    """Compare SiglipVisionModel output with HuggingFace using a small model."""
    import torch
    from transformers import SiglipVisionConfig as HfSiglipVisionConfig
    from transformers import SiglipVisionModel as HfSiglipVisionModel

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

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

    batch_size = 2
    pixel_values_torch = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        hf_output = hf_model(pixel_values_torch, output_hidden_states=True)
        hf_last_hidden_np = hf_output.last_hidden_state.detach().cpu().numpy()

    lev_config = SiglipVisionConfig.from_hf_config(hf_config)
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        hf_model.save_pretrained(f"{tmpdir}/hf_model")

        Vocab = hax.Axis("vocab", 1)
        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, Vocab, lev_config, key=jax.random.PRNGKey(0))

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", 224)
        Width = hax.Axis("width", 224)

        pixel_values_jax = hax.named(
            jnp.array(pixel_values_torch.numpy(), dtype=jnp.float32), (Batch, Channels, Height, Width)
        )

        lev_output = lev_model(pixel_values_jax, output_hidden_states=True, key=jax.random.PRNGKey(1))

    lev_last_hidden_np = np.array(lev_output.last_hidden_state.array)

    # 4-layer model: use 1e-3 tolerance
    assert np.allclose(hf_last_hidden_np, lev_last_hidden_np, rtol=1e-3, atol=1e-3)


@skip_if_no_torch
def test_siglip_vision_real_image():
    """Test SigLIP vision model with real image using HF processor."""
    import torch
    from transformers import SiglipImageProcessor
    from transformers import SiglipVisionModel as HfSiglipVisionModel

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    image = get_single_image()
    model_name = "google/siglip-base-patch16-224"

    try:
        processor = SiglipImageProcessor.from_pretrained(model_name)
        torch_model = HfSiglipVisionModel.from_pretrained(model_name)
        torch_model.eval()
        torch_model = torch_model.float()
    except Exception as e:
        pytest.skip(f"Failed to load HF model/processor: {e}")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"].float()

    with torch.no_grad():
        vision_outputs = torch_model(pixel_values_torch, output_hidden_states=True)
        torch_output = vision_outputs.last_hidden_state.detach().cpu().numpy()

    lev_config = SiglipVisionConfig.from_hf_config(torch_model.config)
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/hf_model")

        Vocab = hax.Axis("vocab", 1)
        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, Vocab, lev_config, key=jax.random.PRNGKey(0))

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        pixel_values_np = pixel_values_torch.cpu().numpy()
        batch_size, num_channels, height, width = pixel_values_np.shape

        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", num_channels)
        Height = hax.Axis("height", height)
        Width = hax.Axis("width", width)

        pixel_values_jax = hax.named(jnp.array(pixel_values_np, dtype=jnp.float32), (Batch, Channels, Height, Width))

        lev_output = lev_model(pixel_values_jax, output_hidden_states=True, key=jax.random.PRNGKey(1))

    lev_output_np = np.array(lev_output.last_hidden_state.array)

    assert torch_output.shape == lev_output_np.shape
    assert not np.any(np.isnan(lev_output_np))
    assert not np.any(np.isinf(lev_output_np))

    # 12-layer full model: numerical differences compound, use looser tolerance
    assert np.allclose(torch_output, lev_output_np, rtol=1e-3, atol=1e-2)

    # Test Levanter -> HF conversion
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/converted_model"

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        converter.save_pretrained(lev_model, save_path, save_tokenizer=False)

        converted_hf_model = HfSiglipVisionModel.from_pretrained(save_path)
        converted_hf_model.eval()
        converted_hf_model = converted_hf_model.float()

        with torch.no_grad():
            converted_outputs = converted_hf_model(pixel_values_torch)
            converted_output_np = converted_outputs.last_hidden_state.detach().cpu().numpy()

        assert not np.any(np.isnan(converted_output_np))
        assert np.allclose(lev_output_np, converted_output_np, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
