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
from jax import random
from jax.sharding import Mesh

from levanter.models.siglip2 import Siglip2VisionConfig, Siglip2VisionModel
from levanter.utils.activation import ActivationFunctionEnum
from test_image_utils import get_single_image
from test_utils import use_test_mesh

from test_utils import skip_if_no_torch


def _hf_siglip2_vision_config():
    """Return a tiny Siglip2VisionConfig for testing."""
    from transformers import Siglip2VisionConfig as HfSiglip2VisionConfig

    return HfSiglip2VisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )


@skip_if_no_torch
def test_siglip2_vision_from_hf_config():
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_siglip2_vision_config()
    config = Siglip2VisionConfig.from_hf_config(hf_config)

    assert config.hidden_size == hf_config.hidden_size
    assert config.intermediate_size == hf_config.intermediate_size
    assert config.num_hidden_layers == hf_config.num_hidden_layers
    assert config.num_attention_heads == hf_config.num_attention_heads
    assert config.hidden_act == ActivationFunctionEnum.gelu_new


@skip_if_no_torch
def test_siglip2_vision_to_hf_config():
    """Test conversion from Levanter config to HuggingFace config."""
    config = Siglip2VisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act=ActivationFunctionEnum.gelu_new,
    )

    hf_config = config.to_hf_config()

    assert hf_config.hidden_size == config.hidden_size
    assert hf_config.intermediate_size == config.intermediate_size
    assert hf_config.num_hidden_layers == config.num_hidden_layers
    assert hf_config.hidden_act == "gelu_pytorch_tanh"


@skip_if_no_torch
def test_siglip2_vision_config_roundtrip():
    """Test that converting HF -> Levanter -> HF preserves the config."""
    hf_config_1 = _hf_siglip2_vision_config()
    levanter_config = Siglip2VisionConfig.from_hf_config(hf_config_1)
    hf_config_2 = levanter_config.to_hf_config()

    assert hf_config_2.hidden_size == hf_config_1.hidden_size
    assert hf_config_2.intermediate_size == hf_config_1.intermediate_size
    assert hf_config_2.num_hidden_layers == hf_config_1.num_hidden_layers
    assert hf_config_2.num_attention_heads == hf_config_1.num_attention_heads
    assert hf_config_2.hidden_act == hf_config_1.hidden_act


@skip_if_no_torch
def test_siglip2_vision_activation_function_mapping():
    """Test that various activation functions are correctly mapped."""
    from transformers import Siglip2VisionConfig as HfSiglip2VisionConfig

    test_cases = [
        ("gelu_pytorch_tanh", ActivationFunctionEnum.gelu_new),
        ("gelu", ActivationFunctionEnum.gelu),
        ("quick_gelu", ActivationFunctionEnum.quick_gelu),
    ]

    for hf_act, expected_lev_act in test_cases:
        hf_config = HfSiglip2VisionConfig(hidden_act=hf_act)
        levanter_config = Siglip2VisionConfig.from_hf_config(hf_config)
        assert levanter_config.hidden_act == expected_lev_act


@skip_if_no_torch
def test_siglip2_vision_roundtrip():
    """Test loading HuggingFace weights into Levanter and roundtrip conversion."""
    import torch
    from transformers import Siglip2VisionModel as HfSiglip2VisionModel

    jax.config.update("jax_default_matmul_precision", "float32")

    hf_config = _hf_siglip2_vision_config()
    torch.random.manual_seed(0)
    torch_model = HfSiglip2VisionModel(hf_config)
    torch_model.eval()

    batch_size = 2
    num_patches = 64
    patch_input_dim = hf_config.num_channels * hf_config.patch_size * hf_config.patch_size
    pixel_values_torch = torch.randn(batch_size, num_patches, patch_input_dim, dtype=torch.float32)

    # Run HF model through encoder
    with torch.no_grad():
        hf_vision = torch_model.vision_model
        patch_embeds = hf_vision.embeddings.patch_embedding(pixel_values_torch)
        position_ids = torch.arange(num_patches)
        pos_embeds = hf_vision.embeddings.position_embedding(position_ids)
        hidden_states = patch_embeds + pos_embeds

        attention_mask = torch.ones(batch_size, 1, num_patches, num_patches)
        encoder_output = hf_vision.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = encoder_output.last_hidden_state
        torch_output = hf_vision.post_layernorm(hidden_states).detach().cpu().numpy()

    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        config = Siglip2VisionConfig.from_hf_config(hf_config)
        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")

        Vocab = hax.Axis("vocab", 1)
        model_template = eqx.filter_eval_shape(Siglip2VisionModel.init, Vocab, config, key=random.PRNGKey(0))
        state_dict = converter.load_state_dict(f"{tmpdir}/torch_model")
        model = from_torch_compatible_state_dict(model_template, state_dict)

        Batch = hax.Axis("batch", batch_size)
        NumPatches = hax.Axis("num_patches", num_patches)
        PatchInput = hax.Axis("patch_input", patch_input_dim)

        pixel_values = hax.named(
            jnp.array(pixel_values_torch.numpy(), dtype=jnp.float32),
            (Batch, NumPatches, PatchInput),
        )

        jax_output = model(pixel_values, key=None)
        jax_output_array = jax_output.last_hidden_state.array

        # Multi-layer model: use 1e-2 tolerance
        assert torch_output.shape == jax_output_array.shape
        assert np.allclose(torch_output, jax_output_array, rtol=1e-3, atol=1e-2)

        # Test roundtrip: save Levanter model and load back as HF
        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = HfSiglip2VisionModel.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        with torch.no_grad():
            hf_vision2 = torch_model2.vision_model
            patch_embeds = hf_vision2.embeddings.patch_embedding(pixel_values_torch)
            pos_embeds = hf_vision2.embeddings.position_embedding(position_ids)
            hidden_states = patch_embeds + pos_embeds
            encoder_output = hf_vision2.encoder(hidden_states, attention_mask=attention_mask)
            hidden_states = encoder_output.last_hidden_state
            torch_output2 = hf_vision2.post_layernorm(hidden_states).detach().cpu().numpy()

        assert np.allclose(torch_output2, jax_output_array, rtol=1e-3, atol=1e-2)


@skip_if_no_torch
def test_siglip2_vision_real_image():
    """Test Siglip2 vision model with real image using HF processor."""
    import torch
    from transformers import AutoModel, AutoProcessor

    jax.config.update("jax_default_matmul_precision", "float32")

    image = get_single_image()
    model_name = "google/siglip2-so400m-patch16-naflex"

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        torch_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
        torch_model.eval()
        torch_model = torch_model.float()
    except Exception as e:
        pytest.skip(f"Failed to load HF model/processor: {e}")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"].float()
    batch_size = pixel_values_torch.shape[0]
    num_patches = pixel_values_torch.shape[1]

    if "spatial_shapes" in inputs:
        spatial_shapes = inputs["spatial_shapes"]
    else:
        grid_size = int(num_patches**0.5)
        spatial_shapes = torch.tensor([[grid_size, grid_size]] * batch_size, dtype=torch.long)

    # Run HF model
    with torch.no_grad():
        if hasattr(torch_model, "vision_model"):
            hf_vision = torch_model.vision_model
            hf_config = torch_model.config.vision_config
        else:
            hf_vision = torch_model
            hf_config = torch_model.config

        attention_mask = torch.ones(batch_size, num_patches, dtype=torch.long)
        vision_outputs = hf_vision(pixel_values_torch, attention_mask=attention_mask, spatial_shapes=spatial_shapes)
        torch_output = vision_outputs.last_hidden_state.detach().cpu().numpy()

    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        config = Siglip2VisionConfig.from_hf_config(hf_config)
        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")

        Vocab = hax.Axis("vocab", 1)
        model_template = eqx.filter_eval_shape(Siglip2VisionModel.init, Vocab, config, key=random.PRNGKey(0))
        state_dict = converter.load_state_dict(f"{tmpdir}/torch_model")
        model = from_torch_compatible_state_dict(model_template, state_dict)

        pixel_values_np = pixel_values_torch.cpu().numpy().astype(np.float32)
        Batch = hax.Axis("batch", batch_size)
        NumPatches = hax.Axis("num_patches", num_patches)
        PatchInput = hax.Axis("patch_input", pixel_values_np.shape[2])

        pixel_values = hax.named(jnp.array(pixel_values_np, dtype=jnp.float32), (Batch, NumPatches, PatchInput))
        spatial_shapes_np = spatial_shapes.cpu().numpy()

        jax_output = model(pixel_values, spatial_shapes=spatial_shapes_np)
        jax_output_array = jax_output.last_hidden_state.array

    assert torch_output.shape == jax_output_array.shape
    assert not np.any(np.isnan(jax_output_array))
    assert not np.any(np.isinf(jax_output_array))

    # Multi-layer full model: use percentile-based check
    diff = np.abs(torch_output - jax_output_array)
    p99_diff = np.percentile(diff, 99)
    assert p99_diff < 0.1, f"P99 diff too large: {p99_diff}"

    # Test Levanter -> HF conversion
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/converted_model"
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        converter.save_pretrained(model, save_path, save_tokenizer=False)

        converted_hf_model = AutoModel.from_pretrained(save_path, trust_remote_code=True)
        converted_hf_model.eval()
        converted_hf_model = converted_hf_model.float()

        with torch.no_grad():
            if hasattr(converted_hf_model, "vision_model"):
                converted_vision = converted_hf_model.vision_model
            else:
                converted_vision = converted_hf_model

            converted_outputs = converted_vision(
                pixel_values_torch, attention_mask=attention_mask, spatial_shapes=spatial_shapes
            )
            converted_output_np = converted_outputs.last_hidden_state.detach().cpu().numpy()

        assert not np.any(np.isnan(converted_output_np))
        diff_lev_hf = np.abs(jax_output_array - converted_output_np)
        p99_diff_lev_hf = np.percentile(diff_lev_hf, 99)
        assert p99_diff_lev_hf < 0.1, f"Levanter -> HF p99 diff too large: {p99_diff_lev_hf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
