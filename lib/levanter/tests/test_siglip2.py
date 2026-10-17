# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


import tempfile

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import PretrainedConfig
from haliax.partitioning import ResourceAxis
from jax.sharding import Mesh

from levanter.layers.attention import AttentionMask
from levanter.models.siglip2 import Siglip2VisionConfig, Siglip2VisionModel
from levanter.utils.activation import ActivationFunctionEnum
from PIL import Image
from levanter.testing.helpers import use_test_mesh

from levanter.testing.helpers import skip_if_no_torch


def _hf_siglip2_vision_config() -> PretrainedConfig:
    """Return a tiny Siglip2VisionConfig for testing."""
    from transformers import (  # noqa: PLC0415  # optional reference
        Siglip2VisionConfig as HfSiglip2VisionConfig,
    )  # noqa: PLC0415  # optional reference

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
def test_siglip2_vision_from_hf_config() -> None:
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_siglip2_vision_config()
    config = Siglip2VisionConfig.from_hf_config(hf_config)

    assert config.hidden_size == hf_config.hidden_size
    assert config.intermediate_size == hf_config.intermediate_size
    assert config.num_hidden_layers == hf_config.num_hidden_layers
    assert config.num_attention_heads == hf_config.num_attention_heads
    assert config.hidden_act == ActivationFunctionEnum.gelu_new


@skip_if_no_torch
def test_siglip2_vision_to_hf_config() -> None:
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
def test_siglip2_vision_config_roundtrip() -> None:
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
def test_siglip2_vision_activation_function_mapping() -> None:
    """Test that various activation functions are correctly mapped."""
    from transformers import (  # noqa: PLC0415  # optional reference
        Siglip2VisionConfig as HfSiglip2VisionConfig,
    )  # noqa: PLC0415  # optional reference

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
def test_siglip2_vision_roundtrip() -> None:
    """Test loading HuggingFace weights into Levanter and roundtrip conversion."""
    import torch  # noqa: PLC0415  # optional reference
    from transformers import (  # noqa: PLC0415  # optional reference
        Siglip2VisionModel as HfSiglip2VisionModel,
    )  # noqa: PLC0415  # optional reference

    jax.config.update("jax_default_matmul_precision", "float32")

    hf_config = _hf_siglip2_vision_config()
    torch.random.manual_seed(0)
    torch_model = HfSiglip2VisionModel(hf_config)
    torch_model.eval()

    batch_size = 2
    num_patches = 64
    patch_input_dim = hf_config.num_channels * hf_config.patch_size * hf_config.patch_size
    pixel_values_torch = torch.randn(batch_size, num_patches, patch_input_dim, dtype=torch.float32)

    spatial_shapes = torch.tensor([[8, 8]] * batch_size)
    attention_mask = torch.ones(batch_size, num_patches, dtype=torch.long)
    with torch.no_grad():
        torch_output = torch_model(
            pixel_values_torch, pixel_attention_mask=attention_mask, spatial_shapes=spatial_shapes
        ).last_hidden_state.numpy()

    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        config = Siglip2VisionConfig.from_hf_config(torch_model.config)
        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")

        model = converter.load_pretrained(Siglip2VisionModel)

        Batch = hax.Axis("batch", batch_size)
        NumPatches = hax.Axis("num_patches", num_patches)
        PatchInput = hax.Axis("patch_input", patch_input_dim)

        pixel_values = hax.named(
            jnp.array(pixel_values_torch.numpy(), dtype=jnp.float32),
            (Batch, NumPatches, PatchInput),
        )

        jax_output_array = eqx.filter_jit(
            lambda pixels, shapes: model(pixels, spatial_shapes=shapes).last_hidden_state.array
        )(pixel_values, jnp.asarray(spatial_shapes.numpy()))

        np.testing.assert_allclose(torch_output, jax_output_array, rtol=1e-4, atol=1e-4)

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = HfSiglip2VisionModel.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        with torch.no_grad():
            torch_output2 = torch_model2(
                pixel_values_torch, pixel_attention_mask=attention_mask, spatial_shapes=spatial_shapes
            ).last_hidden_state.numpy()
        np.testing.assert_allclose(torch_output2, jax_output_array, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.timeout(600)
@skip_if_no_torch
def test_siglip2_vision_pretrained_random_pixels() -> None:
    """Test Siglip2 vision model with generated pixels using the HF processor."""
    import torch  # noqa: PLC0415  # optional reference
    from transformers import (  # noqa: PLC0415  # optional reference
        Siglip2VisionModel as HfSiglip2VisionModel,
        AutoProcessor,
    )  # noqa: PLC0415  # optional reference

    jax.config.update("jax_default_matmul_precision", "float32")

    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    model_name = "google/siglip2-so400m-patch16-naflex"

    processor = AutoProcessor.from_pretrained(model_name)
    torch_model = HfSiglip2VisionModel.from_pretrained(model_name, dtype=torch.float32)
    torch_model.eval()
    torch_model = torch_model.float()

    inputs = processor(images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"].float()
    batch_size = pixel_values_torch.shape[0]
    num_patches = pixel_values_torch.shape[1]

    spatial_shapes = inputs["spatial_shapes"]
    attention_mask = inputs["pixel_attention_mask"]
    with torch.no_grad():
        vision_outputs = torch_model(
            pixel_values_torch, pixel_attention_mask=attention_mask, spatial_shapes=spatial_shapes
        )
        torch_output = vision_outputs.last_hidden_state.detach().cpu().numpy()

    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        config = Siglip2VisionConfig.from_hf_config(torch_model.config)
        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")

        model = converter.load_pretrained(Siglip2VisionModel)

        pixel_values_np = pixel_values_torch.cpu().numpy().astype(np.float32)
        Batch = hax.Axis("batch", batch_size)
        NumPatches = hax.Axis("num_patches", num_patches)
        PatchInput = hax.Axis("patch_input", pixel_values_np.shape[2])

        pixel_values = hax.named(jnp.array(pixel_values_np, dtype=jnp.float32), (Batch, NumPatches, PatchInput))
        spatial_shapes_np = spatial_shapes.cpu().numpy()

        mask = AttentionMask.explicit(
            hax.named(jnp.asarray(attention_mask.numpy(), dtype=bool), (Batch, hax.Axis("key_position", num_patches)))
        )
        jax_output = model(pixel_values, mask=mask, spatial_shapes=spatial_shapes_np)
        jax_output_array = jax_output.last_hidden_state.array

    assert torch_output.shape == jax_output_array.shape
    assert not np.any(np.isnan(jax_output_array))
    assert not np.any(np.isinf(jax_output_array))

    np.testing.assert_allclose(torch_output, jax_output_array, rtol=1e-4, atol=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/converted_model"
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        converter.save_pretrained(model, save_path, save_tokenizer=False)

        converted_hf_model = HfSiglip2VisionModel.from_pretrained(save_path)
        converted_hf_model.eval()
        converted_hf_model = converted_hf_model.float()

        with torch.no_grad():
            converted_outputs = converted_hf_model(
                pixel_values_torch, pixel_attention_mask=attention_mask, spatial_shapes=spatial_shapes
            )
            converted_output_np = converted_outputs.last_hidden_state.detach().cpu().numpy()

        assert not np.any(np.isnan(converted_output_np))
        np.testing.assert_allclose(jax_output_array, converted_output_np, rtol=1e-4, atol=1e-4)
