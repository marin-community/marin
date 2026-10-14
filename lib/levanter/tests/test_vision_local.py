# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict
from levanter.data.dataset import ListAsyncDataset
from levanter.data.image import RepeatedImageDataset
from levanter.models.siglip import SiglipVisionConfig, SiglipVisionModel
from levanter.models.vision_transformer import VisionAttention
from levanter.models.siglip2 import Siglip2VisionConfig, Siglip2VisionModel, Siglip2VisionEmbeddings
from levanter.layers.attention import AttentionMask
from levanter.testing.helpers import skip_if_no_torch, use_test_mesh


@pytest.mark.parametrize("encoder", ["siglip", "siglip2"])
def test_vision_checkpoint_roundtrip_preserves_jitted_predictions(encoder: str) -> None:
    parameters = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=16)
    batch = hax.Axis("batch", 1)
    if encoder == "siglip":
        config = SiglipVisionConfig(**parameters, image_size=32)
        model_class = SiglipVisionModel
        axes = (batch, hax.Axis("channels", 3), hax.Axis("height", 32), hax.Axis("width", 32))
    else:
        config = Siglip2VisionConfig(**parameters, num_patches=4)
        model_class = Siglip2VisionModel
        axes = (batch, hax.Axis("num_patches", 4), hax.Axis("patch_input", 768))

    model = model_class.init(hax.Axis("vocab", 1), config, key=jax.random.PRNGKey(0))
    other = model_class.init(hax.Axis("vocab", 1), config, key=jax.random.PRNGKey(1))
    pixels = hax.random.normal(jax.random.PRNGKey(2), axes)
    forward = eqx.filter_jit(lambda network: network(pixels).last_hidden_state.array)
    expected = forward(model)
    assert not np.allclose(expected, forward(other))
    restored = from_torch_compatible_state_dict(other, to_torch_compatible_state_dict(model))
    np.testing.assert_allclose(forward(restored), expected, rtol=1e-5, atol=1e-5)

    def objective(network: SiglipVisionModel | Siglip2VisionModel) -> jax.Array:
        features = network(pixels).last_hidden_state.array
        return jnp.sum(features * jnp.arange(features.shape[-1]))

    gradients = eqx.filter_grad(objective)(restored)
    leaves = jax.tree.leaves(eqx.filter(gradients, eqx.is_array))
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert any(np.any(leaf != 0) for leaf in leaves)


@pytest.mark.asyncio
async def test_image_repetition_crosses_epoch_boundary_and_stops() -> None:
    source = ListAsyncDataset([{"text": "first"}, {"text": "second"}])
    repeated = RepeatedImageDataset(source, epochs=2)
    assert await repeated.async_len() == 4
    assert await repeated.get_batch([1, 2, 3]) == [{"text": "second"}, {"text": "first"}, {"text": "second"}]
    with pytest.raises(IndexError):
        await repeated.get_batch([4])


@pytest.mark.parametrize("encoder", ["siglip", "siglip2"])
@skip_if_no_torch
def test_public_vision_checkpoint_loader(tmp_path, encoder):
    from transformers import (  # noqa: PLC0415 - optional Torch reference
        SiglipVisionModel as HfSiglipVisionModel,
        Siglip2VisionModel as HfSiglip2VisionModel,
    )
    from transformers import (  # noqa: PLC0415 - optional Torch reference
        SiglipConfig as HfSiglipConfig,
        Siglip2Config as HfSiglip2Config,
    )

    parameters = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=2)
    if encoder == "siglip":
        config = SiglipVisionConfig(**parameters, image_size=4)
        hf_model = HfSiglipVisionModel(config.to_hf_config())
        axes = (hax.Axis("batch", 1), hax.Axis("channels", 3), hax.Axis("height", 4), hax.Axis("width", 4))
    else:
        config = Siglip2VisionConfig(**parameters, num_patches=4)
        hf_model = HfSiglip2VisionModel(config.to_hf_config())
        axes = (hax.Axis("batch", 1), hax.Axis("num_patches", 4), hax.Axis("patch_input", 12))
    hf_model.save_pretrained(tmp_path)
    hf_config_class = HfSiglipConfig if encoder == "siglip" else HfSiglip2Config
    hf_config_class(vision_config=hf_model.config.to_dict()).save_pretrained(tmp_path)
    converter = config.hf_checkpoint_converter(str(tmp_path))
    with use_test_mesh():
        loaded = converter.load_pretrained(config.model_type)
        template = config.model_type.init(hax.Axis("vocab", 0), config, key=jax.random.PRNGKey(1))
        expected = from_torch_compatible_state_dict(template, converter.load_state_dict(str(tmp_path)))
    pixels = hax.random.normal(jax.random.PRNGKey(2), axes)
    forward = eqx.filter_jit(lambda model: model(pixels).last_hidden_state.array)
    np.testing.assert_allclose(forward(loaded), forward(expected), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_siglip2_jitted_mixed_shapes_match_hf_embeddings():
    import torch  # noqa: PLC0415 - optional reference
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings as HfEmbeddings  # noqa: PLC0415

    config = Siglip2VisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, patch_size=2, num_patches=16
    )
    reference = HfEmbeddings(config.to_hf_config()).eval()
    template = Siglip2VisionEmbeddings.init(config, key=jax.random.PRNGKey(0))
    weights = {k: jnp.asarray(v.detach().numpy()) for k, v in reference.state_dict().items()}
    model = from_torch_compatible_state_dict(template, weights)
    pixels = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("batch", 3), hax.Axis("num_patches", 20), hax.Axis("patch_input", 12))
    )
    shapes = np.array([[2, 3], [5, 4], [3, 5]], dtype=np.int32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(np.asarray(pixels.array).copy()), torch.from_numpy(shapes)).numpy()
    actual = eqx.filter_jit(lambda model, shape: model(pixels, spatial_shapes=shape).array)(model, jnp.asarray(shapes))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("config_class", [SiglipVisionConfig, Siglip2VisionConfig])
def test_vision_attention_pads_explicit_mask(config_class):
    config = config_class(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, patch_size=2
    )
    attention = VisionAttention.init(config, key=jax.random.PRNGKey(0))
    sequence = hax.Axis("num_patches", 5)
    pixels = hax.random.normal(jax.random.PRNGKey(1), (hax.Axis("batch", 1), sequence, config.Embed))
    mask = AttentionMask.explicit(hax.named(jnp.eye(5, dtype=bool), (sequence, hax.Axis("key_position", 5))))
    actual = eqx.filter_jit(lambda x: attention(x, mask=mask))(pixels)
    # With diagonal attention, each query receives exactly its own projected value.
    expected = attention.out_proj(attention.v_proj(pixels))
    np.testing.assert_allclose(actual.array, expected.array, rtol=1e-5, atol=1e-5)
