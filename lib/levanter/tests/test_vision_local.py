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
from levanter.models.siglip2 import Siglip2VisionConfig, Siglip2VisionModel


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
