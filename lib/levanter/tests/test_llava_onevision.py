# Test file for LLaVA OneVision model
# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import sys
import tempfile

import equinox as eqx
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
from jax.sharding import Mesh

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaOnevisionConfig as HfLlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration as HfLlavaOnevision,
    LlavaOnevisionProcessor,
    PreTrainedTokenizerFast,
    Qwen2Config as HfQwen2Config,
    SiglipVisionConfig as HfSiglipVisionConfig,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    image_size_to_num_patches as hf_image_size_to_num_patches,
)

from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict
from levanter.data.image import create_custom_processor
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.llava_onevision import (
    LlavaInferenceEngine,
    LlavaOnevisionConfig,
    LlavaOnevisionModel,
    LlavaOnevisionMultimodalProjector,
    VLMRequest,
)
from levanter.models.qwen import QwenConfig
from levanter.models.siglip import SiglipVisionConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import DEFAULT_DP_AXES, MeshConfig

# Import test utils for mesh context
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import skip_if_no_torch, use_test_mesh

# Import shared helper functions from test_image_utils
from test_image_utils import (
    DEFAULT_GRID_PINPOINTS,
    compare_logits_by_region,
    create_grid_mask,
    create_lev_jax_tensors,
    get_multi_images,
    get_single_image,
    pad_pixel_values,
    prepare_test_data_single,
)


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
# Config Tests
# =====================


def test_llava_onevision_config_vision_feature_strategy_validation():
    """Test that invalid vision_feature_select_strategy raises an error."""
    with pytest.raises(ValueError, match="vision_feature_select_strategy must be"):
        LlavaOnevisionConfig(
            vision_config=_tiny_vision_config(),
            text_config=_tiny_text_config(),
            vision_feature_select_strategy="invalid_strategy",
        )


@skip_if_no_torch
def test_llava_onevision_from_hf_config():
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_llava_onevision_config()
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    assert config.image_token_index == hf_config.image_token_index
    assert config.video_token_index == hf_config.video_token_index
    assert config.vision_feature_select_strategy == hf_config.vision_feature_select_strategy
    assert config.vision_feature_layer == hf_config.vision_feature_layer
    assert config.vision_aspect_ratio == hf_config.vision_aspect_ratio
    assert config.multimodal_projector_bias == hf_config.multimodal_projector_bias
    assert config.vision_config.hidden_size == 64
    assert config.text_config.hidden_dim == 128


@skip_if_no_torch
def test_llava_onevision_config_roundtrip():
    """Test that converting HF -> Levanter -> HF preserves the config."""
    hf_config_orig = _hf_llava_onevision_config()
    levanter_config = LlavaOnevisionConfig.from_hf_config(hf_config_orig)
    hf_config_roundtrip = levanter_config.to_hf_config(vocab_size=151936)

    assert hf_config_roundtrip.image_token_index == hf_config_orig.image_token_index
    assert hf_config_roundtrip.video_token_index == hf_config_orig.video_token_index
    assert hf_config_roundtrip.projector_hidden_act == hf_config_orig.projector_hidden_act
    assert hf_config_roundtrip.vision_feature_select_strategy == hf_config_orig.vision_feature_select_strategy
    assert hf_config_roundtrip.vision_feature_layer == hf_config_orig.vision_feature_layer
    assert hf_config_roundtrip.vision_aspect_ratio == hf_config_orig.vision_aspect_ratio
    assert hf_config_roundtrip.multimodal_projector_bias == hf_config_orig.multimodal_projector_bias


# =====================
# Error Case Tests
# =====================


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

    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        model.validate_placeholder_mask(input_ids, image_features)


# =====================
# HF Comparison Tests
# =====================


@skip_if_no_torch
def test_llava_onevision_multimodal_projector_vs_hf():
    """Compare multimodal projector output with HuggingFace."""
    hf_config = _hf_llava_onevision_config()
    torch.random.manual_seed(0)
    torch_model = HfLlavaOnevision(hf_config)
    torch_model.eval()

    hf_projector = torch_model.model.multi_modal_projector

    batch_size = 2
    num_patches = 16
    vision_hidden_size = hf_config.vision_config.hidden_size

    vision_features_torch = torch.randn(batch_size, num_patches, vision_hidden_size)

    with torch.no_grad():
        hf_output = hf_projector(vision_features_torch)
        hf_output_np = hf_output.detach().cpu().numpy()

    config = LlavaOnevisionConfig.from_hf_config(hf_config)
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

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

        Vocab = Axis("vocab", hf_config.text_config.vocab_size)
        model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/torch_model")
        model = from_torch_compatible_state_dict(model_template, state_dict)

        lev_projector = model.multi_modal_projector

    Batch = Axis("batch", batch_size)
    NumPatches = Axis("num_patches", num_patches)
    VisionEmbed = Axis("embed", vision_hidden_size)

    vision_features = hax.named(
        jnp.array(vision_features_torch.numpy().astype(np.float32), dtype=jnp.float32),
        (Batch, NumPatches, VisionEmbed),
    )

    @hax.named_jit
    def compute_projector(projector, features):
        return projector(features, key=None)

    lev_output = compute_projector(lev_projector, vision_features).array

    max_diff = np.max(np.abs(hf_output_np - np.array(lev_output)))
    # Single layer comparison: use 1e-4 tolerance
    assert np.allclose(
        hf_output_np, np.array(lev_output), rtol=1e-4, atol=3e-4
    ), f"Multimodal Projector mismatch: max diff = {max_diff}"


@skip_if_no_torch
def test_llava_onevision_full_model_vs_hf():
    """Test LLaVA OneVision full model forward pass matches HuggingFace."""
    # Force float32 precision for accurate comparison with PyTorch
    # TPU default uses bfloat16 which causes ~0.01 numerical differences
    jax.config.update("jax_default_matmul_precision", "float32")

    hf_config = _hf_llava_onevision_config()
    torch.random.manual_seed(0)
    torch_model = HfLlavaOnevision(hf_config)
    torch_model.eval()
    torch_model.model.image_newline = None

    batch_size = 1
    image_height = hf_config.vision_config.image_size
    image_width = hf_config.vision_config.image_size
    num_channels = hf_config.vision_config.num_channels

    pixel_values_4d = torch.randn(batch_size, num_channels, image_height, image_width)
    num_patches_anyres = hf_image_size_to_num_patches(
        [image_height, image_width], hf_config.image_grid_pinpoints, hf_config.vision_config.image_size
    )
    pixel_values_5d = pixel_values_4d.unsqueeze(1).expand(-1, num_patches_anyres, -1, -1, -1).contiguous()

    with torch.no_grad():
        hf_image_features_list = torch_model.model.get_image_features(
            pixel_values=pixel_values_5d, image_sizes=torch.tensor([[image_height, image_width]])
        )
        hf_image_features_concat = torch.cat(hf_image_features_list, dim=0)
        num_image_tokens_full = hf_image_features_concat.shape[0]

    seq_len = 5 + num_image_tokens_full + 5
    input_ids_torch = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    input_ids_torch[0, 5 : 5 + num_image_tokens_full] = hf_config.image_token_index

    config = LlavaOnevisionConfig.from_hf_config(hf_config)
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

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

        Vocab = Axis("vocab", hf_config.text_config.vocab_size)
        model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

        converter = config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/torch_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/torch_model")
        model = from_torch_compatible_state_dict(model_template, state_dict)

    # Test patch embeddings
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

    max_diff = np.max(np.abs(hf_patch_embed_np - np.array(lev_patch_embed)))
    # With float32 precision, patch embedding should match closely
    assert np.allclose(
        hf_patch_embed_np, np.array(lev_patch_embed), rtol=1e-4, atol=1e-4
    ), f"Patch embedding mismatch: max diff = {max_diff}"

    # Test multimodal forward
    image_sizes_full = torch.tensor([[image_height, image_width]] * batch_size, dtype=torch.long)

    with torch.no_grad():
        hf_output = torch_model(
            input_ids=input_ids_torch,
            pixel_values=pixel_values_5d,
            image_sizes=image_sizes_full,
            attention_mask=torch.ones_like(input_ids_torch),
            return_dict=True,
        )
        hf_multimodal_logits = hf_output.logits.detach().cpu().numpy()

    actual_patches = num_patches_anyres
    total_patches = 10
    grid_mask_np = create_grid_mask(actual_patches, total_patches)
    pv_array = pixel_values_5d.numpy().astype(np.float32)
    pv_padded = pad_pixel_values(pv_array[0], total_patches)
    pv_padded = np.expand_dims(pv_padded, 0)

    NumPatchesPadded = Axis("num_patches", total_patches)
    pixel_values_lev_padded = hax.named(
        jnp.array(pv_padded, dtype=jnp.float32),
        (Batch, NumPatchesPadded, Channels, Height, Width),
    )

    GridMaskAxis = Axis("num_patches", total_patches)
    grid_mask = hax.named(
        jnp.array(np.expand_dims(grid_mask_np, 0)),
        (Batch, GridMaskAxis),
    )

    NumImageTokens = Axis("num_image_tokens", num_image_tokens_full)
    unpad_indices_np = np.arange(num_image_tokens_full, dtype=np.int32)
    unpad_indices = hax.named(
        jnp.array(np.expand_dims(unpad_indices_np, 0)),
        (Batch, NumImageTokens),
    )

    PositionFull = Axis("position", seq_len)
    input_ids_multimodal_lev = hax.named(
        jnp.array(input_ids_torch.numpy(), dtype=jnp.int32), (Batch, PositionFull)
    )

    def compute_multimodal(model, input_ids, pixel_values, grid_mask, unpad_indices):
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

    max_diff = np.max(np.abs(hf_multimodal_logits - np.array(lev_multimodal_logits)))
    # Multi-layer comparison: 1e-3
    assert np.allclose(
        hf_multimodal_logits, np.array(lev_multimodal_logits), rtol=1e-3, atol=1e-3
    ), f"Multimodal forward pass mismatch: max diff = {max_diff}"


@skip_if_no_torch
def test_llava_onevision_visual_embeddings_match():
    """Compare HF vs Levanter merged embeddings (text + visual) before LM."""
    # Force float32 precision for accurate comparison with PyTorch
    # TPU default uses bfloat16 which causes ~0.01 numerical differences
    jax.config.update("jax_default_matmul_precision", "float32")
    image = get_single_image()

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
    torch_model.model.image_newline = None
    torch_model.eval()
    torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS

    processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
    processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)

    text = "Describe this image briefly."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)

    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    inputs_lev = processor_lev(images=image, text=prompt, return_tensors="pt")

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

    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with use_test_mesh(mesh=single_device_mesh):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        state_dict = converter.load_state_dict(model_name)
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    lev_model = jtu.tree_map(_to_float32, lev_model)

    batch_size = inputs_lev["input_ids"].shape[0]
    Batch = Axis("batch", batch_size)

    pixel_values_torch = inputs_lev["pixel_values"]
    num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    height = pixel_values_torch.shape[3]
    width = pixel_values_torch.shape[4]

    actual_patches = num_patches
    patch_size = config.vision_config.image_size
    max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
    max_patches_per_dim = max_resolution // patch_size
    total_patches = max_patches_per_dim * max_patches_per_dim + 1
    grid_mask_np = create_grid_mask(actual_patches, total_patches)

    pv_np = pixel_values_torch.numpy().astype(np.float32)
    pv_padded_np = pad_pixel_values(pv_np[0], total_patches)
    pv_padded_np = np.expand_dims(pv_padded_np, 0)

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
        return model.get_image_features(pixel_values=pixel_values, grid_mask=grid_mask, key=None)

    image_features_result = compute_image_features(lev_model, pixel_values_lev, grid_mask)
    if isinstance(image_features_result, tuple):
        image_features_lev, _ = image_features_result
    else:
        image_features_lev = image_features_result

    batch_ax, num_patches_ax, features_per_patch_ax, embed_ax = image_features_lev.axes

    image_sizes = inputs_hf["image_sizes"].tolist()
    num_hf_image_tokens = hf_image_features.shape[0]
    unpad_indices_np = processor_lev.compute_unpad_indices(
        image_sizes=image_sizes,
        height=patch_size,
        width=patch_size,
        max_num_features=num_hf_image_tokens,
    )
    NumImageTokens = Axis("num_image_tokens", num_hf_image_tokens)
    unpad_indices = hax.named(jnp.array(unpad_indices_np, dtype=jnp.int32), (Batch, NumImageTokens))

    total_image_tokens = num_patches_ax.size * features_per_patch_ax.size
    ImageTokens = Axis("image_tokens", total_image_tokens)
    image_features_flat = hax.flatten_axes(image_features_lev, (num_patches_ax, features_per_patch_ax), ImageTokens)

    def gather_unpadded(features, indices):
        return features[indices]

    image_features_reordered = jax.vmap(gather_unpadded)(image_features_flat.array, unpad_indices.array)

    hf_raw_features = hf_image_features.cpu().numpy()
    lev_raw_features = np.array(image_features_reordered[0])

    overall_diff = np.mean(np.abs(hf_raw_features - lev_raw_features))
    # Multi-layer comparison: 1e-3
    assert overall_diff < 1e-3, f"Image features mismatch: overall_diff={overall_diff:.6e}"


# =====================
# Integration Tests
# =====================


@skip_if_no_torch
def test_llava_onevision_real_image_text():
    """Test with real image and text using processor with feature alignment.

    This test uses the same feature alignment approach as test_llava_onevision_visual_embeddings_match
    to properly compare logits between HF (unpadded) and Levanter (padded) models.
    """
    jax.config.update("jax_default_matmul_precision", "float32")

    # Load real image
    image = get_single_image()

    # Use a small pretrained model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.model.image_newline = None  # Disable image_newline for consistency
        torch_model.eval()
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}")
        return

    # Prepare inputs with processor
    text = "Describe this image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    test_pair = prepare_test_data_single(
        messages=messages,
        images=[image],
        model_name=model_name,
        add_generation_prompt=True,
    )

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

    # HuggingFace forward pass
    with torch.no_grad():
        hf_output = torch_model(**inputs_hf)
        hf_logits = hf_output.logits.detach().cpu().numpy()

    # Convert to Levanter
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    mesh_config = MeshConfig(compute_mapping={"vision_batch": DEFAULT_DP_AXES})
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

    # Compute valid image token count using attention_mask & image_mask intersection
    image_token_id = torch_model.config.image_token_index
    image_mask = test_pair.lev.input_ids == image_token_id
    valid_image_mask = test_pair.lev.attention_mask.astype(bool) & image_mask
    num_valid_image_tokens = int(valid_image_mask.sum())

    # Trim unpad_indices to actual count (remove padding zeros)
    test_pair.lev.unpad_indices = test_pair.lev.unpad_indices[:num_valid_image_tokens]

    # Create JAX tensors with batch_size=1
    jax_tensors = create_lev_jax_tensors(test_pair.lev, batch_size=1)
    input_ids_lev_tensor = jax_tensors.input_ids
    pixel_values_lev_tensor = jax_tensors.pixel_values
    grid_mask = jax_tensors.grid_mask
    unpad_indices = jax_tensors.unpad_indices

    @hax.named_jit
    def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
        return model(
            input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            key=None,
        )

    # Forward pass
    lev_logits = compute_lev(
        lev_model, input_ids_lev_tensor, pixel_values_lev_tensor, grid_mask, unpad_indices
    )
    lev_logits = lev_logits.array

    # Compare logits
    lev_logits_np = np.array(lev_logits)
    if lev_logits_np.ndim == 3:
        lev_logits_np = lev_logits_np[0]  # Remove batch dimension

    hf_logits_flat = hf_logits[0]  # (seq_len, vocab_size)

    # Note: tolerance=1.5e-3 accounts for cross-framework numerical differences
    # between JAX and PyTorch, especially in SigLIP vision encoder attention.
    comparison_result = compare_logits_by_region(
        hf_logits=hf_logits_flat,
        lev_logits=lev_logits_np,
        input_ids=test_pair.hf.input_ids,
        image_token_id=image_token_id,
        tolerance=1.5e-3,
        verbose=False,
        detailed=False,
        attention_mask=test_pair.lev.attention_mask,
    )

    assert comparison_result.passed, "Real image/text test failed"

@skip_if_no_torch
def test_llava_onevision_real_multi_image_text():
    """Test Levanter model with multiple images, comparing HF and Levanter outputs.

    This test validates multi-image behavior where:
    - Both HF and Levanter use base patch per image (no anyres sub-patches)
    - unpad_indices is None for multi-image case
    - grid_mask marks which patches are valid (num_images base patches)
    - HF processor generates correct image tokens with padding_mode=True
    """
    jax.config.update("jax_default_matmul_precision", "float32")
    # Load multiple images
    images = get_multi_images()  # Returns list of 2 images
    num_images = len(images)

    # Use a small pretrained model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    try:
        torch_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        torch_model.model.image_newline = None  # Disable image_newline for consistency
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}")
        return

    # Prepare inputs with processor
    text = "Compare these two images and describe the differences."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]

    test_pair = prepare_test_data_single(
        messages=messages,
        images=images,
        model_name=model_name,
        add_generation_prompt=True,
        max_length=16384,  # Larger max_length for multi-image to avoid truncation
    )

    # Verify multi-image preprocessing is correct
    assert test_pair.lev.unpad_indices is None, "Multi-image should have None unpad_indices"
    assert (
        test_pair.lev.grid_mask.sum() == num_images
    ), f"Multi-image should have {num_images} valid patches (base only)"

    # Prepare HF inputs for forward pass
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)

    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values)
    if hf_pixel_values.dim() == 4:
        hf_pixel_values = hf_pixel_values.unsqueeze(0)

    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes)
    if hf_image_sizes.dim() == 1:
        hf_image_sizes = hf_image_sizes.unsqueeze(0)

    # HuggingFace forward pass with batch_num_images for multi-image mode
    with torch.no_grad():
        hf_output = torch_model(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            attention_mask=hf_attention_mask,
            image_sizes=hf_image_sizes,
            batch_num_images=torch.tensor([num_images]),
        )
        hf_logits = hf_output.logits.detach().cpu().numpy()

    # Convert to Levanter
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    mesh_config = MeshConfig(
        axes={"model": 8, "data": 1, "replica": 1},
        compute_mapping={
            "vision_batch": ("model",),
            "vocab": "model",
            "batch": ("replica_dcn", "replica"),
        },
        shared_mapping={
            "heads": "data",
            "mlp": "data",
        },
        param_mapping={
            "heads": "data",
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

        # Create JAX tensors
        jax_tensors = create_lev_jax_tensors(test_pair.lev, batch_size=1)
        input_ids_lev_tensor = jax_tensors.input_ids
        pixel_values_lev_tensor = jax_tensors.pixel_values
        grid_mask = jax_tensors.grid_mask
        unpad_indices = jax_tensors.unpad_indices

        assert unpad_indices is None, "Multi-image should have None unpad_indices in JAX tensors"

        @hax.named_jit
        def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=None,
            )

        lev_logits = compute_lev(
            lev_model, input_ids_lev_tensor, pixel_values_lev_tensor, grid_mask, unpad_indices
        )
        lev_logits = lev_logits.array

        # Verify logits are not NaN/Inf
        assert not jnp.isnan(lev_logits).any(), "Logits contain NaN"
        assert not jnp.isinf(lev_logits).any(), "Logits contain Inf"

        # Compare logits
        lev_logits_np = np.array(lev_logits)
        if lev_logits_np.ndim == 3:
            lev_logits_np = lev_logits_np[0]

        hf_logits_flat = hf_logits[0]

        image_token_id = torch_model.config.image_token_index
        comparison_result = compare_logits_by_region(
            hf_logits=hf_logits_flat,
            lev_logits=lev_logits_np,
            input_ids=test_pair.hf.input_ids,
            image_token_id=image_token_id,
            tolerance=1.5e-3,
            verbose=True,
            detailed=True,
            attention_mask=test_pair.lev.attention_mask,
        )

        assert (
            comparison_result.passed
        ), f"Multi-image test failed"


@skip_if_no_torch
def test_llava_onevision_real_image_text_0_5b_batch():
    """Test with batch padding for better TPU utilization."""
    jax.config.update("jax_default_matmul_precision", "float32")
    image = get_single_image()
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.model.image_newline = None
        torch_model.eval()
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
        processor_hf = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
        processor_lev = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_GRID_PINPOINTS)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}")
        return

    text = "Describe this image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    prompt = processor_hf.apply_chat_template(messages, add_generation_prompt=True)
    inputs_hf = processor_hf(images=image, text=prompt, return_tensors="pt", padding_mode=False)
    inputs_lev = processor_lev(
        images=image, text=prompt, return_tensors="pt", padding="max_length", max_length=8192, padding_mode=True
    )

    with torch.no_grad():
        hf_output = torch_model(**inputs_hf)
        hf_logits = hf_output.logits.detach().cpu().numpy()

    image_token_id = torch_model.config.image_token_index
    input_ids_for_mask = inputs_hf["input_ids"].numpy()[0]
    image_mask = input_ids_for_mask == image_token_id
    num_image_tokens = image_mask.sum()

    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    mesh_config = MeshConfig(
        compute_mapping={
            "vision_batch": DEFAULT_DP_AXES,
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

        seq_len = inputs_lev["input_ids"].shape[1]
        target_batch_size = 8

        Batch = Axis("batch", target_batch_size)
        Position = Axis("position", seq_len)

        input_ids_np = inputs_lev["input_ids"].numpy()
        input_ids_np = np.tile(input_ids_np, (target_batch_size, 1))
        input_ids_lev = hax.named(jnp.array(input_ids_np, dtype=jnp.int32), (Batch, Position))

        pixel_values_torch = inputs_lev["pixel_values"]
        num_patches = pixel_values_torch.shape[1]
        channels = pixel_values_torch.shape[2]
        height = pixel_values_torch.shape[3]
        width = pixel_values_torch.shape[4]

        Channels = Axis("channels", channels)
        Height = Axis("height", height)
        Width = Axis("width", width)

        actual_patches = pixel_values_torch.shape[1]
        patch_size = config.vision_config.image_size
        max_resolution = max(max(h, w) for h, w in config.image_grid_pinpoints)
        max_patches_per_dim = max_resolution // patch_size
        total_patches = max_patches_per_dim * max_patches_per_dim + 1
        grid_mask_np = create_grid_mask(actual_patches, total_patches)

        pv_np = pixel_values_torch.numpy().astype(np.float32)
        pv_padded_single = pad_pixel_values(pv_np[0], total_patches)
        pv_padded_np = np.tile(np.expand_dims(pv_padded_single, 0), (target_batch_size, 1, 1, 1, 1))

        NumPatchesPadded = Axis("num_patches", total_patches)
        GridMaskAxis = Axis("grid_mask", total_patches)
        pixel_values_lev = hax.named(
            jnp.array(pv_padded_np, dtype=jnp.float32),
            (Batch, NumPatchesPadded, Channels, Height, Width),
        )
        grid_mask_tiled = np.tile(np.expand_dims(grid_mask_np, 0), (target_batch_size, 1))
        grid_mask = hax.named(jnp.array(grid_mask_tiled), (Batch, GridMaskAxis))

        image_sizes = inputs_lev["image_sizes"].tolist()
        unpad_indices_np = processor_lev.compute_unpad_indices(
            image_sizes=image_sizes,
            height=patch_size,
            width=patch_size,
            max_num_features=num_image_tokens,
        )
        unpad_indices_np = np.tile(unpad_indices_np, (target_batch_size, 1))
        NumImageTokens = Axis("num_image_tokens", num_image_tokens)
        unpad_indices = hax.named(jnp.array(unpad_indices_np, dtype=jnp.int32), (Batch, NumImageTokens))

        @hax.named_jit
        def compute_lev(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=None,
            )

        lev_logits = compute_lev(lev_model, input_ids_lev, pixel_values_lev, grid_mask, unpad_indices)
        lev_logits.array.block_until_ready()

        lev_logits_np = np.array(lev_logits.array[0])
        hf_logits_flat = hf_logits[0]

        tolerance = 1.5e-3
        attention_mask_np = inputs_lev["attention_mask"].numpy()[0]
        result = compare_logits_by_region(
            hf_logits=hf_logits_flat,
            lev_logits=lev_logits_np,
            input_ids=input_ids_for_mask,
            image_token_id=image_token_id,
            tolerance=tolerance,
            verbose=False,
            detailed=False,
            attention_mask=attention_mask_np,
        )

        assert result.passed, f"Batch test failed"


# =====================
# Image Feature Tests
# =====================


@skip_if_no_torch
def test_get_image_features_vs_hf_real_single_image():
    """Compare raw image features with HF using a real single image."""
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    jax.config.update("jax_default_matmul_precision", "float32")
    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        hf_config = torch_model.config
        processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}, error: {e}")
        return

    image = get_single_image()

    inputs = processor(text="Describe this image.", images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"]

    batch_size = pixel_values_torch.shape[0]
    num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    patch_height = pixel_values_torch.shape[3]
    patch_width = pixel_values_torch.shape[4]

    pixel_values_flat = pixel_values_torch.reshape(-1, channels, patch_height, patch_width)

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

    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, config, key=random.PRNGKey(0))

    mesh_config = MeshConfig(
        axes={"model": 8, "data": 1, "replica": 1},
        compute_mapping={
            "vision_batch": ("model",),
            "vocab": "model",
            "batch": ("replica_dcn", "replica"),
        },
        shared_mapping={
            "heads": "data",
            "mlp": "data",
        },
        param_mapping={
            "heads": "data",
        },
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
        state_dict = converter.load_state_dict(model_name)
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    lev_model = jtu.tree_map(_to_float32, lev_model)

    pv_np = pixel_values_torch.numpy().astype(np.float32)
    grid_mask_np = np.ones((batch_size, num_patches), dtype=bool)

    Batch = Axis("batch", batch_size)
    NumPatches = Axis("num_patches", num_patches)
    Channels = Axis("channels", channels)
    Height = Axis("height", patch_height)
    Width = Axis("width", patch_width)

    pixel_values_lev = hax.named(jnp.array(pv_np, dtype=jnp.float32), (Batch, NumPatches, Channels, Height, Width))
    grid_mask = hax.named(jnp.array(grid_mask_np), (Batch, NumPatches))

    @hax.named_jit
    def compute_lev_single(model, pixel_values, grid_mask):
        return model.get_image_features(pixel_values=pixel_values, grid_mask=grid_mask, key=None)

    lev_result = compute_lev_single(lev_model, pixel_values_lev, grid_mask)
    lev_image_features = lev_result[0] if isinstance(lev_result, tuple) else lev_result

    hf_array = hf_raw_features.detach().numpy()
    lev_array = np.array(lev_image_features.array)

    hf_array_reshaped = hf_array.reshape(batch_size, num_patches, -1, hf_array.shape[-1])

    assert hf_array_reshaped.shape == lev_array.shape, f"Shape mismatch: HF={hf_array_reshaped.shape}, Lev={lev_array.shape}"

    mean_diff = np.mean(np.abs(hf_array_reshaped - lev_array))
    # Multi-layer comparison: 1e-3
    assert mean_diff < 1e-3, f"Values don't match: mean diff = {mean_diff}"


@skip_if_no_torch
def test_get_image_features_vs_hf_real_multi_image():
    """Compare raw image features with HF using real multiple images."""
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        hf_config = torch_model.config
        processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}, error: {e}")
        return

    image = get_single_image()
    images = [image, image, image]

    inputs = processor(text="Describe these images.", images=images, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"]

    batch_size = pixel_values_torch.shape[0]
    num_patches = pixel_values_torch.shape[1]
    channels = pixel_values_torch.shape[2]
    patch_height = pixel_values_torch.shape[3]
    patch_width = pixel_values_torch.shape[4]

    pixel_values_flat = pixel_values_torch.reshape(-1, channels, patch_height, patch_width)

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

    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    mesh_config = MeshConfig(
        compute_mapping={
            "vision_batch": DEFAULT_DP_AXES,
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

        original_batch_size = batch_size
        target_batch_size = 8

        pv_np = pixel_values_torch.numpy().astype(np.float32)
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

        @hax.named_jit
        def compute_lev_multi(model, pixel_values, grid_mask):
            return model.get_image_features(pixel_values=pixel_values, grid_mask=grid_mask, key=None)

        lev_result = compute_lev_multi(lev_model, pixel_values_lev, grid_mask)
        lev_image_features = lev_result[0] if isinstance(lev_result, tuple) else lev_result

        hf_array = hf_raw_features.detach().numpy()
        lev_array = np.array(lev_image_features.array)[:original_batch_size]

        hf_array_reshaped = hf_array.reshape(original_batch_size, num_patches, -1, hf_array.shape[-1])

        assert hf_array_reshaped.shape == lev_array.shape, f"Shape mismatch: HF={hf_array_reshaped.shape}, Lev={lev_array.shape}"

        mean_diff = np.mean(np.abs(hf_array_reshaped - lev_array))
        # Multi-layer comparison: 1e-3
        assert mean_diff < 1e-3, f"Values don't match: mean diff = {mean_diff}"


# =====================
# Generation Engine Tests
# =====================


@pytest.mark.slow
@skip_if_no_torch
def test_llava_onevision_generation_with_inference_engine():
    """Test generation using Levanter's LlavaInferenceEngine with VLMRequest (single image)."""
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    image = get_single_image()

    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        torch_model.model.image_newline = None
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}")
        return

    text = "Describe the image in detail."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    test_pair = prepare_test_data_single(
        messages=messages,
        images=[image],
        model_name=model_name,
        add_generation_prompt=True,
    )

    # HuggingFace generation
    max_new_tokens = 100
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)
    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes).unsqueeze(0)

    with torch.no_grad():
        hf_output_ids = torch_model.generate(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            attention_mask=hf_attention_mask,
            image_sizes=hf_image_sizes,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_len = hf_input_ids.shape[1]
    hf_generated_ids = hf_output_ids[0, prompt_len:].cpu().numpy()

    # Levanter generation
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)

    trainer_config = TrainerConfig()  # Default: model_axis_size=1, all devices on data axis
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        mesh = trainer_config.device_mesh
        compute_dtype = jnp.float32
        Vocab = Axis("vocab", hf_config.text_config.vocab_size)

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

        jax_tensors = create_lev_jax_tensors(test_pair.lev, batch_size=1)

        # Configure InferenceEngine
        prompt_len_lev = len(test_pair.hf.input_ids)
        estimated_max_seq_len = prompt_len_lev + max_new_tokens + 64
        page_size = 16

        engine_config = InferenceEngineConfig(
            max_seq_len=estimated_max_seq_len,
            page_size=page_size,
            max_seqs=1,
            max_rounds=32,
            max_stop_seqs=1,
            max_stop_tokens=4,
            max_pages=800,  # Reduced from 200 to avoid OOM
            compute_dtype=compute_dtype,
        )

        engine = LlavaInferenceEngine.from_model_with_config(
            model=lev_model,
            tokenizer=processor.tokenizer,
            config=engine_config,
            Vocab=Vocab,
            mesh=mesh,
        )

        prompt_tokens = test_pair.hf.input_ids.tolist()

        eos_token_id = processor.tokenizer.eos_token_id
        if eos_token_id is not None:
            stop_tokens = hax.named(jnp.array([[eos_token_id]], dtype=jnp.int32), ("stop_seq", "position"))
        else:
            stop_tokens = None

        decode_params = SeqDecodingParams(
            max_num_tokens=estimated_max_seq_len,
            temperature=0.0,
            key=random.PRNGKey(42),
            stop_tokens=stop_tokens,
        )

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

        result = engine.generate([vlm_request])

    lev_generated_ids = np.array(result.tokens[0])

    # Compare results
    min_len = min(len(hf_generated_ids), len(lev_generated_ids))
    matching_tokens = sum(1 for i in range(min_len) if hf_generated_ids[i] == lev_generated_ids[i])
    match_ratio = matching_tokens / min_len if min_len > 0 else 0

    min_expected_tokens = len(hf_generated_ids) // 2
    assert len(lev_generated_ids) >= min_expected_tokens, (
        f"Levanter generated too few tokens: {len(lev_generated_ids)} < {min_expected_tokens}"
    )
    assert match_ratio >= 0.99, f"Token match ratio too low: {match_ratio:.1%}"


@pytest.mark.slow
@skip_if_no_torch
def test_llava_onevision_generation_with_inference_engine_multi():
    """Test generation using Levanter's LlavaInferenceEngine with VLMRequest (multi image)."""
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    images = get_multi_images()
    num_images = len(images)

    try:
        torch_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model.eval()
        torch_model.model.image_newline = None
        torch_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not download model: {model_name}")
        return

    text = "Compare these two images and describe the differences."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]

    test_pair = prepare_test_data_single(
        messages=messages,
        images=images,
        model_name=model_name,
        add_generation_prompt=True,
    )

    assert test_pair.lev.unpad_indices is None, "Multi-image should have None unpad_indices"
    assert test_pair.lev.grid_mask.sum() == num_images, f"Multi-image should have {num_images} valid patches"

    # HuggingFace generation
    max_new_tokens = 100
    hf_input_ids = torch.tensor(test_pair.hf.input_ids).unsqueeze(0)
    hf_attention_mask = torch.tensor(test_pair.hf.attention_mask).unsqueeze(0)
    hf_pixel_values = torch.tensor(test_pair.hf.pixel_values)
    if hf_pixel_values.dim() == 4:
        hf_pixel_values = hf_pixel_values.unsqueeze(0)
    hf_image_sizes = torch.tensor(test_pair.hf.image_sizes)
    if hf_image_sizes.dim() == 1:
        hf_image_sizes = hf_image_sizes.unsqueeze(0)

    with torch.no_grad():
        hf_output_ids = torch_model.generate(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            attention_mask=hf_attention_mask,
            image_sizes=hf_image_sizes,
            batch_num_images=torch.tensor([num_images]),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = hf_input_ids.shape[1]
    hf_generated_ids = hf_output_ids[0, prompt_len:].cpu().numpy()

    # Levanter generation
    hf_config = torch_model.config
    config = LlavaOnevisionConfig.from_hf_config(hf_config)
    text_config_updated = dataclasses.replace(config.text_config, attn_backend="dot", flash_attention_block_size=None)
    config = dataclasses.replace(config, text_config=text_config_updated)

    trainer_config = TrainerConfig()

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        mesh = trainer_config.device_mesh
        compute_dtype = jnp.float32
        Vocab = Axis("vocab", hf_config.text_config.vocab_size)

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

        jax_tensors = create_lev_jax_tensors(test_pair.lev, batch_size=1)

        # Configure InferenceEngine
        prompt_len_lev = len(test_pair.hf.input_ids)
        estimated_max_seq_len = prompt_len_lev + max_new_tokens + 64
        page_size = 16

        engine_config = InferenceEngineConfig(
            max_seq_len=estimated_max_seq_len,
            page_size=page_size,
            max_seqs=1,
            max_rounds=32,
            max_stop_seqs=1,
            max_stop_tokens=4,
            max_pages=800,
            compute_dtype=compute_dtype,
        )

        engine = LlavaInferenceEngine.from_model_with_config(
            model=lev_model,
            tokenizer=processor.tokenizer,
            config=engine_config,
            Vocab=Vocab,
            mesh=mesh,
        )

        prompt_tokens = test_pair.hf.input_ids.tolist()

        eos_token_id = processor.tokenizer.eos_token_id
        if eos_token_id is not None:
            stop_tokens = hax.named(jnp.array([[eos_token_id]], dtype=jnp.int32), ("stop_seq", "position"))
        else:
            stop_tokens = None

        decode_params = SeqDecodingParams(
            max_num_tokens=estimated_max_seq_len,
            temperature=0.0,
            key=random.PRNGKey(42),
            stop_tokens=stop_tokens,
        )

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

        result = engine.generate([vlm_request])

    lev_generated_ids = np.array(result.tokens[0])

    # Compare results
    min_len = min(len(hf_generated_ids), len(lev_generated_ids))
    matching_tokens = sum(1 for i in range(min_len) if hf_generated_ids[i] == lev_generated_ids[i])
    match_ratio = matching_tokens / min_len if min_len > 0 else 0

    assert match_ratio >= 0.99, f"Token match ratio too low: {match_ratio:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
