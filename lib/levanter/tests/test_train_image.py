# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Test training for vision-language models (LLaVA OneVision).

This test validates the training pipeline for image-text models,
similar to test_train_asr.py for audio models.
"""

import dataclasses
import os
import tempfile

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoConfig

from levanter.main.train_vlm import compute_vlm_loss
from levanter.models.llava_onevision import LlavaOnevisionConfig
from levanter.models.siglip import SiglipVisionConfig
from test_image_utils import (
    prepare_test_data,
    compare_logits_by_region,
    create_lev_jax_tensors,
    DEFAULT_GRID_PINPOINTS,
)
from test_image_utils import get_real_data, get_single_image

# Define skip_if_no_torch locally to avoid conftest dependencies
try:
    import torch  # noqa: F401

    skip_if_no_torch = pytest.mark.skipif(False, reason="torch is available")
except ImportError:
    skip_if_no_torch = pytest.mark.skip(reason="torch not available")

# =====================
# Module-level constants
# =====================
MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
MODEL_NAME_7B = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 4

# Environment configuration for VLM tests.
# These must be set at module level before JAX initialization to take effect.
# - CUDA_VISIBLE_DEVICES="": Force torch to use CPU (avoid GPU memory conflicts)
# - JAX_PLATFORMS="tpu": Force JAX to use TPU backend
# - JAX_DEFAULT_DTYPE_BITS="32": Use float32 for numerical precision in tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")


# =====================
# Helper functions
# =====================


def _load_levanter_config(model_name=MODEL_NAME, enable_flash_attention=False, gradient_checkpointing=True):
    """Load and configure LlavaOnevisionConfig with common settings.

    Args:
        model_name: HuggingFace model name to load config from
        enable_flash_attention: Whether to enable flash attention for text model
        gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Configured LlavaOnevisionConfig
    """
    from levanter.layers.attention import AttentionBackend

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    lev_config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Configure vision tower - disable flash attention to avoid XLA errors
    vision_config_updated = dataclasses.replace(
        lev_config.vision_config,
        use_flash_attention=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Configure text model
    attn_backend = AttentionBackend.DEFAULT if enable_flash_attention else AttentionBackend.VANILLA
    text_config_updated = dataclasses.replace(
        lev_config.text_config,
        attn_backend=attn_backend,
        gradient_checkpointing=gradient_checkpointing,
    )

    return dataclasses.replace(
        lev_config,
        vision_config=vision_config_updated,
        text_config=text_config_updated,
        gradient_checkpointing=gradient_checkpointing,
    )


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_numerical_correctness():
    """
    Verify numerical correctness of Levanter VLM vs HuggingFace implementation.

    Uses real parquet dataset and compares forward pass outputs.
    Following the pattern from test_llava_hf_levanter_consistency_no_padding:
    - HF uses processor with do_pad=False (variable-shape processing)
    - Levanter uses processor with do_pad=True (fixed-shape processing with grid_mask)
    """
    import torch
    from transformers import AutoModelForVision2Seq
    from haliax import Axis
    from levanter.models.llava_onevision import LlavaOnevisionModel

    # Use real HuggingFace model for comparison
    model_name = MODEL_NAME
    grid_pinpoints = DEFAULT_GRID_PINPOINTS
    num_samples = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # ========== Load test data using prepare_test_data ==========
        print("\n--- Loading test data using prepare_test_data ---")
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=9,
            grid_pinpoints=grid_pinpoints,
        )

    # ========== Load HuggingFace model ==========
    print("\n--- Loading HuggingFace model ---")
    hf_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.image_newline = None
    hf_model.eval()

    # ========== Load Levanter model ==========
    print("\n--- Loading Levanter model ---")
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    lev_config = LlavaOnevisionConfig.from_hf_config(hf_config)

    # Disable flash attention for fair comparison
    text_config_updated = dataclasses.replace(
        lev_config.text_config, attn_backend="dot", flash_attention_block_size=None
    )
    lev_config = dataclasses.replace(lev_config, text_config=text_config_updated)

    from jax import random

    Vocab = Axis("vocab", hf_config.text_config.vocab_size)
    model_template = eqx.filter_eval_shape(LlavaOnevisionModel.init, Vocab, lev_config, key=random.PRNGKey(0))
    converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
    state_dict = converter.load_state_dict(model_name)
    from levanter.compat.hf_checkpoints import from_torch_compatible_state_dict

    lev_model = from_torch_compatible_state_dict(model_template, state_dict)

    # Convert model weights to float32
    import jax.tree_util as jtu

    def to_float32(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float32)
        return x

    lev_model = jtu.tree_map(to_float32, lev_model)

    # Forward function for Levanter
    @eqx.filter_jit
    def compute_forward(model, input_ids, pixel_values, grid_mask, unpad_indices):
        return model(input_ids, pixel_values=pixel_values, grid_mask=grid_mask, unpad_indices=unpad_indices, key=None)

    # ========== Test each sample ==========
    all_max_diffs = []
    all_mean_diffs = []
    all_passed = []

    print(f"\n=== Testing {num_samples} samples ===")

    for sample_idx, pair in enumerate(test_pairs):
        print(f"\n  Sample {sample_idx}:")

        # --- HF Forward Pass (using pair.hf data) ---
        hf_input_ids = torch.from_numpy(pair.hf.input_ids).unsqueeze(0)
        hf_pixel_values = torch.from_numpy(pair.hf.pixel_values).unsqueeze(0)
        hf_image_sizes = torch.from_numpy(pair.hf.image_sizes).unsqueeze(0)

        print(f"    HF input_ids shape: {hf_input_ids.shape}")
        print(f"    HF pixel_values shape: {hf_pixel_values.shape}")

        with torch.no_grad():
            hf_output = hf_model(
                input_ids=hf_input_ids,
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
            )
            hf_logits = hf_output.logits[0].numpy()

        print(f"    HF logits shape: {hf_logits.shape}")

        # --- Levanter Forward Pass (using pair.lev data, already has grid_mask and padding) ---
        print(f"    Lev input_ids shape: {pair.lev.input_ids.shape}")
        print(f"    Lev pixel_values shape: {pair.lev.pixel_values.shape}")
        print(f"    Lev grid_mask valid patches: {pair.lev.grid_mask.sum()}")

        # Create named arrays using create_lev_jax_tensors helper
        # Use batch_size=1 since this test doesn't use device_mesh sharding
        jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=1)

        lev_logits = compute_forward(
            lev_model,
            jax_tensors.input_ids,
            jax_tensors.pixel_values,
            jax_tensors.grid_mask,
            jax_tensors.unpad_indices,
        )
        lev_logits_np = np.array(lev_logits.array)[0]

        print(f"    Lev logits shape: {lev_logits_np.shape}")

        # Compare logits using region-based comparison
        image_token_id = hf_model.config.image_token_index
        result = compare_logits_by_region(
            hf_logits=hf_logits,
            lev_logits=lev_logits_np,
            input_ids=pair.hf.input_ids,
            image_token_id=image_token_id,
            tolerance=1e-2,
            verbose=True,
        )

        all_max_diffs.append(result.overall_max_diff)
        all_mean_diffs.append(result.overall_mean_diff)
        all_passed.append(result.passed)

    # --- Summary ---
    print("\n--- Summary ---")
    avg_max_diff = np.mean(all_max_diffs)
    avg_mean_diff = np.mean(all_mean_diffs)
    pass_rate = np.mean(all_passed)
    print(f"  Average max diff: {avg_max_diff:.6f}")
    print(f"  Average mean diff: {avg_mean_diff:.6f}")
    print(f"  Pass rate: {pass_rate:.2%}")

    # Assert all samples passed
    assert all(all_passed), f"Not all samples passed: {sum(all_passed)}/{len(all_passed)}"
    print("\nNumerical correctness test passed!")


# =====================
# Unit tests for image data loading
# =====================


@skip_if_no_torch
def test_batch_image_processor():
    """Test BatchImageProcessor with synthetic conversation data."""
    try:
        from transformers import AutoProcessor
        from PIL import Image
    except ImportError:
        pytest.skip("transformers or PIL not available")

    from levanter.data.image import BatchImageProcessor

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Use a larger max_length to accommodate all image tokens
    # LLaVA OneVision generates ~1500+ tokens for a single image due to anyres processing
    max_length = MAX_LENGTH

    batch_processor = BatchImageProcessor(
        processor,
        max_length=max_length,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
    )

    # Create synthetic data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image (smaller to reduce number of patches)
        img_path = f"{tmpdir}/test.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_path)

        example = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is this?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "This is a test image."},
                    ],
                },
            ],
            "images": [img_path],
        }

        # Process the batch
        results = batch_processor([example])

        assert len(results) == 1
        result = results[0]

        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["input_ids"].shape == (max_length,)


@skip_if_no_torch
def test_image_mixture_dataset_config():
    """Test ImageMixtureDatasetConfig creation."""
    from levanter.data.image import (
        ImageMixtureDatasetConfig,
        ImageDatasetSourceConfig,
        ConversationDatasetSourceConfig,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ImageMixtureDatasetConfig(
            cache_dir=tmpdir,
            configs={
                "ds1": ImageDatasetSourceConfig(
                    train_urls=[f"{tmpdir}/train1.jsonl"],
                    cache_dir=f"{tmpdir}/ds1",
                ),
                "ds2": ConversationDatasetSourceConfig(
                    train_urls=[f"{tmpdir}/train2.jsonl"],
                    cache_dir=f"{tmpdir}/ds2",
                ),
            },
            train_weights={"ds1": 0.6, "ds2": 0.4},
        )

        assert len(config.configs) == 2
        assert config.train_weights["ds1"] == 0.6
        assert config.train_weights["ds2"] == 0.4


# =====================
# Integration tests with LLaVA model
# =====================


@skip_if_no_torch
def test_llava_forward_pass_with_image_data():
    """Test LLaVA forward pass with image data from the data loader."""
    from levanter.data.loader import ImageDataLoader
    from levanter.data.image import BatchImageProcessor
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from levanter.store.cache import SerialCacheWriter
    from PIL import Image

    # Import custom processor for padding support
    from levanter.data.image import create_custom_processor, DEFAULT_IMAGE_GRID_PINPOINTS

    model_name = MODEL_NAME

    # Use the default grid_pinpoints (anyres_max_9 configuration)
    grid_pinpoints = DEFAULT_IMAGE_GRID_PINPOINTS

    # Create padded processor for Levanter (do_pad=True)
    processor = create_custom_processor(model_name, do_pad=True)

    # Use a larger max_length to accommodate all image tokens
    # LLaVA OneVision generates ~1500+ tokens for a single image due to anyres processing
    max_length = MAX_LENGTH
    patch_size = 384
    vision_feature_height = patch_size // 14
    # Calculate max_num_patches from grid_pinpoints (e.g., anyres_max_9 -> 9)
    max_num_patches = len(grid_pinpoints)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        img_path = f"{tmpdir}/test.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        img.save(img_path)

        # Create batch processor with grid_pinpoints for grid_mask computation
        batch_processor = BatchImageProcessor(
            processor,
            max_length=max_length,
            padding=True,
            messages_key="messages",
            images_key="images",
            mask_prompt=False,
            grid_pinpoints=grid_pinpoints,
            patch_size=patch_size,
            vision_feature_height=vision_feature_height,
            max_num_patches=max_num_patches,
        )

        # Create cache with test data
        cache_dir = f"{tmpdir}/cache"
        with SerialCacheWriter(cache_dir, batch_processor.output_exemplar) as writer:
            for i in range(8):
                example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": f"What is this image {i}?"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"This is test image {i}."},
                            ],
                        },
                    ],
                    "images": [img_path],
                }
                try:
                    results = batch_processor([example])
                    writer.write_batch(results)
                except ValueError as e:
                    print(f"Skipping example {i} due to processing error: {e}")
                    continue

        cache = writer.result()
        cache_len = len(cache)

        if cache_len < 2:
            raise ValueError("Not enough examples cached")

        # Load model config (disable gradient checkpointing and use vanilla attention for testing)
        config = _load_levanter_config(model_name, enable_flash_attention=False, gradient_checkpointing=False)

        # Load model
        trainer_config = TrainerConfig()

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

            # Get example shape info
            all_examples = cache.get_batch_sync(list(range(cache_len)))
            max_num_patches = max(ex["pixel_values"].shape[0] for ex in all_examples)
            first_ex = all_examples[0]
            seq_len = first_ex["input_ids"].shape[0]

            # Create axes
            Pos = hax.Axis("position", seq_len)
            NumPatches = hax.Axis("num_patches", max_num_patches)
            Channels = hax.Axis("channels", 3)
            Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
            Width = hax.Axis("width", first_ex["pixel_values"].shape[3])

            # Calculate NumImageTokens for unpad_indices support
            vision_feature_height = config.vision_config.image_size // config.vision_config.patch_size
            features_per_patch = vision_feature_height * vision_feature_height
            max_image_tokens = max_num_patches * features_per_patch
            NumImageTokens = hax.Axis("num_image_tokens", max_image_tokens)

            # Create dataloader
            batch_size = min(8, cache_len)
            axis_resources = trainer_config.compute_axis_mapping

            from jax._src.mesh import get_concrete_mesh

            mesh = get_concrete_mesh()

            loader = ImageDataLoader(
                data=cache,
                batch_size=batch_size,
                Pos=Pos,
                NumPatches=NumPatches,
                Channels=Channels,
                Height=Height,
                Width=Width,
                axis_resources=axis_resources,
                mesh=mesh,
                max_buffered_batches=0,
                NumImageTokens=NumImageTokens,
                allow_nondivisible_batch_size=True,
            )

            # Get batch and run forward pass
            batch_iter = iter(loader)
            batch = next(batch_iter)

            @hax.named_jit
            def compute_forward(model, input_ids, pixel_values, grid_mask, unpad_indices):
                return model(
                    input_ids,
                    pixel_values=pixel_values,
                    grid_mask=grid_mask,
                    unpad_indices=unpad_indices,
                    key=None,
                )

            logits = compute_forward(
                lev_model,
                batch.input_ids,
                batch.pixel_values,
                batch.grid_mask,
                batch.unpad_indices,
            )

            # Verify output shape
            assert logits.array.shape[0] == batch_size
            assert logits.array.shape[1] == seq_len


# =====================
# VLM Training Correctness Tests
# =====================


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_loss_consistency():
    """
    Test 1: Verify that Levanter's VLM loss computation works end-to-end with real parquet data.

    Uses ImageDataLoader to load batched data from parquet file, matching the actual training pipeline.
    """
    from levanter.data.loader import ImageDataLoader
    from levanter.data.image import ProcessedImageCache, ConversationUrlDataSource
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from levanter.store.cache import CacheOptions

    # Import custom processor for padding support
    from levanter.data.image import create_custom_processor, DEFAULT_IMAGE_GRID_PINPOINTS

    model_name = MODEL_NAME

    # Create custom processor with proper anyres configuration (do_pad=True for Levanter)
    processor = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_IMAGE_GRID_PINPOINTS)
    # Set vision_aspect_ratio to enable max_num_patches extraction
    processor.image_processor.vision_aspect_ratio = f"anyres_max_{len(DEFAULT_IMAGE_GRID_PINPOINTS)}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Create data source from parquet file
        source = ConversationUrlDataSource([parquet_path], messages_key="messages", images_key="images")

        # Build cache using ProcessedImageCache.build_or_load with custom processor
        print("\n=== Building cache from parquet data ===")
        cache = ProcessedImageCache.build_or_load(
            cache_dir=tmpdir,
            source=source,
            processor=processor,
            max_length=MAX_LENGTH,
            padding=True,
            messages_key="messages",
            images_key="images",
            cache_options=CacheOptions.default(),
        )
        tree_cache = cache.cache
        cache_len = len(tree_cache)
        print(f"  Cache length: {cache_len}")

        if cache_len < 2:
            raise ValueError("Not enough examples cached")

        # Load model config (disable gradient checkpointing and use vanilla attention for testing)
        config = _load_levanter_config(model_name, enable_flash_attention=False, gradient_checkpointing=False)

        # Load model
        trainer_config = TrainerConfig()

        with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
            compute_dtype = jnp.float32
            converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
            parameter_axis_mapping = trainer_config.parameter_axis_mapping

            print("\n=== Loading Levanter model ===")
            lev_model = converter.load_pretrained(
                LlavaOnevisionModel,
                ref=model_name,
                config=config,
                axis_mapping=parameter_axis_mapping,
                dtype=compute_dtype,
                resize_vocab_to_match_tokenizer=False,
            )

            # Get example shape info
            all_examples = tree_cache.get_batch_sync(list(range(min(cache_len, 10))))
            max_num_patches = max(ex["pixel_values"].shape[0] for ex in all_examples)
            first_ex = all_examples[0]
            seq_len = first_ex["input_ids"].shape[0]

            # Create axes
            Pos = hax.Axis("position", seq_len)
            NumPatches = hax.Axis("num_patches", max_num_patches)
            Channels = hax.Axis("channels", 3)
            Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
            Width = hax.Axis("width", first_ex["pixel_values"].shape[3])

            # Calculate NumImageTokens for unpad_indices support
            vision_feature_height = config.vision_config.image_size // config.vision_config.patch_size
            features_per_patch = vision_feature_height * vision_feature_height
            max_image_tokens = max_num_patches * features_per_patch
            NumImageTokens = hax.Axis("num_image_tokens", max_image_tokens)

            # Create dataloader (uses TreeCache directly)
            batch_size = min(4, cache_len)
            axis_resources = trainer_config.compute_axis_mapping

            from jax._src.mesh import get_concrete_mesh

            mesh = get_concrete_mesh()

            print("\n=== Creating ImageDataLoader ===")
            loader = ImageDataLoader(
                data=tree_cache,
                batch_size=batch_size,
                Pos=Pos,
                NumPatches=NumPatches,
                Channels=Channels,
                Height=Height,
                Width=Width,
                axis_resources=axis_resources,
                mesh=mesh,
                max_buffered_batches=0,
                NumImageTokens=NumImageTokens,
                allow_nondivisible_batch_size=True,
            )

            # Get batch and compute loss
            batch_iter = iter(loader)
            batch = next(batch_iter)

            @hax.named_jit
            def compute_loss_fn(model, batch):
                return compute_vlm_loss(model, batch, key=None)

            print("\n=== Computing loss ===")
            loss = compute_loss_fn(lev_model, batch)
            loss_value = float(loss.scalar())

            print("\n=== Loss Computation Result ===")
            print(f"  Batch size: {batch_size}")
            print(f"  Sequence length: {seq_len}")
            print(f"  Max patches: {max_num_patches}")
            print(f"  Loss: {loss_value:.6f}")

            # Verify loss is reasonable (not NaN, not too large)
            assert not np.isnan(loss_value), "Loss is NaN"
            assert loss_value < 100.0, f"Loss too large: {loss_value}"
            assert loss_value > 0.0, f"Loss should be positive: {loss_value}"

            print("\n Loss consistency test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_gradient_consistency():
    """
    Test 2: Verify that Levanter's VLM gradients flow correctly end-to-end with real parquet data.

    Uses ImageDataLoader to load batched data from parquet file, matching the actual training pipeline.
    Verifies gradients reach all model components (vision tower, projector, language model).
    """
    from levanter.data.loader import ImageDataLoader
    from levanter.data.image import ProcessedImageCache, ConversationUrlDataSource
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from levanter.store.cache import CacheOptions

    # Import custom processor for padding support
    from levanter.data.image import create_custom_processor, DEFAULT_IMAGE_GRID_PINPOINTS

    model_name = MODEL_NAME

    # Create custom processor with proper anyres configuration (do_pad=True for Levanter)
    processor = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=DEFAULT_IMAGE_GRID_PINPOINTS)
    # Set vision_aspect_ratio to enable max_num_patches extraction
    processor.image_processor.vision_aspect_ratio = f"anyres_max_{len(DEFAULT_IMAGE_GRID_PINPOINTS)}"

    max_length = MAX_LENGTH

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Create data source from parquet file
        source = ConversationUrlDataSource([parquet_path], messages_key="messages", images_key="images")

        # Build cache using ProcessedImageCache.build_or_load with custom processor
        print("\n=== Building cache from parquet data ===")
        cache = ProcessedImageCache.build_or_load(
            cache_dir=tmpdir,
            source=source,
            processor=processor,
            max_length=max_length,
            padding=True,
            messages_key="messages",
            images_key="images",
            cache_options=CacheOptions.default(),
        )
        tree_cache = cache.cache
        cache_len = len(tree_cache)
        print(f"  Cache length: {cache_len}")

        if cache_len < 2:
            raise ValueError("Not enough examples cached")

        # Load model config (enable flash attention + gradient checkpointing to save memory)
        config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)

        # Configure trainer with data parallelism
        trainer_config = TrainerConfig(
            per_device_parallelism=1,  # 1 sample per device
        )

        with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
            compute_dtype = jnp.float32  # Must use float32 for numerical accuracy
            converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
            parameter_axis_mapping = trainer_config.parameter_axis_mapping

            print("\n=== Loading Levanter model ===")
            print(f"  Data axis size: {trainer_config.data_axis_size}")
            print(f"  FSDP axis: {trainer_config.fsdp_axis}")
            lev_model = converter.load_pretrained(
                LlavaOnevisionModel,
                ref=model_name,
                config=config,
                axis_mapping=parameter_axis_mapping,
                dtype=compute_dtype,
                resize_vocab_to_match_tokenizer=False,
            )

            # Get example shape info
            all_examples = tree_cache.get_batch_sync(list(range(min(cache_len, 10))))
            max_num_patches = max(ex["pixel_values"].shape[0] for ex in all_examples)
            first_ex = all_examples[0]
            seq_len = first_ex["input_ids"].shape[0]

            # Create axes
            Pos = hax.Axis("position", seq_len)
            NumPatches = hax.Axis("num_patches", max_num_patches)
            Channels = hax.Axis("channels", 3)
            Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
            Width = hax.Axis("width", first_ex["pixel_values"].shape[3])

            # Calculate NumImageTokens for unpad_indices support
            vision_feature_height = config.vision_config.image_size // config.vision_config.patch_size
            features_per_patch = vision_feature_height * vision_feature_height
            max_image_tokens = max_num_patches * features_per_patch
            NumImageTokens = hax.Axis("num_image_tokens", max_image_tokens)

            # Use batch_size=1 to reduce memory for logits computation
            # (logits = batch * seq * vocab = 8 * 8192 * 152000 * 4 bytes = 37GB is too large)
            batch_size = 1
            axis_resources = trainer_config.compute_axis_mapping

            from jax._src.mesh import get_concrete_mesh

            mesh = get_concrete_mesh()

            print("\n=== Creating ImageDataLoader ===")
            print(f"  Batch size: {batch_size}")
            loader = ImageDataLoader(
                data=tree_cache,
                batch_size=batch_size,
                Pos=Pos,
                NumPatches=NumPatches,
                Channels=Channels,
                Height=Height,
                Width=Width,
                axis_resources=axis_resources,
                mesh=mesh,
                max_buffered_batches=0,
                NumImageTokens=NumImageTokens,
                allow_nondivisible_batch_size=True,
            )

            # Get batch and compute gradients
            batch_iter = iter(loader)
            batch = next(batch_iter)

            def compute_loss_for_grad(model):
                loss = compute_vlm_loss(model, batch, key=None)
                return loss.scalar()

            print("\n=== Computing gradients ===")
            lev_loss, lev_grads = eqx.filter_value_and_grad(compute_loss_for_grad)(lev_model)
            lev_loss_value = float(lev_loss)

            print("\n=== Gradient Computation Result ===")
            print(f"  Batch size: {batch_size}")
            print(f"  Loss: {lev_loss_value:.6f}")

            # Verify loss is reasonable
            assert not np.isnan(lev_loss_value), "Loss is NaN"
            assert lev_loss_value < 100.0, f"Loss too large: {lev_loss_value}"
            assert lev_loss_value > 0.0, f"Loss should be positive: {lev_loss_value}"

            # Convert gradients to state dict for analysis
            lev_grad_dict = hax.state_dict.to_torch_compatible_state_dict(lev_grads)

            # Check gradients exist for all components
            has_vision_grads = any("vision_tower" in k for k in lev_grad_dict.keys() if lev_grad_dict[k] is not None)
            has_projector_grads = any(
                "multi_modal_projector" in k for k in lev_grad_dict.keys() if lev_grad_dict[k] is not None
            )
            has_lm_grads = any("language_model" in k for k in lev_grad_dict.keys() if lev_grad_dict[k] is not None)

            print("\n=== Gradient Flow Verification ===")
            print(f"  Vision tower has gradients: {has_vision_grads}")
            print(f"  Projector has gradients: {has_projector_grads}")
            print(f"  Language model has gradients: {has_lm_grads}")

            # Verify gradient norms are reasonable (not zero, not exploding)
            lm_head_grad = lev_grad_dict.get("language_model.lm_head.weight", None)
            if lm_head_grad is not None:
                lm_head_norm = np.linalg.norm(lm_head_grad)
                print(f"  LM head gradient norm: {lm_head_norm:.6f}")
                assert lm_head_norm > 0.0, "LM head gradient is zero"
                assert lm_head_norm < 1e6, f"LM head gradient exploded: {lm_head_norm}"

            proj_grad = lev_grad_dict.get("multi_modal_projector.linear_1.weight", None)
            if proj_grad is not None:
                proj_norm = np.linalg.norm(proj_grad)
                print(f"  Projector L1 gradient norm: {proj_norm:.6f}")
                assert proj_norm > 0.0, "Projector gradient is zero"
                assert proj_norm < 1e6, f"Projector gradient exploded: {proj_norm}"

            assert has_vision_grads, "Vision tower should have gradients"
            assert has_projector_grads, "Projector should have gradients"
            assert has_lm_grads, "Language model should have gradients"

            print("\n Gradient consistency test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_gradient_flow():
    """
    Test 3: Verify gradients flow correctly through all VLM components.

    This test ensures that:
    1. Gradients reach the vision encoder (not just text model)
    2. Gradients pass through the projector
    3. No gradient explosion or vanishing
    """
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    model_name = MODEL_NAME

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Load test data using unified prepare_test_data()
        print("\n=== Loading first test sample ===")
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[0],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=9,
            grid_pinpoints=DEFAULT_GRID_PINPOINTS,
        )
        pair = test_pairs[0]

    # Load Levanter model
    # Load model config (enable flash attention + gradient checkpointing to save memory)
    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)

    # Configure trainer with data parallelism
    trainer_config = TrainerConfig(
        per_device_parallelism=1,  # 1 sample per device
    )

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=lev_config,
            axis_mapping=parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        # Prepare tensors using create_lev_jax_tensors (includes loss_mask)
        jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=8)

        from levanter.data.image import ImageTextExample as ImgTextEx

        batch_example = ImgTextEx(
            pixel_values=jax_tensors.pixel_values,
            input_ids=jax_tensors.input_ids,
            loss_mask=jax_tensors.loss_mask,
            grid_mask=jax_tensors.grid_mask,
            unpad_indices=jax_tensors.unpad_indices,
        )

        def compute_loss(model):
            loss = compute_vlm_loss(model, batch_example, key=None)
            return loss.scalar()

        _, grads = eqx.filter_value_and_grad(compute_loss)(lev_model)

        # Convert to dict for analysis
        grad_dict = hax.state_dict.to_torch_compatible_state_dict(grads)

        print("\n=== Gradient Flow Analysis ===")

        # Analyze gradients by component
        vision_grad_norms = []
        projector_grad_norms = []
        lm_grad_norms = []

        for key, grad in grad_dict.items():
            if grad is None:
                continue
            grad_norm = float(np.linalg.norm(grad))

            if "vision_tower" in key:
                vision_grad_norms.append((key, grad_norm))
            elif "multi_modal_projector" in key:
                projector_grad_norms.append((key, grad_norm))
            elif "language_model" in key:
                lm_grad_norms.append((key, grad_norm))

        # Report statistics
        print(f"\nVision Tower gradients ({len(vision_grad_norms)} params with grads):")
        if vision_grad_norms:
            norms = [n for _, n in vision_grad_norms]
            print(f"  Min norm: {min(norms):.6e}")
            print(f"  Max norm: {max(norms):.6e}")
            print(f"  Mean norm: {np.mean(norms):.6e}")

        print(f"\nProjector gradients ({len(projector_grad_norms)} params with grads):")
        if projector_grad_norms:
            norms = [n for _, n in projector_grad_norms]
            print(f"  Min norm: {min(norms):.6e}")
            print(f"  Max norm: {max(norms):.6e}")
            print(f"  Mean norm: {np.mean(norms):.6e}")

        print(f"\nLanguage Model gradients ({len(lm_grad_norms)} params with grads):")
        if lm_grad_norms:
            norms = [n for _, n in lm_grad_norms]
            print(f"  Min norm: {min(norms):.6e}")
            print(f"  Max norm: {max(norms):.6e}")
            print(f"  Mean norm: {np.mean(norms):.6e}")

        # Assertions
        assert len(vision_grad_norms) > 0, "Vision tower should have gradients"
        assert len(projector_grad_norms) > 0, "Projector should have gradients"
        assert len(lm_grad_norms) > 0, "Language model should have gradients"

        # Check for gradient explosion
        all_norms = [n for _, n in vision_grad_norms + projector_grad_norms + lm_grad_norms]
        max_norm = max(all_norms)
        assert max_norm < 1e6, f"Gradient explosion detected: max norm = {max_norm}"

        # Check for gradient vanishing in vision tower
        vision_mean_norm = np.mean([n for _, n in vision_grad_norms])
        assert vision_mean_norm > 1e-10, f"Vision tower gradients too small: {vision_mean_norm}"

        print("\nPASS: Gradient flow test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_training_step_reproducibility():
    """
    Test 4: Verify that training steps are reproducible with the same seed.

    This test ensures that:
    1. Same random seed produces identical results
    2. Gradient computation is deterministic
    3. Loss values match exactly across runs
    """
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    model_name = MODEL_NAME

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Load test data using unified prepare_test_data()
        print("\n=== Loading first test sample ===")
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[0],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=9,
            grid_pinpoints=DEFAULT_GRID_PINPOINTS,
        )
        pair = test_pairs[0]

    # Load model config (enable flash attention + gradient checkpointing to save memory)
    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)

    # Configure trainer with data parallelism
    trainer_config = TrainerConfig(
        per_device_parallelism=1,  # 1 sample per device
    )

    def run_training_step():
        """Run a single training step and return loss + gradients."""
        with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
            converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)

            model = converter.load_pretrained(
                LlavaOnevisionModel,
                ref=model_name,
                config=lev_config,
                axis_mapping=trainer_config.parameter_axis_mapping,
                dtype=jnp.float32,
                resize_vocab_to_match_tokenizer=False,
            )

            # Create JAX tensors using unified API
            jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=8)

            from levanter.data.image import ImageTextExample as ImgTextEx

            batch_example = ImgTextEx(
                pixel_values=jax_tensors.pixel_values,
                input_ids=jax_tensors.input_ids,
                loss_mask=jax_tensors.loss_mask,
                grid_mask=jax_tensors.grid_mask,
                unpad_indices=jax_tensors.unpad_indices,
            )

            def compute_loss(model):
                loss = compute_vlm_loss(model, batch_example, key=None)
                return loss.scalar()

            # Compute gradients
            loss, grads = eqx.filter_value_and_grad(compute_loss)(model)

            # Get gradient norms for comparison
            grad_dict = hax.state_dict.to_torch_compatible_state_dict(grads)
            sample_grad_key = "language_model.lm_head.weight"
            sample_grad = grad_dict.get(sample_grad_key, None)

            return float(loss), sample_grad

    print("\n=== Training Reproducibility Test ===")

    # Run training step twice
    loss1, grads1 = run_training_step()
    loss2, grads2 = run_training_step()

    print(f"\nRun 1 loss: {loss1:.10f}")
    print(f"Run 2 loss: {loss2:.10f}")
    print(f"Loss difference: {abs(loss1 - loss2):.2e}")

    # Losses should be identical (deterministic computation)
    assert loss1 == loss2, f"Losses not identical: {loss1} vs {loss2}"

    # Gradients should be identical
    if grads1 is not None and grads2 is not None:
        grad_diff = np.max(np.abs(grads1 - grads2))
        print(f"Max gradient difference: {grad_diff:.2e}")
        assert grad_diff == 0.0, f"Gradients not identical: max diff = {grad_diff}"

    print("\nPASS: Training reproducibility test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_loss_mask_correctness():
    """
    Test 5: Verify that loss masking correctly excludes user prompts.

    This test ensures that:
    1. Loss is only computed on assistant responses (labels != -100)
    2. Image tokens and user prompts are properly masked
    3. The mask is correctly shifted for next-token prediction
    """
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    model_name = MODEL_NAME

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Load test data using unified prepare_test_data()
        print("\n=== Loading first test sample ===")
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[0],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=9,
            grid_pinpoints=DEFAULT_GRID_PINPOINTS,
        )
        pair = test_pairs[0]

    # Get labels from processed data
    labels_np = np.array(pair.lev.labels)

    print("\n=== Loss Mask Analysis ===")

    # Analyze the labels
    total_positions = len(labels_np)
    masked_positions = np.sum(labels_np == -100)
    unmasked_positions = total_positions - masked_positions

    print(f"Total positions: {total_positions}")
    print(f"Masked positions (labels=-100): {masked_positions} ({100*masked_positions/total_positions:.1f}%)")
    print(f"Unmasked positions (compute loss): {unmasked_positions} ({100*unmasked_positions/total_positions:.1f}%)")

    # Verify that unmasked positions exist
    assert unmasked_positions > 0, "No unmasked positions found - training would have no signal!"

    # Load model config (enable flash attention + gradient checkpointing to save memory)
    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)

    # Configure trainer with data parallelism
    trainer_config = TrainerConfig(
        per_device_parallelism=1,  # 1 sample per device
    )

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=lev_config,
            axis_mapping=trainer_config.parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        # Create JAX tensors using unified API
        jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=8)

        from levanter.data.image import ImageTextExample as ImgTextEx

        batch_example = ImgTextEx(
            pixel_values=jax_tensors.pixel_values,
            input_ids=jax_tensors.input_ids,
            loss_mask=jax_tensors.loss_mask,
            grid_mask=jax_tensors.grid_mask,
            unpad_indices=jax_tensors.unpad_indices,
        )

        # Verify loss_mask has correct number of valid positions
        # (without computing full logits which would OOM)
        Pos_axis = batch_example.input_ids.resolve_axis("position")
        loss_mask_shifted = hax.roll(batch_example.loss_mask, -1, Pos_axis)
        num_valid_per_sample = hax.sum(loss_mask_shifted, axis=Pos_axis)

        # Each sample in batch is duplicated, so check first sample
        actual_valid = int(num_valid_per_sample.array[0])
        expected_valid = unmasked_positions

        print("\nLoss mask stats:")
        print(f"  Number of valid positions per sample: {actual_valid}")
        print(f"  Expected valid positions: ~{expected_valid}")

        # Compute loss using compute_vlm_loss (memory efficient)
        avg_loss = compute_vlm_loss(lev_model, batch_example, key=None)
        print(f"  Average loss: {float(avg_loss.scalar()):.6f}")

        # Allow some tolerance due to edge effects
        assert (
            abs(actual_valid - expected_valid) <= 2
        ), f"Valid position mismatch: expected ~{expected_valid}, got {actual_valid}"

        print("\nPASS: Loss mask correctness test passed!")


# =====================
# Text-only and Mixed Batch Tests
# =====================

# Test image loaded from HuggingFace dataset


@skip_if_no_torch
def test_text_only_conversation():
    """Test BatchImageProcessor with text-only conversations (no images)."""
    from transformers import AutoProcessor
    from levanter.data.image import BatchImageProcessor, ImageTextExample
    from haliax import NamedArray, Axis

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bp = BatchImageProcessor(processor, max_length=2048, padding=True)

    # Text-only conversation
    text_only_messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]},
    ]

    batch = [{"messages": text_only_messages, "images": []}]
    results = bp(batch)

    assert len(results) == 1
    result = results[0]

    # Text-only should have None pixel_values and image_sizes
    assert result["pixel_values"] is None, "Text-only should have None pixel_values"
    assert result["image_sizes"] is None, "Text-only should have None image_sizes"
    assert result["input_ids"].shape == (2048,), "input_ids should be padded to max_length"
    assert result["labels"].shape == (2048,), "labels should be padded to max_length"

    # Check that labels have some non-ignored values (assistant response)
    non_ignore_count = np.sum(result["labels"] != -100)
    assert non_ignore_count > 0, "Labels should have some non-ignored values for assistant response"

    # Test ImageTextExample with text-only
    Position = Axis("position", 2048)
    input_ids_named = NamedArray(result["input_ids"], (Position,))
    labels_named = NamedArray(result["labels"], (Position,))

    example = ImageTextExample.init(
        pixel_values=None,
        input_ids=input_ids_named,
        labels=labels_named,
    )

    assert example.pixel_values is None, "ImageTextExample should have None pixel_values"
    assert example.loss_mask is not None, "ImageTextExample should have loss_mask"

    print("PASS: Text-only conversation test passed!")


@skip_if_no_torch
def test_mixed_batch():
    """Test BatchImageProcessor with mixed batch (some with images, some text-only)."""
    from transformers import AutoProcessor
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bp = BatchImageProcessor(processor, max_length=MAX_LENGTH, padding=True)

    # Load test image
    test_image = get_single_image()

    # Text-only conversation
    text_only_messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "2 + 2 equals 4."}]},
    ]

    # Image conversation
    image_messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see in this image?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "I see a colorful image."}]},
    ]

    # Mixed batch: text-only first, then image
    mixed_batch = [
        {"messages": text_only_messages, "images": []},
        {"messages": image_messages, "images": [test_image]},
    ]

    results = bp(mixed_batch)
    assert len(results) == 2

    # First example: text-only
    assert results[0]["pixel_values"] is None, "First example (text-only) should have None pixel_values"
    assert results[0]["image_sizes"] is None, "First example (text-only) should have None image_sizes"

    # Second example: with image
    assert results[1]["pixel_values"] is not None, "Second example should have pixel_values"
    assert results[1]["image_sizes"] is not None, "Second example should have image_sizes"
    assert results[1]["pixel_values"].ndim == 4, "pixel_values should be 4D (num_patches, C, H, W)"

    # Reverse order: image first, then text-only
    reverse_batch = [
        {"messages": image_messages, "images": [test_image]},
        {"messages": text_only_messages, "images": []},
    ]

    results = bp(reverse_batch)
    assert len(results) == 2

    # First example: with image
    assert results[0]["pixel_values"] is not None, "First example should have pixel_values"
    assert results[0]["image_sizes"] is not None, "First example should have image_sizes"

    # Second example: text-only
    assert results[1]["pixel_values"] is None, "Second example (text-only) should have None pixel_values"
    assert results[1]["image_sizes"] is None, "Second example (text-only) should have None image_sizes"

    print("PASS: Mixed batch test passed!")


@skip_if_no_torch
def test_multiround_image_input():
    """Test BatchImageProcessor with multi-turn conversations containing multiple images."""
    from transformers import AutoProcessor
    from PIL import Image
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bp = BatchImageProcessor(processor, max_length=MAX_LENGTH, padding=True)

    # Use small images to avoid truncation issues
    test_image = Image.new("RGB", (100, 100), color="red")
    test_image_2 = Image.new("RGB", (100, 100), color="blue")

    # Multi-turn conversation with multiple images
    multi_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "This is the first image. What do you see?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I see a colorful picture in the first image."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Now look at this second image. How is it different?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The second image appears smaller and has a different composition."}],
        },
    ]

    batch = [{"messages": multi_image_messages, "images": [test_image, test_image_2]}]
    results = bp(batch)

    assert len(results) == 1
    result = results[0]

    # Should have pixel_values for multiple images
    assert result["pixel_values"] is not None, "Multi-image should have pixel_values"
    assert result["image_sizes"] is not None, "Multi-image should have image_sizes"

    # Check labels - should have assistant responses
    non_ignore_count = np.sum(result["labels"] != -100)
    assert non_ignore_count > 0, "Labels should have non-ignored values for assistant responses"

    # The assistant responses should include both turns
    # Check that we have reasonable number of non-ignored tokens
    print(f"Non-ignored label count: {non_ignore_count}")

    print("PASS: Multi-round image input test passed!")


@skip_if_no_torch
def test_multiround_mixed_conversation():
    """Test multi-turn conversation mixing text-only and image turns."""
    from transformers import AutoProcessor
    from PIL import Image
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bp = BatchImageProcessor(processor, max_length=MAX_LENGTH, padding=True)

    # Use small image to avoid truncation issues
    test_image = Image.new("RGB", (100, 100), color="green")

    # Multi-turn with text first, then image
    mixed_turns_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello! Can you help me analyze some images?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Of course! Please share the images you'd like me to analyze."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Here's an image. What can you tell me about it?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This appears to be a detailed photograph with various elements."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Can you describe the colors?"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The image contains a rich palette of colors including various shades."}
            ],
        },
    ]

    batch = [{"messages": mixed_turns_messages, "images": [test_image]}]
    results = bp(batch)

    assert len(results) == 1
    result = results[0]

    # Should have pixel_values for the image
    assert result["pixel_values"] is not None, "Should have pixel_values"
    assert result["image_sizes"] is not None, "Should have image_sizes"

    # Check labels - should have all assistant responses (3 turns)
    non_ignore_count = np.sum(result["labels"] != -100)
    assert non_ignore_count > 0, "Labels should have non-ignored values"

    # All 3 assistant turns should be included
    # We should have more non-ignored tokens than a single turn
    print(f"Non-ignored label count: {non_ignore_count}")
    assert non_ignore_count > 10, "Should have substantial non-ignored tokens for 3 assistant turns"

    print("PASS: Multi-round mixed conversation test passed!")


@skip_if_no_torch
def test_labels_mask_correctness_text_only():
    """Verify that _create_labels correctly masks text-only conversations."""
    from transformers import AutoProcessor
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    bp = BatchImageProcessor(processor, max_length=2048, padding=True)

    # Text-only conversation with known content
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is Python?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Python is a programming language."}]},
    ]

    batch = [{"messages": messages, "images": []}]
    results = bp(batch)
    result = results[0]

    # Decode and verify
    input_ids = result["input_ids"]
    labels = result["labels"]

    # Count non-ignored labels
    non_ignore_indices = np.where(labels != -100)[0]
    print(f"Non-ignored positions: {len(non_ignore_indices)}")

    # Verify that only assistant content is included
    # The non-ignored tokens should correspond to assistant content + <|im_end|>
    assert len(non_ignore_indices) > 0, "Should have some non-ignored labels"

    # Decode the non-ignored tokens
    non_ignore_tokens = input_ids[non_ignore_indices]
    decoded = tokenizer.decode(non_ignore_tokens, skip_special_tokens=False)
    print(f"Non-ignored content: {decoded}")

    # The decoded content should contain the assistant response
    assert "Python" in decoded or "programming" in decoded, "Non-ignored content should include assistant response"

    print("PASS: Labels mask correctness (text-only) test passed!")


@skip_if_no_torch
def test_labels_mask_correctness_with_image():
    """Verify that _create_labels correctly masks conversations with images."""
    from transformers import AutoProcessor
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    bp = BatchImageProcessor(processor, max_length=MAX_LENGTH, padding=True)

    # Load test image
    test_image = get_single_image()

    # Conversation with image
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A beautiful photograph."}]},
    ]

    batch = [{"messages": messages, "images": [test_image]}]
    results = bp(batch)
    result = results[0]

    input_ids = result["input_ids"]
    labels = result["labels"]

    # Count non-ignored labels
    non_ignore_indices = np.where(labels != -100)[0]
    print(f"Non-ignored positions: {len(non_ignore_indices)}")

    # Non-ignored tokens should be assistant content
    non_ignore_tokens = input_ids[non_ignore_indices]
    decoded = tokenizer.decode(non_ignore_tokens, skip_special_tokens=False)
    print(f"Non-ignored content: {decoded}")

    # Should contain assistant response
    assert "beautiful" in decoded or "photograph" in decoded, "Non-ignored content should include assistant response"

    # Image tokens should NOT be in the non-ignored set
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id != tokenizer.unk_token_id:
        assert image_token_id not in non_ignore_tokens, "Image tokens should be masked"

    print("PASS: Labels mask correctness (with image) test passed!")


@skip_if_no_torch
def test_replace_tokenizer_with_qwen3():
    """Test that _replace_tokenizer correctly replaces processor tokenizer with Qwen3 tokenizer."""
    from transformers import AutoProcessor, AutoTokenizer
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Store original tokenizer reference
    original_tokenizer = processor.tokenizer

    # Create BatchImageProcessor with LLM tokenizer
    _ = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # Verify tokenizer was replaced
    assert processor.tokenizer is llm_tokenizer, "Tokenizer should be replaced"
    assert id(processor.tokenizer) != id(original_tokenizer), "Tokenizer ID should be different"
    assert id(processor.tokenizer) == id(llm_tokenizer), "Tokenizer should be the LLM tokenizer"

    print("PASS: Tokenizer replacement test passed!")


@skip_if_no_torch
def test_replace_tokenizer_qwen3_thinking_tokens():
    """Test that replaced Qwen3 tokenizer can correctly encode Qwen3-specific thinking tokens.

    Qwen3 has special <think> and </think> tokens (IDs 151667 and 151668) that are not
    present in the original processor tokenizer. After replacement, these should be
    encoded as single tokens instead of being split into multiple tokens.
    """
    from transformers import AutoProcessor, AutoTokenizer
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Test encoding BEFORE replacement
    text_with_thinking = "<think>Let me think...</think>Answer is 42."
    original_encoding = processor.tokenizer.encode(text_with_thinking, add_special_tokens=False)

    # The original tokenizer should NOT have <think> and </think> as single tokens
    # It will split them into multiple tokens like ['<', 'think', '>']
    think_token_id = 151667  # Qwen3's <think> token ID
    end_think_token_id = 151668  # Qwen3's </think> token ID

    assert (
        think_token_id not in original_encoding
    ), f"Original tokenizer should not have <think> as single token, got: {original_encoding}"
    assert (
        end_think_token_id not in original_encoding
    ), f"Original tokenizer should not have </think> as single token, got: {original_encoding}"

    # Create BatchImageProcessor with LLM tokenizer (this replaces the tokenizer)
    _ = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # Test encoding AFTER replacement
    new_encoding = processor.tokenizer.encode(text_with_thinking, add_special_tokens=False)

    # After replacement, <think> and </think> should be single tokens
    assert (
        think_token_id in new_encoding
    ), f"Replaced tokenizer should have <think> as single token (ID {think_token_id}), got: {new_encoding}"
    assert (
        end_think_token_id in new_encoding
    ), f"Replaced tokenizer should have </think> as single token (ID {end_think_token_id}), got: {new_encoding}"

    # Verify the token count is different (fewer tokens after replacement)
    assert len(new_encoding) < len(original_encoding), (
        f"Replaced tokenizer should produce fewer tokens: "
        f"original={len(original_encoding)}, new={len(new_encoding)}"
    )

    print(f"Original encoding ({len(original_encoding)} tokens): {original_encoding}")
    print(f"New encoding ({len(new_encoding)} tokens): {new_encoding}")
    print("PASS: Qwen3 thinking tokens test passed!")


@skip_if_no_torch
def test_replace_tokenizer_critical_tokens_match():
    """Test that critical special tokens match between processor and Qwen3 tokenizer."""
    from transformers import AutoProcessor, AutoTokenizer
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Verify critical tokens match before replacement (should pass without assertion error)
    critical_tokens = ["<|im_start|>", "<|im_end|>", "assistant", "user", "system"]

    for token in critical_tokens:
        proc_id = processor.tokenizer.convert_tokens_to_ids(token)
        llm_id = llm_tokenizer.convert_tokens_to_ids(token)
        assert proc_id == llm_id, f"Token '{token}' ID mismatch: processor={proc_id}, llm={llm_id}"
        print(f"  {token}: {proc_id} OK")

    # Create BatchImageProcessor - should not raise any assertion errors
    _ = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # Verify vocab size matches
    assert processor.tokenizer.vocab_size == llm_tokenizer.vocab_size

    print("PASS: Critical tokens match test passed!")


@skip_if_no_torch
def test_replace_tokenizer_processing_with_thinking():
    """Test that BatchImageProcessor works correctly with Qwen3 thinking tokens in conversation."""
    from transformers import AutoProcessor, AutoTokenizer
    from PIL import Image
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Create processor with Qwen3 tokenizer
    bp = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # Create a small test image
    test_image = Image.new("RGB", (100, 100), color="blue")

    # Conversation with Qwen3 thinking tokens in assistant response
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>Let me analyze this image carefully...</think>I see a blue square."}
            ],
        },
    ]

    batch = [{"messages": messages, "images": [test_image]}]
    results = bp(batch)

    assert len(results) == 1
    result = results[0]

    # Verify the output structure
    assert result["pixel_values"] is not None
    assert result["input_ids"] is not None
    assert result["labels"] is not None

    # Verify thinking tokens are in the input_ids
    input_ids = result["input_ids"]
    think_token_id = 151667
    end_think_token_id = 151668

    assert think_token_id in input_ids, f"<think> token should be in input_ids: {input_ids[:50]}..."
    assert end_think_token_id in input_ids, "</think> token should be in input_ids"

    # Verify labels have non-ignored values (assistant response should be included)
    non_ignore_count = np.sum(result["labels"] != -100)
    assert non_ignore_count > 0, "Labels should have non-ignored values for assistant response"

    print(f"Input IDs length: {len(input_ids)}")
    print(f"Non-ignored labels count: {non_ignore_count}")
    print("PASS: Processing with thinking tokens test passed!")


@skip_if_no_torch
def test_replace_tokenizer_uses_qwen3_image_token():
    """Test that processor uses Qwen3's <|image_pad|> token after tokenizer replacement."""
    from transformers import AutoProcessor, AutoTokenizer
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Before replacement: processor uses <image> token
    assert processor.image_token == "<image>"
    old_processor_image_id = processor.image_token_id
    print(f"Original: image_token='{processor.image_token}', id={old_processor_image_id}")

    # Qwen3 tokenizer has <|image_pad|> token pre-defined
    qwen3_image_token = "<|image_pad|>"
    qwen3_image_id = llm_tokenizer.convert_tokens_to_ids(qwen3_image_token)
    assert qwen3_image_id is not None, "Qwen3 should have <|image_pad|> token"
    print(f"Qwen3 <|image_pad|> ID: {qwen3_image_id}")

    # Create BatchImageProcessor with Qwen3 tokenizer
    _ = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # After replacement: processor should use Qwen3's <|image_pad|> token
    assert (
        processor.image_token == qwen3_image_token
    ), f"Processor should use Qwen3's image token: got '{processor.image_token}'"
    assert (
        processor.image_token_id == qwen3_image_id
    ), f"Processor image_token_id should match Qwen3: got {processor.image_token_id}"
    print(f"Updated: image_token='{processor.image_token}', id={processor.image_token_id}")

    # Same for video token
    assert processor.video_token == "<|video_pad|>"
    qwen3_video_id = llm_tokenizer.convert_tokens_to_ids("<|video_pad|>")
    assert processor.video_token_id == qwen3_video_id

    # Verify encoding works correctly with the new image token
    text_with_image = f"Hello {qwen3_image_token} world"
    encoded = processor.tokenizer.encode(text_with_image, add_special_tokens=False)
    assert qwen3_image_id in encoded, f"<|image_pad|> token should be in encoded output: {encoded}"

    print("PASS: Qwen3 image token test passed!")


@skip_if_no_torch
def test_get_token_ids_and_update_model_config():
    """Test that get_token_ids returns correct values and can update model config."""
    from transformers import AutoProcessor, AutoTokenizer
    from levanter.data.image import BatchImageProcessor
    from levanter.models.qwen import Qwen3Config

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Original processor token IDs
    original_image_token_id = processor.image_token_id
    original_video_token_id = processor.video_token_id
    print(f"Original image_token_id: {original_image_token_id}")
    print(f"Original video_token_id: {original_video_token_id}")

    # Create BatchImageProcessor with Qwen3 tokenizer
    bp = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # Get updated token IDs
    token_ids = bp.get_token_ids()
    print(f"New image_token_id: {token_ids['image_token_id']}")
    print(f"New video_token_id: {token_ids['video_token_id']}")
    print(f"vocab_size: {token_ids['vocab_size']}")

    # Token IDs should have changed (new tokens added to Qwen3 tokenizer)
    assert (
        token_ids["image_token_id"] != original_image_token_id
    ), f"image_token_id should change: original={original_image_token_id}, new={token_ids['image_token_id']}"

    # Create a sample model config
    vision_config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    text_config = Qwen3Config(
        hidden_dim=128,
        intermediate_dim=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
    )
    model_config = LlavaOnevisionConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=original_image_token_id,  # Original value
        video_token_index=original_video_token_id,
    )

    # Update model config with new token IDs
    updated_config = model_config.with_token_ids(
        image_token_id=token_ids["image_token_id"],
        video_token_id=token_ids["video_token_id"],
    )

    # Verify the config was updated
    assert updated_config.image_token_index == token_ids["image_token_id"]
    assert updated_config.video_token_index == token_ids["video_token_id"]
    print(f"Updated model config image_token_index: {updated_config.image_token_index}")
    print(f"Updated model config video_token_index: {updated_config.video_token_index}")

    # Original config should be unchanged (immutable)
    assert model_config.image_token_index == original_image_token_id

    print("PASS: get_token_ids and update model config test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
