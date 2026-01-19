# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
from transformers import AutoProcessor

from levanter.data.image import BatchImageProcessor, load_image
from levanter.store.cache import SerialCacheWriter
import jax
import jax.numpy as jnp

from test_image_utils import get_real_data, DEFAULT_GRID_PINPOINTS, SINGLE_PATCH_GRID_PINPOINTS
import numpy as np
import haliax as hax
from jax.sharding import Mesh


@pytest.fixture
def processor():
    return AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-si-hf")


@pytest.fixture
def dataset():
    return get_real_data()


def test_batch_image_processor(processor, dataset):
    """Test core BatchImageProcessor functionality."""
    batch_processor = BatchImageProcessor(
        processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
    )
    examples = [dataset[i] for i in range(4)]
    results = batch_processor(examples)

    assert len(results) == 4
    for result in results:
        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "image_sizes" in result
        assert "combined_mask" in result
        assert "position_ids" in result

        # Check shapes
        assert result["input_ids"].shape == (2048,), f"Expected (2048,), got {result['input_ids'].shape}"
        assert result["attention_mask"].shape == (2048,), f"Expected (2048,), got {result['attention_mask'].shape}"
        assert result["combined_mask"].shape == (2048,), f"Expected (2048,), got {result['combined_mask'].shape}"
        assert result["position_ids"].shape == (2048,), f"Expected (2048,), got {result['position_ids'].shape}"

        # pixel_values should have proper dimensions
        assert result["pixel_values"].ndim >= 3


def test_image_data_loader(processor, dataset):
    """Test ImageDataLoader with cached data."""
    from levanter.data.image import ImageDataLoader, ImageTextExample

    batch_processor = BatchImageProcessor(
        processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with SerialCacheWriter(tmpdir, batch_processor.output_exemplar) as writer:
            for i in range(8):
                example = dataset[i]
                try:
                    results = batch_processor([example])
                    writer.write_batch(results)
                except ValueError as e:
                    if "Mismatch in `image` token count" in str(e):
                        continue
                    raise

        cache = writer.result()
        cache_len = len(cache)

        if cache_len < 2:
            pytest.skip("Not enough examples cached for dataloader test")

        all_examples = cache.get_batch_sync(list(range(cache_len)))
        max_num_patches = max(ex["pixel_values"].shape[0] for ex in all_examples)
        first_ex = all_examples[0]
        seq_len = first_ex["input_ids"].shape[0]

        Pos = hax.Axis("position", seq_len)
        NumPatches = hax.Axis("num_patches", max_num_patches)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
        Width = hax.Axis("width", first_ex["pixel_values"].shape[3])

        devices = np.array(jax.devices("cpu")[:1])
        mesh = Mesh(devices, ("data",))

        batch_size = min(2, cache_len)
        axis_resources = {"batch": "data"}

        with mesh:
            loader = ImageDataLoader(
                data=cache,
                batch_size=batch_size,
                Pos=Pos,
                NumPatches=NumPatches,
                Channels=Channels,
                Height=Height,
                Width=Width,
                mesh=mesh,
                axis_resources=axis_resources,
                max_buffered_batches=0,
            )

            batch = next(iter(loader))
            assert isinstance(batch, ImageTextExample)
            assert batch.pixel_values.array.shape[0] == batch_size
            assert batch.input_ids.array.shape[0] == batch_size


def test_llava_with_image_dataloader(processor, dataset):
    """Test LLaVA OneVision model - compare HF and Levanter outputs (disable_anyres mode)."""
    jax.config.update("jax_default_matmul_precision", "float32")

    import dataclasses
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from levanter.data.image import ImageDataLoader, ImageTextExample, create_custom_processor
    from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
    from levanter.layers.attention import AttentionBackend
    from levanter.trainer import TrainerConfig
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision, AutoConfig
    import equinox as eqx

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    # Use disable_anyres mode - only base patch per image
    image_processor = processor.image_processor
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS  # Use single patch grid for disable_anyres
    patch_size = getattr(image_processor, "size", {}).get("height", 384)
    vision_feature_height = patch_size // 14
    max_num_patches = 1  # Only 1 patch per image in disable_anyres mode

    padded_processor = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=grid_pinpoints, vision_aspect_ratio="single")
    unpadded_processor = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=grid_pinpoints, vision_aspect_ratio="single")

    batch_processor = BatchImageProcessor(
        padded_processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
        grid_pinpoints=grid_pinpoints,
        patch_size=patch_size,
        vision_feature_height=vision_feature_height,
        max_num_patches=max_num_patches,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cached_dataset_indices = []
        with SerialCacheWriter(tmpdir, batch_processor.output_exemplar) as writer:
            for i in range(8):
                example = dataset[i]
                try:
                    results = batch_processor([example])
                    writer.write_batch(results)
                    cached_dataset_indices.append(i)
                except ValueError as e:
                    if "Mismatch in `image` token count" in str(e):
                        continue
                    raise

        cache = writer.result()
        cache_len = len(cache)

        if cache_len < 2:
            pytest.skip("Not enough examples cached for test")

        all_examples = cache.get_batch_sync(list(range(cache_len)))
        max_num_patches_actual = max(ex["pixel_values"].shape[0] for ex in all_examples)
        first_ex = all_examples[0]
        seq_len = first_ex["input_ids"].shape[0]

        Pos = hax.Axis("position", seq_len)
        NumPatches = hax.Axis("num_patches", max_num_patches_actual)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
        Width = hax.Axis("width", first_ex["pixel_values"].shape[3])
        features_per_patch = vision_feature_height * vision_feature_height
        max_image_tokens = max_num_patches_actual * features_per_patch
        NumImageTokens = hax.Axis("num_image_tokens", max_image_tokens)

        # Load HF model with disable_anyres configuration
        hf_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        hf_model.model.config.image_grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS
        hf_model.model.config.vision_aspect_ratio = "single"  # disable_anyres mode
        hf_model.model.image_newline = None
        hf_model.eval()

        # Load Levanter model
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config = LlavaOnevisionConfig.from_hf_config(hf_config)
        vision_config_updated = dataclasses.replace(
            config.vision_config,
            use_flash_attention=False,
            attn_backend=AttentionBackend.VANILLA,
            gradient_checkpointing=False,
        )
        text_config_updated = dataclasses.replace(
            config.text_config, attn_backend=AttentionBackend.VANILLA, gradient_checkpointing=False
        )
        config = dataclasses.replace(
            config, vision_config=vision_config_updated, text_config=text_config_updated, gradient_checkpointing=False
        )

        trainer_config = TrainerConfig()

        with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
            converter = config.hf_checkpoint_converter(ref_checkpoint=model_name)
            lev_model = converter.load_pretrained(
                LlavaOnevisionModel,
                ref=model_name,
                config=config,
                axis_mapping=trainer_config.parameter_axis_mapping,
                dtype=jnp.float32,
                resize_vocab_to_match_tokenizer=False,
            )

            batch_size = min(4, cache_len)
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
                axis_resources=trainer_config.compute_axis_mapping,
                mesh=mesh,
                max_buffered_batches=0,
                allow_nondivisible_batch_size=True,
                NumImageTokens=NumImageTokens,
            )

            batch = next(iter(loader))
            assert isinstance(batch, ImageTextExample)

            batch_input_ids = np.array(batch.input_ids.array)
            batch_pixel_values = np.array(batch.pixel_values.array)
            batch_grid_mask = np.array(batch.grid_mask.array) if batch.grid_mask is not None else None

            image_token_id = hf_model.config.image_token_index

            # Monkey-patch image_size_to_num_patches to return 1 for disable_anyres mode
            original_image_size_to_num_patches = llava_modeling.image_size_to_num_patches

            def patched_image_size_to_num_patches(*args, **kwargs):
                return 1  # Force 1 patch per image for disable_anyres

            llava_modeling.image_size_to_num_patches = patched_image_size_to_num_patches

            # HF forward pass
            hf_logits_list = []
            hf_input_ids_list = []
            hf_image_sizes_list = []
            try:
                for sample_idx in range(batch_size):
                    dataset_idx = cached_dataset_indices[sample_idx]
                    raw_example = dataset[dataset_idx]
                    messages = raw_example["messages"]
                    images = raw_example.get("images", None)
                    prompt_text = unpadded_processor.apply_chat_template(messages, add_generation_prompt=False)

                    if images is not None and len(images) > 0:
                        pil_images = [load_image(img) for img in images]
                        hf_inputs = unpadded_processor(text=prompt_text, images=pil_images, return_tensors="pt")
                        # For disable_anyres mode: truncate to base patch only
                        # HF processor outputs base + anyres patches, but we only want base patch
                        if "pixel_values" in hf_inputs and hf_inputs["pixel_values"].dim() == 5:
                            hf_inputs["pixel_values"] = hf_inputs["pixel_values"][:, 0:1, :, :, :]
                    else:
                        hf_inputs = unpadded_processor(text=prompt_text, return_tensors="pt")

                    hf_input_ids = hf_inputs["input_ids"]
                    hf_input_ids_list.append(hf_input_ids[0].numpy())
                    hf_image_sizes_list.append(hf_inputs.get("image_sizes"))

                    with torch.no_grad():
                        hf_output = hf_model(**hf_inputs)
                        hf_logits_list.append(hf_output.logits[0].numpy())
            finally:
                # Restore original function
                llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

            # Levanter forward pass
            @eqx.filter_jit
            def compute_forward_single(model, input_ids, pixel_values, grid_mask, unpad_indices):
                return model(
                    input_ids, pixel_values=pixel_values, grid_mask=grid_mask, unpad_indices=unpad_indices, key=None
                )

            lev_logits_list = []
            for sample_idx in range(batch_size):
                input_ids_np = batch_input_ids[sample_idx : sample_idx + 1]
                pixel_values_np = batch_pixel_values[sample_idx : sample_idx + 1]
                grid_mask_np = batch_grid_mask[sample_idx : sample_idx + 1] if batch_grid_mask is not None else None
                has_image = grid_mask_np is not None and grid_mask_np[0].any()

                Batch1 = hax.Axis("batch", 1)
                input_ids_lev = hax.named(jnp.array(input_ids_np, dtype=jnp.int32), (Batch1, Pos))
                pixel_values_lev = hax.named(
                    jnp.array(pixel_values_np, dtype=jnp.float32), (Batch1, NumPatches, Channels, Height, Width)
                )
                grid_mask_lev = (
                    hax.named(jnp.array(grid_mask_np, dtype=jnp.bool_), (Batch1, NumPatches))
                    if grid_mask_np is not None
                    else None
                )

                # In disable_anyres mode, unpad_indices is always None
                # (no unpadding needed for single patch per image)
                unpad_indices_lev = None

                lev_logits_sample = compute_forward_single(
                    lev_model, input_ids_lev, pixel_values_lev, grid_mask_lev, unpad_indices_lev
                )
                lev_logits_sample.array.block_until_ready()
                lev_logits_list.append(np.array(lev_logits_sample.array)[0])

            # Compare outputs
            all_correlations = []
            all_pred_match_rates = []
            for sample_idx in range(batch_size):
                hf_logit = hf_logits_list[sample_idx]
                lev_logit = lev_logits_list[sample_idx]
                min_len = min(len(hf_logit), len(lev_logit))
                hf_compare = hf_logit[:min_len]
                lev_compare = lev_logit[:min_len]

                correlation = np.corrcoef(hf_compare.flatten(), lev_compare.flatten())[0, 1]
                all_correlations.append(correlation)

                hf_preds = np.argmax(hf_compare, axis=-1)
                lev_preds = np.argmax(lev_compare, axis=-1)
                pred_match_rate = np.mean(hf_preds == lev_preds)
                all_pred_match_rates.append(pred_match_rate)

            avg_correlation = np.mean(all_correlations)
            avg_pred_match = np.mean(all_pred_match_rates)

            assert avg_correlation > 0.99, f"Average correlation too low: {avg_correlation}"
            assert avg_pred_match > 0.90, f"Average prediction match too low: {avg_pred_match}"


def test_cache_vs_streaming_data_consistency():
    """Test that cache mode and streaming mode produce consistent data.

    Note: With lazy shard loading, streaming datasets no longer report actual length
    (they return sys.maxsize for infinite iteration). We still test that data
    retrieved from both modes is consistent.
    """
    import asyncio
    import sys
    from levanter.data.image import ImageMixtureDatasetConfig, ConversationDatasetSourceConfig

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        cache_config = ImageMixtureDatasetConfig(
            cache_dir=f"{tmpdir}/cache",
            configs={
                "train": ConversationDatasetSourceConfig(
                    train_urls=[f"file://{parquet_path}"],
                    validation_urls=[f"file://{parquet_path}"],
                    cache_dir=f"{tmpdir}/cache/train",
                )
            },
            train_weights={"train": 1.0},
            processor=model_name,
            max_length=8192,
            use_cache=True,
        )

        cache_datasets = cache_config.training_sets()
        cache_dataset = list(cache_datasets.values())[0]
        cache_len = asyncio.run(cache_dataset.async_len())

        streaming_config = ImageMixtureDatasetConfig(
            cache_dir=f"{tmpdir}/streaming_cache",
            configs={
                "train": ConversationDatasetSourceConfig(
                    train_urls=[f"file://{parquet_path}"],
                    validation_urls=[f"file://{parquet_path}"],
                    cache_dir=f"{tmpdir}/streaming_cache/train",
                )
            },
            train_weights={"train": 1.0},
            processor=model_name,
            max_length=8192,
            use_cache=False,
        )

        streaming_datasets = streaming_config.training_sets()
        streaming_dataset = list(streaming_datasets.values())[0]
        streaming_len = asyncio.run(streaming_dataset.async_len())

        # Streaming dataset returns sys.maxsize for step-based infinite iteration
        assert streaming_len == sys.maxsize

        # Test that we can get batches from both modes
        num_to_compare = min(5, cache_len)
        indices = list(range(num_to_compare))
        cache_examples = asyncio.run(cache_dataset.get_batch(indices))
        streaming_examples = asyncio.run(streaming_dataset.get_batch(indices))

        assert len(cache_examples) == num_to_compare
        assert len(streaming_examples) == num_to_compare

        # Both should have valid data
        for ex in cache_examples:
            assert "input_ids" in ex
            assert "pixel_values" in ex
        for ex in streaming_examples:
            assert "input_ids" in ex
            assert "pixel_values" in ex

        # Close streaming dataset before temp dir is cleaned up
        streaming_dataset.close()


def test_streaming_dataset_basic():
    """Test StreamingImageDataset functionality.

    Note: With lazy shard loading, streaming datasets use step-based infinite
    iteration mode where is_finite() returns False and async_len() returns sys.maxsize.
    """
    import asyncio
    import sys
    from levanter.data.image import ImageMixtureDatasetConfig, ConversationDatasetSourceConfig, StreamingImageDataset

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        config = ImageMixtureDatasetConfig(
            cache_dir=f"{tmpdir}/cache",
            configs={
                "train": ConversationDatasetSourceConfig(
                    train_urls=[f"file://{parquet_path}"],
                    validation_urls=[f"file://{parquet_path}"],
                    cache_dir=f"{tmpdir}/cache/train",
                )
            },
            train_weights={"train": 1.0},
            processor=model_name,
            max_length=2048,
            use_cache=False,
        )

        datasets = config.training_sets()
        dataset = list(datasets.values())[0]
        assert isinstance(dataset, StreamingImageDataset)

        async def run_tests():
            length = await dataset.async_len()
            # Step-based mode returns sys.maxsize for infinite iteration
            assert length == sys.maxsize
            # Step-based mode is infinite (not finite)
            assert not dataset.is_finite()
            batch = await dataset.get_batch([0, 1, 2])
            assert len(batch) == 3
            for ex in batch:
                assert "input_ids" in ex
                assert "pixel_values" in ex
            return True

        result = asyncio.run(run_tests())
        assert result

        # Close dataset before temp dir is cleaned up
        dataset.close()


def test_image_data_loader_variable_length_sequences():
    """Test that ImageDataLoader handles variable-length sequences correctly.

    This verifies end-to-end padding support for input_ids, loss_mask,
    combined_mask, and position_ids when sequence lengths vary.
    """
    from levanter.data.dataset import ListAsyncDataset
    from levanter.data.image import ImageDataLoader, ImageTextExample

    # Create test data with DIFFERENT sequence lengths
    seq_lens = [100, 150, 200, 80]  # Variable lengths
    target_seq_len = 256  # Pos.size > max(seq_lens)
    num_patches = 2
    batch_size = len(seq_lens)

    test_dicts = []
    for seq_len in seq_lens:
        d = {
            "input_ids": np.arange(seq_len, dtype=np.int32),
            "attention_mask": np.ones(seq_len, dtype=np.int32),
            "loss_mask": np.ones(seq_len, dtype=np.float32),
            "pixel_values": np.random.randn(num_patches, 3, 384, 384).astype(np.float32),
            "grid_mask": np.array([True, True], dtype=np.bool_),
            "combined_mask": np.ones(seq_len, dtype=np.int32),
            "position_ids": np.arange(seq_len, dtype=np.int32),
        }
        test_dicts.append(d)

    dataset = ListAsyncDataset(test_dicts, is_complete=True)

    # Define axes (Pos.size > max sequence length)
    Pos = hax.Axis("position", target_seq_len)
    NumPatches = hax.Axis("num_patches", num_patches)
    Channels = hax.Axis("channels", 3)
    Height = hax.Axis("height", 384)
    Width = hax.Axis("width", 384)

    # Create simple CPU mesh
    devices = np.array(jax.devices("cpu")[:1])
    mesh = Mesh(devices, ("data",))
    axis_resources = {"batch": "data"}

    with mesh:
        loader = ImageDataLoader(
            data=dataset,
            batch_size=batch_size,
            Pos=Pos,
            NumPatches=NumPatches,
            Channels=Channels,
            Height=Height,
            Width=Width,
            mesh=mesh,
            axis_resources=axis_resources,
            max_buffered_batches=0,
        )

        batch = next(iter(loader.iter_from_step(0)))

    # Verify shapes
    assert batch.input_ids.array.shape == (batch_size, target_seq_len), (
        f"Expected input_ids shape ({batch_size}, {target_seq_len}), got {batch.input_ids.array.shape}"
    )
    assert batch.loss_mask.array.shape == (batch_size, target_seq_len), (
        f"Expected loss_mask shape ({batch_size}, {target_seq_len}), got {batch.loss_mask.array.shape}"
    )
    assert batch.combined_mask.array.shape == (batch_size, target_seq_len), (
        f"Expected combined_mask shape ({batch_size}, {target_seq_len}), got {batch.combined_mask.array.shape}"
    )
    assert batch.position_ids.array.shape == (batch_size, target_seq_len), (
        f"Expected position_ids shape ({batch_size}, {target_seq_len}), got {batch.position_ids.array.shape}"
    )

    # Verify padding is correct (values after seq_len should be 0)
    input_ids_np = np.array(batch.input_ids.array)
    loss_mask_np = np.array(batch.loss_mask.array)
    combined_mask_np = np.array(batch.combined_mask.array)
    position_ids_np = np.array(batch.position_ids.array)

    for i, seq_len in enumerate(seq_lens):
        # Original values preserved for input_ids
        assert np.all(input_ids_np[i, :seq_len] == np.arange(seq_len)), (
            f"Example {i}: input_ids values not preserved"
        )
        # Padding is 0 for input_ids
        assert np.all(input_ids_np[i, seq_len:] == 0), (
            f"Example {i}: input_ids padding not zero"
        )

        # Original values preserved for loss_mask (all ones)
        assert np.all(loss_mask_np[i, :seq_len] == 1.0), (
            f"Example {i}: loss_mask values not preserved"
        )
        # Padding is 0 for loss_mask
        assert np.all(loss_mask_np[i, seq_len:] == 0.0), (
            f"Example {i}: loss_mask padding not zero"
        )

        # Original values preserved for combined_mask (all ones)
        assert np.all(combined_mask_np[i, :seq_len] == 1), (
            f"Example {i}: combined_mask values not preserved"
        )
        # Padding is 0 for combined_mask
        assert np.all(combined_mask_np[i, seq_len:] == 0), (
            f"Example {i}: combined_mask padding not zero"
        )

        # Original values preserved for position_ids
        assert np.all(position_ids_np[i, :seq_len] == np.arange(seq_len)), (
            f"Example {i}: position_ids values not preserved"
        )
        # Padding is 0 for position_ids
        assert np.all(position_ids_np[i, seq_len:] == 0), (
            f"Example {i}: position_ids padding not zero"
        )

    print("Variable-length sequence test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
