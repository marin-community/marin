# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
from transformers import AutoProcessor

from levanter.data.image import BatchImageProcessor, load_image
from levanter.store.cache import SerialCacheWriter
import jax
import jax.numpy as jnp

from test_image_utils import get_real_data, DEFAULT_GRID_PINPOINTS
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
        assert "loss_mask" in result
        assert result["input_ids"].shape == (2048,)


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
    """Test LLaVA OneVision model - compare HF and Levanter outputs."""
    jax.config.update("jax_default_matmul_precision", "float32")

    import dataclasses
    import torch
    from levanter.data.image import ImageDataLoader, ImageTextExample, create_custom_processor
    from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
    from levanter.layers.attention import AttentionBackend
    from levanter.trainer import TrainerConfig
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision, AutoConfig
    import equinox as eqx

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    image_processor = processor.image_processor
    grid_pinpoints = getattr(image_processor, "image_grid_pinpoints", None)
    patch_size = getattr(image_processor, "size", {}).get("height", 384)
    vision_feature_height = patch_size // 14
    vision_aspect_ratio = getattr(image_processor, "vision_aspect_ratio", "anyres_max_9")
    max_num_patches = None
    if vision_aspect_ratio and "anyres_max_" in vision_aspect_ratio:
        max_num_patches = int(vision_aspect_ratio.split("anyres_max_")[-1])

    padded_processor = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=grid_pinpoints)
    unpadded_processor = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=grid_pinpoints)

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

        # Load HF model
        hf_model = HfLlavaOnevision.from_pretrained(model_name, torch_dtype=torch.float32)
        hf_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
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

            # HF forward pass
            hf_logits_list = []
            hf_input_ids_list = []
            hf_image_sizes_list = []
            for sample_idx in range(batch_size):
                dataset_idx = cached_dataset_indices[sample_idx]
                raw_example = dataset[dataset_idx]
                messages = raw_example["messages"]
                images = raw_example.get("images", None)
                prompt_text = unpadded_processor.apply_chat_template(messages, add_generation_prompt=False)

                if images is not None and len(images) > 0:
                    pil_images = [load_image(img) for img in images]
                    hf_inputs = unpadded_processor(text=prompt_text, images=pil_images, return_tensors="pt")
                else:
                    hf_inputs = unpadded_processor(text=prompt_text, return_tensors="pt")

                hf_input_ids = hf_inputs["input_ids"]
                hf_input_ids_list.append(hf_input_ids[0].numpy())
                hf_image_sizes_list.append(hf_inputs.get("image_sizes"))

                with torch.no_grad():
                    hf_output = hf_model(**hf_inputs)
                    hf_logits_list.append(hf_output.logits[0].numpy())

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

                if has_image:
                    hf_ids = hf_input_ids_list[sample_idx]
                    num_hf_image_tokens = (hf_ids == image_token_id).sum()
                    hf_image_sizes = hf_image_sizes_list[sample_idx]
                    image_sizes_list = [hf_image_sizes[0].tolist()]
                    unpad_indices_np = padded_processor.compute_unpad_indices(
                        image_sizes=image_sizes_list,
                        height=patch_size,
                        width=patch_size,
                        max_num_features=int(num_hf_image_tokens),
                    )
                    NumImageTokensSample = hax.Axis("num_image_tokens", int(num_hf_image_tokens))
                    unpad_indices_lev = hax.named(
                        jnp.array(unpad_indices_np, dtype=jnp.int32), (Batch1, NumImageTokensSample)
                    )
                else:
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
    """Test that cache mode and streaming mode produce identical data."""
    import asyncio
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

        assert cache_len == streaming_len

        num_to_compare = min(10, cache_len)
        indices = list(range(num_to_compare))
        cache_examples = asyncio.run(cache_dataset.get_batch(indices))
        streaming_examples = asyncio.run(streaming_dataset.get_batch(indices))

        for i in range(num_to_compare):
            cache_ex = cache_examples[i]
            streaming_ex = streaming_examples[i]
            assert np.array_equal(cache_ex["input_ids"], streaming_ex["input_ids"])
            assert np.array_equal(cache_ex["attention_mask"], streaming_ex["attention_mask"])
            pixel_max_diff = np.abs(cache_ex["pixel_values"] - streaming_ex["pixel_values"]).max()
            assert pixel_max_diff < 1e-5


def test_streaming_dataset_basic():
    """Test StreamingImageDataset functionality."""
    import asyncio
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
            assert length > 0
            assert dataset.is_finite()
            batch = await dataset.get_batch([0, 1, 2])
            assert len(batch) == 3
            for ex in batch:
                assert "input_ids" in ex
                assert "pixel_values" in ex
            return True

        result = asyncio.run(run_tests())
        assert result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
