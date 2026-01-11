# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile

import pytest
from transformers import AutoProcessor

from levanter.data.image import (
    BatchImageProcessor,
    ImageDatasetSourceConfig,
    ConversationDatasetSourceConfig,
    load_image,
)
from levanter.data.sharded_datasource import (
    ImageTextUrlDataSource,
    ImageConversationUrlDataSource,
)
from levanter.store.cache import SerialCacheWriter
import jax
import jax.numpy as jnp

# Force torch to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Force JAX to use TPU
os.environ["JAX_PLATFORMS"] = "tpu"
# Force JAX to use float32
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"
# Enable float32 mode in JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

# Import test data utilities for loading from HuggingFace dataset
from test_image_utils import get_real_data  # noqa: E402

import numpy as np  # noqa: E402

# Import shared helper functions from test_image_utils
from test_image_utils import DEFAULT_GRID_PINPOINTS  # noqa: E402
import haliax as hax  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

# =============================================================================
# Tests for ShardedDataSource classes
# =============================================================================


class TestImageTextUrlDataSource:
    """Tests for ImageTextUrlDataSource."""

    @pytest.fixture
    def image_text_jsonl(self, tmp_path):
        """Create a JSONL file with image-text pairs."""
        data = [
            {"image": "/path/to/image1.jpg", "text": "A cat on the mat"},
            {"image": "/path/to/image2.jpg", "text": "A dog in the park"},
            {"image": "/path/to/image3.jpg", "text": "A bird on the tree"},
        ]
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return str(jsonl_path)

    def test_shard_names(self, image_text_jsonl):
        """Test that shard names match the input URLs."""
        ds = ImageTextUrlDataSource([image_text_jsonl])
        assert len(ds.shard_names) == 1

    def test_open_shard_at_row_zero(self, image_text_jsonl):
        """Test reading from the beginning of a shard."""
        ds = ImageTextUrlDataSource([image_text_jsonl])
        shard_name = ds.shard_names[0]
        records = list(ds.open_shard_at_row(shard_name, 0))
        assert len(records) == 3
        assert records[0]["text"] == "A cat on the mat"
        assert records[2]["text"] == "A bird on the tree"

    def test_open_shard_at_row_nonzero(self, image_text_jsonl):
        """Test reading from a specific row."""
        ds = ImageTextUrlDataSource([image_text_jsonl])
        shard_name = ds.shard_names[0]
        records = list(ds.open_shard_at_row(shard_name, 1))
        assert len(records) == 2
        assert records[0]["text"] == "A dog in the park"

    def test_custom_keys(self, tmp_path):
        """Test with custom image and text keys."""
        data = [
            {"img": "/path/img1.jpg", "caption": "Caption 1"},
            {"img": "/path/img2.jpg", "caption": "Caption 2"},
        ]
        jsonl_path = tmp_path / "custom.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        ds = ImageTextUrlDataSource([str(jsonl_path)], image_key="img", text_key="caption")
        shard_name = ds.shard_names[0]
        records = list(ds.open_shard(shard_name))
        assert len(records) == 2
        assert records[0]["image"] == "/path/img1.jpg"
        assert records[0]["text"] == "Caption 1"


class TestImageConversationUrlDataSource:
    """Tests for ImageConversationUrlDataSource."""

    @pytest.fixture
    def conversation_jsonl(self, tmp_path):
        """Create a JSONL file with conversation data."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "This is a cat."}]},
                ],
                "images": ["/path/to/cat.jpg"],
            },
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
                ],
                # No images in this example
            },
        ]
        jsonl_path = tmp_path / "conv.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return str(jsonl_path)

    def test_shard_names(self, conversation_jsonl):
        """Test that shard names are correct."""
        ds = ImageConversationUrlDataSource([conversation_jsonl])
        assert len(ds.shard_names) == 1

    def test_open_shard(self, conversation_jsonl):
        """Test reading conversation data."""
        ds = ImageConversationUrlDataSource([conversation_jsonl])
        shard_name = ds.shard_names[0]
        records = list(ds.open_shard(shard_name))
        assert len(records) == 2

        # First record has images
        assert len(records[0]["messages"]) == 2
        assert records[0]["images"] == ["/path/to/cat.jpg"]

        # Second record has no images
        assert records[1]["images"] == []

    def test_open_shard_at_row(self, conversation_jsonl):
        """Test reading from a specific row."""
        ds = ImageConversationUrlDataSource([conversation_jsonl])
        shard_name = ds.shard_names[0]
        records = list(ds.open_shard_at_row(shard_name, 1))
        assert len(records) == 1
        assert records[0]["images"] == []


class TestImageDatasetSourceConfig:
    """Tests for ImageDatasetSourceConfig."""

    def test_urls_for_split(self, tmp_path):
        """Test URL expansion for splits."""
        config = ImageDatasetSourceConfig(
            train_urls=[str(tmp_path / "train*.jsonl")],
            validation_urls=[str(tmp_path / "val*.jsonl")],
        )

        # Create some test files
        (tmp_path / "train1.jsonl").touch()
        (tmp_path / "train2.jsonl").touch()
        (tmp_path / "val1.jsonl").touch()

        train_urls = config.urls_for_split("train")
        assert len(train_urls) == 2

        val_urls = config.urls_for_split("validation")
        assert len(val_urls) == 1

    def test_invalid_split(self):
        """Test that invalid split raises error."""
        config = ImageDatasetSourceConfig()
        with pytest.raises(ValueError, match="Unknown split"):
            config.urls_for_split("test")

    def test_get_shard_source_from_urls(self, tmp_path):
        """Test getting shard source from URLs."""
        # Create a JSONL file with image-text pairs
        data = [
            {"image": "/path/to/img1.jpg", "text": "A cat"},
            {"image": "/path/to/img2.jpg", "text": "A dog"},
        ]
        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        config = ImageDatasetSourceConfig(
            train_urls=[str(jsonl_path)],
            image_key="image",
            text_key="text",
        )

        source = config.get_shard_source("train")
        assert source is not None
        records = list(source)
        assert len(records) == 2
        assert records[0]["image"] == "/path/to/img1.jpg"
        assert records[0]["text"] == "A cat"

    def test_get_shard_source_empty_urls(self):
        """Test that get_shard_source returns None for empty URLs."""
        config = ImageDatasetSourceConfig(
            train_urls=[],
        )
        source = config.get_shard_source("train")
        assert source is None

    def test_doc_iterator(self, tmp_path):
        """Test doc_iterator for URL-based data."""
        data = [
            {"image": "/path/img1.jpg", "text": "Text 1"},
            {"image": "/path/img2.jpg", "text": "Text 2"},
        ]
        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        config = ImageDatasetSourceConfig(
            train_urls=[str(jsonl_path)],
        )

        docs = list(config.doc_iterator("train"))
        assert len(docs) == 2
        assert docs[0]["text"] == "Text 1"


class TestConversationDatasetSourceConfig:
    """Tests for ConversationDatasetSourceConfig."""

    def test_urls_for_split(self, tmp_path):
        """Test URL expansion for splits."""
        config = ConversationDatasetSourceConfig(
            train_urls=[str(tmp_path / "train.jsonl")],
            validation_urls=[str(tmp_path / "val.jsonl")],
        )

        # Create test files
        (tmp_path / "train.jsonl").touch()
        (tmp_path / "val.jsonl").touch()

        train_urls = config.urls_for_split("train")
        assert len(train_urls) == 1

        val_urls = config.urls_for_split("validation")
        assert len(val_urls) == 1

    def test_get_shard_source_from_urls(self, tmp_path):
        """Test getting shard source from URLs."""
        # Create a conversation JSONL file
        data = [
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                "images": [],
            }
        ]
        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(data[0]) + "\n")

        config = ConversationDatasetSourceConfig(
            train_urls=[str(jsonl_path)],
        )

        source = config.get_shard_source("train")
        assert source is not None
        records = list(source)
        assert len(records) == 1

    def test_get_shard_source_with_images(self, tmp_path):
        """Test getting shard source with images."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "A beautiful sunset"}]},
                ],
                "images": ["/path/to/sunset.jpg"],
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare these"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "Both show cats"}]},
                ],
                "images": ["/path/cat1.jpg", "/path/cat2.jpg"],
            },
        ]
        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        config = ConversationDatasetSourceConfig(
            train_urls=[str(jsonl_path)],
        )

        source = config.get_shard_source("train")
        records = list(source)
        assert len(records) == 2
        assert len(records[0]["images"]) == 1
        assert len(records[1]["images"]) == 2

    def test_doc_iterator(self, tmp_path):
        """Test doc_iterator for conversation data."""
        data = [
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
                "images": [],
            },
            {
                "messages": [{"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}],
                "images": [],
            },
        ]
        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        config = ConversationDatasetSourceConfig(
            train_urls=[str(jsonl_path)],
        )

        docs = list(config.doc_iterator("train"))
        assert len(docs) == 2
        assert docs[0]["messages"][0]["role"] == "user"

    def test_custom_keys(self, tmp_path):
        """Test with custom message and image keys."""
        data = [
            {
                "conversation": [{"role": "user", "content": "Hello"}],
                "photos": ["/path/photo.jpg"],
            }
        ]
        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(data[0]) + "\n")

        config = ConversationDatasetSourceConfig(
            train_urls=[str(jsonl_path)],
            messages_key="conversation",
            images_key="photos",
        )

        source = config.get_shard_source("train")
        records = list(source)
        assert len(records) == 1
        assert records[0]["messages"][0]["role"] == "user"
        assert records[0]["images"] == ["/path/photo.jpg"]

    def test_invalid_split(self):
        """Test that invalid split raises error."""
        config = ConversationDatasetSourceConfig()
        with pytest.raises(ValueError, match="Unknown split"):
            config.urls_for_split("test")


class TestImageMixtureDatasetConfig:
    """Tests for ImageMixtureDatasetConfig."""

    def test_post_init_empty_configs(self):
        """Test that empty configs raises error."""
        from levanter.data.image import ImageMixtureDatasetConfig

        with pytest.raises(ValueError, match="At least one dataset must be provided"):
            ImageMixtureDatasetConfig(
                configs={},
                train_weights={},
            )

    def test_post_init_mismatched_keys(self):
        """Test that mismatched keys raise error."""
        from levanter.data.image import ImageMixtureDatasetConfig

        with pytest.raises(ValueError, match="keys in configs and weights must be the same"):
            ImageMixtureDatasetConfig(
                configs={"dataset1": ImageDatasetSourceConfig()},
                train_weights={"dataset2": 1.0},
            )

    def test_valid_config(self, tmp_path):
        """Test creating a valid mixture config."""
        from levanter.data.image import ImageMixtureDatasetConfig

        config = ImageMixtureDatasetConfig(
            cache_dir=str(tmp_path),
            configs={
                "ds1": ImageDatasetSourceConfig(
                    train_urls=[str(tmp_path / "train1.jsonl")],
                    cache_dir=str(tmp_path / "ds1"),
                ),
                "ds2": ImageDatasetSourceConfig(
                    train_urls=[str(tmp_path / "train2.jsonl")],
                    cache_dir=str(tmp_path / "ds2"),
                ),
            },
            train_weights={"ds1": 0.6, "ds2": 0.4},
        )

        assert len(config.configs) == 2
        assert config.train_weights["ds1"] == 0.6
        assert config.sources == config.configs

    def test_shuffle_options(self, tmp_path):
        """Test different shuffle configurations."""
        from levanter.data.image import ImageMixtureDatasetConfig

        # Test shuffle=False
        config = ImageMixtureDatasetConfig(
            configs={"ds": ImageDatasetSourceConfig(cache_dir=str(tmp_path))},
            train_weights={"ds": 1.0},
            shuffle=False,
        )
        assert config.shuffle is False

        # Test shuffle=True
        config = ImageMixtureDatasetConfig(
            configs={"ds": ImageDatasetSourceConfig(cache_dir=str(tmp_path))},
            train_weights={"ds": 1.0},
            shuffle=True,
        )
        assert config.shuffle is True

        # Test shuffle as era length
        config = ImageMixtureDatasetConfig(
            configs={"ds": ImageDatasetSourceConfig(cache_dir=str(tmp_path))},
            train_weights={"ds": 1.0},
            shuffle=1000,
        )
        assert config.shuffle == 1000

    def test_conversation_and_image_mixture(self, tmp_path):
        """Test mixing conversation and image-text datasets."""
        from levanter.data.image import ImageMixtureDatasetConfig

        config = ImageMixtureDatasetConfig(
            cache_dir=str(tmp_path),
            configs={
                "image_text": ImageDatasetSourceConfig(
                    train_urls=[str(tmp_path / "images.jsonl")],
                    cache_dir=str(tmp_path / "images"),
                ),
                "conversations": ConversationDatasetSourceConfig(
                    train_urls=[str(tmp_path / "conversations.jsonl")],
                    cache_dir=str(tmp_path / "conversations"),
                ),
            },
            train_weights={"image_text": 0.5, "conversations": 0.5},
        )

        assert len(config.configs) == 2
        assert isinstance(config.configs["image_text"], ImageDatasetSourceConfig)
        assert isinstance(config.configs["conversations"], ConversationDatasetSourceConfig)


@pytest.fixture
def processor():
    return AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-si-hf")


@pytest.fixture
def dataset():
    return get_real_data()


def test_load_image_from_bytes(dataset):
    """Test loading an image from HuggingFace bytes format or PIL Image."""
    from PIL import Image

    example = dataset[0]
    image_data = example["images"][0]

    # Image can be either a dict with bytes key or already a PIL Image
    if isinstance(image_data, Image.Image):
        # Already a PIL Image (from HuggingFace dataset with decoded images)
        image = image_data
    else:
        # Should have bytes key
        assert "bytes" in image_data
        # Load the image
        image = load_image(image_data)

    # Check it's a valid PIL image
    assert image.mode == "RGB"
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_batch_image_processor(processor, dataset):
    """Test BatchImageProcessor with conversation data."""
    batch_processor = BatchImageProcessor(
        processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,  # Disable masking for simpler testing
    )

    # Get first few examples
    examples = [dataset[i] for i in range(4)]

    # Process the batch
    results = batch_processor(examples)

    assert len(results) == 4

    for result in results:
        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "image_sizes" in result
        assert "loss_mask" in result

        # Check shapes
        assert result["input_ids"].shape == (2048,), f"Expected (2048,), got {result['input_ids'].shape}"
        assert result["attention_mask"].shape == (2048,), f"Expected (2048,), got {result['attention_mask'].shape}"
        assert result["loss_mask"].shape == (2048,), f"Expected (2048,), got {result['loss_mask'].shape}"

        # pixel_values should have proper dimensions
        assert result["pixel_values"].ndim >= 3


def test_batch_image_processor_with_masking(processor, dataset):
    """Test BatchImageProcessor with label masking enabled."""
    batch_processor = BatchImageProcessor(
        processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=True,
    )

    # Get a single example
    example = dataset[0]

    # Process
    results = batch_processor([example])

    assert len(results) == 1
    result = results[0]

    # loss_mask should be mostly 0.0 (masked) for non-assistant tokens
    # At least some tokens should be masked
    assert (result["loss_mask"] == 0.0).any(), "Expected some tokens to be masked"


def test_serial_cache_write_and_read(processor, dataset):
    """Test writing and reading from a serial cache."""
    # Use a large max_length to avoid truncation issues with image tokens
    # Some examples may have many images, so we need enough space
    batch_processor = BatchImageProcessor(
        processor,
        max_length=8192,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write to cache
        with SerialCacheWriter(tmpdir, batch_processor.output_exemplar) as writer:
            for i in range(10):
                example = dataset[i]
                try:
                    results = batch_processor([example])
                    writer.write_batch(results)
                except ValueError as e:
                    if "Mismatch in `image` token count" in str(e):
                        # Skip examples that are too long even with large max_length
                        continue
                    raise

        cache = writer.result()

        # Read back from cache - get available examples
        cache_len = len(cache)
        if cache_len > 0:
            cached_examples = cache.get_batch_sync(list(range(min(cache_len, 10))))

            assert len(cached_examples) > 0

            for ex in cached_examples:
                assert ex["input_ids"].shape == (8192,), f"Expected (8192,), got {ex['input_ids'].shape}"
                assert ex["attention_mask"].shape == (8192,), f"Expected (8192,), got {ex['attention_mask'].shape}"
                assert ex["loss_mask"].shape == (8192,), f"Expected (8192,), got {ex['loss_mask'].shape}"


def test_metadata(processor):
    """Test that metadata is properly generated."""
    batch_processor = BatchImageProcessor(
        processor,
        max_length=2048,
        padding=True,
    )

    metadata = batch_processor.metadata
    assert "processor" in metadata
    assert "max_length" in metadata
    assert metadata["max_length"] == 2048


@pytest.mark.asyncio
async def test_hf_image_ray_pipeline():
    """Test image data pipeline, similar to test_hf_audio_ray_pipeline.

    This test:
    1. Creates a cache from parquet data using SerialCacheWriter
    2. Wraps it in ProcessedImageCache for async access
    3. Fetches batches asynchronously
    4. Verifies the output shapes and keys
    """
    from levanter.data.image import ProcessedImageCache

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-si-hf")
    dataset = get_real_data()

    batch_processor = BatchImageProcessor(
        processor,
        max_length=8192,  # Use larger max_length to avoid truncation issues with image tokens
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build cache using SerialCacheWriter
        with SerialCacheWriter(tmpdir, batch_processor.output_exemplar) as writer:
            for i in range(15):  # Process enough examples
                example = dataset[i]
                try:
                    results = batch_processor([example])
                    writer.write_batch(results)
                except ValueError as e:
                    if "Mismatch in `image` token count" in str(e):
                        # Skip examples that are too long
                        continue
                    raise

        cache = writer.result()
        processed_cache = ProcessedImageCache(cache)

        # Fetch and verify batches asynchronously
        cache_len = len(cache)
        if cache_len < 10:
            # If we don't have enough examples, just test what we have
            num_to_test = cache_len
        else:
            num_to_test = 10

        for i in range(num_to_test):
            t = (await processed_cache.get_batch([i]))[0]
            # Verify the expected keys and shapes
            assert "pixel_values" in t, "pixel_values should be present"
            assert "input_ids" in t, "input_ids should be present"
            assert "attention_mask" in t, "attention_mask should be present"
            assert "loss_mask" in t, "loss_mask should be present"
            assert t["input_ids"].shape == (8192,), f"Expected input_ids shape (8192,), got {t['input_ids'].shape}"
            assert t["attention_mask"].shape == (
                8192,
            ), f"Expected attention_mask shape (8192,), got {t['attention_mask'].shape}"
            assert t["loss_mask"].shape == (8192,), f"Expected loss_mask shape (8192,), got {t['loss_mask'].shape}"
            # pixel_values should have proper dimensions (num_patches, channels, height, width)
            assert t["pixel_values"].ndim >= 3, f"Expected pixel_values ndim >= 3, got {t['pixel_values'].ndim}"


def test_image_data_loader(processor, dataset):
    """Test ImageDataLoader with cached data."""
    from levanter.data.loader import ImageDataLoader, ImageTextExample

    batch_processor = BatchImageProcessor(
        processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        mask_prompt=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # First create a cache with some examples
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

        # Get example shape info - find max num_patches across all cached examples
        all_examples = cache.get_batch_sync(list(range(cache_len)))
        max_num_patches = max(ex["pixel_values"].shape[0] for ex in all_examples)
        first_ex = all_examples[0]
        seq_len = first_ex["input_ids"].shape[0]

        # Create axes - use max_num_patches to ensure all examples can be padded to this size
        Pos = hax.Axis("position", seq_len)
        NumPatches = hax.Axis("num_patches", max_num_patches)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
        Width = hax.Axis("width", first_ex["pixel_values"].shape[3])

        # Create a simple mesh for testing with proper axis resources
        devices = np.array(jax.devices("cpu")[:1])
        mesh = Mesh(devices, ("data",))

        # Create the dataloader with matching axis_resources
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
                max_buffered_batches=0,  # Disable background iteration for testing
            )

            # Get one batch
            batch_iter = iter(loader)
            batch = next(batch_iter)

            # Verify the batch structure
            assert isinstance(batch, ImageTextExample)
            assert batch.pixel_values.array.shape[0] == batch_size
            assert batch.input_ids.array.shape[0] == batch_size
            # ImageTextExample uses loss_mask instead of attention_mask/labels
            assert batch.loss_mask.array.shape[0] == batch_size
            # Check grid_mask if present
            if batch.grid_mask is not None:
                assert batch.grid_mask.array.shape[0] == batch_size


def test_llava_with_image_dataloader(processor, dataset):
    """Test LLaVA OneVision model using ImageDataLoader.

    This test:
    1. Creates a cache from the dataset using padded processor
    2. Uses ImageDataLoader to get a batch
    3. Runs the batch through both HuggingFace and Levanter models
    4. Compares outputs for consistency
    """
    import time
    import dataclasses
    import torch
    from levanter.data.loader import ImageDataLoader, ImageTextExample
    from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
    from levanter.layers.attention import AttentionBackend
    from levanter.trainer import TrainerConfig
    from transformers import LlavaOnevisionForConditionalGeneration as HfLlavaOnevision

    print("\n=== Test: LLaVA OneVision with ImageDataLoader ===")

    # Use smaller model for testing
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    # Import custom processor for padding support
    from levanter.data.image import create_custom_processor

    # Get grid_pinpoints and related params from the standard processor
    image_processor = processor.image_processor
    grid_pinpoints = getattr(image_processor, "image_grid_pinpoints", None)
    patch_size = getattr(image_processor, "size", {}).get("height", 384)
    # vision_feature_height = patch_size // 14 for SigLIP
    vision_feature_height = patch_size // 14
    # Parse max_num_patches from vision_aspect_ratio (e.g., "anyres_max_9" -> 9)
    vision_aspect_ratio = getattr(image_processor, "vision_aspect_ratio", "anyres_max_9")
    max_num_patches = None
    if vision_aspect_ratio and "anyres_max_" in vision_aspect_ratio:
        max_num_patches = int(vision_aspect_ratio.split("anyres_max_")[-1])

    # Create padded processor for Levanter (do_pad=True generates correct input_ids for padded pixel_values)
    padded_processor = create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=grid_pinpoints)

    batch_processor = BatchImageProcessor(
        padded_processor,  # Use padded processor instead of standard processor
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

    # Create unpadded processor for HF (to get correct input_ids for unpadded pixel_values)
    unpadded_processor = create_custom_processor(model_name, do_pad=False, image_grid_pinpoints=grid_pinpoints)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a cache with some examples, tracking which dataset indices were cached
        print("\n--- Creating cache ---")
        start_time = time.time()
        cached_dataset_indices = []  # Track which dataset samples were successfully cached
        with SerialCacheWriter(tmpdir, batch_processor.output_exemplar) as writer:
            for i in range(8):
                example = dataset[i]
                try:
                    results = batch_processor([example])
                    writer.write_batch(results)
                    cached_dataset_indices.append(i)  # Track successful cache
                except ValueError as e:
                    if "Mismatch in `image` token count" in str(e):
                        continue
                    raise

        cache = writer.result()
        cache_len = len(cache)
        print(f"  Cache created with {cache_len} examples in {time.time() - start_time:.2f}s")
        print(f"  Cached dataset indices: {cached_dataset_indices}")

        if cache_len < 2:
            pytest.skip("Not enough examples cached for test")

        # Get shape info
        all_examples = cache.get_batch_sync(list(range(cache_len)))
        max_num_patches = max(ex["pixel_values"].shape[0] for ex in all_examples)
        first_ex = all_examples[0]
        seq_len = first_ex["input_ids"].shape[0]

        print(f"  max_num_patches: {max_num_patches}")
        print(f"  seq_len: {seq_len}")
        print(f"  pixel_values shape: {first_ex['pixel_values'].shape}")

        # Create axes
        Pos = hax.Axis("position", seq_len)
        NumPatches = hax.Axis("num_patches", max_num_patches)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
        Width = hax.Axis("width", first_ex["pixel_values"].shape[3])
        # NumImageTokens: total patches * features per patch (for unpad_indices)
        features_per_patch = vision_feature_height * vision_feature_height  # e.g., 27*27 = 729
        max_image_tokens = max_num_patches * features_per_patch
        NumImageTokens = hax.Axis("num_image_tokens", max_image_tokens)

        # Load HuggingFace model for comparison
        print("\n--- Loading HuggingFace model for comparison ---")
        start_time = time.time()
        hf_model = HfLlavaOnevision.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        # Update HF model config to match the processor's grid_pinpoints (anyres_max_9)
        hf_model.model.config.image_grid_pinpoints = DEFAULT_GRID_PINPOINTS
        hf_model.model.image_newline = None  # Disable image_newline for consistency
        hf_model.eval()
        print(f"  HF model loaded in {time.time() - start_time:.2f}s")

        # Load model config
        print(f"\n--- Loading model config: {model_name} ---")
        start_time = time.time()
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config = LlavaOnevisionConfig.from_hf_config(hf_config)

        # Use VANILLA attention backend for consistency comparison with HF
        vision_config_updated = dataclasses.replace(
            config.vision_config,
            use_flash_attention=False,
            attn_backend=AttentionBackend.VANILLA,
            gradient_checkpointing=False,
        )
        text_config_updated = dataclasses.replace(
            config.text_config,
            attn_backend=AttentionBackend.VANILLA,
            gradient_checkpointing=False,
        )
        config = dataclasses.replace(
            config,
            vision_config=vision_config_updated,
            text_config=text_config_updated,
            gradient_checkpointing=False,
        )
        print(f"  Config loaded in {time.time() - start_time:.2f}s")

        # Load model with trainer mesh
        print("\n--- Loading Levanter model ---")
        start_time = time.time()
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
            print(f"  Levanter model loaded in {time.time() - start_time:.2f}s")

            # Create dataloader
            print("\n--- Creating ImageDataLoader ---")
            batch_size = min(4, cache_len)
            axis_resources = trainer_config.compute_axis_mapping

            # Get the mesh from the trainer config context
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
                allow_nondivisible_batch_size=True,
                NumImageTokens=NumImageTokens,
            )

            # Get first batch
            print("\n--- Getting first batch from dataloader ---")
            start_time = time.time()
            batch_iter = iter(loader)
            batch = next(batch_iter)
            print(f"  Batch loaded in {time.time() - start_time:.2f}s")

            # Verify batch structure
            assert isinstance(batch, ImageTextExample)
            print(f"  pixel_values shape: {batch.pixel_values.array.shape}")
            print(f"  input_ids shape: {batch.input_ids.array.shape}")

            # --- HuggingFace Forward Pass (using raw data with unpadded processor) ---
            # Process raw samples with do_pad=False to get correctly matched input_ids and pixel_values
            print("\n--- HuggingFace Forward Pass (processing raw data with do_pad=False) ---")
            start_time = time.time()

            # Extract Levanter batch inputs for later comparison
            batch_input_ids = np.array(batch.input_ids.array)  # (batch_size, seq_len)
            batch_pixel_values = np.array(batch.pixel_values.array)  # (batch_size, num_patches, C, H, W)
            batch_grid_mask = np.array(batch.grid_mask.array) if batch.grid_mask is not None else None
            _batch_loss_mask = np.array(batch.loss_mask.array) if batch.loss_mask is not None else None

            # HF forward pass for each sample using raw data
            hf_logits_list = []
            hf_input_ids_list = []  # Store HF input_ids for alignment
            hf_image_sizes_list = []  # Store HF image_sizes for unpad_indices computation
            for sample_idx in range(batch_size):
                # Get raw data from dataset using tracked indices
                dataset_idx = cached_dataset_indices[sample_idx]
                raw_example = dataset[dataset_idx]

                # Process with unpadded processor (do_pad=False)
                messages = raw_example["messages"]
                images = raw_example.get("images", None)

                # Format for processor
                # Use add_generation_prompt=False to match BatchImageProcessor default
                prompt_text = unpadded_processor.apply_chat_template(messages, add_generation_prompt=False)

                if images is not None and len(images) > 0:
                    pil_images = [load_image(img) for img in images]
                    hf_inputs = unpadded_processor(
                        text=prompt_text,
                        images=pil_images,
                        return_tensors="pt",
                    )
                else:
                    hf_inputs = unpadded_processor(
                        text=prompt_text,
                        return_tensors="pt",
                    )

                hf_input_ids = hf_inputs["input_ids"]
                hf_input_ids_list.append(hf_input_ids[0].numpy())
                hf_image_sizes_list.append(hf_inputs.get("image_sizes"))  # May be None for text-only

                # Run HF forward
                with torch.no_grad():
                    hf_output = hf_model(**hf_inputs)
                    hf_logit = hf_output.logits[0].numpy()

                hf_logits_list.append(hf_logit)
                print(
                    f"    Sample {sample_idx} (dataset[{dataset_idx}]): input_ids={hf_input_ids.shape}, logits={hf_logit.shape}"
                )

            print(f"  HF forward time: {time.time() - start_time:.2f}s")

            # --- Levanter Forward Pass (per sample to match HF's variable-length processing) ---
            print("\n--- Levanter Forward Pass (per sample) ---")

            # Get the image token ID from HF model config
            image_token_id = hf_model.config.image_token_index

            # Debug: Check pad token vs image token
            pad_token_id = padded_processor.tokenizer.pad_token_id
            print(f"  image_token_id={image_token_id}, pad_token_id={pad_token_id}")
            if pad_token_id == image_token_id:
                print("  WARNING: pad_token_id == image_token_id! This will cause confusion in comparisons.")

            # Process each sample individually with correct unpad_indices
            # We need to exit the mesh context to avoid sharding issues with batch size 1
            lev_logits_list = []

            # Define forward function outside the loop (uses eqx.filter_jit for flexibility)
            import equinox as eqx

            @eqx.filter_jit
            def compute_forward_single(model, input_ids, pixel_values, grid_mask, unpad_indices):
                return model(
                    input_ids, pixel_values=pixel_values, grid_mask=grid_mask, unpad_indices=unpad_indices, key=None
                )

            for sample_idx in range(batch_size):
                input_ids_np = batch_input_ids[sample_idx : sample_idx + 1]  # (1, seq_len)
                pixel_values_np = batch_pixel_values[sample_idx : sample_idx + 1]  # (1, num_patches, C, H, W)
                grid_mask_np = batch_grid_mask[sample_idx : sample_idx + 1] if batch_grid_mask is not None else None

                # Check if this sample has images (using grid_mask instead of image_sizes)
                has_image = grid_mask_np is not None and grid_mask_np[0].any()

                # Create named arrays for this sample
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
                    # Count actual HF image tokens (from unpadded processor)
                    # This is the target number of features we want to produce
                    hf_ids = hf_input_ids_list[sample_idx]
                    num_hf_image_tokens = (hf_ids == image_token_id).sum()

                    # Compute unpad_indices for this specific sample
                    # Use HF's image token count as max_num_features to produce same number of features
                    # Get image sizes from HF processor output (stored during HF forward pass)
                    hf_image_sizes = hf_image_sizes_list[sample_idx]
                    image_sizes_list = [hf_image_sizes[0].tolist()]  # [(h, w)]
                    unpad_indices_np_sample = padded_processor.compute_unpad_indices(
                        image_sizes=image_sizes_list,
                        height=patch_size,
                        width=patch_size,
                        max_num_features=int(num_hf_image_tokens),
                    )
                    # unpad_indices_np_sample shape: (1, num_hf_image_tokens)
                    NumImageTokensSample = hax.Axis("num_image_tokens", int(num_hf_image_tokens))
                    unpad_indices_lev = hax.named(
                        jnp.array(unpad_indices_np_sample, dtype=jnp.int32), (Batch1, NumImageTokensSample)
                    )
                else:
                    unpad_indices_lev = None

                # Run Levanter forward
                lev_logits_sample = compute_forward_single(
                    lev_model, input_ids_lev, pixel_values_lev, grid_mask_lev, unpad_indices_lev
                )
                lev_logits_sample.array.block_until_ready()
                lev_logits_list.append(np.array(lev_logits_sample.array)[0])  # Remove batch dim
                print(f"    Sample {sample_idx}: processed (has_image={has_image})")

            # --- Compare HF and Levanter outputs ---
            # Compare ALL tokens at valid positions (like test_llava_onevision_real_image_text)
            # Both HF and Levanter should have same sequence length since we use matching input_ids
            print("\n--- Comparing HF and Levanter outputs ---")

            all_correlations = []
            all_pred_match_rates = []

            for sample_idx in range(min(batch_size, 4)):
                hf_logit = hf_logits_list[sample_idx]  # (hf_seq_len, vocab_size)
                lev_logit = lev_logits_list[sample_idx]  # (lev_seq_len, vocab_size)
                hf_ids = hf_input_ids_list[sample_idx]  # (hf_seq_len,)
                lev_ids = batch_input_ids[sample_idx]  # (lev_seq_len,)

                # Find image token positions in both sequences
                hf_image_mask = hf_ids == image_token_id
                lev_image_mask = lev_ids == image_token_id

                hf_num_image = hf_image_mask.sum()
                lev_num_image = lev_image_mask.sum()

                hf_has_image = hf_num_image > 0
                lev_has_image = lev_num_image > 0

                print(
                    f"    Sample {sample_idx}: HF seq_len={len(hf_ids)}, Lev seq_len={len(lev_ids)}, "
                    f"HF images={hf_num_image}, Lev images={lev_num_image}"
                )

                if hf_has_image and lev_has_image:
                    # Compare by region like test_llava_onevision_real_image_text
                    hf_first_image = np.where(hf_image_mask)[0][0]
                    lev_first_image = np.where(lev_image_mask)[0][0]
                    hf_last_image = np.where(hf_image_mask)[0][-1]
                    lev_last_image = np.where(lev_image_mask)[0][-1]

                    # Debug: Print image token positions
                    print(
                        f"      HF image range: [{hf_first_image}, {hf_last_image}] (contiguous: {hf_last_image - hf_first_image + 1 == hf_num_image})"
                    )
                    print(
                        f"      Lev image range: [{lev_first_image}, {lev_last_image}] (contiguous: {lev_last_image - lev_first_image + 1 == lev_num_image})"
                    )

                    # Debug: Check if image tokens are truly contiguous
                    hf_image_positions = np.where(hf_image_mask)[0]
                    lev_image_positions = np.where(lev_image_mask)[0]
                    if not np.array_equal(hf_image_positions, np.arange(hf_first_image, hf_last_image + 1)):
                        print("      WARNING: HF image tokens are NOT contiguous!")
                        gaps = np.where(np.diff(hf_image_positions) > 1)[0]
                        for g in gaps[:3]:
                            print(f"        Gap at positions {hf_image_positions[g]} -> {hf_image_positions[g+1]}")
                    if not np.array_equal(lev_image_positions, np.arange(lev_first_image, lev_last_image + 1)):
                        print("      WARNING: Lev image tokens are NOT contiguous!")
                        gaps = np.where(np.diff(lev_image_positions) > 1)[0]
                        for g in gaps[:3]:
                            print(f"        Gap at positions {lev_image_positions[g]} -> {lev_image_positions[g+1]}")

                    regions = []

                    # Pre-image text (should match exactly)
                    pre_len = min(hf_first_image, lev_first_image)
                    if pre_len > 0:
                        hf_pre = hf_logit[:pre_len]
                        lev_pre = lev_logit[:pre_len]
                        pre_diff = np.abs(hf_pre - lev_pre).mean()
                        regions.append(("pre-image", pre_len, pre_diff, hf_pre, lev_pre))

                    # Image tokens (compare HF's N tokens with Levanter's first N)
                    # With unpad_indices, Levanter's first N image token positions have valid features
                    hf_image_start = hf_first_image
                    lev_image_start = lev_first_image
                    image_len = min(hf_num_image, lev_num_image)  # Should be equal with unpad_indices
                    hf_image = hf_logit[hf_image_start : hf_image_start + image_len]
                    lev_image = lev_logit[lev_image_start : lev_image_start + image_len]
                    image_diff = np.abs(hf_image - lev_image).mean()
                    regions.append(("image", image_len, image_diff, hf_image, lev_image))

                    # Post-image text (align by offset from end of image tokens)
                    # Use first_image + image_len to find where valid image tokens end
                    # (not last_image which may include extra padded placeholders)
                    hf_post_start = hf_first_image + hf_num_image
                    lev_post_start = lev_first_image + hf_num_image  # Use HF's count for Lev too
                    hf_post_len = len(hf_ids) - hf_post_start
                    lev_post_len = len(lev_ids) - lev_post_start
                    post_len = min(hf_post_len, lev_post_len)

                    # Debug: Find where Levanter's actual content ends (before padding)
                    # Look for the first padding token after the image tokens
                    lev_content_mask = lev_ids != pad_token_id
                    lev_content_positions = np.where(lev_content_mask)[0]
                    if len(lev_content_positions) > 0:
                        lev_actual_end = lev_content_positions[-1] + 1  # Exclusive end
                        lev_post_actual_len = lev_actual_end - lev_post_start
                        print(
                            f"      Lev actual content ends at {lev_actual_end}, post-image actual length: {lev_post_actual_len}"
                        )
                    else:
                        lev_actual_end = len(lev_ids)
                        lev_post_actual_len = lev_post_len

                    # Only compare non-padded tokens
                    hf_post_actual_len = len(hf_ids) - hf_post_start
                    post_len = min(hf_post_actual_len, lev_post_actual_len)
                    print(
                        f"      Comparing post-image: HF has {hf_post_actual_len}, Lev has {lev_post_actual_len}, comparing {post_len}"
                    )

                    # Debug: Check if post-image tokens match
                    if post_len > 0:
                        hf_post_ids = hf_ids[hf_post_start : hf_post_start + post_len]
                        lev_post_ids = lev_ids[lev_post_start : lev_post_start + post_len]
                        ids_match = np.array_equal(hf_post_ids, lev_post_ids)
                        if not ids_match:
                            mismatch_positions = np.where(hf_post_ids != lev_post_ids)[0]
                            print(f"      WARNING: Post-image token mismatch at positions: {mismatch_positions}")
                            for pos in mismatch_positions[:5]:  # Show first 5 mismatches
                                print(f"        pos {pos}: HF={hf_post_ids[pos]}, Lev={lev_post_ids[pos]}")

                        hf_post = hf_logit[hf_post_start : hf_post_start + post_len]
                        lev_post = lev_logit[lev_post_start : lev_post_start + post_len]

                        # Calculate diff excluding mismatched token positions
                        if not ids_match:
                            match_mask = hf_post_ids == lev_post_ids
                            post_diff_matched = np.abs(hf_post[match_mask] - lev_post[match_mask]).mean()
                            post_diff_all = np.abs(hf_post - lev_post).mean()
                            print(f"      post-image diff (matched only): {post_diff_matched:.6f}")
                            print(f"      post-image diff (all): {post_diff_all:.6f}")
                            # Use matched positions only for regions
                            post_diff = post_diff_matched
                            regions.append(
                                (
                                    "post-image",
                                    np.sum(match_mask),
                                    post_diff,
                                    hf_post[match_mask],
                                    lev_post[match_mask],
                                )
                            )
                        else:
                            post_diff = np.abs(hf_post - lev_post).mean()
                            regions.append(("post-image", post_len, post_diff, hf_post, lev_post))

                    # Print region stats
                    for name, length, diff, _, _ in regions:
                        print(f"      {name}: {length} tokens, mean_diff={diff:.6f}")

                    # Combine all regions for overall comparison
                    hf_compare = np.concatenate([r[3] for r in regions], axis=0)
                    lev_compare = np.concatenate([r[4] for r in regions], axis=0)
                else:
                    # Text-only sample - compare full sequences
                    min_len = min(len(hf_logit), len(lev_logit))
                    hf_compare = hf_logit[:min_len]
                    lev_compare = lev_logit[:min_len]

                # Calculate correlation
                correlation = np.corrcoef(hf_compare.flatten(), lev_compare.flatten())[0, 1]
                all_correlations.append(correlation)

                # Compare argmax predictions
                hf_preds = np.argmax(hf_compare, axis=-1)
                lev_preds = np.argmax(lev_compare, axis=-1)
                pred_match_rate = np.mean(hf_preds == lev_preds)
                all_pred_match_rates.append(pred_match_rate)

                # Calculate diff stats
                abs_diff = np.abs(hf_compare - lev_compare)
                max_abs_diff = np.max(abs_diff)
                mean_abs_diff = np.mean(abs_diff)

                print(
                    f"      OVERALL: {len(hf_compare)} tokens compared, "
                    f"corr={correlation:.4f}, pred_match={pred_match_rate:.4f}, "
                    f"max_diff={max_abs_diff:.4f}, mean_diff={mean_abs_diff:.6f}"
                )

            # Overall statistics
            if all_correlations:
                avg_correlation = np.mean(all_correlations)
                avg_pred_match = np.mean(all_pred_match_rates)
                print(f"\n  Average correlation: {avg_correlation:.6f}")
                print(f"  Average prediction match rate: {avg_pred_match:.4f}")

                # All tokens (text + image) should match closely with unpad_indices
                assert avg_correlation > 0.99, f"Average correlation too low: {avg_correlation}"
                assert avg_pred_match > 0.90, f"Average prediction match too low: {avg_pred_match}"

            print("  All samples pass consistency check with HuggingFace!")


def test_cache_vs_streaming_data_consistency():
    """Test that cache mode (use_cache=True) and streaming mode (use_cache=False) produce identical data.

    This test ensures that:
    1. Both modes load and process the same raw data
    2. The processed outputs (input_ids, pixel_values, loss_mask) are identical
    3. Streaming mode is a valid drop-in replacement for cache mode

    Note: This is a sync test because cache building internally uses asyncio.run(),
    which cannot be called from within an async test.
    """
    import asyncio
    from levanter.data.image import (
        ImageMixtureDatasetConfig,
        ConversationDatasetSourceConfig,
    )

    print("\n=== Test: Cache vs Streaming Data Consistency ===")

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file for this test
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)
        print(f"  Saved HF dataset to temporary parquet: {parquet_path}")
        # ====== Create config with caching enabled ======
        print("\n--- Building dataset with caching (use_cache=True) ---")
        cache_config = ImageMixtureDatasetConfig(
            cache_dir=f"{tmpdir}/cache",
            configs={
                "train": ConversationDatasetSourceConfig(
                    train_urls=[f"file://{parquet_path}"],
                    validation_urls=[f"file://{parquet_path}"],
                    cache_dir=f"{tmpdir}/cache/train",
                ),
            },
            train_weights={"train": 1.0},
            processor=model_name,
            max_length=8192,
            use_cache=True,  # Use caching mode
        )

        # Build cached dataset (this internally uses asyncio.run)
        cache_datasets = cache_config.training_sets()
        cache_dataset = list(cache_datasets.values())[0]

        # Get cache length synchronously
        cache_len = asyncio.run(cache_dataset.async_len())
        print(f"  Cache dataset loaded with {cache_len} examples")

        # ====== Create config with streaming enabled ======
        print("\n--- Building dataset with streaming (use_cache=False) ---")
        streaming_config = ImageMixtureDatasetConfig(
            cache_dir=f"{tmpdir}/streaming_cache",  # Different dir to avoid conflict
            configs={
                "train": ConversationDatasetSourceConfig(
                    train_urls=[f"file://{parquet_path}"],
                    validation_urls=[f"file://{parquet_path}"],
                    cache_dir=f"{tmpdir}/streaming_cache/train",
                ),
            },
            train_weights={"train": 1.0},
            processor=model_name,
            max_length=8192,  # Must match cache config for fair comparison
            use_cache=False,  # Use streaming mode
        )

        # Build streaming dataset
        streaming_datasets = streaming_config.training_sets()
        streaming_dataset = list(streaming_datasets.values())[0]

        # Get streaming length
        streaming_len = asyncio.run(streaming_dataset.async_len())
        print(f"  Streaming dataset loaded with {streaming_len} examples")

        # ====== Compare lengths ======
        print("\n--- Comparing dataset lengths ---")
        print(f"  Cache length: {cache_len}")
        print(f"  Streaming length: {streaming_len}")
        assert cache_len == streaming_len, f"Length mismatch: cache={cache_len}, streaming={streaming_len}"

        # ====== Compare first N examples ======
        num_to_compare = min(10, cache_len)
        print(f"\n--- Comparing first {num_to_compare} examples ---")

        # Get examples from both datasets
        indices = list(range(num_to_compare))
        cache_examples = asyncio.run(cache_dataset.get_batch(indices))
        streaming_examples = asyncio.run(streaming_dataset.get_batch(indices))

        all_input_ids_match = True
        all_attention_mask_match = True
        all_pixel_values_match = True
        all_loss_mask_match = True

        for i in range(num_to_compare):
            cache_ex = cache_examples[i]
            streaming_ex = streaming_examples[i]

            # Compare input_ids
            input_ids_match = np.array_equal(cache_ex["input_ids"], streaming_ex["input_ids"])
            if not input_ids_match:
                all_input_ids_match = False
                print(f"  Example {i}: input_ids MISMATCH")
                # Find first difference
                diff_idx = np.where(cache_ex["input_ids"] != streaming_ex["input_ids"])[0]
                if len(diff_idx) > 0:
                    first_diff = diff_idx[0]
                    print(
                        f"    First diff at position {first_diff}: cache={cache_ex['input_ids'][first_diff]}, streaming={streaming_ex['input_ids'][first_diff]}"
                    )

            # Compare attention_mask
            attention_mask_match = np.array_equal(cache_ex["attention_mask"], streaming_ex["attention_mask"])
            if not attention_mask_match:
                all_attention_mask_match = False
                print(f"  Example {i}: attention_mask MISMATCH")

            # Compare pixel_values
            pixel_diff = np.abs(cache_ex["pixel_values"] - streaming_ex["pixel_values"])
            pixel_max_diff = pixel_diff.max()
            pixel_values_match = pixel_max_diff < 1e-5  # Allow small numerical tolerance
            if not pixel_values_match:
                all_pixel_values_match = False
                print(f"  Example {i}: pixel_values MISMATCH (max_diff={pixel_max_diff:.6f})")

            # Compare loss_mask
            loss_mask_match = np.array_equal(cache_ex["loss_mask"], streaming_ex["loss_mask"])
            if not loss_mask_match:
                all_loss_mask_match = False
                print(f"  Example {i}: loss_mask MISMATCH")

            # Print success for each example
            if input_ids_match and attention_mask_match and pixel_values_match and loss_mask_match:
                print(f"  Example {i}: ✓ All fields match")

        # ====== Summary ======
        print("\n--- Summary ---")
        print(f"  input_ids match: {all_input_ids_match}")
        print(f"  attention_mask match: {all_attention_mask_match}")
        print(f"  pixel_values match: {all_pixel_values_match}")
        print(f"  loss_mask match: {all_loss_mask_match}")

        # Assert all match
        assert all_input_ids_match, "input_ids mismatch between cache and streaming modes"
        assert all_attention_mask_match, "attention_mask mismatch between cache and streaming modes"
        assert all_pixel_values_match, "pixel_values mismatch between cache and streaming modes"
        assert all_loss_mask_match, "loss_mask mismatch between cache and streaming modes"

        print("\n✓ Cache and streaming modes produce identical data!")


def test_streaming_dataset_basic():
    """Basic test for StreamingImageDataset functionality."""
    import asyncio
    from levanter.data.image import (
        ImageMixtureDatasetConfig,
        ConversationDatasetSourceConfig,
        StreamingImageDataset,
    )

    print("\n=== Test: Streaming Dataset Basic Functionality ===")

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save HF dataset to a temporary parquet file for this test
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)
        print(f"  Saved HF dataset to temporary parquet: {parquet_path}")

        # Create config with streaming
        config = ImageMixtureDatasetConfig(
            cache_dir=f"{tmpdir}/cache",
            configs={
                "train": ConversationDatasetSourceConfig(
                    train_urls=[f"file://{parquet_path}"],
                    validation_urls=[f"file://{parquet_path}"],
                    cache_dir=f"{tmpdir}/cache/train",
                ),
            },
            train_weights={"train": 1.0},
            processor=model_name,
            max_length=2048,
            use_cache=False,  # Use streaming mode
        )

        # Build streaming dataset
        datasets = config.training_sets()
        dataset = list(datasets.values())[0]

        # Verify it's a StreamingImageDataset
        assert isinstance(dataset, StreamingImageDataset), f"Expected StreamingImageDataset, got {type(dataset)}"

        # Test async methods
        async def run_tests():
            # Test async_len
            length = await dataset.async_len()
            print(f"  Dataset length: {length}")
            assert length > 0, "Dataset should have examples"

            # Test is_finite
            assert dataset.is_finite(), "Streaming dataset should be finite"

            # Test final_length_is_known (after loading)
            is_known = await dataset.final_length_is_known()
            assert is_known, "Final length should be known after loading"

            # Test get_batch
            batch = await dataset.get_batch([0, 1, 2])
            assert len(batch) == 3, f"Expected 3 examples, got {len(batch)}"

            # Verify batch structure
            for i, ex in enumerate(batch):
                assert "input_ids" in ex, f"Example {i} missing input_ids"
                assert "pixel_values" in ex, f"Example {i} missing pixel_values"
                assert "attention_mask" in ex, f"Example {i} missing attention_mask"
                assert "loss_mask" in ex, f"Example {i} missing loss_mask"
                assert "image_sizes" in ex, f"Example {i} missing image_sizes"

                # Verify shapes
                assert ex["input_ids"].shape == (2048,), f"Example {i} input_ids wrong shape: {ex['input_ids'].shape}"
                assert ex["attention_mask"].shape == (
                    2048,
                ), f"Example {i} attention_mask wrong shape: {ex['attention_mask'].shape}"
                assert ex["loss_mask"].shape == (2048,), f"Example {i} loss_mask wrong shape: {ex['loss_mask'].shape}"
                print(f"  Example {i}: input_ids={ex['input_ids'].shape}, pixel_values={ex['pixel_values'].shape}")

            return True

        result = asyncio.run(run_tests())
        assert result, "Streaming dataset tests failed"

        print("\n✓ Streaming dataset basic functionality works!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
