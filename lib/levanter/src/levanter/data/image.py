# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Image data processing module for vision-language models like LLaVA OneVision.

This module provides utilities for:
- Loading and preprocessing images from various sources (URLs, HuggingFace datasets)
- Processing conversation-format data with interleaved images and text
- Converting images to model-ready tensors with proper axes
- Batching and caching processed image-text pairs

Conversation Format Example:
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This image shows..."}
            ]
        }
    ],
    "images": ["path/to/image.jpg"]  # or PIL Images, or URLs
}
"""

import abc
import asyncio
import dataclasses
import json
import logging
import math
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, cast

import braceexpand
import datasets
import equinox as eqx
import fsspec
import numpy
import numpy as np
from draccus import field
from haliax import Axis, NamedArray

from levanter.data.mixture import MixtureDataset, StopStrategy
from jaxtyping import PRNGKeyArray
from typing_extensions import TypedDict

from levanter.compat.hf_checkpoints import load_processor
from levanter.data import AsyncDataset
from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import EpochDataset, MappedAsyncDataset
from levanter.data.sharded_datasource import (
    ShardedDataSource,
    UrlBackedShardedDataSource,
    WrappedHFDataSource,
    _sniff_format_for_dataset,
)
from levanter.store.cache import CacheOptions, TreeCache, build_or_load_cache
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag

# JAX-related imports for ImageDataLoader
# Note: JAX distributed should be initialized before importing this module in distributed environments
import haliax as hax
import jax
from haliax.partitioning import ResourceMapping
from jax.sharding import Mesh, PartitionSpec
from levanter.data.loader import DataLoader, DataLoaderIterator, _Batch
from levanter.schedule import IntSchedule
from levanter.shapes import NamedShapeSpec, ShapeSpec

silence_transformer_nag()
from transformers import (  # noqa: E402
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.image_processing_utils import select_best_resolution  # noqa: E402
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array  # noqa: E402
from transformers.processing_utils import MultiModalData, ProcessingKwargs, Unpack  # noqa: E402
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402
from transformers.video_utils import VideoInput  # noqa: E402

# Image loading dependencies - imported at module level for performance
from io import BytesIO  # noqa: E402

import requests  # noqa: E402
from PIL import Image  # noqa: E402

logger = logging.getLogger("levanter.data.image")


class ImageTextUrlDataSource(UrlBackedShardedDataSource[dict]):
    """
    Dataset for image-text pairs from various file formats (JSON, JSONL, Parquet).

    This data source reads image-text pairs where:
    - image_key: points to the image data (can be path, URL, bytes, or HF dict format)
    - text_key: points to the text description/caption

    Supports HuggingFace-style image formats:
    - {"bytes": <raw_bytes>}
    - {"path": "path/to/image.jpg"}
    - Direct path string or URL
    """

    def __init__(self, urls, image_key="image", text_key="text"):
        super().__init__(urls)
        self.image_key = image_key
        self.text_key = text_key

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            format = _sniff_format_for_dataset(url)
            match format:
                case ".jsonl":
                    for line in f:
                        if i >= row:
                            data = json.loads(line)
                            yield {
                                "image": data[self.image_key],
                                "text": data[self.text_key],
                            }
                        i += 1
                case ".json":
                    data = json.load(f)
                    for doc in data[row:]:
                        yield {
                            "image": doc[self.image_key],
                            "text": doc[self.text_key],
                        }
                case _:
                    raise ValueError(f"Unknown format {format}")


class ImageConversationUrlDataSource(UrlBackedShardedDataSource[dict]):
    """
    Dataset for conversation-format image-text data (VLM training format).

    This data source reads conversation data with interleaved images and text,
    used for vision-language model training like LLaVA.

    Expected data format:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": ["path/to/image.jpg"]  # or PIL Images, URLs, or bytes
    }
    """

    def __init__(self, urls, messages_key="messages", images_key="images"):
        super().__init__(urls)
        self.messages_key = messages_key
        self.images_key = images_key

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        format = _sniff_format_for_dataset(url)
        if format == ".parquet":
            # Handle parquet files
            import pyarrow.parquet as pq

            with fsspec.open(url, "rb") as f:
                table = pq.read_table(f)
                data = table.to_pydict()
                num_rows = table.num_rows
                for idx in range(row, num_rows):
                    yield {
                        "messages": data[self.messages_key][idx],
                        "images": data.get(self.images_key, [[]])[idx],
                    }
        else:
            with fsspec.open(url, "r", compression="infer") as f:
                match format:
                    case ".jsonl":
                        for line in f:
                            if i >= row:
                                data = json.loads(line)
                                yield {
                                    "messages": data[self.messages_key],
                                    "images": data.get(self.images_key, []),
                                }
                            i += 1
                    case ".json":
                        data = json.load(f)
                        for doc in data[row:]:
                            yield {
                                "messages": doc[self.messages_key],
                                "images": doc.get(self.images_key, []),
                            }
                    case _:
                        raise ValueError(f"Unknown format {format}")


class CustomVLMProcessor(ProcessorMixin):
    """
    Custom VLM processor that combines components from different sources.

    This allows using a different tokenizer (e.g., Qwen3-1.7B) while keeping
    the image/video processing from the original processor. Instead of mutating
    the original processor's tokenizer, this creates a new processor instance
    that properly combines the components.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    # Critical tokens for validation when combining processors
    CRITICAL_SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]
    CRITICAL_ROLE_TOKENS = ["assistant", "user", "system"]

    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor=None,
        *,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        num_image_tokens=None,
        vision_feature_select_strategy=None,
        vision_aspect_ratio=None,
        use_full_padded_tokens=False,
        **kwargs,
    ):
        """
        Initialize the custom processor with combined components.

        Args:
            image_processor: Image processor from the original VLM processor
            tokenizer: New tokenizer to use (e.g., from Qwen3-1.7B)
            video_processor: Optional video processor from the original VLM processor
            chat_template: Chat template for formatting conversations
            image_token: Token used for image placeholders
            video_token: Token used for video placeholders
            num_image_tokens: Number of tokens per image
            vision_feature_select_strategy: Strategy for selecting vision features
            vision_aspect_ratio: Aspect ratio mode (e.g., "anyres_max_9")
            use_full_padded_tokens: If True, use full padded token count (num_tiles * base_tokens)
                                   for Levanter. If False (default), use HF-style unpadded count.
        """
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_aspect_ratio = vision_aspect_ratio
        self.use_full_padded_tokens = use_full_padded_tokens
        self.image_token = image_token
        self.video_token = video_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(video_token)

        # ProcessorMixin validates arguments against the `attributes` class variable.
        # When video_processor is None, we need to exclude it from attributes to avoid type validation error.
        if video_processor is None:
            # Set instance-level attributes (overrides class attribute lookup)
            object.__setattr__(self, "attributes", ["image_processor", "tokenizer"])
            super().__init__(image_processor, tokenizer, chat_template=chat_template)
            self.video_processor = None  # Set manually after init
        else:
            super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    @classmethod
    def from_processor_and_tokenizer(
        cls,
        original_processor: ProcessorMixin,
        new_tokenizer: PreTrainedTokenizerBase,
        use_full_padded_tokens: bool = False,
    ) -> "CustomVLMProcessor":
        """
        Create a CustomVLMProcessor by combining original processor components with a new tokenizer.

        This factory method validates that the new tokenizer is compatible with the original
        processor's tokenizer, then creates a new processor instance that combines them.

        Args:
            original_processor: The original VLM processor (e.g., LlavaOnevisionProcessor)
            new_tokenizer: The new tokenizer to use (e.g., from Qwen3-1.7B)
            use_full_padded_tokens: If True, use full padded token count for Levanter.
                                   If False (default), use HF-style unpadded count.

        Returns:
            A new CustomVLMProcessor instance

        Raises:
            AssertionError: If tokenizers are incompatible (vocab_size, critical tokens, etc.)
            NotImplementedError: If the new tokenizer type is not supported
        """
        old_tokenizer = original_processor.tokenizer

        # Validate vocab_size matches
        assert old_tokenizer.vocab_size == new_tokenizer.vocab_size, (
            f"Tokenizer vocab size mismatch: processor has {old_tokenizer.vocab_size}, "
            f"new tokenizer has {new_tokenizer.vocab_size}"
        )

        # Validate critical special tokens have the same IDs
        for token in cls.CRITICAL_SPECIAL_TOKENS:
            old_id = old_tokenizer.convert_tokens_to_ids(token)
            new_id = new_tokenizer.convert_tokens_to_ids(token)
            assert old_id == new_id, (
                f"Critical special token '{token}' ID mismatch: " f"processor has {old_id}, new tokenizer has {new_id}"
            )

        # Validate role tokens have the same IDs
        for token in cls.CRITICAL_ROLE_TOKENS:
            old_id = old_tokenizer.convert_tokens_to_ids(token)
            new_id = new_tokenizer.convert_tokens_to_ids(token)
            assert old_id == new_id, (
                f"Critical role token '{token}' ID mismatch: " f"processor has {old_id}, new tokenizer has {new_id}"
            )

        # Validate eos_token_id matches
        assert old_tokenizer.eos_token_id == new_tokenizer.eos_token_id, (
            f"eos_token_id mismatch: processor has {old_tokenizer.eos_token_id}, "
            f"new tokenizer has {new_tokenizer.eos_token_id}"
        )

        # Detect Qwen3 tokenizer and use appropriate image/video tokens
        # Qwen3 has <|image_pad|>, <|video_pad|>, <think>, </think> tokens
        qwen3_image_token = "<|image_pad|>"
        qwen3_image_token_id = new_tokenizer.convert_tokens_to_ids(qwen3_image_token)
        is_qwen3 = qwen3_image_token_id != new_tokenizer.unk_token_id

        if is_qwen3:
            image_token = "<|image_pad|>"
            video_token = "<|video_pad|>"
            logger.info(f"Using Qwen3 tokens: image={image_token}, video={video_token}")
        else:
            raise NotImplementedError(f"Tokenizer {type(new_tokenizer).__name__} is not supported")

        # Read vision_aspect_ratio from image_processor first (may have been overridden),
        # then fall back to processor-level attribute
        vision_aspect_ratio = getattr(
            original_processor.image_processor, "vision_aspect_ratio",
            getattr(original_processor, "vision_aspect_ratio", None)
        )

        result = cls(
            image_processor=original_processor.image_processor,
            tokenizer=new_tokenizer,
            video_processor=getattr(original_processor, "video_processor", None),
            chat_template=getattr(original_processor, "chat_template", None),
            image_token=image_token,
            video_token=video_token,
            num_image_tokens=getattr(original_processor, "num_image_tokens", None),
            vision_feature_select_strategy=getattr(original_processor, "vision_feature_select_strategy", None),
            vision_aspect_ratio=vision_aspect_ratio,
            use_full_padded_tokens=use_full_padded_tokens,
        )

        logger.info(
            f"Created CustomVLMProcessor with {type(new_tokenizer).__name__} "
            f"(vocab_size={new_tokenizer.vocab_size})"
        )

        return result

    def __call__(
        self,
        images=None,
        text=None,
        **kwargs,
    ):
        """
        Process images and text, expanding image placeholders to correct token count.

        This method:
        1. Processes images with the image_processor
        2. Expands <image> placeholders in text to num_image_tokens copies of image_token
        3. Tokenizes the expanded text

        Args:
            images: Image input (PIL Image, list of PIL Images, etc.)
            text: Text input (str or list of str)
            **kwargs: Additional arguments passed to tokenizer (return_tensors, padding, etc.)

        Returns:
            BatchFeature with processed images and tokenized text
        """
        from transformers import BatchFeature

        # 1. Process images with image_processor
        # Filter out tokenizer-specific kwargs that shouldn't go to image_processor
        image_processor_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("padding", "max_length", "truncation", "return_tensors", "padding_mode")
        }
        image_inputs = {}
        if images is not None:
            # Auto-wrap flat image list into nested list based on <image> token count in text.
            # This enables the image processor to correctly handle multi-image samples.
            # When multiple images belong to one text sample, the processor outputs 1 patch
            # per image (base only) instead of multiple patches per image (anyres).
            #
            # Example:
            #   images = [img1, img2, img3]  # flat
            #   text = "<image> <image> <image> describe these"
            #   → images = [[img1, img2, img3]]  # nested for one sample
            is_flat_list = (
                isinstance(images, (list, tuple))
                and len(images) > 0
                and not isinstance(images[0], (list, tuple))
            )

            if is_flat_list and text is not None:
                # Count <image> tokens in each text sample (using the original <image> placeholder)
                # Chat templates produce <image> placeholders which we later convert
                original_image_token = "<image>"
                if isinstance(text, str):
                    text_list = [text]
                else:
                    text_list = text
                num_images_per_sample = [t.count(original_image_token) for t in text_list]

                # Validate total images match before reorganizing
                total_expected = sum(num_images_per_sample)
                if total_expected == len(images):
                    # Reorganize flat list into nested list
                    images_nested = []
                    idx = 0
                    for num_img in num_images_per_sample:
                        images_nested.append(list(images[idx : idx + num_img]))
                        idx += num_img
                    images = images_nested

            image_inputs = self.image_processor(images, **image_processor_kwargs)
            # Convert to dict if BatchFeature
            if hasattr(image_inputs, "data"):
                image_inputs = dict(image_inputs.data)

        # Extract actual patch counts from HF processor output (ground truth for token count)
        # This ensures token count matches pixel_values, avoiding mismatches from different
        # resolution selection algorithms between HF processor and _get_number_of_features
        actual_patch_counts = None
        if images is not None:
            pixel_values = image_inputs.get("pixel_values")
            if pixel_values is not None:
                if isinstance(pixel_values, (list, tuple)):
                    # Multiple images: each can be 4D (num_patches, C, H, W) or 3D (C, H, W)
                    actual_patch_counts = []
                    for i, pv in enumerate(pixel_values):
                        if len(pv.shape) == 4:
                            actual_patch_counts.append(pv.shape[0])  # num_patches
                        else:
                            actual_patch_counts.append(1)  # single patch (3D: C, H, W)
                elif hasattr(pixel_values, 'shape'):
                    if len(pixel_values.shape) == 4:
                        # 4D: (num_patches, C, H, W)
                        actual_patch_counts = [pixel_values.shape[0]]
                    else:
                        # 3D: (C, H, W) - single patch
                        actual_patch_counts = [1]

        # 2. Expand image tokens in text
        if text is not None and images is not None:
            text = self._expand_image_tokens(text, image_inputs, actual_patch_counts)

        # 3. Tokenize expanded text
        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors", None),
                padding=kwargs.get("padding", False),
                max_length=kwargs.get("max_length"),
                truncation=kwargs.get("truncation", False),
            )
            # Convert to dict if BatchEncoding
            if hasattr(text_inputs, "data"):
                text_inputs = dict(text_inputs.data)

        return BatchFeature(data={**image_inputs, **text_inputs})

    def _expand_image_tokens(
        self,
        text,
        image_inputs: dict,
        actual_patch_counts: Optional[List[int]] = None,
    ):
        """
        Expand <image> placeholders to num_image_tokens copies of image_token with vision markers.

        Key insight: Chat template produces "<image>" but we want:
        "<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>"

        For single image with anyres enabled, calculates actual token count based on
        image dimensions and grid pinpoints. For multi-image or disable_anyres,
        uses fixed base token count.

        Args:
            text: Text input (str or list of str)
            image_inputs: Dict with processed image data (pixel_values, image_sizes, etc.)
            actual_patch_counts: Optional list of actual patch counts from HF processor output.
                               When provided, this is used as ground truth for token count
                               to ensure consistency with pixel_values shape.

        Returns:
            Expanded text with image tokens properly expanded
        """
        if isinstance(text, str):
            text = [text]
            was_single = True
        else:
            was_single = False

        # The chat template uses "<image>" as placeholder (standard LLaVA format)
        CHAT_TEMPLATE_PLACEHOLDER = "<image>"

        # Get image_sizes for anyres calculation
        image_sizes = image_inputs.get("image_sizes")

        # Determine if anyres is enabled
        is_anyres_enabled = (
            self.vision_aspect_ratio is not None and self.vision_aspect_ratio != "single"
        )

        # Get patch size from image processor
        patch_size = getattr(self.image_processor, "size", {}).get("height", 384)

        expanded = []
        image_idx = 0  # Track which image we're processing across all samples

        for sample in text:
            # Count images in this sample (from chat template's <image>)
            num_images_in_sample = sample.count(CHAT_TEMPLATE_PLACEHOLDER)
            if num_images_in_sample == 0:
                expanded.append(sample)
                continue

            result = sample
            # Vision markers to wrap image tokens
            vision_start = "<|vision_start|>"
            vision_end = "<|vision_end|>"

            for _ in range(num_images_in_sample):
                # Determine tokens for this image
                is_single_image = num_images_in_sample == 1
                base_tokens = self.num_image_tokens or 576

                # For Levanter (use_full_padded_tokens=True): use actual_patch_counts * base_tokens
                # For HF (use_full_padded_tokens=False): use _get_number_of_features for unpadded count
                if self.use_full_padded_tokens:
                    # Levanter mode: full padded tokens (unpadding handled via grid_mask/unpad_indices)
                    if is_anyres_enabled and actual_patch_counts is not None and image_idx < len(actual_patch_counts):
                        # actual_patch_counts[i] = number of patches (tiles) from HF processor
                        tokens_per_image = actual_patch_counts[image_idx] * base_tokens
                    elif is_anyres_enabled and is_single_image and image_sizes is not None:
                        orig_height, orig_width = image_sizes[image_idx]
                        tokens_per_image = self._get_number_of_features(
                            orig_height, orig_width, patch_size, patch_size
                        )
                    else:
                        tokens_per_image = base_tokens
                elif is_anyres_enabled and is_single_image and image_sizes is not None:
                    # Fallback: calculate based on image dimensions (may not match HF processor)
                    orig_height, orig_width = image_sizes[image_idx]
                    tokens_per_image = self._get_number_of_features(
                        orig_height, orig_width, patch_size, patch_size
                    )
                else:
                    # Multi-image or disable_anyres: use base tokens
                    # Levanter doesn't use newline separator
                    tokens_per_image = base_tokens

                # Replace <image> with <|vision_start|><|image_pad|>*N<|vision_end|>
                result = result.replace(
                    CHAT_TEMPLATE_PLACEHOLDER,  # "<image>" from chat template
                    vision_start + (self.image_token * tokens_per_image) + vision_end,
                    1,  # Replace only first occurrence
                )
                image_idx += 1
            expanded.append(result)

        return expanded[0] if was_single else expanded

    def _get_unpadded_features(
        self,
        orig_height: int,
        orig_width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int,
    ) -> tuple[int, int]:
        """
        Calculate unpadded features based on original aspect ratio.
        Mirrors HF's LlavaOnevisionProcessor._get_unpadded_features.

        Args:
            orig_height/orig_width: Original image dimensions
            patches_height/patches_width: Patches per tile dimension (e.g., 27)
            scale_height/scale_width: Number of tiles in each dimension

        Returns:
            Tuple of (unpadded_features, newline_features)
        """
        import math

        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = orig_width / orig_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            # Wider image - remove top/bottom padding
            scale_factor = current_width / orig_width
            new_height = int(round(orig_height * scale_factor, 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            # Taller image - remove left/right padding
            scale_factor = current_height / orig_height
            new_width = int(round(orig_width * scale_factor, 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        # Apply anyres_max limit if specified
        if self.vision_aspect_ratio and self.vision_aspect_ratio.startswith("anyres_max_"):
            max_num_patches = int(self.vision_aspect_ratio.split("anyres_max_")[-1])
            ratio = math.sqrt(
                current_height * current_width / (max_num_patches * patches_height**2)
            )
            if ratio > 1.1:
                unpadded_features = int(current_height // ratio) * int(current_width // ratio)
                newline_features = int(current_height // ratio)

        return unpadded_features, newline_features

    def _get_number_of_features(
        self,
        orig_height: int,
        orig_width: int,
        height: int,
        width: int,
    ) -> int:
        """
        Calculate actual image tokens for single image with anyres.
        Mirrors HF's LlavaOnevisionProcessor._get_number_of_features.

        Args:
            orig_height/orig_width: Original image dimensions
            height/width: Patch size (e.g., 384)

        Returns:
            Total number of image tokens. Depends on use_full_padded_tokens:
            - True (Levanter): num_tiles * base_tokens (full padded, unpadding via grid_mask)
            - False (HF): base_features + unpadded_features (HF-style spatial unpadding)
        """
        import math
        from transformers.models.llava_onevision.image_processing_llava_onevision import (
            select_best_resolution,
        )

        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        # 1. Select best resolution from grid pinpoints
        height_best, width_best = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height = height_best // height
        scale_width = width_best // width

        if self.use_full_padded_tokens:
            # Levanter mode: full padded tokens, unpadding handled via grid_mask/unpad_indices
            num_sub_patches = scale_height * scale_width
            num_tiles = num_sub_patches + 1  # sub-patches + base tile
            return num_tiles * self.num_image_tokens
        else:
            # HF mode: calculate unpadded feature count (matches HF's spatial unpadding)
            patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
            unpadded_features, newline_features = self._get_unpadded_features(
                orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
            )
            return self.num_image_tokens + unpadded_features 

    def _compute_unpad_indices(
        self,
        orig_height: int,
        orig_width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int,
        features_per_patch: int,
    ) -> np.ndarray:
        """
        Compute indices to reorder Levanter's padded features to HF's unpadded order.

        HF's pack_image_features applies spatial unpadding based on original image aspect ratio.
        This function computes the mapping from HF's feature positions to Levanter's sequential
        feature layout.

        Args:
            orig_height: Original image height
            orig_width: Original image width
            patches_height: Number of patches per tile in height (e.g., 27)
            patches_width: Number of patches per tile in width (e.g., 27)
            scale_height: Number of tiles in height (e.g., 3 for 3x3 grid)
            scale_width: Number of tiles in width (e.g., 3 for 3x3 grid)
            features_per_patch: Features per patch/tile (e.g., 729)

        Returns:
            unpad_indices: Array of shape (num_unpadded_features,) where
                          unpad_indices[i] = Levanter index for HF position i
        """
        # Base features are identity mapping (base patch is always first)
        base_indices = np.arange(features_per_patch)

        # Grid spatial dimensions after combining all tiles
        curr_height = patches_height * scale_height  # e.g., 81 for 3x3 grid of 27x27 patches
        curr_width = patches_width * scale_width

        # Compute unpadding bounds based on original aspect ratio
        # This matches HF's unpad_image logic
        original_aspect_ratio = orig_width / orig_height
        current_aspect_ratio = curr_width / curr_height

        if original_aspect_ratio > current_aspect_ratio:
            # Wider image - remove top/bottom padding
            scale_factor = curr_width / orig_width
            new_height = int(round(orig_height * scale_factor, 7))
            padding = (curr_height - new_height) // 2
            row_start = padding
            row_end = curr_height - padding  # Symmetric padding like HF
            col_start = 0
            col_end = curr_width
        else:
            # Taller image - remove left/right padding
            scale_factor = curr_height / orig_height
            new_width = int(round(orig_width * scale_factor, 7))
            padding = (curr_width - new_width) // 2
            row_start = 0
            row_end = curr_height
            col_start = padding
            col_end = curr_width - padding  # Symmetric padding like HF

        # Build mapping from HF grid position to Levanter grid index
        # HF order: row-major through unpadded region
        # Levanter order: patch-by-patch (tile-by-tile), then row-major within each patch
        grid_indices = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                # Convert global (row, col) to Levanter's patch-based index
                # Which tile (patch) does this position belong to?
                tile_row = row // patches_height
                tile_col = col // patches_width
                # Local position within the tile
                local_row = row % patches_height
                local_col = col % patches_width

                # Tile index in row-major order (0-indexed grid patch, excluding base)
                tile_idx = tile_row * scale_width + tile_col
                # Local feature index within the tile
                local_idx = local_row * patches_width + local_col

                # Levanter index: base_features + tile_idx * features_per_patch + local_idx
                # +1 because tile_idx=0 is the first grid tile, but Levanter's patch 0 is the base
                lev_idx = features_per_patch + tile_idx * features_per_patch + local_idx
                grid_indices.append(lev_idx)

        return np.concatenate([base_indices, np.array(grid_indices, dtype=np.int32)])

    def compute_unpad_indices(
        self,
        image_sizes: list,
        height: int,
        width: int,
        max_num_features: int,
    ) -> np.ndarray:
        """
        Compute unpad indices for a batch of images.

        Args:
            image_sizes: List of (orig_height, orig_width) tuples for each image
            height: Processed tile height (e.g., 384)
            width: Processed tile width (e.g., 384)
            max_num_features: Maximum number of features to pad to

        Returns:
            unpad_indices: Array of shape (batch, max_num_features) padded with zeros
        """
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        # Handle single resolution mode (no grid patches, only base)
        # Check both empty grid_pinpoints and vision_aspect_ratio="single"
        is_single_resolution = (
            not image_grid_pinpoints
            or (self.vision_aspect_ratio and not self.vision_aspect_ratio.startswith("anyres"))
        )
        if is_single_resolution:
            # For single resolution, return identity mapping (no reordering needed)
            batch_size = len(image_sizes)
            return np.zeros((batch_size, max_num_features), dtype=np.int32)

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))

        batch_indices = []
        for orig_height, orig_width in image_sizes:
            # Find best resolution for this image
            height_best_resolution, width_best_resolution = select_best_resolution(
                [orig_height, orig_width], image_grid_pinpoints
            )
            scale_height = height_best_resolution // height
            scale_width = width_best_resolution // width

            # Compute unpad indices for this image
            indices = self._compute_unpad_indices(
                orig_height,
                orig_width,
                patches_height,
                patches_width,
                scale_height,
                scale_width,
                self.num_image_tokens,
            )
            batch_indices.append(indices)

        # Pad all indices to max_num_features
        padded_indices = np.zeros((len(batch_indices), max_num_features), dtype=np.int32)
        for i, indices in enumerate(batch_indices):
            padded_indices[i, : len(indices)] = indices

        return padded_indices


def expand_urls_with_folder_support(urls: List[str]) -> List[str]:
    """Expand URLs/paths to a list of file paths.

    Supports:
    - Single file paths: /path/to/file.parquet
    - Glob patterns: /path/to/*.parquet
    - Directories: /path/to/folder/ (will find all *.parquet files recursively)
    - file:// prefixed paths: file:///path/to/folder/
    - Brace expansion: /path/to/{train,val}*.parquet

    Args:
        urls: List of URLs/paths that may include directories, globs, or brace patterns

    Returns:
        List of expanded file paths
    """

    def expand_single_path(url: str) -> List[str]:
        """Expand a single path/url to a list of file paths."""
        # Handle file:// prefix
        if url.startswith("file://"):
            local_path = url[7:]  # Remove file:// prefix
            prefix = "file://"
        else:
            local_path = url
            prefix = ""

        # Check if it's a directory (without glob pattern)
        if os.path.isdir(local_path):
            # Find all parquet files in the directory (recursively)
            parquet_files = []
            for root, dirs, files in os.walk(local_path):
                for f in files:
                    if f.endswith(".parquet"):
                        full_path = os.path.join(root, f)
                        parquet_files.append(f"{prefix}{full_path}")
            parquet_files.sort()  # Sort for deterministic ordering
            if parquet_files:
                logger.info(f"Found {len(parquet_files)} parquet files in directory: {local_path}")
            else:
                logger.warning(f"No parquet files found in directory: {local_path}")
            return parquet_files
        elif "*" in local_path:
            # Use fsspec for glob expansion
            fs, path = fsspec.core.url_to_fs(url)
            globbed = fs.glob(path)
            if globbed:
                # Add protocol prefix back (fs.glob returns paths without protocol)
                proto = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
                return [f"{proto}://{p}" if proto else p for p in globbed]
            return [url]
        else:
            # Single file
            return [url]

    result = []
    for pat in urls:
        for url in braceexpand.braceexpand(pat):
            result.extend(expand_single_path(url))

    return result


# Type definitions for conversation data
ConversationMessage = TypedDict(
    "ConversationMessage",
    {
        "role": str,  # "user", "assistant", "system"
        "content": List[Dict[str, Any]],  # [{"type": "image"}, {"type": "text", "text": "..."}]
    },
)

ConversationDict = TypedDict(
    "ConversationDict",
    {
        "messages": List[ConversationMessage],
        "images": List[Any],  # List of images (PIL, paths, URLs, or bytes)
    },
    total=False,
)


# Type definitions for processed image-text data
# pixel_values and image_sizes are optional to support text-only examples
class ImageTextDict(TypedDict, total=False):
    """Processed image-text data for VLM training.

    For text-only examples, pixel_values and image_sizes will be None.
    """

    pixel_values: Optional[np.ndarray]  # (TOTAL_PATCHES, channels, height, width) - FIXED shape, padded
    input_ids: np.ndarray  # (seq_len,)
    attention_mask: np.ndarray  # (seq_len,)
    image_sizes: Optional[np.ndarray]  # (num_images, 2) or None - original image sizes (H, W)
    loss_mask: np.ndarray  # (seq_len,) float32 - 1.0 for compute loss, 0.0 for ignore
    # Grid mask for fixed-shape processing - indicates which patches are valid (not padding)
    grid_mask: Optional[np.ndarray]  # (TOTAL_PATCHES,) boolean - True for valid patches
    # Unpad indices for anyres processing
    unpad_indices: Optional[np.ndarray]  # (num_image_tokens,) - indices for unpadding image features
    # Actual number of unpadded features (before padding unpad_indices)
    num_unpadded_features: Optional[int]  # scalar - used for combined_mask computation
    # Precomputed validity mask for attention (1 for valid text/image, 0 for padding)
    combined_mask: Optional[np.ndarray]  # (seq_len,) int32 - validity mask
    # Precomputed position IDs (cumsum of combined_mask - 1, clamped to 0)
    position_ids: Optional[np.ndarray]  # (seq_len,) int32 - position IDs


ImageTextDict_exemplar: ImageTextDict = {
    "pixel_values": np.zeros((1, 3, 384, 384), dtype=np.float32),
    "input_ids": np.zeros((1,), dtype=np.int32),
    "attention_mask": np.zeros((1,), dtype=np.int32),
    "image_sizes": np.zeros((1, 2), dtype=np.int32),
    "loss_mask": np.zeros((1,), dtype=np.float32),
    "grid_mask": None,  # Always included, may be None
    "unpad_indices": None,  # Always included, may be None
    "num_unpadded_features": None,  # Always included, may be None
    "combined_mask": None,  # Always included, may be None
    "position_ids": None,  # Always included, may be None
}


def load_image_from_path_or_url(path_or_url: str) -> Image.Image:
    """Load an image from a local path, URL, or cloud storage.

    Args:
        path_or_url: Local file path, URL, or cloud storage path (gs://, s3://) to the image

    Returns:
        PIL Image in RGB format
    """
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    elif path_or_url.startswith(("gs://", "s3://")):
        with fsspec.open(path_or_url, "rb") as f:
            image = Image.open(f)
            image.load()
    else:
        image = Image.open(path_or_url)

    return image.convert("RGB")


def load_image(image_data: Any) -> Image.Image:
    """Load an image from various formats.

    Args:
        image_data: Can be PIL Image, numpy array, path string, URL, or HF dict with bytes

    Returns:
        PIL Image in RGB format
    """
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    elif isinstance(image_data, str):
        return load_image_from_path_or_url(image_data)
    elif isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data).convert("RGB")
    elif isinstance(image_data, dict):
        if "bytes" in image_data:
            # HuggingFace dataset format
            return Image.open(BytesIO(image_data["bytes"])).convert("RGB")
        elif "path" in image_data:
            return load_image_from_path_or_url(image_data["path"])
        else:
            raise ValueError(f"Unknown image dict format: {image_data.keys()}")
    else:
        raise ValueError(f"Unsupported image type: {type(image_data)}")


def _extract_anyres_params(
    processor: ProcessorMixin,
) -> Tuple[Optional[List[List[int]]], int, Optional[int], Optional[int]]:
    """Extract grid_pinpoints and related params from HF processor for anyres support.

    Args:
        processor: HuggingFace processor (e.g., LlavaOnevisionProcessor)

    Returns:
        Tuple of (grid_pinpoints, patch_size, vision_feature_height, max_num_patches)
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None, 384, None, None

    grid_pinpoints = getattr(image_processor, "image_grid_pinpoints", None)
    size_dict = getattr(image_processor, "size", {})
    patch_size = size_dict.get("height", 384) if isinstance(size_dict, dict) else 384
    vision_feature_height = patch_size // 14
    max_num_patches = None

    # Try to get max_num_patches from vision_aspect_ratio (LLaVA-specific)
    vision_aspect_ratio = getattr(image_processor, "vision_aspect_ratio", None)

    # Handle disable_anyres case: vision_aspect_ratio="single" means no grid patches
    if vision_aspect_ratio == "single" or (
        vision_aspect_ratio and isinstance(vision_aspect_ratio, str) and not vision_aspect_ratio.startswith("anyres")
    ):
        max_num_patches = 0  # No grid patches, only base patch
    elif vision_aspect_ratio and isinstance(vision_aspect_ratio, str) and "anyres_max_" in vision_aspect_ratio:
        try:
            max_num_patches = int(vision_aspect_ratio.split("anyres_max_")[-1])
        except (ValueError, IndexError):
            pass

    # Handle empty grid_pinpoints (shouldn't happen with proper config)
    if max_num_patches is None and grid_pinpoints is not None and len(grid_pinpoints) == 0:
        max_num_patches = 0  # No grid patches, only base patch
    # Fallback: compute from grid_pinpoints if available
    elif max_num_patches is None and grid_pinpoints:
        max_resolution = max(max(h, w) for h, w in grid_pinpoints)
        max_patches_per_dim = max_resolution // patch_size
        max_num_patches = max_patches_per_dim * max_patches_per_dim  # +1 for base is added in _pad_pixel_values

    return grid_pinpoints, patch_size, vision_feature_height, max_num_patches


class BatchImageProcessor(BatchProcessor[Dict[str, Any], ImageTextDict]):
    """
    A batch processor that converts conversation-format data into model-ready inputs.

    This processor handles the conversation format used by VLMs like LLaVA:
    - Applies chat template to convert messages to text with image placeholders
    - Processes images using the HuggingFace processor
    - Creates loss_mask for training (1.0 for assistant tokens, 0.0 for others)

    Input format:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": [<image_data>]  # PIL, path, URL, or HF bytes dict
    }

    Output format (ImageTextDict):
    {
        "pixel_values": np.ndarray or None,
            # Shape: (TOTAL_PATCHES, C, H, W) where TOTAL_PATCHES = max_num_patches + 1
            # Preprocessed image patches ready for the vision encoder. Padded to fixed size
            # for JIT compatibility. For single image: includes base patch + anyres grid patches.
            # For multiple images: only base patches (one per image). None for text-only examples.

        "input_ids": np.ndarray,
            # Shape: (seq_len,) dtype: int32
            # Tokenized text sequence with image placeholder tokens inserted where images appear.
            # The image placeholder token (e.g., <|image_pad|>) is repeated for each image feature.

        "attention_mask": np.ndarray,
            # Shape: (seq_len,) dtype: int32
            # Binary mask indicating valid tokens (1) vs padding tokens (0).
            # Used to prevent attention to padding positions.

        "image_sizes": np.ndarray or None,
            # Shape: (num_images, 2) dtype: int32
            # Original image dimensions as (height, width) for each image.
            # Used by the model for spatial unpadding in anyres processing. None for text-only.

        "loss_mask": np.ndarray,
            # Shape: (seq_len,) dtype: float32
            # Training loss mask for causal language modeling. 1.0 for assistant response
            # tokens that should contribute to the loss; 0.0 for all other tokens
            # (system, user, special) that should be ignored during training.

        "grid_mask": np.ndarray or None,
            # Shape: (TOTAL_PATCHES,) dtype: bool
            # Boolean mask indicating which patches are real (True) vs padding (False).
            # Enables fixed-shape tensors for JIT while tracking actual patch count.
            # None if max_num_patches is not configured.

        "unpad_indices": np.ndarray or None,
            # Shape: (num_image_tokens,) dtype: int32
            # Index mapping from HuggingFace's unpadded feature order to Levanter's padded order.
            # Used to reorder vision features after encoding to match HF's spatial unpadding.
            # Only computed for single-image anyres case; None otherwise.
    }
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 2048,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        add_generation_prompt: bool = False,
        mask_prompt: bool = True,
        max_num_patches: int = 9,
        override_resources: Optional[Dict[str, Any]] = None,
        # Parameters for computing grid_mask for JIT-compatible VLM training
        grid_pinpoints: Optional[List[List[int]]] = None,
        patch_size: int = 384,
        vision_feature_height: Optional[int] = None,
        disable_anyres: bool = False,
        use_full_padded_tokens: bool = True,
    ):
        """
        Initialize the BatchImageProcessor.

        Args:
            processor: HuggingFace processor (e.g., AutoProcessor.from_pretrained(...))
            tokenizer: Optional tokenizer to replace the processor's tokenizer.
                       Use this to ensure tokenization matches the LLM's tokenizer (e.g., Qwen3-1.7B).
                       If provided, critical special tokens will be verified for consistency.
            max_length: Maximum sequence length for tokenization
            padding: Whether to pad sequences to max_length
            messages_key: Key for messages list in input dictionaries
            images_key: Key for images list in input dictionaries
            add_generation_prompt: Whether to add generation prompt at the end
            mask_prompt: Whether to mask non-assistant tokens (set loss_mask to 0.0)
            override_resources: Optional resource overrides
            grid_pinpoints: List of grid resolutions for anyres processing, e.g., [[384,384], [768,384], ...]
            patch_size: Size of each image patch (default 384)
            vision_feature_height: Vision encoder output tokens per spatial dim (e.g., 27 for 384/14)
            max_num_patches: Maximum number of patches for anyres constraint (e.g., 9 for anyres_max_9)
            disable_anyres: If True, only use base patch for single images (no anyres sub-patches)
            use_full_padded_tokens: If True, use full padded token count (num_tiles × base_tokens)
                                   for Levanter mode. If False, use HF-style unpadded count.
        """
        # Store model names for process-local recreation (needed for Ray distributed workers)
        # HuggingFace tokenizers cannot be safely pickled across process boundaries due to
        # Rust backend state issues ("Already borrowed" error), so we store names and recreate
        self._processor_name_or_path: Optional[str] = getattr(processor, 'name_or_path', None)
        if self._processor_name_or_path is None:
            # Try to get from tokenizer or image_processor
            self._processor_name_or_path = getattr(getattr(processor, 'tokenizer', None), 'name_or_path', None)
        self._tokenizer_name_or_path: Optional[str] = getattr(tokenizer, 'name_or_path', None) if tokenizer else None

        # Store the actual processor (will be excluded from pickle in __getstate__)
        self._processor = processor
        self._custom_tokenizer = tokenizer

        self.max_length = max_length
        self.padding = padding
        self.messages_key = messages_key
        self.images_key = images_key
        self.add_generation_prompt = add_generation_prompt
        self.mask_prompt = mask_prompt
        self.override_resources = override_resources
        self.disable_anyres = disable_anyres
        self.use_full_padded_tokens = use_full_padded_tokens

        # Parameters for computing grid_mask for JIT-compatible VLM training
        self.grid_pinpoints = grid_pinpoints
        self.patch_size = patch_size
        self.vision_feature_height = vision_feature_height
        self.max_num_patches = max_num_patches

        # Override processor's num_image_tokens when vision_feature_height is provided
        # This ensures the processor expands image placeholders to the correct number of tokens
        # Critical: HF processor defaults to 729 (27x27), but custom vision encoders may output
        # different sizes (e.g., 576 = 24x24 for SigLIP with patch_size=16)
        if vision_feature_height is not None:
            num_image_tokens = vision_feature_height * vision_feature_height
            # Always set num_image_tokens - don't check hasattr, just set it
            processor.num_image_tokens = num_image_tokens

        # Configure processor for disable_anyres mode
        # This controls how many image tokens the processor expands <image> placeholders to
        # Without this, the processor uses its default anyres config and produces more tokens
        if disable_anyres:
            # Set single resolution mode in processor AND image_processor
            # IMPORTANT: Must set on both because CustomVLMProcessor.from_processor_and_tokenizer
            # reads from image_processor first, then falls back to processor
            if hasattr(processor, 'image_grid_pinpoints'):
                processor.image_grid_pinpoints = grid_pinpoints if grid_pinpoints else [[patch_size, patch_size]]
            if hasattr(processor, 'vision_aspect_ratio'):
                processor.vision_aspect_ratio = "single"
            # Also set on image_processor (critical for CustomVLMProcessor)
            if hasattr(processor, 'image_processor'):
                if hasattr(processor.image_processor, 'image_grid_pinpoints'):
                    processor.image_processor.image_grid_pinpoints = grid_pinpoints if grid_pinpoints else [[patch_size, patch_size]]
                if hasattr(processor.image_processor, 'vision_aspect_ratio'):
                    processor.image_processor.vision_aspect_ratio = "single"

        # Pre-compute grid_pinpoints arrays for vectorized _compute_grid_shape
        # Note: empty list [] is treated as no grid pinpoints (disable_anyres case)
        if grid_pinpoints is not None and len(grid_pinpoints) > 0:
            self._grid_h = np.array([p[0] for p in grid_pinpoints], dtype=np.float64)
            self._grid_w = np.array([p[1] for p in grid_pinpoints], dtype=np.float64)
            self._grid_area = self._grid_h * self._grid_w
        else:
            self._grid_h = None
            self._grid_w = None
            self._grid_area = None

        # Create a custom processor with the new tokenizer if specified
        if tokenizer is not None:
            self._processor = CustomVLMProcessor.from_processor_and_tokenizer(
                processor, tokenizer,
                use_full_padded_tokens=True,
            )

        # Cache padding mode for __call__
        self._padding_mode = "max_length" if self.padding else False

        # Pre-compute token IDs (safe to pickle since they're just int/np.array values)
        self._cached_token_ids = self._compute_token_ids()

    @property
    def processor(self) -> ProcessorMixin:
        """Lazily access the processor, recreating it if needed after unpickling."""
        if self._processor is None:
            self._recreate_processor()
        return self._processor

    def _recreate_processor(self) -> None:
        """Recreate processor and tokenizer in current process after unpickling.

        This is called automatically when accessing self.processor after the object
        has been unpickled in a Ray worker process. The tokenizer cannot be pickled
        due to Rust backend state issues, so we recreate it from the stored model name.
        """
        from levanter.compat.hf_checkpoints import load_processor, load_tokenizer

        if self._processor_name_or_path is None:
            raise RuntimeError(
                "Cannot recreate processor: _processor_name_or_path is None. "
                "This may happen if the original processor didn't have a name_or_path attribute."
            )

        # Recreate the base processor
        self._processor = load_processor(self._processor_name_or_path, trust_remote_code=True)

        # Apply vision_feature_height override if needed
        if self.vision_feature_height is not None:
            num_image_tokens = self.vision_feature_height * self.vision_feature_height
            self._processor.num_image_tokens = num_image_tokens

        # Apply disable_anyres configuration if needed
        # IMPORTANT: Must set on both processor AND image_processor because
        # CustomVLMProcessor.from_processor_and_tokenizer reads from image_processor first
        if self.disable_anyres:
            grid_config = self.grid_pinpoints if self.grid_pinpoints else [[self.patch_size, self.patch_size]]
            if hasattr(self._processor, 'image_grid_pinpoints'):
                self._processor.image_grid_pinpoints = grid_config
            if hasattr(self._processor, 'vision_aspect_ratio'):
                self._processor.vision_aspect_ratio = "single"
            # Also set on image_processor (critical for CustomVLMProcessor)
            if hasattr(self._processor, 'image_processor'):
                if hasattr(self._processor.image_processor, 'image_grid_pinpoints'):
                    self._processor.image_processor.image_grid_pinpoints = grid_config
                if hasattr(self._processor.image_processor, 'vision_aspect_ratio'):
                    self._processor.image_processor.vision_aspect_ratio = "single"

        # Recreate custom tokenizer if it was originally provided
        if self._tokenizer_name_or_path is not None:
            custom_tokenizer = load_tokenizer(self._tokenizer_name_or_path, trust_remote_code=True)
            self._processor = CustomVLMProcessor.from_processor_and_tokenizer(
                self._processor, custom_tokenizer,
                use_full_padded_tokens=True,
            )

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling, excluding unpicklable tokenizer/processor objects.

        HuggingFace tokenizers with Rust backends cannot be safely pickled across process
        boundaries (causes "RuntimeError: Already borrowed"). We exclude them and recreate
        them lazily in the worker process using the stored model names.
        """
        state = self.__dict__.copy()
        # Remove unpicklable processor/tokenizer objects - they will be recreated lazily
        state['_processor'] = None
        state['_custom_tokenizer'] = None
        # Note: _cached_token_ids is kept since it's just int/np.array values (safe to pickle)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # _processor is None, will be lazily recreated on first access via self.processor

    def _compute_token_ids(self) -> Dict[str, Any]:
        """Compute token IDs for loss mask creation.

        Creates a temporary tokenizer to avoid concurrent access issues with
        the shared processor's tokenizer (HF fast tokenizers are not thread-safe).
        Returns dict with im_start_id, im_end_id, num_assistant_tokens, and assistant_token_ids_array.
        These are safe to pickle since they're just int/np.array values.
        """
        from levanter.compat.hf_checkpoints import load_tokenizer

        # Create a temporary tokenizer instance to avoid "Already borrowed" error
        # The shared processor.tokenizer may be accessed concurrently by other threads
        tokenizer_path = self._tokenizer_name_or_path or self._processor_name_or_path
        if tokenizer_path is None:
            raise RuntimeError("Cannot compute token IDs: no tokenizer path available")

        temp_tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)

        im_start_id = temp_tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = temp_tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_ids = temp_tokenizer.encode("assistant", add_special_tokens=False)
        return {
            "im_start_id": im_start_id,
            "im_end_id": im_end_id,
            "num_assistant_tokens": len(assistant_ids),
            "assistant_token_ids_array": np.array(assistant_ids, dtype=np.int32),
        }

    def _get_cached_token_ids(self) -> Dict[str, Any]:
        """Get pre-computed token IDs for loss mask creation."""
        return self._cached_token_ids

    def get_token_ids(self) -> Dict[str, Optional[int]]:
        """Get current token IDs from the processor.

        Returns a dict with keys:
        - image_token_id: Token ID for <image> placeholder
        - video_token_id: Token ID for <video> placeholder
        - vocab_size: Vocabulary size of the tokenizer

        These values can be used to update model config (e.g., LlavaOnevisionConfig)
        when the tokenizer has been replaced.

        Example:
            >>> bp = BatchImageProcessor(processor, tokenizer=qwen3_tokenizer)
            >>> token_ids = bp.get_token_ids()
            >>> # Update model config
            >>> model_config = dataclasses.replace(
            ...     model_config,
            ...     image_token_index=token_ids["image_token_id"],
            ...     video_token_index=token_ids["video_token_id"],
            ... )
        """
        return {
            "image_token_id": getattr(self.processor, "image_token_id", None),
            "video_token_id": getattr(self.processor, "video_token_id", None),
            "vocab_size": self.processor.tokenizer.vocab_size,
        }

    def _compute_grid_shape(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Compute (gh, gw) grid shape for an image using HF's select_best_resolution.

        This is used for pre-computing grid shapes during CPU preprocessing so they can be
        passed as concrete Python ints to pack_image_features, enabling JIT compilation.

        IMPORTANT: This method MUST use the same algorithm as CustomVLMProcessor._get_number_of_features
        to ensure grid_mask and token count are consistent. Both use HF's select_best_resolution.

        Args:
            image_size: (height, width) of the original image

        Returns:
            (gh, gw) grid dimensions as Python ints
        """
        if self.grid_pinpoints is None or self.patch_size is None:
            return (1, 1)  # Default for no anyres

        from transformers.models.llava_onevision.image_processing_llava_onevision import (
            select_best_resolution,
        )

        orig_h, orig_w = image_size
        best_h, best_w = select_best_resolution([orig_h, orig_w], self.grid_pinpoints)
        gh = best_h // self.patch_size
        gw = best_w // self.patch_size
        return (gh, gw)

    def _compute_unpad_indices_for_image(
        self,
        orig_height: int,
        orig_width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int,
        features_per_patch: int,
    ) -> np.ndarray:
        """Compute indices to reorder Levanter's padded features to HF's unpadded order.

        HF's pack_image_features applies spatial unpadding based on original image aspect ratio.
        This function computes the mapping from HF's feature positions to Levanter's sequential
        feature layout.

        Args:
            orig_height: Original image height
            orig_width: Original image width
            patches_height: Number of patches per tile in height (e.g., 27)
            patches_width: Number of patches per tile in width (e.g., 27)
            scale_height: Number of tiles in height (e.g., 3 for 3x3 grid)
            scale_width: Number of tiles in width (e.g., 3 for 3x3 grid)
            features_per_patch: Features per patch/tile (e.g., 729)

        Returns:
            unpad_indices: Array of shape (num_unpadded_features,) where
                          unpad_indices[i] = Levanter index for HF position i
        """
        # Base features are identity mapping (base patch is always first)
        base_indices = np.arange(features_per_patch, dtype=np.int32)

        # Grid spatial dimensions after combining all tiles
        curr_height = patches_height * scale_height
        curr_width = patches_width * scale_width

        # Compute unpadding bounds based on original aspect ratio
        # This matches HF's unpad_image logic
        original_aspect_ratio = orig_width / orig_height
        current_aspect_ratio = curr_width / curr_height

        if original_aspect_ratio > current_aspect_ratio:
            # Wider image - remove top/bottom padding
            scale_factor = curr_width / orig_width
            new_height = int(round(orig_height * scale_factor, 7))
            padding = (curr_height - new_height) // 2
            row_start = padding
            row_end = curr_height - padding
            col_start = 0
            col_end = curr_width
        else:
            # Taller image - remove left/right padding
            scale_factor = curr_height / orig_height
            new_width = int(round(orig_width * scale_factor, 7))
            padding = (curr_width - new_width) // 2
            row_start = 0
            row_end = curr_height
            col_start = padding
            col_end = curr_width - padding

        # Build mapping from HF grid position to Levanter grid index (vectorized)
        # HF order: row-major through unpadded region
        # Levanter order: patch-by-patch (tile-by-tile), then row-major within each patch

        # Create grid of all (row, col) positions in the unpadded region
        rows = np.arange(row_start, row_end, dtype=np.int32)
        cols = np.arange(col_start, col_end, dtype=np.int32)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        row_flat = row_grid.ravel()
        col_flat = col_grid.ravel()

        # Compute tile indices and local positions (vectorized)
        tile_rows = row_flat // patches_height
        tile_cols = col_flat // patches_width
        local_rows = row_flat % patches_height
        local_cols = col_flat % patches_width

        # Compute Levanter indices (vectorized)
        tile_indices = tile_rows * scale_width + tile_cols
        local_indices = local_rows * patches_width + local_cols
        grid_indices = features_per_patch + tile_indices * features_per_patch + local_indices

        return np.concatenate([base_indices, grid_indices])

    def _pad_pixel_values(
        self, pixel_values: np.ndarray, valid_patches: int, batch_max_patches: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad pixel_values to target size and create grid_mask.

        Args:
            pixel_values: Image patches array of shape (actual_patches, C, H, W)
            valid_patches: Number of patches to mark as valid in grid_mask
            batch_max_patches: Target patches for this batch (e.g., max images in batch).
                              For multi-image disable_anyres: this is the number of images.
                              For single-image anyres: dynamically pads to batch's actual max,
                              but not exceeding max_num_patches+1.

        Returns:
            Tuple of (padded_pixel_values, grid_mask)
        """
        assert self.max_num_patches is not None
        upper_limit = self.max_num_patches + 1  # +1 for base patch, this is the max allowed for anyres
        actual_patches = pixel_values.shape[0]

        # Determine total_patches based on mode:
        # - Multi-image mode: batch_max_patches > 1 AND actual_patches == batch_max_patches
        #   (each image contributes exactly 1 base patch)
        # - Single-image disable_anyres mode: only 1 patch (base patch)
        # - Single-image anyres mode: actual_patches > 1 (single image with multiple patches)
        #   Should use upper_limit to preserve all anyres patches
        if batch_max_patches is not None:
            is_multi_image = batch_max_patches > 1 and actual_patches == batch_max_patches
            if is_multi_image:
                # Multi-image case: pad to batch_max_patches (one patch per image)
                total_patches = batch_max_patches
            elif self.disable_anyres:
                # Single-image disable_anyres: only 1 patch needed (base patch)
                total_patches = batch_max_patches  # which is 1
            else:
                # Single-image anyres: preserve all patches up to upper_limit
                total_patches = upper_limit
        else:
            total_patches = upper_limit

        # Create grid_mask: True for valid patches, False for padding
        grid_mask = np.zeros(total_patches, dtype=np.bool_)
        grid_mask[:valid_patches] = True

        # Pad or truncate pixel_values to fixed size
        if actual_patches < total_patches:
            pad_size = total_patches - actual_patches
            padding = np.zeros((pad_size,) + pixel_values.shape[1:], dtype=pixel_values.dtype)
            pixel_values = np.concatenate([pixel_values, padding], axis=0)
        elif actual_patches > total_patches:
            pixel_values = pixel_values[:total_patches]
            grid_mask[:] = True

        return pixel_values, grid_mask

    def _create_loss_mask(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Create loss mask for training by identifying assistant response tokens.

        For causal LM training, we only compute loss on assistant responses.
        Returns a float32 mask where 1.0 indicates tokens that should contribute
        to the loss, and 0.0 indicates tokens that should be ignored.

        This is an efficient vectorized implementation that works directly on token IDs
        without decoding, similar to HuggingFace's return_assistant_tokens_mask.

        The algorithm identifies assistant response spans by looking for:
            <|im_start|>assistant{whitespace}...content...<|im_end|>

        Uses cumsum trick for O(n) complexity without Python loops.

        Args:
            input_ids: Token IDs array

        Returns:
            Loss mask array (float32) with 1.0 for valid positions, 0.0 for masked
        """
        if not self.mask_prompt:
            return np.ones(len(input_ids), dtype=np.float32)

        n = len(input_ids)
        # Get lazily cached token IDs (computed on first access)
        token_ids = self._get_cached_token_ids()
        num_ast = token_ids["num_assistant_tokens"]
        im_start_id = token_ids["im_start_id"]
        im_end_id = token_ids["im_end_id"]
        assistant_token_ids_array = token_ids["assistant_token_ids_array"]

        empty_mask = np.zeros(n, dtype=np.float32)

        if n < 3:
            return empty_mask

        # Find all <|im_start|> positions and filter to valid ones
        im_start_positions = np.where(input_ids == im_start_id)[0]
        valid_positions = im_start_positions[im_start_positions + 1 + num_ast <= n]
        if len(valid_positions) == 0:
            return empty_mask

        # Vectorized check for assistant tokens following <|im_start|>
        offsets = np.arange(1, num_ast + 1)
        check_indices = valid_positions[:, None] + offsets
        check_tokens = input_ids[check_indices]
        matches = np.all(check_tokens == assistant_token_ids_array, axis=1)
        pattern_starts = valid_positions[matches]
        if len(pattern_starts) == 0:
            return empty_mask

        # Find all <|im_end|> positions
        im_end_positions = np.where(input_ids == im_end_id)[0]
        if len(im_end_positions) == 0:
            return empty_mask

        # Content starts after: <|im_start|> + assistant_tokens
        # Note: The \n after "assistant" is INCLUDED in loss (matches HF behavior)
        content_starts = pattern_starts + 1 + num_ast
        valid_mask = content_starts < n
        content_starts = content_starts[valid_mask]
        if len(content_starts) == 0:
            return empty_mask

        # Use searchsorted to find matching <|im_end|> for each content_start
        end_indices = np.searchsorted(im_end_positions, content_starts, side="left")
        valid_ends = end_indices < len(im_end_positions)
        content_starts = content_starts[valid_ends]
        end_indices = end_indices[valid_ends]
        if len(content_starts) == 0:
            return empty_mask

        # End positions include <|im_end|> token
        end_positions = im_end_positions[end_indices] + 1

        # Use diff + cumsum to create interval mask efficiently
        diff = np.zeros(n + 1, dtype=np.int8)
        np.add.at(diff, content_starts, 1)
        np.add.at(diff, end_positions, -1)
        mask = np.cumsum(diff[:-1]) > 0

        return mask.astype(np.float32)

    def _compute_combined_mask(
        self,
        input_ids: np.ndarray,
        grid_mask: Optional[np.ndarray],
        unpad_indices: Optional[np.ndarray],
        features_per_patch: int,
        total_patches: int,
        num_unpadded_features: Optional[int] = None,
    ) -> np.ndarray:
        """Compute combined validity mask on CPU.

        This precomputes the mask that would otherwise be computed in the model's
        _merge_embeddings() method on GPU/TPU.

        Args:
            input_ids: Token IDs (seq_len,)
            grid_mask: Valid patch mask (TOTAL_PATCHES,) or None
            unpad_indices: Unpad indices (num_image_tokens,) or None
            features_per_patch: Number of features per patch (e.g., 729 for 27x27)
            total_patches: Total number of patches (TOTAL_PATCHES)
            num_unpadded_features: Actual number of unpadded features (before padding).
                                   Required when unpad_indices is provided.

        Returns:
            combined_mask: Validity mask (seq_len,) int32
        """
        # Get image_token_id and pad_token_id from processor
        image_token_id = getattr(self.processor, "image_token_id", None)
        pad_token_id = self.processor.tokenizer.pad_token_id

        # Create special_image_mask: positions where image tokens should be
        special_image_mask = (input_ids == image_token_id) if image_token_id is not None else np.zeros_like(input_ids, dtype=bool)

        # Create text_mask: valid text tokens (not padding)
        text_mask = (input_ids != pad_token_id).astype(np.int32)

        if unpad_indices is not None and num_unpadded_features is not None:
            # Only the first num_unpadded_features image placeholders are valid
            # The remaining are padding and should not get incrementing position IDs
            image_token_indices = np.cumsum(special_image_mask.astype(np.int32)) - 1
            image_validity = (image_token_indices < num_unpadded_features).astype(np.int32)
            combined_mask = np.where(special_image_mask, image_validity, text_mask).astype(np.int32)
        elif grid_mask is not None:
            # Need to check grid_mask validity for each placeholder position
            # Use ACTUAL valid patches from grid_mask (more robust than total_patches parameter)
            num_valid_patches = np.sum(grid_mask.astype(np.int32))
            num_valid_tokens = num_valid_patches * features_per_patch
            grid_mask_expanded = np.repeat(grid_mask.astype(np.int32), features_per_patch)

            # Compute image token indices for each position
            image_token_indices = np.cumsum(special_image_mask.astype(np.int32)) - 1
            image_token_indices = np.clip(image_token_indices, 0, max(num_valid_tokens - 1, 0))

            # Gather image validity from expanded grid_mask
            image_validity = grid_mask_expanded[image_token_indices]
            combined_mask = np.where(special_image_mask, image_validity, text_mask).astype(np.int32)
        else:
            # No images - just use text_mask
            combined_mask = text_mask

        return combined_mask

    def _compute_position_ids(self, combined_mask: np.ndarray) -> np.ndarray:
        """Compute position IDs from combined_mask using cumsum.

        This precomputes the position IDs that would otherwise be computed in the model's
        _merge_embeddings() method on GPU/TPU.

        Args:
            combined_mask: Validity mask (seq_len,) int32

        Returns:
            position_ids: Position IDs (seq_len,) int32
        """
        position_ids = np.cumsum(combined_mask.astype(np.int32)) - 1
        return np.maximum(position_ids, 0).astype(np.int32)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Sequence[ImageTextDict]:
        """
        Process a batch of conversation data.

        Args:
            batch: Sequence of conversation dictionaries with 'messages' and 'images' keys.

        Returns:
            Sequence of processed ImageTextDict
        """
        batch_size = len(batch)
        all_images: list = []
        all_texts: list[str] = []
        images_per_example: list[int] = []

        # Collect all images and texts - avoid repeated dict.get calls
        for item_idx, item in enumerate(batch):
            messages = item.get(self.messages_key, [])
            images_data = item.get(self.images_key, [])

            # Handle None images_data (parquet null values)
            if images_data is None:
                images_data = []

            # Sanitize: remove <image> from text content when {"type": "image"} dicts exist
            # This prevents double-counting when both formats coexist in the same message
            if messages:
                for msg in messages:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        has_image_dicts = any(
                            isinstance(c, dict) and c.get("type") == "image" for c in content
                        )
                        if has_image_dicts:
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                                    if "<image>" in c["text"]:
                                        c["text"] = c["text"].replace("<image>", "")

            # Validate messages/images consistency before processing
            # Count image references in messages
            num_image_refs = 0
            if messages:
                for msg in messages:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "image":
                                num_image_refs += 1
                    elif isinstance(content, str):
                        num_image_refs += content.count("<image>")

            if num_image_refs > 0 and len(images_data) == 0:
                raise ValueError(
                    f"Data inconsistency: messages reference {num_image_refs} image(s) but images list is empty! "
                    f"batch_item_idx={item_idx}, "
                    f"messages={json.dumps(messages, ensure_ascii=False, default=str)[:500]}, "
                    f"images_key='{self.images_key}', "
                    f"images_data={images_data!r}"
                )

            # Load all images for this example
            all_images.extend(load_image(img) for img in images_data)
            images_per_example.append(len(images_data))

            # Apply chat template to get the text with image placeholders
            template_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=self.add_generation_prompt,
            )
            all_texts.append(template_text)

            # Check apply_chat_template output for image placeholders that our pre-check missed
            # This catches message formats like {"type": "image_url"} that produce <image> tokens
            template_image_count = template_text.count("<image>")
            if template_image_count > 0 and len(images_data) == 0:
                raise ValueError(
                    f"Data inconsistency: apply_chat_template produced {template_image_count} "
                    f"<image> placeholder(s) but images list is empty! "
                    f"This suggests messages have image references in a format our pre-check missed. "
                    f"batch_item_idx={item_idx}, "
                    f"template_text={template_text[:500]!r}, "
                    f"messages={json.dumps(messages, ensure_ascii=False, default=str)[:500]}"
                )

        # Process all images and texts together in one call
        if all_images:
            processed: BatchFeature = self.processor(
                images=all_images,
                text=all_texts,
                return_tensors="np",
                padding=self._padding_mode,
                max_length=self.max_length,
                truncation=True,
            )
        else:
            # Text-only processing
            processed: BatchFeature = self.processor(
                text=all_texts,
                return_tensors="np",
                padding=self._padding_mode,
                max_length=self.max_length,
                truncation=True,
            )

        # Extract and convert batch arrays once (avoid per-example astype calls)
        input_ids_batch = processed["input_ids"].astype(np.int32)
        attention_mask_batch = processed["attention_mask"].astype(np.int32)

        # Pre-extract pixel_values and image_sizes if available
        has_pixel_values = "pixel_values" in processed
        has_image_sizes = "image_sizes" in processed
        pv = processed["pixel_values"] if has_pixel_values else None
        # image_sizes might be a list (from some processors), convert to numpy array
        img_sizes = np.array(processed["image_sizes"]).astype(np.int32) if has_image_sizes else None

        # Pre-compute cumulative image indices for fast slicing
        # cumsum gives end indices: [n0, n0+n1, n0+n1+n2, ...]
        cum_images = np.cumsum(images_per_example)

        # Compute max images in batch for consistent padding in disable_anyres mode
        # This ensures all examples have the same total_patches for proper batching
        max_num_images_in_batch = int(max(images_per_example)) if len(images_per_example) > 0 else 1

        # Build output list
        out: list[ImageTextDict] = []
        for i in range(batch_size):
            input_ids = input_ids_batch[i]
            num_images = images_per_example[i]

            # Calculate image index range for this example
            pv_end = cum_images[i]
            pv_start = pv_end - num_images

            # Get pixel_values for this example and create grid_mask
            grid_mask = None
            unpad_indices = None
            num_unpadded_features = None  # Actual number of unpadded features (before padding)
            if num_images > 0 and has_pixel_values:
                assert pv is not None  # Guarded by has_pixel_values
                if num_images == 1 and not self.disable_anyres:
                    # Single image with anyres: use all patches from processor output
                    pixel_values = pv[pv_start]
                    if self.max_num_patches is not None:
                        pixel_values, grid_mask = self._pad_pixel_values(
                            pixel_values,
                            valid_patches=pixel_values.shape[0],
                            batch_max_patches=max_num_images_in_batch,
                        )
                elif num_images == 1 and self.disable_anyres:
                    # Single image with disable_anyres: only use base patch (first patch)
                    # HF processor always outputs base + anyres patches, but we only want base
                    pixel_values = pv[pv_start][0:1]  # Take only first patch
                    if self.max_num_patches is not None:
                        pixel_values, grid_mask = self._pad_pixel_values(
                            pixel_values,
                            valid_patches=1,  # Only 1 valid patch (base)
                            batch_max_patches=max_num_images_in_batch,
                        )
                else:
                    # Multiple images: only use base patch (first patch) from each image
                    # This matches HF behavior where multi-image doesn't use anyres
                    # pv[pv_start + img_idx] has shape (patches, C, H, W), take [0] for base patch
                    base_patches = [pv[pv_start + img_idx][0] for img_idx in range(num_images)]
                    pixel_values = np.stack(base_patches, axis=0)  # (num_images, C, H, W)
                    if self.max_num_patches is not None:
                        pixel_values, grid_mask = self._pad_pixel_values(
                            pixel_values,
                            valid_patches=num_images,
                            batch_max_patches=max_num_images_in_batch,
                        )
            else:
                pixel_values = None

            # Get image sizes for this example
            if num_images > 0 and has_image_sizes:
                assert img_sizes is not None  # Guarded by has_image_sizes
                image_sizes = img_sizes[pv_start:pv_end]
                if image_sizes.ndim == 1:
                    image_sizes = image_sizes.reshape(1, 2)
            else:
                image_sizes = None

            # Compute unpad_indices for image feature reordering
            # - For anyres mode (single image): compute proper unpad indices based on grid shape
            # - For disable_anyres mode or multi-image: keep None, model uses grid_mask for validity
            use_anyres = self.max_num_patches is None or self.max_num_patches > 0
            if num_images > 0 and self.vision_feature_height:
                patches_height = patches_width = self.vision_feature_height
                features_per_patch = patches_height * patches_width

                if num_images == 1 and has_image_sizes and self.grid_pinpoints and use_anyres:
                    # Single image with anyres: compute proper unpad indices based on grid shape
                    assert image_sizes is not None  # Guarded by has_image_sizes
                    orig_height, orig_width = int(image_sizes[0, 0]), int(image_sizes[0, 1])
                    gh, gw = self._compute_grid_shape((orig_height, orig_width))
                    unpad_indices_raw = self._compute_unpad_indices_for_image(
                        orig_height=orig_height,
                        orig_width=orig_width,
                        patches_height=patches_height,
                        patches_width=patches_width,
                        scale_height=gh,
                        scale_width=gw,
                        features_per_patch=features_per_patch,
                    )
                    # Store actual number of unpadded features before padding
                    num_unpadded_features = np.int32(len(unpad_indices_raw))
                    # Pad unpad_indices to fixed size for consistent array shapes
                    if self.max_num_patches is not None:
                        max_features = (self.max_num_patches + 1) * features_per_patch
                        unpad_indices = np.zeros(max_features, dtype=np.int32)
                        unpad_indices[: len(unpad_indices_raw)] = unpad_indices_raw
                    else:
                        unpad_indices = unpad_indices_raw
                # else: multi-image or disable_anyres - keep unpad_indices = None
                # Model uses grid_mask to determine which patches are valid

            # Compute combined_mask and position_ids on CPU
            features_per_patch = (
                self.vision_feature_height * self.vision_feature_height
                if self.vision_feature_height else 729  # default 27x27
            )
            # For multi-image: use num_images as total_patches (one base patch per image)
            # For single-image anyres: use (max_num_patches + 1) for grid tiles + base
            if num_images > 1:
                total_patches = num_images
            elif self.max_num_patches is not None:
                total_patches = self.max_num_patches + 1
            else:
                total_patches = 1

            combined_mask = self._compute_combined_mask(
                input_ids=input_ids,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                features_per_patch=features_per_patch,
                total_patches=total_patches,
                num_unpadded_features=num_unpadded_features,
            )
            position_ids = self._compute_position_ids(combined_mask)

            # Post-processing consistency check: verify input_ids image tokens match pixel_values/grid_mask
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is not None:
                image_token_count = int(np.sum(input_ids == image_token_id))
                has_valid_images = (
                    pixel_values is not None
                    and grid_mask is not None
                    and np.any(grid_mask)
                )
                if image_token_count > 0 and not has_valid_images:
                    raise ValueError(
                        f"Post-processing consistency error: input_ids has {image_token_count} "
                        f"image tokens (token_id={image_token_id}) but pixel_values is "
                        f"{'None' if pixel_values is None else 'present'}, "
                        f"grid_mask={grid_mask}, num_images={num_images}. "
                        f"This means image tokens were injected into input_ids without "
                        f"corresponding image data. "
                        f"chat_template_text={all_texts[i][:500]!r}, "
                        f"images_per_example[i]={images_per_example[i]}, "
                        f"batch_item_idx={i}"
                    )

            # Create labels and build result
            result: ImageTextDict = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask_batch[i],
                "image_sizes": image_sizes,
                "loss_mask": self._create_loss_mask(input_ids),
                "grid_mask": grid_mask,
                "unpad_indices": unpad_indices,
                "num_unpadded_features": num_unpadded_features,
                "combined_mask": combined_mask,
                "position_ids": position_ids,
            }
            out.append(result)

        return out

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "processor": type(self.processor).__name__,
            "max_length": self.max_length,
            "padding": self.padding,
            "mask_prompt": self.mask_prompt,
        }

    @property
    def output_exemplar(self):
        exemplar = dict(ImageTextDict_exemplar)
        # Override with sized arrays when max_num_patches is configured
        if self.max_num_patches is not None:
            total_patches = self.max_num_patches + 1
            # Fixed-size pixel_values for cache schema
            exemplar["pixel_values"] = np.zeros((total_patches, 3, self.patch_size, self.patch_size), dtype=np.float32)
            exemplar["grid_mask"] = np.zeros((total_patches,), dtype=np.bool_)
            # Include sized unpad_indices and num_unpadded_features when vision_feature_height is also configured
            if self.vision_feature_height is not None:
                features_per_patch = self.vision_feature_height * self.vision_feature_height
                max_features = (self.max_num_patches + 1) * features_per_patch
                exemplar["unpad_indices"] = np.zeros((max_features,), dtype=np.int32)
                exemplar["num_unpadded_features"] = np.int32(0)  # Scalar for actual unpadded feature count
        # Add combined_mask and position_ids with max_length shape
        exemplar["combined_mask"] = np.zeros((self.max_length,), dtype=np.int32)
        exemplar["position_ids"] = np.zeros((self.max_length,), dtype=np.int32)
        return exemplar

    @property
    def num_cpus(self) -> int:
        return 2  # Image processing can benefit from multiple CPUs

    @property
    def num_gpus(self) -> int:
        return 0


@dataclass
class ImageDatasetSourceConfig:
    """Configuration for a simple image-text dataset source (single image + text pairs)."""

    id: Optional[str] = None  # HuggingFace dataset id or path
    name: Optional[str] = None  # Dataset configuration name

    stream: bool = True  # Whether to use streaming
    image_key: str = "image"  # Key for image field
    text_key: str = "text"  # Key for text field

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type: ignore
    cache_dir: str = "cache/"

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[Dict[str, Any]]]:
        """Get a sharded data source for the specified split."""
        if self.id is not None:
            try:
                ds = WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.stream)
            except ValueError as e:
                if str(e).startswith("Bad split"):
                    logger.warning(f"Split {split} not found for {self.id} {self.name}")
                    return None
                raise

            if len(ds.shard_names) == 0:
                return None

            def extract_fields(x):
                return {
                    "image": x[self.image_key],
                    "text": x[self.text_key],
                }

            return ds.map(extract_fields)
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return ImageTextUrlDataSource(split_urls, image_key=self.image_key, text_key=self.text_key)

    def doc_iterator(self, split: str) -> Iterator[Dict[str, Any]]:
        """Iterate over documents in the specified split."""
        if self.id is not None:
            data = datasets.load_dataset(self.id, split=split, name=self.name, streaming=self.stream)
            for doc in data:
                yield {
                    "image": doc[self.image_key],
                    "text": doc[self.text_key],
                }
        else:
            urls = self.urls_for_split(split)
            yield from ImageTextUrlDataSource(urls, image_key=self.image_key, text_key=self.text_key)

    def urls_for_split(self, split: str) -> List[str]:
        """Get URLs for the specified split.

        Supports:
        - Single file paths: /path/to/file.parquet
        - Glob patterns: /path/to/*.parquet
        - Directories: /path/to/folder/ (will find all *.parquet files)
        - file:// prefixed paths: file:///path/to/folder/
        - Brace expansion: /path/to/{train,val}*.parquet
        """
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split: {split}")

        return expand_urls_with_folder_support(list(urls))


@dataclass
class ConversationDatasetSourceConfig:
    """Configuration for a conversation-format image-text dataset source.

    This is used for VLM training data with conversation format like LLaVA.
    Supports single image, multiple images, and interleaved image/text content.

    1. Single image:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": ["path/to/image.jpg"]
    }

    2. Multiple images:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare these two images."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": ["path/to/image1.jpg", "path/to/image2.jpg"]
    }

    3. Interleaved image and text:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "First image:"},
                {"type": "image"},
                {"type": "text", "text": "Second image:"},
                {"type": "image"},
                {"type": "text", "text": "What are the differences?"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": ["path/to/image1.jpg", "path/to/image2.jpg"]
    }

    Note: {"type": "image"} placeholders are replaced with images from the "images" list in order.
    """

    id: Optional[str] = None  # HuggingFace dataset id or path
    name: Optional[str] = None  # Dataset configuration name

    stream: bool = True  # Whether to use streaming
    messages_key: str = "messages"  # Key for messages field
    images_key: str = "images"  # Key for images field

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type: ignore
    cache_dir: str = "cache/"

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[ConversationDict]]:
        """Get a sharded data source for the specified split."""
        if self.id is not None:
            try:
                ds = WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.stream)
            except ValueError as e:
                if str(e).startswith("Bad split"):
                    logger.warning(f"Split {split} not found for {self.id} {self.name}")
                    return None
                raise

            if len(ds.shard_names) == 0:
                return None

            def extract_fields(x):
                return {
                    "messages": x[self.messages_key],
                    "images": x.get(self.images_key, []),
                }

            return ds.map(extract_fields)
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return cast(
                ShardedDataSource[ConversationDict],
                ImageConversationUrlDataSource(split_urls, messages_key=self.messages_key, images_key=self.images_key),
            )

    def doc_iterator(self, split: str) -> Iterator[ConversationDict]:
        """Iterate over documents in the specified split."""
        if self.id is not None:
            data = datasets.load_dataset(self.id, split=split, name=self.name, streaming=self.stream)
            for doc in data:
                yield {
                    "messages": doc[self.messages_key],
                    "images": doc.get(self.images_key, []),
                }
        else:
            urls = self.urls_for_split(split)
            for doc in ImageConversationUrlDataSource(
                urls, messages_key=self.messages_key, images_key=self.images_key
            ):
                yield cast(ConversationDict, doc)

    def urls_for_split(self, split: str) -> List[str]:
        """Get URLs for the specified split.

        Supports:
        - Single file paths: /path/to/file.parquet
        - Glob patterns: /path/to/*.parquet
        - Directories: /path/to/folder/ (will find all *.parquet files)
        - file:// prefixed paths: file:///path/to/folder/
        - Brace expansion: /path/to/{train,val}*.parquet
        """
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split: {split}")

        return expand_urls_with_folder_support(list(urls))


@dataclass
class ImageTaskConfig(abc.ABC):
    """Base configuration for image-text tasks."""

    processor: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    tokenizer: Optional[str] = None
    """Optional custom tokenizer name (e.g., "Qwen/Qwen3-1.7B").
    If specified, creates a CustomVLMProcessor that combines the processor's image processing
    with the custom tokenizer. This is useful when training with a different tokenizer than
    the one bundled with the processor."""
    max_length: int = 2048
    padding: bool = True
    image_grid_pinpoints: Optional[List[List[int]]] = None
    """Override image grid pinpoints for anyres processing.
    Set to [[image_size, image_size]] to disable anyres (single resolution only)."""

    vision_aspect_ratio: Optional[str] = None
    """Override vision aspect ratio for anyres processing.
    Set to "single" to disable anyres (single resolution only).
    Common values: "single" (base patch only), "anyres_max_9" (up to 9 grid patches)."""

    use_full_padded_tokens: bool = True
    """Use Levanter-style full padded token count for training.

    When True (default, Levanter mode): token_count = num_tiles × features_per_patch
    When False (HF mode): token_count = base_tokens + unpadded_features

    Levanter mode is required for training because the model uses grid_mask for JIT-compatible
    fixed-shape tensors. HF mode is for inference with HF's spatial unpadding."""

    vision_feature_height: Optional[int] = None
    """Vision encoder output tokens per spatial dimension.
    E.g., 24 for SigLIP2 (384/16=24), 27 for LLaVA original (378/14=27).
    If not specified, uses the HF processor's default (usually 27×27=729)."""

    @cached_property
    def the_processor(self) -> ProcessorMixin:
        proc = load_processor(self.processor)

        # Override image_grid_pinpoints if specified (e.g., for disable_anyres)
        if self.image_grid_pinpoints is not None and hasattr(proc, "image_processor"):
            proc.image_processor.image_grid_pinpoints = self.image_grid_pinpoints

        # Override vision_aspect_ratio if specified (e.g., for disable_anyres)
        if self.vision_aspect_ratio is not None and hasattr(proc, "image_processor"):
            proc.image_processor.vision_aspect_ratio = self.vision_aspect_ratio

        # Replace SiglipImageProcessor with LlavaOnevisionImageProcessor for anyres support
        # SiglipImageProcessor doesn't support multi-patch output needed for anyres
        if hasattr(proc, "image_processor"):
            image_processor_class = type(proc.image_processor).__name__
            if image_processor_class == "SiglipImageProcessor":
                try:
                    from transformers.models.llava_onevision.image_processing_llava_onevision import (
                        LlavaOnevisionImageProcessor,
                    )
                    from levanter.compat.hf_checkpoints import hf_load_with_retry

                    logger.info(
                        f"Replacing {image_processor_class} with LlavaOnevisionImageProcessor "
                        f"for anyres support (loading from {self.processor})"
                    )
                    new_image_processor = hf_load_with_retry(
                        LlavaOnevisionImageProcessor.from_pretrained,
                        self.processor, trust_remote_code=True
                    )
                    # Copy over any overridden attributes
                    if self.vision_aspect_ratio is not None:
                        new_image_processor.vision_aspect_ratio = self.vision_aspect_ratio
                    if self.image_grid_pinpoints is not None:
                        new_image_processor.image_grid_pinpoints = self.image_grid_pinpoints
                    proc.image_processor = new_image_processor
                    logger.info(
                        f"LlavaOnevisionImageProcessor loaded: "
                        f"vision_aspect_ratio={getattr(new_image_processor, 'vision_aspect_ratio', 'N/A')}, "
                        f"image_grid_pinpoints count={len(getattr(new_image_processor, 'image_grid_pinpoints', []))}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load LlavaOnevisionImageProcessor: {e}")

        # Override num_image_tokens based on vision_feature_height
        # This is critical for correct token expansion when vision encoder uses different patch size
        if self.vision_feature_height is not None:
            num_image_tokens = self.vision_feature_height * self.vision_feature_height
            proc.num_image_tokens = num_image_tokens
            logger.info(f"Set processor.num_image_tokens={num_image_tokens} "
                       f"(vision_feature_height={self.vision_feature_height})")

        # If custom tokenizer is specified, wrap with CustomVLMProcessor
        if self.tokenizer is not None:
            from levanter.compat.hf_checkpoints import load_tokenizer

            custom_tokenizer = load_tokenizer(self.tokenizer, trust_remote_code=True)
            proc = CustomVLMProcessor.from_processor_and_tokenizer(
                proc, custom_tokenizer,
                use_full_padded_tokens=self.use_full_padded_tokens,
            )

        return proc

    @cached_property
    def pad_token_id(self) -> int:
        return self.the_processor.tokenizer.pad_token_id

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.the_processor.tokenizer

    @abc.abstractmethod
    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> AsyncDataset[ImageTextDict]:
        pass

    @abc.abstractmethod
    def validation_sets(self) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        pass


class StreamingImageDataset(AsyncDataset[ImageTextDict]):
    """
    Streaming dataset that processes images on-the-fly without caching to disk.

    This avoids the disk space overhead of caching preprocessed pixel_values,
    at the cost of reprocessing images each epoch.

    Key design (Lazy Shard Loading):
    - Only loads shard metadata at startup (fast - just row counts from parquet metadata)
    - Loads shards on-demand as data is accessed
    - Pre-fetches next shard when remaining data in current shard < PREFETCH_THRESHOLD
    - Evicts old shards to limit memory usage (keeps only recent 2 shards)
    - Background thread prefetches processed data sequentially ahead of consumption
    - Uses per-processor locks for HF tokenizer thread-safety

    Flow:
        Shard loading: [load shard 0] -> [prefetch shard 1 when 1K left] -> [evict shard 0] -> ...
        Prefetch thread: [process 0-31] -> [process 32-63] -> [process 64-95] -> ...
        Main thread:     [access 0-31, pop from cache] -> [access 32-63, pop] -> ...
    """

    # How many processed examples to cache in memory
    DEFAULT_CACHE_SIZE = 2048  # ~256 examples * ~2MB each = ~512MB

    # When remaining rows in current shard < this threshold, prefetch next shard
    PREFETCH_THRESHOLD = 2000

    # Per-processor locks - each processor instance gets its own lock
    # This allows different processors to run in parallel while ensuring
    # thread-safety for each individual processor (HF tokenizers are not thread-safe)
    _processor_locks: Optional[weakref.WeakKeyDictionary] = None  # Lazy init
    _processor_locks_lock: Optional[threading.Lock] = None  # Lazy init

    @classmethod
    def _init_class_locks(cls):
        """Initialize class-level locks lazily."""
        if cls._processor_locks is None:
            cls._processor_locks = weakref.WeakKeyDictionary()
            cls._processor_locks_lock = threading.Lock()

    @classmethod
    def _get_processor_lock(cls, processor) -> threading.Lock:
        """Get or create a lock for a specific processor instance."""
        cls._init_class_locks()
        assert cls._processor_locks_lock is not None
        assert cls._processor_locks is not None
        with cls._processor_locks_lock:
            if processor not in cls._processor_locks:
                cls._processor_locks[processor] = threading.Lock()
            return cls._processor_locks[processor]

    def __init__(
        self,
        source: ShardedDataSource[Dict[str, Any]],
        processor: ProcessorMixin,
        max_length: int = 2048,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        cache_size: int = DEFAULT_CACHE_SIZE,
        max_num_patches: Optional[int] = None,
        vision_feature_height: Optional[int] = None,
    ):
        super().__init__()
        self.source = source
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
        self.messages_key = messages_key
        self.images_key = images_key
        self.cache_size = cache_size

        # Extract grid_pinpoints and related params from processor for anyres support
        grid_pinpoints, patch_size, extracted_vision_feature_height, extracted_max_num_patches = _extract_anyres_params(
            processor
        )
        # Use passed max_num_patches if provided, otherwise use extracted value
        if max_num_patches is not None:
            final_max_num_patches = max_num_patches
        else:
            final_max_num_patches = extracted_max_num_patches

        # Use passed vision_feature_height if provided, otherwise use extracted value
        # This allows overriding when the model uses a different patch size than the HF processor
        if vision_feature_height is not None:
            final_vision_feature_height = vision_feature_height
        else:
            final_vision_feature_height = extracted_vision_feature_height

        # Build the batch processor (runs on CPU in background thread)
        self._batch_processor = BatchImageProcessor(
            processor,
            max_length=max_length,
            padding=padding,
            messages_key=messages_key,
            images_key=images_key,
            grid_pinpoints=grid_pinpoints,
            patch_size=patch_size,
            vision_feature_height=final_vision_feature_height,
            max_num_patches=final_max_num_patches,
        )

        # Use per-processor lock - HuggingFace tokenizer is NOT thread-safe
        # Each processor instance gets its own lock, allowing different processors
        # to run in parallel while ensuring thread-safety for each one
        self._processor_lock = self._get_processor_lock(processor)

        # Lazy shard loading state (step-based mode - no pre-counting)
        self._shard_names: List[str] = []
        self._loaded_shards: Dict[int, List[Dict[str, Any]]] = {}  # shard_idx -> rows
        self._shard_load_lock = threading.Lock()
        self._prefetch_shard_thread: Optional[threading.Thread] = None

        # Sequential iteration state (step-based mode)
        self._current_shard_idx: int = 0
        self._current_local_idx: int = 0
        self._current_shard_data: Optional[List[Dict[str, Any]]] = None
        self._iteration_lock = threading.Lock()

        # Initialization flag (replaces _length check)
        self._initialized: bool = False
        self._data_lock = threading.Lock()
        self._data_loaded = threading.Event()

        # Prefetch cache for PROCESSED data (large - includes pixel_values)
        # Key: monotonically increasing counter, Value: ImageTextDict
        # Items are popped when accessed - cache only holds prefetched but not-yet-accessed data
        self._processed_cache: OrderedDict[int, ImageTextDict] = OrderedDict()
        self._cache_key_counter: int = 0  # Monotonic counter to avoid key collision when items are popped
        self._cache_lock = threading.Lock()

        # Background sequential prefetch
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
        self._prefetch_error: Optional[BaseException] = None

        # Pipeline start is deferred until first get_batch() call to support seeking
        self._pipeline_started: bool = False
        self._resume_items_to_skip: int = 0

    def _ensure_data_loaded(self):
        """Initialize dataset in step-based mode (no pre-counting of rows).

        In distributed training, each worker only loads its assigned subset of shards
        to avoid GCS bandwidth contention. Shards are interleaved across workers:
        - Worker 0: shards 0, N, 2N, ...
        - Worker 1: shards 1, N+1, 2N+1, ...
        - ...
        where N = num_workers
        """
        if self._initialized:
            return

        with self._data_lock:
            if self._initialized:
                return

            logger.info("Initializing streaming dataset (step-based mode)...")
            t0 = time.time()

            # Get worker ID and total worker count for distributed training
            process_index = jax.process_index()  # 0-based worker ID
            num_processes = jax.process_count()  # Total number of workers

            # Get all shard names
            all_shard_names = list(self.source.shard_names)
            total_shards = len(all_shard_names)
            logger.info(f"Found {total_shards} total shards in {time.time()-t0:.2f}s")

            # Each worker only gets its assigned subset of shards (interleaved)
            # This prevents all workers from loading the same shards simultaneously
            if num_processes > 1:
                self._shard_names = [
                    name for i, name in enumerate(all_shard_names)
                    if i % num_processes == process_index
                ]
                logger.info(
                    f"Worker {process_index}/{num_processes}: assigned {len(self._shard_names)} shards "
                    f"(indices {process_index}, {process_index + num_processes}, ...)"
                )
            else:
                # Single process - use all shards
                self._shard_names = all_shard_names

            self._initialized = True
            self._data_loaded.set()

            # NOTE: First shard loading and prefetch thread start are deferred
            # to _start_pipeline(), which is called on the first get_batch() call.
            # This allows computing the correct seek position from the indices.

    def set_resume_item_count(self, count: int):
        """Set the number of items to skip on resume (per process).

        Must be called BEFORE the first get_batch() call (before pipeline starts).
        """
        self._resume_items_to_skip = count

    def _start_pipeline(self, items_to_skip: int = 0):
        """Start the data pipeline, optionally seeking to a position first.

        Called on the first get_batch() call. Deferred from _ensure_data_loaded()
        so that we can compute the seek position from the batch indices.
        """
        if self._pipeline_started:
            return
        self._pipeline_started = True

        if items_to_skip > 0 and self._shard_names:
            self._seek_to_item(items_to_skip)
        elif self._shard_names:
            self._load_current_shard()

        self._start_prefetch_thread()

    def _seek_to_item(self, items_to_skip: int):
        """Fast-forward the sequential stream by skipping items.

        Uses shard metadata (row counts) to skip entire shards efficiently,
        only loading the target shard that contains the seek position.
        """
        total_items_in_all_shards = 0
        shard_row_counts = []

        # First pass: collect row counts using metadata (fast for parquet)
        for shard_name in self._shard_names:
            try:
                num_rows = self.source.shard_num_rows(shard_name)
            except Exception:
                # Fallback: load shard to get count (slow)
                logger.warning(f"shard_num_rows not available for {shard_name}, loading full shard to count")
                shard_data = list(self.source.open_shard(shard_name))
                num_rows = len(shard_data)
            shard_row_counts.append(num_rows)
            total_items_in_all_shards += num_rows

        if total_items_in_all_shards == 0:
            logger.warning("No data available for seeking")
            return

        # Handle wrap-around: if we've consumed more than one full pass, take modulo
        items_to_skip = items_to_skip % total_items_in_all_shards

        # Second pass: find target shard and row
        skipped = 0
        for shard_idx, num_rows in enumerate(shard_row_counts):
            if items_to_skip < num_rows:
                self._current_shard_idx = shard_idx
                self._current_local_idx = items_to_skip
                self._load_current_shard()
                logger.info(
                    f"Seeked to shard {shard_idx}/{len(self._shard_names)}, "
                    f"row {items_to_skip}/{num_rows} "
                    f"(skipped {skipped + items_to_skip} total items)"
                )
                return
            items_to_skip -= num_rows
            skipped += num_rows

        # Should not reach here due to modulo above, but just in case
        self._current_shard_idx = 0
        self._current_local_idx = 0
        self._load_current_shard()

    def _load_shard(self, shard_idx: int):
        """Load a specific shard into memory."""
        if shard_idx >= len(self._shard_names):
            return
        if shard_idx in self._loaded_shards:
            return

        with self._shard_load_lock:
            if shard_idx in self._loaded_shards:
                return

            shard_name = self._shard_names[shard_idx]
            logger.info(f"Loading shard {shard_idx}: {shard_name}")

            t0 = time.time()
            shard_data = list(self.source.open_shard(shard_name))
            self._loaded_shards[shard_idx] = shard_data
            logger.info(f"Loaded shard {shard_idx} ({len(shard_data)} rows) in {time.time()-t0:.2f}s")

            # Evict old shards to limit memory usage
            self._evict_old_shards(keep_recent=2)

    def _evict_old_shards(self, keep_recent: int = 2):
        """Evict old shards, keeping only the most recent N shards."""
        if len(self._loaded_shards) <= keep_recent:
            return

        loaded_indices = sorted(self._loaded_shards.keys())
        for idx in loaded_indices[:-keep_recent]:
            del self._loaded_shards[idx]
            logger.debug(f"Evicted shard {idx}")

    def _load_current_shard(self):
        """Load the current shard into memory (step-based mode)."""
        if self._current_shard_idx >= len(self._shard_names):
            return

        shard_name = self._shard_names[self._current_shard_idx]
        logger.info(f"Loading shard {self._current_shard_idx}: {shard_name}")

        t0 = time.time()
        self._current_shard_data = list(self.source.open_shard(shard_name))
        logger.info(f"Loaded shard {self._current_shard_idx} ({len(self._current_shard_data)} rows) in {time.time()-t0:.2f}s")

        # Also store in _loaded_shards for compatibility
        self._loaded_shards[self._current_shard_idx] = self._current_shard_data

        # Evict old shards to limit memory usage
        self._evict_old_shards(keep_recent=2)

    def _get_next_item(self) -> Dict[str, Any]:
        """Get next item using sequential iteration (step-based mode)."""
        self._ensure_data_loaded()

        with self._iteration_lock:
            # Ensure current shard is loaded
            if self._current_shard_data is None:
                self._load_current_shard()

            # If current shard is exhausted, move to next
            while self._current_local_idx >= len(self._current_shard_data):
                self._current_shard_idx += 1
                if self._current_shard_idx >= len(self._shard_names):
                    # Wrap around to first shard (for infinite iteration)
                    self._current_shard_idx = 0
                self._current_local_idx = 0
                self._load_current_shard()

                # Prefetch next shard in background
                next_shard = self._current_shard_idx + 1
                if next_shard < len(self._shard_names):
                    self._prefetch_shard_async(next_shard)

            item = self._current_shard_data[self._current_local_idx]
            self._current_local_idx += 1
            return item

    def _prefetch_shard_async(self, shard_idx: int):
        """Asynchronously prefetch a shard in background thread."""
        if self._prefetch_shard_thread is not None and self._prefetch_shard_thread.is_alive():
            return  # Already prefetching

        def prefetch():
            self._load_shard(shard_idx)

        self._prefetch_shard_thread = threading.Thread(target=prefetch, daemon=True)
        self._prefetch_shard_thread.start()

    def _start_prefetch_thread(self):
        """Start background thread to prefetch data sequentially (step-based mode)."""
        if self._prefetch_thread is not None:
            return

        def prefetch_worker():
            """Background worker that prefetches data sequentially (step-based mode).

            Uses _get_next_item() for sequential iteration through shards.
            """
            batch_size = 32

            while not self._stop_prefetch.is_set():
                if not self._initialized:
                    self._stop_prefetch.wait(0.05)
                    continue

                # Check cache size
                cache_len = len(self._processed_cache)
                if cache_len >= self.cache_size:
                    # Cache is full, wait
                    self._stop_prefetch.wait(0.05)
                    continue

                # Process batch using sequential iteration
                try:
                    raw_items = []
                    for _ in range(min(batch_size, self.cache_size - cache_len)):
                        raw_items.append(self._get_next_item())

                    if not raw_items:
                        continue

                    with self._processor_lock:
                        processed = self._batch_processor(raw_items)

                    with self._cache_lock:
                        for item in processed:
                            # Use monotonic counter to avoid key collision when items are popped
                            self._processed_cache[self._cache_key_counter] = item
                            self._cache_key_counter += 1
                        # Evict oldest entries if over limit
                        while len(self._processed_cache) > self.cache_size:
                            self._processed_cache.popitem(last=False)
                except ValueError as e:
                    # Store exception for main thread to detect, then exit cleanly
                    logger.error(f"Prefetch hit data error: {e}")
                    self._prefetch_error = e
                    self._stop_prefetch.set()
                    return
                except Exception as e:
                    logger.warning(f"Prefetch failed (transient): {e}")
                    self._stop_prefetch.wait(0.1)

        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        logger.debug("Started sequential prefetch thread (step-based mode)")

    def _process_items(self, count: int) -> List[ImageTextDict]:
        """Process items using sequential iteration - must be called with _processor_lock held."""
        raw_items = [self._get_next_item() for _ in range(count)]
        processed = self._batch_processor(raw_items)
        return list(processed)

    def _get_from_cache_or_process(self, indices: Sequence[int]) -> List[ImageTextDict]:
        """Get items from cache (step-based mode).

        In step-based mode, we return batch_size items in sequential order from
        the cache. On the first call, uses pre-set _resume_items_to_skip for
        checkpoint resume support (set via set_resume_item_count()).
        """
        self._ensure_data_loaded()

        # Start the pipeline on first call, using pre-computed resume position
        if not self._pipeline_started:
            with self._data_lock:
                if not self._pipeline_started:
                    items_to_skip = self._resume_items_to_skip
                    if items_to_skip > 0:
                        logger.info(
                            f"Starting pipeline with resume: seeking to item {items_to_skip}"
                        )
                    self._start_pipeline(items_to_skip)

        batch_size = len(indices)
        results: List[ImageTextDict] = []

        # Wait for cache to have enough items (only prefetch thread produces data)
        while len(results) < batch_size:
            if self._prefetch_error is not None:
                raise RuntimeError(
                    f"Prefetch thread died with error: {self._prefetch_error}"
                ) from self._prefetch_error

            with self._cache_lock:
                sorted_keys = sorted(self._processed_cache.keys())
                needed = batch_size - len(results)
                for key in sorted_keys[:needed]:
                    results.append(self._processed_cache.pop(key))

            if len(results) < batch_size:
                # Wait a bit for prefetch thread to add more items
                time.sleep(0.01)

        return results

    async def async_len(self) -> int:
        """Return a large number for step-based mode (infinite dataset)."""
        self._ensure_data_loaded()
        return sys.maxsize

    async def final_length_is_known(self) -> bool:
        """Return False for step-based mode - length is not known."""
        return False

    def is_finite(self) -> bool:
        """Return False for step-based mode - dataset iterates infinitely."""
        return False

    async def current_len(self) -> Optional[int]:
        """Return None for step-based mode - current length is not tracked."""
        return None

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ImageTextDict]:
        """Get a batch of processed items."""
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_from_cache_or_process, indices)

    def close(self):
        """Stop background threads and clean up resources."""
        self._stop_prefetch.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)
        if self._prefetch_shard_thread is not None:
            self._prefetch_shard_thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        """Clean up background thread."""
        self.close()

    @staticmethod
    def build(
        source: ShardedDataSource[Dict[str, Any]],
        processor: ProcessorMixin,
        max_length: int = 2048,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        cache_size: int = DEFAULT_CACHE_SIZE,
        max_num_patches: Optional[int] = None,
        vision_feature_height: Optional[int] = None,
    ) -> "StreamingImageDataset":
        """Build a streaming dataset from a source.

        Args:
            vision_feature_height: Override the vision feature height extracted from processor.
                This is useful when using a custom vision encoder with different patch size.
                For example, if your vision encoder outputs 24x24=576 features per image,
                set this to 24 (not 27 which is the default for some HF processors).
        """
        return StreamingImageDataset(
            source=source,
            processor=processor,
            max_length=max_length,
            padding=padding,
            messages_key=messages_key,
            images_key=images_key,
            cache_size=cache_size,
            max_num_patches=max_num_patches,
            vision_feature_height=vision_feature_height,
        )


class ProcessedImageCache(AsyncDataset[ImageTextDict]):
    """
    Cache for preprocessed image-text data.
    """

    def __init__(self, cache: TreeCache[ImageTextDict]):
        super().__init__()
        self.cache = cache

    async def async_len(self) -> int:
        return await self.cache.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.cache.final_length_is_known()

    def is_finite(self) -> bool:
        return self.cache.is_finite()

    async def current_len(self) -> Optional[int]:
        return await self.cache.current_len()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ImageTextDict]:
        return await self.cache.get_batch(indices)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        source: ShardedDataSource[Dict[str, Any]],
        processor: ProcessorMixin,
        max_length: int = 2048,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        cache_options: CacheOptions = CacheOptions.default(),
        split: str = "",
    ) -> "ProcessedImageCache":
        # Extract grid_pinpoints and related params from processor for anyres support
        grid_pinpoints, patch_size, vision_feature_height, max_num_patches = _extract_anyres_params(processor)

        bp = BatchImageProcessor(
            processor,
            max_length=max_length,
            padding=padding,
            messages_key=messages_key,
            images_key=images_key,
            grid_pinpoints=grid_pinpoints,
            patch_size=patch_size,
            vision_feature_height=vision_feature_height,
            max_num_patches=max_num_patches,
        )
        cache = build_or_load_cache(cache_dir, source, bp, options=cache_options)

        if cache.is_finished:
            logger.info(f"Cache {cache_dir} is complete.")
        else:
            logger.info(f"Cache {cache_dir} is incomplete. Blocking until at least one chunk is complete.")

        return ProcessedImageCache(cache)

    @staticmethod
    def load(cache_dir: str) -> "ProcessedImageCache":
        """Load a ProcessedImageCache from a directory."""
        try:
            cache = TreeCache.load(cache_dir, ImageTextDict_exemplar, options=None)
            return ProcessedImageCache(cache)
        except FileNotFoundError:
            raise FileNotFoundError(f"{cache_dir} is not a complete cache")
        except Exception:
            logger.exception("Error loading cache")
            raise


@dataclass
class ImageIODatasetConfig(ImageDatasetSourceConfig, ImageTaskConfig):
    """Configuration for loading image-text data from HuggingFace or URLs."""

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> ProcessedImageCache:
        ds = self.build_or_load_cache(self.train_split, options)
        if ds is None:
            raise ValueError("No training set!")
        return ds

    def validation_set(self) -> Optional[ProcessedImageCache]:
        return self.build_or_load_cache(self.validation_split)

    def validation_sets(self) -> Mapping[str, ProcessedImageCache]:
        if self._has_validation_set:
            validation_set = self.validation_set()
            if validation_set is not None:
                return {"": validation_set}
        return {}

    @cached_property
    def _has_validation_set(self) -> bool:
        if len(self.validation_urls) > 0:
            return True

        if self.id is not None:
            try:
                dataset = datasets.load_dataset(
                    self.id, name=self.name, streaming=self.stream, split=self.validation_split
                )
                next(iter(dataset))
                return True
            except StopIteration:
                return False

        return False

    def build_or_load_cache(
        self,
        split: str,
        cache_options: CacheOptions = CacheOptions.default(),
    ) -> Optional[ProcessedImageCache]:
        split_cache_dir = os.path.join(self.cache_dir, split)

        try:
            return ProcessedImageCache.load(split_cache_dir)
        except FileNotFoundError:
            pass

        source = self.get_shard_source(split)
        if source is None:
            logger.info(f"No data for {split}")
            return None

        logger.info(f"Building cache for {split}...")

        # For simple image-text pairs, we need to convert to conversation format
        # The BatchImageProcessor expects messages_key and images_key
        return ProcessedImageCache.build_or_load(
            split_cache_dir,
            source,
            self.the_processor,
            max_length=self.max_length,
            padding=self.padding,
            messages_key="messages",  # Will be created by source mapping
            images_key="images",
            cache_options=cache_options,
        )


@dataclass
class ConversationIODatasetConfig(ConversationDatasetSourceConfig, ImageTaskConfig):
    """Configuration for loading conversation-format image-text data from HuggingFace or URLs."""

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> ProcessedImageCache:
        ds = self.build_or_load_cache(self.train_split, options)
        if ds is None:
            raise ValueError("No training set!")
        return ds

    def validation_set(self) -> Optional[ProcessedImageCache]:
        return self.build_or_load_cache(self.validation_split)

    def validation_sets(self) -> Mapping[str, ProcessedImageCache]:
        if self._has_validation_set:
            validation_set = self.validation_set()
            if validation_set is not None:
                return {"": validation_set}
        return {}

    @cached_property
    def _has_validation_set(self) -> bool:
        if len(self.validation_urls) > 0:
            return True

        if self.id is not None:
            try:
                dataset = datasets.load_dataset(
                    self.id, name=self.name, streaming=self.stream, split=self.validation_split
                )
                next(iter(dataset))
                return True
            except StopIteration:
                return False

        return False

    def build_or_load_cache(
        self,
        split: str,
        cache_options: CacheOptions = CacheOptions.default(),
    ) -> Optional[ProcessedImageCache]:
        split_cache_dir = os.path.join(self.cache_dir, split)

        try:
            return ProcessedImageCache.load(split_cache_dir)
        except FileNotFoundError:
            pass

        source = self.get_shard_source(split)
        if source is None:
            logger.info(f"No data for {split}")
            return None

        logger.info(f"Building cache for {split}...")

        return ProcessedImageCache.build_or_load(
            split_cache_dir,
            source,
            self.the_processor,
            max_length=self.max_length,
            padding=self.padding,
            messages_key=self.messages_key,
            images_key=self.images_key,
            cache_options=cache_options,
        )


class ImageTextExample(eqx.Module):
    """Example for vision-language model training/inference.

    Supports both image+text and text-only examples.
    For text-only, pixel_values is None and grid_mask is None.

    Uses fixed-shape processing for JIT compatibility:
    - pixel_values are padded to TOTAL_PATCHES = max_patches + 1
    - grid_mask indicates which patches are valid (True) vs padding (False)
    """

    pixel_values: Optional[NamedArray]  # (TOTAL_PATCHES, channels, height, width) - FIXED shape, padded
    input_ids: NamedArray  # (position,)
    loss_mask: Optional[NamedArray] = None  # (position,) - mask for loss computation (1.0 for valid, 0.0 for masked)
    # Boolean mask indicating valid patches (True for actual, False for padding)
    # Shape: (TOTAL_PATCHES,) where TOTAL_PATCHES = max_patches + 1
    grid_mask: Optional[NamedArray] = None
    # Pre-computed indices to reorder features to HF's unpadded order
    # Shape: (num_image_tokens,) - maps HF position to Levanter index
    unpad_indices: Optional[NamedArray] = None
    # Pre-computed validity mask for attention (1 for valid text/image, 0 for padding)
    # Shape: (position,) int32 - computed on CPU during data preprocessing
    combined_mask: Optional[NamedArray] = None
    # Pre-computed position IDs (cumsum of combined_mask - 1, clamped to 0)
    # Shape: (position,) int32 - computed on CPU during data preprocessing
    position_ids: Optional[NamedArray] = None

    @staticmethod
    def init(
        pixel_values: Optional[NamedArray],
        input_ids: NamedArray,
        loss_mask: Optional[NamedArray] = None,
        grid_mask: Optional[NamedArray] = None,
    ) -> "ImageTextExample":
        """Initialize an ImageTextExample with optional loss masking.

        Args:
            pixel_values: Image pixel values (FIXED shape, padded), or None for text-only
            input_ids: Token IDs
            loss_mask: Loss mask (float32) with 1.0 for valid tokens, 0.0 for masked.
            grid_mask: Boolean mask indicating valid patches (TOTAL_PATCHES,)
        """
        result_loss_mask = None
        if loss_mask is not None:
            # Ensure float32 dtype for loss computation
            mask_array = loss_mask.array if hasattr(loss_mask, "array") else loss_mask
            if mask_array.dtype != np.float32:
                mask_array = mask_array.astype(np.float32)
            result_loss_mask = NamedArray(mask_array, loss_mask.axes)

        return ImageTextExample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            loss_mask=result_loss_mask,
            grid_mask=grid_mask,
        )


class ImageTextDataset(MappedAsyncDataset[ImageTextDict, ImageTextExample]):
    """Dataset that converts ImageTextDict to ImageTextExample with proper axes."""

    def __init__(
        self,
        dataset: AsyncDataset[ImageTextDict],
        Position: Axis,
        NumPatches: Axis,
        Channels: Axis,
        Height: Axis,
        Width: Axis,
        key: Optional[PRNGKeyArray] = None,
        pixel_dtype: Optional[np.dtype] = None,
        grid_pinpoints: Optional[List[List[int]]] = None,
        patch_size: int = 384,
    ):
        """
        Args:
            dataset: Source dataset providing ImageTextDict
            Position: Axis for sequence position
            NumPatches: Axis for number of image patches
            Channels: Axis for image channels
            Height: Axis for image height
            Width: Axis for image width
            key: Optional random key
            pixel_dtype: dtype for pixel values when moving to device.
                        If None, uses the original dtype (float32).
                        Set to jnp.bfloat16 to save memory on TPU.
            grid_pinpoints: List of grid resolutions for anyres processing.
            patch_size: Size of each image patch (default 384).
        """
        self.dataset = dataset
        self.Position = Position
        self.NumPatches = NumPatches
        self.Channels = Channels
        self.Height = Height
        self.Width = Width
        self.key = key
        self.pixel_dtype = pixel_dtype
        self.grid_pinpoints = grid_pinpoints
        self.patch_size = patch_size

        # Process on CPU with numpy, avoid jnp.asarray() which would allocate on TPU
        # Use NamedArray constructor directly instead of hax.named() to keep data as numpy
        # DataLoader will handle the conversion to JAX arrays during batching
        def _convert_example(inputs: ImageTextDict) -> ImageTextExample:
            # All processing on CPU with numpy
            pv = inputs.get("pixel_values")

            # Handle text-only examples (pixel_values is None)
            if pv is None:
                pixel_values = None
            elif pv.ndim == 4:
                # (num_patches, channels, height, width)
                actual_num_patches = pv.shape[0]
                target_num_patches = self.NumPatches.size

                if actual_num_patches < target_num_patches:
                    # Pad with numpy (CPU)
                    pad_size = target_num_patches - actual_num_patches
                    padding = np.zeros((pad_size,) + pv.shape[1:], dtype=pv.dtype)
                    pv = np.concatenate([pv, padding], axis=0)
                elif actual_num_patches > target_num_patches:
                    pv = pv[:target_num_patches]

                # Convert to target dtype if specified (e.g., bfloat16 for TPU)
                # Use numpy for dtype conversion to keep data on CPU
                if self.pixel_dtype is not None:
                    np_dtype = np.dtype(self.pixel_dtype)
                    pv = pv.astype(np_dtype)

                # Use NamedArray directly to avoid jnp.asarray() in hax.named()
                # This keeps data as numpy array until DataLoader batches it
                pixel_values = NamedArray(pv, (self.NumPatches, self.Channels, self.Height, self.Width))
            elif pv.ndim == 3:
                if self.pixel_dtype is not None:
                    np_dtype = np.dtype(self.pixel_dtype)
                    pv = pv.astype(np_dtype)
                pixel_values = NamedArray(pv, (self.Channels, self.Height, self.Width))
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pv.shape}")

            # Keep input_ids as numpy array
            input_ids = NamedArray(inputs["input_ids"], (self.Position,))

            loss_mask = None
            if "loss_mask" in inputs:
                loss_mask = NamedArray(inputs["loss_mask"], (self.Position,))

            # Extract grid_mask from preprocessing (for fixed-shape processing)
            gm_arr = inputs.get("grid_mask")
            if gm_arr is not None:
                # Create NamedArray for grid_mask
                NumPatches = Axis("num_patches", gm_arr.shape[0])
                grid_mask = NamedArray(gm_arr, (NumPatches,))
            else:
                grid_mask = None

            out = ImageTextExample.init(
                pixel_values,
                input_ids,
                loss_mask=loss_mask,
                grid_mask=grid_mask,
            )
            return out

        super().__init__(self.dataset, _convert_example)


@dataclass
class ImageMixtureDatasetConfig(ImageTaskConfig):
    """Configuration for a mixture of image-text datasets with their associated weights.

    This class supports mixing multiple image-text data sources for training,
    similar to AudioMixtureDatasetConfig for audio data.

    Example:
        config = ImageMixtureDatasetConfig(
            cache_dir="cache/",
            configs={
                "coco": ImageDatasetSourceConfig(id="coco-dataset", ...),
                "llava": ConversationDatasetSourceConfig(id="llava-dataset", ...),
            },
            train_weights={"coco": 0.3, "llava": 0.7},
        )
    """

    cache_dir: Optional[str] = "cache/"

    # Data source configs and weights
    configs: Dict[str, Union[ImageDatasetSourceConfig, ConversationDatasetSourceConfig]] = field(default_factory=dict)
    """Configuration of each dataset source (URLs, HF dataset ID, etc.)"""
    train_weights: Dict[str, float] = field(default_factory=dict)
    """Weights for each dataset source. They will be normalized to sum to 1."""
    shuffle: bool | int = False
    """Whether to shuffle the dataset. True means shuffle the whole dataset, False means don't shuffle.
    If you want to shuffle in eras, set this to the era length."""
    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    mixture_block_size: int = 2048
    """Block size for the mixture dataset."""
    use_cache: bool = True
    """Whether to cache preprocessed data. Set to False for streaming mode (saves disk space)."""
    vision_feature_height: Optional[int] = None
    """Override vision feature height extracted from HF processor.

    This is useful when using a custom vision encoder with different patch size than
    what the HF processor expects. For example:
    - HF processor (llava-onevision) expects 27x27=729 features (384/14)
    - Your vision encoder (SigLIP with patch_size=16) outputs 24x24=576 features (384/16)

    Set this to 24 (= image_size / patch_size) to match your model's actual output."""

    # Packing configuration
    enable_packing: bool = False
    """Enable sequence packing for VLM training. Multiple samples are combined into
    a single training example with attention masks preventing cross-sample attention.
    IMPORTANT: Packing requires disable_anyres=True in model config.
    For streaming mode, provide pack_assignments_path. Otherwise, use_cache=True is required."""

    max_segments_per_pack: int = 64
    """Maximum number of samples that can be packed into a single training example."""

    packing_cache_dir: Optional[str] = None
    """Directory to cache pack assignments. If None, uses cache_dir/packing/."""

    pack_assignments_path: Optional[str] = None
    """Path to pre-computed pack assignments JSON file. When provided, enables streaming
    packing mode using PackedVLMDataset. This allows use_cache=False with packing.
    Generate this file using: scripts/compute_vlm_pack_assignments.py"""

    # Preprocessed data configuration
    use_preprocessed: bool = False
    """Use preprocessed and pre-packed parquet files. When True, data is loaded from
    preprocessed_train_urls instead of configs. The preprocessed files should contain
    already-packed samples with input_ids, segment_ids, position_ids, images, etc.
    Generate these files using: scripts/preprocess_vlm_data.py --enable-packing"""

    preprocessed_train_urls: Optional[List[str]] = None
    """URLs to preprocessed parquet files. Required when use_preprocessed=True.
    Example: ["gs://bucket/preprocessed/*.parquet"]"""

    preprocessed_max_patches: int = 10
    """Maximum number of image patches per packed sample in preprocessed data."""

    def __post_init__(self):
        # Skip configs validation when using preprocessed data
        if self.use_preprocessed:
            if not self.preprocessed_train_urls:
                raise ValueError("preprocessed_train_urls must be provided when use_preprocessed=True")
            return

        if len(self.configs) == 0:
            raise ValueError("At least one dataset must be provided")

        if set(self.configs.keys()) != set(self.train_weights.keys()):
            raise ValueError(
                f"The keys in configs and weights must be the same; got {self.configs.keys()} and"
                f" {self.train_weights.keys()}"
            )

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
        epochs: Optional[int] = None,
        max_num_patches: Optional[int] = None,
    ) -> AsyncDataset[ImageTextDict]:
        # Check for preprocessed mode first
        if self.use_preprocessed:
            if not self.preprocessed_train_urls:
                raise ValueError("preprocessed_train_urls must be provided when use_preprocessed=True")

            from levanter.data.vlm_packing import PreprocessedPackedVLMDataset

            # Expand glob patterns to get actual file paths
            all_paths = []
            for url_pattern in self.preprocessed_train_urls:
                fs, fs_path = fsspec.core.url_to_fs(url_pattern)
                matching = fs.glob(fs_path)
                if url_pattern.startswith("gs://"):
                    matching = [f"gs://{p}" for p in matching]
                elif url_pattern.startswith("s3://"):
                    matching = [f"s3://{p}" for p in matching]
                all_paths.extend(sorted(matching))

            if not all_paths:
                raise ValueError(f"No files found matching preprocessed_train_urls: {self.preprocessed_train_urls}")

            logger.info(f"Using preprocessed packed data: {len(all_paths)} files")

            max_patches = max_num_patches if max_num_patches is not None else self.preprocessed_max_patches

            dataset = PreprocessedPackedVLMDataset(
                parquet_paths=all_paths,
                image_processor=self.the_processor.image_processor,
                max_num_patches=max_patches,
                patch_size=384,
            )

            # Apply shuffle if configured
            if key is None:
                key = jax.random.PRNGKey(0)

            if self.shuffle is True:
                dataset = dataset.shuffle(key)
            elif isinstance(self.shuffle, int):
                dataset = dataset.era_shuffle(self.shuffle, key=key)

            # Apply epoch wrapping if configured
            if epochs and epochs > 0:
                dataset = EpochDataset(dataset, max_epochs=epochs)

            return dataset

        image_datasets = self.training_sets(max_num_patches=max_num_patches)

        if key is None:
            key = jax.random.PRNGKey(0)

        mix_key, shuffle_key = jax.random.split(key)

        # Shuffle components, not the overall mixture, to preserve "stable batch" property
        def shuffle_ds(ds, key):
            if self.shuffle is True:
                ds = ds.shuffle(key)
            elif isinstance(self.shuffle, int):
                ds = ds.era_shuffle(self.shuffle, key=key)
            return ds

        if self.shuffle:
            out_datasets = {}
            key_iter = key_iterator(shuffle_key)
            for name, ds in image_datasets.items():
                out_datasets[name] = shuffle_ds(ds, next(key_iter))
            image_datasets = out_datasets

        # Wrap each dataset in EpochDataset if epochs is specified and > 0
        # This is applied before mixing so each dataset cycles for the specified epochs
        if epochs and epochs > 0:
            logger.info(f"Wrapping each dataset in EpochDataset with max_epochs={epochs}")
            epoch_wrapped_datasets = {}
            for name, ds in image_datasets.items():
                epoch_wrapped_datasets[name] = EpochDataset(ds, max_epochs=epochs)
            image_datasets = epoch_wrapped_datasets

        mixture = MixtureDataset(
            datasets=image_datasets,
            weights=self.train_weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )

        # Apply packing if enabled
        if self.enable_packing:
            # Streaming packing requires pre-computed pack assignments
            if not self.use_cache and self.pack_assignments_path is None:
                raise ValueError(
                    "VLM Packing requires either use_cache=True or pack_assignments_path. "
                    "For streaming mode, provide pack_assignments_path from compute_vlm_pack_assignments.py"
                )

            if self.pack_assignments_path is not None:
                # Streaming packing: use pre-computed pack assignments
                from levanter.data.vlm_packing import PackedVLMDataset

                logger.info(f"Using streaming packing with pack_assignments: {self.pack_assignments_path}")

                # Determine disable_anyres from vision_aspect_ratio
                disable_anyres = (self.vision_aspect_ratio == "single")

                # PackedVLMDataset wraps the raw data source, not the processed mixture
                # It reads parquet directly and applies packing based on pre-computed assignments
                # IMPORTANT: Must pass all BatchImageProcessor configuration parameters!
                mixture = PackedVLMDataset(
                    pack_assignments_file=self.pack_assignments_path,
                    processor=self.the_processor,
                    max_length=self.max_length,
                    # BatchImageProcessor configuration - critical for correct token expansion
                    tokenizer=self.the_tokenizer,
                    disable_anyres=disable_anyres,
                    grid_pinpoints=self.image_grid_pinpoints,
                    vision_feature_height=self.vision_feature_height,
                    patch_size=384,  # Standard VLM patch size
                    prefetch_threshold=0.5,  # Default: start prefetching at 80% of shard
                )
            else:
                # Cached packing: compute pack assignments on-the-fly
                from levanter.data.vlm_packing import VLMPackerConfig, VLMPrepackedDataset

                # Compute features_per_patch from vision_feature_height
                if self.vision_feature_height is not None:
                    features_per_patch = self.vision_feature_height ** 2
                else:
                    features_per_patch = 576  # Default for 384x384 images with patch_size=16

                # Compute max_patches (disable_anyres mode means 1 patch per image typically)
                # For packing, we'll use a reasonable default
                max_patches = max_num_patches if max_num_patches is not None else 10

                # Packing cache directory
                packing_cache_dir = self.packing_cache_dir
                if packing_cache_dir is None and self.cache_dir is not None:
                    packing_cache_dir = os.path.join(self.cache_dir, "packing")

                packer_config = VLMPackerConfig(
                    max_length=self.max_length,
                    max_patches=max_patches,
                    max_segments=self.max_segments_per_pack,
                    features_per_patch=features_per_patch,
                    pad_token_id=self.pad_token_id,
                )

                logger.info(f"Enabling VLM packing with max_length={self.max_length}, max_patches={max_patches}, "
                            f"max_segments={self.max_segments_per_pack}, features_per_patch={features_per_patch}")

                mixture = VLMPrepackedDataset(
                    base_dataset=mixture,
                    config=packer_config,
                    cache_dir=packing_cache_dir,
                )

        return mixture

    def training_sets(self, max_num_patches: Optional[int] = None) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        if self.use_cache:
            return self.build_caches("train")
        else:
            return self.build_streaming_datasets("train", max_num_patches=max_num_patches)

    def validation_sets(self, max_num_patches: Optional[int] = None) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        if self.use_cache:
            return self.build_caches("validation")
        else:
            return self.build_streaming_datasets("validation", max_num_patches=max_num_patches)

    def build_streaming_datasets(
        self, split: str, max_num_patches: Optional[int] = None
    ) -> Dict[str, StreamingImageDataset]:
        """Build streaming datasets that process images on-the-fly without caching."""
        datasets_dict = {}

        # Use provided max_num_patches, otherwise try to extract from processor
        if max_num_patches is None:
            _, _, _, max_num_patches = _extract_anyres_params(self.the_processor)

        for name, source_config in self.configs.items():
            weight = self.train_weights.get(name, 0)

            if weight == 0 and split == "train":
                continue

            # Get the shard source
            if split == "train":
                source = source_config.get_shard_source(source_config.train_split)
            elif split == "validation":
                source = source_config.get_shard_source(source_config.validation_split)
            else:
                source = source_config.get_shard_source(split)

            if source is None:
                logger.warning(f"Skipping {name} for split {split} because no source was provided")
                continue

            # Determine messages_key and images_key
            if isinstance(source_config, ConversationDatasetSourceConfig):
                messages_key = source_config.messages_key
                images_key = source_config.images_key
            else:
                # For simple image-text pairs, the source already maps to messages/images format
                messages_key = "messages"
                images_key = "images"

            # Build streaming dataset
            streaming_ds = StreamingImageDataset.build(
                source=source,
                processor=self.the_processor,
                max_length=self.max_length,
                padding=self.padding,
                messages_key=messages_key,
                images_key=images_key,
                max_num_patches=max_num_patches,
                vision_feature_height=self.vision_feature_height,
            )

            datasets_dict[name] = streaming_ds
            # Get dataset size and log it
            try:
                dataset_len = asyncio.run(streaming_ds.async_len())
                if dataset_len == sys.maxsize:
                    logger.info(f"Built streaming dataset for {name} ({split}): streaming mode")
                else:
                    logger.info(f"Built streaming dataset for {name} ({split}): {dataset_len:,} datapoints")
            except Exception:
                logger.info(f"Built streaming dataset for {name} ({split})")

        return datasets_dict

    def build_caches(self, split: str) -> Dict[str, ProcessedImageCache]:
        # Forward all "Task" config fields to the dataset config for building
        task_config_fields = set(x.name for x in dataclasses.fields(ImageTaskConfig))
        task_config_dict = {k: v for k, v in self.__dict__.items() if k in task_config_fields and k != "cache_dir"}

        caches = {}
        for name, source_config in self.configs.items():
            weight = self.train_weights.get(name, 0)

            if weight == 0 and split == "train":
                continue

            source_config_dict = dict(**source_config.__dict__)

            if source_config.cache_dir is None:
                # Replace with the main cache_dir/{name}
                if self.cache_dir is None:
                    raise ValueError(
                        "If the 'main' cache_dir is None, then all component cache_dirs must be non-None, but"
                        f" {name}'s cache_dir is None."
                    )
                cache_dir = os.path.join(self.cache_dir, name)
                source_config_dict["cache_dir"] = cache_dir

            # Choose the correct config class based on source config type
            if isinstance(source_config, ConversationDatasetSourceConfig):
                dataset = ConversationIODatasetConfig(
                    **source_config_dict,
                    **task_config_dict,
                )
            else:
                dataset = ImageIODatasetConfig(
                    **source_config_dict,
                    **task_config_dict,
                )

            if split == "train":
                cache = dataset.build_or_load_cache(dataset.train_split)
            elif split == "validation":
                cache = dataset.build_or_load_cache(dataset.validation_split)
            else:
                cache = dataset.build_or_load_cache(split)

            # Drop the data source and corresponding weight if the cache is not built
            if cache is None:
                logger.warning(f"Skipping {name} for split {split} because no source was provided")
            else:
                caches[name] = cache
                # Get cache size and log it
                try:
                    cache_len = asyncio.run(cache.async_len())
                    logger.info(f"Built cache for {name} ({split}): {cache_len:,} datapoints")
                except Exception:
                    logger.info(f"Built cache for {name} ({split})")

        return caches

    @property
    def sources(self) -> Mapping[str, Union[ImageDatasetSourceConfig, ConversationDatasetSourceConfig]]:
        return self.configs


# =============================================================================
# LLaVA-OneVision Processor Classes
# =============================================================================
# Adapted from HuggingFace Transformers library:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/processing_llava_onevision.py
#
# Original code copyright 2024 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
#
# We acknowledge the LLaVA-OneVision team for their excellent work:
# https://github.com/LLaVA-VL/LLaVA-NeXT
# Paper: https://arxiv.org/abs/2408.03326
#
# These classes provide custom processor implementation for LLaVA-OneVision models
# with additional support for padding mode and fixed-shape processing.

# Get a transformers logger for the processor
_processor_logger = transformers_logging.get_logger(__name__)


class LlavaOnevisionProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "image_kwargs": {},
    }


class LlavaOnevisionProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-Onevision processor which wraps a LLaVa-Onevision video processor, LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaOnevisionVideoProcessor`], [`LlavaOnevisionImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaOnevisionVideoProcessor.__call__`], [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaOnevisionImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`LlavaOnevisionVideoProcessor`], *optional*):
            The video processor is a required input.
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LlavaOnevisionImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    video_processor_class = "LlavaOnevisionVideoProcessor"
    optional_attributes = ["video_processor", "chat_template"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        num_image_tokens=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        vision_aspect_ratio="anyres_max_9",
        max_image_tiles: Optional[int] = None,
        use_full_padded_tokens: bool = False,
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
        self.use_full_padded_tokens = use_full_padded_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_aspect_ratio = vision_aspect_ratio

        # For padding mode: max_image_tiles is the total number of tiles (including base)
        # e.g., for anyres_max_9, max_image_tiles = 9 + 1 = 10
        self.max_image_tiles = max_image_tiles
        if max_image_tiles is not None and num_image_tokens is not None:
            self.max_image_tokens = max_image_tiles * num_image_tokens
        else:
            self.max_image_tokens = None

        super().__init__(image_processor, tokenizer, video_processor=video_processor, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: Optional[VideoInput] = None,
        padding_mode: bool = False,
        **kwargs: Unpack[LlavaOnevisionProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = video_inputs = {}

        if images is not None:
            # Auto-wrap flat image list into nested list based on <image> token count in each text.
            # This enables HF processor to correctly compute batch_num_images.
            #
            # Example:
            #   images = [img1, img2, img3, img4, img5, img6]  # flat
            #   text = ["<image> <image> desc", "<image> desc", "<image> <image> <image> desc"]
            #   → images = [[img1, img2], [img3], [img4, img5, img6]]  # nested
            #   → batch_num_images = [2, 1, 3]
            is_flat_list = (
                isinstance(images, (list, tuple))
                and len(images) > 0
                and not isinstance(images[0], (list, tuple))
            )

            if is_flat_list:
                # Count <image> tokens in each text sample
                text_list = [text] if isinstance(text, str) else text
                num_images_per_sample = [t.count(self.image_token) for t in text_list]

                # Validate total images match before reorganizing
                total_expected = sum(num_images_per_sample)
                if total_expected == len(images):
                    # Reorganize flat list into nested list
                    images_nested = []
                    idx = 0
                    for num_img in num_images_per_sample:
                        images_nested.append(list(images[idx : idx + num_img]))
                        idx += num_img
                    images = images_nested
                # else: leave as-is, let HF processor handle mismatch error

            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            batch_num_images = iter(image_inputs["batch_num_images"])
            image_sizes = iter(image_inputs["image_sizes"])
            height, width = get_image_size(
                to_numpy_array(image_inputs["pixel_values"][0][0]),
                channel_dim=output_kwargs["images_kwargs"].get("data_format"),
            )
            text, num_image_tokens = self._expand_image_tokens(
                text,
                image_sizes,
                height,
                width,
                self.image_token,
                batch_num_images,
                padding_mode=padding_mode,
            )

        if videos is not None:
            video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])

            one_video = video_inputs.get("pixel_values_videos")[0]
            if isinstance(video_inputs.get("pixel_values_videos")[0], (list, tuple)):
                one_video = np.array(one_video)
            else:
                one_video = to_numpy_array(one_video)
            height, width = get_image_size(one_video[0], channel_dim=output_kwargs["images_kwargs"].get("data_format"))
            num_frames = one_video.shape[0]  # frame dim is always after batch dim
            patches_height_width = int(math.sqrt(self.num_image_tokens))
            pooled_height_width = math.ceil(patches_height_width / 2)
            num_video_tokens = num_frames * pooled_height_width * pooled_height_width
            text = [sample.replace(self.video_token, self.video_token * num_video_tokens) for sample in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=return_tensors)

    def _expand_image_tokens(
        self,
        text: list[TextInput],
        image_sizes: Iterable[Union[list[int], int]],
        height: int,
        width: int,
        special_token: str,
        batch_num_images: Iterable[int],
        padding_mode: bool = False,
    ):
        prompt_strings = []
        max_num_vision_tokens = 0
        for sample in text:
            if special_token in sample:
                # Count actual number of image tokens in the sample
                # batch_num_images may not be reliable for multi-image
                num_images = sample.count(special_token)
                _ = next(batch_num_images)  # consume iterator to stay in sync
                is_multi_image = num_images != 1
            else:
                is_multi_image = False
                num_images = 0
            while special_token in sample:
                original_size = next(image_sizes)  # should consume iterable

                # In padding mode:
                # - Multi-image: use base tokens only (729) - no anyres for multi-image
                # - Single image: use max tokens (7290) for JIT compatibility
                if padding_mode and self.max_image_tokens is not None:
                    if is_multi_image:
                        num_image_tokens = self.num_image_tokens  # Base patch only
                    else:
                        num_image_tokens = self.max_image_tokens  # Full anyres
                elif is_multi_image:
                    num_image_tokens = self.num_image_tokens
                else:
                    if not isinstance(original_size, (list, tuple)):
                        # cast to list to avoid numerical precision errors when calculating unpadding
                        original_size = original_size.tolist()
                    orig_height, orig_width = original_size
                    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)

                assert num_image_tokens is not None  # Always assigned in branches above
                max_num_vision_tokens = max(max_num_vision_tokens, num_image_tokens)
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1

                sample = sample.replace(special_token, "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        text = [sample.replace("<placeholder>", special_token) for sample in prompt_strings]
        return text, max_num_vision_tokens

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        # When not using anyres, only return base features (single resolution mode)
        if not self.vision_aspect_ratio.startswith("anyres"):
            return self.num_image_tokens

        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        # Levanter mode: return full padded token count (num_tiles × base_tokens)
        # Vision encoder outputs features for all tiles, unpadding handled via grid_mask/unpad_indices
        if self.use_full_padded_tokens:
            num_sub_patches = scale_height * scale_width
            num_tiles = num_sub_patches + 1  # sub-patches + base tile
            total_features = num_tiles * self.num_image_tokens  # e.g., 10 × 729 = 7290
            return total_features

        # HF-style: return unpadded features count
        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens
        num_image_tokens = unpadded_features + base_features
        return num_image_tokens

    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        # Only apply max_num_patches limit in anyres mode
        if self.vision_aspect_ratio.startswith("anyres_max_"):
            max_num_patches = int(self.vision_aspect_ratio.replace("anyres_max_", ""))
            ratio = math.sqrt(current_height * current_width / (max_num_patches * patches_height**2))
            if ratio > 1.1:
                unpadded_features = int(current_height // ratio) * int(current_width // ratio)
                newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    def _compute_unpad_indices(
        self,
        orig_height: int,
        orig_width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int,
        features_per_patch: int,
    ) -> np.ndarray:
        """
        Compute indices to reorder Levanter's padded features to HF's unpadded order.

        HF's pack_image_features applies spatial unpadding based on original image aspect ratio.
        This function computes the mapping from HF's feature positions to Levanter's sequential
        feature layout.

        Args:
            orig_height: Original image height
            orig_width: Original image width
            patches_height: Number of patches per tile in height (e.g., 27)
            patches_width: Number of patches per tile in width (e.g., 27)
            scale_height: Number of tiles in height (e.g., 3 for 3x3 grid)
            scale_width: Number of tiles in width (e.g., 3 for 3x3 grid)
            features_per_patch: Features per patch/tile (e.g., 729)

        Returns:
            unpad_indices: Array of shape (num_unpadded_features,) where
                          unpad_indices[i] = Levanter index for HF position i
        """
        # Base features are identity mapping (base patch is always first)
        base_indices = np.arange(features_per_patch)

        # Grid spatial dimensions after combining all tiles
        curr_height = patches_height * scale_height  # e.g., 81 for 3x3 grid of 27x27 patches
        curr_width = patches_width * scale_width

        # Compute unpadding bounds based on original aspect ratio
        # This matches HF's unpad_image logic
        original_aspect_ratio = orig_width / orig_height
        current_aspect_ratio = curr_width / curr_height

        if original_aspect_ratio > current_aspect_ratio:
            # Wider image - remove top/bottom padding
            scale_factor = curr_width / orig_width
            new_height = int(round(orig_height * scale_factor, 7))
            padding = (curr_height - new_height) // 2
            row_start = padding
            row_end = curr_height - padding  # Symmetric padding like HF
            col_start = 0
            col_end = curr_width
        else:
            # Taller image - remove left/right padding
            scale_factor = curr_height / orig_height
            new_width = int(round(orig_width * scale_factor, 7))
            padding = (curr_width - new_width) // 2
            row_start = 0
            row_end = curr_height
            col_start = padding
            col_end = curr_width - padding  # Symmetric padding like HF

        # Build mapping from HF grid position to Levanter grid index
        # HF order: row-major through unpadded region
        # Levanter order: patch-by-patch (tile-by-tile), then row-major within each patch
        grid_indices = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                # Convert global (row, col) to Levanter's patch-based index
                # Which tile (patch) does this position belong to?
                tile_row = row // patches_height
                tile_col = col // patches_width
                # Local position within the tile
                local_row = row % patches_height
                local_col = col % patches_width

                # Tile index in row-major order (0-indexed grid patch, excluding base)
                tile_idx = tile_row * scale_width + tile_col
                # Local feature index within the tile
                local_idx = local_row * patches_width + local_col

                # Levanter index: base_features + tile_idx * features_per_patch + local_idx
                # +1 because tile_idx=0 is the first grid tile, but Levanter's patch 0 is the base
                lev_idx = features_per_patch + tile_idx * features_per_patch + local_idx
                grid_indices.append(lev_idx)

        return np.concatenate([base_indices, np.array(grid_indices, dtype=np.int32)])

    def compute_unpad_indices(
        self,
        image_sizes: list,
        height: int,
        width: int,
        max_num_features: int,
    ) -> np.ndarray:
        """
        Compute unpad indices for a batch of images.

        Args:
            image_sizes: List of (orig_height, orig_width) tuples for each image
            height: Processed tile height (e.g., 384)
            width: Processed tile width (e.g., 384)
            max_num_features: Maximum number of features to pad to

        Returns:
            unpad_indices: Array of shape (batch, max_num_features) padded with zeros
        """
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        # Handle single resolution mode (no grid patches, only base)
        # Check both empty grid_pinpoints and vision_aspect_ratio="single"
        is_single_resolution = (
            not image_grid_pinpoints
            or not self.vision_aspect_ratio.startswith("anyres")
        )
        if is_single_resolution:
            # For single resolution, return identity mapping (no reordering needed)
            batch_size = len(image_sizes)
            return np.zeros((batch_size, max_num_features), dtype=np.int32)
        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))

        batch_indices = []
        for orig_height, orig_width in image_sizes:
            # Find best resolution for this image
            height_best_resolution, width_best_resolution = select_best_resolution(
                [orig_height, orig_width], image_grid_pinpoints
            )
            scale_height = height_best_resolution // height
            scale_width = width_best_resolution // width

            # Compute unpad indices for this image
            indices = self._compute_unpad_indices(
                orig_height,
                orig_width,
                patches_height,
                patches_width,
                scale_height,
                scale_width,
                self.num_image_tokens,
            )
            batch_indices.append(indices)

        # Pad all indices to max_num_features
        padded_indices = np.zeros((len(batch_indices), max_num_features), dtype=np.int32)
        for i, indices in enumerate(batch_indices):
            padded_indices[i, : len(indices)] = indices

        return padded_indices

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (list[list[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (list[list[str]], *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
            audio_lengths (list[int], *optional*):
                The input length formatted as per each audio.
        Returns:
            dict[str, list[int]]: A dictionary mapping each modality ("image", "video", "audio")
            to a list containing the number of placeholder tokens required. If the model doesn't accept
            a certain modality or no input sizes are provided, the dict value is set to an empty list.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LlavaOnevisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            assert isinstance(size, dict)  # size should be a dict with height/width or shortest_edge
            size = (
                (size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"]))
            )
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)  # llava-ov doesn't batch pixels as Idefics, thus `1` patch`
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, processed_height, processed_width
                )
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                batch_num_image_tokens.append(num_image_tokens)
            vision_data.update({"num_image_tokens": batch_num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


DEFAULT_IMAGE_GRID_PINPOINTS = [
    [384, 384],
    [384, 768],
    [384, 1152],
    [768, 384],
    [768, 768],
    [768, 1152],
    [1152, 384],
    [1152, 768],
    [1152, 1152],
]


def create_custom_processor(
    model_name,
    do_pad=True,
    image_grid_pinpoints=None,
    max_image_tiles=None,
    vision_aspect_ratio=None,
):
    """
    Create a LlavaOnevisionProcessor with custom do_pad setting.

    Args:
        model_name: HuggingFace model name
        do_pad: Whether to pad image patches (True for Levanter, False for HF reference)
        image_grid_pinpoints: Optional custom grid pinpoints. If None, uses DEFAULT_IMAGE_GRID_PINPOINTS.
        max_image_tiles: Maximum number of image tiles (including base) for padding mode.
                         For anyres_max_9, this would be 10 (9 + 1 base).
                         Required when using padding_mode=True when calling the processor.
        vision_aspect_ratio: Optional aspect ratio mode. Use "single" to disable anyres
                             and only output base patch. If None, uses model config default.
    """
    from transformers import AutoConfig, AutoImageProcessor
    from levanter.compat.hf_checkpoints import load_tokenizer, load_processor, hf_load_with_retry

    if image_grid_pinpoints is None:
        image_grid_pinpoints = DEFAULT_IMAGE_GRID_PINPOINTS

    # Load config with retry logic for rate limits
    config = hf_load_with_retry(AutoConfig.from_pretrained, model_name)

    # Load the HF processor to get the chat template (with sync/retry)
    hf_processor = load_processor(model_name)
    chat_template = hf_processor.chat_template

    # Load tokenizer from HF (with sync/retry)
    tokenizer = load_tokenizer(model_name)

    # Load image processor from HF and configure do_pad (with retry)
    image_processor = hf_load_with_retry(AutoImageProcessor.from_pretrained, model_name)
    image_processor.do_pad = do_pad
    image_processor.image_grid_pinpoints = image_grid_pinpoints

    # Calculate num_image_tokens (patches per image = (image_size / patch_size)^2)
    image_size = config.vision_config.image_size  # e.g., 384
    patch_size = config.vision_config.patch_size  # e.g., 14
    num_image_tokens = (image_size // patch_size) ** 2  # e.g., 729

    # Use provided vision_aspect_ratio or fall back to config
    # "single" disables anyres and only outputs base patch
    effective_vision_aspect_ratio = vision_aspect_ratio if vision_aspect_ratio else config.vision_aspect_ratio

    # Create the custom processor with required parameters
    processor = LlavaOnevisionProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        num_image_tokens=num_image_tokens,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        vision_aspect_ratio=effective_vision_aspect_ratio,
        chat_template=chat_template,
        max_image_tiles=max_image_tiles,
    )
    return processor


# =====================================================================================
# ImageDataLoader and ImageDataLoaderIterator
# Note: JAX distributed should be initialized before importing this module
# =====================================================================================


class ImageDataLoader(DataLoader):
    """
    Data loader for image-text (VLM) data.

    This loader extends DataLoader with special handling for vision-language models:
    - Variable number of image patches per example
    - Multiple fields (pixel_values, input_ids, image_sizes, labels, loss_mask)
    - Proper batching and padding for TPU efficiency

    The loader expects data in ImageTextDict format from ProcessedImageCache.
    """

    def __init__(
        self,
        data: AsyncDataset[ImageTextDict],
        batch_size: int | IntSchedule | hax.Axis,
        *,
        Pos: hax.Axis,
        NumPatches: hax.Axis,
        Channels: hax.Axis = hax.Axis("channels", 3),
        Height: hax.Axis = hax.Axis("height", 384),
        Width: hax.Axis = hax.Axis("width", 384),
        batch_axis_name: str | None = None,
        max_buffered_batches: int | None = 64,
        mesh: Mesh | None = None,
        axis_resources: ResourceMapping | None = None,
        prefetch_size: int = 32,
        pad_final_batch: bool = True,
        allow_nondivisible_batch_size: bool = False,
        pixel_dtype: Optional[numpy.dtype] = None,
        NumImageTokens: Optional[hax.Axis] = None,
    ):
        """
        Initialize ImageDataLoader.

        Args:
            data: AsyncDataset providing ImageTextDict examples
            batch_size: Batch size or schedule
            Pos: Position axis for sequence length
            NumPatches: Axis for number of image patches
            Channels: Axis for image channels (default: 3)
            Height: Axis for patch height (default: 384)
            Width: Axis for patch width (default: 384)
            batch_axis_name: Name for batch axis
            max_buffered_batches: Max batches to buffer
            mesh: JAX mesh for sharding
            axis_resources: Resource mapping for sharding
            prefetch_size: Number of batches to prefetch
            pad_final_batch: Whether to pad final batch
            allow_nondivisible_batch_size: Allow non-divisible batch sizes
            pixel_dtype: dtype for pixel values (default: float32). Set to bfloat16 to save memory.
            NumImageTokens: Axis for number of image tokens (for unpad_indices). If provided,
                           unpad_indices will be included in batches for HF-compatible feature ordering.

        Note:
            grid_mask is computed during batching and included in the ImageTextExample data
            for JIT-compatible VLM training.
        """
        # Set image-specific attributes before calling super().__init__()
        # because _make_padding_example (called in super) may need these
        self.Pos = Pos
        self.NumPatches = NumPatches
        self.Channels = Channels
        self.Height = Height
        self.Width = Width
        self.NumImageTokens = NumImageTokens
        self.pixel_dtype = pixel_dtype if pixel_dtype is not None else numpy.float32

        # Call parent constructor
        super().__init__(
            data=data,
            batch_size=batch_size,
            batch_axis_name=batch_axis_name,
            max_buffered_batches=max_buffered_batches,
            mesh=mesh,
            axis_resources=axis_resources,
            prefetch_size=prefetch_size,
            pad_final_batch=pad_final_batch,
            allow_nondivisible_batch_size=allow_nondivisible_batch_size,
        )

    def _make_padding_example(self, ex: ImageTextDict) -> ImageTextDict:
        """Create a zero-padded example for padding incomplete batches."""
        padding_dict: dict[str, Any] = {}
        for key, value in ex.items():
            if value is None:
                padding_dict[key] = None
            else:
                padding_dict[key] = numpy.zeros_like(value)
        return cast(ImageTextDict, padding_dict)

    def iter_from_step(self, start_from_batch: int | None = None):
        start_from_batch = int(start_from_batch) if start_from_batch is not None else None
        return ImageDataLoaderIterator(self, start_from_batch=start_from_batch)


class ImageDataLoaderIterator(DataLoaderIterator):
    """Iterator for ImageDataLoader.

    Inherits batch production and data retrieval from DataLoaderIterator,
    overriding only the image-specific batching logic.
    """

    def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec | tuple) -> PartitionSpec:
        """Get partition spec for a given set of axes."""
        if isinstance(shape_spec, NamedShapeSpec):
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.dl.axis_resources)
        elif isinstance(shape_spec, tuple) and len(shape_spec) > 0 and isinstance(shape_spec[0], hax.Axis):
            # Handle tuple of hax.Axis objects directly
            return hax.partitioning.pspec_for_axis(shape_spec, self.dl.axis_resources)
        else:
            # ShapeSpec - shouldn't happen for image data, but handle it for type safety
            batch_name = hax.partitioning.physical_axis_name(self.dl.batch_axis_name, self.dl.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))

    def _pad_pixel_values_to_num_patches(self, pixel_values: numpy.ndarray, target_num_patches: int) -> numpy.ndarray:
        """Pad pixel_values to target number of patches.

        Args:
            pixel_values: Image patches array of shape (actual_patches, C, H, W)
            target_num_patches: Target number of patches

        Returns:
            Padded pixel_values of shape (target_num_patches, C, H, W)
        """
        actual_patches = pixel_values.shape[0]
        if actual_patches >= target_num_patches:
            return pixel_values[:target_num_patches]

        # Pad to target size
        pad_size = target_num_patches - actual_patches
        padding = numpy.zeros((pad_size,) + pixel_values.shape[1:], dtype=pixel_values.dtype)
        return numpy.concatenate([pixel_values, padding], axis=0)

    def _pad_sequence_to_pos(self, arr: numpy.ndarray, target_len: int, pad_value=0) -> numpy.ndarray:
        """Pad or truncate sequence to target length (Pos.size).

        Args:
            arr: 1D array to pad/truncate
            target_len: Target sequence length
            pad_value: Value to use for padding (default 0)

        Returns:
            Array of length target_len
        """
        actual_len = len(arr)
        if actual_len == target_len:
            return arr
        if actual_len > target_len:
            return arr[:target_len]
        # Pad to target length
        padded = numpy.zeros(target_len, dtype=arr.dtype)
        if pad_value != 0:
            padded.fill(pad_value)
        padded[:actual_len] = arr
        return padded

    def _batchify_local_data(self, batch: _Batch[ImageTextDict]) -> ImageTextExample:
        """
        Stack individual ImageTextDict examples into a batched ImageTextExample.
        Uses jax.make_array_from_callback for proper device placement.
        """
        padded_batch_size = self.dl._round_batch_size(batch.global_size)
        Batch = hax.Axis(self.dl.batch_axis_name, padded_batch_size)

        # Get target sizes from the axes
        target_seq_len = self.dl.Pos.size
        target_num_patches = self.dl.NumPatches.size

        # Determine axes for each field
        # Always include NumPatches dimension (even when size=1) for model compatibility
        pixel_axes = (Batch, self.dl.NumPatches, self.dl.Channels, self.dl.Height, self.dl.Width)
        input_axes = (Batch, self.dl.Pos)

        # Cache for local data
        local_data_cache: dict[int, ImageTextDict] = {}

        def get_local_data(idx: int) -> ImageTextDict:
            if idx not in local_data_cache:
                if idx in batch.data_by_local_index:
                    local_data_cache[idx] = batch.data_by_local_index[idx]
                else:
                    local_data_cache[idx] = self.dl._padding_example
            return local_data_cache[idx]

        # Helper to create sharded arrays
        def make_sharded_array(
            shape: tuple[int, ...],
            axes: tuple[hax.Axis, ...],
            dtype: numpy.dtype,
            get_data_fn,
        ) -> hax.NamedArray:
            """Create a properly sharded NamedArray."""
            pspec = self._pspec_for(axes)
            sharding = jax.sharding.NamedSharding(self.dl.mesh, pspec)

            def callback(indices):
                batch_slice = indices[0]
                begin, end, stride = batch_slice.indices(padded_batch_size)
                assert stride == 1, "Stride must be 1"

                # Collect data for this slice
                data_list = []
                for i in range(begin, end):
                    data_list.append(get_data_fn(get_local_data(i)))

                stacked = numpy.stack(data_list, axis=0)
                # Apply remaining indices
                other_indices = indices[1:]
                if not all(idx == slice(None) for idx in other_indices):
                    stacked = stacked[(..., *other_indices)]
                return stacked

            raw_array = jax.make_array_from_callback(shape, sharding, callback)
            return hax.NamedArray(raw_array, axes)

        # Create pixel_values
        pixel_shape = tuple(ax.size for ax in pixel_axes)

        def get_pixel_values(d: ImageTextDict) -> numpy.ndarray:
            pv = d["pixel_values"]
            if pv is None:
                # Pure text sample - return zeros (same as padding)
                # Shape: (num_patches, channels, height, width)
                return numpy.zeros(pixel_shape[1:], dtype=self.dl.pixel_dtype)
            if pv.ndim == 4 and target_num_patches > 1:
                pv = self._pad_pixel_values_to_num_patches(pv, target_num_patches)
            elif pv.ndim == 4 and target_num_patches == 1:
                # When target_num_patches=1 (disable_anyres), keep only the first patch
                # pixel_values shape: (actual_patches, C, H, W) -> (1, C, H, W)
                pv = pv[:1]
            return pv.astype(self.dl.pixel_dtype)

        pixel_values = make_sharded_array(pixel_shape, pixel_axes, self.dl.pixel_dtype, get_pixel_values)

        # Create input_ids
        input_shape = tuple(ax.size for ax in input_axes)

        def get_input_ids(d: ImageTextDict) -> numpy.ndarray:
            ids = d["input_ids"].astype(numpy.int32)
            return self._pad_sequence_to_pos(ids, target_seq_len, pad_value=0)

        input_ids = make_sharded_array(input_shape, input_axes, numpy.int32, get_input_ids)

        # Get loss_mask directly from preprocessed data
        def get_loss_mask(d: ImageTextDict) -> numpy.ndarray:
            mask = d["loss_mask"].astype(numpy.float32)
            return self._pad_sequence_to_pos(mask, target_seq_len, pad_value=0.0)

        loss_mask = make_sharded_array(input_shape, input_axes, numpy.float32, get_loss_mask)

        # Create grid_mask as a NamedArray for JIT-compatible VLM training
        # grid_mask indicates which patches are valid (True) vs padding (False)
        grid_mask_axes = (Batch, self.dl.NumPatches)
        grid_mask_shape = (padded_batch_size, target_num_patches)

        def get_grid_mask(d: ImageTextDict) -> numpy.ndarray:
            # Use cached grid_mask from BatchImageProcessor if available
            cached_mask = d.get("grid_mask")
            if cached_mask is not None:
                # Pad or truncate to target size if needed
                if len(cached_mask) > target_num_patches:
                    logger.warning(f"Truncating grid_mask from {len(cached_mask)} to {target_num_patches}")
                    return cached_mask[:target_num_patches]
                if len(cached_mask) == target_num_patches:
                    return cached_mask
                mask = numpy.zeros(target_num_patches, dtype=numpy.bool_)
                mask[: len(cached_mask)] = cached_mask
                return mask
            # Fallback: compute from pixel_values shape (for backwards compatibility)
            pv = d["pixel_values"]
            if pv is None:
                # Pure text sample - all patches are padding (no valid image patches)
                # This ensures combined_mask will mask out all image placeholder tokens
                return numpy.zeros(target_num_patches, dtype=numpy.bool_)
            actual_patches = pv.shape[0] if pv.ndim == 4 else 1
            mask = numpy.zeros(target_num_patches, dtype=numpy.bool_)
            mask[:actual_patches] = True
            return mask

        grid_mask = make_sharded_array(grid_mask_shape, grid_mask_axes, numpy.bool_, get_grid_mask)

        # Create unpad_indices if NumImageTokens is configured
        unpad_indices = None
        if self.dl.NumImageTokens is not None:
            unpad_axes = (Batch, self.dl.NumImageTokens)
            unpad_shape = (padded_batch_size, self.dl.NumImageTokens.size)

            def get_unpad_indices(d: ImageTextDict) -> numpy.ndarray:
                indices = d.get("unpad_indices")
                if indices is not None:
                    target_size = self.dl.NumImageTokens.size
                    # Pad or truncate to target size
                    if len(indices) < target_size:
                        padded = numpy.zeros(target_size, dtype=numpy.int32)
                        padded[: len(indices)] = indices
                        return padded
                    if len(indices) > target_size:
                        logger.warning(f"Truncating unpad_indices from {len(indices)} to {target_size}")
                    return indices[:target_size].astype(numpy.int32)
                return numpy.zeros(self.dl.NumImageTokens.size, dtype=numpy.int32)

            unpad_indices = make_sharded_array(unpad_shape, unpad_axes, numpy.int32, get_unpad_indices)

        # Create combined_mask NamedArray (precomputed validity mask)
        def get_combined_mask(d: ImageTextDict) -> numpy.ndarray:
            mask = d.get("combined_mask")
            if mask is not None:
                mask = mask.astype(numpy.int32)
                return self._pad_sequence_to_pos(mask, target_seq_len, pad_value=0)
            # Fallback: create from attention_mask
            attn = d["attention_mask"].astype(numpy.int32)
            return self._pad_sequence_to_pos(attn, target_seq_len, pad_value=0)

        combined_mask = make_sharded_array(input_shape, input_axes, numpy.int32, get_combined_mask)

        # Create position_ids NamedArray (precomputed position IDs)
        def get_position_ids(d: ImageTextDict) -> numpy.ndarray:
            pos_ids = d.get("position_ids")
            if pos_ids is not None:
                pos_ids = pos_ids.astype(numpy.int32)
                return self._pad_sequence_to_pos(pos_ids, target_seq_len, pad_value=0)
            # Fallback: compute from attention_mask using cumsum
            attn_mask = d["attention_mask"].astype(numpy.int32)
            attn_mask = self._pad_sequence_to_pos(attn_mask, target_seq_len, pad_value=0)
            pos_ids = numpy.cumsum(attn_mask) - 1
            return numpy.maximum(pos_ids, 0).astype(numpy.int32)

        position_ids = make_sharded_array(input_shape, input_axes, numpy.int32, get_position_ids)

        return ImageTextExample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            loss_mask=loss_mask,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            combined_mask=combined_mask,
            position_ids=position_ids,
        )

    async def _do_retrieve_batch_of_batches(self, batch_specs: list[_Batch[None]]) -> list[_Batch[ImageTextDict]]:
        """Retrieve the data for a batch of batches."""
        global_indices_for_each_batch = []

        for batch in batch_specs:
            global_offset = batch.global_data_offset
            local_indices_for_device = self.dl.local_data_indices_by_device_for_step(batch.index)

            distinct_local_indices_this_batch = set()
            for indices in local_indices_for_device.values():
                for local_index in indices:
                    if local_index >= batch.global_size:
                        continue
                    distinct_local_indices_this_batch.add(local_index)

            global_indices_for_this_batch = [global_offset + i for i in distinct_local_indices_this_batch]
            global_indices_for_each_batch.append(global_indices_for_this_batch)

        indices_for_this_batch_of_batches: list[int] = [
            i for indices in global_indices_for_each_batch for i in indices
        ]

        individual_datums = await self.run_and_report_slowness(
            self.dl.data_store.get_batch(indices_for_this_batch_of_batches),
            f"Waiting for {len(indices_for_this_batch_of_batches)} image items.",
        )

        global_map: dict[int, ImageTextDict] = dict(zip(indices_for_this_batch_of_batches, individual_datums))

        out: list[_Batch[ImageTextDict]] = []

        for batch, global_indices_batch in zip(batch_specs, global_indices_for_each_batch, strict=False):
            local_index_to_example = {}
            for global_index in global_indices_batch:
                local_index = global_index - batch.global_data_offset
                local_index_to_example[local_index] = global_map[global_index]

            out.append(dataclasses.replace(batch, data_by_local_index=local_index_to_example))

        return out

