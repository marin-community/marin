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
import logging
import math
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, cast

import braceexpand
import datasets
import equinox as eqx
import fsspec
import jax
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
    ConversationUrlDataSource,
    ImageTextUrlDataSource,
    ShardedDataSource,
    WrappedHFDataSource,
)
from levanter.store.cache import CacheOptions, TreeCache, build_or_load_cache
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag

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
            fs = fsspec.core.url_to_fs(url)[0]
            globbed = fs.glob(url)
            return globbed if globbed else [url]
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
    labels: np.ndarray  # (seq_len,)
    # Grid mask for fixed-shape processing - indicates which patches are valid (not padding)
    grid_mask: Optional[np.ndarray]  # (TOTAL_PATCHES,) boolean - True for valid patches
    # Unpad indices for anyres processing
    unpad_indices: Optional[np.ndarray]  # (num_image_tokens,) - indices for unpadding image features


ImageTextDict_exemplar: ImageTextDict = {
    "pixel_values": np.zeros((1, 3, 384, 384), dtype=np.float32),
    "input_ids": np.zeros((1,), dtype=np.int32),
    "attention_mask": np.zeros((1,), dtype=np.int32),
    "image_sizes": np.zeros((1, 2), dtype=np.int32),
    "labels": np.zeros((1,), dtype=np.int32),
    "grid_mask": None,  # Always included, may be None
    "unpad_indices": None,  # Always included, may be None
}


def load_image_from_path_or_url(path_or_url: str) -> Image.Image:
    """Load an image from a local path or URL.

    Args:
        path_or_url: Local file path or URL to the image

    Returns:
        PIL Image in RGB format
    """
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
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

    vision_aspect_ratio = getattr(image_processor, "vision_aspect_ratio", None)
    if vision_aspect_ratio and isinstance(vision_aspect_ratio, str) and "anyres_max_" in vision_aspect_ratio:
        try:
            max_num_patches = int(vision_aspect_ratio.split("anyres_max_")[-1])
        except (ValueError, IndexError):
            pass

    return grid_pinpoints, patch_size, vision_feature_height, max_num_patches


class BatchImageProcessor(BatchProcessor[Dict[str, Any], ImageTextDict]):
    """
    A batch processor that converts conversation-format data into model-ready inputs.

    This processor handles the conversation format used by VLMs like LLaVA:
    - Applies chat template to convert messages to text with image placeholders
    - Processes images using the HuggingFace processor
    - Creates labels for training (masking non-assistant tokens with -100)

    Input format:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": [<image_data>]  # PIL, path, URL, or HF bytes dict
    }
    """

    # Ignore index for loss computation (standard value used by HuggingFace)
    IGNORE_INDEX = -100

    # Critical special tokens that must match between processor and LLM tokenizer
    # These are essential for chat template formatting and label masking
    CRITICAL_SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]
    # Tokens used for role identification in chat templates
    CRITICAL_ROLE_TOKENS = ["assistant", "user", "system"]

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
            mask_prompt: Whether to mask (set to -100) non-assistant tokens in labels
            override_resources: Optional resource overrides
            grid_pinpoints: List of grid resolutions for anyres processing, e.g., [[384,384], [768,384], ...]
            patch_size: Size of each image patch (default 384)
            vision_feature_height: Vision encoder output tokens per spatial dim (e.g., 27 for 384/14)
            max_num_patches: Maximum number of patches for anyres constraint (e.g., 9 for anyres_max_9)
        """
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
        self.messages_key = messages_key
        self.images_key = images_key
        self.add_generation_prompt = add_generation_prompt
        self.mask_prompt = mask_prompt
        self.override_resources = override_resources

        # Parameters for computing grid_mask for JIT-compatible VLM training
        self.grid_pinpoints = grid_pinpoints
        self.patch_size = patch_size
        self.vision_feature_height = vision_feature_height
        self.max_num_patches = max_num_patches

        # Pre-compute grid_pinpoints arrays for vectorized _compute_grid_shape
        if grid_pinpoints is not None:
            self._grid_h = np.array([p[0] for p in grid_pinpoints], dtype=np.float64)
            self._grid_w = np.array([p[1] for p in grid_pinpoints], dtype=np.float64)
            self._grid_area = self._grid_h * self._grid_w
        else:
            self._grid_h = None
            self._grid_w = None
            self._grid_area = None

        # Replace processor's tokenizer with provided tokenizer if specified
        if tokenizer is not None:
            self._replace_tokenizer(tokenizer)

        # Cache padding mode for __call__
        self._padding_mode = "max_length" if self.padding else False

        # Eagerly cache token IDs for _create_labels (after any tokenizer replacement)
        final_tokenizer = self.processor.tokenizer
        self._cached_im_start_id: int = final_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._cached_im_end_id: int = final_tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_ids = final_tokenizer.encode("assistant", add_special_tokens=False)
        self._cached_num_assistant_tokens: int = len(assistant_ids)
        self._cached_assistant_token_ids_array: np.ndarray = np.array(assistant_ids, dtype=np.int32)

    def _replace_tokenizer(self, new_tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Replace the processor's tokenizer with a new tokenizer.

        This is useful when you want to use an LLM's tokenizer (e.g., Qwen3-1.7B) instead of
        the processor's default tokenizer, to ensure consistent tokenization during training.

        The method will:
        1. Verify critical special tokens match between old and new tokenizer
        2. Add image/video tokens to the new tokenizer if missing
        3. Update processor's image_token_id/video_token_id to match the new tokenizer

        Args:
            new_tokenizer: The new tokenizer to use (e.g., from AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"))

        Raises:
            AssertionError: If critical special tokens don't match between old and new tokenizer
        """
        old_tokenizer = self.processor.tokenizer

        # Verify vocab size matches
        assert old_tokenizer.vocab_size == new_tokenizer.vocab_size, (
            f"Tokenizer vocab size mismatch: processor has {old_tokenizer.vocab_size}, "
            f"new tokenizer has {new_tokenizer.vocab_size}"
        )

        # Verify critical special tokens have the same IDs
        for token in self.CRITICAL_SPECIAL_TOKENS:
            old_id = old_tokenizer.convert_tokens_to_ids(token)
            new_id = new_tokenizer.convert_tokens_to_ids(token)
            assert old_id == new_id, (
                f"Critical special token '{token}' ID mismatch: " f"processor has {old_id}, new tokenizer has {new_id}"
            )

        # Verify role tokens have the same IDs
        for token in self.CRITICAL_ROLE_TOKENS:
            old_id = old_tokenizer.convert_tokens_to_ids(token)
            new_id = new_tokenizer.convert_tokens_to_ids(token)
            assert old_id == new_id, (
                f"Critical role token '{token}' ID mismatch: " f"processor has {old_id}, new tokenizer has {new_id}"
            )

        # Verify eos_token_id matches
        assert old_tokenizer.eos_token_id == new_tokenizer.eos_token_id, (
            f"eos_token_id mismatch: processor has {old_tokenizer.eos_token_id}, "
            f"new tokenizer has {new_tokenizer.eos_token_id}"
        )

        # Check if this is a Qwen3 tokenizer by looking for Qwen3-specific tokens
        # Qwen3 has <|image_pad|>, <|video_pad|>, <think>, </think> tokens
        qwen3_image_token = "<|image_pad|>"
        qwen3_video_token = "<|video_pad|>"
        # convert_tokens_to_ids returns unk_token_id for unknown tokens, not None
        qwen3_image_token_id = new_tokenizer.convert_tokens_to_ids(qwen3_image_token)
        is_qwen3 = qwen3_image_token_id != new_tokenizer.unk_token_id

        if is_qwen3:
            # Update processor's image_token to Qwen3's <|image_pad|>
            new_image_id = new_tokenizer.convert_tokens_to_ids(qwen3_image_token)
            old_image_id = getattr(self.processor, "image_token_id", None)
            self.processor.image_token = qwen3_image_token
            self.processor.image_token_id = new_image_id
            logger.info(f"Updated processor image_token: {old_image_id} -> {new_image_id} ({qwen3_image_token})")

            # Update processor's video_token to Qwen3's <|video_pad|>
            new_video_id = new_tokenizer.convert_tokens_to_ids(qwen3_video_token)
            old_video_id = getattr(self.processor, "video_token_id", None)
            self.processor.video_token = qwen3_video_token
            self.processor.video_token_id = new_video_id
            logger.info(f"Updated processor video_token: {old_video_id} -> {new_video_id} ({qwen3_video_token})")
        else:
            raise NotImplementedError(f"Tokenizer {type(new_tokenizer).__name__} is not supported")

        # Replace the tokenizer
        self.processor.tokenizer = new_tokenizer
        logger.info(
            f"Replaced processor tokenizer with {type(new_tokenizer).__name__} "
            f"(vocab_size={new_tokenizer.vocab_size})"
        )

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
        """Compute (gh, gw) grid shape for an image using vectorized numpy operations.

        This is used for pre-computing grid shapes during CPU preprocessing so they can be
        passed as concrete Python ints to pack_image_features, enabling JIT compilation.

        Args:
            image_size: (height, width) of the original image

        Returns:
            (gh, gw) grid dimensions as Python ints
        """
        if self._grid_h is None or self.patch_size is None:
            return (1, 1)  # Default for no anyres

        orig_h, orig_w = image_size
        orig_area = orig_h * orig_w

        # Vectorized computation of scales for all grid resolutions
        # scale = min(w/orig_w, h/orig_h) for each resolution
        scales = np.minimum(self._grid_w / orig_w, self._grid_h / orig_h)

        # Compute scaled dimensions and effective area
        scaled_h = (orig_h * scales).astype(np.int64)
        scaled_w = (orig_w * scales).astype(np.int64)
        eff = np.minimum(scaled_h * scaled_w, orig_area)

        # Compute waste (area not used)
        waste = self._grid_area - eff

        # Combined score: maximize eff first, then minimize waste
        # Use large multiplier to ensure eff dominates
        scores = eff.astype(np.float64) * 1e12 - waste
        best_idx = int(np.argmax(scores))

        assert self.grid_pinpoints is not None
        best_h, best_w = self.grid_pinpoints[best_idx]
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

    def _pad_pixel_values(self, pixel_values: np.ndarray, valid_patches: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad pixel_values to fixed TOTAL_PATCHES size and create grid_mask.

        Args:
            pixel_values: Image patches array of shape (actual_patches, C, H, W)
            valid_patches: Number of patches to mark as valid in grid_mask

        Returns:
            Tuple of (padded_pixel_values, grid_mask)
        """
        assert self.max_num_patches is not None
        total_patches = self.max_num_patches + 1  # +1 for base patch
        actual_patches = pixel_values.shape[0]

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

    def _create_labels(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Create labels for training by masking non-assistant tokens.

        For causal LM training, we only compute loss on assistant responses.
        All other tokens (system, user, special tokens) are masked with IGNORE_INDEX.

        This is an efficient vectorized implementation that works directly on token IDs
        without decoding, similar to HuggingFace's return_assistant_tokens_mask.

        The algorithm identifies assistant response spans by looking for:
            <|im_start|>assistant{whitespace}...content...<|im_end|>

        Uses cumsum trick for O(n) complexity without Python loops.

        Args:
            input_ids: Token IDs array

        Returns:
            Labels array with IGNORE_INDEX for masked positions
        """
        if not self.mask_prompt:
            return input_ids.copy()

        n = len(input_ids)
        num_ast = self._cached_num_assistant_tokens
        empty_labels = np.full_like(input_ids, self.IGNORE_INDEX)

        if n < 3:
            return empty_labels

        # Find all <|im_start|> positions and filter to valid ones
        im_start_positions = np.where(input_ids == self._cached_im_start_id)[0]
        valid_positions = im_start_positions[im_start_positions + 1 + num_ast <= n]
        if len(valid_positions) == 0:
            return empty_labels

        # Vectorized check for assistant tokens following <|im_start|>
        offsets = np.arange(1, num_ast + 1)
        check_indices = valid_positions[:, None] + offsets
        check_tokens = input_ids[check_indices]
        matches = np.all(check_tokens == self._cached_assistant_token_ids_array, axis=1)
        pattern_starts = valid_positions[matches]
        if len(pattern_starts) == 0:
            return empty_labels

        # Find all <|im_end|> positions
        im_end_positions = np.where(input_ids == self._cached_im_end_id)[0]
        if len(im_end_positions) == 0:
            return empty_labels

        # Content starts after: <|im_start|> + assistant_tokens
        # Note: The \n after "assistant" is INCLUDED in loss (matches HF behavior)
        content_starts = pattern_starts + 1 + num_ast
        valid_mask = content_starts < n
        content_starts = content_starts[valid_mask]
        if len(content_starts) == 0:
            return empty_labels

        # Use searchsorted to find matching <|im_end|> for each content_start
        end_indices = np.searchsorted(im_end_positions, content_starts, side="left")
        valid_ends = end_indices < len(im_end_positions)
        content_starts = content_starts[valid_ends]
        end_indices = end_indices[valid_ends]
        if len(content_starts) == 0:
            return empty_labels

        # End positions include <|im_end|> token
        end_positions = im_end_positions[end_indices] + 1

        # Use diff + cumsum to create interval mask efficiently
        diff = np.zeros(n + 1, dtype=np.int8)
        np.add.at(diff, content_starts, 1)
        np.add.at(diff, end_positions, -1)
        mask = np.cumsum(diff[:-1]) > 0

        return np.where(mask, input_ids, self.IGNORE_INDEX).astype(input_ids.dtype)

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
        for item in batch:
            messages = item.get(self.messages_key, [])
            images_data = item.get(self.images_key, [])

            # Load all images for this example
            all_images.extend(load_image(img) for img in images_data)
            images_per_example.append(len(images_data))

            # Apply chat template to get the text with image placeholders
            all_texts.append(
                self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=self.add_generation_prompt,
                )
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
        img_sizes = processed["image_sizes"].astype(np.int32) if has_image_sizes else None

        # Pre-compute cumulative image indices for fast slicing
        # cumsum gives end indices: [n0, n0+n1, n0+n1+n2, ...]
        cum_images = np.cumsum(images_per_example)

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
            if num_images > 0 and has_pixel_values:
                assert pv is not None  # Guarded by has_pixel_values
                if num_images == 1:
                    # Single image: use anyres with all patches
                    pixel_values = pv[pv_start]
                    if self.max_num_patches is not None:
                        pixel_values, grid_mask = self._pad_pixel_values(
                            pixel_values, valid_patches=pixel_values.shape[0]
                        )
                else:
                    # Multiple images: only use base patch (first patch) from each image
                    # This matches HF behavior where multi-image doesn't use anyres
                    base_patches = [pv[j][0] for j in range(pv_start, pv_end)]
                    pixel_values = np.stack(base_patches, axis=0)  # (num_images, C, H, W)
                    if self.max_num_patches is not None:
                        pixel_values, grid_mask = self._pad_pixel_values(pixel_values, valid_patches=num_images)
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

            # Compute unpad_indices only for single-image anyres case
            # Multi-image doesn't use anyres (each image is just 1 base patch)
            if num_images == 1 and has_image_sizes and self.grid_pinpoints and self.vision_feature_height:
                assert image_sizes is not None  # Guarded by has_image_sizes
                orig_height, orig_width = int(image_sizes[0, 0]), int(image_sizes[0, 1])
                gh, gw = self._compute_grid_shape((orig_height, orig_width))
                patches_height = patches_width = self.vision_feature_height
                features_per_patch = patches_height * patches_width
                unpad_indices = self._compute_unpad_indices_for_image(
                    orig_height=orig_height,
                    orig_width=orig_width,
                    patches_height=patches_height,
                    patches_width=patches_width,
                    scale_height=gh,
                    scale_width=gw,
                    features_per_patch=features_per_patch,
                )

            # Create labels and build result
            result: ImageTextDict = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask_batch[i],
                "image_sizes": image_sizes,
                "labels": self._create_labels(input_ids),
                "grid_mask": grid_mask,
                "unpad_indices": unpad_indices,
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
            exemplar["grid_mask"] = np.zeros((total_patches,), dtype=np.bool_)
            # Include sized unpad_indices when vision_feature_height is also configured
            if self.vision_feature_height is not None:
                features_per_patch = self.vision_feature_height * self.vision_feature_height
                max_features = (self.max_num_patches + 1) * features_per_patch
                exemplar["unpad_indices"] = np.zeros((max_features,), dtype=np.int32)
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

    Expected data format:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ],
        "images": ["path/to/image.jpg"]
    }
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
                ConversationUrlDataSource(split_urls, messages_key=self.messages_key, images_key=self.images_key),
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
            for doc in ConversationUrlDataSource(urls, messages_key=self.messages_key, images_key=self.images_key):
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
    max_length: int = 2048
    padding: bool = True

    @cached_property
    def the_processor(self) -> ProcessorMixin:
        return load_processor(self.processor)

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

    Key design:
    - Loads ALL raw data into memory at startup (raw data is small - just text/metadata)
    - Background thread prefetches data sequentially ahead of consumption
    - Prefetch cache: items are removed when accessed, freeing space for more prefetch
    - Uses per-processor locks for HF tokenizer thread-safety

    Flow:
        Prefetch thread: [process 0-31] -> [process 32-63] -> [process 64-95] -> ...
        Main thread:     [access 0-31, pop from cache] -> [access 32-63, pop] -> ...
    """

    # How many processed examples to cache in memory
    DEFAULT_CACHE_SIZE = 256  # ~256 examples * ~2MB each = ~512MB

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
        grid_pinpoints, patch_size, vision_feature_height, max_num_patches = _extract_anyres_params(processor)

        # Build the batch processor (runs on CPU in background thread)
        self._batch_processor = BatchImageProcessor(
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

        # Use per-processor lock - HuggingFace tokenizer is NOT thread-safe
        # Each processor instance gets its own lock, allowing different processors
        # to run in parallel while ensuring thread-safety for each one
        self._processor_lock = self._get_processor_lock(processor)

        # RAW data stored in memory (small - just text/paths, not images)
        # This avoids slow re-reading of jsonl files
        self._raw_data: Optional[List[Dict[str, Any]]] = None
        self._length: Optional[int] = None
        self._data_lock = threading.Lock()
        self._data_loaded = threading.Event()

        # Prefetch cache for PROCESSED data (large - includes pixel_values)
        # Key: global_idx, Value: ImageTextDict
        # Items are popped when accessed - cache only holds prefetched but not-yet-accessed data
        self._processed_cache: OrderedDict[int, ImageTextDict] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Background sequential prefetch
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()

    def _ensure_data_loaded(self):
        """Load all raw data into memory (synchronous)."""
        if self._raw_data is not None:
            return

        with self._data_lock:
            if self._raw_data is not None:
                return

            logger.info("Loading raw data into memory for streaming...")

            # Pre-allocate list and use list extend for better performance
            raw_data: list[Dict[str, Any]] = []
            for shard_name in self.source.shard_names:
                # Use list extend instead of individual appends
                shard_data = list(self.source.open_shard(shard_name))
                raw_data.extend(shard_data)

            self._raw_data = raw_data
            self._length = len(raw_data)
            self._data_loaded.set()
            logger.info(f"Loaded {self._length} raw examples into memory")

            # Start background prefetch thread
            self._start_prefetch_thread()

    def _start_prefetch_thread(self):
        """Start background thread to prefetch data sequentially."""
        if self._prefetch_thread is not None:
            return

        def prefetch_worker():
            """Background worker that prefetches data sequentially.

            Simple sequential prefetch - processes data from index 0 to end,
            keeping the cache filled ahead of consumption.
            """
            batch_size = 32
            next_idx = 0

            while not self._stop_prefetch.is_set():
                if self._length is None or self._raw_data is None:
                    self._stop_prefetch.wait(0.05)
                    continue

                # Check cache size
                cache_len = len(self._processed_cache)
                if cache_len >= self.cache_size:
                    # Cache is full, wait
                    self._stop_prefetch.wait(0.05)
                    continue

                # Wrap around for epoch support
                if next_idx >= self._length:
                    next_idx = 0

                # Find indices not in cache
                end_idx = min(next_idx + batch_size, self._length)
                with self._cache_lock:
                    indices_to_prefetch = [i for i in range(next_idx, end_idx) if i not in self._processed_cache]

                if not indices_to_prefetch:
                    next_idx = end_idx
                    continue

                # Process batch
                try:
                    raw_items = [self._raw_data[i] for i in indices_to_prefetch]
                    with self._processor_lock:
                        processed = self._batch_processor(raw_items)

                    with self._cache_lock:
                        for idx, item in zip(indices_to_prefetch, processed):
                            self._processed_cache[idx] = item
                        # Evict oldest entries if over limit
                        while len(self._processed_cache) > self.cache_size:
                            self._processed_cache.popitem(last=False)
                except Exception as e:
                    logger.warning(f"Prefetch failed for indices {indices_to_prefetch}: {e}")

                next_idx = end_idx

        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        logger.debug("Started sequential prefetch thread")

    def _process_items(self, indices: Sequence[int]) -> List[ImageTextDict]:
        """Process items - must be called with _processor_lock held."""
        assert self._raw_data is not None, "Data not loaded"
        raw_items = [self._raw_data[i] for i in indices]
        processed = self._batch_processor(raw_items)
        return list(processed)

    def _get_from_cache_or_process(self, indices: Sequence[int]) -> List[ImageTextDict]:
        """Get items from cache or process them.

        Strategy: Remove accessed items from cache immediately to free space for prefetch.
        This ensures the background prefetch thread can always work ahead.
        """
        self._ensure_data_loaded()

        results: List[Optional[ImageTextDict]] = [None] * len(indices)
        indices_to_process: List[Tuple[int, int]] = []  # (result_idx, global_idx)

        # Check cache and pop accessed items (they won't be needed again soon)
        with self._cache_lock:
            for result_idx, global_idx in enumerate(indices):
                if global_idx in self._processed_cache:
                    # Pop from cache - accessed data won't be reused in sequential access
                    results[result_idx] = self._processed_cache.pop(global_idx)
                else:
                    indices_to_process.append((result_idx, global_idx))

        # Process missing items outside of cache lock
        if indices_to_process:
            global_indices = [gidx for _, gidx in indices_to_process]

            # Get raw items without lock (read-only access to _raw_data)
            assert self._raw_data is not None, "Data not loaded"
            raw_items = [self._raw_data[i] for i in global_indices]

            # Only hold processor lock during actual processing
            with self._processor_lock:
                processed = self._batch_processor(raw_items)

            # Store results (no need to cache since we just processed on-demand)
            for (result_idx, _), item in zip(indices_to_process, processed):
                results[result_idx] = item

        return results  # type: ignore

    async def async_len(self) -> int:
        self._ensure_data_loaded()
        assert self._length is not None, "Data not loaded"
        return self._length

    async def final_length_is_known(self) -> bool:
        self._ensure_data_loaded()
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        self._ensure_data_loaded()
        return self._length

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ImageTextDict]:
        """Get a batch of processed items."""
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_from_cache_or_process, indices)

    def __del__(self):
        """Clean up background thread."""
        self._stop_prefetch.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)

    @staticmethod
    def build(
        source: ShardedDataSource[Dict[str, Any]],
        processor: ProcessorMixin,
        max_length: int = 2048,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> "StreamingImageDataset":
        """Build a streaming dataset from a source."""
        return StreamingImageDataset(
            source=source,
            processor=processor,
            max_length=max_length,
            padding=padding,
            messages_key=messages_key,
            images_key=images_key,
            cache_size=cache_size,
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

    @staticmethod
    def init(
        pixel_values: Optional[NamedArray],
        input_ids: NamedArray,
        labels: Optional[NamedArray] = None,
        ignore_id: Optional[int] = None,
        grid_mask: Optional[NamedArray] = None,
    ) -> "ImageTextExample":
        """Initialize an ImageTextExample with optional loss masking.

        Args:
            pixel_values: Image pixel values (FIXED shape, padded), or None for text-only
            input_ids: Token IDs
            labels: Training labels with -100 for tokens to ignore (HF-compatible).
                    If provided, loss_mask is created from labels != -100.
            ignore_id: Alternative way to create loss_mask from input_ids != ignore_id.
                       Only used if labels is None.
            grid_mask: Boolean mask indicating valid patches (TOTAL_PATCHES,)
        """
        if labels is not None:
            # HuggingFace-compatible: use labels to create loss mask
            # labels == -100 means the token should be ignored
            # Use numpy operations to keep data on CPU during data loading
            # Use bool (1 byte) instead of float32 (4 bytes) to save memory
            # Will be converted to float during loss computation
            labels_array = labels.array if hasattr(labels, "array") else labels
            mask_array = (labels_array != -100).astype(np.bool_)
            # Use NamedArray directly to avoid jnp.asarray()
            loss_mask = NamedArray(mask_array, labels.axes)
        elif ignore_id is not None:
            # Legacy behavior: use input_ids to create loss mask
            input_ids_array = input_ids.array if hasattr(input_ids, "array") else input_ids
            mask_array = (input_ids_array != ignore_id).astype(np.bool_)
            loss_mask = NamedArray(mask_array, input_ids.axes)
        else:
            loss_mask = None

        return ImageTextExample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            loss_mask=loss_mask,
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
        ignore_index: Optional[int] = None,
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
            ignore_index: Token ID to ignore in loss computation
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
        self.ignore_id = ignore_index
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

            labels = None
            if "labels" in inputs:
                labels = NamedArray(inputs["labels"], (self.Position,))

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
                labels=labels,
                ignore_id=self.ignore_id,
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

    def __post_init__(self):
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
    ) -> AsyncDataset[ImageTextDict]:
        image_datasets = self.training_sets()

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

        return mixture

    def training_sets(self) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        if self.use_cache:
            return self.build_caches("train")
        else:
            return self.build_streaming_datasets("train")

    def validation_sets(self) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        if self.use_cache:
            return self.build_caches("validation")
        else:
            return self.build_streaming_datasets("validation")

    def build_streaming_datasets(self, split: str) -> Dict[str, StreamingImageDataset]:
        """Build streaming datasets that process images on-the-fly without caching."""
        datasets_dict = {}

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
            )

            datasets_dict[name] = streaming_ds
            # Get dataset size and log it
            try:
                dataset_len = asyncio.run(streaming_ds.async_len())
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
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
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
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

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

        max_num_patches = int(self.vision_aspect_ratio.strip("anyres_max_"))
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


def create_custom_processor(model_name, do_pad=True, image_grid_pinpoints=None, max_image_tiles=None):
    """
    Create a LlavaOnevisionProcessor with custom do_pad setting.

    Args:
        model_name: HuggingFace model name
        do_pad: Whether to pad image patches (True for Levanter, False for HF reference)
        image_grid_pinpoints: Optional custom grid pinpoints. If None, uses DEFAULT_IMAGE_GRID_PINPOINTS.
        max_image_tiles: Maximum number of image tiles (including base) for padding mode.
                         For anyres_max_9, this would be 10 (9 + 1 base).
                         Required when using padding_mode=True when calling the processor.
    """
    from transformers import AutoTokenizer, AutoConfig, AutoImageProcessor, AutoProcessor

    if image_grid_pinpoints is None:
        image_grid_pinpoints = DEFAULT_IMAGE_GRID_PINPOINTS

    # Load config
    config = AutoConfig.from_pretrained(model_name)

    # Load the HF processor to get the chat template
    hf_processor = AutoProcessor.from_pretrained(model_name)
    chat_template = hf_processor.chat_template

    # Load tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load image processor from HF and configure do_pad
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image_processor.do_pad = do_pad
    image_processor.image_grid_pinpoints = image_grid_pinpoints

    # Calculate num_image_tokens (patches per image = (image_size / patch_size)^2)
    image_size = config.vision_config.image_size  # e.g., 384
    patch_size = config.vision_config.patch_size  # e.g., 14
    num_image_tokens = (image_size // patch_size) ** 2  # e.g., 729

    # Create the custom processor with required parameters
    processor = LlavaOnevisionProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        num_image_tokens=num_image_tokens,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        vision_aspect_ratio=config.vision_aspect_ratio,
        chat_template=chat_template,
        max_image_tiles=max_image_tiles,
    )
    return processor
