# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLM (Vision-Language Model) Sequence Packing

Implements sequence packing for VLM training to improve GPU/TPU utilization by
combining multiple short samples into a single training example. Attention masks
prevent samples from attending to each other.

This module follows the patterns from packing.py but handles the additional
complexity of image data (pixel_values, image_segment_ids).

Key components:
- PackedImageTextDict: Data structure for packed VLM samples
- VLMPrepackedDataset: Dataset that loads pre-computed pack assignments
- compute_vlm_sample_lengths: Extract lengths from cache for pack planning
- pack_vlm_samples: Wrapper around pack_documents for dual constraints

IMPORTANT: VLM Packing requires disable_anyres=True in the model config.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TypedDict

import numpy as np

from levanter.data import AsyncDataset
from levanter.data.packing import pack_documents

logger = logging.getLogger(__name__)


class PackedImageTextDict(TypedDict, total=False):
    """
    Data structure for packed VLM samples.

    Contains multiple samples packed together with segment IDs to track
    which tokens/patches belong to which original sample.
    """

    # Core fields (concatenated + padded)
    input_ids: np.ndarray  # (max_length,) - concatenated tokens from all samples
    pixel_values: np.ndarray  # (max_patches, C, H, W) - concatenated patches
    loss_mask: np.ndarray  # (max_length,) - concatenated loss masks

    # Packing-specific fields
    segment_ids: np.ndarray  # (max_length,) - segment ID per token (0,1,2,... or -1 for pad)
    image_segment_ids: np.ndarray  # (max_patches,) - segment ID per patch
    position_ids: np.ndarray  # (max_length,) - position within each segment (resets per segment)
    num_segments: int  # number of samples in this pack

    # Optional fields from original ImageTextDict (carried through for compatibility)
    attention_mask: np.ndarray  # (max_length,) - 1 for valid, 0 for pad


@dataclass
class VLMPackerConfig:
    """Configuration for VLM sequence packing."""

    max_length: int
    """Maximum sequence length (tokens) for packed examples."""

    max_patches: int
    """Maximum number of image patches for packed examples."""

    max_segments: int = 64
    """Maximum number of samples that can be packed together."""

    features_per_patch: int = 576
    """Number of features per image patch (vision_feature_height^2)."""

    pad_token_id: int = 0
    """Token ID to use for padding."""

    image_token_id: int = 151646
    """Token ID for <image> placeholder."""


def compute_vlm_sample_lengths(
    dataset: AsyncDataset,
    features_per_patch: int,
    batch_size: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Compute sample lengths from a dataset for pack planning.

    This extracts token counts and patch counts from each sample in the dataset,
    returning them in a PyTree format suitable for pack_documents().

    The effective token count accounts for image token expansion:
    - Each <image> placeholder is replaced by features_per_patch tokens
    - effective_tokens = token_count + patch_count * (features_per_patch - 1)

    Args:
        dataset: AsyncDataset containing ImageTextDict samples
        features_per_patch: Number of features per image patch
        batch_size: Batch size for reading samples

    Returns:
        Dict with "tokens" and "patches" arrays suitable for pack_documents()
    """
    n_samples = asyncio.get_event_loop().run_until_complete(dataset.async_len())
    logger.info(f"Computing sample lengths for {n_samples} samples...")

    token_lengths = []
    patch_counts = []

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        indices = list(range(batch_start, batch_end))

        # Batch load samples
        samples = asyncio.get_event_loop().run_until_complete(dataset.get_batch(indices))

        for sample in samples:
            # Token count: actual tokens without padding
            if "attention_mask" in sample and sample["attention_mask"] is not None:
                token_count = int(np.sum(sample["attention_mask"]))
            else:
                token_count = len(sample["input_ids"])

            # Patch count: valid patches
            if sample.get("grid_mask") is not None:
                patch_count = int(np.sum(sample["grid_mask"]))
            elif sample.get("pixel_values") is not None:
                pv = sample["pixel_values"]
                if pv is not None and len(pv) > 0:
                    patch_count = len(pv)
                else:
                    patch_count = 0
            else:
                patch_count = 0

            # Effective token count accounts for image token expansion
            # Each <image> placeholder is replaced by features_per_patch tokens
            effective_tokens = token_count + patch_count * (features_per_patch - 1)

            token_lengths.append(effective_tokens)
            patch_counts.append(patch_count)

        if batch_end % 10000 == 0:
            logger.info(f"Processed {batch_end}/{n_samples} samples")

    logger.info(f"Sample length computation complete. Token lengths: mean={np.mean(token_lengths):.1f}, "
                f"max={np.max(token_lengths)}, Patch counts: mean={np.mean(patch_counts):.1f}, "
                f"max={np.max(patch_counts)}")

    return {
        "tokens": np.array(token_lengths, dtype=np.int32),
        "patches": np.array(patch_counts, dtype=np.int32),
    }


def pack_vlm_samples(
    lengths: Dict[str, np.ndarray],
    max_tokens: int,
    max_patches: int,
    max_segments: int = 64,
) -> List[range]:
    """
    Compute pack assignments using dual constraints (tokens AND patches).

    This is a wrapper around pack_documents() that handles the VLM-specific
    dual constraint of both token count and patch count.

    Args:
        lengths: Dict with "tokens" and "patches" arrays
        max_tokens: Maximum tokens per pack
        max_patches: Maximum patches per pack
        max_segments: Maximum samples per pack

    Returns:
        List of ranges, where each range contains sample indices for one pack
    """
    return pack_documents(
        lengths=lengths,
        max_length={"tokens": max_tokens, "patches": max_patches},
        max_segments_per_example=max_segments,
        slice_too_long_examples=True,  # Oversized samples become solo packs
    )


def save_pack_assignments(
    assignments: List[range],
    cache_dir: str,
    config: Optional[VLMPackerConfig] = None,
) -> str:
    """
    Save pack assignments to cache directory.

    Args:
        assignments: List of ranges from pack_vlm_samples()
        cache_dir: Directory to save the assignments
        config: Optional packer config for metadata

    Returns:
        Path to the saved file
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "vlm_pack_assignments.json")

    # Convert ranges to lists for JSON serialization
    data = {
        "assignments": [[r.start, r.stop] for r in assignments],
        "num_packs": len(assignments),
        "version": "1.0",
    }

    if config is not None:
        data["config"] = {
            "max_length": config.max_length,
            "max_patches": config.max_patches,
            "max_segments": config.max_segments,
            "features_per_patch": config.features_per_patch,
        }

    with open(path, "w") as f:
        json.dump(data, f)

    logger.info(f"Saved {len(assignments)} pack assignments to {path}")
    return path


def load_pack_assignments(cache_dir: str) -> List[range]:
    """
    Load pack assignments from cache directory.

    Args:
        cache_dir: Directory containing the assignments file

    Returns:
        List of ranges, where each range contains sample indices for one pack
    """
    path = os.path.join(cache_dir, "vlm_pack_assignments.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pack assignments not found at {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Convert lists back to ranges
    assignments = [range(start, stop) for start, stop in data["assignments"]]

    logger.info(f"Loaded {len(assignments)} pack assignments from {path}")
    return assignments


class VLMPrepackedDataset(AsyncDataset):
    """
    Pre-packed VLM dataset that combines multiple samples into packed examples.

    This class follows the GreedyPrepackedDataset pattern from packing.py but
    handles VLM-specific data (pixel_values, image_segment_ids).

    Key features:
    - Pre-computes pack assignments at initialization (or loads from cache)
    - async_len() returns number of packs (for step calculation)
    - get_batch() loads and assembles packed samples on the fly

    Usage:
        # Create dataset with automatic pack assignment computation
        packed_ds = VLMPrepackedDataset(
            base_dataset=cached_vlm_dataset,
            config=VLMPackerConfig(max_length=2048, max_patches=10),
        )

        # Or load pre-computed assignments
        packed_ds = VLMPrepackedDataset(
            base_dataset=cached_vlm_dataset,
            config=config,
            cache_dir="/path/to/cache",  # Will load vlm_pack_assignments.json
        )
    """

    def __init__(
        self,
        base_dataset: AsyncDataset,
        config: VLMPackerConfig,
        cache_dir: Optional[str] = None,
        precomputed_lengths: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize the pre-packed dataset.

        Args:
            base_dataset: Source dataset containing ImageTextDict samples
            config: Packer configuration
            cache_dir: Optional cache directory for loading/saving pack assignments
            precomputed_lengths: Optional pre-computed sample lengths
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config
        self.cache_dir = cache_dir

        # Try to load existing pack assignments
        if cache_dir is not None:
            try:
                self._pack_indices = load_pack_assignments(cache_dir)
                logger.info(f"Loaded {len(self._pack_indices)} pre-computed pack assignments")
                return
            except FileNotFoundError:
                logger.info("No pre-computed pack assignments found, computing new ones...")

        # Compute pack assignments
        if precomputed_lengths is not None:
            lengths = precomputed_lengths
        else:
            lengths = compute_vlm_sample_lengths(
                base_dataset,
                features_per_patch=config.features_per_patch,
            )

        self._pack_indices = pack_vlm_samples(
            lengths=lengths,
            max_tokens=config.max_length,
            max_patches=config.max_patches,
            max_segments=config.max_segments,
        )

        logger.info(f"Computed {len(self._pack_indices)} packs from "
                    f"{asyncio.get_event_loop().run_until_complete(base_dataset.async_len())} samples")

        # Save pack assignments if cache_dir is provided
        if cache_dir is not None:
            save_pack_assignments(self._pack_indices, cache_dir, config)

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        """Return number of packed samples (for step calculation)."""
        return len(self._pack_indices)

    async def final_length_is_known(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self._pack_indices)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[PackedImageTextDict]:
        """
        Load and assemble packed samples for the given pack indices.

        Args:
            indices: Indices into the pack assignments

        Returns:
            List of PackedImageTextDict, one per requested pack
        """
        results = []

        for idx in indices:
            pack_range = self._pack_indices[idx]
            sample_indices = list(pack_range)

            # Batch load original samples
            samples = await self.base_dataset.get_batch(sample_indices)

            # Assemble into packed format
            packed = self._assemble_pack(list(samples))
            results.append(packed)

        return results

    def _assemble_pack(self, samples: List) -> PackedImageTextDict:
        """
        Assemble multiple samples into a single packed sample.

        This creates segment_ids and image_segment_ids to track which
        tokens/patches belong to which original sample.

        Args:
            samples: List of ImageTextDict samples to pack together

        Returns:
            PackedImageTextDict with all samples packed and segment IDs assigned
        """
        config = self.config

        all_input_ids = []
        all_loss_masks = []
        all_pixel_values = []
        segment_ids = []
        image_segment_ids = []

        for seg_id, sample in enumerate(samples):
            # 1. Token processing
            ids = sample["input_ids"]
            # Remove padding if present (use attention_mask)
            if "attention_mask" in sample and sample["attention_mask"] is not None:
                valid_len = int(np.sum(sample["attention_mask"]))
                ids = ids[:valid_len]

            all_input_ids.append(ids)
            segment_ids.extend([seg_id] * len(ids))

            # Loss mask
            if "loss_mask" in sample and sample["loss_mask"] is not None:
                loss_mask = sample["loss_mask"]
                if len(loss_mask) > len(ids):
                    loss_mask = loss_mask[:len(ids)]
                all_loss_masks.append(loss_mask)
            else:
                # Default: all tokens contribute to loss
                all_loss_masks.append(np.ones(len(ids), dtype=np.float32))

            # 2. Pixel values processing
            pv = sample.get("pixel_values")
            grid_mask = sample.get("grid_mask")

            if pv is not None and len(pv) > 0:
                all_pixel_values.append(pv)

                if grid_mask is not None:
                    # Valid patch -> seg_id, invalid patch -> -1
                    patch_seg_ids = np.where(grid_mask, seg_id, -1)
                else:
                    # All patches valid if no grid_mask
                    patch_seg_ids = np.full(len(pv), seg_id, dtype=np.int32)

                image_segment_ids.extend(patch_seg_ids.tolist())

        # 3. Concatenate and pad
        if all_input_ids:
            input_ids = np.concatenate(all_input_ids)
        else:
            input_ids = np.array([], dtype=np.int32)

        # Pad to max_length
        pad_length = config.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = np.pad(input_ids, (0, pad_length), constant_values=config.pad_token_id)
            segment_ids = segment_ids + [-1] * pad_length
        elif pad_length < 0:
            # Truncate if too long (shouldn't happen with proper pack planning)
            input_ids = input_ids[:config.max_length]
            segment_ids = segment_ids[:config.max_length]
            logger.warning(f"Pack exceeded max_length, truncated to {config.max_length}")

        segment_ids = np.array(segment_ids, dtype=np.int32)

        # Loss mask
        if all_loss_masks:
            loss_mask = np.concatenate(all_loss_masks)
            loss_mask = np.pad(loss_mask, (0, max(0, config.max_length - len(loss_mask))))[:config.max_length]
        else:
            loss_mask = np.zeros(config.max_length, dtype=np.float32)

        # Pixel values
        if all_pixel_values:
            pixel_values = np.concatenate(all_pixel_values, axis=0)

            # Pad to max_patches
            pad_patches = config.max_patches - len(pixel_values)
            if pad_patches > 0:
                pad_shape = (pad_patches,) + pixel_values.shape[1:]
                pixel_values = np.concatenate([pixel_values, np.zeros(pad_shape, dtype=pixel_values.dtype)], axis=0)
                image_segment_ids = image_segment_ids + [-1] * pad_patches
            elif pad_patches < 0:
                pixel_values = pixel_values[:config.max_patches]
                image_segment_ids = image_segment_ids[:config.max_patches]
        else:
            # No images - create placeholder
            pixel_values = np.zeros((config.max_patches, 3, 384, 384), dtype=np.float32)
            image_segment_ids = [-1] * config.max_patches

        image_segment_ids = np.array(image_segment_ids, dtype=np.int32)

        # 4. Compute position_ids (reset per segment)
        position_ids = self._compute_per_segment_positions(segment_ids)

        # 5. Attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (segment_ids >= 0).astype(np.int32)

        return PackedImageTextDict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            image_segment_ids=image_segment_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_segments=len(samples),
        )

    def _compute_per_segment_positions(self, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute position IDs that reset for each segment.

        For packed sequences, each segment should have positions starting from 0.
        Example:
            segment_ids:  [0, 0, 0, 1, 1, -1, -1]
            position_ids: [0, 1, 2, 0, 1,  0,  0]

        Args:
            segment_ids: Array of segment IDs (-1 for padding)

        Returns:
            Array of position IDs that reset per segment
        """
        position_ids = np.zeros_like(segment_ids)
        current_seg = -2  # Invalid starting value
        pos_counter = 0

        for i, seg_id in enumerate(segment_ids):
            if seg_id < 0:
                # Padding - use 0
                position_ids[i] = 0
            elif seg_id != current_seg:
                # New segment - reset counter
                current_seg = seg_id
                pos_counter = 0
                position_ids[i] = pos_counter
                pos_counter += 1
            else:
                # Same segment - increment
                position_ids[i] = pos_counter
                pos_counter += 1

        return position_ids


def build_vlm_prepacked_dataset(
    base_dataset: AsyncDataset,
    max_length: int,
    max_patches: int,
    features_per_patch: int = 576,
    max_segments: int = 64,
    pad_token_id: int = 0,
    cache_dir: Optional[str] = None,
) -> VLMPrepackedDataset:
    """
    Convenience function to create a VLMPrepackedDataset.

    Args:
        base_dataset: Source dataset containing ImageTextDict samples
        max_length: Maximum sequence length for packed examples
        max_patches: Maximum number of image patches for packed examples
        features_per_patch: Number of features per image patch
        max_segments: Maximum samples per pack
        pad_token_id: Token ID for padding
        cache_dir: Optional cache directory for pack assignments

    Returns:
        VLMPrepackedDataset ready for training
    """
    config = VLMPackerConfig(
        max_length=max_length,
        max_patches=max_patches,
        max_segments=max_segments,
        features_per_patch=features_per_patch,
        pad_token_id=pad_token_id,
    )

    return VLMPrepackedDataset(
        base_dataset=base_dataset,
        config=config,
        cache_dir=cache_dir,
    )
