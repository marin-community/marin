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
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

import fsspec
import numpy as np
import pyarrow.parquet as pq

from levanter.data import AsyncDataset
from levanter.data.packing import pack_documents

logger = logging.getLogger(__name__)


def _convert_numpy_to_python(obj):
    """
    Recursively convert numpy arrays and scalars to Python native types.

    This is needed because parquet stores lists as numpy arrays, but Jinja2
    templates (used by HuggingFace chat templates) expect Python lists.

    Args:
        obj: Any object that might contain numpy arrays

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return [_convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(item) for item in obj]
    else:
        return obj


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
    combined_mask: np.ndarray  # (max_length,) int32 - validity mask for position ID computation


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

    Note: The token count from attention_mask already includes expanded image tokens
    (each <image> placeholder was expanded to features_per_patch tokens by the HF processor).

    Args:
        dataset: AsyncDataset containing ImageTextDict samples
        features_per_patch: Number of features per image patch (deprecated, no longer used)
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

            # attention_mask already includes expanded image tokens from HF processor
            # (each <image> placeholder was expanded to patch_count * features_per_patch tokens)
            # So token_count is already the effective token count
            token_lengths.append(token_count)
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

        # 6. Combined mask for position ID computation
        # For packed data, combined_mask is the same as attention_mask (valid tokens vs padding)
        # This is required for the model to use precomputed position_ids
        combined_mask = attention_mask.copy()

        return PackedImageTextDict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            image_segment_ids=image_segment_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            combined_mask=combined_mask,
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


# =============================================================================
# Offline VLM Pack Assignment (Phase 1: Preprocessing)
# =============================================================================


@dataclass
class PackAssignmentConfig:
    """Configuration for pack assignment preprocessing."""

    max_length: int = 2048
    """Maximum sequence length (tokens) for packed examples."""

    max_patches: int = 10
    """Maximum number of image patches for packed examples."""

    max_segments: int = 64
    """Maximum number of samples that can be packed together."""

    features_per_patch: int = 576
    """Number of features per image patch (e.g., 24*24=576 for LLaVA)."""

    image_column: str = "image"
    """Column name for image data in parquet."""

    text_column: str = "text"
    """Column name for text data in parquet."""


@dataclass
class ShardInfo:
    """Information about a parquet shard."""

    path: str
    start_idx: int
    num_samples: int


@dataclass
class PackAssignmentResult:
    """Result of pack assignment computation."""

    assignments: List[Tuple[int, int]]
    """List of (start, end) tuples for each pack."""

    shard_info: List[ShardInfo]
    """Information about each shard."""

    config: PackAssignmentConfig
    """Config used for assignment."""

    num_packs: int
    """Total number of packs."""

    num_samples: int
    """Total number of samples."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "assignments": self.assignments,
            "shard_info": [asdict(s) for s in self.shard_info],
            "config": asdict(self.config),
            "num_packs": self.num_packs,
            "num_samples": self.num_samples,
            "version": "2.0",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackAssignmentResult":
        """Load from dict."""
        return cls(
            assignments=[(s, e) for s, e in data["assignments"]],
            shard_info=[ShardInfo(**s) for s in data["shard_info"]],
            config=PackAssignmentConfig(**data["config"]),
            num_packs=data["num_packs"],
            num_samples=data["num_samples"],
        )


# =============================================================================
# Parallel Processing for Pack Assignments
# =============================================================================


def _process_single_parquet_thread(
    path: str,
    config: PackAssignmentConfig,
    tokenizer,
) -> Tuple[str, List[int], List[int], int]:
    """
    Process a single parquet file - for use with ThreadPoolExecutor.

    This worker function accepts the tokenizer directly (threads share memory).

    Args:
        path: Path to parquet file (supports GCS, S3, local)
        config: Pack assignment configuration
        tokenizer: Pre-loaded tokenizer object

    Returns:
        Tuple of (path, token_lengths, patch_counts, num_samples)
    """
    # Read parquet file (supports GCS, S3, local paths via fsspec)
    fs, fs_path = fsspec.core.url_to_fs(path)
    with fs.open(fs_path, 'rb') as f:
        table = pq.read_table(f)
    df = table.to_pandas()

    token_lengths = []
    patch_counts = []

    for i, row in df.iterrows():
        # Get text
        text = row.get(config.text_column, "")
        if text is None:
            text = ""

        # Tokenize text (without image tokens)
        text_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        text_token_count = len(text_tokens)

        # Count images
        image_data = row.get(config.image_column)
        if image_data is None:
            num_images = 0
        elif isinstance(image_data, list):
            num_images = len(image_data)
        elif isinstance(image_data, (bytes, str)):
            num_images = 1
        else:
            # Assume single image for other types
            num_images = 1

        # For disable_anyres=True, each image = 1 patch
        patch_count = num_images

        # Effective tokens = text tokens + image token expansion
        # Each image expands to features_per_patch tokens
        effective_tokens = text_token_count + num_images * (config.features_per_patch + 1)

        token_lengths.append(effective_tokens)
        patch_counts.append(patch_count)

    return path, token_lengths, patch_counts, len(df)


def compute_sample_lengths_parallel(
    parquet_paths: List[str],
    tokenizer_name: str,
    config: PackAssignmentConfig,
    num_workers: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], List[ShardInfo]]:
    """
    Parallel version of compute_sample_lengths_lightweight.

    Uses multiprocessing to process parquet files in parallel, providing
    significant speedup for large datasets (100M+ samples).

    Args:
        parquet_paths: List of paths to parquet files
        tokenizer_name: HuggingFace model name for tokenizer (e.g., "Qwen/Qwen3-1.7B")
        config: Pack assignment configuration
        num_workers: Number of parallel workers. If None, uses min(cpu_count, num_files, 32)

    Returns:
        Tuple of:
        - Dict with "tokens" and "patches" arrays suitable for pack_documents()
        - List of ShardInfo for tracking shard boundaries
    """
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(parquet_paths), 32)

    logger.info(f"Processing {len(parquet_paths)} parquet files with {num_workers} parallel workers...")

    # Load tokenizer in main thread - threads share memory so no need for per-worker loading
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    logger.info("Tokenizer loaded. Starting parallel processing...")

    # Use ThreadPoolExecutor for I/O-bound tasks
    # - Threads share memory: GCS auth, fsspec state, tokenizer all shared
    # - Python GIL is released during I/O operations (file reads)
    # - No serialization/pickle overhead
    results = []
    total_samples = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_parquet_thread, path, config, tokenizer): path
            for path in parquet_paths
        }

        if use_tqdm:
            # Use tqdm progress bar
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing parquet files",
                unit="file",
            )
            for future in pbar:
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_samples += result[3]
                    pbar.set_postfix({"samples": f"{total_samples:,}"})
                except Exception as e:
                    logger.error(f"Failed: {path} - {e}")
                    raise
            pbar.close()
        else:
            # Fallback to logger-based progress
            completed = 0
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    total_samples += result[3]
                    if completed % 10 == 0 or completed == len(parquet_paths):
                        logger.info(f"  Progress: {completed}/{len(parquet_paths)} files, {total_samples:,} samples")
                except Exception as e:
                    logger.error(f"Failed: {path} - {e}")
                    raise

    # Sort results by original order and merge
    path_to_result = {r[0]: r for r in results}

    all_tokens = []
    all_patches = []
    shard_info = []
    current_idx = 0

    for path in parquet_paths:
        _, tokens, patches, num_samples = path_to_result[path]
        all_tokens.extend(tokens)
        all_patches.extend(patches)
        shard_info.append(ShardInfo(path=path, start_idx=current_idx, num_samples=num_samples))
        current_idx += num_samples

    total_samples = len(all_tokens)
    logger.info(f"Parallel processing complete. Total: {total_samples} samples from {len(parquet_paths)} files")
    logger.info(f"  Token lengths: mean={np.mean(all_tokens):.1f}, max={np.max(all_tokens)}")
    logger.info(f"  Patch counts: mean={np.mean(all_patches):.1f}, max={np.max(all_patches)}")

    return {
        "tokens": np.array(all_tokens, dtype=np.int32),
        "patches": np.array(all_patches, dtype=np.int32),
    }, shard_info


def compute_sample_lengths_lightweight(
    parquet_paths: List[str],
    tokenizer,
    config: PackAssignmentConfig,
    processor=None,
) -> Tuple[Dict[str, np.ndarray], List[ShardInfo]]:
    """
    Compute sample lengths from raw parquet files without full image preprocessing.

    This is a lightweight operation that:
    - Tokenizes text to get text token count
    - Counts images to estimate patch count
    - Does NOT load/decode images

    For each sample, the effective token count is:
        text_tokens + num_images * features_per_patch

    Args:
        parquet_paths: List of paths to parquet files
        tokenizer: HuggingFace tokenizer for text tokenization
        config: Pack assignment configuration
        processor: Optional HuggingFace processor for applying chat template to
                   conversation-format text. Required if text_column contains
                   message dicts instead of plain strings.

    Returns:
        Tuple of:
        - Dict with "tokens" and "patches" arrays suitable for pack_documents()
        - List of ShardInfo for tracking shard boundaries
    """
    token_lengths = []
    patch_counts = []
    shard_info = []
    current_idx = 0

    logger.info(f"Computing sample lengths from {len(parquet_paths)} parquet files...")

    for path in parquet_paths:
        logger.info(f"Processing {path}...")

        # Read parquet file (supports GCS, S3, local paths via fsspec)
        fs, fs_path = fsspec.core.url_to_fs(path)
        with fs.open(fs_path, 'rb') as f:
            table = pq.read_table(f)
        df = table.to_pandas()
        num_samples = len(df)

        shard_info.append(ShardInfo(
            path=path,
            start_idx=current_idx,
            num_samples=num_samples,
        ))

        for i, row in df.iterrows():
            # Get text data (may be plain string or conversation format)
            text_data = row.get(config.text_column, "")
            if text_data is None:
                text_data = ""

            # Check if text_data is conversation format (list of message dicts)
            # If so, apply chat template to convert to string
            if isinstance(text_data, (list, np.ndarray)) and len(text_data) > 0:
                # Conversation format: apply chat template
                if processor is not None:
                    try:
                        # Convert numpy array to list if needed
                        messages = list(text_data) if isinstance(text_data, np.ndarray) else text_data
                        # Convert numpy arrays inside messages to native Python types
                        def convert_message(msg):
                            if isinstance(msg, np.ndarray):
                                return list(msg)
                            if isinstance(msg, dict):
                                return {k: convert_message(v) for k, v in msg.items()}
                            if isinstance(msg, list):
                                return [convert_message(item) for item in msg]
                            return msg
                        messages = [convert_message(m) for m in messages]
                        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
                    except Exception as e:
                        logger.warning(f"Failed to apply chat template: {e}. Using empty string.")
                        text = ""
                else:
                    logger.warning("Conversation format detected but no processor provided. Using empty string.")
                    text = ""
            else:
                text = str(text_data) if text_data else ""

            # Tokenize text (without image tokens)
            # Note: We don't expand <image> placeholders here, just count base tokens
            text_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            text_token_count = len(text_tokens)

            # Count images
            image_data = row.get(config.image_column)
            if image_data is None:
                num_images = 0
            elif isinstance(image_data, list):
                num_images = len(image_data)
            elif isinstance(image_data, (bytes, str)):
                num_images = 1
            else:
                # Assume single image for other types
                num_images = 1

            # For disable_anyres=True, each image = 1 patch
            patch_count = num_images

            # Effective tokens = text tokens + image token expansion
            # Each image expands to features_per_patch tokens
            effective_tokens = text_token_count + num_images * (config.features_per_patch+1)

            token_lengths.append(effective_tokens)
            patch_counts.append(patch_count)

        current_idx += num_samples
        logger.info(f"  Processed {num_samples} samples, total: {current_idx}")

    total_samples = len(token_lengths)
    logger.info(f"Sample length computation complete. Total samples: {total_samples}")
    logger.info(f"  Token lengths: mean={np.mean(token_lengths):.1f}, max={np.max(token_lengths)}")
    logger.info(f"  Patch counts: mean={np.mean(patch_counts):.1f}, max={np.max(patch_counts)}")

    return {
        "tokens": np.array(token_lengths, dtype=np.int32),
        "patches": np.array(patch_counts, dtype=np.int32),
    }, shard_info


def compute_pack_assignments(
    parquet_paths: List[str],
    output_file: str,
    tokenizer,
    config: Optional[PackAssignmentConfig] = None,
    num_workers: int = 1,
    processor=None,
) -> PackAssignmentResult:
    """
    Compute pack assignments from raw parquet files and save to JSON.

    This is the offline preprocessing step that determines which samples
    should be packed together based on their lengths.

    Args:
        parquet_paths: List of paths to raw parquet files
        output_file: Path to save pack_assignments.json
        tokenizer: HuggingFace tokenizer (object) or tokenizer name (str) for text tokenization
        config: Pack assignment configuration (uses defaults if None)
        num_workers: Number of parallel workers for processing parquet files.
                     Default 1 (sequential). Set > 1 for parallel processing.
                     Parallel mode provides significant speedup for large datasets.
        processor: Optional HuggingFace processor for applying chat template to
                   conversation-format text. Required if text_column contains
                   message dicts instead of plain strings.

    Returns:
        PackAssignmentResult with assignments and metadata
    """
    if config is None:
        config = PackAssignmentConfig()

    # 1. Compute sample lengths (parallel or sequential)
    if num_workers > 1:
        # Parallel processing - need tokenizer name (string)
        if isinstance(tokenizer, str):
            tokenizer_name = tokenizer
        else:
            tokenizer_name = tokenizer.name_or_path
        logger.info(f"Using parallel processing with {num_workers} workers")
        lengths, shard_info = compute_sample_lengths_parallel(
            parquet_paths, tokenizer_name, config, num_workers
        )
    else:
        # Sequential processing - can use tokenizer object directly
        logger.info("Using sequential processing (set num_workers > 1 for parallel)")
        lengths, shard_info = compute_sample_lengths_lightweight(
            parquet_paths, tokenizer, config, processor=processor
        )

    num_samples = len(lengths["tokens"])
    logger.info(f"Total samples: {num_samples}")

    # 2. Run greedy packing
    pack_ranges = pack_vlm_samples(
        lengths=lengths,
        max_tokens=config.max_length,
        max_patches=config.max_patches,
        max_segments=config.max_segments,
    )

    # Convert ranges to tuples
    assignments = [(r.start, r.stop) for r in pack_ranges]
    num_packs = len(assignments)

    logger.info(f"Created {num_packs} packs from {num_samples} samples")
    logger.info(f"  Compression ratio: {num_samples / num_packs:.2f}x")

    # 3. Create result
    result = PackAssignmentResult(
        assignments=assignments,
        shard_info=shard_info,
        config=config,
        num_packs=num_packs,
        num_samples=num_samples,
    )

    # 4. Save to file
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Saved pack assignments to {output_file}")

    return result


def load_pack_assignment_result(path: str) -> PackAssignmentResult:
    """Load pack assignments from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return PackAssignmentResult.from_dict(data)


# =============================================================================
# Streaming Training Dataset (Phase 2: Training)
# =============================================================================


@dataclass
class PackedVLMDatasetState:
    """Checkpoint state for PackedVLMDataset."""

    pack_idx: int = 0
    """Current pack index."""

    step: int = 0
    """Training step number."""


class SequentialShardLoader:
    """
    Sequential parquet shard loader with prefetching.

    Since training iterates through packs sequentially, and samples within
    packs are also ordered, we use a simple strategy:
    1. Keep current shard in memory
    2. Prefetch next shard in background when approaching shard boundary
    3. Release old shard when switching

    This is more efficient than LRU for sequential access patterns.
    """

    def __init__(self, shard_info: List[ShardInfo], prefetch_threshold: float = 0.8):
        """
        Args:
            shard_info: List of shard metadata
            prefetch_threshold: Start prefetching when progress exceeds this (0.0-1.0)
        """
        self._shard_info = shard_info
        self._prefetch_threshold = prefetch_threshold

        # Current shard state
        self._current_shard_idx: int = -1
        self._current_df: Optional[Any] = None

        # Prefetch state
        self._next_df: Optional[Any] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_shard_idx: int = -1

    def get_sample(self, global_idx: int) -> Dict[str, Any]:
        """Get a sample by global index, with automatic shard management."""
        shard_idx, local_idx = self._find_shard(global_idx)

        # Switch shard if needed
        if shard_idx != self._current_shard_idx:
            self._switch_to_shard(shard_idx)

        # Maybe start prefetching next shard
        self._maybe_start_prefetch(shard_idx, local_idx)

        raw_dict = self._current_df.iloc[local_idx].to_dict()
        # Convert numpy arrays to Python lists for compatibility with Jinja2 templates
        return _convert_numpy_to_python(raw_dict)

    def _find_shard(self, global_idx: int) -> Tuple[int, int]:
        """Find which shard contains the global index."""
        for shard_idx, shard in enumerate(self._shard_info):
            if shard.start_idx <= global_idx < shard.start_idx + shard.num_samples:
                local_idx = global_idx - shard.start_idx
                return shard_idx, local_idx
        raise IndexError(f"Global index {global_idx} not found in any shard")

    def _switch_to_shard(self, shard_idx: int):
        """Switch to a new shard, using prefetched data if available."""
        with self._prefetch_lock:
            # Check if we have this shard prefetched
            if shard_idx == self._prefetch_shard_idx and self._next_df is not None:
                logger.debug(f"Using prefetched shard {shard_idx}")
                self._current_df = self._next_df
                self._next_df = None
                self._prefetch_shard_idx = -1
            else:
                # Load synchronously
                logger.debug(f"Loading shard {shard_idx} synchronously")
                self._current_df = self._load_shard_sync(shard_idx)

            self._current_shard_idx = shard_idx

    def _load_shard_sync(self, shard_idx: int) -> Any:
        """Load a shard synchronously (supports GCS, S3, local paths via fsspec)."""
        shard = self._shard_info[shard_idx]
        logger.info(f"Loading shard: {shard.path}")
        fs, fs_path = fsspec.core.url_to_fs(shard.path)
        with fs.open(fs_path, 'rb') as f:
            table = pq.read_table(f)
        return table.to_pandas()

    def _maybe_start_prefetch(self, shard_idx: int, local_idx: int):
        """Start prefetching next shard if we're near the end of current."""
        shard = self._shard_info[shard_idx]
        progress = local_idx / shard.num_samples if shard.num_samples > 0 else 1.0

        # Check if we should prefetch
        if progress >= self._prefetch_threshold:
            next_shard_idx = shard_idx + 1
            if next_shard_idx < len(self._shard_info):
                self._start_prefetch(next_shard_idx)

    def _start_prefetch(self, shard_idx: int):
        """Start prefetching a shard in background thread."""
        with self._prefetch_lock:
            # Don't prefetch if already prefetching or already have this shard
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            if self._prefetch_shard_idx == shard_idx:
                return

            logger.debug(f"Starting prefetch for shard {shard_idx}")
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                args=(shard_idx,),
                daemon=True,
            )
            self._prefetch_thread.start()

    def _prefetch_worker(self, shard_idx: int):
        """Background worker to prefetch a shard."""
        try:
            df = self._load_shard_sync(shard_idx)
            with self._prefetch_lock:
                self._next_df = df
                self._prefetch_shard_idx = shard_idx
            logger.debug(f"Prefetch complete for shard {shard_idx}")
        except Exception as e:
            logger.warning(f"Prefetch failed for shard {shard_idx}: {e}")


class PackedVLMDataset(AsyncDataset):
    """
    Streaming VLM dataset that reads raw parquet + pack assignments.

    This class:
    - Reads pack assignments computed offline
    - Loads raw samples from parquet on demand
    - Applies BatchImageProcessor during training
    - Supports checkpoint save/restore

    The images are kept in original format (URL or bytes) in parquet,
    and only processed to pixel_values during training.

    Usage:
        # Load pre-computed pack assignments
        dataset = PackedVLMDataset(
            pack_assignments_file="pack_assignments.json",
            processor=hf_processor,  # For BatchImageProcessor
            state_file="vlm_data_state.json",  # Optional checkpoint
        )

        # Get a pack
        pack = await dataset.get_batch([0])

        # Save checkpoint
        dataset.save_state(step=1000)
    """

    def __init__(
        self,
        pack_assignments_file: str,
        processor,
        max_length: int = 2048,
        prefetch_threshold: float = 0.8,
        state_file: Optional[str] = None,
        # BatchImageProcessor configuration
        tokenizer=None,
        disable_anyres: bool = False,
        grid_pinpoints: Optional[List[List[int]]] = None,
        vision_feature_height: Optional[int] = None,
        max_num_patches: int = 9,
        patch_size: int = 384,
    ):
        """
        Initialize the packed VLM dataset.

        Args:
            pack_assignments_file: Path to pack_assignments.json
            processor: HuggingFace processor (for BatchImageProcessor)
            max_length: Maximum sequence length
            prefetch_threshold: Start prefetching next shard when progress exceeds this
            state_file: Optional path to checkpoint state file
            tokenizer: Custom tokenizer (e.g., Qwen3) to replace processor's tokenizer
            disable_anyres: If True, disable anyres processing
            grid_pinpoints: Grid resolutions for anyres processing
            vision_feature_height: Vision encoder output tokens per spatial dim
            max_num_patches: Maximum number of patches for anyres
            patch_size: Size of each image patch
        """
        super().__init__()

        # Load pack assignments
        self._assignments = load_pack_assignment_result(pack_assignments_file)
        self._pack_ranges = [range(s, e) for s, e in self._assignments.assignments]

        # Sequential shard loader with prefetching (not LRU - we read sequentially)
        self._shard_loader = SequentialShardLoader(
            shard_info=self._assignments.shard_info,
            prefetch_threshold=prefetch_threshold,
        )

        # Image processor with full configuration (delayed import to avoid circular dependencies)
        from levanter.data.image import BatchImageProcessor
        self._batch_processor = BatchImageProcessor(
            processor,
            tokenizer=tokenizer,
            max_length=max_length,
            disable_anyres=disable_anyres,
            grid_pinpoints=grid_pinpoints,
            vision_feature_height=vision_feature_height,
            max_num_patches=max_num_patches,
            patch_size=patch_size,
        )

        # Config from assignments
        self._config = self._assignments.config
        self._max_length = max_length

        # Checkpoint state
        self._state = PackedVLMDatasetState()
        self._state_file = state_file
        if state_file and os.path.exists(state_file):
            self._restore_state()

        logger.info(f"PackedVLMDataset initialized: {self._assignments.num_packs} packs, "
                    f"{self._assignments.num_samples} samples, "
                    f"{len(self._assignments.shard_info)} shards")

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        """Return number of packs."""
        return self._assignments.num_packs

    async def final_length_is_known(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return self._assignments.num_packs

    def _get_sample(self, global_idx: int) -> Dict[str, Any]:
        """Get a single raw sample by global index."""
        return self._shard_loader.get_sample(global_idx)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[PackedImageTextDict]:
        """
        Load and assemble packed samples for the given pack indices.

        This method:
        1. Looks up which samples belong to each pack
        2. Loads raw samples from parquet
        3. Processes images with BatchImageProcessor
        4. Assembles into PackedImageTextDict format

        Args:
            indices: Pack indices to load

        Returns:
            List of PackedImageTextDict, one per requested pack
        """
        results = []

        for pack_idx in indices:
            pack_range = self._pack_ranges[pack_idx]

            # 1. Load raw samples
            raw_samples = [self._get_sample(i) for i in pack_range]

            # 2. Process with BatchImageProcessor
            # BatchImageProcessor.__call__ takes a batch and returns a batch
            processed_samples = list(self._batch_processor(raw_samples))

            # 3. Assemble into packed format
            if processed_samples:
                packed = self._assemble_pack(processed_samples)
            else:
                # Empty pack - create placeholder
                packed = self._create_empty_pack()

            results.append(packed)

            # Update state
            self._state.pack_idx = pack_idx + 1

        return results

    def _assemble_pack(self, samples: List[Dict]) -> PackedImageTextDict:
        """
        Assemble multiple processed samples into a single packed sample.

        Similar to VLMPrepackedDataset._assemble_pack but works with
        freshly processed samples.
        """
        config = self._config

        all_input_ids = []
        all_loss_masks = []
        all_pixel_values = []
        segment_ids = []
        image_segment_ids = []

        for seg_id, sample in enumerate(samples):
            # Token processing
            ids = np.array(sample["input_ids"])
            if "attention_mask" in sample and sample["attention_mask"] is not None:
                valid_len = int(np.sum(sample["attention_mask"]))
                ids = ids[:valid_len]

            all_input_ids.append(ids)
            segment_ids.extend([seg_id] * len(ids))

            # Loss mask
            if "loss_mask" in sample and sample["loss_mask"] is not None:
                loss_mask = np.array(sample["loss_mask"])[:len(ids)]
                all_loss_masks.append(loss_mask)
            else:
                all_loss_masks.append(np.ones(len(ids), dtype=np.float32))

            # Pixel values
            pv = sample.get("pixel_values")
            if pv is not None and len(pv) > 0:
                pv = np.array(pv)
                all_pixel_values.append(pv)
                num_patches = len(pv)
                image_segment_ids.extend([seg_id] * num_patches)

        # Concatenate and pad
        if all_input_ids:
            input_ids = np.concatenate(all_input_ids)
        else:
            input_ids = np.array([], dtype=np.int32)

        # Pad to max_length
        pad_length = self._max_length - len(input_ids)
        if pad_length > 0:
            input_ids = np.pad(input_ids, (0, pad_length), constant_values=0)
            segment_ids = segment_ids + [-1] * pad_length
        elif pad_length < 0:
            input_ids = input_ids[:self._max_length]
            segment_ids = segment_ids[:self._max_length]

        segment_ids = np.array(segment_ids, dtype=np.int32)

        # Loss mask
        if all_loss_masks:
            loss_mask = np.concatenate(all_loss_masks)
            loss_mask = np.pad(loss_mask, (0, max(0, self._max_length - len(loss_mask))))[:self._max_length]
        else:
            loss_mask = np.zeros(self._max_length, dtype=np.float32)

        # Pixel values
        max_patches = config.max_patches
        if all_pixel_values:
            pixel_values = np.concatenate(all_pixel_values, axis=0)
            pad_patches = max_patches - len(pixel_values)
            if pad_patches > 0:
                pad_shape = (pad_patches,) + pixel_values.shape[1:]
                pixel_values = np.concatenate([pixel_values, np.zeros(pad_shape, dtype=pixel_values.dtype)], axis=0)
                image_segment_ids = image_segment_ids + [-1] * pad_patches
            elif pad_patches < 0:
                pixel_values = pixel_values[:max_patches]
                image_segment_ids = image_segment_ids[:max_patches]
        else:
            pixel_values = np.zeros((max_patches, 3, 384, 384), dtype=np.float32)
            image_segment_ids = [-1] * max_patches

        image_segment_ids = np.array(image_segment_ids, dtype=np.int32)

        # Position IDs (reset per segment)
        position_ids = self._compute_per_segment_positions(segment_ids)

        # Attention mask
        attention_mask = (segment_ids >= 0).astype(np.int32)

        # Combined mask for position ID computation
        # For packed data, combined_mask is the same as attention_mask (valid tokens vs padding)
        # This is required for the model to use precomputed position_ids
        combined_mask = attention_mask.copy()

        return PackedImageTextDict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            image_segment_ids=image_segment_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            combined_mask=combined_mask,
            num_segments=len(samples),
        )

    def _compute_per_segment_positions(self, segment_ids: np.ndarray) -> np.ndarray:
        """Compute position IDs that reset for each segment."""
        position_ids = np.zeros_like(segment_ids)
        current_seg = -2
        pos_counter = 0

        for i, seg_id in enumerate(segment_ids):
            if seg_id < 0:
                position_ids[i] = 0
            elif seg_id != current_seg:
                current_seg = seg_id
                pos_counter = 0
                position_ids[i] = pos_counter
                pos_counter += 1
            else:
                position_ids[i] = pos_counter
                pos_counter += 1

        return position_ids

    def _create_empty_pack(self) -> PackedImageTextDict:
        """Create an empty pack (placeholder for failed processing)."""
        max_patches = self._config.max_patches
        return PackedImageTextDict(
            input_ids=np.zeros(self._max_length, dtype=np.int32),
            pixel_values=np.zeros((max_patches, 3, 384, 384), dtype=np.float32),
            loss_mask=np.zeros(self._max_length, dtype=np.float32),
            segment_ids=np.full(self._max_length, -1, dtype=np.int32),
            image_segment_ids=np.full(max_patches, -1, dtype=np.int32),
            position_ids=np.zeros(self._max_length, dtype=np.int32),
            attention_mask=np.zeros(self._max_length, dtype=np.int32),
            num_segments=0,
        )

    def save_state(self, step: int):
        """Save checkpoint state."""
        if self._state_file is None:
            logger.warning("No state_file configured, cannot save state")
            return

        self._state.step = step
        state_dict = asdict(self._state)

        os.makedirs(os.path.dirname(self._state_file) or ".", exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(state_dict, f)

        logger.debug(f"Saved state: pack_idx={self._state.pack_idx}, step={step}")

    def _restore_state(self):
        """Restore checkpoint state from file."""
        try:
            with open(self._state_file, "r") as f:
                state_dict = json.load(f)
            self._state = PackedVLMDatasetState(**state_dict)
            logger.info(f"Restored state: pack_idx={self._state.pack_idx}, step={self._state.step}")
        except Exception as e:
            logger.warning(f"Failed to restore state: {e}")
            self._state = PackedVLMDatasetState()

    @property
    def current_pack_idx(self) -> int:
        """Current pack index for checkpoint."""
        return self._state.pack_idx

    @current_pack_idx.setter
    def current_pack_idx(self, value: int):
        """Set current pack index (for resuming from checkpoint)."""
        self._state.pack_idx = value