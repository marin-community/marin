#!/usr/bin/env python
"""
Preprocess VLM data: tokenize text, compute loss_mask, keep images as-is.

This script performs offline preprocessing to avoid redundant tokenization during training.
The output parquet files contain:
- input_ids: tokenized text with expanded image tokens
- attention_mask: valid tokens mask
- loss_mask: 1.0 for assistant tokens, 0.0 for others
- images: original images (paths/bytes/URLs) - NOT pixel_values
- num_images: number of images in the sample

Usage:
    python scripts/preprocess_vlm_data.py \
        --input-pattern "gs://marin-vlm/raw_data/*.parquet" \
        --output-dir "gs://marin-vlm/preprocessed/" \
        --tokenizer "Qwen/Qwen3-1.7B" \
        --processor "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
        --max-length 2048 \
        --features-per-patch 576 \
        --num-workers 50 \
        --checkpoint-dir "gs://marin-vlm/checkpoints_preprocess"

Requirements:
    pip install fsspec gcsfs  # For GCS support
    pip install s3fs          # For S3 support (optional)
"""
import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_gcs_to_local(gcs_path: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a GCS path to local cache for HuggingFace loading.
    Uses recursive download to properly handle nested directories.

    Args:
        gcs_path: GCS path (gs://bucket/path) or local/HF Hub path
        cache_dir: Optional cache directory (defaults to ~/.cache/vlm_preprocess)

    Returns:
        Local path to the downloaded directory, or original path if not GCS
    """
    import shutil

    if not gcs_path.startswith("gs://"):
        # Not a GCS path, return as-is (local path or HF Hub name)
        return gcs_path

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/vlm_preprocess")

    # Create a cache key from the GCS path
    cache_key = gcs_path.replace("gs://", "").replace("/", "_")
    local_path = os.path.join(cache_dir, cache_key)

    # Check if cache exists AND has contents (not empty from failed previous attempt)
    if os.path.exists(local_path) and os.listdir(local_path):
        logger.info(f"Using cached: {gcs_path} -> {local_path}")
        return local_path

    # Remove empty directory if it exists (from failed previous attempt)
    if os.path.exists(local_path):
        logger.info(f"Removing empty cache directory: {local_path}")
        shutil.rmtree(local_path)

    logger.info(f"Downloading: {gcs_path} -> {local_path}")

    fs, fs_path = fsspec.core.url_to_fs(gcs_path)

    # Use recursive download to handle nested directories
    fs.get(fs_path, local_path, recursive=True)

    logger.info(f"Downloaded to {local_path}")
    return local_path


@dataclass
class PreprocessConfig:
    """Configuration for VLM preprocessing."""
    max_length: int = 2048
    features_per_patch: int = 576  # 24*24 for SigLIP with patch_size=16
    messages_key: str = "messages"
    images_key: str = "images"
    image_pad_token: str = "<|image_pad|>"
    # Vision markers to wrap image tokens (matching BatchImageProcessor)
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"


def list_parquet_files(pattern: str) -> List[str]:
    """
    List parquet files matching the given pattern.

    Supports GCS (gs://), S3 (s3://), and local paths.

    Args:
        pattern: Glob pattern (e.g., "gs://bucket/data/*.parquet")

    Returns:
        Sorted list of full paths
    """
    fs, path_pattern = fsspec.core.url_to_fs(pattern)
    matching_paths = sorted(fs.glob(path_pattern))

    if not matching_paths:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Reconstruct full paths with protocol prefix
    if pattern.startswith("gs://"):
        full_paths = [f"gs://{p}" for p in matching_paths]
    elif pattern.startswith("s3://"):
        full_paths = [f"s3://{p}" for p in matching_paths]
    else:
        full_paths = matching_paths

    logger.info(f"Found {len(full_paths)} parquet files matching {pattern}")
    return full_paths


def _convert_numpy_to_python(obj):
    """
    Recursively convert numpy arrays and scalars to Python native types.

    This is needed because parquet stores lists as numpy arrays, but Jinja2
    templates (used by HuggingFace chat templates) expect Python lists.
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


def serialize_image(image) -> str:
    """
    Serialize an image to a string for parquet storage.

    HuggingFace datasets store images as dicts with 'bytes' and/or 'path' keys.
    This function converts them to strings that can be stored in parquet and
    later deserialized.

    Args:
        image: Can be:
            - str: path or URL (returned as-is)
            - dict: {'bytes': b'...', 'path': '...'} (HuggingFace format)
            - bytes: raw image bytes

    Returns:
        String representation:
            - For paths/URLs: the string itself
            - For bytes: "base64:" prefix + base64-encoded bytes
    """
    import base64

    if isinstance(image, str):
        # Already a string (path or URL)
        return image
    elif isinstance(image, bytes):
        # Raw bytes - base64 encode
        return "base64:" + base64.b64encode(image).decode("utf-8")
    elif isinstance(image, dict):
        # HuggingFace Image format: {'bytes': b'...', 'path': '...'}
        if "path" in image and image["path"]:
            # Prefer path if available
            return image["path"]
        elif "bytes" in image and image["bytes"]:
            # Fall back to bytes
            return "base64:" + base64.b64encode(image["bytes"]).decode("utf-8")
        else:
            logger.warning(f"Image dict has no 'path' or 'bytes': {image.keys()}")
            return ""
    else:
        # Unknown format - try to convert to string
        logger.warning(f"Unknown image type: {type(image)}")
        return str(image)


def create_loss_mask(input_ids: np.ndarray, tokenizer) -> np.ndarray:
    """
    Create loss mask for training by identifying assistant response tokens.

    For causal LM training, we only compute loss on assistant responses.
    Returns a float32 mask where 1.0 indicates tokens that should contribute
    to the loss, and 0.0 indicates tokens that should be ignored.

    The algorithm identifies assistant response spans by looking for:
        <|im_start|>assistant{whitespace}...content...<|im_end|>

    Args:
        input_ids: Token IDs array
        tokenizer: HuggingFace tokenizer

    Returns:
        Loss mask array (float32) with 1.0 for valid positions, 0.0 for masked
    """
    n = len(input_ids)
    if n < 3:
        return np.zeros(n, dtype=np.float32)

    # Get token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    num_ast = len(assistant_ids)
    assistant_token_ids_array = np.array(assistant_ids, dtype=np.int32)

    # Find all <|im_start|> positions and filter to valid ones
    im_start_positions = np.where(input_ids == im_start_id)[0]
    valid_positions = im_start_positions[im_start_positions + 1 + num_ast <= n]
    if len(valid_positions) == 0:
        return np.zeros(n, dtype=np.float32)

    # Vectorized check for assistant tokens following <|im_start|>
    offsets = np.arange(1, num_ast + 1)
    check_indices = valid_positions[:, None] + offsets
    check_tokens = input_ids[check_indices]
    matches = np.all(check_tokens == assistant_token_ids_array, axis=1)
    pattern_starts = valid_positions[matches]
    if len(pattern_starts) == 0:
        return np.zeros(n, dtype=np.float32)

    # Find all <|im_end|> positions
    im_end_positions = np.where(input_ids == im_end_id)[0]
    if len(im_end_positions) == 0:
        return np.zeros(n, dtype=np.float32)

    # For each pattern start, find the corresponding end
    # Use searchsorted to efficiently find the first end after each start
    loss_mask = np.zeros(n, dtype=np.float32)

    # Content starts after <|im_start|>assistant (includes the newline/whitespace after "assistant")
    # This matches HuggingFace and BatchImageProcessor behavior
    content_starts = pattern_starts + 1 + num_ast  # No extra +1, newline IS included

    for content_start in content_starts:
        # Find first <|im_end|> after content_start
        end_idx = np.searchsorted(im_end_positions, content_start)
        if end_idx < len(im_end_positions):
            content_end = im_end_positions[end_idx]
            # Include content tokens AND <|im_end|> token (matches BatchImageProcessor)
            loss_mask[content_start:content_end + 1] = 1.0

    return loss_mask


def preprocess_sample(
    sample: Dict[str, Any],
    processor,
    tokenizer,
    config: PreprocessConfig,
) -> Dict[str, Any]:
    """
    Preprocess a single VLM sample.

    Args:
        sample: Raw sample with messages and images
        processor: HuggingFace processor (for chat template)
        tokenizer: HuggingFace tokenizer
        config: Preprocessing configuration

    Returns:
        Preprocessed sample with input_ids, attention_mask, loss_mask, images, num_images
    """
    messages = sample.get(config.messages_key, [])
    images = sample.get(config.images_key, [])

    # Convert numpy arrays to Python lists for Jinja2 template
    messages = _convert_numpy_to_python(messages)
    images = _convert_numpy_to_python(images)

    num_images = len(images) if images else 0

    # 1. Apply chat template (without tokenizing)
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}")
        # Return empty sample
        return {
            "input_ids": np.zeros(config.max_length, dtype=np.int32),
            "attention_mask": np.zeros(config.max_length, dtype=np.int32),
            "loss_mask": np.zeros(config.max_length, dtype=np.float32),
            "images": images,
            "num_images": num_images,
        }

    # 2. Expand <image> placeholders to image_pad tokens with vision markers
    # Each <image> → <|vision_start|><|image_pad|>*N<|vision_end|>
    # This matches BatchImageProcessor behavior exactly
    expanded_image_placeholder = (
        config.vision_start_token +
        (config.image_pad_token * config.features_per_patch) +
        config.vision_end_token
    )
    text = text.replace("<image>", expanded_image_placeholder)

    # 3. Tokenize
    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    input_ids = encoding["input_ids"][0].astype(np.int32)
    attention_mask = encoding["attention_mask"][0].astype(np.int32)

    # 4. Compute loss_mask
    loss_mask = create_loss_mask(input_ids, tokenizer)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "images": images,  # Keep original
        "num_images": num_images,
    }


@dataclass
class PackingConfig:
    """Configuration for packing."""
    enable_packing: bool = False
    max_patches: int = 10
    max_segments: int = 64
    pad_token_id: int = 0


def process_shard(
    shard_path: str,
    output_path: str,
    tokenizer_name: str,
    processor_name: str,
    config: PreprocessConfig,
    packing_args: Optional[Tuple[bool, int, int, int]] = None,  # (enable, max_patches, max_segments, pad_token_id)
) -> Tuple[str, int, int, int]:
    """
    Process a single parquet shard.

    Args:
        shard_path: Path to input parquet file
        output_path: Path to output parquet file
        tokenizer_name: HuggingFace tokenizer name/path
        processor_name: HuggingFace processor name/path
        config: Preprocessing configuration

    Returns:
        Tuple of (output_path, input_rows, output_rows, num_errors)
    """
    from transformers import AutoProcessor, AutoTokenizer

    # Load tokenizer and processor (in worker process)
    # Support GCS paths by downloading to local cache
    tokenizer_path = download_gcs_to_local(tokenizer_name)
    processor_path = download_gcs_to_local(processor_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True, local_files_only=True)

    # Download input parquet to local temp file using gcloud command (faster than gcsfs)
    if shard_path.startswith("gs://"):
        import subprocess
        logger.info(f"Downloading {shard_path}...")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/dev/shm") as tmp_input:
            local_input_path = tmp_input.name
        subprocess.run(["gcloud", "storage", "cp", shard_path, local_input_path], check=True, capture_output=True)
        logger.info(f"Downloaded to {local_input_path}")
        table = pq.read_table(local_input_path)
        logger.info(f"Read table with {len(table)} rows")
        os.remove(local_input_path)
    elif shard_path.startswith("s3://"):
        fs, fs_path = fsspec.core.url_to_fs(shard_path)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/dev/shm") as tmp_input:
            fs.get(fs_path, tmp_input.name)
            local_input_path = tmp_input.name
        table = pq.read_table(local_input_path)
        os.remove(local_input_path)
    else:
        table = pq.read_table(shard_path)

    # Keep original images column from pyarrow (avoid pandas conversion issues)
    original_images_column = table.column("images")

    df = table.to_pandas()
    input_rows = len(df)

    # Process each row
    processed_rows = []
    num_errors = 0
    logger.info(f"Processing {len(df)} rows from {shard_path}...")

    for idx in range(len(df)):
        if idx % 100 == 0:
            logger.info(f"  Processing row {idx}/{len(df)}...")
        try:
            row = df.iloc[idx].to_dict()
            row = _convert_numpy_to_python(row)
            processed = preprocess_sample(row, processor, tokenizer, config)
            processed_rows.append(processed)
        except Exception as e:
            logger.warning(f"Error processing row {idx} in {shard_path}: {e}")
            num_errors += 1
            # Add placeholder
            processed_rows.append({
                "input_ids": np.zeros(config.max_length, dtype=np.int32),
                "attention_mask": np.zeros(config.max_length, dtype=np.int32),
                "loss_mask": np.zeros(config.max_length, dtype=np.float32),
                "images": [],
                "num_images": 0,
            })

    # Serialize images to strings for parquet storage
    def serialize_images_list(images):
        """Serialize a list of images to strings."""
        if not images:
            return []
        return [serialize_image(img) for img in images]

    # Apply packing if enabled
    if packing_args and packing_args[0]:  # packing_args = (enable, max_patches, max_segments, pad_token_id)
        enable_packing, max_patches, max_segments, pad_token_id = packing_args
        # Compute lengths for packing
        token_lengths = []
        patch_counts = []
        for row in processed_rows:
            token_count = int(np.sum(row["attention_mask"]))
            token_lengths.append(token_count)
            patch_counts.append(row["num_images"])

        # Pack samples within this shard
        packs = greedy_pack(
            token_lengths,
            patch_counts,
            max_tokens=config.max_length,
            max_patches=max_patches,
            max_segments=max_segments,
        )

        # Assemble packed samples
        packed_rows = []
        for pack in packs:
            samples = [processed_rows[i] for i in pack]
            # Serialize images before packing
            for s in samples:
                s["images"] = serialize_images_list(s["images"])
            packed = assemble_pack(samples, config.max_length, max_patches, pad_token_id)
            packed_rows.append(packed)

        # Schema for packed output
        schema = pa.schema([
            ("input_ids", pa.list_(pa.int32())),
            ("attention_mask", pa.list_(pa.int32())),
            ("loss_mask", pa.list_(pa.float32())),
            ("segment_ids", pa.list_(pa.int32())),
            ("position_ids", pa.list_(pa.int32())),
            ("images", pa.list_(pa.string())),
            ("image_segment_ids", pa.list_(pa.int32())),
            ("num_segments", pa.int32()),
            ("num_images", pa.int32()),
        ])

        columns = {
            "input_ids": [row["input_ids"] for row in packed_rows],
            "attention_mask": [row["attention_mask"] for row in packed_rows],
            "loss_mask": [row["loss_mask"] for row in packed_rows],
            "segment_ids": [row["segment_ids"] for row in packed_rows],
            "position_ids": [row["position_ids"] for row in packed_rows],
            "images": [row["images"] for row in packed_rows],
            "image_segment_ids": [row["image_segment_ids"] for row in packed_rows],
            "num_segments": [row["num_segments"] for row in packed_rows],
            "num_images": [row["num_images"] for row in packed_rows],
        }
        output_rows = len(packed_rows)
        logger.info("Creating parquet table...")
        out_table = pa.table(columns, schema=schema)
        logger.info(f"Table created, num_rows={out_table.num_rows}")
    else:
        # No packing - output preprocessed samples directly
        logger.info("Building output columns...")

        logger.info("  Converting input_ids...")
        input_ids_col = pa.array([row["input_ids"].tolist() for row in processed_rows], type=pa.list_(pa.int32()))
        logger.info("  Converting attention_mask...")
        attention_mask_col = pa.array([row["attention_mask"].tolist() for row in processed_rows], type=pa.list_(pa.int32()))
        logger.info("  Converting loss_mask...")
        loss_mask_col = pa.array([row["loss_mask"].tolist() for row in processed_rows], type=pa.list_(pa.float32()))
        logger.info("  Converting num_images...")
        num_images_col = pa.array([row["num_images"] for row in processed_rows], type=pa.int32())

        # Use original pyarrow images column directly (preserves binary format)
        logger.info("  Using original images column...")
        output_rows = len(processed_rows)

        logger.info("Creating parquet table...")
        out_table = pa.table({
            "input_ids": input_ids_col,
            "attention_mask": attention_mask_col,
            "loss_mask": loss_mask_col,
            "images": original_images_column,
            "num_images": num_images_col,
        })
        logger.info(f"Table created, num_rows={out_table.num_rows}")

    logger.info("Writing to temp file...")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/dev/shm") as tmp:
        tmp_path = tmp.name
    logger.info(f"Temp file: {tmp_path}")
    pq.write_table(out_table, tmp_path)
    logger.info("Write complete")

    try:
        # Upload to remote if needed
        if output_path.startswith("gs://"):
            import subprocess
            logger.info(f"Uploading to {output_path}...")
            subprocess.run(["gcloud", "storage", "cp", tmp_path, output_path], check=True, capture_output=True)
            logger.info("Upload complete")
        elif output_path.startswith("s3://"):
            out_fs, out_fs_path = fsspec.core.url_to_fs(output_path)
            out_fs.put(tmp_path, out_fs_path)
        else:
            # Local - ensure directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            import shutil
            shutil.move(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Cleanup
    del df, table, processed_rows, out_table
    gc.collect()

    return output_path, input_rows, output_rows, num_errors


def process_shard_wrapper(args):
    """Wrapper for process_shard to work with ProcessPoolExecutor."""
    return process_shard(*args)


def save_checkpoint(
    checkpoint_dir: str,
    processed_shards: List[str],
    shard_idx: int,
) -> None:
    """Save checkpoint to checkpoint_dir."""
    checkpoint = {
        "processed_shards": processed_shards,
        "last_shard_idx": shard_idx,
        "timestamp": time.time(),
    }

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")

    # Write to temp file first, then move
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(checkpoint, tmp)
        tmp_path = tmp.name

    fs, fs_path = fsspec.core.url_to_fs(checkpoint_path)
    if checkpoint_path.startswith(("gs://", "s3://")):
        fs.put(tmp_path, fs_path)
        os.remove(tmp_path)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        import shutil
        shutil.move(tmp_path, checkpoint_path)

    logger.info(f"Saved checkpoint at shard {shard_idx}")


def load_checkpoint(checkpoint_dir: str) -> Optional[Dict]:
    """Load checkpoint from checkpoint_dir."""
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")

    try:
        fs, fs_path = fsspec.core.url_to_fs(checkpoint_path)
        if not fs.exists(fs_path):
            return None

        with fs.open(fs_path, "r") as f:
            checkpoint = json.load(f)

        logger.info(f"Loaded checkpoint: last_shard_idx={checkpoint['last_shard_idx']}")
        return checkpoint
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


# ============================================================================
# Packing Functions
# ============================================================================


def compute_sample_lengths(parquet_paths: List[str]) -> Tuple[List[int], List[int], int]:
    """
    Compute token and patch lengths for all samples across parquet files.

    Args:
        parquet_paths: List of preprocessed parquet file paths

    Returns:
        Tuple of (token_lengths, patch_counts, total_samples)
    """
    token_lengths = []
    patch_counts = []
    total_samples = 0

    for path in tqdm(parquet_paths, desc="Computing lengths"):
        fs, fs_path = fsspec.core.url_to_fs(path)
        with fs.open(fs_path, "rb") as f:
            table = pq.read_table(f)

        for i in range(len(table)):
            # Token count: sum of attention_mask
            attention_mask = table["attention_mask"][i].as_py()
            token_count = sum(attention_mask)
            token_lengths.append(token_count)

            # Patch count = num_images (with disable_anyres=True, 1 image = 1 patch)
            num_images = table["num_images"][i].as_py()
            patch_counts.append(num_images)
            total_samples += 1

    return token_lengths, patch_counts, total_samples


def greedy_pack(
    token_lengths: List[int],
    patch_counts: List[int],
    max_tokens: int,
    max_patches: int,
    max_segments: int,
) -> List[List[int]]:
    """
    Greedy bin-packing algorithm for VLM samples.

    Packs samples into bins respecting both token and patch constraints.

    Args:
        token_lengths: Token count per sample
        patch_counts: Patch count per sample
        max_tokens: Maximum tokens per pack
        max_patches: Maximum patches per pack
        max_segments: Maximum samples per pack

    Returns:
        List of packs, where each pack is a list of sample indices
    """
    n = len(token_lengths)
    packs = []
    current_pack = []
    current_tokens = 0
    current_patches = 0

    for i in range(n):
        tokens = token_lengths[i]
        patches = patch_counts[i]

        # Check if sample fits in current pack
        can_fit = (
            current_tokens + tokens <= max_tokens
            and current_patches + patches <= max_patches
            and len(current_pack) < max_segments
        )

        if can_fit:
            current_pack.append(i)
            current_tokens += tokens
            current_patches += patches
        else:
            # Start new pack
            if current_pack:
                packs.append(current_pack)
            current_pack = [i]
            current_tokens = tokens
            current_patches = patches

    # Don't forget the last pack
    if current_pack:
        packs.append(current_pack)

    return packs


def load_sample_from_parquets(
    sample_idx: int,
    parquet_paths: List[str],
    sample_counts: List[int],
) -> Dict[str, Any]:
    """
    Load a single sample by global index from multiple parquet files.

    Args:
        sample_idx: Global sample index
        parquet_paths: List of parquet file paths
        sample_counts: Cumulative sample counts per file

    Returns:
        Sample dict
    """
    # Find which file contains this sample
    file_idx = 0
    offset = 0
    for i, count in enumerate(sample_counts):
        if sample_idx < count:
            file_idx = i
            break
        offset = count

    local_idx = sample_idx - offset

    fs, fs_path = fsspec.core.url_to_fs(parquet_paths[file_idx])
    with fs.open(fs_path, "rb") as f:
        table = pq.read_table(f)

    row = {col: table[col][local_idx].as_py() for col in table.column_names}
    return row


def assemble_pack(
    samples: List[Dict[str, Any]],
    max_length: int,
    max_patches: int,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Assemble multiple samples into a single packed sample.

    Args:
        samples: List of preprocessed samples
        max_length: Maximum sequence length
        max_patches: Maximum patches
        pad_token_id: Token ID for padding

    Returns:
        Packed sample dict with segment_ids, position_ids, etc.
    """
    all_input_ids = []
    all_loss_masks = []
    all_images = []
    segment_ids = []
    image_segment_ids = []
    position_ids = []

    for seg_id, sample in enumerate(samples):
        # Get input_ids and remove padding
        input_ids = np.array(sample["input_ids"], dtype=np.int32)
        attention_mask = np.array(sample["attention_mask"], dtype=np.int32)
        valid_len = int(np.sum(attention_mask))
        input_ids = input_ids[:valid_len]

        all_input_ids.append(input_ids)
        segment_ids.extend([seg_id] * valid_len)
        position_ids.extend(list(range(valid_len)))

        # Loss mask
        loss_mask = np.array(sample["loss_mask"], dtype=np.float32)[:valid_len]
        all_loss_masks.append(loss_mask)

        # Images
        images = sample.get("images", [])
        if images:
            all_images.extend(images)
            image_segment_ids.extend([seg_id] * len(images))

    # Concatenate
    if all_input_ids:
        input_ids = np.concatenate(all_input_ids)
    else:
        input_ids = np.array([], dtype=np.int32)

    if all_loss_masks:
        loss_mask = np.concatenate(all_loss_masks)
    else:
        loss_mask = np.array([], dtype=np.float32)

    # Pad to max_length
    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        input_ids = np.pad(input_ids, (0, pad_length), constant_values=pad_token_id)
        loss_mask = np.pad(loss_mask, (0, pad_length), constant_values=0.0)
        segment_ids = segment_ids + [-1] * pad_length
        position_ids = position_ids + [0] * pad_length
    elif pad_length < 0:
        # Truncate
        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]
        segment_ids = segment_ids[:max_length]
        position_ids = position_ids[:max_length]

    # Attention mask (1 for valid, 0 for pad)
    attention_mask = np.array([1 if s >= 0 else 0 for s in segment_ids], dtype=np.int32)

    # Pad image_segment_ids to max_patches
    pad_patches = max_patches - len(image_segment_ids)
    if pad_patches > 0:
        image_segment_ids = image_segment_ids + [-1] * pad_patches
    elif pad_patches < 0:
        image_segment_ids = image_segment_ids[:max_patches]
        all_images = all_images[:max_patches]

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "loss_mask": loss_mask.tolist(),
        "segment_ids": segment_ids,
        "position_ids": position_ids,
        "images": all_images,
        "image_segment_ids": image_segment_ids,
        "num_segments": len(samples),
        "num_images": len([s for s in image_segment_ids if s >= 0]),
    }


def run_packing(
    preprocessed_paths: List[str],
    output_dir: str,
    max_length: int,
    max_patches: int,
    max_segments: int,
    num_workers: int = 1,
) -> Tuple[int, int]:
    """
    Run packing on preprocessed parquet files.

    Args:
        preprocessed_paths: List of preprocessed parquet file paths
        output_dir: Output directory for packed parquet files
        max_length: Maximum sequence length per pack
        max_patches: Maximum patches per pack
        max_segments: Maximum samples per pack
        num_workers: Number of parallel workers

    Returns:
        Tuple of (num_packs, num_samples)
    """
    logger.info("=" * 60)
    logger.info("Starting Packing Phase")
    logger.info("=" * 60)

    # 1. Compute sample lengths
    logger.info("Step 1: Computing sample lengths...")
    token_lengths, patch_counts, total_samples = compute_sample_lengths(preprocessed_paths)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Token lengths: mean={np.mean(token_lengths):.1f}, max={np.max(token_lengths)}")
    logger.info(f"Patch counts: mean={np.mean(patch_counts):.1f}, max={np.max(patch_counts)}")

    # 2. Compute pack assignments
    logger.info("Step 2: Computing pack assignments...")
    packs = greedy_pack(token_lengths, patch_counts, max_length, max_patches, max_segments)
    logger.info(f"Created {len(packs)} packs from {total_samples} samples")
    logger.info(f"Packing efficiency: {total_samples / len(packs):.2f} samples/pack")

    # 3. Compute cumulative sample counts for loading
    sample_counts = []
    cumsum = 0
    for path in preprocessed_paths:
        fs, fs_path = fsspec.core.url_to_fs(path)
        with fs.open(fs_path, "rb") as f:
            table = pq.read_table(f)
        cumsum += len(table)
        sample_counts.append(cumsum)

    # 4. Create packed parquet files
    logger.info("Step 3: Creating packed parquet files...")

    # Schema for packed output
    packed_schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.float32())),
        ("segment_ids", pa.list_(pa.int32())),
        ("position_ids", pa.list_(pa.int32())),
        ("images", pa.list_(pa.string())),
        ("image_segment_ids", pa.list_(pa.int32())),
        ("num_segments", pa.int32()),
        ("num_images", pa.int32()),
    ])

    # Cache for loaded parquet files
    parquet_cache = {}

    def load_sample(sample_idx: int) -> Dict[str, Any]:
        """Load sample with caching."""
        # Find which file contains this sample
        file_idx = 0
        offset = 0
        for i, count in enumerate(sample_counts):
            if sample_idx < count:
                file_idx = i
                break
            offset = count

        local_idx = sample_idx - offset
        path = preprocessed_paths[file_idx]

        if path not in parquet_cache:
            fs, fs_path = fsspec.core.url_to_fs(path)
            with fs.open(fs_path, "rb") as f:
                parquet_cache[path] = pq.read_table(f)

        table = parquet_cache[path]
        row = {col: table[col][local_idx].as_py() for col in table.column_names}
        return row

    # Process packs in batches and write to parquet files
    packs_per_file = 5000
    packed_rows = []
    file_idx = 0

    for pack_idx, pack in tqdm(enumerate(packs), total=len(packs), desc="Creating packs"):
        # Load samples for this pack
        samples = [load_sample(idx) for idx in pack]

        # Assemble packed sample
        packed = assemble_pack(samples, max_length, max_patches)
        packed_rows.append(packed)

        # Write to file periodically
        if len(packed_rows) >= packs_per_file:
            output_path = os.path.join(output_dir, f"packed_{file_idx:05d}.parquet")
            _write_packed_parquet(packed_rows, output_path, packed_schema)
            logger.info(f"Wrote {len(packed_rows)} packs to {output_path}")
            packed_rows = []
            file_idx += 1
            parquet_cache.clear()  # Clear cache to save memory
            gc.collect()

    # Write remaining packs
    if packed_rows:
        output_path = os.path.join(output_dir, f"packed_{file_idx:05d}.parquet")
        _write_packed_parquet(packed_rows, output_path, packed_schema)
        logger.info(f"Wrote {len(packed_rows)} packs to {output_path}")

    logger.info(f"Packing complete: {len(packs)} packs written to {output_dir}")
    return len(packs), total_samples


def _write_packed_parquet(rows: List[Dict], output_path: str, schema: pa.Schema) -> None:
    """Write packed rows to parquet file."""
    columns = {
        "input_ids": [row["input_ids"] for row in rows],
        "attention_mask": [row["attention_mask"] for row in rows],
        "loss_mask": [row["loss_mask"] for row in rows],
        "segment_ids": [row["segment_ids"] for row in rows],
        "position_ids": [row["position_ids"] for row in rows],
        "images": [row["images"] for row in rows],
        "image_segment_ids": [row["image_segment_ids"] for row in rows],
        "num_segments": [row["num_segments"] for row in rows],
        "num_images": [row["num_images"] for row in rows],
    }

    out_table = pa.table(columns, schema=schema)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        pq.write_table(out_table, tmp.name)
        tmp_path = tmp.name

    try:
        fs, fs_path = fsspec.core.url_to_fs(output_path)
        if output_path.startswith(("gs://", "s3://")):
            fs.put(tmp_path, fs_path)
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            import shutil
            shutil.move(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VLM data: tokenize text, compute loss_mask, keep images as-is",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-pattern",
        required=True,
        help="Glob pattern for input parquet files (e.g., 'gs://bucket/data/*.parquet')",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for preprocessed parquet files (can be GCS, S3, or local)",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name/path (e.g., 'Qwen/Qwen3-1.7B')",
    )
    parser.add_argument(
        "--processor",
        required=True,
        help="HuggingFace processor name/path (e.g., 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf')",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--features-per-patch",
        type=int,
        default=576,
        help="Number of features per image patch (default: 576, i.e., 24*24)",
    )
    parser.add_argument(
        "--messages-key",
        default="messages",
        help="Column name for messages in parquet (default: 'messages')",
    )
    parser.add_argument(
        "--images-key",
        default="images",
        help="Column name for images in parquet (default: 'images')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving/loading checkpoints (supports GCS/S3)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N shards (default: 10)",
    )
    # Packing arguments
    parser.add_argument(
        "--enable-packing",
        action="store_true",
        help="Enable sequence packing after preprocessing",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=10,
        help="Maximum number of image patches per pack (default: 10). With disable_anyres=True, this equals max images.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=64,
        help="Maximum number of samples per pack (default: 64)",
    )

    args = parser.parse_args()

    # Create config
    config = PreprocessConfig(
        max_length=args.max_length,
        features_per_patch=args.features_per_patch,
        messages_key=args.messages_key,
        images_key=args.images_key,
    )

    # Create packing args tuple if enabled (for multiprocessing compatibility)
    packing_args = None
    if args.enable_packing:
        packing_args = (True, args.max_patches, args.max_segments, 0)  # (enable, max_patches, max_segments, pad_token_id)
        logger.info(f"Packing enabled: max_patches={args.max_patches}, max_segments={args.max_segments}")

    # Pre-download GCS paths in main process to avoid race condition
    # when multiple workers try to download simultaneously
    logger.info("Pre-downloading tokenizer and processor if needed...")
    tokenizer_path = download_gcs_to_local(args.tokenizer)
    processor_path = download_gcs_to_local(args.processor)
    logger.info(f"Tokenizer path: {tokenizer_path}")
    logger.info(f"Processor path: {processor_path}")

    # List input files
    logger.info(f"Listing parquet files matching: {args.input_pattern}")
    input_paths = list_parquet_files(args.input_pattern)
    logger.info(f"Found {len(input_paths)} parquet files")
    for i, path in enumerate(input_paths[:5]):
        logger.info(f"  [{i}] {path}")
    if len(input_paths) > 5:
        logger.info(f"  ... and {len(input_paths) - 5} more")

    # Load checkpoint if exists
    start_idx = 0
    processed_shards = []
    if args.checkpoint_dir:
        checkpoint = load_checkpoint(args.checkpoint_dir)
        if checkpoint:
            start_idx = checkpoint["last_shard_idx"] + 1
            processed_shards = checkpoint["processed_shards"]
            logger.info(f"Resuming from shard {start_idx}")

    # Process shards
    total_input_rows = 0
    total_output_rows = 0
    total_errors = 0
    failed_shards = 0

    if args.num_workers > 1:
        # Parallel processing
        logger.info(f"Processing with {args.num_workers} workers...")

        # Prepare arguments for each shard
        tasks = []
        for i, input_path in enumerate(input_paths[start_idx:], start=start_idx):
            # Generate output path
            input_name = os.path.basename(input_path)
            output_path = os.path.join(args.output_dir, input_name)
            tasks.append((
                input_path,
                output_path,
                tokenizer_path,  # Use pre-downloaded local path
                processor_path,  # Use pre-downloaded local path
                config,
                packing_args,
            ))

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_shard_wrapper, task): i
                      for i, task in enumerate(tasks, start=start_idx)}

            pbar = tqdm(as_completed(futures), total=len(tasks), desc="Processing shards")
            for future in pbar:
                shard_idx = futures[future]
                try:
                    output_path, input_rows, output_rows, num_errors = future.result()
                    total_input_rows += input_rows
                    total_output_rows += output_rows
                    total_errors += num_errors
                    processed_shards.append(output_path)
                    logger.info(f"[{shard_idx + 1}/{len(input_paths)}] {os.path.basename(input_paths[shard_idx])}: "
                               f"input={input_rows} rows → output={output_rows} rows "
                               f"({num_errors} errors)")

                    # Save checkpoint
                    if args.checkpoint_dir and (shard_idx + 1) % args.checkpoint_interval == 0:
                        save_checkpoint(args.checkpoint_dir, processed_shards, shard_idx)

                except Exception as e:
                    logger.error(f"Failed to process shard {shard_idx}: {e}")
                    failed_shards += 1
    else:
        # Sequential processing
        logger.info("Processing sequentially...")

        for i, input_path in tqdm(enumerate(input_paths[start_idx:], start=start_idx),
                                   total=len(input_paths) - start_idx,
                                   desc="Processing shards"):
            input_name = os.path.basename(input_path)
            output_path = os.path.join(args.output_dir, input_name)

            try:
                output_path, input_rows, output_rows, num_errors = process_shard(
                    input_path,
                    output_path,
                    tokenizer_path,  # Use pre-downloaded local path
                    processor_path,  # Use pre-downloaded local path
                    config,
                    packing_args,
                )
                total_input_rows += input_rows
                total_output_rows += output_rows
                total_errors += num_errors
                processed_shards.append(output_path)
                logger.info(f"[{i + 1}/{len(input_paths)}] {os.path.basename(input_path)}: "
                           f"input={input_rows} rows → output={output_rows} rows "
                           f"({num_errors} errors)")

                # Save checkpoint
                if args.checkpoint_dir and (i + 1) % args.checkpoint_interval == 0:
                    save_checkpoint(args.checkpoint_dir, processed_shards, i)

            except Exception as e:
                logger.error(f"Failed to process shard {i}: {e}")
                failed_shards += 1

    # Final checkpoint - only save if we have successful shards
    if args.checkpoint_dir and processed_shards:
        save_checkpoint(args.checkpoint_dir, processed_shards, len(input_paths) - 1)
    elif args.checkpoint_dir and not processed_shards:
        logger.warning("No shards processed successfully, not saving checkpoint")

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Input pattern:      {args.input_pattern}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Number of shards:   {len(input_paths)}")
    print(f"Total input rows:   {total_input_rows}")
    print(f"Total output rows:  {total_output_rows}")
    print(f"Total row errors:   {total_errors}")
    print(f"Failed shards:      {failed_shards}")
    print(f"Parallel workers:   {args.num_workers}")
    if args.checkpoint_dir:
        print(f"Checkpoint dir:     {args.checkpoint_dir}")
    if args.enable_packing:
        print(f"Packing enabled:    True")
        print(f"Max patches:        {args.max_patches}")
        print(f"Max segments:       {args.max_segments}")
        if total_input_rows > 0:
            print(f"Packing efficiency: {total_input_rows / total_output_rows:.2f} samples/pack")
    print("=" * 60)


if __name__ == "__main__":
    main()
