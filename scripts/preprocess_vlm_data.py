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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_gcs_to_local(gcs_path: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a GCS path to local cache for HuggingFace loading.

    Args:
        gcs_path: GCS path (gs://bucket/path) or local/HF Hub path
        cache_dir: Optional cache directory (defaults to ~/.cache/vlm_preprocess)

    Returns:
        Local path to the downloaded directory, or original path if not GCS
    """
    if not gcs_path.startswith("gs://"):
        # Not a GCS path, return as-is (local path or HF Hub name)
        return gcs_path

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/vlm_preprocess")

    # Create a cache key from the GCS path
    cache_key = gcs_path.replace("gs://", "").replace("/", "_")
    local_path = os.path.join(cache_dir, cache_key)

    if os.path.exists(local_path):
        logger.info(f"Using cached: {gcs_path} -> {local_path}")
        return local_path

    logger.info(f"Downloading: {gcs_path} -> {local_path}")
    os.makedirs(local_path, exist_ok=True)

    fs, fs_path = fsspec.core.url_to_fs(gcs_path)

    # Download all files in the directory
    files = fs.ls(fs_path)
    for file_path in files:
        if fs.isfile(file_path):
            file_name = os.path.basename(file_path)
            local_file = os.path.join(local_path, file_name)
            fs.get(file_path, local_file)

    logger.info(f"Downloaded {len(files)} files to {local_path}")
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


def process_shard(
    shard_path: str,
    output_path: str,
    tokenizer_name: str,
    processor_name: str,
    config: PreprocessConfig,
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    # Read input parquet
    fs, fs_path = fsspec.core.url_to_fs(shard_path)
    with fs.open(fs_path, "rb") as f:
        table = pq.read_table(f)
    df = table.to_pandas()
    input_rows = len(df)

    # Process each row
    processed_rows = []
    num_errors = 0

    for idx in range(len(df)):
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

    # Convert to pyarrow table
    # Create schema with explicit types
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.float32())),
        ("images", pa.list_(pa.string())),  # Store as list of strings (paths/URLs)
        ("num_images", pa.int32()),
    ])

    # Convert processed rows to columnar format
    columns = {
        "input_ids": [row["input_ids"].tolist() for row in processed_rows],
        "attention_mask": [row["attention_mask"].tolist() for row in processed_rows],
        "loss_mask": [row["loss_mask"].tolist() for row in processed_rows],
        "images": [row["images"] if isinstance(row["images"], list) else [] for row in processed_rows],
        "num_images": [row["num_images"] for row in processed_rows],
    }

    # Write output parquet
    out_table = pa.table(columns, schema=schema)
    out_fs, out_fs_path = fsspec.core.url_to_fs(output_path)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        pq.write_table(out_table, tmp.name)
        tmp_path = tmp.name

    try:
        # Upload to remote if needed
        if output_path.startswith(("gs://", "s3://")):
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
    output_rows = len(columns["input_ids"])
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

    args = parser.parse_args()

    # Create config
    config = PreprocessConfig(
        max_length=args.max_length,
        features_per_patch=args.features_per_patch,
        messages_key=args.messages_key,
        images_key=args.images_key,
    )

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
                args.tokenizer,
                args.processor,
                config,
            ))

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_shard_wrapper, task): i
                      for i, task in enumerate(tasks, start=start_idx)}

            for future in as_completed(futures):
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
    else:
        # Sequential processing
        logger.info("Processing sequentially...")

        for i, input_path in enumerate(input_paths[start_idx:], start=start_idx):
            input_name = os.path.basename(input_path)
            output_path = os.path.join(args.output_dir, input_name)

            try:
                output_path, input_rows, output_rows, num_errors = process_shard(
                    input_path,
                    output_path,
                    args.tokenizer,
                    args.processor,
                    config,
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

    # Final checkpoint
    if args.checkpoint_dir:
        save_checkpoint(args.checkpoint_dir, processed_shards, len(input_paths) - 1)

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Input pattern:      {args.input_pattern}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Number of shards:   {len(input_paths)}")
    print(f"Total input rows:   {total_input_rows}")
    print(f"Total output rows:  {total_output_rows}")
    print(f"Total errors:       {total_errors}")
    print(f"Parallel workers:   {args.num_workers}")
    if args.checkpoint_dir:
        print(f"Checkpoint dir:     {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
