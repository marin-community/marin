#!/usr/bin/env python
"""
Compute VLM pack assignments from parquet files (supports GCS, S3, and local paths).

This script performs offline preprocessing to determine which samples should be packed
together for VLM training. The resulting pack_assignments.json can be used with
PackedVLMDataset for efficient streaming training.

Usage:
    python scripts/compute_vlm_pack_assignments.py \
        --input-pattern "gs://marin-vlm/stage2_sharded/*.parquet" \
        --output "gs://marin-vlm/stage2_sharded/pack_assignments.json" \
        --model "Qwen/Qwen3-1.7B" \
        --max-length 2048 \
        --max-patches 10 \
        --num-workers 50 \
        --checkpoint-dir "gs://marin-vlm/checkpoints_packing"

Requirements:
    pip install fsspec gcsfs  # For GCS support
    pip install s3fs          # For S3 support (optional)
"""
import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import fsspec

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from lib.levanter.src.levanter.data.vlm_packing import (
    PackAssignmentConfig,
    compute_pack_assignments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_parquet_files(pattern: str) -> list[str]:
    """
    List parquet files matching the given pattern.

    Supports GCS (gs://), S3 (s3://), and local paths.

    Args:
        pattern: Glob pattern (e.g., "gs://bucket/data/*.parquet")

    Returns:
        Sorted list of full paths
    """
    # Parse the pattern to get filesystem and path
    fs, path_pattern = fsspec.core.url_to_fs(pattern)

    # Glob for matching files
    matching_paths = sorted(fs.glob(path_pattern))

    if not matching_paths:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Reconstruct full paths with protocol prefix
    if pattern.startswith("gs://"):
        full_paths = [f"gs://{p}" for p in matching_paths]
    elif pattern.startswith("s3://"):
        full_paths = [f"s3://{p}" for p in matching_paths]
    else:
        # Local paths
        full_paths = matching_paths

    logger.info(f"Found {len(full_paths)} parquet files matching {pattern}")
    return full_paths


def upload_to_remote(local_path: str, remote_path: str) -> None:
    """
    Upload a local file to a remote location (GCS, S3, etc.).

    Args:
        local_path: Path to local file
        remote_path: Remote destination path
    """
    fs, fs_path = fsspec.core.url_to_fs(remote_path)

    with open(local_path, "rb") as src:
        with fs.open(fs_path, "wb") as dst:
            dst.write(src.read())

    logger.info(f"Uploaded to {remote_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute VLM pack assignments from parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-pattern",
        required=True,
        help="Glob pattern for input parquet files (e.g., 'gs://bucket/data/*.parquet')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for pack_assignments.json (can be GCS, S3, or local)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model name for tokenizer (default: Qwen/Qwen2-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length for packed examples (default: 2048)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=10,
        help="Maximum number of image patches per packed example (default: 10)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=64,
        help="Maximum number of samples per pack (default: 64)",
    )
    parser.add_argument(
        "--features-per-patch",
        type=int,
        default=576,
        help="Number of features per image patch (default: 576, i.e., 24*24)",
    )
    parser.add_argument(
        "--image-column",
        default="images",
        help="Column name for image data in parquet (default: 'images')",
    )
    parser.add_argument(
        "--text-column",
        default="messages",
        help="Column name for text data in parquet (default: 'messages')",
    )
    parser.add_argument(
        "--processor",
        default=None,
        help="HuggingFace processor name for conversation format data (e.g., 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'). "
             "Required if text_column contains messages (conversation format) instead of plain strings.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers for processing parquet files (default: 1, sequential). "
             "Set > 1 for parallel processing, e.g., --num-workers 8 for 8x speedup on large datasets.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving/loading checkpoints (supports GCS/S3). Enables resume on interrupt. "
             "Example: gs://my-bucket/checkpoints or /tmp/checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N shards (default: 10). Lower = more frequent saves.",
    )

    args = parser.parse_args()

    # 1. List parquet files
    logger.info(f"Listing parquet files matching: {args.input_pattern}")
    parquet_paths = list_parquet_files(args.input_pattern)
    logger.info(f"Found {len(parquet_paths)} parquet files")
    for i, path in enumerate(parquet_paths[:5]):
        logger.info(f"  [{i}] {path}")
    if len(parquet_paths) > 5:
        logger.info(f"  ... and {len(parquet_paths) - 5} more")

    # 2. Load tokenizer
    logger.info(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 2b. Load processor (if specified, for conversation format data)
    processor = None
    if args.processor:
        from transformers import AutoProcessor
        logger.info(f"Loading processor from: {args.processor}")
        processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)

    # 3. Create config
    config = PackAssignmentConfig(
        max_length=args.max_length,
        max_patches=args.max_patches,
        max_segments=args.max_segments,
        features_per_patch=args.features_per_patch,
        image_column=args.image_column,
        text_column=args.text_column,
    )
    logger.info(f"Config: max_length={config.max_length}, max_patches={config.max_patches}, "
                f"max_segments={config.max_segments}, features_per_patch={config.features_per_patch}")

    # 4. Compute pack assignments (save to temp file first)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        local_output = tmp.name

    try:
        logger.info(f"Computing pack assignments (num_workers={args.num_workers})...")
        if args.checkpoint_dir:
            logger.info(f"Checkpointing enabled: {args.checkpoint_dir} (every {args.checkpoint_interval} shards)")
        result = compute_pack_assignments(
            parquet_paths=parquet_paths,
            output_file=local_output,
            tokenizer=args.model if args.num_workers > 1 else tokenizer,  # Pass name for parallel, object for sequential
            config=config,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            processor=processor,  # For conversation format data
        )

        # 5. Upload to remote if needed
        if args.output.startswith(("gs://", "s3://")):
            logger.info(f"Uploading to remote: {args.output}")
            upload_to_remote(local_output, args.output)
        else:
            # Local output - copy or rename
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            import shutil
            shutil.copy2(local_output, args.output)
            logger.info(f"Saved to: {args.output}")

        # 6. Print summary
        print("\n" + "=" * 60)
        print("Pack Assignment Summary")
        print("=" * 60)
        print(f"Input pattern:     {args.input_pattern}")
        print(f"Output file:       {args.output}")
        print(f"Parallel workers:  {args.num_workers}")
        if args.checkpoint_dir:
            print(f"Checkpoint dir:    {args.checkpoint_dir}")
        print(f"Number of shards:  {len(result.shard_info)}")
        print(f"Total samples:     {result.num_samples}")
        print(f"Total packs:       {result.num_packs}")
        print(f"Compression ratio: {result.num_samples / result.num_packs:.2f}x")
        print("=" * 60)

    finally:
        # Clean up temp file
        if os.path.exists(local_output):
            os.remove(local_output)


if __name__ == "__main__":
    main()
