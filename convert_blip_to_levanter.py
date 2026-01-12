#!/usr/bin/env python3
"""
Convert BLIP LAION CC SBU JSON dataset to Levanter image conversation format (parquet).

Input: JSON file with conversations format, images are under images/ folder
Output: parquet file with conversation format for image captioning

Example:
    # Basic conversion
    uv run python convert_blip_to_levanter.py 
        /home/ruili/dataset/blip_laion_cc_sbu_558k.json 
        /home/ruili/dataset/output.parquet
    
    # With custom images directory
    uv run python convert_blip_to_levanter.py \\
        /home/ruili/dataset/blip_laion_cc_sbu_558k.json \\
        /home/ruili/dataset/output.parquet \\
        --images-dir /home/ruili/dataset/images
    
    # Process only first 1000 rows
    uv run python convert_blip_to_levanter.py \\
        /home/ruili/dataset/blip_laion_cc_sbu_558k.json \\
        /home/ruili/dataset/output.parquet \\
        --max-rows 1000
    
    # Shuffle data before processing
    uv run python convert_blip_to_levanter.py /home/ruili/dataset/blip_laion_cc_sbu_558k.json /home/ruili/dataset/output.parquet --shuffle
    
    # Shuffle with fixed seed for reproducibility
    uv run python convert_blip_to_levanter.py \\
        /home/ruili/dataset/blip_laion_cc_sbu_558k.json \\
        /home/ruili/dataset/output.parquet \\
        --shuffle --seed 42

    # Output sharded parquet files (5000 rows per shard)
    uv run python convert_blip_to_levanter.py /home/ruili/dataset/blip_laion_cc_sbu_558k.json /home/ruili/dataset/output_shards/ --rows-per-shard 5000  --shuffle
    # Output: output_shards/train-00000.parquet, train-00001.parquet, ...
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up temporary directory before importing datasets
if 'TMPDIR' not in os.environ:
    for temp_dir in ['/tmp', '/var/tmp', '/usr/tmp', str(Path.home() / '.tmp')]:
        if os.path.exists(temp_dir) and os.access(temp_dir, os.W_OK):
            os.environ['TMPDIR'] = temp_dir
            break
    else:
        temp_dir = Path.cwd() / '.tmp'
        temp_dir.mkdir(exist_ok=True)
        os.environ['TMPDIR'] = str(temp_dir)

from datasets import Dataset


def read_image_bytes(image_path: str) -> bytes:
    """Read image file and return bytes."""
    with open(image_path, 'rb') as f:
        return f.read()


def parse_conversation_to_levanter(
    conversations: List[Dict[str, str]],
    image_path: str,
    images_base_dir: str
) -> Dict[str, Any]:
    """
    Convert BLIP conversation format to Levanter format.
    
    Args:
        conversations: List of conversation dicts with 'from' and 'value' keys
        image_path: Relative image path from JSON (e.g., "00453/004539375.jpg")
        images_base_dir: Base directory for images (e.g., "/home/ruili/dataset/images")
        
    Returns:
        Dictionary in Levanter conversation format with embedded image bytes
    """
    # Build full image path and read bytes
    full_image_path = os.path.join(images_base_dir, image_path)
    image_bytes = read_image_bytes(full_image_path)
    
    # Parse conversations
    messages = []
    
    for conv in conversations:
        role = conv.get('from', '').lower()
        value = conv.get('value', '')
        
        if role == 'human':
            # Split by <image> tag - handle multiple occurrences
            parts = value.split('<image>')
            user_content = []
            
            # Process each part
            for i, part in enumerate(parts):
                # Add text part if non-empty
                if part.strip():
                    user_content.append({"type": "text", "text": part.strip()})
                
                # Add image after each part except the last one
                if i < len(parts) - 1:
                    user_content.append({"type": "image", "text": None})
            
            # If no <image> tag found, add image at the end
            if '<image>' not in value:
                # Add text if exists
                if value.strip():
                    user_content.append({"type": "text", "text": value.strip()})
                # Add image
                user_content.append({"type": "image", "text": None})
            
            # Add user message
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content
                })
                
        elif role == 'gpt' or role == 'assistant':
            # Assistant response
            if value.strip():
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": value.strip()}]
                })
    
    # Ensure we have at least user and assistant messages
    if not messages:
        raise ValueError("No valid messages found in conversation")
    
    # Return with embedded image bytes (same format as train_subset_200.parquet)
    return {
        "messages": messages,
        "images": [{"bytes": image_bytes}]
    }


def convert_json_to_parquet(
    input_json_path: str,
    output_path: str,
    images_base_dir: str,
    max_rows: Optional[int] = None,
    start_row: int = 0,
    shuffle: bool = False,
    seed: Optional[int] = None,
    rows_per_shard: Optional[int] = None
):
    """
    Convert JSON file to Levanter conversation format parquet file(s).

    Args:
        input_json_path: Path to input JSON file
        output_path: Path to output parquet file or directory (if sharding)
        images_base_dir: Base directory for images
        max_rows: Maximum number of rows to process (None for all)
        start_row: Starting row index
        shuffle: Whether to shuffle the data before processing
        seed: Random seed for shuffling (None for random seed)
        rows_per_shard: If set, output sharded parquet files with this many rows each.
                       Output will be output_path/train-00000.parquet, train-00001.parquet, etc.
    """
    print(f"Reading JSON file: {input_json_path}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle data if requested
    if shuffle:
        if seed is not None:
            random.seed(seed)
            print(f"Shuffling data with seed {seed}...")
        else:
            print("Shuffling data...")
        random.shuffle(data)
        print("Data shuffled.")

    total_rows = len(data)
    end_row = start_row + max_rows if max_rows else total_rows
    end_row = min(end_row, total_rows)

    print(f"Processing rows {start_row} to {end_row} (total: {total_rows})")

    # Setup output path
    output_path = Path(output_path)
    if rows_per_shard:
        # Sharded output - output_path is a directory
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sharded output mode: {rows_per_shard} rows per shard")
    else:
        # Single file output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process and collect rows
    converted_rows: List[Dict[str, Any]] = []
    skipped_count = 0
    shard_idx = 0
    total_written = 0

    def write_shard(rows: List[Dict[str, Any]], shard_num: int) -> int:
        """Write a shard and return number of rows written."""
        if not rows:
            return 0
        shard_path = output_dir / f"train-{shard_num:05d}.parquet"
        dataset = Dataset.from_list(rows)
        dataset.to_parquet(str(shard_path))
        print(f"  Written shard {shard_num}: {shard_path} ({len(rows)} rows)")
        return len(rows)

    for idx in range(start_row, end_row):
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{end_row}... (converted: {len(converted_rows) + total_written}, skipped: {skipped_count})")

        item = data[idx]

        try:
            image_path = item.get('image', '')
            if not image_path:
                print(f"Warning: Row {idx} has no image path, skipping", file=sys.stderr)
                skipped_count += 1
                continue

            conversations = item.get('conversations', [])
            if not conversations:
                print(f"Warning: Row {idx} has no conversations, skipping", file=sys.stderr)
                skipped_count += 1
                continue

            # Check if image file exists
            full_image_path = os.path.join(images_base_dir, image_path)
            if not os.path.exists(full_image_path):
                print(f"Warning: Image not found at {full_image_path}, skipping row {idx}", file=sys.stderr)
                skipped_count += 1
                continue

            conversation = parse_conversation_to_levanter(
                conversations,
                image_path,
                images_base_dir
            )
            converted_rows.append(conversation)

            # Write shard if we've accumulated enough rows
            if rows_per_shard and len(converted_rows) >= rows_per_shard:
                total_written += write_shard(converted_rows, shard_idx)
                shard_idx += 1
                converted_rows = []

        except Exception as e:
            print(f"Error processing row {idx}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    # Write remaining rows
    if converted_rows:
        if rows_per_shard:
            # Write final shard
            total_written += write_shard(converted_rows, shard_idx)
            shard_idx += 1
        else:
            # Single file output
            print(f"Creating HuggingFace dataset from {len(converted_rows)} rows...")
            dataset = Dataset.from_list(converted_rows)
            print(f"Writing to parquet: {output_path}...")
            dataset.to_parquet(str(output_path))
            total_written = len(converted_rows)

    # Summary
    print()
    print("=" * 50)
    print("Conversion complete!")
    if rows_per_shard:
        print(f"Output directory: {output_dir}")
        print(f"Total shards: {shard_idx}")
        print(f"Rows per shard: {rows_per_shard}")
    else:
        print(f"Output file: {output_path}")
    print(f"Successfully converted: {total_written} rows")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BLIP LAION CC SBU JSON dataset to Levanter conversation format (parquet)."
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "output_parquet",
        type=str,
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Base directory for images (default: same directory as input JSON + 'images')"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: all)"
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Starting row index (default: 0)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling (default: random)"
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=None,
        help="If set, output sharded parquet files with this many rows each. "
             "Output will be output_path/train-00000.parquet, train-00001.parquet, etc."
    )

    args = parser.parse_args()
    
    # Determine images directory
    if args.images_dir:
        images_base_dir = args.images_dir
    else:
        # Default: same directory as input JSON + 'images'
        input_path = Path(args.input_json)
        images_base_dir = str(input_path.parent / 'images')
    
    print(f"Input JSON: {args.input_json}")
    print(f"Output Parquet: {args.output_parquet}")
    print(f"Images directory: {images_base_dir}")
    print()
    
    if not os.path.exists(args.input_json):
        print(f"Error: Input JSON file not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(images_base_dir):
        print(f"Warning: Images directory not found: {images_base_dir}", file=sys.stderr)
        print("Continuing anyway, but images will be checked during conversion...", file=sys.stderr)
    
    convert_json_to_parquet(
        args.input_json,
        args.output_parquet,
        images_base_dir,
        args.max_rows,
        args.start_row,
        args.shuffle,
        args.seed,
        args.rows_per_shard
    )


if __name__ == "__main__":
    main()

