# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect a single row from an input parquet shard.

Downloads the shard and prints raw contents: messages, caption text, image token stats.

Usage:
    uv run experiments/unified/inspect_parquet_row.py --shard_index 0 --row_index 5
    uv run experiments/unified/inspect_parquet_row.py --shard_index 0 --row_index 5 \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized
"""

import argparse
import json
import os
import tempfile

import pyarrow.parquet as pq

from experiments.unified.vlm_tokenize_captions import (
    VISUAL_TOKEN_OFFSET,
    extract_caption,
    gcs_download,
)
from marin.utils import fsspec_glob


def main():
    parser = argparse.ArgumentParser(description="Inspect a single row from an input parquet shard.")
    parser.add_argument("--shard_index", type=int, required=True, help="Index of the parquet shard")
    parser.add_argument("--row_index", type=int, required=True, help="Row index within the shard")
    parser.add_argument("--input_path", default="gs://marin-vlm/stage2_sharded_full_tokenized")
    args = parser.parse_args()

    # List shards and pick the one at shard_index
    all_shards = sorted(fsspec_glob(f"{args.input_path}/train-*.parquet"))
    print(f"Found {len(all_shards)} shards")
    shard_path = all_shards[args.shard_index]
    gcs_shard = shard_path if shard_path.startswith("gs://") else f"gs://{shard_path}"

    # Download to temp
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "shard.parquet")
        gcs_download(gcs_shard, local)
        table = pq.read_table(local)

    print(f"Shard has {len(table)} rows, columns: {table.column_names}")

    row_idx = args.row_index
    messages = table.column("messages")[row_idx].as_py()
    image_token_lists = table.column("image_tokens")[row_idx].as_py()

    # Raw messages
    print(f"\n=== Shard {args.shard_index}, Row {row_idx} ===")
    print(f"\nMessages ({len(messages)} turns):")
    for msg in messages:
        role = msg["role"]
        for part in msg["content"]:
            if part["type"] == "text":
                text = part["text"]
                print(f"  [{role}] text: {text[:200]}{'...' if len(text) > 200 else ''}")
            elif part["type"] == "image":
                print(f"  [{role}] image")
            else:
                print(f"  [{role}] {part['type']}: {json.dumps(part)[:100]}")

    # Caption extraction
    caption = extract_caption(messages)
    print(f"\nExtracted caption: {caption[:200] if caption else None}{'...' if caption and len(caption) > 200 else ''}")

    # Image tokens
    n_images = sum(1 for msg in messages for part in msg["content"] if part["type"] == "image")
    print(f"\nImages in messages: {n_images}")
    print(f"Image token lists: {len(image_token_lists)}")
    if n_images != len(image_token_lists):
        print(f"  *** MISMATCH: {n_images} images vs {len(image_token_lists)} token lists ***")

    for img_i, tokens in enumerate(image_token_lists):
        n_tok = len(tokens)
        n_above = sum(1 for t in tokens if t >= VISUAL_TOKEN_OFFSET)
        print(f"\n  Image {img_i}: {n_tok} tokens")
        if tokens:
            print(f"    ID range: [{min(tokens)}, {max(tokens)}]")
            print(f"    Tokens >= {VISUAL_TOKEN_OFFSET} (already shifted?): {n_above}/{n_tok}")


if __name__ == "__main__":
    main()
