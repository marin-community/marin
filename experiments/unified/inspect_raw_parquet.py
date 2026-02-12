# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect a raw row from a pre-tokenized parquet shard (before vlm_tokenize_captions.py).

Shows the caption text, image token counts/ranges, and message structure.

Usage:
    uv run experiments/unified/inspect_raw_parquet.py --shard_index 0 --row_index 5
    uv run experiments/unified/inspect_raw_parquet.py --shard_index 0 --row_index 0 \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Inspect raw parquet row before tokenization.")
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--row_index", type=int, required=True)
    parser.add_argument("--input_path", default="gs://marin-vlm/stage2_sharded_full_tokenized")
    args = parser.parse_args()

    all_shards = sorted(fsspec_glob(f"{args.input_path}/train-*.parquet"))
    print(f"Found {len(all_shards)} shards")
    shard_path = all_shards[args.shard_index]
    gcs_shard = shard_path if shard_path.startswith("gs://") else f"gs://{shard_path}"

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "shard.parquet")
        print(f"Downloading {gcs_shard} ...")
        gcs_download(gcs_shard, local)
        table = pq.read_table(local)

    print(f"Shard {args.shard_index} has {len(table)} rows")
    print(f"Columns: {table.column_names}")

    row_idx = args.row_index
    messages = table.column("messages")[row_idx].as_py()
    image_token_lists = table.column("image_tokens")[row_idx].as_py()

    caption = extract_caption(messages)

    print(f"\n=== Shard {args.shard_index}, Row {row_idx} ===")

    # Messages structure
    print(f"\nMessages ({len(messages)} turns):")
    for msg_i, msg in enumerate(messages):
        role = msg["role"]
        parts = msg["content"]
        part_types = [p["type"] for p in parts]
        print(f"  [{msg_i}] role={role}, parts={part_types}")
        for p in parts:
            if p["type"] == "text" and p["text"]:
                print(f"       text: {p['text'][:200]}{'...' if len(p['text']) > 200 else ''}")

    # Caption
    print(f"\nCaption: {caption}")

    # Image tokens
    n_images = sum(1 for msg in messages for part in msg["content"] if part["type"] == "image")
    print(f"\nImages in messages: {n_images}")
    print(f"Image token lists: {len(image_token_lists)}")

    for img_i, tokens in enumerate(image_token_lists):
        print(f"\n  Image {img_i}: {len(tokens)} tokens")
        print(f"    Raw range: [{min(tokens)}, {max(tokens)}]")
        print(f"    Shifted range: [{min(tokens) + VISUAL_TOKEN_OFFSET}, {max(tokens) + VISUAL_TOKEN_OFFSET}]")
        print(f"    First 10: {tokens[:10]}")
        print(f"    Last 10:  {tokens[-10:]}")


if __name__ == "__main__":
    main()
