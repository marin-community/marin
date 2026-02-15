# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize a single row from a converted eval benchmark parquet.

Prints all text fields (messages, answer, choices, metadata) and saves
any images to disk for inspection.

Usage:
    # Local parquet
    uv run experiments/unified/inspect_eval_parquet.py \
        /tmp/eval_benchmarks/textvqa/eval-textvqa-00000.parquet --row 5

    # GCS parquet
    uv run experiments/unified/inspect_eval_parquet.py \
        gs://marin-vlm/eval_benchmarks/textvqa/eval-textvqa-00000.parquet --row 0

    # Custom output directory for saved images
    uv run experiments/unified/inspect_eval_parquet.py \
        /tmp/eval_benchmarks/mmmu/eval-mmmu-00000.parquet --row 42 --save-dir /tmp/inspect
"""

import argparse
import io
import json
import os
import tempfile

import pyarrow.parquet as pq
from PIL import Image


def load_table(parquet_path: str):
    """Load a parquet table from a local or GCS path."""
    if parquet_path.startswith("gs://"):
        from experiments.unified.vlm_tokenize_captions import gcs_download

        tmp = tempfile.mkdtemp(prefix="inspect_eval_")
        local = os.path.join(tmp, "shard.parquet")
        print(f"Downloading {parquet_path} ...")
        gcs_download(parquet_path, local)
        return pq.read_table(local)
    return pq.read_table(parquet_path)


def inspect_row(table, row_idx: int, save_dir: str):
    """Print a single row's contents and save images."""
    os.makedirs(save_dir, exist_ok=True)

    n_rows = len(table)
    print(f"\n=== Row {row_idx} / {n_rows} ===")

    # Scalar fields
    for col in ["benchmark", "task_type", "question_id", "source", "split"]:
        if col in table.column_names:
            val = table.column(col)[row_idx].as_py()
            print(f"{col}: {val}")

    # Messages
    messages = table.column("messages")[row_idx].as_py()
    print(f"\nMessages ({len(messages)} turns):")
    img_save_idx = 0
    for msg_i, msg in enumerate(messages):
        role = msg["role"]
        print(f"  [{msg_i}] role={role}")
        for part in msg["content"]:
            if part["type"] == "image":
                img_path = os.path.join(save_dir, f"row_{row_idx}_img_{img_save_idx}.png")
                # Try to save the actual image from the images column
                images = table.column("images")[row_idx].as_py()
                if img_save_idx < len(images):
                    img_bytes = images[img_save_idx]["bytes"]
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)
                    print(f"      [image] {img.size[0]}x{img.size[1]} → {img_path}")
                else:
                    print(f"      [image] (no image data at index {img_save_idx})")
                img_save_idx += 1
            elif part["type"] == "text":
                text = part.get("text", "")
                # Indent multiline text
                lines = text.split("\n")
                print(f"      [text]  {lines[0]}")
                for line in lines[1:]:
                    print(f"              {line}")

    # Answer and choices
    answer = table.column("answer")[row_idx].as_py()
    print(f"\nAnswer: {answer}")

    if "choices" in table.column_names:
        choices = table.column("choices")[row_idx].as_py()
        print(f"Choices: {choices}")

    # Metadata
    if "metadata" in table.column_names:
        metadata_raw = table.column("metadata")[row_idx].as_py()
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
                print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"Metadata: {metadata_raw}")

    # Image summary
    images = table.column("images")[row_idx].as_py()
    print(f"\nImages: {len(images)} total, saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a row from an eval benchmark parquet.")
    parser.add_argument("parquet_path", help="Path to the parquet file (local or gs://)")
    parser.add_argument("--row", type=int, default=0, help="Row index to inspect (default: 0)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/tmp/inspect_eval",
        help="Directory to save extracted images (default: /tmp/inspect_eval)",
    )
    args = parser.parse_args()

    table = load_table(args.parquet_path)
    print(f"Loaded {len(table)} rows, columns: {table.column_names}")

    if args.row >= len(table):
        print(f"Error: row {args.row} out of range (table has {len(table)} rows)")
        return

    inspect_row(table, args.row, args.save_dir)


if __name__ == "__main__":
    main()
