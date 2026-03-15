# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect a row from a tokenized eval benchmark parquet (with image_tokens column).

Prints all text fields (messages, answer, choices, metadata), image token
statistics, and optionally previews the full tokenized sequence that
tokenize_eval_benchmarks.py would produce.

Usage:
    # Basic row inspection (no tokenizer needed)
    uv run experiments/unified/inspect_eval_tokenized_parquet.py \
        gs://marin-vlm/eval_benchmarks_tokenized/textvqa/eval-textvqa-00000.parquet --row 0

    # With tokenized sequence preview (loads tokenizer)
    uv run experiments/unified/inspect_eval_tokenized_parquet.py \
        gs://marin-vlm/eval_benchmarks_tokenized/textvqa/eval-textvqa-00000.parquet \
        --row 0 --preview

    # Summary statistics across all rows
    uv run experiments/unified/inspect_eval_tokenized_parquet.py \
        gs://marin-vlm/eval_benchmarks_tokenized/textvqa/eval-textvqa-00000.parquet --summary

    # Generation benchmark
    uv run experiments/unified/inspect_eval_tokenized_parquet.py \
        gs://marin-vlm/eval_benchmarks_tokenized/cifar10_small/eval-cifar10_small-00000.parquet \
        --row 0 --preview
"""

import argparse
import io
import json
import os
import tempfile
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from experiments.unified.vlm_tokenize_captions import (
    ENDOFTEXT_ID,
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
)


def load_table(parquet_path: str):
    """Load a parquet table from a local or GCS path."""
    if parquet_path.startswith("gs://"):
        from experiments.unified.vlm_tokenize_captions import gcs_download

        tmp = tempfile.mkdtemp(prefix="inspect_eval_tok_")
        local = os.path.join(tmp, "shard.parquet")
        print(f"Downloading {parquet_path} ...")
        gcs_download(parquet_path, local)
        return pq.read_table(local)
    return pq.read_table(parquet_path)


def inspect_row(table, row_idx: int, save_dir: str):
    """Print a single row's contents, image token stats, and save images if available."""
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
    img_placeholder_idx = 0
    for msg_i, msg in enumerate(messages):
        role = msg["role"]
        print(f"  [{msg_i}] role={role}")
        for part in msg["content"]:
            if part["type"] == "image":
                # Try to save raw image if images column exists
                if "images" in table.column_names:
                    images = table.column("images")[row_idx].as_py()
                    if img_placeholder_idx < len(images):
                        img_bytes = images[img_placeholder_idx]["bytes"]
                        img = Image.open(io.BytesIO(img_bytes))
                        img_path = os.path.join(save_dir, f"row_{row_idx}_img_{img_placeholder_idx}.png")
                        img.save(img_path)
                        print(f"      [image {img_placeholder_idx}] {img.size[0]}x{img.size[1]} -> {img_path}")
                    else:
                        print(f"      [image {img_placeholder_idx}] (no image data)")
                else:
                    print(f"      [image {img_placeholder_idx}] (no images column)")
                img_placeholder_idx += 1
            elif part["type"] == "text":
                text = part.get("text", "")
                lines = text.split("\n")
                print(f"      [text]  {lines[0]}")
                for line in lines[1:]:
                    print(f"              {line}")

    # Answer and choices
    if "answer" in table.column_names:
        answer = table.column("answer")[row_idx].as_py()
        print(f"\nAnswer: {answer}")

    if "choices" in table.column_names:
        choices = table.column("choices")[row_idx].as_py()
        if choices:
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

    # Image tokens
    if "image_tokens" in table.column_names:
        image_token_lists = table.column("image_tokens")[row_idx].as_py()
        print(f"\nImage tokens ({len(image_token_lists)} images):")
        for img_i, tokens in enumerate(image_token_lists):
            if not tokens:
                print(f"  [image {img_i}] empty")
                continue
            arr = np.array(tokens)
            print(
                f"  [image {img_i}] {len(tokens)} codes, "
                f"range=[{arr.min()}, {arr.max()}], "
                f"unique={len(np.unique(arr))}, "
                f"shifted_range=[{arr.min() + VISUAL_TOKEN_OFFSET}, {arr.max() + VISUAL_TOKEN_OFFSET}]"
            )


def preview_tokenized_sequence(table, row_idx: int):
    """Build and display the full tokenized sequence for a row.

    Uses the same logic as tokenize_eval_benchmarks.py to build input_ids
    and loss_weights, then displays a breakdown similar to inspect_parquet_row.py.
    """

    from experiments.unified.tokenize_eval_benchmarks import (
        GENERATION_TASK_TYPES,
        build_generation_sequence,
        build_understanding_sequence,
    )
    from experiments.unified.unified_pretrain import UNIFIED_TOKENIZER_PATH

    messages = table.column("messages")[row_idx].as_py()
    image_token_lists = table.column("image_tokens")[row_idx].as_py()
    task_type = table.column("task_type")[row_idx].as_py()

    print("\n--- Tokenized Sequence Preview ---")

    # Load tokenizer
    from levanter.compat.hf_checkpoints import load_tokenizer

    tok = load_tokenizer(UNIFIED_TOKENIZER_PATH)

    if task_type in GENERATION_TASK_TYPES:
        result = build_generation_sequence(messages, image_token_lists, tok)
        mode = "generation (text-first)"
    else:
        result = build_understanding_sequence(messages, image_token_lists, tok)
        mode = "understanding (image-first)"

    if result is None:
        print("  (could not build sequence — missing data)")
        return

    ids = result["input_ids"]
    weights = result["loss_weights"]

    n_total = len(ids)
    n_visual = int(np.sum(ids >= VISUAL_TOKEN_OFFSET))
    n_special = int(np.sum(np.isin(ids, [VISION_START_ID, VISION_END_ID, ENDOFTEXT_ID])))
    n_text = n_total - n_visual - n_special

    # Decode text tokens
    non_visual_ids = [int(t) for t in ids if t < VISUAL_TOKEN_OFFSET]
    decoded_text = tok.decode(non_visual_ids)

    n_loss_active = int(np.sum(weights > 0))

    print(f"  Mode: {mode}")
    print(f"  Total tokens: {n_total}")
    print(f"    Visual tokens (>= {VISUAL_TOKEN_OFFSET}): {n_visual}")
    print(f"    Text tokens: {n_text}")
    print(f"    Special tokens: {n_special}")
    print(f"  Loss active on: {n_loss_active} / {n_total} tokens ({n_loss_active / n_total * 100:.1f}%)")
    print(f"  Decoded text: {decoded_text}")
    print(f"  First 10 token IDs: {[int(t) for t in ids[:10]]}")
    print(f"  First 10 weights:   {[float(w) for w in weights[:10]]}")
    print(f"  Last 10 token IDs:  {[int(t) for t in ids[-10:]]}")
    print(f"  Last 10 weights:    {[float(w) for w in weights[-10:]]}")


def print_summary(table):
    """Print aggregate statistics across all rows in the parquet."""
    n_rows = len(table)
    print(f"\n=== Summary ({n_rows} rows) ===")

    # Task type distribution
    if "task_type" in table.column_names:
        task_types = [table.column("task_type")[i].as_py() for i in range(n_rows)]
        counts = Counter(task_types)
        print("\nTask type distribution:")
        for task_type, count in counts.most_common():
            print(f"  {task_type}: {count}")

    # Benchmark distribution
    if "benchmark" in table.column_names:
        benchmarks = [table.column("benchmark")[i].as_py() for i in range(n_rows)]
        counts = Counter(benchmarks)
        print("\nBenchmark distribution:")
        for benchmark, count in counts.most_common():
            print(f"  {benchmark}: {count}")

    # Image token statistics
    if "image_tokens" in table.column_names:
        all_token_counts = []
        all_num_images = []
        for i in range(n_rows):
            image_token_lists = table.column("image_tokens")[i].as_py()
            all_num_images.append(len(image_token_lists))
            for tokens in image_token_lists:
                all_token_counts.append(len(tokens))

        if all_token_counts:
            arr = np.array(all_token_counts)
            print("\nImage token counts (per image):")
            print(f"  Total images: {len(all_token_counts)}")
            print(
                f"  Tokens/image: min={arr.min()}, mean={arr.mean():.1f}, max={arr.max()}, median={np.median(arr):.0f}"
            )

        if all_num_images:
            arr = np.array(all_num_images)
            print("\nImages per row:")
            print(f"  min={arr.min()}, mean={arr.mean():.1f}, max={arr.max()}")
            counts = Counter(all_num_images)
            for n_img, count in sorted(counts.items()):
                print(f"  {n_img} images: {count} rows")


def main():
    parser = argparse.ArgumentParser(description="Inspect a row from a tokenized eval benchmark parquet.")
    parser.add_argument("parquet_path", help="Path to the parquet file (local or gs://)")
    parser.add_argument("--row", type=int, default=0, help="Row index to inspect (default: 0)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/tmp/inspect_eval_tokenized",
        help="Directory to save extracted images (default: /tmp/inspect_eval_tokenized)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the full tokenized sequence (loads the unified tokenizer)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate statistics across all rows",
    )
    args = parser.parse_args()

    table = load_table(args.parquet_path)
    print(f"Loaded {len(table)} rows, columns: {table.column_names}")

    if args.summary:
        print_summary(table)

    if args.row >= len(table):
        print(f"Error: row {args.row} out of range (table has {len(table)} rows)")
        return

    inspect_row(table, args.row, args.save_dir)

    if args.preview:
        preview_tokenized_sequence(table, args.row)


if __name__ == "__main__":
    main()
