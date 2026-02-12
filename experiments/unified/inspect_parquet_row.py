# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect records from the tokenized Levanter cache output.

Reads directly from the output cache produced by vlm_tokenize_captions.py
and displays token breakdown, decoded text, and loss weights.

Usage:
    uv run experiments/unified/inspect_parquet_row.py --row_index 5
    uv run experiments/unified/inspect_parquet_row.py --row_index 0 \
        --cache_path gs://marin-vlm/stage2_sharded_full_tokenized_llama3/train
"""

import argparse

import numpy as np

from experiments.unified.unified_pretrain import UNIFIED_TOKENIZER_PATH
from experiments.unified.vlm_tokenize_captions import (
    ENDOFTEXT_ID,
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
)


def print_record(index, record, tok):
    ids = record["input_ids"]
    weights = record["loss_weights"]

    n_total = len(ids)
    n_visual = sum(1 for t in ids if t >= VISUAL_TOKEN_OFFSET)
    n_text = sum(1 for t in ids if t < VISUAL_TOKEN_OFFSET and t not in (VISION_START_ID, VISION_END_ID, ENDOFTEXT_ID))

    # Decode all non-visual tokens (text + special tokens like vision_start, vision_end, endoftext)
    non_visual_ids = [int(t) for t in ids if t < VISUAL_TOKEN_OFFSET]
    decoded_text = tok.decode(non_visual_ids)

    # Determine ordering from token layout
    if n_visual > 0 and ids[0] == VISION_START_ID:
        ordering = "image-first (understanding)"
    elif n_visual > 0:
        ordering = "text-first (generation)"
    else:
        ordering = "text-only"

    print(f"\n=== Record {index} ({n_total} tokens, {ordering}) ===")
    print(f"  Visual tokens (>= {VISUAL_TOKEN_OFFSET}): {n_visual}")
    print(f"  Text tokens: {n_text}")
    print(f"  Special tokens: {n_total - n_visual - n_text}")
    print(f"  Decoded text: {decoded_text}")
    print(f"  Loss weights: min={weights.min():.2f}, max={weights.max():.2f}")
    print(f"  First 10 tokens: {[int(t) for t in ids[:10]]}")
    print(f"  First 10 weights: {[float(w) for w in weights[:10]]}")
    print(f"  Last 10 tokens:  {[int(t) for t in ids[-10:]]}")
    print(f"  Last 10 weights: {[float(w) for w in weights[-10:]]}")


def main():
    parser = argparse.ArgumentParser(description="Inspect records from tokenized Levanter cache.")
    parser.add_argument("--row_index", type=int, required=True, help="Record index in the cache")
    parser.add_argument("--cache_path", default="gs://marin-vlm/stage2_sharded_full_tokenized_llama3/train")
    parser.add_argument("--tokenizer", default=UNIFIED_TOKENIZER_PATH)
    args = parser.parse_args()

    from levanter.compat.hf_checkpoints import load_tokenizer
    from levanter.store.cache import TreeCache

    print(f"Loading cache from {args.cache_path} ...")
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "loss_weights": np.zeros((0,), dtype=np.float32)}
    cache = TreeCache.load(args.cache_path, exemplar=exemplar)
    print(f"Cache has {len(cache)} records")

    tok = load_tokenizer(args.tokenizer)
    record = cache[args.row_index]
    print_record(args.row_index, record, tok)


if __name__ == "__main__":
    main()
