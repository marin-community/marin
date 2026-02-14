# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize eval benchmark parquets into Levanter cache format for training-time evaluation.

Reads parquet shards that have been augmented with an `image_tokens` column (via TokLIP
encoding) and builds token sequences with per-token loss weights:

  - Understanding (image-first): loss only on answer tokens
  - Generation (text-first): loss only on visual tokens

Output cache directory structure (consumed by Levanter's validation_sets):

    {output_path}/{benchmark}/validation/
        input_ids/{offsets, data}
        loss_weights/{offsets, data}
        shard_ledger.json

Prerequisites:
    1. Eval parquets must exist at {input_path}/{benchmark}/ with columns:
       messages, image_tokens, task_type (and optionally answer, benchmark, etc.)
    2. The `image_tokens` column must contain TokLIP codes (list[list[int]])
       produced by running the TokLIP encoder on the raw images.

Usage:
    # Tokenize all eval benchmarks
    uv run experiments/unified/tokenize_eval_benchmarks.py

    # Tokenize specific benchmarks
    uv run experiments/unified/tokenize_eval_benchmarks.py \
        --benchmarks textvqa chartqa

    # Understanding benchmarks only
    uv run experiments/unified/tokenize_eval_benchmarks.py \
        --benchmarks textvqa chartqa ai2d mmmu

    # Generation benchmarks only
    uv run experiments/unified/tokenize_eval_benchmarks.py \
        --benchmarks cifar10_small imagenet_small

    # Custom input/output paths
    uv run experiments/unified/tokenize_eval_benchmarks.py \
        --input_path gs://marin-vlm/eval_benchmarks_tokenized \
        --output_path gs://marin-vlm/unified_eval_cache \
        --benchmarks textvqa
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq
import transformers

from experiments.unified.unified_pretrain import UNIFIED_TOKENIZER_PATH
from experiments.unified.vlm_tokenize_captions import (
    ENDOFTEXT_ID,
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
    gcs_download,
    gcs_upload,
)
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

# --- Default paths ---

DEFAULT_INPUT_PATH = "gs://marin-vlm/eval_benchmarks_tokenized"
DEFAULT_OUTPUT_PATH = "gs://marin-vlm/unified_eval_cache"

ALL_BENCHMARKS = [
    # Understanding
    "textvqa",
    "chartqa",
    "ai2d",
    "mmmu",
    # Generation
    "cifar10_small",
    "cifar10",
    "imagenet_small",
    "imagenet",
]

GENERATION_TASK_TYPES = {"generation"}


# --- Sequence building ---


def _extract_role_content(messages: list[dict], role: str) -> list[dict]:
    """Extract content parts for a given role from a message list."""
    for msg in messages:
        if msg["role"] == role:
            return msg["content"]
    return []


def build_understanding_sequence(
    messages: list[dict],
    image_token_lists: list[list[int]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, np.ndarray] | None:
    """Build an image-first (understanding) sequence with loss only on answer tokens.

    Format: [user visual+text] [assistant answer text] <endoftext>
    Loss weights: 0.0 for user content (images + question), 1.0 for answer + eos.

    Handles multi-image cases (e.g., MMMU) by inserting visual tokens in order
    of appearance in the user content.
    """
    user_content = _extract_role_content(messages, "user")
    assistant_content = _extract_role_content(messages, "assistant")

    # Build user region: interleaved visual + text tokens
    user_ids: list[int] = []
    image_idx = 0
    for part in user_content:
        if part["type"] == "image":
            if image_idx < len(image_token_lists):
                raw_tokens = image_token_lists[image_idx]
                shifted = [t + VISUAL_TOKEN_OFFSET for t in raw_tokens]
                user_ids.append(VISION_START_ID)
                user_ids.extend(shifted)
                user_ids.append(VISION_END_ID)
                image_idx += 1
        elif part["type"] == "text" and part.get("text"):
            text_ids = tokenizer.encode(part["text"], add_special_tokens=False)
            user_ids.extend(text_ids)

    # Build assistant region: answer text tokens
    assistant_ids: list[int] = []
    for part in assistant_content:
        if part["type"] == "text" and part.get("text"):
            text_ids = tokenizer.encode(part["text"], add_special_tokens=False)
            assistant_ids.extend(text_ids)

    if not assistant_ids:
        return None

    # Full sequence: user + assistant + eos
    all_ids = user_ids + assistant_ids + [ENDOFTEXT_ID]
    n_user = len(user_ids)

    input_ids = np.array(all_ids, dtype=np.int32)
    loss_weights = np.zeros(len(all_ids), dtype=np.float32)
    loss_weights[n_user:] = 1.0  # answer + eos

    return {"input_ids": input_ids, "loss_weights": loss_weights}


def build_generation_sequence(
    messages: list[dict],
    image_token_lists: list[list[int]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, np.ndarray] | None:
    """Build a text-first (generation) sequence with loss only on visual tokens.

    Format: [user prompt text] <vision_start> V₁..Vₙ <vision_end> <endoftext>
    Loss weights: 0.0 for prompt, 1.0 for visual region + eos.
    """
    user_content = _extract_role_content(messages, "user")

    # Build prompt region: text tokens only
    prompt_ids: list[int] = []
    for part in user_content:
        if part["type"] == "text" and part.get("text"):
            text_ids = tokenizer.encode(part["text"], add_special_tokens=False)
            prompt_ids.extend(text_ids)

    if not image_token_lists:
        return None

    # Use the first image's tokens
    raw_tokens = image_token_lists[0]
    shifted = [t + VISUAL_TOKEN_OFFSET for t in raw_tokens]
    visual_ids = [VISION_START_ID, *shifted, VISION_END_ID]

    # Full sequence: prompt + visual + eos
    all_ids = prompt_ids + visual_ids + [ENDOFTEXT_ID]
    n_prompt = len(prompt_ids)

    input_ids = np.array(all_ids, dtype=np.int32)
    loss_weights = np.zeros(len(all_ids), dtype=np.float32)
    loss_weights[n_prompt:] = 1.0  # visual + eos

    return {"input_ids": input_ids, "loss_weights": loss_weights}


def process_eval_rows(
    table,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Iterator[dict[str, np.ndarray]]:
    """Process rows in a parquet table, yielding token sequences.

    Reads `task_type` column to determine understanding vs. generation ordering.
    """
    messages_col = table.column("messages")
    image_tokens_col = table.column("image_tokens")
    task_type_col = table.column("task_type")

    total = len(table)
    skipped = 0

    for i in range(total):
        messages = messages_col[i].as_py()
        image_token_lists = image_tokens_col[i].as_py()
        task_type = task_type_col[i].as_py()

        if task_type in GENERATION_TASK_TYPES:
            result = build_generation_sequence(messages, image_token_lists, tokenizer)
        else:
            result = build_understanding_sequence(messages, image_token_lists, tokenizer)

        if result is None:
            skipped += 1
            continue

        yield result

        if (i + 1) % 1000 == 0:
            logger.info("  ... processed %d/%d rows", i + 1, total)

    if skipped > 0:
        logger.info("Skipped %d/%d rows (missing data)", skipped, total)


# --- Shard processing ---


def process_shard_local(
    local_parquet_path: str,
    local_output_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    source_shard: str,
) -> dict:
    """Process a single parquet shard and write Levanter cache locally."""
    from zephyr.writers import write_levanter_cache

    table = pq.read_table(local_parquet_path)
    records = process_eval_rows(table, tokenizer)

    metadata = {
        "source_shard": source_shard,
        "visual_token_offset": VISUAL_TOKEN_OFFSET,
        "format": "eval_benchmark",
    }

    result = write_levanter_cache(records, local_output_path, metadata)
    logger.info(
        "Shard %s: %d records, %d tokens",
        source_shard,
        result["count"],
        result["token_count"],
    )
    return result


# --- Benchmark tokenization ---


def _cache_tokenizer_locally(tokenizer_path: str) -> str:
    """Download a (possibly GCS-hosted) tokenizer to a local temp directory."""
    from levanter.compat.hf_checkpoints import load_tokenizer

    local_dir = tempfile.mkdtemp(prefix="tokenizer_cache_")
    tok = load_tokenizer(tokenizer_path)
    tok.save_pretrained(local_dir)
    logger.info("Tokenizer cached locally (vocab_size=%d)", len(tok))
    return local_dir


def tokenize_benchmark(
    benchmark: str,
    input_path: str,
    output_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Tokenize a single eval benchmark into a consolidated Levanter cache.

    Processes all parquet shards for the benchmark, writes per-shard caches,
    then consolidates into {output_path}/{benchmark}/validation/.
    """
    from levanter.store.cache import consolidate_shard_caches

    # Find parquet shards
    pattern = f"{input_path}/{benchmark}/eval-{benchmark}-*.parquet"
    shard_paths = sorted(fsspec_glob(pattern))
    if not shard_paths:
        logger.warning("No shards found for %s (pattern: %s)", benchmark, pattern)
        return {"benchmark": benchmark, "total_records": 0, "total_tokens": 0, "shards": 0}

    logger.info("Found %d shards for %s", len(shard_paths), benchmark)

    work_dir = tempfile.mkdtemp(prefix=f"eval_tok_{benchmark}_")
    local_parquet_dir = os.path.join(work_dir, "parquets")
    local_cache_dir = os.path.join(work_dir, "caches")
    os.makedirs(local_parquet_dir)
    os.makedirs(local_cache_dir)

    shard_cache_paths_local = []
    shard_cache_paths_gcs = []
    total_records = 0
    total_tokens = 0

    for shard_idx, shard_path in enumerate(shard_paths):
        gcs_shard = shard_path if shard_path.startswith("gs://") else f"gs://{shard_path}"
        filename = shard_path.rsplit("/", 1)[-1]
        shard_name = filename.replace(".parquet", "")
        local_parquet = os.path.join(local_parquet_dir, filename)
        local_cache = os.path.join(local_cache_dir, f"part-{shard_name}")

        # Download parquet
        logger.info("Downloading shard %d/%d: %s", shard_idx + 1, len(shard_paths), filename)
        gcs_download(gcs_shard, local_parquet)

        # Process
        result = process_shard_local(local_parquet, local_cache, tokenizer, shard_path)
        total_records += result["count"]
        total_tokens += result["token_count"]

        # Upload shard cache to GCS
        gcs_cache_path = f"{output_path}/{benchmark}/validation/part-{shard_name}"
        gcs_upload(local_cache, gcs_cache_path)

        shard_cache_paths_local.append(local_cache)
        shard_cache_paths_gcs.append(gcs_cache_path)

    # Consolidate shard caches into final validation cache
    final_path = f"{output_path}/{benchmark}/validation"
    logger.info("Consolidating %d shard caches into %s", len(shard_cache_paths_gcs), final_path)

    exemplar = {
        "input_ids": np.zeros((0,), dtype=np.int32),
        "loss_weights": np.zeros((0,), dtype=np.float32),
    }
    consolidate_shard_caches(
        shard_cache_paths=shard_cache_paths_gcs,
        output_path=final_path,
        exemplar=exemplar,
    )
    logger.info("Consolidation complete for %s", benchmark)

    # Clean up local files
    shutil.rmtree(work_dir)

    stats = {
        "benchmark": benchmark,
        "total_records": total_records,
        "total_tokens": total_tokens,
        "shards": len(shard_paths),
    }
    logger.info(
        "Completed %s: %d records, %d tokens, %d shards",
        benchmark,
        total_records,
        total_tokens,
        len(shard_paths),
    )
    return stats


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="Tokenize eval benchmark parquets into Levanter cache format.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
        help=f"Benchmarks to tokenize (default: all). Choices: {ALL_BENCHMARKS}",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="GCS path containing benchmark parquets (with image_tokens column)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="GCS path for output Levanter caches",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=UNIFIED_TOKENIZER_PATH,
        help="Path to the unified tokenizer",
    )
    args = parser.parse_args()

    # Cache tokenizer locally
    local_tokenizer_path = _cache_tokenizer_locally(args.tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_tokenizer_path)

    all_stats = []
    for benchmark in args.benchmarks:
        logger.info("=" * 60)
        logger.info("Tokenizing benchmark: %s", benchmark)
        stats = tokenize_benchmark(
            benchmark=benchmark,
            input_path=args.input_path,
            output_path=args.output_path,
            tokenizer=tokenizer,
        )
        all_stats.append(stats)

    # Print summary
    logger.info("=" * 60)
    logger.info("Tokenization Summary:")
    total_records = 0
    total_tokens = 0
    for stats in all_stats:
        logger.info(
            "  %s: %d records, %d tokens, %d shards",
            stats["benchmark"],
            stats["total_records"],
            stats["total_tokens"],
            stats["shards"],
        )
        total_records += stats["total_records"]
        total_tokens += stats["total_tokens"]
    logger.info(
        "Total: %d records, %d tokens across %d benchmarks",
        total_records,
        total_tokens,
        len(all_stats),
    )

    # Clean up tokenizer cache
    shutil.rmtree(local_tokenizer_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
