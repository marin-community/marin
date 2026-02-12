# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize pre-tokenized TokLIP image-caption data into unified sequences for VLM pre-training.

Reads parquet shards from GCS containing TokLIP image tokens and captions,
produces dual-ordering (image-first + text-first) token sequences with per-token
loss weights, and writes them to Levanter cache format.

Usage:
    uv run experiments/unified/vlm_tokenize_captions.py \
        --output_path gs://marin-vlm/vlm_pretraining_cache \
        --start_shard 0 --end_shard 10 \
        --generation_ratio 0.5
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

import draccus
import gcsfs
import numpy as np
import pyarrow.parquet as pq
import transformers

from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)

# --- Constants (from unified_model_scaling_law.md) ---

VISUAL_TOKEN_OFFSET = 151_936  # TokLIP codebook index + offset = unified token ID
VISION_START_ID = 151_652  # <|vision_start|>
VISION_END_ID = 151_653  # <|vision_end|>
ENDOFTEXT_ID = 151_643  # <|endoftext|>


def extract_caption(messages: list[dict]) -> str | None:
    """Extract the assistant's caption text from a chat-format message list.

    Returns None if no assistant text is found.
    """
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for part in msg["content"]:
            if part["type"] == "text" and part["text"]:
                return part["text"]
    return None


def build_image_first_sequence(
    caption_ids: list[int],
    shifted_image_tokens: list[int],
    w_visual: float,
) -> dict[str, np.ndarray]:
    """Build an image-first (understanding-oriented) sequence.

    Format: <|vision_start|> V₁..V₅₇₆ <|vision_end|> T₁..Tₙ <|endoftext|>
    Loss weights: w_visual for visual region, 1.0 for text + endoftext.
    """
    n_visual = len(shifted_image_tokens)
    n_text = len(caption_ids)

    # vision_start + image_tokens + vision_end + caption + endoftext
    total_len = 1 + n_visual + 1 + n_text + 1

    input_ids = np.empty(total_len, dtype=np.int32)
    loss_weights = np.empty(total_len, dtype=np.float32)

    # Visual region: vision_start + image tokens + vision_end
    input_ids[0] = VISION_START_ID
    input_ids[1 : 1 + n_visual] = shifted_image_tokens
    input_ids[1 + n_visual] = VISION_END_ID
    loss_weights[: 2 + n_visual] = w_visual

    # Text region: caption + endoftext
    text_start = 2 + n_visual
    input_ids[text_start : text_start + n_text] = caption_ids
    input_ids[text_start + n_text] = ENDOFTEXT_ID
    loss_weights[text_start:] = 1.0

    return {"input_ids": input_ids, "loss_weights": loss_weights}


def build_text_first_sequence(
    caption_ids: list[int],
    shifted_image_tokens: list[int],
    w_visual: float,
) -> dict[str, np.ndarray]:
    """Build a text-first (generation-oriented) sequence.

    Format: T₁..Tₙ <|vision_start|> V₁..V₅₇₆ <|vision_end|> <|endoftext|>
    Loss weights: w_visual for text tokens, 1.0 for visual region + endoftext.
    """
    n_text = len(caption_ids)
    n_visual = len(shifted_image_tokens)

    # caption + vision_start + image_tokens + vision_end + endoftext
    total_len = n_text + 1 + n_visual + 1 + 1

    input_ids = np.empty(total_len, dtype=np.int32)
    loss_weights = np.empty(total_len, dtype=np.float32)

    # Text region: caption
    input_ids[:n_text] = caption_ids
    loss_weights[:n_text] = w_visual

    # Visual region: vision_start + image tokens + vision_end + endoftext
    vis_start = n_text
    input_ids[vis_start] = VISION_START_ID
    input_ids[vis_start + 1 : vis_start + 1 + n_visual] = shifted_image_tokens
    input_ids[vis_start + 1 + n_visual] = VISION_END_ID
    input_ids[vis_start + 2 + n_visual] = ENDOFTEXT_ID
    loss_weights[vis_start:] = 1.0

    return {"input_ids": input_ids, "loss_weights": loss_weights}


def process_parquet_rows(
    table,
    tokenizer: transformers.PreTrainedTokenizer,
    w_visual: float,
    generation_ratio: float,
) -> Iterator[dict[str, np.ndarray]]:
    """Process all rows in a PyArrow table, yielding understanding or generation records.

    Each row produces a single sequence. The first (1 - generation_ratio) fraction
    of valid rows yield image-first (understanding) sequences; the remaining
    generation_ratio fraction yield text-first (generation) sequences.
    """
    messages_col = table.column("messages")
    image_tokens_col = table.column("image_tokens")

    # First pass: count valid rows to compute the threshold
    valid_indices = []
    for i in range(len(table)):
        messages = messages_col[i].as_py()
        image_token_lists = image_tokens_col[i].as_py()
        caption = extract_caption(messages)
        if caption is not None and image_token_lists:
            valid_indices.append(i)

    skipped = len(table) - len(valid_indices)
    threshold = int(len(valid_indices) * (1.0 - generation_ratio))

    for rank, i in enumerate(valid_indices):
        messages = messages_col[i].as_py()
        image_token_lists = image_tokens_col[i].as_py()

        caption = extract_caption(messages)

        # Use the first image's tokens (all examples in this dataset are single-image)
        raw_image_tokens = image_token_lists[0]
        shifted_image_tokens = [t + VISUAL_TOKEN_OFFSET for t in raw_image_tokens]

        # Tokenize caption (no BOS per doc spec)
        caption_ids = tokenizer.encode(caption, add_special_tokens=False)

        if rank < threshold:
            yield build_image_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        else:
            yield build_text_first_sequence(caption_ids, shifted_image_tokens, w_visual)

    if skipped > 0:
        logger.warning("Skipped %d rows with missing caption or image tokens", skipped)


def process_shard(
    shard_path: str,
    output_path: str,
    tokenizer_name: str,
    w_visual: float,
    generation_ratio: float,
) -> dict:
    """Process a single parquet shard and write to Levanter cache.

    This function is designed to run in a worker process.
    """
    from zephyr.writers import write_levanter_cache

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    fs = gcsfs.GCSFileSystem()

    with fs.open(shard_path) as f:
        table = pq.read_table(f)

    records = process_parquet_rows(table, tokenizer, w_visual, generation_ratio)
    metadata = {
        "source_shard": shard_path,
        "tokenizer": tokenizer_name,
        "visual_token_offset": VISUAL_TOKEN_OFFSET,
        "w_visual": w_visual,
        "generation_ratio": generation_ratio,
        "format": "dual_ordering_pretraining",
    }

    result = write_levanter_cache(records, output_path, metadata)
    logger.info("Shard %s → %s: %d records, %d tokens", shard_path, output_path, result["count"], result["token_count"])
    return result


def _list_shard_paths(input_path: str, start_shard: int | None, end_shard: int | None) -> list[str]:
    """List parquet shard paths, optionally filtering by index range."""
    all_paths = sorted(fsspec_glob(f"{input_path}/train-*.parquet"))
    if not all_paths:
        raise ValueError(f"No parquet shards found at {input_path}")
    logger.info("Found %d total shards at %s", len(all_paths), input_path)

    if start_shard is not None or end_shard is not None:
        start = start_shard or 0
        end = end_shard if end_shard is not None else len(all_paths) - 1
        all_paths = all_paths[start : end + 1]
        logger.info("Selected shards %d-%d (%d shards)", start, end, len(all_paths))

    return all_paths


@dataclasses.dataclass
class TokenizeVLMConfig:
    input_path: str = "gs://marin-vlm/stage2_sharded_full_tokenized"
    output_path: str = "gs://marin-vlm/vlm_pretraining_cache"
    tokenizer: str = "Qwen/Qwen3-0.6B"
    w_visual: float = 0.5
    generation_ratio: float = 0.5
    num_workers: int = 32
    start_shard: int | None = None
    end_shard: int | None = None


@draccus.wrap()
def main(config: TokenizeVLMConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    shard_paths = _list_shard_paths(config.input_path, config.start_shard, config.end_shard)

    # Build per-shard output paths
    shard_outputs = []
    for shard_path in shard_paths:
        shard_name = shard_path.rsplit("/", 1)[-1].replace(".parquet", "")
        out = f"{config.output_path}/train/part-{shard_name}"
        shard_outputs.append((shard_path, out))

    # Skip shards that already have a .success sentinel
    pending = []
    for shard_path, out_path in shard_outputs:
        if fsspec_exists(f"{out_path}/.success"):
            logger.info("Skipping already-completed shard: %s", out_path)
        else:
            pending.append((shard_path, out_path))

    logger.info("%d shards pending (%d already done)", len(pending), len(shard_outputs) - len(pending))

    if not pending:
        logger.info("All shards already processed. Running consolidation only.")
    else:
        num_workers = min(config.num_workers, len(pending))
        logger.info("Processing %d shards with %d workers", len(pending), num_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(process_shard, shard_path, out_path, config.tokenizer, config.w_visual, config.generation_ratio): shard_path
                for shard_path, out_path in pending
            }

            completed = 0
            for future in as_completed(futures):
                shard_path = futures[future]
                try:
                    future.result()
                    completed += 1
                    if completed % 100 == 0:
                        logger.info("Progress: %d/%d shards completed", completed, len(pending))
                except Exception:
                    logger.exception("Failed to process shard %s", shard_path)
                    raise

    # Consolidate all shard caches
    all_shard_cache_paths = [out for _, out in shard_outputs]
    existing_paths = [p for p in all_shard_cache_paths if fsspec_exists(f"{p}/.success")]
    logger.info("Consolidating %d shard caches into %s/train", len(existing_paths), config.output_path)

    from levanter.store.cache import consolidate_shard_caches

    # Get exemplar from first shard

    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "loss_weights": np.zeros((0,), dtype=np.float32)}
    consolidate_shard_caches(
        shard_cache_paths=existing_paths,
        output_path=f"{config.output_path}/train",
        exemplar=exemplar,
    )
    logger.info("Consolidation complete.")


if __name__ == "__main__":
    main()
