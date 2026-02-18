# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize pre-tokenized TokLIP image-caption data into unified sequences for pre-training.

Reads parquet shards from GCS containing TokLIP image tokens and captions,
produces dual-ordering (image-first + text-first) token sequences with per-token
loss weights, and writes them to Levanter cache format.

Requires the unified tokenizer to be created first via create_unified_tokenizer()
in experiments/unified/unified_pretrain.py.

Usage:
    # Dual ordering (default): both image-first and text-first per row
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --start_shard 0 --end_shard 10

    # Process all shards with dual ordering
    uv run experiments/unified/vlm_tokenize_captions.py

    # Custom input parquet path
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --output_path gs://marin-vlm/stage2_sharded_full_tokenized_llama3 \
        --start_shard 0 --end_shard 10 --dual_ordering false

    # Ratio-controlled: 30% generation (text-first), 70% understanding (image-first)
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --start_shard 0 --end_shard 100 \
        --dual_ordering false --generation_ratio 0.3

    # All understanding (image-first only)
    uv run experiments/unified/vlm_tokenize_captions.py \
        --dual_ordering false --generation_ratio 0.0

    # All generation (text-first only)
    uv run experiments/unified/vlm_tokenize_captions.py \
        --dual_ordering false --generation_ratio 1.0

    # Custom visual loss weight and worker count
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --start_shard 0 --end_shard 50 \
        --w_visual 0.3 --num_workers 16
"""

import dataclasses
import logging
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

import draccus
import numpy as np
import pyarrow.parquet as pq
import transformers

from experiments.unified.unified_pretrain import UNIFIED_CACHE_PATH, UNIFIED_TOKENIZER_PATH
from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)

# --- Constants (Llama3 base tokenizer + TokLIP-L visual tokens) ---

VISUAL_TOKEN_OFFSET = 128_256  # Llama3 vocab size; TokLIP index c → unified ID c + 128256
VISION_START_ID = 128_004  # <|vision_start|> (repurposed reserved_special_token_2)
VISION_END_ID = 128_005  # <|vision_end|> (repurposed reserved_special_token_3)
ENDOFTEXT_ID = 128_001  # <|end_of_text|> (Llama3 EOS)


# --- GCS helpers (subprocess-based, no gcsfs) ---


def gcs_download(remote_path: str, local_path: str) -> None:
    """Download a file from GCS using gcloud storage cp."""
    result = subprocess.run(
        ["gcloud", "storage", "cp", "--quiet", remote_path, local_path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcloud storage cp failed: {result.stderr.strip()}")


def gcs_upload(local_path: str, remote_path: str) -> None:
    """Upload a file or directory to GCS using gcloud storage cp."""
    cmd = ["gcloud", "storage", "cp", "--quiet"]
    if os.path.isdir(local_path):
        cmd.append("-r")
        # Ensure trailing slash so gcloud copies contents correctly
        if not local_path.endswith("/"):
            local_path += "/"
    cmd.extend([local_path, remote_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"gcloud storage cp upload failed: {result.stderr.strip()}")


# --- Sequence building ---


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
    dual_ordering: bool,
    generation_ratio: float,
) -> Iterator[dict[str, np.ndarray]]:
    """Process all rows in a PyArrow table, yielding token sequences.

    Supports two parquet schemas:
    - **Messages format**: columns ``messages`` (chat-format) + ``image_tokens`` (nested list).
    - **Caption format**: columns ``caption`` (plain string) + ``image_tokens`` (flat list).

    When dual_ordering is True, every valid row produces both an image-first
    (understanding) and a text-first (generation) sequence.

    When dual_ordering is False, each row produces a single sequence. The first
    (1 - generation_ratio) fraction of valid rows yield image-first sequences;
    the remaining generation_ratio fraction yield text-first sequences.
    """
    # Auto-detect parquet schema
    has_caption_col = "caption" in table.column_names
    has_messages_col = "messages" in table.column_names

    if not has_caption_col and not has_messages_col:
        raise ValueError(f"Parquet must have either 'caption' or 'messages' column, got: {table.column_names}")

    image_tokens_col = table.column("image_tokens")
    if has_caption_col:
        caption_col = table.column("caption")
    else:
        messages_col = table.column("messages")

    # First pass: collect valid row indices
    valid_indices = []
    mismatched = 0
    for i in range(len(table)):
        image_tokens_raw = image_tokens_col[i].as_py()

        if has_caption_col:
            caption = caption_col[i].as_py()
            if not caption or not image_tokens_raw:
                continue
        else:
            messages = messages_col[i].as_py()
            image_token_lists = image_tokens_raw
            caption = extract_caption(messages)
            if caption is None or not image_token_lists:
                continue

            n_images = sum(1 for msg in messages for part in msg["content"] if part["type"] == "image")
            if n_images != len(image_token_lists):
                mismatched += 1
                continue

        valid_indices.append(i)

    skipped = len(table) - len(valid_indices) - mismatched
    n_valid = len(valid_indices)

    logger.info(
        "Processing %d valid rows (%d skipped, %d mismatched image/token count, generation_ratio=%.2f)",
        n_valid,
        skipped,
        mismatched,
        generation_ratio,
    )

    rng = random.Random(42)

    for rank, i in enumerate(valid_indices):
        if has_caption_col:
            caption = caption_col[i].as_py()
            raw_image_tokens = image_tokens_col[i].as_py()
        else:
            messages = messages_col[i].as_py()
            caption = extract_caption(messages)
            # Use the first image's tokens (all examples in this dataset are single-image)
            raw_image_tokens = image_tokens_col[i].as_py()[0]

        shifted_image_tokens = [t + VISUAL_TOKEN_OFFSET for t in raw_image_tokens]

        # Tokenize caption (no BOS per doc spec)
        caption_ids = tokenizer.encode(caption, add_special_tokens=False)

        if dual_ordering:
            yield build_image_first_sequence(caption_ids, shifted_image_tokens, w_visual)
            yield build_text_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        elif rng.random() < generation_ratio:
            yield build_text_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        else:
            yield build_image_first_sequence(caption_ids, shifted_image_tokens, w_visual)

        if (rank + 1) % 1000 == 0:
            logger.info("  ... processed %d/%d rows", rank + 1, n_valid)


# --- Shard processing (local files only, no GCS in workers) ---


def process_shard(
    local_parquet_path: str,
    local_output_path: str,
    tokenizer_path: str,
    w_visual: float,
    dual_ordering: bool,
    generation_ratio: float,
    source_shard: str,
) -> dict:
    """Process a single parquet shard from local disk and write cache to local disk.

    This function is designed to run in a worker process. It only touches local files.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    from zephyr.writers import write_levanter_cache

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    table = pq.read_table(local_parquet_path)

    records = process_parquet_rows(table, tokenizer, w_visual, dual_ordering, generation_ratio)
    metadata = {
        "source_shard": source_shard,
        "tokenizer": tokenizer_path,
        "visual_token_offset": VISUAL_TOKEN_OFFSET,
        "w_visual": w_visual,
        "dual_ordering": dual_ordering,
        "generation_ratio": generation_ratio,
        "format": "dual_ordering_pretraining",
    }

    result = write_levanter_cache(records, local_output_path, metadata)
    logger.info("Shard %s: %d records, %d tokens", source_shard, result["count"], result["token_count"])
    return result


# --- Shard listing ---


def _list_shard_paths(input_path: str, start_shard: int | None, end_shard: int | None) -> list[str]:
    """List parquet shard paths, optionally filtering by index range."""
    all_paths = sorted(fsspec_glob(f"{input_path}/*.parquet"))
    if not all_paths:
        raise ValueError(f"No parquet shards found at {input_path}")
    logger.info("Found %d total shards at %s", len(all_paths), input_path)

    if start_shard is not None or end_shard is not None:
        start = start_shard or 0
        end = end_shard if end_shard is not None else len(all_paths)
        all_paths = all_paths[start:end]
        logger.info("Selected shards [%d, %d) (%d shards)", start, end, len(all_paths))

    return all_paths


# --- Config and main ---


@dataclasses.dataclass
class TokenizeVLMConfig:
    input_path: str = "gs://marin-vlm/stage2_sharded_full_tokenized"
    output_path: str = UNIFIED_CACHE_PATH
    tokenizer: str = UNIFIED_TOKENIZER_PATH
    w_visual: float = 0.5
    dual_ordering: bool = True
    generation_ratio: float = 0.5
    num_workers: int = 32
    start_shard: int | None = None
    end_shard: int | None = None


def _cache_tokenizer_locally(tokenizer_path: str) -> str:
    """Download a (possibly GCS-hosted) tokenizer to a local directory.

    Returns the local directory path.
    """
    from levanter.compat.hf_checkpoints import load_tokenizer

    local_dir = tempfile.mkdtemp(prefix="tokenizer_cache_")
    tok = load_tokenizer(tokenizer_path)
    tok.save_pretrained(local_dir)
    logger.info("Tokenizer cached locally (vocab_size=%d)", len(tok))
    return local_dir


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
        if not fsspec_exists(f"{out_path}/.success"):
            pending.append((shard_path, out_path))

    logger.info("%d shards pending (%d already done)", len(pending), len(shard_outputs) - len(pending))

    if not pending:
        logger.info("All shards already processed. Running consolidation only.")
    else:
        # Pre-cache tokenizer locally
        local_tokenizer_path = _cache_tokenizer_locally(config.tokenizer)

        num_workers = min(config.num_workers, len(pending))
        chunk_size = num_workers
        num_chunks = math.ceil(len(pending) / chunk_size)
        logger.info("Processing %d shards in %d chunks (%d workers per chunk)", len(pending), num_chunks, num_workers)

        total_completed = 0
        for chunk_idx in range(num_chunks):
            chunk = pending[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            chunk_label = f"Chunk {chunk_idx + 1}/{num_chunks}"

            # Create per-chunk work directory (cleaned up after each chunk)
            work_dir = tempfile.mkdtemp(prefix=f"vlm_chunk{chunk_idx}_")
            local_parquet_dir = os.path.join(work_dir, "parquets")
            local_cache_dir = os.path.join(work_dir, "caches")
            os.makedirs(local_parquet_dir)
            os.makedirs(local_cache_dir)

            # Step 1: Download this chunk's parquet shards from GCS
            logger.info("[%s] Downloading %d shards ...", chunk_label, len(chunk))
            chunk_jobs = []
            for shard_path, gcs_out_path in chunk:
                gcs_shard = shard_path if shard_path.startswith("gs://") else f"gs://{shard_path}"
                filename = shard_path.rsplit("/", 1)[-1]
                local_parquet = os.path.join(local_parquet_dir, filename)
                shard_name = filename.replace(".parquet", "")
                local_cache = os.path.join(local_cache_dir, f"part-{shard_name}")

                gcs_download(gcs_shard, local_parquet)
                chunk_jobs.append((local_parquet, local_cache, gcs_out_path, shard_path))

            # Step 2: Process in parallel (workers only touch local files)
            logger.info("[%s] Processing %d shards with %d workers ...", chunk_label, len(chunk_jobs), num_workers)
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        process_shard,
                        local_parquet,
                        local_cache,
                        local_tokenizer_path,
                        config.w_visual,
                        config.dual_ordering,
                        config.generation_ratio,
                        source_shard,
                    ): source_shard
                    for local_parquet, local_cache, _gcs_out, source_shard in chunk_jobs
                }

                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        future.result()
                        total_completed += 1
                        logger.info("[%s] Shard done (%d/%d total)", chunk_label, total_completed, len(pending))
                    except Exception:
                        logger.exception("Failed to process shard %s", source)
                        raise

            # Step 3: Upload results to GCS
            logger.info("[%s] Uploading %d shard caches ...", chunk_label, len(chunk_jobs))
            for _local_parquet, local_cache, gcs_out_path, _source in chunk_jobs:
                gcs_dest = gcs_out_path if gcs_out_path.startswith("gs://") else f"gs://{gcs_out_path}"
                gcs_upload(local_cache, gcs_dest)

            # Step 4: Clean up local files for this chunk
            shutil.rmtree(work_dir)
            logger.info("[%s] Done and cleaned up.", chunk_label)

    # Consolidate all shard caches
    all_shard_cache_paths = [out for _, out in shard_outputs]
    existing_paths = [p for p in all_shard_cache_paths if fsspec_exists(f"{p}/.success")]
    logger.info("Consolidating %d shard caches into %s/train", len(existing_paths), config.output_path)

    from levanter.store.cache import consolidate_shard_caches

    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "loss_weights": np.zeros((0,), dtype=np.float32)}
    consolidate_shard_caches(
        shard_cache_paths=existing_paths,
        output_path=f"{config.output_path}/train",
        exemplar=exemplar,
    )
    logger.info("Consolidation complete.")


if __name__ == "__main__":
    main()
