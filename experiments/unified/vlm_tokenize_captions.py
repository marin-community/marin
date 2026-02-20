# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize pre-tokenized TokLIP image-caption data into unified sequences for pre-training.

Reads parquet shards from GCS containing TokLIP image tokens and captions,
produces dual-ordering (image-first + text-first) token sequences with per-token
loss weights, and writes them to Levanter cache format.

Supports two parquet schemas:
- **Messages format**: columns ``messages`` (chat-format) + ``image_tokens`` (nested list).
- **Caption format**: columns ``caption`` (plain string) + ``image_tokens`` (flat list).

Two processing modes:
- **Sequential** (default): 1:1 mapping from input parquet to output shard cache.
- **Shuffled** (``--shuffle true``): cross-source mixing via size-limited batching
  and proportional sampling. Each output shard receives proportional slices from
  every file in the batch, guaranteeing cross-source mixing. Controlled by
  ``--max_batch_gb`` (larger batches = more files = better mixing, more memory).

Requires the unified tokenizer to be created first via create_unified_tokenizer()
in experiments/unified/unified_pretrain.py.

Usage:
    # Sequential mode (1:1, default)
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --start_shard 0 --end_shard 10

    # Shuffled mode with cross-source mixing
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/hf_85m_tokenized \
        --output_path gs://marin-vlm/hf_85m_levanter_cache \
        --shuffle true --max_batch_gb 10 --rows_per_shard 8000 --seed 42 \
        --num_workers 32

    # Shuffled mode, small test run
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/hf_85m_tokenized \
        --output_path gs://marin-vlm/hf_85m_levanter_cache_test \
        --shuffle true --max_batch_gb 1 --rows_per_shard 8000 \
        --start_shard 0 --end_shard 100 --num_workers 16

    # Ratio-controlled: 30% generation (text-first), 70% understanding (image-first)
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --start_shard 0 --end_shard 100 \
        --dual_ordering false --generation_ratio 0.3

    # Custom visual loss weight and worker count
    uv run experiments/unified/vlm_tokenize_captions.py \
        --input_path gs://marin-vlm/stage2_sharded_full_tokenized \
        --start_shard 0 --end_shard 50 \
        --w_visual 0.3 --num_workers 16
"""

import dataclasses
import json
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
from tqdm import tqdm

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


def _collect_valid_indices(table) -> tuple[list[int], int]:
    """First pass over a parquet table: collect valid row indices.

    Returns (valid_indices, mismatched_count).
    """
    has_caption_col = "caption" in table.column_names
    image_tokens_col = table.column("image_tokens")

    valid_indices = []
    mismatched = 0

    for i in range(len(table)):
        image_tokens_raw = image_tokens_col[i].as_py()

        if has_caption_col:
            caption = table.column("caption")[i].as_py()
            if not caption or not image_tokens_raw:
                continue
        else:
            messages = table.column("messages")[i].as_py()
            image_token_lists = image_tokens_raw
            caption = extract_caption(messages)
            if caption is None or not image_token_lists:
                continue

            n_images = sum(1 for msg in messages for part in msg["content"] if part["type"] == "image")
            if n_images != len(image_token_lists):
                mismatched += 1
                continue

        valid_indices.append(i)

    return valid_indices, mismatched


def _split_train_val_indices(valid_indices: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    """Deterministically split valid indices into train and val sets.

    Every ``val_step``-th valid row goes to val. This is reproducible and
    independent of shard ordering or shuffle state.
    """
    if val_fraction <= 0.0:
        return valid_indices, []

    val_step = max(int(1.0 / val_fraction), 1)
    val_indices = [valid_indices[j] for j in range(0, len(valid_indices), val_step)]
    val_set = set(val_indices)
    train_indices = [i for i in valid_indices if i not in val_set]
    return train_indices, val_indices


def process_parquet_rows(
    table,
    tokenizer: transformers.PreTrainedTokenizer,
    w_visual: float,
    dual_ordering: bool,
    generation_ratio: float,
    val_fraction: float = 0.0,
    split: str = "train",
) -> Iterator[dict[str, np.ndarray]]:
    """Process rows in a PyArrow table, yielding token sequences.

    Supports two parquet schemas:
    - **Messages format**: columns ``messages`` (chat-format) + ``image_tokens`` (nested list).
    - **Caption format**: columns ``caption`` (plain string) + ``image_tokens`` (flat list).

    Args:
        split: Which rows and ordering to produce:
            - ``"train"``: train rows with dual_ordering / generation_ratio logic.
            - ``"val_understanding"``: val rows, image-first sequences only.
            - ``"val_generation"``: val rows, text-first sequences only.
        val_fraction: Fraction of valid rows to hold out for validation (0.0 to disable).
    """
    has_caption_col = "caption" in table.column_names
    has_messages_col = "messages" in table.column_names

    if not has_caption_col and not has_messages_col:
        raise ValueError(f"Parquet must have either 'caption' or 'messages' column, got: {table.column_names}")

    image_tokens_col = table.column("image_tokens")

    valid_indices, mismatched = _collect_valid_indices(table)
    train_indices, val_indices = _split_train_val_indices(valid_indices, val_fraction)

    if split == "train":
        row_indices = train_indices
    elif split in ("val_understanding", "val_generation"):
        row_indices = val_indices
    else:
        raise ValueError(f"Unknown split: {split!r}")

    skipped = len(table) - len(valid_indices) - mismatched
    logger.info(
        "Processing %d %s rows (%d total valid, %d train, %d val, %d skipped, %d mismatched)",
        len(row_indices),
        split,
        len(valid_indices),
        len(train_indices),
        len(val_indices),
        skipped,
        mismatched,
    )

    rng = random.Random(42)

    for i in row_indices:
        if has_caption_col:
            caption = table.column("caption")[i].as_py()
            raw_image_tokens = image_tokens_col[i].as_py()
        else:
            messages = table.column("messages")[i].as_py()
            caption = extract_caption(messages)
            raw_image_tokens = image_tokens_col[i].as_py()[0]

        shifted_image_tokens = [t + VISUAL_TOKEN_OFFSET for t in raw_image_tokens]
        caption_ids = tokenizer.encode(caption, add_special_tokens=False)

        if split == "val_understanding":
            yield build_image_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        elif split == "val_generation":
            yield build_text_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        elif dual_ordering:
            yield build_image_first_sequence(caption_ids, shifted_image_tokens, w_visual)
            yield build_text_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        elif rng.random() < generation_ratio:
            yield build_text_first_sequence(caption_ids, shifted_image_tokens, w_visual)
        else:
            yield build_image_first_sequence(caption_ids, shifted_image_tokens, w_visual)


# --- Shard processing (local files only, no GCS in workers) ---


def process_shard(
    local_parquet_path: str,
    local_output_path: str,
    tokenizer_path: str,
    w_visual: float,
    dual_ordering: bool,
    generation_ratio: float,
    source_shard: str,
    val_fraction: float = 0.0,
) -> dict:
    """Process a single parquet shard from local disk and write cache(s) to local disk.

    Writes up to 3 caches:
    - ``{local_output_path}`` — training records
    - ``{local_output_path}_val_und`` — val understanding (image-first) records
    - ``{local_output_path}_val_gen`` — val generation (text-first) records

    This function is designed to run in a worker process. It only touches local files.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    from zephyr.writers import write_levanter_cache

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    table = pq.read_table(local_parquet_path)

    metadata = {
        "source_shard": source_shard,
        "tokenizer": tokenizer_path,
        "visual_token_offset": VISUAL_TOKEN_OFFSET,
        "w_visual": w_visual,
        "dual_ordering": dual_ordering,
        "generation_ratio": generation_ratio,
        "format": "dual_ordering_pretraining",
    }

    # Train cache
    train_records = process_parquet_rows(
        table, tokenizer, w_visual, dual_ordering, generation_ratio, val_fraction, "train"
    )
    train_result = write_levanter_cache(train_records, local_output_path, metadata)
    logger.info(
        "Shard %s train: %d records, %d tokens", source_shard, train_result["count"], train_result["token_count"]
    )

    result = {**train_result, "val_und_path": None, "val_gen_path": None, "val_und_count": 0, "val_gen_count": 0}

    # Val caches (if val_fraction > 0)
    if val_fraction > 0:
        val_und_path = f"{local_output_path}_val_und"
        val_und_records = process_parquet_rows(
            table, tokenizer, w_visual, dual_ordering, generation_ratio, val_fraction, "val_understanding"
        )
        val_und_result = write_levanter_cache(val_und_records, val_und_path, {**metadata, "split": "val_understanding"})
        result["val_und_count"] = val_und_result["count"]
        if val_und_result["count"] > 0:
            result["val_und_path"] = val_und_path
            logger.info(
                "Shard %s val_understanding: %d records, %d tokens",
                source_shard,
                val_und_result["count"],
                val_und_result["token_count"],
            )

        val_gen_path = f"{local_output_path}_val_gen"
        val_gen_records = process_parquet_rows(
            table, tokenizer, w_visual, dual_ordering, generation_ratio, val_fraction, "val_generation"
        )
        val_gen_result = write_levanter_cache(val_gen_records, val_gen_path, {**metadata, "split": "val_generation"})
        result["val_gen_count"] = val_gen_result["count"]
        if val_gen_result["count"] > 0:
            result["val_gen_path"] = val_gen_path
            logger.info(
                "Shard %s val_generation: %d records, %d tokens",
                source_shard,
                val_gen_result["count"],
                val_gen_result["token_count"],
            )

    return result


def tokenize_shard(
    local_parquet_path: str,
    tokenizer_path: str,
    w_visual: float,
    dual_ordering: bool,
    generation_ratio: float,
    val_fraction: float = 0.0,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """Tokenize a single parquet shard and return records without writing a cache.

    Used by the shuffle path: workers tokenize in parallel, main process
    collects results into a shared buffer for cross-source mixing.

    Returns:
        (train_records, val_understanding_records, val_generation_records)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    table = pq.read_table(local_parquet_path)

    train_records = list(
        process_parquet_rows(table, tokenizer, w_visual, dual_ordering, generation_ratio, val_fraction, "train")
    )
    val_und_records: list[dict[str, np.ndarray]] = []
    val_gen_records: list[dict[str, np.ndarray]] = []

    if val_fraction > 0:
        val_und_records = list(
            process_parquet_rows(
                table, tokenizer, w_visual, dual_ordering, generation_ratio, val_fraction, "val_understanding"
            )
        )
        val_gen_records = list(
            process_parquet_rows(
                table, tokenizer, w_visual, dual_ordering, generation_ratio, val_fraction, "val_generation"
            )
        )

    return train_records, val_und_records, val_gen_records


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
    shuffle: bool = False
    """Enable cross-source shuffling. When True, rows from multiple input files
    are mixed together in each output shard instead of the default 1:1 mapping."""
    rows_per_shard: int = 8000
    """Number of records per output shard (only used when shuffle=True)."""
    max_batch_gb: float = 10.0
    """Maximum total parquet file size per batch in GB (only used when shuffle=True).
    Larger batches mix more files per proportional-sampling round → better cross-source mixing,
    but use more memory (~6x expansion from parquet to tokenized records in RAM)."""
    seed: int = 42
    """Random seed for shuffling."""
    download_dir: str = "/dev/shm"
    """Directory for temporary parquet downloads (default: /dev/shm for faster I/O)."""
    staging_dir: str = "/run/shards"
    """Directory for staging output shard caches before uploading to GCS."""
    val_fraction: float = 0.005
    """Fraction of valid rows to hold out for validation (0.0 to disable).
    Val rows produce separate understanding (image-first) and generation (text-first)
    caches at {output_path}/val_understanding/validation/ and
    {output_path}/val_generation/validation/."""


def _cache_tokenizer_locally(tokenizer_path: str, download_dir: str = "/dev/shm") -> str:
    """Download a (possibly GCS-hosted) tokenizer to a local directory.

    Returns the local directory path.
    """
    from levanter.compat.hf_checkpoints import load_tokenizer

    os.makedirs(download_dir, exist_ok=True)
    local_dir = tempfile.mkdtemp(prefix="tokenizer_cache_", dir=download_dir)
    tok = load_tokenizer(tokenizer_path)
    tok.save_pretrained(local_dir)
    logger.info("Tokenizer cached locally at %s (vocab_size=%d)", local_dir, len(tok))
    return local_dir


@draccus.wrap()
def main(config: TokenizeVLMConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    shard_paths = _list_shard_paths(config.input_path, config.start_shard, config.end_shard)

    if config.shuffle:
        _main_shuffled(config, shard_paths)
    else:
        _main_sequential(config, shard_paths)


def _write_metadata(config: "TokenizeVLMConfig", stats: dict) -> None:
    """Write a JSON metadata summary to {output_path}/metadata.json."""
    meta = {
        "input_path": config.input_path,
        "output_path": config.output_path,
        "shuffle": config.shuffle,
        "dual_ordering": config.dual_ordering,
        "generation_ratio": config.generation_ratio,
        "w_visual": config.w_visual,
        "val_fraction": config.val_fraction,
        **stats,
    }
    logger.info("=== Processing Summary ===")
    for key, val in stats.items():
        logger.info("  %-30s %s", key, val)

    meta_path = f"{config.output_path}/metadata.json"
    staging = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(meta, staging, indent=2)
        staging.close()
        gcs_upload(staging.name, meta_path if meta_path.startswith("gs://") else f"gs://{meta_path}")
        logger.info("Metadata written to %s", meta_path)
    finally:
        os.unlink(staging.name)


def _consolidate(paths: list[str], output_path: str, label: str) -> None:
    """Consolidate shard caches into a single output path."""
    from levanter.store.cache import consolidate_shard_caches

    existing = [p for p in paths if fsspec_exists(f"{p}/.success")]
    if not existing:
        logger.info("No %s shard caches to consolidate.", label)
        return
    logger.info("Consolidating %d %s shard caches into %s", len(existing), label, output_path)
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "loss_weights": np.zeros((0,), dtype=np.float32)}
    consolidate_shard_caches(shard_cache_paths=existing, output_path=output_path, exemplar=exemplar)
    logger.info("%s consolidation complete.", label.capitalize())


def _main_sequential(config: TokenizeVLMConfig, shard_paths: list[str]):
    """Original 1:1 processing: each input parquet → one output shard cache."""
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

    # Track val shard paths for consolidation and counts
    val_und_gcs_paths: list[str] = []
    val_gen_gcs_paths: list[str] = []
    total_train_records = 0
    total_train_tokens = 0
    total_val_und_records = 0
    total_val_gen_records = 0

    if not pending:
        logger.info("All shards already processed. Running consolidation only.")
    else:
        # Pre-cache tokenizer locally
        local_tokenizer_path = _cache_tokenizer_locally(config.tokenizer, config.download_dir)

        num_workers = min(config.num_workers, len(pending))
        chunk_size = num_workers
        num_chunks = math.ceil(len(pending) / chunk_size)
        logger.info(
            "Processing %d shards in %d chunks (%d workers per chunk), download_dir=%s, val_fraction=%.4f",
            len(pending),
            num_chunks,
            num_workers,
            config.download_dir,
            config.val_fraction,
        )

        os.makedirs(config.download_dir, exist_ok=True)
        os.makedirs(config.staging_dir, exist_ok=True)
        total_completed = 0
        for chunk_idx in tqdm(range(num_chunks), desc="Chunks", unit="chunk"):
            chunk = pending[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            chunk_label = f"Chunk {chunk_idx + 1}/{num_chunks}"

            # Parquet downloads go to download_dir, cache output goes to staging_dir
            parquet_work_dir = tempfile.mkdtemp(prefix=f"vlm_chunk{chunk_idx}_", dir=config.download_dir)
            local_parquet_dir = os.path.join(parquet_work_dir, "parquets")
            cache_work_dir = tempfile.mkdtemp(prefix=f"vlm_cache{chunk_idx}_", dir=config.staging_dir)
            local_cache_dir = os.path.join(cache_work_dir, "caches")
            os.makedirs(local_parquet_dir)
            os.makedirs(local_cache_dir)

            # Step 1: Download this chunk's parquet shards from GCS
            logger.info("[%s] Downloading %d shards ...", chunk_label, len(chunk))
            chunk_jobs = []
            for shard_path, gcs_out_path in tqdm(chunk, desc="Downloading", unit="file", leave=False):
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
                        config.val_fraction,
                    ): (local_cache, source_shard)
                    for local_parquet, local_cache, _gcs_out, source_shard in chunk_jobs
                }

                for future in tqdm(
                    as_completed(futures),
                    desc=f"Processing ({total_completed}/{len(pending)})",
                    total=len(futures),
                    unit="shard",
                    leave=False,
                ):
                    local_cache, source = futures[future]
                    try:
                        result = future.result()
                        total_completed += 1
                        total_train_records += result["count"]
                        total_train_tokens += result["token_count"]
                        total_val_und_records += result.get("val_und_count", 0)
                        total_val_gen_records += result.get("val_gen_count", 0)

                        # Track val cache paths
                        if result.get("val_und_path"):
                            shard_name = local_cache.rsplit("/", 1)[-1]
                            val_und_gcs = f"{config.output_path}/val_understanding/validation/{shard_name}_val_und"
                            val_und_gcs_paths.append(val_und_gcs)
                        if result.get("val_gen_path"):
                            shard_name = local_cache.rsplit("/", 1)[-1]
                            val_gen_gcs = f"{config.output_path}/val_generation/validation/{shard_name}_val_gen"
                            val_gen_gcs_paths.append(val_gen_gcs)
                    except Exception:
                        logger.exception("Failed to process shard %s", source)
                        raise

            # Step 3: Upload results to GCS (train + val caches)
            logger.info("[%s] Uploading shard caches ...", chunk_label)
            for _local_parquet, local_cache, gcs_out_path, _source in tqdm(
                chunk_jobs, desc="Uploading", unit="shard", leave=False
            ):
                # Upload train cache
                gcs_dest = gcs_out_path if gcs_out_path.startswith("gs://") else f"gs://{gcs_out_path}"
                gcs_upload(local_cache, gcs_dest)

                # Upload val caches if they exist
                val_und_local = f"{local_cache}_val_und"
                if os.path.isdir(val_und_local):
                    shard_name = local_cache.rsplit("/", 1)[-1]
                    val_und_gcs = f"{config.output_path}/val_understanding/validation/{shard_name}_val_und"
                    gcs_upload(val_und_local, val_und_gcs if val_und_gcs.startswith("gs://") else f"gs://{val_und_gcs}")

                val_gen_local = f"{local_cache}_val_gen"
                if os.path.isdir(val_gen_local):
                    shard_name = local_cache.rsplit("/", 1)[-1]
                    val_gen_gcs = f"{config.output_path}/val_generation/validation/{shard_name}_val_gen"
                    gcs_upload(val_gen_local, val_gen_gcs if val_gen_gcs.startswith("gs://") else f"gs://{val_gen_gcs}")

            # Step 4: Clean up local files for this chunk
            shutil.rmtree(parquet_work_dir)
            shutil.rmtree(cache_work_dir)
            logger.info("[%s] Done and cleaned up.", chunk_label)

    # Consolidate all shard caches
    all_train_paths = [out for _, out in shard_outputs]
    _consolidate(all_train_paths, f"{config.output_path}/train", "train")

    if config.val_fraction > 0:
        _consolidate(
            val_und_gcs_paths,
            f"{config.output_path}/val_understanding/validation",
            "val_understanding",
        )
        _consolidate(
            val_gen_gcs_paths,
            f"{config.output_path}/val_generation/validation",
            "val_generation",
        )

    _write_metadata(
        config,
        {
            "mode": "sequential",
            "total_shards": len(shard_paths),
            "train_records": total_train_records,
            "train_tokens": total_train_tokens,
            "val_understanding_records": total_val_und_records,
            "val_generation_records": total_val_gen_records,
        },
    )


def _main_shuffled(config: TokenizeVLMConfig, shard_paths: list[str]):
    """Cross-source shuffled processing with proportional sampling.

    Follows the same approach as convert_llava_onevision_to_levanter.py:
    1. File-level shuffle: randomize input file order for cross-source mixing
    2. Size-limited batching: group files into batches capped at max_batch_gb
    3. Per-file record shuffle: shuffle tokenized records within each file
    4. Proportional distribution: each output shard receives a proportional
       slice from every file in the batch, guaranteeing cross-source mixing
    5. Shard-level shuffle: shuffle each output shard buffer before writing

    Val records (~0.5% of data) are accumulated in memory across batches and
    written as single caches at the end.
    """
    import fsspec as _fsspec

    from zephyr.writers import write_levanter_cache

    random.seed(config.seed)
    random.shuffle(shard_paths)
    logger.info("Shuffled %d input files (seed=%d)", len(shard_paths), config.seed)

    # --- Get file sizes and create size-limited batches ---
    logger.info("Querying file sizes for %d files ...", len(shard_paths))
    fs = _fsspec.filesystem("gs")
    files_with_sizes: list[tuple[str, int]] = []
    for path in shard_paths:
        gcs_path = path if path.startswith("gs://") else f"gs://{path}"
        info = fs.info(gcs_path)
        files_with_sizes.append((path, info["size"]))

    max_batch_bytes = int(config.max_batch_gb * 1024 * 1024 * 1024)
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_size = 0
    for path, size in files_with_sizes:
        if current_size + size > max_batch_bytes and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(path)
        current_size += size
    if current_batch:
        batches.append(current_batch)

    total_gb = sum(s for _, s in files_with_sizes) / (1024**3)
    logger.info(
        "Created %d batches (max %.1f GB each) from %.1f GB total",
        len(batches),
        config.max_batch_gb,
        total_gb,
    )

    # --- Setup ---
    os.makedirs(config.download_dir, exist_ok=True)
    os.makedirs(config.staging_dir, exist_ok=True)
    local_tokenizer_path = _cache_tokenizer_locally(config.tokenizer, config.download_dir)

    metadata = {
        "tokenizer": config.tokenizer,
        "visual_token_offset": VISUAL_TOKEN_OFFSET,
        "w_visual": config.w_visual,
        "dual_ordering": config.dual_ordering,
        "generation_ratio": config.generation_ratio,
        "shuffle": True,
        "seed": config.seed,
        "rows_per_shard": config.rows_per_shard,
        "format": "shuffled_pretraining",
    }

    num_workers = min(config.num_workers, len(shard_paths))
    shard_idx = 0
    all_output_paths: list[str] = []
    total_records = 0

    # Accumulate val records across batches (small — ~0.5% of data)
    all_val_und_records: list[dict[str, np.ndarray]] = []
    all_val_gen_records: list[dict[str, np.ndarray]] = []

    logger.info(
        "Shuffle mode: %d files in %d batches, rows_per_shard=%d, num_workers=%d, val_fraction=%.4f",
        len(shard_paths),
        len(batches),
        config.rows_per_shard,
        num_workers,
        config.val_fraction,
    )

    # --- Process each batch ---
    for batch_idx, batch in tqdm(enumerate(batches), desc="Batches", total=len(batches), unit="batch"):
        batch_label = f"Batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)"
        file_records: dict[str, list[dict[str, np.ndarray]]] = {}

        # Download and tokenize in sub-chunks of num_workers
        num_sub_chunks = math.ceil(len(batch) / num_workers)
        for sub_idx in range(num_sub_chunks):
            sub_chunk = batch[sub_idx * num_workers : (sub_idx + 1) * num_workers]
            sub_label = f"{batch_label} sub {sub_idx + 1}/{num_sub_chunks}"

            # Download sub-chunk to /dev/shm (or configured download_dir)
            work_dir = tempfile.mkdtemp(prefix=f"vlm_batch{batch_idx}_sub{sub_idx}_", dir=config.download_dir)
            local_parquet_dir = os.path.join(work_dir, "parquets")
            os.makedirs(local_parquet_dir)

            local_parquets = []
            for shard_path in tqdm(sub_chunk, desc="Downloading", unit="file", leave=False):
                gcs_shard = shard_path if shard_path.startswith("gs://") else f"gs://{shard_path}"
                filename = shard_path.rsplit("/", 1)[-1]
                local_parquet = os.path.join(local_parquet_dir, filename)
                gcs_download(gcs_shard, local_parquet)
                local_parquets.append(local_parquet)

            # Tokenize in parallel
            logger.info("[%s] Tokenizing %d files ...", sub_label, len(local_parquets))
            with ProcessPoolExecutor(max_workers=min(num_workers, len(local_parquets))) as pool:
                futures = {
                    pool.submit(
                        tokenize_shard,
                        lp,
                        local_tokenizer_path,
                        config.w_visual,
                        config.dual_ordering,
                        config.generation_ratio,
                        config.val_fraction,
                    ): lp
                    for lp in local_parquets
                }

                for future in tqdm(
                    as_completed(futures), desc="Tokenizing", total=len(futures), unit="file", leave=False
                ):
                    source = futures[future]
                    try:
                        train_records, val_und, val_gen = future.result()
                        random.shuffle(train_records)
                        file_records[source] = train_records
                        all_val_und_records.extend(val_und)
                        all_val_gen_records.extend(val_gen)
                    except Exception:
                        logger.exception("Failed to tokenize %s", source)
                        raise

            # Free disk immediately
            shutil.rmtree(work_dir)

        # --- Proportional distribution across output shards ---
        batch_total = sum(len(r) for r in file_records.values())
        total_records += batch_total
        num_shards_in_batch = math.ceil(batch_total / config.rows_per_shard)

        logger.info(
            "[%s] Distributing %d records from %d files across %d shards",
            batch_label,
            batch_total,
            len(file_records),
            num_shards_in_batch,
        )

        shard_buffers: list[list[dict[str, np.ndarray]]] = [[] for _ in range(num_shards_in_batch)]

        for _source_path, records in file_records.items():
            n = len(records)
            if n == 0:
                continue
            a_i = n / num_shards_in_batch

            for shard_k in range(num_shards_in_batch):
                start = int(shard_k * a_i)
                end = int((shard_k + 1) * a_i) if shard_k < num_shards_in_batch - 1 else n
                shard_buffers[shard_k].extend(records[start:end])

            records.clear()

        file_records.clear()

        # --- Shuffle each shard buffer, write locally, then upload ---
        logger.info("[%s] Shuffling and writing %d shards ...", batch_label, num_shards_in_batch)
        staging_batch_dir = tempfile.mkdtemp(prefix=f"vlm_shards_batch{batch_idx}_", dir=config.staging_dir)
        for buf in tqdm(shard_buffers, desc="Writing shards", unit="shard", leave=False):
            if not buf:
                continue
            random.shuffle(buf)
            shard_name = f"shard-{shard_idx:06d}"
            local_out = os.path.join(staging_batch_dir, shard_name)
            write_levanter_cache(iter(buf), local_out, metadata)

            gcs_out = f"{config.output_path}/train/{shard_name}"
            gcs_upload(local_out, gcs_out if gcs_out.startswith("gs://") else f"gs://{gcs_out}")
            all_output_paths.append(gcs_out)
            logger.info("Wrote and uploaded shard %06d (%d records)", shard_idx, len(buf))
            shard_idx += 1
            buf.clear()

        del shard_buffers
        shutil.rmtree(staging_batch_dir)
        logger.info("[%s] Done. Total records so far: %d", batch_label, total_records)

    logger.info("Wrote %d shuffled shards (%d total records)", len(all_output_paths), total_records)

    # --- Write val caches ---
    val_und_gcs_paths: list[str] = []
    val_gen_gcs_paths: list[str] = []
    total_val_und = len(all_val_und_records)
    total_val_gen = len(all_val_gen_records)

    if config.val_fraction > 0 and (all_val_und_records or all_val_gen_records):
        logger.info(
            "Writing val caches: %d understanding, %d generation records",
            total_val_und,
            total_val_gen,
        )
        staging_val_dir = tempfile.mkdtemp(prefix="vlm_val_", dir=config.staging_dir)

        if all_val_und_records:
            local_val_und = os.path.join(staging_val_dir, "val_und")
            write_levanter_cache(iter(all_val_und_records), local_val_und, {**metadata, "split": "val_understanding"})
            gcs_val_und = f"{config.output_path}/val_understanding/validation/shard-000000"
            gcs_upload(local_val_und, gcs_val_und if gcs_val_und.startswith("gs://") else f"gs://{gcs_val_und}")
            val_und_gcs_paths.append(gcs_val_und)

        if all_val_gen_records:
            local_val_gen = os.path.join(staging_val_dir, "val_gen")
            write_levanter_cache(iter(all_val_gen_records), local_val_gen, {**metadata, "split": "val_generation"})
            gcs_val_gen = f"{config.output_path}/val_generation/validation/shard-000000"
            gcs_upload(local_val_gen, gcs_val_gen if gcs_val_gen.startswith("gs://") else f"gs://{gcs_val_gen}")
            val_gen_gcs_paths.append(gcs_val_gen)

        shutil.rmtree(staging_val_dir)
        all_val_und_records.clear()
        all_val_gen_records.clear()

    # Consolidate all shard caches
    _consolidate(all_output_paths, f"{config.output_path}/train", "train")

    if config.val_fraction > 0:
        _consolidate(val_und_gcs_paths, f"{config.output_path}/val_understanding/validation", "val_understanding")
        _consolidate(val_gen_gcs_paths, f"{config.output_path}/val_generation/validation", "val_generation")

    _write_metadata(
        config,
        {
            "mode": "shuffled",
            "total_input_files": len(shard_paths),
            "total_batches": len(batches),
            "train_records": total_records,
            "train_shards": len(all_output_paths),
            "val_understanding_records": total_val_und,
            "val_generation_records": total_val_gen,
        },
    )


if __name__ == "__main__":
    main()
