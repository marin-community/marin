# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer sweep for Marin issue #5821 on the 2026-05-26 Datakit testbed.

This builds proportional windows from the 1.1T-token normalized testbed sample:

* randomized 50B-token-equivalent window: tokenizer-training corpus
* window [100B, 200B): holdout corpus to tokenize with the trained vocabularies
* optional window [0, 50B): train corpus retokenization for downstream runs

The Llama and GPT-OSS HF-family tokenizers are initialized from their upstream
tokenizer repositories and trained at 262k, then derived to 128k, 32k, and 8k
using the same training corpus and trainer configuration. TokenMonster uses its
native binary training/export flow at 262k, then derives smaller vocabularies
with ``exportvocab -resize``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import tempfile
import unicodedata
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import draccus
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from huggingface_hub import hf_hub_download
from levanter.tokenizers import TokenizerBackend
from marin.datakit.normalize import NormalizedData
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, InputName, executor_main, this_output_path, versioned
import marin.execution.executor as executor_module
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep
from marin.utils import fsspec_glob, fsspec_mkdirs
from rigging.filesystem import open_url, url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-eu-west4"
NORMALIZED_BASE = "gs://marin-eu-west4/data/datakit/sample/2026-05-26"
TOTAL_TOKENIZED_TOKENS = 1_099_611_681_172
WINDOW_TOKENS = 100_000_000_000
WINDOW_FRACTION = WINDOW_TOKENS / TOTAL_TOKENIZED_TOKENS
TOKENIZER_TRAIN_TOKENS = int(os.environ.get("TOKENIZER_SWEEP_TOKENIZER_TRAIN_TOKENS", "50000000000"))
TOKENIZER_TRAIN_FRACTION = TOKENIZER_TRAIN_TOKENS / TOTAL_TOKENIZED_TOKENS
RETOKENIZE_TRAIN_TOKENS = int(os.environ.get("TOKENIZER_SWEEP_RETOKENIZE_TRAIN_TOKENS", "50000000000"))
RETOKENIZE_TRAIN_START_TOKENS = int(os.environ.get("TOKENIZER_SWEEP_RETOKENIZE_TRAIN_START_TOKENS", "0"))
RETOKENIZE_TRAIN_FRACTION = RETOKENIZE_TRAIN_TOKENS / TOTAL_TOKENIZED_TOKENS
RETOKENIZE_TRAIN_START_FRACTION = RETOKENIZE_TRAIN_START_TOKENS / TOTAL_TOKENIZED_TOKENS

RUN_ID = "tokenizer-sweep-20260526"
MAX_STEP_CONCURRENCY = int(os.environ.get("TOKENIZER_SWEEP_MAX_STEP_CONCURRENCY", "24"))
VOCAB_SIZES = (262_144, 131_072, 32_768, 8_192)
HF_TOKENIZER_FAMILIES = {
    "gpt-oss": ("openai/gpt-oss-20b", False),
    "llama": ("meta-llama/Meta-Llama-3.1-8B", False),
    "gpt-oss-place-digits": ("openai/gpt-oss-20b", True),
    "llama-place-digits": ("meta-llama/Meta-Llama-3.1-8B", True),
}
OFFICIAL_TRUNCATED_TOKENIZER_FAMILIES = {
    "llama-official": "meta-llama/Meta-Llama-3.1-8B",
}
PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS = 510
PLACE_ALIGNED_DIGIT_CHUNK_SIZE = 3
PLACE_ALIGNED_DIGIT_PRETOKENIZER_REVISION = "bounded-leading-triplets-v2"


def _env_regions(name: str, default: str) -> tuple[str, ...]:
    return tuple(region.strip() for region in os.environ.get(name, default).split(",") if region.strip())


def _token_count_label(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    return str(tokens)


RETOKENIZE_TRAIN_LABEL = os.environ.get(
    "TOKENIZER_SWEEP_RETOKENIZE_TRAIN_LABEL",
    f"train{_token_count_label(RETOKENIZE_TRAIN_TOKENS)}",
)


TOKENIZER_SWEEP_RESOURCE_REVISION = os.environ.get(
    "TOKENIZER_SWEEP_RESOURCE_REVISION",
    "regional-highmem-v2-tokenmonster-oom",
)
TOKENIZER_SWEEP_REGIONS = _env_regions("TOKENIZER_SWEEP_REGIONS", "europe-west4")
TOKENIZER_SWEEP_TM_CORPUS_BATCH_SIZE = int(os.environ.get("TOKENIZER_SWEEP_TM_CORPUS_BATCH_SIZE", "64"))
TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES = int(os.environ.get("TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES", "1000000000"))
TOKENIZER_SWEEP_HF_BATCH_SIZE = int(os.environ.get("TOKENIZER_SWEEP_HF_BATCH_SIZE", "1024"))
TOKENIZER_SWEEP_HF_TRAIN_THREADS = int(
    os.environ.get("TOKENIZER_SWEEP_HF_TRAIN_THREADS", os.environ.get("TOKENIZER_SWEEP_HF_CPU", "64"))
)
TOKENIZER_SWEEP_HF_CORPUS_MAX_BYTES = int(
    os.environ.get("TOKENIZER_SWEEP_HF_CORPUS_MAX_BYTES", "0")
)
TOKENIZER_SWEEP_TRAIN_SAMPLE_MODE = os.environ.get(
    "TOKENIZER_SWEEP_TRAIN_SAMPLE_MODE",
    "random-shards",
).lower()
TOKENIZER_SWEEP_TRAIN_RANDOM_SEED = int(os.environ.get("TOKENIZER_SWEEP_TRAIN_RANDOM_SEED", "5821"))
TOKENIZER_SWEEP_TM_GETALLTOKENS_WORKERS = int(os.environ.get("TOKENIZER_SWEEP_TM_GETALLTOKENS_WORKERS", "1"))
TOKENIZER_SWEEP_TM_TRAINVOCAB_WORKERS = int(os.environ.get("TOKENIZER_SWEEP_TM_TRAINVOCAB_WORKERS", "8"))
TOKENIZER_SWEEP_TM_TRAIN_REVISION = os.environ.get("TOKENIZER_SWEEP_TM_TRAIN_REVISION", "tokenmonster-1gb-corpus-v1")
TOKENIZER_SWEEP_TM_CHUNK_SIZE = os.environ.get("TOKENIZER_SWEEP_TM_CHUNK_SIZE", "10MB")
TOKENIZER_SWEEP_TM_MICRO_CHUNKS = int(os.environ.get("TOKENIZER_SWEEP_TM_MICRO_CHUNKS", "10"))
TOKENIZER_SWEEP_TM_MIN_OCCUR_CHUNK = int(os.environ.get("TOKENIZER_SWEEP_TM_MIN_OCCUR_CHUNK", "2"))
TOKENIZER_SWEEP_TM_MIN_OCCUR_MICRO_CHUNK = int(os.environ.get("TOKENIZER_SWEEP_TM_MIN_OCCUR_MICRO_CHUNK", "1"))
TOKENIZER_SWEEP_PREEMPTIBLE_TRAINING = os.environ.get("TOKENIZER_SWEEP_PREEMPTIBLE_TRAINING", "false").lower() in {
    "1",
    "true",
    "yes",
}
TOKENIZER_SWEEP_PREEMPTIBLE_TOKENIZATION = os.environ.get(
    "TOKENIZER_SWEEP_PREEMPTIBLE_TOKENIZATION", "true"
).lower() in {
    "1",
    "true",
    "yes",
}
TOKENIZER_SWEEP_WORKER_TPU_TYPE = os.environ.get("TOKENIZER_SWEEP_WORKER_TPU_TYPE")


def _resource_config(
    *,
    cpu: int,
    ram: str,
    disk: str = "16g",
    preemptible: bool = True,
) -> ResourceConfig:
    if TOKENIZER_SWEEP_WORKER_TPU_TYPE:
        return ResourceConfig.with_tpu(
            TOKENIZER_SWEEP_WORKER_TPU_TYPE,
            cpu=cpu,
            ram=ram,
            disk=disk,
            regions=TOKENIZER_SWEEP_REGIONS,
            preemptible=preemptible,
        )
    return ResourceConfig(cpu=cpu, ram=ram, disk=disk, regions=TOKENIZER_SWEEP_REGIONS, preemptible=preemptible)

_SAMPLE_RESOURCES = _resource_config(cpu=1, ram="8g")
_HF_TRAIN_RESOURCES = _resource_config(
    cpu=int(os.environ.get("TOKENIZER_SWEEP_HF_CPU", "64")),
    ram=os.environ.get("TOKENIZER_SWEEP_HF_RAM", "768g"),
    disk=os.environ.get("TOKENIZER_SWEEP_HF_DISK", "1000g"),
    preemptible=TOKENIZER_SWEEP_PREEMPTIBLE_TRAINING,
)
_TM_CORPUS_RESOURCES = _resource_config(
    cpu=int(os.environ.get("TOKENIZER_SWEEP_TM_CORPUS_CPU", "16")),
    ram=os.environ.get("TOKENIZER_SWEEP_TM_CORPUS_RAM", "512g"),
    disk=os.environ.get("TOKENIZER_SWEEP_TM_CORPUS_DISK", "3000g"),
)
_TM_TRAIN_RESOURCES = _resource_config(
    cpu=int(os.environ.get("TOKENIZER_SWEEP_TM_TRAIN_CPU", "32")),
    ram=os.environ.get("TOKENIZER_SWEEP_TM_TRAIN_RAM", "768g"),
    disk=os.environ.get("TOKENIZER_SWEEP_TM_TRAIN_DISK", "3000g"),
    preemptible=TOKENIZER_SWEEP_PREEMPTIBLE_TRAINING,
)
_TOKENIZE_WORKER_RESOURCES = _resource_config(
    cpu=int(os.environ.get("TOKENIZER_SWEEP_TOKENIZE_WORKER_CPU", "1")),
    ram=os.environ.get("TOKENIZER_SWEEP_TOKENIZE_WORKER_RAM", "10g"),
    disk=os.environ.get("TOKENIZER_SWEEP_TOKENIZE_WORKER_DISK", "5g"),
    preemptible=TOKENIZER_SWEEP_PREEMPTIBLE_TOKENIZATION,
)


def _skip_executor_info_writes_for_generated_dag() -> None:
    """Avoid pre-run per-step metadata writes for this large generated DAG.

    Executor status files and step outputs are still written normally. This
    only skips the data-browser JSON sidecar files that otherwise issue one
    preflight write per step/dependency and can block before remote jobs launch.
    """

    def write_infos(self):
        caller = os.path.basename(self.executor_info.caller_path).replace(".py", "")
        self.executor_info_path = os.path.join(self.executor_info_base_path, f"{caller}-metadata-skipped.json")
        logger.info("Skipping executor info sidecar writes for generated tokenizer sweep DAG")
        logger.info("Executor info placeholder: %s", self.executor_info_path)

    executor_module.Executor.write_infos = write_infos


def _part_name(idx: int, total: int) -> str:
    return f"part-{idx:05d}-of-{total:05d}.parquet"


def _load_normalized_data(path: str) -> NormalizedData:
    for name in (".artifact", ".artifact.json"):
        try:
            with open_url(f"{path.rstrip('/')}/{name}") as f:
                return NormalizedData.model_validate_json(f.read())
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"No NormalizedData artifact found under {path}")


def _copy_shard(src: str, dst: str) -> int:
    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    parent = os.path.dirname(dst_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)
    src_fs.copy(src_path, dst_path)
    return int(src_fs.size(src_path) or 0)


def _write_row_range(src: str, dst: str, start_row: int, stop_row: int) -> tuple[int, int]:
    """Write rows ``[start_row, stop_row)`` from one parquet file."""
    if stop_row <= start_row:
        raise ValueError(f"empty row range for {src}: [{start_row}, {stop_row})")

    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    parent = os.path.dirname(dst_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)

    with src_fs.open(src_path, "rb") as sf:
        pf = pq.ParquetFile(sf)
        rows_in = pf.metadata.num_rows
        stop_row = min(stop_row, rows_in)
        cursor = 0
        wrote = 0
        with dst_fs.open(dst_path, "wb") as df, pq.ParquetWriter(df, pf.schema_arrow) as writer:
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                rg_start = cursor
                rg_stop = cursor + rg_rows
                cursor = rg_stop
                take_start = max(start_row, rg_start)
                take_stop = min(stop_row, rg_stop)
                if take_stop <= take_start:
                    continue
                table = pf.read_row_group(rg_idx)
                table = table.slice(take_start - rg_start, take_stop - take_start)
                writer.write_table(table)
                wrote += table.num_rows
    return rows_in, wrote


def _stable_hash_int(*parts: object) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "big")


def sample_normalized_random_shards(
    *,
    source: NormalizedData,
    output_path: str,
    sample_fraction: float,
    seed: int,
) -> NormalizedData:
    """Copy a deterministic pseudo-random shard/row sample from normalized data."""
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1]; got {sample_fraction}")

    shards = sorted(fsspec_glob(f"{source.main_output_dir.rstrip('/')}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {source.main_output_dir}")

    first_fs, first_path = url_to_fs(shards[0])
    with first_fs.open(first_path, "rb") as first_file:
        rows_per_file = pq.ParquetFile(first_file).metadata.num_rows
    total_rows_est = rows_per_file * len(shards)
    target_rows = max(1, min(total_rows_est, int(math.ceil(total_rows_est * sample_fraction))))

    shard_order = sorted(
        range(len(shards)),
        key=lambda idx: _stable_hash_int("tokenizer-train-shard", seed, source.main_output_dir, idx),
    )

    selected: list[tuple[int, int | None, int | None]] = []
    rows_remaining = target_rows
    for shard_idx in shard_order:
        if rows_remaining <= 0:
            break
        if rows_remaining >= rows_per_file:
            selected.append((shard_idx, None, None))
            rows_remaining -= rows_per_file
            continue

        rows_to_take = rows_remaining
        max_start = max(0, rows_per_file - rows_to_take)
        start_row = (
            _stable_hash_int("tokenizer-train-row", seed, source.main_output_dir, shard_idx) % (max_start + 1)
            if max_start
            else 0
        )
        selected.append((shard_idx, start_row, start_row + rows_to_take))
        rows_remaining = 0

    main_out = f"{output_path.rstrip('/')}/outputs/main"
    output_total = len(selected)
    logger.info(
        "random-shards sampler: %s -> %s fraction=%.6f target_rows=%d selected_shards=%d seed=%d",
        source.main_output_dir,
        main_out,
        sample_fraction,
        target_rows,
        output_total,
        seed,
    )

    def copy_or_slice(local_idx: int, item: tuple[int, int | None, int | None]) -> tuple[int, int]:
        shard_idx, start_row, stop_row = item
        src = shards[shard_idx]
        dst = f"{main_out}/{_part_name(local_idx, output_total)}"
        if start_row is None or stop_row is None:
            _copy_shard(src, dst)
            return rows_per_file, rows_per_file
        return _write_row_range(src, dst, start_row, stop_row)

    rows_out = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(copy_or_slice, i, item) for i, item in enumerate(selected)]
        for fut in futures:
            _, wrote = fut.result()
            rows_out += wrote

    return NormalizedData(
        main_output_dir=main_out,
        dup_output_dir=source.dup_output_dir,
        num_partitions=output_total,
        counters={
            "sampler/random_shards": 1,
            "sampler/random_seed": seed,
            "sampler/window_fraction_ppm": int(sample_fraction * 1_000_000),
            "sampler/rows_out": rows_out,
            "sampler/target_rows": target_rows,
            "sampler/selected_shards": output_total,
            "sampler/total_shards": len(shards),
        },
    )


def sample_normalized_window(
    *,
    source: NormalizedData,
    output_path: str,
    start_fraction: float,
    sample_fraction: float,
) -> NormalizedData:
    """Copy a deterministic contiguous shard/row window from normalized data."""
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError(f"start_fraction must be in [0, 1); got {start_fraction}")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1]; got {sample_fraction}")
    if start_fraction + sample_fraction > 1.0:
        raise ValueError(f"window exceeds source: {start_fraction=} {sample_fraction=}")

    shards = sorted(fsspec_glob(f"{source.main_output_dir.rstrip('/')}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {source.main_output_dir}")

    first_fs, first_path = url_to_fs(shards[0])
    with first_fs.open(first_path, "rb") as first_file:
        rows_per_file = pq.ParquetFile(first_file).metadata.num_rows
    total_rows_est = rows_per_file * len(shards)
    start_row = int(math.floor(total_rows_est * start_fraction))
    stop_row = min(total_rows_est, int(math.ceil(total_rows_est * (start_fraction + sample_fraction))))

    first_shard = start_row // rows_per_file
    last_shard = (stop_row - 1) // rows_per_file
    output_total = last_shard - first_shard + 1
    main_out = f"{output_path.rstrip('/')}/outputs/main"

    logger.info(
        "window sampler: %s -> %s start=%.6f fraction=%.6f rows=[%d,%d) shards=[%d,%d]",
        source.main_output_dir,
        main_out,
        start_fraction,
        sample_fraction,
        start_row,
        stop_row,
        first_shard,
        last_shard,
    )

    def copy_or_slice(local_idx: int, shard_idx: int) -> tuple[int, int]:
        src = shards[shard_idx]
        dst = f"{main_out}/{_part_name(local_idx, output_total)}"
        shard_start = shard_idx * rows_per_file
        shard_stop = shard_start + rows_per_file
        take_start = max(start_row, shard_start) - shard_start
        take_stop = min(stop_row, shard_stop) - shard_start
        if take_start == 0 and take_stop >= rows_per_file:
            _copy_shard(src, dst)
            return rows_per_file, rows_per_file
        return _write_row_range(src, dst, take_start, take_stop)

    rows_out = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(copy_or_slice, i, shard_idx) for i, shard_idx in enumerate(range(first_shard, last_shard + 1))]
        for fut in futures:
            _, wrote = fut.result()
            rows_out += wrote

    return NormalizedData(
        main_output_dir=main_out,
        dup_output_dir=source.dup_output_dir,
        num_partitions=output_total,
        counters={
            "sampler/window_start_fraction_ppm": int(start_fraction * 1_000_000),
            "sampler/window_fraction_ppm": int(sample_fraction * 1_000_000),
            "sampler/rows_out": rows_out,
            "sampler/selected_shards": output_total,
            "sampler/total_shards": len(shards),
        },
    )


def sample_window_step(
    *,
    name: str,
    normalized_path: str,
    start_fraction: float,
    sample_fraction: float,
    window_tokens: int = WINDOW_TOKENS,
    sample_mode: str = "contiguous",
    random_seed: int = TOKENIZER_SWEEP_TRAIN_RANDOM_SEED,
) -> StepSpec:
    def run(output_path: str) -> NormalizedData:
        if sample_mode == "random-shards":
            if start_fraction != 0.0:
                raise ValueError("random-shards sampling does not support start_fraction")
            return sample_normalized_random_shards(
                source=_load_normalized_data(normalized_path),
                output_path=output_path,
                sample_fraction=sample_fraction,
                seed=random_seed,
            )
        if sample_mode != "contiguous":
            raise ValueError(f"Unknown sample_mode={sample_mode!r}; expected 'contiguous' or 'random-shards'")
        return sample_normalized_window(
            source=_load_normalized_data(normalized_path),
            output_path=output_path,
            start_fraction=start_fraction,
            sample_fraction=sample_fraction,
        )

    return StepSpec(
        name=name,
        hash_attrs={
            "normalized_path": normalized_path,
            "start_fraction": start_fraction,
            "sample_fraction": sample_fraction,
            "window_tokens": window_tokens,
            "total_tokenized_tokens": TOTAL_TOKENIZED_TOKENS,
            "sample_mode": sample_mode,
            "random_seed": random_seed if sample_mode == "random-shards" else None,
        },
        fn=remote(run, resources=_SAMPLE_RESOURCES),
    )


def existing_normalized_sources() -> dict[str, str]:
    """Return all 2026-05-26 source artifacts that exist in the regional sample."""
    if not NORMALIZED_BASE.startswith("gs://"):
        raise ValueError(f"expected GCS NORMALIZED_BASE, got {NORMALIZED_BASE}")
    fs = fsspec.filesystem("gcs")
    base = NORMALIZED_BASE.removeprefix("gs://").rstrip("/")
    artifact_paths: set[str] = set()

    def artifact_in(entries: list[str]) -> str | None:
        for entry in entries:
            if os.path.basename(entry) in {".artifact.json", ".artifact"}:
                return entry
        return None

    def inspect_prefix(path: str) -> list[str]:
        try:
            entries = fs.ls(path, detail=False)
        except FileNotFoundError:
            return []
        found = []
        own_artifact = artifact_in(entries)
        if own_artifact is not None:
            found.append(own_artifact)

        for entry in entries:
            name = os.path.basename(entry.rstrip("/"))
            if name.startswith(".") or name == "outputs":
                continue
            try:
                child_entries = fs.ls(entry, detail=False)
            except FileNotFoundError:
                continue
            child_artifact = artifact_in(child_entries)
            if child_artifact is not None:
                found.append(child_artifact)
                continue
            for grandchild in child_entries:
                grandchild_name = os.path.basename(grandchild.rstrip("/"))
                if grandchild_name.startswith(".") or grandchild_name == "outputs":
                    continue
                try:
                    grandchild_entries = fs.ls(grandchild, detail=False)
                except FileNotFoundError:
                    continue
                grandchild_artifact = artifact_in(grandchild_entries)
                if grandchild_artifact is not None:
                    found.append(grandchild_artifact)
        return found

    top_level = sorted(fs.ls(base, detail=False))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for found in pool.map(inspect_prefix, top_level):
            artifact_paths.update(found)

    paths: dict[str, str] = {}
    for artifact_path in sorted(artifact_paths):
        path = f"gs://{artifact_path.rsplit('/', 1)[0]}"
        source_name = path.removeprefix(f"{NORMALIZED_BASE}/")
        paths[source_name] = path

    if not paths:
        raise ValueError(f"No normalized source artifacts found under {NORMALIZED_BASE}")
    logger.info("Discovered %d existing normalized sources under %s", len(paths), NORMALIZED_BASE)
    return paths


def _iter_text_batches(paths: list[str], *, batch_size: int = 1024) -> Iterator[list[str]]:
    batch: list[str] = []
    for pattern in paths:
        for shard in sorted(fsspec_glob(pattern)):
            fs, resolved = url_to_fs(shard)
            with fs.open(resolved, "rb") as f:
                pf = pq.ParquetFile(f)
                for record_batch in pf.iter_batches(columns=["text"], batch_size=batch_size):
                    texts = record_batch.column("text").to_pylist()
                    batch.extend(str(t) for t in texts if t is not None)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    if batch:
        yield batch


def _iter_limited_round_robin_text_batches(
    paths: list[str],
    *,
    batch_size: int,
    max_bytes: int,
    stats: dict[str, int] | None = None,
) -> Iterator[list[str]]:
    """Yield up to ``max_bytes`` of text, round-robin across source patterns."""
    docs = 0
    bytes_read = 0
    batch: list[str] = []
    has_byte_limit = max_bytes > 0
    iterators = [_iter_text_batches([pattern], batch_size=batch_size) for pattern in paths]
    active = list(range(len(iterators)))
    try:
        while active and (not has_byte_limit or bytes_read < max_bytes):
            next_active: list[int] = []
            for idx in active:
                try:
                    source_batch = next(iterators[idx])
                except StopIteration:
                    continue

                next_active.append(idx)
                for text in source_batch:
                    encoded_len = len(text.encode("utf-8")) + 1
                    if has_byte_limit and docs > 0 and bytes_read + encoded_len > max_bytes:
                        break
                    batch.append(text)
                    docs += 1
                    bytes_read += encoded_len
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                    if has_byte_limit and bytes_read >= max_bytes:
                        break
                if has_byte_limit and bytes_read >= max_bytes:
                    break
            active = next_active
        if batch:
            yield batch
    finally:
        for iterator in iterators:
            iterator.close()
        if stats is not None:
            stats["documents"] = docs
            stats["bytes"] = bytes_read


def _iter_limited_shuffled_text_batches(
    paths: list[str],
    *,
    batch_size: int,
    max_bytes: int,
    seed: int,
    stats: dict[str, int] | None = None,
) -> Iterator[list[str]]:
    """Yield up to ``max_bytes`` of text from a deterministic shuffled file order."""
    docs = 0
    bytes_read = 0
    batch: list[str] = []
    has_byte_limit = max_bytes > 0
    shards: list[str] = []
    for pattern in paths:
        shards.extend(fsspec_glob(pattern))
    shards = sorted(
        set(shards),
        key=lambda shard: _stable_hash_int("hf-corpus-shard", seed, shard),
    )
    try:
        for shard in shards:
            if has_byte_limit and bytes_read >= max_bytes:
                break
            fs, resolved = url_to_fs(shard)
            with fs.open(resolved, "rb") as f:
                pf = pq.ParquetFile(f)
                row_groups = sorted(
                    range(pf.num_row_groups),
                    key=lambda rg_idx: _stable_hash_int("hf-corpus-row-group", seed, shard, rg_idx),
                )
                for rg_idx in row_groups:
                    for record_batch in pf.iter_batches(
                        columns=["text"],
                        batch_size=batch_size,
                        row_groups=[rg_idx],
                    ):
                        for text_value in record_batch.column("text").to_pylist():
                            if text_value is None:
                                continue
                            text = str(text_value)
                            encoded_len = len(text.encode("utf-8")) + 1
                            if has_byte_limit and docs > 0 and bytes_read + encoded_len > max_bytes:
                                break
                            batch.append(text)
                            docs += 1
                            bytes_read += encoded_len
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                            if has_byte_limit and bytes_read >= max_bytes:
                                break
                        if has_byte_limit and bytes_read >= max_bytes:
                            break
                    if has_byte_limit and bytes_read >= max_bytes:
                        break
        if batch:
            yield batch
    finally:
        if stats is not None:
            stats["documents"] = docs
            stats["bytes"] = bytes_read
            stats["shards"] = len(shards)


def _is_numeric_char(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _place_aligned_digit_run_pieces(run: str) -> list[str]:
    """Split one numeric run into bounded, right-aligned groups of three."""
    pieces: list[str] = []
    for chunk_start in range(0, len(run), PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS):
        chunk = run[chunk_start : chunk_start + PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS]
        leading = len(chunk) % PLACE_ALIGNED_DIGIT_CHUNK_SIZE
        if leading:
            pieces.append(chunk[:leading])
        pieces.extend(
            chunk[i : i + PLACE_ALIGNED_DIGIT_CHUNK_SIZE]
            for i in range(leading, len(chunk), PLACE_ALIGNED_DIGIT_CHUNK_SIZE)
        )
    return pieces


def place_aligned_digit_pieces(text: str) -> list[str]:
    """Split text at 4915-style place-aligned numeric boundaries.

    Contiguous numeric runs are isolated from surrounding text, capped at 510
    characters, and then split into right-aligned groups of three.
    """
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        run_is_numeric = _is_numeric_char(text[cursor])
        run_start = cursor
        cursor += 1
        while cursor < len(text) and _is_numeric_char(text[cursor]) == run_is_numeric:
            cursor += 1
        run = text[run_start:cursor]
        if run_is_numeric:
            pieces.extend(_place_aligned_digit_run_pieces(run))
        else:
            pieces.append(run)
    return pieces


def _place_aligned_digit_pretokenizer(original_pretokenizer):
    from tokenizers import Regex
    from tokenizers import pre_tokenizers

    leading_width = PLACE_ALIGNED_DIGIT_CHUNK_SIZE - 1
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(rf"\p{{N}}{{1,{PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS}}}"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(
                Regex(
                    rf"^\p{{N}}{{1,{leading_width}}}"
                    rf"(?=(?:\p{{N}}{{{PLACE_ALIGNED_DIGIT_CHUNK_SIZE}}})+$)"
                ),
                behavior="isolated",
            ),
            pre_tokenizers.Split(
                Regex(rf"\p{{N}}{{{PLACE_ALIGNED_DIGIT_CHUNK_SIZE}}}"),
                behavior="isolated",
            ),
            original_pretokenizer,
        ]
    )


def _apply_place_aligned_digit_pretokenizer(tokenizer) -> None:
    tokenizer.backend_tokenizer.pre_tokenizer = _place_aligned_digit_pretokenizer(
        tokenizer.backend_tokenizer.pre_tokenizer
    )


def _mirror_hf_tokenizer(local_dir: str, tokenizer_name: str) -> None:
    from huggingface_hub import __version__ as hf_hub_version

    mirror_base = f"mirror://tokenizers/{tokenizer_name}/hf-hub-{hf_hub_version}"
    for filename in sorted(os.listdir(local_dir)):
        src = os.path.join(local_dir, filename)
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as sf, fsspec.open(f"{mirror_base}/{filename}", "wb") as df:
            shutil.copyfileobj(sf, df)


def _copy_dir_to_url(local_dir: str, dst_url: str) -> None:
    for filename in sorted(os.listdir(local_dir)):
        src = os.path.join(local_dir, filename)
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as sf, fsspec.open(f"{dst_url.rstrip('/')}/{filename}", "wb") as df:
            shutil.copyfileobj(sf, df)


def _merge_parts(merge: str | list[str]) -> tuple[str, str]:
    if isinstance(merge, str):
        left, right = merge.split(" ", 1)
        return left, right
    if len(merge) != 2:
        raise ValueError(f"Expected two-part BPE merge, got {merge!r}")
    return merge[0], merge[1]


def _rewrite_special_token_ids(value, token_to_new_id: dict[str, int]):
    """Rewrite numeric special-token ids in tokenizer sidecar JSON objects."""
    if isinstance(value, dict):
        if "content" in value and "id" in value and value["content"] in token_to_new_id:
            value = dict(value)
            value["id"] = token_to_new_id[value["content"]]
        elif "tokens" in value and "ids" in value:
            tokens = value.get("tokens") or []
            if all(token in token_to_new_id for token in tokens):
                value = dict(value)
                value["ids"] = [token_to_new_id[token] for token in tokens]
        return {k: _rewrite_special_token_ids(v, token_to_new_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_special_token_ids(v, token_to_new_id) for v in value]
    return value


def _derive_hf_bpe_tokenizer_dir(base_dir: str, target_size: int, output_dir: str) -> None:
    """Derive a smaller BPE tokenizer from a trained 262k tokenizer directory."""
    tokenizer_json_path = os.path.join(base_dir, "tokenizer.json")
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    model = tokenizer_json.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"Can only derive BPE tokenizers; got model type {model.get('type')!r}")

    added_tokens = tokenizer_json.get("added_tokens") or []
    special_tokens = [tok for tok in added_tokens if tok.get("special")]
    special_contents = [tok["content"] for tok in special_tokens]
    if len(set(special_contents)) != len(special_contents):
        raise ValueError("Duplicate special-token contents in tokenizer.json")

    model_vocab_size = target_size - len(special_tokens)
    if model_vocab_size <= 0:
        raise ValueError(f"target_size={target_size} leaves no room after {len(special_tokens)} special tokens")

    old_vocab: dict[str, int] = model["vocab"]
    retained_model_tokens = [
        token
        for token, _ in sorted(old_vocab.items(), key=lambda item: item[1])
        if token not in set(special_contents)
    ][:model_vocab_size]
    if len(retained_model_tokens) < model_vocab_size:
        raise ValueError(
            f"Base tokenizer has only {len(retained_model_tokens)} regular tokens; cannot derive {target_size}"
        )

    new_vocab = {token: idx for idx, token in enumerate(retained_model_tokens)}
    special_id_map = {token: model_vocab_size + idx for idx, token in enumerate(special_contents)}

    retained_token_set = set(new_vocab)
    new_merges = []
    for merge in model.get("merges") or []:
        left, right = _merge_parts(merge)
        if left in retained_token_set and right in retained_token_set and f"{left}{right}" in retained_token_set:
            new_merges.append(merge)

    derived_json = dict(tokenizer_json)
    derived_json["model"] = dict(model)
    derived_json["model"]["vocab"] = new_vocab
    derived_json["model"]["merges"] = new_merges
    derived_json["added_tokens"] = _rewrite_special_token_ids(added_tokens, special_id_map)
    if derived_json.get("post_processor") is not None:
        derived_json["post_processor"] = _rewrite_special_token_ids(derived_json["post_processor"], special_id_map)

    os.makedirs(output_dir, exist_ok=True)
    for filename in sorted(os.listdir(base_dir)):
        src = os.path.join(base_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))

    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(derived_json, f, ensure_ascii=False)

    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        tokenizer_config = _rewrite_special_token_ids(tokenizer_config, special_id_map)
        tokenizer_config["model_max_length"] = int(tokenizer_config.get("model_max_length", 1_000_000_000_000_000_000))
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)


def _train_hf_family(
    output_path: str,
    *,
    family: str,
    base_tokenizer: str,
    train_patterns: list[str],
    place_aligned_digits: bool,
) -> dict:
    from transformers import AutoTokenizer

    base_size = VOCAB_SIZES[0]
    corpus_stats: dict[str, int] = {}
    logger.info("Training %s base tokenizer at vocab size %d from %s", family, base_size, base_tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(TOKENIZER_SWEEP_HF_TRAIN_THREADS)
    logger.info("Using HF tokenizer training parallelism with %d Rayon threads", TOKENIZER_SWEEP_HF_TRAIN_THREADS)
    base = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    if place_aligned_digits:
        logger.info(
            "Applying 4915 place-aligned digit pretokenizer to %s with max numeric run %d",
            family,
            PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS,
        )
        _apply_place_aligned_digit_pretokenizer(base)
    tokenizer = base.train_new_from_iterator(
        _iter_limited_shuffled_text_batches(
            train_patterns,
            batch_size=TOKENIZER_SWEEP_HF_BATCH_SIZE,
            max_bytes=TOKENIZER_SWEEP_HF_CORPUS_MAX_BYTES,
            seed=TOKENIZER_SWEEP_TRAIN_RANDOM_SEED,
            stats=corpus_stats,
        ),
        vocab_size=base_size,
        length=None,
        new_special_tokens=[],
    )
    base_local_dir = tempfile.mkdtemp(prefix=f"{family}-{base_size}-")
    tokenizer.save_pretrained(base_local_dir)

    results = {}
    for size in VOCAB_SIZES:
        local_dir = base_local_dir
        if size != base_size:
            logger.info("Deriving %s tokenizer at vocab size %d from %d base", family, size, base_size)
            local_dir = tempfile.mkdtemp(prefix=f"{family}-{size}-")
            _derive_hf_bpe_tokenizer_dir(base_local_dir, size, local_dir)

        tokenizer_name = f"marin-community/{RUN_ID}-{family}-{size // 1024}k"
        _mirror_hf_tokenizer(local_dir, tokenizer_name)
        _copy_dir_to_url(local_dir, f"{output_path}/{family}/{size}")
        results[str(size)] = {
            "tokenizer": tokenizer_name,
            "backend": "hf",
            "path": f"{output_path}/{family}/{size}",
            "derived_from": str(base_size),
        }

    with open_url(f"{output_path}/metadata.json", "w") as f:
        json.dump(
            {
                "family": family,
                "base_tokenizer": base_tokenizer,
                "results": results,
                "corpus": {
                    "documents": corpus_stats.get("documents", 0),
                    "bytes": corpus_stats.get("bytes", 0),
                    "max_bytes": TOKENIZER_SWEEP_HF_CORPUS_MAX_BYTES,
                    "sources": len(train_patterns),
                    "shards": corpus_stats.get("shards", 0),
                    "format": "deterministic-shuffled-parquet-text",
                    "seed": TOKENIZER_SWEEP_TRAIN_RANDOM_SEED,
                    "upstream_tokenizer_repo": base_tokenizer,
                },
                "pretokenizer": {
                    "place_aligned_digits": place_aligned_digits,
                    "numeric_run_cap": PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS if place_aligned_digits else None,
                    "numeric_chunk_size": PLACE_ALIGNED_DIGIT_CHUNK_SIZE if place_aligned_digits else None,
                    "revision": PLACE_ALIGNED_DIGIT_PRETOKENIZER_REVISION if place_aligned_digits else None,
                    "issue": "https://github.com/marin-community/marin/issues/4915"
                    if place_aligned_digits
                    else None,
                },
            },
            f,
            indent=2,
        )
    return results


def train_hf_family_step(
    family: str,
    base_tokenizer: str,
    train_samples: list[StepSpec],
    *,
    place_aligned_digits: bool,
) -> StepSpec:
    train_patterns = [f"{s.output_path}/outputs/main/*.parquet" for s in train_samples]

    def run(output_path: str) -> dict:
        return _train_hf_family(
            output_path,
            family=family,
            base_tokenizer=base_tokenizer,
            train_patterns=train_patterns,
            place_aligned_digits=place_aligned_digits,
        )

    return StepSpec(
        name=f"tokenizers/{RUN_ID}/{family}",
        deps=train_samples,
        hash_attrs={
            "family": family,
            "base_tokenizer": base_tokenizer,
            "vocab_sizes": VOCAB_SIZES,
            "window_tokens": TOKENIZER_TRAIN_TOKENS,
            "derive_from": f"train {VOCAB_SIZES[0]} once, then truncate BPE vocab/merges",
            "resource_revision": TOKENIZER_SWEEP_RESOURCE_REVISION,
            "regions": TOKENIZER_SWEEP_REGIONS,
            "ram": _HF_TRAIN_RESOURCES.ram,
            "preemptible": _HF_TRAIN_RESOURCES.preemptible,
            "corpus_max_bytes": TOKENIZER_SWEEP_HF_CORPUS_MAX_BYTES,
            "batch_size": TOKENIZER_SWEEP_HF_BATCH_SIZE,
            "train_threads": TOKENIZER_SWEEP_HF_TRAIN_THREADS,
            "tokenizers_parallelism": "true",
            "sampling": "deterministic-shuffled-files-and-row-groups",
            "sampling_seed": TOKENIZER_SWEEP_TRAIN_RANDOM_SEED,
            "place_aligned_digits": place_aligned_digits,
            "digit_max_run_chars": PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS if place_aligned_digits else None,
            "digit_chunk_size": PLACE_ALIGNED_DIGIT_CHUNK_SIZE if place_aligned_digits else None,
            "digit_pretokenizer_revision": PLACE_ALIGNED_DIGIT_PRETOKENIZER_REVISION
            if place_aligned_digits
            else None,
        },
        fn=remote(run, resources=_HF_TRAIN_RESOURCES),
    )


def _derive_official_hf_family(output_path: str, *, family: str, base_tokenizer: str) -> dict:
    from transformers import AutoTokenizer

    base_size = VOCAB_SIZES[0]
    logger.info("Deriving %s tokenizer sizes from official %s", family, base_tokenizer)
    base = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    base_local_dir = tempfile.mkdtemp(prefix=f"{family}-{base_size}-")
    base.save_pretrained(base_local_dir)
    size_filter_raw = _env_filter("TOKENIZER_SWEEP_SIZES")
    derive_sizes = tuple(int(size) for size in size_filter_raw) if size_filter_raw is not None else VOCAB_SIZES

    results = {}
    for size in derive_sizes:
        local_dir = base_local_dir
        if size != base_size:
            logger.info("Deriving %s tokenizer at vocab size %d from official %d base", family, size, base_size)
            local_dir = tempfile.mkdtemp(prefix=f"{family}-{size}-")
            _derive_hf_bpe_tokenizer_dir(base_local_dir, size, local_dir)

        tokenizer_name = f"marin-community/{RUN_ID}-{family}-{size // 1024}k"
        _mirror_hf_tokenizer(local_dir, tokenizer_name)
        _copy_dir_to_url(local_dir, f"{output_path}/{family}/{size}")
        results[str(size)] = {
            "tokenizer": tokenizer_name,
            "backend": "hf",
            "path": f"{output_path}/{family}/{size}",
            "derived_from": base_tokenizer,
        }

    with open_url(f"{output_path}/metadata.json", "w") as f:
        json.dump(
            {
                "family": family,
                "base_tokenizer": base_tokenizer,
                "results": results,
                "corpus": None,
                "format": "official-hf-bpe-truncated-by-rank",
            },
            f,
            indent=2,
        )
    return results


def official_truncated_hf_family_step(family: str, base_tokenizer: str) -> StepSpec:
    def run(output_path: str) -> dict:
        return _derive_official_hf_family(output_path, family=family, base_tokenizer=base_tokenizer)

    return StepSpec(
        name=f"tokenizers/{RUN_ID}/{family}",
        hash_attrs={
            "family": family,
            "base_tokenizer": base_tokenizer,
            "vocab_sizes": VOCAB_SIZES,
            "derive_sizes": os.environ.get("TOKENIZER_SWEEP_SIZES", ",".join(str(size) for size in VOCAB_SIZES)),
            "derive_from": "official HF tokenizer, then truncate BPE vocab/merges by rank",
            "resource_revision": TOKENIZER_SWEEP_RESOURCE_REVISION,
            "regions": TOKENIZER_SWEEP_REGIONS,
            "ram": _HF_TRAIN_RESOURCES.ram,
            "preemptible": _HF_TRAIN_RESOURCES.preemptible,
        },
        fn=remote(run, resources=_HF_TRAIN_RESOURCES),
    )


def _write_tokenmonster_corpus(output_path: str, *, train_patterns: list[str]) -> dict:
    """Stream TokenMonster's required plain-text corpus to the output path.

    TokenMonster's trainer is designed for a single representative text file
    around 1GB, not the full 100B-token tokenizer sweep window. Keep source
    coverage broad by taking batches round-robin across the sampled sources,
    then stop at the configured byte budget.
    """
    dst = f"{output_path.rstrip('/')}/train.txt"
    docs = 0
    bytes_written = 0
    iterators = [
        _iter_text_batches([pattern], batch_size=TOKENIZER_SWEEP_TM_CORPUS_BATCH_SIZE)
        for pattern in train_patterns
    ]
    active = list(range(len(iterators)))
    with fsspec.open(dst, "wb") as raw_out:
        out = io.TextIOWrapper(raw_out, encoding="utf-8", write_through=False)
        try:
            while active and bytes_written < TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES:
                next_active: list[int] = []
                for idx in active:
                    try:
                        batch = next(iterators[idx])
                    except StopIteration:
                        continue

                    next_active.append(idx)
                    for text in batch:
                        encoded_len = len(text.encode("utf-8")) + 1
                        if docs > 0 and bytes_written + encoded_len > TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES:
                            break
                        out.write(text)
                        out.write("\n")
                        docs += 1
                        bytes_written += encoded_len
                    if docs % 100_000 == 0:
                        out.flush()
                    if bytes_written >= TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES:
                        break
                active = next_active
        finally:
            for iterator in iterators:
                iterator.close()
        out.flush()
    metadata = {
        "path": dst,
        "documents": docs,
        "bytes": bytes_written,
        "max_bytes": TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES,
        "sources": len(train_patterns),
        "format": "round-robin-source-plain-text-lines",
    }
    with open_url(f"{output_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def tokenmonster_corpus_step(train_samples: list[StepSpec]) -> StepSpec:
    train_patterns = [f"{s.output_path}/outputs/main/*.parquet" for s in train_samples]

    def run(output_path: str) -> dict:
        return _write_tokenmonster_corpus(output_path, train_patterns=train_patterns)

    return StepSpec(
        name=f"tokenizers/{RUN_ID}/tokenmonster-corpus",
        deps=train_samples,
        hash_attrs={
            "window_tokens": TOKENIZER_TRAIN_TOKENS,
            "format": "plain-text-lines",
            "resource_revision": TOKENIZER_SWEEP_RESOURCE_REVISION,
            "regions": TOKENIZER_SWEEP_REGIONS,
            "writer": "direct-gcs-stream",
            "ram": _TM_CORPUS_RESOURCES.ram,
            "batch_size": TOKENIZER_SWEEP_TM_CORPUS_BATCH_SIZE,
            "max_bytes": TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES,
            "sampling": "round-robin-over-tokenizer-train-samples",
        },
        fn=remote(run, resources=_TM_CORPUS_RESOURCES),
    )


def _download_tokenmonster_binary(name: str, dest_dir: str) -> str:
    candidates = [
        f"binaries/linux_x86_64/{name}",
        f"binaries/linux-amd64/{name}",
        f"binaries/{name}",
    ]
    last_error: Exception | None = None
    for repo_path in candidates:
        try:
            path = hf_hub_download("alasdairforsythe/tokenmonster", repo_path)
            dst = os.path.join(dest_dir, name)
            shutil.copy2(path, dst)
            os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return dst
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not download TokenMonster binary {name}") from last_error


def _train_tokenmonster(output_path: str, *, corpus_path: str) -> dict:
    work_base = os.environ.get("TOKENIZER_SWEEP_TM_WORK_BASE", os.getcwd())
    os.makedirs(work_base, exist_ok=True)
    work = tempfile.mkdtemp(prefix="tokenmonster-train-", dir=work_base)
    corpus = os.path.join(work, "train.txt")
    with fsspec.open(corpus_path, "rb") as src, open(corpus, "wb") as dst:
        shutil.copyfileobj(src, dst)

    getalltokens = _download_tokenmonster_binary("getalltokens", work)
    trainvocab = _download_tokenmonster_binary("trainvocab", work)
    exportvocab = _download_tokenmonster_binary("exportvocab", work)
    dictionary = os.path.join(work, "dictionary.txt")
    train_dir = os.path.join(work, "trainvocab-output")

    subprocess.run(
        [
            getalltokens,
            "-dataset",
            corpus,
            "-output",
            dictionary,
            "-charset",
            "UTF-8",
            "-mode",
            "clean",
            "-capcode",
            "2",
            "-workers",
            str(TOKENIZER_SWEEP_TM_GETALLTOKENS_WORKERS),
            "-chunk-size",
            TOKENIZER_SWEEP_TM_CHUNK_SIZE,
            "-micro-chunks",
            str(TOKENIZER_SWEEP_TM_MICRO_CHUNKS),
            "-min-occur-chunk",
            str(TOKENIZER_SWEEP_TM_MIN_OCCUR_CHUNK),
            "-min-occur-micro-chunk",
            str(TOKENIZER_SWEEP_TM_MIN_OCCUR_MICRO_CHUNK),
        ],
        check=True,
    )
    subprocess.run(
        [
            trainvocab,
            "-dataset",
            corpus,
            "-dictionary",
            dictionary,
            "-dir",
            train_dir,
            "-vocab-size",
            str(VOCAB_SIZES[0]),
            "-include-256-bytes",
            "-workers",
            str(TOKENIZER_SWEEP_TM_TRAINVOCAB_WORKERS),
        ],
        check=True,
    )

    results = {}
    base_vocab = os.path.join(work, "tokenmonster-262k.vocab")
    subprocess.run([exportvocab, "-input", train_dir, "-output", base_vocab, "-reset-token-ids"], check=True)
    for size in VOCAB_SIZES:
        out_vocab = base_vocab if size == VOCAB_SIZES[0] else os.path.join(work, f"tokenmonster-{size}.vocab")
        if size != VOCAB_SIZES[0]:
            subprocess.run(
                [
                    exportvocab,
                    "-input-vocab",
                    base_vocab,
                    "-resize",
                    str(size),
                    "-output",
                    out_vocab,
                    "-reset-token-ids",
                ],
                check=True,
            )
        dst = f"{output_path.rstrip('/')}/tokenmonster/{size}/tokenizer.vocab"
        with open(out_vocab, "rb") as src, fsspec.open(dst, "wb") as dst_f:
            shutil.copyfileobj(src, dst_f)
        results[str(size)] = {"tokenizer": dst, "backend": "tokenmonster", "path": dst}

    with open_url(f"{output_path}/metadata.json", "w") as f:
        json.dump({"family": "tokenmonster", "results": results, "corpus_path": corpus_path}, f, indent=2)
    return results


def train_tokenmonster_step(corpus_step: StepSpec) -> StepSpec:
    corpus_path = f"{corpus_step.output_path}/train.txt"

    def run(output_path: str) -> dict:
        return _train_tokenmonster(output_path, corpus_path=corpus_path)

    return StepSpec(
        name=f"tokenizers/{RUN_ID}/tokenmonster",
        deps=[corpus_step],
        hash_attrs={
            "vocab_sizes": VOCAB_SIZES,
            "mode": "clean",
            "capcode": 2,
            "window_tokens": TOKENIZER_TRAIN_TOKENS,
            "corpus_max_bytes": TOKENIZER_SWEEP_TM_CORPUS_MAX_BYTES,
            "resource_revision": TOKENIZER_SWEEP_RESOURCE_REVISION,
            "regions": TOKENIZER_SWEEP_REGIONS,
            "ram": _TM_TRAIN_RESOURCES.ram,
            "preemptible": _TM_TRAIN_RESOURCES.preemptible,
            "getalltokens_workers": TOKENIZER_SWEEP_TM_GETALLTOKENS_WORKERS,
            "getalltokens_chunk_size": TOKENIZER_SWEEP_TM_CHUNK_SIZE,
            "getalltokens_micro_chunks": TOKENIZER_SWEEP_TM_MICRO_CHUNKS,
            "getalltokens_min_occur_chunk": TOKENIZER_SWEEP_TM_MIN_OCCUR_CHUNK,
            "getalltokens_min_occur_micro_chunk": TOKENIZER_SWEEP_TM_MIN_OCCUR_MICRO_CHUNK,
            "trainvocab_workers": TOKENIZER_SWEEP_TM_TRAINVOCAB_WORKERS,
            "train_dir_name": "trainvocab-output",
            "train_revision": TOKENIZER_SWEEP_TM_TRAIN_REVISION,
        },
        fn=remote(run, resources=_TM_TRAIN_RESOURCES),
    )


@dataclass(frozen=True)
class TokenizeAfterTokenizerConfig:
    tokenize: TokenizeConfig
    tokenizer_done: str


def tokenize_after_tokenizer(config: TokenizeAfterTokenizerConfig):
    return tokenize(config.tokenize)


def holdout_tokenize_step(
    *,
    bucket_name: str,
    sampled_exec: ExecutorStep,
    tokenizer_name: str,
    tokenizer_backend: TokenizerBackend,
    tokenizer_exec: ExecutorStep,
) -> TokenizerStep:
    return ExecutorStep(
        name=os.path.join("data/datakit", "tokenized", RUN_ID, bucket_name),
        fn=tokenize_after_tokenizer,
        config=TokenizeAfterTokenizerConfig(
            tokenize=TokenizeConfig(
                train_paths=[sampled_exec / "outputs/main/*.parquet"],
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer_name),
                tokenizer_backend=versioned(tokenizer_backend),
                worker_resources=versioned(_TOKENIZE_WORKER_RESOURCES),
            ),
            tokenizer_done=tokenizer_exec / "metadata.json",
        ),
    )


def _env_filter(name: str) -> set[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def build_steps(phase: str = "all") -> list[ExecutorStep]:
    family_filter = _env_filter("TOKENIZER_SWEEP_FAMILIES")
    size_filter_raw = _env_filter("TOKENIZER_SWEEP_SIZES")
    size_filter = {int(size) for size in size_filter_raw} if size_filter_raw is not None else None

    sources = existing_normalized_sources()
    train_samples: dict[str, StepSpec] = {}
    holdout_samples: dict[str, StepSpec] = {}
    retokenize_train_samples: dict[str, StepSpec] = {}
    for source_name, normalized_path in sources.items():
        safe_source_name = source_name.replace("/", "__")
        train_samples[source_name] = sample_window_step(
            name=f"data/datakit/tokenizer_sweep/{RUN_ID}/train/{safe_source_name}",
            normalized_path=normalized_path,
            start_fraction=0.0,
            sample_fraction=TOKENIZER_TRAIN_FRACTION,
            window_tokens=TOKENIZER_TRAIN_TOKENS,
            sample_mode=TOKENIZER_SWEEP_TRAIN_SAMPLE_MODE,
            random_seed=TOKENIZER_SWEEP_TRAIN_RANDOM_SEED,
        )
        retokenize_train_samples[source_name] = sample_window_step(
            name=f"data/datakit/tokenizer_sweep/{RUN_ID}/{RETOKENIZE_TRAIN_LABEL}/{safe_source_name}",
            normalized_path=normalized_path,
            start_fraction=RETOKENIZE_TRAIN_START_FRACTION,
            sample_fraction=RETOKENIZE_TRAIN_FRACTION,
            window_tokens=RETOKENIZE_TRAIN_TOKENS,
        )
        holdout_samples[source_name] = sample_window_step(
            name=f"data/datakit/tokenizer_sweep/{RUN_ID}/holdout/{safe_source_name}",
            normalized_path=normalized_path,
            start_fraction=WINDOW_FRACTION,
            sample_fraction=WINDOW_FRACTION,
        )

    train_sample_list = list(train_samples.values())
    tokenizer_steps: dict[str, tuple[StepSpec, TokenizerBackend, dict[int, str]]] = {}
    for family, (base, place_aligned_digits) in HF_TOKENIZER_FAMILIES.items():
        step = train_hf_family_step(
            family,
            base,
            train_sample_list,
            place_aligned_digits=place_aligned_digits,
        )
        tokenizer_steps[family] = (
            step,
            TokenizerBackend.HF,
            {size: f"marin-community/{RUN_ID}-{family}-{size // 1024}k" for size in VOCAB_SIZES},
        )
    for family, base in OFFICIAL_TRUNCATED_TOKENIZER_FAMILIES.items():
        step = official_truncated_hf_family_step(family, base)
        tokenizer_steps[family] = (
            step,
            TokenizerBackend.HF,
            {size: f"marin-community/{RUN_ID}-{family}-{size // 1024}k" for size in VOCAB_SIZES},
        )

    corpus = tokenmonster_corpus_step(train_sample_list)
    tokenmonster_step = train_tokenmonster_step(corpus)
    tokenizer_steps["tokenmonster"] = (
        tokenmonster_step,
        TokenizerBackend.TOKENMONSTER,
        {
            size: f"{tokenmonster_step.output_path}/tokenmonster/{size}/tokenizer.vocab"
            for size in VOCAB_SIZES
        },
    )
    if family_filter is not None:
        unknown = family_filter - set(tokenizer_steps)
        if unknown:
            raise ValueError(f"Unknown TOKENIZER_SWEEP_FAMILIES entries: {sorted(unknown)}")
        tokenizer_steps = {family: value for family, value in tokenizer_steps.items() if family in family_filter}
    if size_filter is not None:
        unknown_sizes = size_filter - set(VOCAB_SIZES)
        if unknown_sizes:
            raise ValueError(f"Unknown TOKENIZER_SWEEP_SIZES entries: {sorted(unknown_sizes)}")

    holdout_execs = {source_name: step.as_executor_step() for source_name, step in holdout_samples.items()}
    retokenize_train_execs = {
        source_name: step.as_executor_step() for source_name, step in retokenize_train_samples.items()
    }
    tokenizer_execs = {family: step.as_executor_step() for family, (step, _, _) in tokenizer_steps.items()}

    if phase == "prep":
        prep_steps = [*holdout_execs.values(), *tokenizer_execs.values()]
        logger.info(
            "Tokenizer sweep prep DAG: %d sources, %d holdout samples, %d tokenizer-training targets",
            len(sources),
            len(holdout_execs),
            len(tokenizer_execs),
        )
        return prep_steps
    if phase not in {"all", "train_tokenization"}:
        raise ValueError(
            f"Unknown TOKENIZER_SWEEP_PHASE={phase!r}; expected 'prep', 'all', or 'train_tokenization'"
        )

    if phase == "train_tokenization":
        sample_execs = retokenize_train_execs
        sample_names = retokenize_train_samples
        output_prefix = RETOKENIZE_TRAIN_LABEL
    else:
        sample_execs = holdout_execs
        sample_names = holdout_samples
        output_prefix = ""

    tokenized_steps: list[ExecutorStep] = []
    for family, (tokenizer_step, backend, names_by_size) in tokenizer_steps.items():
        for size, tokenizer_name in names_by_size.items():
            if size_filter is not None and size not in size_filter:
                continue
            for source_name in sample_names:
                safe_source_name = source_name.replace("/", "__")
                bucket_name = f"{family}-{size // 1024}k/{safe_source_name}"
                if output_prefix:
                    bucket_name = f"{output_prefix}/{bucket_name}"
                tokenized_steps.append(
                    holdout_tokenize_step(
                        bucket_name=bucket_name,
                        sampled_exec=sample_execs[source_name],
                        tokenizer_name=tokenizer_name,
                        tokenizer_backend=backend,
                        tokenizer_exec=tokenizer_execs[family],
                    )
                )
    logger.info(
        "Tokenizer sweep DAG: %d sources, %d tokenizer families, %d vocab sizes, %d tokenization steps",
        len(sources),
        len(tokenizer_steps),
        len(size_filter or set(VOCAB_SIZES)),
        len(tokenized_steps),
    )
    return tokenized_steps


def main() -> None:
    config = draccus.parse(ExecutorMainConfig)
    if config.prefix is None:
        config = dataclasses.replace(config, prefix=STAGING_PREFIX)
    os.environ["MARIN_PREFIX"] = config.prefix
    phase = os.environ.get("TOKENIZER_SWEEP_PHASE", "all")
    _skip_executor_info_writes_for_generated_dag()
    executor_main(config, build_steps(phase), max_concurrent=MAX_STEP_CONCURRENCY)


if __name__ == "__main__":
    configure_logging()
    main()
