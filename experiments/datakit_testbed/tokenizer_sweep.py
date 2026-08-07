# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable tokenizer sweep for Marin issue #5821-style experiments.

This builds proportional windows from a normalized Datakit corpus:

* randomized tokenizer-training corpus
* held-out corpus to tokenize with the trained vocabularies
* optional train-corpus retokenization for downstream model runs

By default, this reproduces the Llama/GPT-OSS tokenizer sweep from issue #5821:
initialize from upstream tokenizer repositories, train a 262k tokenizer on a
50B-token-equivalent sample, then derive 128k, 32k, and 8k vocabularies from
that same trained tokenizer. Typed config below makes the corpus, windows,
vocab sizes, and tokenizer families configurable for future sweeps.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import draccus
import fsspec
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from huggingface_hub import __version__ as hf_hub_version
from levanter.tokenizers import TokenizerBackend
from marin.datakit.normalize import NormalizedData
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import StoragePath, open_url, url_to_fs
from rigging.log_setup import configure_logging
from tokenizers import Regex, pre_tokenizers
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS = 510
PLACE_ALIGNED_DIGIT_CHUNK_SIZE = 3
PLACE_ALIGNED_DIGIT_PRETOKENIZER_REVISION = "bounded-leading-triplets-v2"
DEFAULT_TOKENIZER_SWEEP_TPU_TYPES = ("v5p-8", "v6e-4", "v4-8", "v5litepod-4")


def _default_tokenizer_sweep_tpu_types() -> list[str]:
    return list(DEFAULT_TOKENIZER_SWEEP_TPU_TYPES)


@dataclass(frozen=True)
class ResourceConfigSpec:
    cpu: int
    ram: str
    disk: str = "16g"
    preemptible: bool = True
    tpu_types: list[str] = field(default_factory=list)

    def to_resource_config(self, regions: Sequence[str]) -> ResourceConfig:
        if self.tpu_types:
            return ResourceConfig.with_tpu(
                self.tpu_types,
                cpu=self.cpu,
                ram=self.ram,
                disk=self.disk,
                regions=regions,
                preemptible=self.preemptible,
            )
        return ResourceConfig(
            cpu=self.cpu,
            ram=self.ram,
            disk=self.disk,
            regions=regions,
            preemptible=self.preemptible,
        )


@dataclass(frozen=True)
class CorpusConfig:
    normalized_base: str = "gs://marin-eu-west4/data/datakit/sample/2026-05-26"
    total_tokenized_tokens: int = 1_099_611_681_172


@dataclass(frozen=True)
class WindowConfig:
    tokens: int
    start_tokens: int = 0
    sample_mode: str = "contiguous"

    def fraction(self, corpus: CorpusConfig) -> float:
        return self.tokens / corpus.total_tokenized_tokens

    def start_fraction(self, corpus: CorpusConfig) -> float:
        return self.start_tokens / corpus.total_tokenized_tokens


@dataclass(frozen=True)
class HfTokenizerFamilyConfig:
    name: str
    base_tokenizer: str
    place_aligned_digits: bool = False
    train_new_from_iterator: bool = True


@dataclass(frozen=True)
class TokenizerSweepConfig:
    run_id: str = "tokenizer-sweep-issue-5821"
    staging_prefix: str = "gs://marin-eu-west4"
    max_step_concurrency: int = 24
    phase: str = "all"
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    tokenizer_train: WindowConfig = field(
        default_factory=lambda: WindowConfig(tokens=50_000_000_000, sample_mode="random-shards")
    )
    holdout: WindowConfig = field(
        default_factory=lambda: WindowConfig(tokens=100_000_000_000, start_tokens=100_000_000_000)
    )
    train_retokenize: WindowConfig = field(default_factory=lambda: WindowConfig(tokens=50_000_000_000))
    retokenize_train_label: str | None = None
    train_random_seed: int = 5821
    vocab_sizes: list[int] = field(default_factory=lambda: [262_144, 131_072, 32_768, 8_192])
    hf_families: list[HfTokenizerFamilyConfig] = field(
        default_factory=lambda: [
            HfTokenizerFamilyConfig("gpt-oss", "openai/gpt-oss-20b"),
            HfTokenizerFamilyConfig("llama", "meta-llama/Meta-Llama-3.1-8B"),
            HfTokenizerFamilyConfig("gpt-oss-place-digits", "openai/gpt-oss-20b", place_aligned_digits=True),
            HfTokenizerFamilyConfig("llama-place-digits", "meta-llama/Meta-Llama-3.1-8B", place_aligned_digits=True),
        ]
    )
    official_truncated_families: list[HfTokenizerFamilyConfig] = field(default_factory=list)
    family_filter: list[str] | None = None
    size_filter: list[int] | None = None
    regions: list[str] = field(default_factory=lambda: ["europe-west4"])
    resource_revision: str = "regional-highmem-v2"
    sample_resource: ResourceConfigSpec = field(
        default_factory=lambda: ResourceConfigSpec(cpu=1, ram="8g", tpu_types=_default_tokenizer_sweep_tpu_types())
    )
    hf_train_resource: ResourceConfigSpec = field(
        default_factory=lambda: ResourceConfigSpec(
            cpu=64,
            ram="768g",
            disk="1000g",
            preemptible=False,
            tpu_types=_default_tokenizer_sweep_tpu_types(),
        )
    )
    tokenize_worker_resource: ResourceConfigSpec = field(
        default_factory=lambda: ResourceConfigSpec(
            cpu=1,
            ram="10g",
            disk="5g",
            preemptible=True,
            tpu_types=_default_tokenizer_sweep_tpu_types(),
        )
    )
    hf_batch_size: int = 1024
    hf_train_threads: int = 64
    hf_corpus_max_bytes: int = 0

    @property
    def train_label(self) -> str:
        if self.retokenize_train_label:
            return self.retokenize_train_label
        return f"train{_token_count_label(self.train_retokenize.tokens)}"

    def validate(self) -> None:
        if not self.vocab_sizes:
            raise ValueError("vocab_sizes must be non-empty")
        if len(set(self.vocab_sizes)) != len(self.vocab_sizes):
            raise ValueError(f"vocab_sizes must be unique, got {self.vocab_sizes}")
        if self.corpus.total_tokenized_tokens <= 0:
            raise ValueError("corpus.total_tokenized_tokens must be positive")
        for name, window in {
            "tokenizer_train": self.tokenizer_train,
            "holdout": self.holdout,
            "train_retokenize": self.train_retokenize,
        }.items():
            if window.tokens <= 0:
                raise ValueError(f"{name}.tokens must be positive")
            if window.start_tokens < 0:
                raise ValueError(f"{name}.start_tokens must be non-negative")
            if window.start_tokens + window.tokens > self.corpus.total_tokenized_tokens:
                raise ValueError(f"{name} exceeds corpus token budget")
            if window.sample_mode not in {"contiguous", "random-shards"}:
                raise ValueError(f"{name}.sample_mode must be 'contiguous' or 'random-shards'")
        if self.hf_corpus_max_bytes < 0:
            raise ValueError("hf_corpus_max_bytes must be non-negative")
        family_names = [family.name for family in [*self.hf_families, *self.official_truncated_families]]
        if len(set(family_names)) != len(family_names):
            raise ValueError(f"Tokenizer family names must be unique, got {family_names}")


def issue_5821_default_config() -> TokenizerSweepConfig:
    return TokenizerSweepConfig()


def _token_count_label(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    return str(tokens)


def _part_name(idx: int, total: int) -> str:
    return f"part-{idx:05d}-of-{total:05d}.parquet"


def _glob_paths(pattern: str) -> list[str]:
    return [str(path) for path in StoragePath(pattern).glob()]


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
    _dst_fs, dst_path = url_to_fs(dst)
    StoragePath(dst).parent.mkdirs()
    src_fs.copy(src_path, dst_path)
    return int(src_fs.size(src_path) or 0)


def _write_row_range(src: str, dst: str, start_row: int, stop_row: int) -> tuple[int, int]:
    """Write rows ``[start_row, stop_row)`` from one parquet file."""
    if stop_row <= start_row:
        raise ValueError(f"empty row range for {src}: [{start_row}, {stop_row})")

    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    StoragePath(dst).parent.mkdirs()

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

    shards = sorted(_glob_paths(f"{source.main_output_dir.rstrip('/')}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {source.main_output_dir}")

    first_fs, first_path = url_to_fs(shards[0])
    with first_fs.open(first_path, "rb") as first_file:
        rows_per_file = pq.ParquetFile(first_file).metadata.num_rows
    total_rows_est = rows_per_file * len(shards)
    target_rows = max(1, min(total_rows_est, math.ceil(total_rows_est * sample_fraction)))

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

    shards = sorted(_glob_paths(f"{source.main_output_dir.rstrip('/')}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {source.main_output_dir}")

    first_fs, first_path = url_to_fs(shards[0])
    with first_fs.open(first_path, "rb") as first_file:
        rows_per_file = pq.ParquetFile(first_file).metadata.num_rows
    total_rows_est = rows_per_file * len(shards)
    start_row = math.floor(total_rows_est * start_fraction)
    stop_row = min(total_rows_est, math.ceil(total_rows_est * (start_fraction + sample_fraction)))

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
        futures = [
            pool.submit(copy_or_slice, i, shard_idx) for i, shard_idx in enumerate(range(first_shard, last_shard + 1))
        ]
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
    window_tokens: int,
    total_tokenized_tokens: int,
    sample_resources: ResourceConfig,
    sample_mode: str = "contiguous",
    random_seed: int = 0,
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
            "total_tokenized_tokens": total_tokenized_tokens,
            "sample_mode": sample_mode,
            "random_seed": random_seed if sample_mode == "random-shards" else None,
        },
        fn=remote(run, resources=sample_resources),
    )


def existing_normalized_sources(normalized_base: str) -> dict[str, str]:
    """Return all source artifacts that exist in the configured normalized sample."""
    if not normalized_base.startswith("gs://"):
        raise ValueError(f"expected GCS normalized_base, got {normalized_base}")
    fs = fsspec.filesystem("gcs")
    base = normalized_base.removeprefix("gs://").rstrip("/")
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
        source_name = path.removeprefix(f"{normalized_base}/")
        paths[source_name] = path

    if not paths:
        raise ValueError(f"No normalized source artifacts found under {normalized_base}")
    logger.info("Discovered %d existing normalized sources under %s", len(paths), normalized_base)
    return paths


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
        shards.extend(_glob_paths(pattern))
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
    leading_width = PLACE_ALIGNED_DIGIT_CHUNK_SIZE - 1
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(rf"\p{{N}}{{1,{PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS}}}"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(
                Regex(rf"^\p{{N}}{{1,{leading_width}}}" rf"(?=(?:\p{{N}}{{{PLACE_ALIGNED_DIGIT_CHUNK_SIZE}}})+$)"),
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
        token for token, _ in sorted(old_vocab.items(), key=lambda item: item[1]) if token not in set(special_contents)
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
    config: TokenizerSweepConfig,
    family: str,
    base_tokenizer: str,
    train_patterns: list[str],
    place_aligned_digits: bool,
) -> dict:
    base_size = config.vocab_sizes[0]
    corpus_stats: dict[str, int] = {}
    logger.info("Training %s base tokenizer at vocab size %d from %s", family, base_size, base_tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(config.hf_train_threads)
    logger.info("Using HF tokenizer training parallelism with %d Rayon threads", config.hf_train_threads)
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
            batch_size=config.hf_batch_size,
            max_bytes=config.hf_corpus_max_bytes,
            seed=config.train_random_seed,
            stats=corpus_stats,
        ),
        vocab_size=base_size,
        length=None,
        new_special_tokens=[],
    )
    base_local_dir = tempfile.mkdtemp(prefix=f"{family}-{base_size}-")
    tokenizer.save_pretrained(base_local_dir)

    results = {}
    for size in config.vocab_sizes:
        local_dir = base_local_dir
        if size != base_size:
            logger.info("Deriving %s tokenizer at vocab size %d from %d base", family, size, base_size)
            local_dir = tempfile.mkdtemp(prefix=f"{family}-{size}-")
            _derive_hf_bpe_tokenizer_dir(base_local_dir, size, local_dir)

        tokenizer_name = f"marin-community/{config.run_id}-{family}-{size // 1024}k"
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
                    "max_bytes": config.hf_corpus_max_bytes,
                    "sources": len(train_patterns),
                    "shards": corpus_stats.get("shards", 0),
                    "format": "deterministic-shuffled-parquet-text",
                    "seed": config.train_random_seed,
                    "upstream_tokenizer_repo": base_tokenizer,
                },
                "pretokenizer": {
                    "place_aligned_digits": place_aligned_digits,
                    "numeric_run_cap": PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS if place_aligned_digits else None,
                    "numeric_chunk_size": PLACE_ALIGNED_DIGIT_CHUNK_SIZE if place_aligned_digits else None,
                    "revision": PLACE_ALIGNED_DIGIT_PRETOKENIZER_REVISION if place_aligned_digits else None,
                    "issue": "https://github.com/marin-community/marin/issues/4915" if place_aligned_digits else None,
                },
            },
            f,
            indent=2,
        )
    return results


def train_hf_family_step(
    config: TokenizerSweepConfig,
    family_config: HfTokenizerFamilyConfig,
    train_samples: list[StepSpec],
) -> StepSpec:
    train_patterns = [f"{s.output_path}/outputs/main/*.parquet" for s in train_samples]

    def run(output_path: str) -> dict:
        return _train_hf_family(
            output_path,
            config=config,
            family=family_config.name,
            base_tokenizer=family_config.base_tokenizer,
            train_patterns=train_patterns,
            place_aligned_digits=family_config.place_aligned_digits,
        )

    resources = config.hf_train_resource.to_resource_config(config.regions)

    return StepSpec(
        name=f"tokenizers/{config.run_id}/{family_config.name}",
        deps=train_samples,
        hash_attrs={
            "family": family_config.name,
            "base_tokenizer": family_config.base_tokenizer,
            "vocab_sizes": config.vocab_sizes,
            "window_tokens": config.tokenizer_train.tokens,
            "derive_from": f"train {config.vocab_sizes[0]} once, then truncate BPE vocab/merges",
            "resource_revision": config.resource_revision,
            "regions": config.regions,
            "ram": resources.ram,
            "preemptible": resources.preemptible,
            "corpus_max_bytes": config.hf_corpus_max_bytes,
            "batch_size": config.hf_batch_size,
            "train_threads": config.hf_train_threads,
            "tokenizers_parallelism": "true",
            "sampling": "deterministic-shuffled-files-and-row-groups",
            "sampling_seed": config.train_random_seed,
            "place_aligned_digits": family_config.place_aligned_digits,
            "digit_max_run_chars": PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS if family_config.place_aligned_digits else None,
            "digit_chunk_size": PLACE_ALIGNED_DIGIT_CHUNK_SIZE if family_config.place_aligned_digits else None,
            "digit_pretokenizer_revision": (
                PLACE_ALIGNED_DIGIT_PRETOKENIZER_REVISION if family_config.place_aligned_digits else None
            ),
        },
        fn=remote(run, resources=resources),
    )


def _derive_official_hf_family(
    output_path: str,
    *,
    config: TokenizerSweepConfig,
    family: str,
    base_tokenizer: str,
) -> dict:
    base_size = config.vocab_sizes[0]
    logger.info("Deriving %s tokenizer sizes from official %s", family, base_tokenizer)
    base = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    base_local_dir = tempfile.mkdtemp(prefix=f"{family}-{base_size}-")
    base.save_pretrained(base_local_dir)
    derive_sizes = config.size_filter if config.size_filter is not None else config.vocab_sizes

    results = {}
    for size in derive_sizes:
        local_dir = base_local_dir
        if size != base_size:
            logger.info("Deriving %s tokenizer at vocab size %d from official %d base", family, size, base_size)
            local_dir = tempfile.mkdtemp(prefix=f"{family}-{size}-")
            _derive_hf_bpe_tokenizer_dir(base_local_dir, size, local_dir)

        tokenizer_name = f"marin-community/{config.run_id}-{family}-{size // 1024}k"
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


def official_truncated_hf_family_step(config: TokenizerSweepConfig, family_config: HfTokenizerFamilyConfig) -> StepSpec:
    def run(output_path: str) -> dict:
        return _derive_official_hf_family(
            output_path,
            config=config,
            family=family_config.name,
            base_tokenizer=family_config.base_tokenizer,
        )

    resources = config.hf_train_resource.to_resource_config(config.regions)

    return StepSpec(
        name=f"tokenizers/{config.run_id}/{family_config.name}",
        hash_attrs={
            "family": family_config.name,
            "base_tokenizer": family_config.base_tokenizer,
            "vocab_sizes": config.vocab_sizes,
            "derive_sizes": config.size_filter if config.size_filter is not None else config.vocab_sizes,
            "derive_from": "official HF tokenizer, then truncate BPE vocab/merges by rank",
            "resource_revision": config.resource_revision,
            "regions": config.regions,
            "ram": resources.ram,
            "preemptible": resources.preemptible,
        },
        fn=remote(run, resources=resources),
    )


def holdout_tokenize_step(
    *,
    config: TokenizerSweepConfig,
    bucket_name: str,
    sampled_step: StepSpec,
    tokenizer_name: str,
    tokenizer_backend: TokenizerBackend,
    tokenizer_step: StepSpec,
) -> StepSpec:
    def run(output_path: str) -> None:
        tokenize(
            TokenizeConfig(
                train_paths=[f"{sampled_step.output_path}/outputs/main/*.parquet"],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=tokenizer_name,
                tokenizer_backend=tokenizer_backend,
                worker_resources=config.tokenize_worker_resource.to_resource_config(config.regions),
            )
        )

    return StepSpec(
        name=os.path.join("data/datakit", "tokenized", config.run_id, bucket_name),
        deps=[sampled_step, tokenizer_step],
        hash_attrs={
            "tokenizer": tokenizer_name,
            "tokenizer_backend": tokenizer_backend.value,
            "tokenize_worker_resources": dataclasses.asdict(config.tokenize_worker_resource),
            "regions": config.regions,
        },
        fn=run,
    )


def build_steps(config: TokenizerSweepConfig | None = None, phase: str | None = None) -> list[StepSpec]:
    config = config or issue_5821_default_config()
    config.validate()
    phase = phase or config.phase
    family_filter = set(config.family_filter) if config.family_filter is not None else None
    size_filter = set(config.size_filter) if config.size_filter is not None else None
    sample_resources = config.sample_resource.to_resource_config(config.regions)

    sources = existing_normalized_sources(config.corpus.normalized_base)
    train_samples: dict[str, StepSpec] = {}
    holdout_samples: dict[str, StepSpec] = {}
    retokenize_train_samples: dict[str, StepSpec] = {}
    for source_name, normalized_path in sources.items():
        safe_source_name = source_name.replace("/", "__")
        train_samples[source_name] = sample_window_step(
            name=f"data/datakit/tokenizer_sweep/{config.run_id}/train/{safe_source_name}",
            normalized_path=normalized_path,
            start_fraction=0.0,
            sample_fraction=config.tokenizer_train.fraction(config.corpus),
            window_tokens=config.tokenizer_train.tokens,
            total_tokenized_tokens=config.corpus.total_tokenized_tokens,
            sample_resources=sample_resources,
            sample_mode=config.tokenizer_train.sample_mode,
            random_seed=config.train_random_seed,
        )
        retokenize_train_samples[source_name] = sample_window_step(
            name=f"data/datakit/tokenizer_sweep/{config.run_id}/{config.train_label}/{safe_source_name}",
            normalized_path=normalized_path,
            start_fraction=config.train_retokenize.start_fraction(config.corpus),
            sample_fraction=config.train_retokenize.fraction(config.corpus),
            window_tokens=config.train_retokenize.tokens,
            total_tokenized_tokens=config.corpus.total_tokenized_tokens,
            sample_resources=sample_resources,
            sample_mode=config.train_retokenize.sample_mode,
            random_seed=config.train_random_seed,
        )
        holdout_samples[source_name] = sample_window_step(
            name=f"data/datakit/tokenizer_sweep/{config.run_id}/holdout/{safe_source_name}",
            normalized_path=normalized_path,
            start_fraction=config.holdout.start_fraction(config.corpus),
            sample_fraction=config.holdout.fraction(config.corpus),
            window_tokens=config.holdout.tokens,
            total_tokenized_tokens=config.corpus.total_tokenized_tokens,
            sample_resources=sample_resources,
            sample_mode=config.holdout.sample_mode,
            random_seed=config.train_random_seed,
        )

    train_sample_list = list(train_samples.values())
    tokenizer_steps: dict[str, tuple[StepSpec, TokenizerBackend, dict[int, str]]] = {}
    for family_config in config.hf_families:
        if not family_config.train_new_from_iterator:
            raise ValueError(
                f"{family_config.name} has train_new_from_iterator=False; "
                "put official truncation families in official_truncated_families"
            )
        step = train_hf_family_step(config, family_config, train_sample_list)
        tokenizer_steps[family_config.name] = (
            step,
            TokenizerBackend.HF,
            {
                size: f"marin-community/{config.run_id}-{family_config.name}-{size // 1024}k"
                for size in config.vocab_sizes
            },
        )
    for family_config in config.official_truncated_families:
        step = official_truncated_hf_family_step(config, family_config)
        tokenizer_steps[family_config.name] = (
            step,
            TokenizerBackend.HF,
            {
                size: f"marin-community/{config.run_id}-{family_config.name}-{size // 1024}k"
                for size in config.vocab_sizes
            },
        )
    if family_filter is not None:
        unknown = family_filter - set(tokenizer_steps)
        if unknown:
            raise ValueError(f"Unknown family_filter entries: {sorted(unknown)}")
        tokenizer_steps = {family: value for family, value in tokenizer_steps.items() if family in family_filter}
    if size_filter is not None:
        unknown_sizes = size_filter - set(config.vocab_sizes)
        if unknown_sizes:
            raise ValueError(f"Unknown size_filter entries: {sorted(unknown_sizes)}")

    tokenizer_execs = {family: step for family, (step, _, _) in tokenizer_steps.items()}

    if phase == "prep":
        prep_steps = [*holdout_samples.values(), *tokenizer_execs.values()]
        logger.info(
            "Tokenizer sweep prep DAG: %d sources, %d holdout samples, %d tokenizer-training targets",
            len(sources),
            len(holdout_samples),
            len(tokenizer_execs),
        )
        return prep_steps
    if phase not in {"all", "train_tokenization"}:
        raise ValueError(f"Unknown phase={phase!r}; expected 'prep', 'all', or 'train_tokenization'")

    if phase == "train_tokenization":
        sample_steps = retokenize_train_samples
        sample_names = retokenize_train_samples
        output_prefix = config.train_label
    else:
        sample_steps = holdout_samples
        sample_names = holdout_samples
        output_prefix = ""

    tokenized_steps: list[StepSpec] = []
    for family, (_tokenizer_step, backend, names_by_size) in tokenizer_steps.items():
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
                        config=config,
                        bucket_name=bucket_name,
                        sampled_step=sample_steps[source_name],
                        tokenizer_name=tokenizer_name,
                        tokenizer_backend=backend,
                        tokenizer_step=tokenizer_execs[family],
                    )
                )
    logger.info(
        "Tokenizer sweep DAG: %d sources, %d tokenizer families, %d vocab sizes, %d tokenization steps",
        len(sources),
        len(tokenizer_steps),
        len(size_filter or set(config.vocab_sizes)),
        len(tokenized_steps),
    )
    return tokenized_steps


@dataclass(frozen=True)
class TokenizerSweepMainConfig:
    prefix: str | None = None
    dry_run: bool = False
    sweep: TokenizerSweepConfig = field(default_factory=issue_5821_default_config)


def main() -> None:
    config = draccus.parse(TokenizerSweepMainConfig)
    prefix = config.prefix or config.sweep.staging_prefix
    os.environ["MARIN_PREFIX"] = prefix
    StepRunner().run(
        build_steps(config.sweep),
        dry_run=config.dry_run,
        max_concurrent=config.sweep.max_step_concurrency,
    )


if __name__ == "__main__":
    configure_logging()
    main()
