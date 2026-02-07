# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import partial
import logging
import os
from typing import TypedDict
import humanfriendly
from marin.utilities.time_logger import log_time
import pyarrow as pa
import pyarrow.json as pa_json

from fray.v2.local_backend import LocalClient
from marin.utilities.wandb_utils import init_wandb
from marin.execution.executor import THIS_OUTPUT_PATH
from marin.utils import fsspec_glob
from zephyr import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.expr import col
from zephyr.readers import SUPPORTED_EXTENSIONS, open_file

logger = logging.getLogger(__name__)


class DedupMode(StrEnum):
    """Mode in which deduplication is performed"""

    EXACT_PARAGRAPH = auto()
    """
    Identify exact duplicate paragraphs within documents.
    """
    EXACT_DOCUMENT = auto()
    """
    Identify exact duplicate documents.
    """
    FUZZY_DOCUMENT = auto()
    """
    Identify documents that are similar but not necessarily identical.
    """


@dataclass(frozen=True)
class DedupConfig:
    """
    Configuration class for running deduplication on docs using Zephyr.

    Attributes:
        input_paths: Path(s) of files to apply deduplication to. This could be across multiple directories/datasets.
        filetypes: File extensions to consider when collecting input files.
        output_path: Path for storing results of deduplication (char spans in docs that are duplicate)
        processes: number of processes to use for deduplication
        mode: switch between decontamination (build filter) and regular deduplication
        text_field: field to use for text content in Parquet files
        fuzzy_minhash_num_perms: Number of permutations for MinHash signature.
            Must be divisible by fuzzy_minhash_num_bands. Defaults are from OLMo 3: 26 bands x 11 rows = 286.
        fuzzy_minhash_num_bands: Number of bands for LSH. More bands = higher recall, lower precision.
        fuzzy_minhash_ngram_size: Size of character n-grams/shingles to extract from text.
        fuzzy_minhash_seed: Random seed for MinHash permutation generation.
    """

    input_paths: str | list[str]
    filetypes: list[str] = field(default_factory=lambda: ["jsonl", "jsonl.gz", "jsonl.zst", "parquet"])
    output_path: str = THIS_OUTPUT_PATH
    processes: int = 1
    mode: DedupMode = DedupMode.EXACT_PARAGRAPH
    # field to use for text content in Parquet files
    text_field: str = "text"
    ray_num_cpus: int = 2
    ray_memory: int = humanfriendly.parse_size("64GB", binary=True)
    # MinHash LSH parameters (only used for FUZZY_DOCUMENT mode)
    fuzzy_minhash_num_perms: int = 286
    fuzzy_minhash_num_bands: int = 26
    fuzzy_minhash_ngram_size: int = 5
    fuzzy_minhash_seed: int = 42


def deduplicate(config: DedupConfig):
    """Main entry point for deduplication"""
    if config.mode == DedupMode.EXACT_PARAGRAPH:
        from marin.processing.classification.deduplication.exact import dedup_exact_paragraph

        return dedup_exact_paragraph(config)
    elif config.mode == DedupMode.EXACT_DOCUMENT:
        from marin.processing.classification.deduplication.exact import dedup_exact_document

        return dedup_exact_document(config)
    elif config.mode == DedupMode.FUZZY_DOCUMENT:
        from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

        return dedup_fuzzy_document(config)
    else:
        raise ValueError(f"Unknown mode {config.mode}")


@dataclass
class DupCounters:
    # TODO (rav): make both method and level Enums
    method: str
    level: str
    total: int = 0
    dups: int = 0
    unique: int = 0
    dup_clusters: int = 0

    def __add__(self, other: "DupCounters") -> "DupCounters":
        assert isinstance(other, DupCounters)

        return DupCounters(
            method=self.method,
            level=self.level,
            total=self.total + other.total,
            dups=self.dups + other.dups,
            unique=self.unique + other.unique,
            dup_clusters=self.dup_clusters + other.dup_clusters,
        )

    def __str__(self) -> str:
        if self.total == 0:
            return f"{self.level} total: 0"
        return (
            f"{self.method.capitalize()} {self.level.lower()} total: {self.total:,}, "
            f"dups: {self.dups:,} ({self.dups / self.total:.2%}), unique: {self.unique:,}, "
            f"dup_clusters: {self.dup_clusters:,}"
        )

    def to_dict(self):
        return {
            f"dedup/{self.method}/{self.level}/total": self.total,
            f"dedup/{self.method}/{self.level}/dups": self.dups,
            f"dedup/{self.method}/{self.level}/unique": self.unique,
            f"dedup/{self.method}/{self.level}/dup_clusters": self.dup_clusters,
        }


def _collect_input_files(*, input_paths: str | list[str], filetypes: list[str]) -> list[str]:
    """Given an input path or list of paths, collect all matching files"""
    input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
    all_files = []
    ext_glob = ",".join(set(filetypes))
    for path in input_paths:
        logger.info(f"Collecting files from path: {path}")
        files = fsspec_glob(f"{path.rstrip('/')}/**/*.{{{ext_glob}}}")
        if files:
            all_files.extend(files)
        else:
            if not any(path.endswith(ext) for ext in filetypes):
                raise FileNotFoundError(f"No files found in path: {path}")
            all_files.append(path)  # Assume it's a single file
    assert all_files, "No input files found for deduplication."
    return all_files


def _init_wandb(config: DedupConfig):
    """Initialize wandb for deduplication tracking."""
    init_wandb(
        run_name=f"{config.mode}",
        tags=[str(config.mode)],
        config={
            "mode": str(config.mode),
            "input_path": config.input_paths,
            "processes": config.processes,
        },
    )


def _get_extension(file_path: str) -> str:
    for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if file_path.endswith(ext):
            return ext
    raise ValueError(f"Unsupported extension: {file_path}.")


def _load_batches(file_path: str, columns: list[str] | None = None, **parquet_kwargs) -> Iterator[pa.RecordBatch]:
    """
    Load file contents as PyArrow RecordBatches.

    This is useful to feed the pyarrow into rust using zero-copy batches.

    Args:
        file_path: Path to the input file (parquet, jsonl, jsonl.gz, or jsonl.zst)
        columns: Optional list of columns to read (parquet only)
        **parquet_kwargs: Additional kwargs passed to ParquetFile.iter_batches()

    Yields:
        pa.RecordBatch objects containing the file data
    """
    if not file_path.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported extension: {file_path}.")
    with open_file(file_path, "rb") as f:
        if file_path.endswith(".parquet"):
            import pyarrow.parquet as pq

            if columns is not None:
                parquet_kwargs = {**parquet_kwargs, "columns": columns}

            parquet_file = pq.ParquetFile(f)
            yield from parquet_file.iter_batches(**parquet_kwargs)
        else:
            yield from pa_json.read_json(f).to_batches()


def _load_dupe_map_shard(shards: list[str]) -> dict[str, dict[str, str]]:
    shard_dup_map = {}

    def add_to_dup_map(record: dict):
        shard_dup_map[record["hash"]] = {"canonical": record["canonical"]}

    with log_time(f"Load duplicate map from {len(shards)} shards"):
        with ZephyrContext(client=LocalClient()) as ctx:
            ctx.execute(
                Dataset.from_list(shards)
                .load_parquet()
                .select("hash", "canonical")
                .filter(col("hash").is_not_null())
                .map(add_to_dup_map),
            )

    return shard_dup_map


def _find_base_path(input_path: str | list[str], input_files: list[str]) -> str:
    # Determine base path for rebasing
    base_path = input_path[0] if isinstance(input_path, list) else input_path
    if base_path in input_files:
        # NOTE: if the base_path is in the input_files, means it's a specific file, so rebase to its directory
        base_path = os.path.dirname(base_path)
    return base_path


def _compute_dedup_stats(shards: list[str], method: str, level: str) -> DupCounters:
    with log_time(f"Compute deduplication stats from {len(shards)} shards"):
        with ZephyrContext(client=LocalClient()) as ctx:
            result: DupCounters = ctx.execute(  # type: ignore[bad-assignment]
                Dataset.from_list(shards)
                .load_parquet()
                .select("cnt")
                .map(
                    lambda c: DupCounters(
                        method=method,
                        level=level,
                        total=c["cnt"],
                        dups=c["cnt"] if c["cnt"] > 1 else 0,
                        unique=int(c["cnt"] == 1),
                        dup_clusters=int(c["cnt"] > 1),
                    )
                )
                .reduce(partial(sum, start=DupCounters(method=method, level=level))),
            )[0]
    return result


class DupeReduceResult(TypedDict):
    hash: str | None
    cnt: int
    canonical: str | None


def _count_reduce(key: str, items: Iterator[pa.StructScalar], *, canonical_id: str) -> DupeReduceResult:
    head = next(items)
    doc_cnt = sum(map(lambda _: 1, items)) + 1
    if doc_cnt == 1:
        return {
            "hash": None,
            "cnt": 1,
            "canonical": None,
        }

    return {
        "hash": key,
        "cnt": doc_cnt,
        "canonical": head[canonical_id],
    }
