# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum, auto
import logging
import os
import pyarrow as pa
import pyarrow.json as pa_json

from fray.v2 import ResourceConfig
from marin.utilities.wandb_utils import init_wandb
from marin.execution.executor import THIS_OUTPUT_PATH
from marin.utils import fsspec_glob
from zephyr.readers import SUPPORTED_EXTENSIONS, open_file

logger = logging.getLogger(__name__)

DEFAULT_FILETYPES: list[str] = ["jsonl", "jsonl.gz", "jsonl.zst", "parquet"]


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
    filetypes: list[str] = field(default_factory=lambda: list(DEFAULT_FILETYPES))
    output_path: str = THIS_OUTPUT_PATH
    processes: int = 1
    mode: DedupMode = DedupMode.EXACT_PARAGRAPH
    # field to use for text content in Parquet files
    text_field: str = "text"
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="32g", disk="5g"))
    # MinHash LSH parameters (only used for FUZZY_DOCUMENT mode)
    fuzzy_minhash_num_perms: int = 286
    fuzzy_minhash_num_bands: int = 26
    fuzzy_minhash_ngram_size: int = 5
    fuzzy_minhash_seed: int = 42


def deduplicate(config: DedupConfig):
    """Main entry point for deduplication. Unpacks config and dispatches to mode-specific functions."""
    if config.mode == DedupMode.EXACT_PARAGRAPH:
        from marin.processing.classification.deduplication.exact import dedup_exact_paragraph

        return dedup_exact_paragraph(
            input_paths=config.input_paths,
            output_path=config.output_path,
            text_field=config.text_field,
            filetypes=config.filetypes,
            worker_resources=config.worker_resources,
        )
    elif config.mode == DedupMode.EXACT_DOCUMENT:
        from marin.processing.classification.deduplication.exact import dedup_exact_document

        return dedup_exact_document(
            input_paths=config.input_paths,
            output_path=config.output_path,
            text_field=config.text_field,
            filetypes=config.filetypes,
            worker_resources=config.worker_resources,
        )
    elif config.mode == DedupMode.FUZZY_DOCUMENT:
        from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

        return dedup_fuzzy_document(
            input_paths=config.input_paths,
            output_path=config.output_path,
            text_field=config.text_field,
            filetypes=config.filetypes,
            fuzzy_minhash_num_perms=config.fuzzy_minhash_num_perms,
            fuzzy_minhash_num_bands=config.fuzzy_minhash_num_bands,
            fuzzy_minhash_ngram_size=config.fuzzy_minhash_ngram_size,
            fuzzy_minhash_seed=config.fuzzy_minhash_seed,
            worker_resources=config.worker_resources,
        )
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

    def __add__(self, other: "DupCounters") -> "DupCounters":
        assert isinstance(other, DupCounters)

        return DupCounters(
            method=self.method,
            level=self.level,
            total=self.total + other.total,
            dups=self.dups + other.dups,
            unique=self.unique + other.unique,
        )

    def __str__(self) -> str:
        if self.total == 0:
            return f"{self.level} total: 0"
        return (
            f"{self.method.capitalize()} {self.level.lower()} total: {self.total:,}, "
            f"dups: {self.dups:,} ({self.dups / self.total:.2%}), unique: {self.unique:,}"
        )

    def to_dict(self):
        return {
            f"dedup/{self.method}/{self.level}/total": self.total,
            f"dedup/{self.method}/{self.level}/dups": self.dups,
            f"dedup/{self.method}/{self.level}/unique": self.unique,
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


def _init_wandb(*, mode: DedupMode, input_paths: str | list[str], processes: int = 1):
    """Initialize wandb for deduplication tracking."""
    init_wandb(
        run_name=f"{mode}",
        tags=[str(mode)],
        config={
            "mode": str(mode),
            "input_path": input_paths,
            "processes": processes,
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


def _find_base_path(input_path: str | list[str], input_files: list[str]) -> str:
    # Determine base path for rebasing
    base_path = input_path[0] if isinstance(input_path, list) else input_path
    if base_path in input_files:
        # NOTE: if the base_path is in the input_files, means it's a specific file, so rebase to its directory
        base_path = os.path.dirname(base_path)
    return base_path
