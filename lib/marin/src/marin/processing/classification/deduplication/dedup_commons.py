# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Callable, Iterator
from enum import StrEnum, auto

import pyarrow as pa
import pyarrow.json as pa_json
import wandb
from zephyr import counters, write_parquet_file
from zephyr.readers import SUPPORTED_EXTENSIONS, open_file

from marin.utilities.wandb_utils import init_wandb
from marin.utils import fsspec_glob, rebase_file_path

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


def _aggregate_shard_counters(shard_results: list[dict], method: str, level: str) -> dict[str, int]:
    """Aggregate per-shard counter dicts into a single counter dict."""
    total = sum(r["total"] for r in shard_results)
    dups = sum(r["dups"] for r in shard_results)
    unique = sum(r["unique"] for r in shard_results)
    return {
        f"dedup/{method}/{level}/total": total,
        f"dedup/{method}/{level}/dups": dups,
        f"dedup/{method}/{level}/unique": unique,
    }


def _collect_input_files(*, input_paths: str | list[str], filetypes: list[str]) -> list[str]:
    """Given an input path or list of paths, collect all matching files and return them sorted."""
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
    return sorted(all_files)


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
            # block_size must be >= the largest single JSON line in the file
            read_options = pa_json.ReadOptions(block_size=64 * 1024 * 1024)  # 64 MB
            yield from pa_json.open_json(f, read_options=read_options)


def _find_base_path(input_path: str | list[str], input_files: list[str]) -> str:
    # Determine base path for rebasing
    if isinstance(input_path, list):
        # Use common ancestor so rebase_file_path never generates ".." segments in GCS paths.
        # os.path.commonpath works on GCS paths since it operates on string prefixes.
        base_path = os.path.commonpath(input_path) if len(input_path) > 1 else input_path[0]
    else:
        base_path = input_path
    if base_path in input_files:
        # NOTE: if the base_path is in the input_files, means it's a specific file, so rebase to its directory
        base_path = os.path.dirname(base_path)
    return base_path


def make_document_dedup_aggregator(
    *,
    idx_to_path: dict[int, str],
    input_paths: str | list[str],
    output_path: str,
    counter_prefix: str,
) -> Callable[[int, Iterator[dict]], dict]:
    """Return a group_by reducer that counts dedup stats and writes parquet output.

    The returned callable maps ``(file_idx, records) -> dict`` with keys
    ``total``, ``dups``, ``unique`` plus whatever ``write_parquet_file`` returns.

    Used identically by both exact-document and fuzzy-document dedup.
    """

    def aggregate(file_idx: int, records: Iterator[dict]) -> dict:
        input_path = idx_to_path[file_idx]
        output_file = rebase_file_path(
            _find_base_path(input_paths, [input_path]),
            input_path,
            f"{output_path}/data/",
            old_extension=_get_extension(input_path),
            new_extension=".parquet",
        )

        total = 0
        dups = 0

        def counting_iter():
            nonlocal total, dups
            for record in records:
                is_dup: bool = record["is_dup"]
                total += 1
                counters.increment(f"{counter_prefix}/total")
                if is_dup:
                    dups += 1
                    counters.increment(f"{counter_prefix}/dups")
                else:
                    counters.increment(f"{counter_prefix}/unique")
                yield record

        def only_dups(records: Iterator[dict]) -> Iterator[dict]:
            for record in records:
                if record["is_dup"]:
                    yield {"id": record["id"], "attributes": {"dup_doc": True}}

        result = write_parquet_file(only_dups(counting_iter()), output_file)
        return {**result, "total": total, "dups": dups, "unique": total - dups}

    return aggregate


def finalize_dedup(shard_results: list[dict], mode: DedupMode, method: str, level: str) -> dict:
    """Aggregate shard counters, log summary, finish wandb, and return result dict.

    Shared epilogue for all three dedup entry points.
    """
    counter_dict = _aggregate_shard_counters(shard_results, method=method, level=level)
    logger.info(
        "%s %s total: %s, dups: %s, unique: %s",
        method.capitalize(),
        level,
        counter_dict[f"dedup/{method}/{level}/total"],
        counter_dict[f"dedup/{method}/{level}/dups"],
        counter_dict[f"dedup/{method}/{level}/unique"],
    )

    if wandb.run:
        wandb.log(counter_dict)
        wandb.finish()

    return {"success": True, "mode": str(mode)} | counter_dict
