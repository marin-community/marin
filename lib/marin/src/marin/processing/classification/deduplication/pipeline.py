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

"""
Deduplication using rbloom bloom filters and zephyr streaming.

This module provides two deduplication workflows:
1. DEDUPLICATE: Remove duplicate paragraphs within a dataset
2. EXACT_DOC_DEDUPLICATE: Remove duplicate documents based on full text hash
"""

from functools import partial
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum, auto
import typing

from marin.execution.executor import THIS_OUTPUT_PATH
from marin.processing.classification.deduplication.connected_components import connected_components
from marin.processing.classification.deduplication.minhash_lsh import minhash_lsh
from marin.utilities.time_logger import log_time
import pyarrow as pa
import pyarrow.json as pa_json
import draccus
import wandb

from marin.utilities.wandb_utils import WANDB_PROJECT, WANDB_ENTITY

from marin.utils import fsspec_glob, rebase_file_path
from zephyr import Dataset, col, flow_backend
from zephyr.backend_factory import create_backend
from zephyr.readers import load_file, open_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class DedupMode(StrEnum):
    EXACT_PARAGRAPH_DEDUPLICATE = auto()
    DOCUMENT_DEDUPLICATE = auto()


@dataclass(frozen=True)
class DedupeConfig:
    """
    Configuration class for running deduplication on docs using Zephyr.

    Deduplication will identify spans of text in documents that are duplicate.

    Attributes:
        input_path (str | list[str]): Path(s) of files to apply deduplication to.
        output_path (str): Path for storing results of deduplication (char spans in docs that are duplicate)
        attribute_name (str): Name for key to store duplicate span info in json
        processes (int): number of processes to use for deduplication
        mode (DedupMode): switch between decontamination (build filter) and regular deduplication
        text_field (str): field to use for text content in Parquet files
    """

    # TODO (rav): had to make this optional to avoid default argument issues in dataclass, what is the
    #   best way to handle this in marin and draccus?
    input_path: str | list[str]
    output_path: str = THIS_OUTPUT_PATH
    # TODO: remove this and just hard code the attribute names
    attribute_name: str = "duplicate_text"
    processes: int = 1
    mode: DedupMode = DedupMode.EXACT_PARAGRAPH_DEDUPLICATE
    # field to use for text content in Parquet files
    text_field: str = "text"


def _collect_input_files(input_path: str | list[str]) -> list[str]:
    """
    Given an input path or list of paths, collect all matching files (jsonl, parquet, etc).
    """
    input_paths = input_path if isinstance(input_path, list) else [input_path]
    all_files = []
    for path in input_paths:
        logger.info(f"Collecting files from path: {path}")
        files = fsspec_glob(f"{path.rstrip('/')}/**/*.{{jsonl,jsonl.gz,jsonl.zst,parquet}}")
        if files:
            all_files.extend(files)
        else:
            if not path.endswith(("jsonl", "jsonl.gz", "jsonl.zst", "parquet")):
                raise FileNotFoundError(f"No files found in path: {path}")
            all_files.append(path)  # Assume it's a single file
    assert all_files, "No input files found for deduplication."
    return all_files


def _init_wandb(config: DedupeConfig, tags: list[str] | None = None):
    """
    Initialize wandb if configured.

    Args:
        config: DedupeConfig containing wandb settings
        tags: Additional tags to add beyond those in config
    """
    if "WANDB_API_KEY" not in os.environ:
        return

    run_name = os.environ.get("WANDB_RUN_NAME")
    if not run_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_name = f"{config.mode}-{timestamp}"

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        tags=[str(config.mode)] + (tags or []),
        config={
            "mode": str(config.mode),
            "input_path": config.input_path,
            "processes": config.processes,
        },
    )


def _get_extension(file_path: str) -> str:
    for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if file_path.endswith(ext):
            return ext
    raise ValueError(f"Unsupported extension: {file_path}.")


def _load_batches(file_path: str, columns: list[str] | None = None, **parquet_kwargs) -> Iterator[pa.RecordBatch]:
    # Private function for now to isolate the `pa.RecordBatch` experiment
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
        create_backend("threadpool").execute(
            Dataset.from_list(shards)
            .load_parquet()
            .select("hash", "canonical")
            .filter(col("hash").is_not_null())
            .map(add_to_dup_map)
        )

    return shard_dup_map


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
            f"dups: {self.dups:,} ({self.dups/self.total:.2%}), unique: {self.unique:,}, "
            f"dup_clusters: {self.dup_clusters:,}"
        )

    def to_dict(self):
        return {
            f"dedup/{self.method}/{self.level}/total": self.total,
            f"dedup/{self.method}/{self.level}/dups": self.dups,
            f"dedup/{self.method}/{self.level}/unique": self.unique,
            f"dedup/{self.method}/{self.level}/dup_clusters": self.dup_clusters,
        }


def _compute_dedup_stats(shards: list[str], method: str, level: str) -> DupCounters:
    with log_time(f"Compute deduplication stats from {len(shards)} shards"):
        result: DupCounters = create_backend("threadpool").execute(  # type: ignore[bad-assignment]
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
            .reduce(partial(sum, start=DupCounters(method=method, level=level)))
        )[0]
    return result


class DupeReduceResult(typing.TypedDict):
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


def _find_base_path(input_path: str | list[str], input_files: list[str]) -> str:
    # Determine base path for rebasing
    base_path = input_path[0] if isinstance(input_path, list) else input_path
    if base_path in input_files:
        # NOTE: if the base_path is in the input_files, means it's a specific file, so rebase to its directory
        base_path = os.path.dirname(base_path)
    return base_path


def _run_deduplication(config: DedupeConfig):
    import dupekit
    from dupekit import Transformation

    input_files = _collect_input_files(config.input_path)

    backend = flow_backend(max_parallelism=config.processes)
    _init_wandb(config, tags=["paragraph"])

    def compute_paragraph_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="resolved_id"),
            Transformation.SplitParagraphs(text_col=config.text_field, id_col="resolved_id"),
            Transformation.Hash(input_col="paragraph_text", output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            Transformation.SelectColumns(columns=["hash", "doc_id"]),
        ]
        return dupekit.transform(batch, pipeline)

    # first compute the full set of duplicate keys.
    duplicate_key_shards = list(
        backend.execute(
            Dataset.from_list(input_files).flat_map(_load_batches)
            # NOTE: when do we want to trigger reshard. Keep in mind that reshard will materialize the
            #   text field!
            # TODO: the resharding logic should be improved, based on size and/or max_parallelism
            .reshard(num_shards=config.processes if len(input_files) > 3 and len(input_files) < 42 else None)
            .map(compute_paragraph_hashes)
            .flat_map(lambda batch: batch.to_pylist())
            .group_by(
                lambda key_fn: key_fn["hash"],
                partial(_count_reduce, canonical_id="doc_id"),
                num_output_shards=42,
            )
            .write_parquet(f"{config.output_path}/metadata/dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
            verbose=True,
        ),
    )

    exact_cnts = _compute_dedup_stats(duplicate_key_shards, method="exact", level="paragraph")
    logger.info(str(exact_cnts))

    if wandb.run:
        wandb.log(exact_cnts.to_dict())

    def mark_exact_dups_paragraphs(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Mark duplicate paragraphs in a single record using exact hash matching."""

        dup_map = _load_dupe_map_shard(duplicate_key_shards)

        for batch in batches:
            yield dupekit.mark_paragraph_duplicates(
                batch,
                dup_map,
                config.attribute_name,
                algorithm=dupekit.HashAlgorithm.Xxh3_128,
            )

    base_path = _find_base_path(config.input_path, input_files)
    backend.execute(
        Dataset.from_list(input_files)
        .flat_map(_load_batches)
        .map_shard(mark_exact_dups_paragraphs)
        .flat_map(lambda batch: batch.to_pylist())
        .write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{config.output_path}/data",
                old_extension=_get_extension(input_files[shard_idx]),
            ),
            skip_existing=True,
        )
    )

    if wandb.run:
        wandb.finish()

    return {"success": True, "mode": "deduplication"} | exact_cnts.to_dict()


def _compute_fuzzy_dedup_stats(shards: list[str], method: str, level: str) -> DupCounters:
    with log_time(f"Compute fuzzy deduplication stats from {len(shards)} shards"):
        result: DupCounters = create_backend("threadpool").execute(  # type: ignore[bad-assignment]
            Dataset.from_list(shards)
            .load_parquet(columns=["component_id"])
            # Compute the per-component statistics and then roll them up into a single counter group
            .group_by(
                key=lambda r: r["component_id"],
                reducer=lambda _, items: DupCounters(
                    method=method,
                    level=level,
                    total=(total := sum(1 for _ in items)),
                    dups=total if total > 1 else 0,
                    unique=1,
                    dup_clusters=int(total > 1),
                ),
            )
            .reduce(partial(sum, start=DupCounters(method=method, level=level)))
        )[0]
    return result


def _load_fuzzy_dupe_map_shard(shards: list[str]) -> dict[str, bool]:
    if not shards:
        logger.warning("No fuzzy duplicate documents found.")
        return {}

    # Map record ID -> is duplicate (bool)
    shard_dup_map = {}

    def add_to_dup_map(record: dict):
        shard_dup_map[record["id"]] = record["fuzzy_duplicate"]

    with log_time(f"Load fuzzy duplicate map from {len(shards)} shards"):
        create_backend("threadpool").execute(Dataset.from_list(shards).load_parquet().map(add_to_dup_map))

    return shard_dup_map


def _run_doc_deduplication(config: DedupeConfig):
    """
    Exact document deduplication: identify duplicate documents based on full text hash.
    This is a temporary implementation, primarily to compare directly with the Ai2 duplodocus.
    """
    import dupekit
    from dupekit import Transformation

    input_files = _collect_input_files(config.input_path)

    backend = flow_backend(max_parallelism=config.processes)
    _init_wandb(config, tags=["exact-doc"])

    def compute_document_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="resolved_id"),
            Transformation.Hash(input_col=config.text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            Transformation.SelectColumns(columns=["hash", "resolved_id"]),
        ]
        return dupekit.transform(batch, pipeline)

    # first compute the full set of duplicate keys.
    duplicate_key_shards = list(
        backend.execute(
            Dataset.from_list(input_files).flat_map(_load_batches)
            # NOTE: when do we want to trigger reshard. Keep in mind that reshard will materialize the
            #   text field!
            # TODO: the resharding logic should be improved, based on size and/or max_parallelism
            .reshard(num_shards=config.processes if len(input_files) > 3 and len(input_files) < 42 else None)
            .map(compute_document_hashes)
            .flat_map(lambda batch: batch.to_pylist())
            .group_by(
                lambda key_fn: key_fn["hash"],
                partial(_count_reduce, canonical_id="resolved_id"),
                num_output_shards=42,
            )
            .write_parquet(f"{config.output_path}/metadata/dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
            verbose=True,
        )
    )

    exact_cnts = _compute_dedup_stats(duplicate_key_shards, method="exact", level="document")
    logger.info(str(exact_cnts))

    doc_minhash_lsh = minhash_lsh(
        Dataset.from_list(input_files)
        .flat_map(load_file)
        .reshard(num_shards=config.processes if len(input_files) < 42 else None)
    )
    converged, cc_files = connected_components(
        doc_minhash_lsh, backend=backend, output_dir=f"{config.output_path}/metadata/cc"
    )
    if not converged:
        # TODO (rav): log the number of changed nodes?
        logger.warning("Connected components did not converge")
    fuzzy_dup_shards = backend.execute(
        Dataset.from_list(cc_files)
        .flat_map(load_file)
        .map(
            lambda r: {
                "id": r["node_id"]["record_id"],
                "fuzzy_duplicate": r["component_id"] != r["node_id"]["record_id_norm"],
            }
        )
        .reshard(num_shards=42)
        .write_parquet(f"{config.output_path}/metadata/fuzzy-dup-key-{{shard:05d}}-of-{{total:05d}}.parquet")
    )

    fuzzy_cnt = _compute_fuzzy_dedup_stats(cc_files, method="fuzzy", level="document")
    logger.info(str(fuzzy_cnt))

    assert (
        exact_cnts.total == fuzzy_cnt.total
    ), f"Exact ({exact_cnts.total}) and fuzzy ({fuzzy_cnt.total}) dedup counts do not match!"

    if wandb.run:
        wandb.log(exact_cnts.to_dict() | fuzzy_cnt.to_dict())

    def mark_dup_documents(batches: Iterator[pa.RecordBatch]) -> Iterator[dict]:
        """Mark exact duplicate documents using exact hash matching."""
        dup_map = _load_dupe_map_shard(duplicate_key_shards)
        fuzzy_dup_map = _load_fuzzy_dupe_map_shard(fuzzy_dup_shards)

        for batch in batches:
            prepared_batch = dupekit.transform(
                batch,
                [
                    Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="id"),
                    Transformation.Hash(
                        input_col=config.text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128
                    ),
                ],
            )
            b = dupekit.mark_document_duplicates(prepared_batch, dup_map, config.attribute_name, hash_col="hash")
            for r in b.to_pylist():
                is_fuzzy_dup = fuzzy_dup_map.get(r["id"], False)
                # TODO: accept fuzzy_duplicate as config option?
                r["attributes"]["fuzzy_duplicate"] = is_fuzzy_dup
                yield r

    base_path = _find_base_path(config.input_path, input_files)
    backend.execute(
        Dataset.from_list(input_files).flat_map(_load_batches)
        # NOTE/TODO: we can't reshard here to increase parallelism because afaiu we want to match
        # the shards of the input files for rebase_file_path to work correctly.
        .map_shard(mark_dup_documents).write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{config.output_path}/data",
                old_extension=_get_extension(input_files[shard_idx]),
            ),
            skip_existing=True,
        ),
        verbose=True,
    )

    if wandb.run:
        wandb.finish()

    return {"success": True, "mode": str(DedupMode.DOCUMENT_DEDUPLICATE)} | exact_cnts.to_dict() | fuzzy_cnt.to_dict()


def deduplicate(config: DedupeConfig):
    """Main entry point for deduplication workflows."""
    if config.mode == DedupMode.EXACT_PARAGRAPH_DEDUPLICATE:
        return _run_deduplication(config)
    elif config.mode == DedupMode.DOCUMENT_DEDUPLICATE:
        return _run_doc_deduplication(config)
    else:
        raise ValueError(f"Unknown mode {config.mode}")


@draccus.wrap()
def main(config: DedupeConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    result = deduplicate(config)
    print(f"Deduplication completed: {result}")


if __name__ == "__main__":
    main()
