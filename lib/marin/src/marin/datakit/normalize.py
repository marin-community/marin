# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit normalize stage — convert raw data into the datakit standard Parquet format.

The normalize step is the "intake" for the datakit pipeline. It reads raw files
(JSONL, Parquet, or other formats supported by Zephyr), enforces a standard
schema (mandatory ``id`` and ``text`` columns), and writes co-partitioned,
sorted Parquet files.

Key guarantees after normalization:
- Every record has a deterministic ``id`` (SHA-256 of the text content).
- If the source data has an existing ID field, it is preserved as ``source_id``.
- Text is present and UTF-8 encoded.
- Each output partition is sorted by ``id``.
- Output files follow the ``part-{shard:05d}-of-{total:05d}.parquet`` naming convention.
"""

import hashlib
import logging
import os
from collections.abc import Iterator

from marin.execution.artifact import PathsMetadata
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_glob
from zephyr import Dataset, ShardInfo, ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

DEFAULT_TEXT_FIELD = "text"


def content_hash_id(text: str) -> str:
    """Generate a deterministic document ID from text content.

    Uses SHA-256 truncated to 16 hex characters for a compact but
    collision-resistant identifier.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _discover_input_files(input_path: str) -> list[str]:
    """Find all supported input files under input_path, excluding dotfiles/directories."""
    extensions = ["jsonl.gz", "jsonl.zst", "jsonl.zstd", "jsonl", "parquet", "vortex"]
    files: list[str] = []
    for ext in extensions:
        files.extend(fsspec_glob(os.path.join(input_path, f"**/*.{ext}")))
    # Exclude hidden directories (e.g. .metrics/ written by download_hf)
    files = [f for f in files if "/." not in f.split(input_path, 1)[-1]]
    if not files:
        raise ValueError(f"No supported input files found under {input_path}")
    return sorted(files)


def _normalize_record(record: dict, text_field: str, source_id_field: str | None) -> dict:
    """Transform a single record into datakit standard format.

    - Extracts and renames the text field to ``text``.
    - Generates a deterministic ``id`` from the text content.
    - Preserves the original ID (if any) as ``source_id``.
    - Preserves all other fields.
    """
    text = record.get(text_field)
    if text is None:
        raise ValueError(f"Record missing required text field {text_field!r}: {list(record.keys())}")
    if not isinstance(text, str):
        text = str(text)

    doc_id = content_hash_id(text)

    normalized: dict = {"id": doc_id, "text": text}

    if source_id_field is not None and source_id_field in record:
        normalized["source_id"] = str(record[source_id_field])

    # Preserve additional columns
    skip_fields = {text_field, source_id_field} if source_id_field else {text_field}
    for key, value in record.items():
        if key not in skip_fields and key not in normalized:
            normalized[key] = value

    return normalized


def normalize(
    input_path: str,
    output_path: str,
    *,
    text_field: str = DEFAULT_TEXT_FIELD,
    source_id_field: str | None = None,
    num_output_shards: int | None = None,
    zephyr_max_workers: int = 64,
) -> PathsMetadata:
    """Run the normalize pipeline.

    Reads raw files, transforms each record to the standard schema,
    repartitions by ``id`` (hash-based), deduplicates, sorts each partition
    by ``id``, and writes Parquet output files.

    Args:
        input_path: Path to raw input files.
        output_path: Directory to write output Parquet files.
        text_field: Name of the field containing the primary text content.
        source_id_field: Name of an existing ID field to preserve as ``source_id``.
        num_output_shards: Number of output Parquet partitions. Defaults to
            the number of input files.
        zephyr_max_workers: Maximum Zephyr worker parallelism.

    Returns:
        PathsMetadata listing the output files.
    """
    input_files = _discover_input_files(input_path)
    logger.info("Normalizing %d input files from %s", len(input_files), input_path)

    shards = num_output_shards or len(input_files)

    def _sort_shard(records: Iterator[dict], _shard_info: ShardInfo) -> Iterator[dict]:
        batch = list(records)
        batch.sort(key=lambda r: r["id"])
        return iter(batch)

    output_pattern = os.path.join(output_path, "part-{shard:05d}-of-{total:05d}.parquet")
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(load_file)
        .map(lambda r: _normalize_record(r, text_field, source_id_field))
        .group_by(
            key=lambda r: r["id"],
            reducer=lambda _key, records: next(iter(records)),
            num_output_shards=shards,
        )
        .map_shard(_sort_shard)
        .write_parquet(output_pattern)
    )

    ctx = ZephyrContext(name="datakit-normalize", max_workers=min(zephyr_max_workers, shards))
    output_files = list(ctx.execute(pipeline))
    logger.info("Wrote %d normalized Parquet partitions to %s", len(output_files), output_path)
    return PathsMetadata(parent_path=output_path, paths=output_files)


def normalize_step(
    name: str,
    *,
    input_path: str,
    text_field: str = DEFAULT_TEXT_FIELD,
    source_id_field: str | None = None,
    num_output_shards: int | None = None,
    zephyr_max_workers: int = 64,
    deps: list[StepSpec] | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec for the normalize stage.

    Args:
        name: Step name (e.g. "fineweb/normalize").
        input_path: Path to raw input files.
        text_field: Name of the field containing the primary text content.
        source_id_field: Name of an existing ID field to preserve as ``source_id``.
        num_output_shards: Number of output Parquet partitions.
        zephyr_max_workers: Maximum Zephyr worker parallelism.
        deps: Upstream dependencies (typically the download step).
        output_path_prefix: Override the default output path prefix.
        override_output_path: Override the computed output path entirely.

    Returns:
        A StepSpec whose output_path contains normalized Parquet files.
    """

    def _run(step_output_path: str) -> PathsMetadata:
        return normalize(
            input_path,
            step_output_path,
            text_field=text_field,
            source_id_field=source_id_field,
            num_output_shards=num_output_shards,
            zephyr_max_workers=zephyr_max_workers,
        )

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={
            "input_path": input_path,
            "text_field": text_field,
            "source_id_field": source_id_field,
        },
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
