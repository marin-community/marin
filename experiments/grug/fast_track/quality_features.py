# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare cached Harrier vectors for quality scoring."""

import hashlib
import math
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.writers import ensure_parent_dir

from experiments.datakit import hero_data
from experiments.datakit.embeddings.harrier.pipeline import (
    HARRIER_DIM,
    HARRIER_REPO,
    HARRIER_REVISION,
    QUANT_RANGE,
    QUANT_SCALE,
    dequantize_to_fp32,
)
from experiments.grug.fast_track.corpus_sample import CorpusSource, RawCorpusPool
from experiments.grug.fast_track.ranked_pool import RankedPool, take_token_prefix

FEATURE_BATCH_ROWS = 1024
FEATURE_JOIN_SHARDS = 512
HARRIER_NORM_FLOOR = 1e-6
FEATURE_FILE_PREFIX = "range-"
FEATURE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("raw_row_index", pa.int64()),
        pa.field("embedding", pa.list_(pa.int8(), HARRIER_DIM)),
    ]
)

HARRIER_FEATURE_IDENTITY: dict[str, str | int | float] = {
    "model": HARRIER_REPO,
    "revision": HARRIER_REVISION,
    "dimension": HARRIER_DIM,
    "quantization_range": QUANT_RANGE,
    "quantization_scale": QUANT_SCALE,
    "stored_dtype": "int8",
    "output_dtype": "float32",
    "normalization": "l2",
    "norm_floor": HARRIER_NORM_FLOOR,
}


@dataclass(frozen=True)
class QualityFeatureSource:
    """One normalized source pin and its matching Harrier output root."""

    source: str
    normalized_path: str
    harrier_path: str


@dataclass(frozen=True)
class QualityFeatureShard:
    """One feature shard in raw sample range order."""

    range_key: int
    path: str
    documents: int


class PreparedQualityPool(Artifact):
    """A fixed raw prefix and its aligned cached Harrier feature shards."""

    raw_manifest_path: str
    raw_seed: int
    tokenizer: str
    tokenizer_hash: str
    requested_tokens: int
    actual_tokens: int
    documents: int
    raw_prefix_shards: tuple[str, ...]
    feature_shards: tuple[QualityFeatureShard, ...]
    sources: tuple[QualityFeatureSource, ...]
    feature_identity: dict[str, str | int | float]


def normalize_harrier_embeddings(raw_embeddings: np.ndarray) -> np.ndarray:
    """Dequantize Harrier rows and return finite unit vectors in float32."""
    raw = np.asarray(raw_embeddings)
    if raw.ndim != 2 or raw.shape[1] != HARRIER_DIM or raw.dtype != np.int8:
        raise ValueError(f"Harrier features must be an int8 matrix with {HARRIER_DIM} columns")
    embeddings = dequantize_to_fp32(raw, scale=QUANT_SCALE)
    if not np.isfinite(embeddings).all():
        raise ValueError("Harrier features must be finite")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, np.maximum(norms, HARRIER_NORM_FLOOR), out=embeddings)
    return embeddings


def pinned_quality_feature_sources(sources: Sequence[CorpusSource]) -> tuple[QualityFeatureSource, ...]:
    """Resolve the checked-in normalized and Harrier paths for corpus sources."""
    pins = tuple(
        QualityFeatureSource(source.name, source.normalized_path, hero_data.harrier(source.name))
        for source in sorted(sources, key=lambda item: item.name)
    )
    if not pins or len({pin.source for pin in pins}) != len(pins):
        raise ValueError("quality feature sources must have distinct normalized source names")
    return pins


def _normalized_row_locators(raw_prefix_shards: tuple[str, ...]) -> Dataset[dict]:
    """Read only IDs and row offsets from the raw prefix."""
    inputs = [{"range_key": index, "path": path} for index, path in enumerate(raw_prefix_shards)]

    def read_shard(files: Iterator[dict], _: ShardInfo) -> Iterator[dict]:
        for item in files:
            row_index = 0
            with StoragePath(item["path"]).open("rb") as stream:
                parquet = pq.ParquetFile(stream)
                for batch in parquet.iter_batches(
                    batch_size=FEATURE_BATCH_ROWS,
                    columns=["source", "id", "normalized_shard", "normalized_row"],
                ):
                    for row in batch.to_pylist():
                        if not isinstance(row["id"], str) or not isinstance(row["source"], str):
                            raise ValueError("raw quality rows require string source and ID fields")
                        if not isinstance(row["normalized_shard"], str) or not isinstance(row["normalized_row"], int):
                            raise ValueError("raw quality rows require normalized shard and row locators")
                        yield {
                            "source": row["source"],
                            "normalized_shard": row["normalized_shard"],
                            "normalized_row": row["normalized_row"],
                            "raw_shard_index": item["range_key"],
                            "raw_row_index": row_index,
                            "id": row["id"],
                        }
                        row_index += 1

    return Dataset.from_list(inputs).reshard(len(inputs)).map_shard(read_shard)


def _embedding_path(source: QualityFeatureSource, normalized_shard: str) -> str:
    shard = StoragePath(normalized_shard)
    if not shard.relative_to(StoragePath(source.normalized_path)):
        raise ValueError(f"normalized shard {normalized_shard} is outside the pinned source {source.source}")
    return str(StoragePath(source.harrier_path) / shard.name)


def _join_source_shard(
    normalized_shard: str,
    locators: Iterator[dict],
    *,
    source_pins: dict[str, QualityFeatureSource],
) -> Iterator[dict]:
    """Read selected embeddings by row group and require each selected ID to match."""
    first = next(locators, None)
    if first is None:
        return
    source_name = first["source"]
    source = source_pins[source_name]
    embedding_path = _embedding_path(source, normalized_shard)

    # The locator list is bounded to one input shard. Keep only the current
    # Parquet row group's locators while the paired embedding rows are read.
    pending_locator = first
    previous_row = -1
    with StoragePath(embedding_path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        row_start = 0
        for row_group in range(parquet.metadata.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            row_end = row_start + row_count
            selected = []
            while pending_locator is not None and pending_locator["normalized_row"] < row_end:
                if pending_locator["source"] != source_name:
                    raise ValueError("one normalized shard has rows from multiple source pins")
                if pending_locator["normalized_row"] < row_start:
                    raise ValueError("normalized row locators are not unique and ordered")
                if pending_locator["normalized_row"] <= previous_row:
                    raise ValueError("normalized row locators are not unique and ordered")
                previous_row = pending_locator["normalized_row"]
                selected.append(pending_locator)
                pending_locator = next(locators, None)
            if selected:
                selected_index = 0
                current_row = row_start
                for batch in parquet.iter_batches(
                    batch_size=FEATURE_BATCH_ROWS,
                    row_groups=[row_group],
                    columns=["id", "embedding"],
                ):
                    for document_id, embedding in zip(
                        batch.column("id").to_pylist(), batch.column("embedding").to_pylist(), strict=True
                    ):
                        if selected_index < len(selected) and current_row == selected[selected_index]["normalized_row"]:
                            locator = selected[selected_index]
                            if document_id != locator["id"]:
                                raise ValueError(
                                    f"Harrier ID differs from normalized ID in {normalized_shard} at row {current_row}"
                                )
                            if len(embedding) != HARRIER_DIM:
                                raise ValueError(f"Harrier embedding has the wrong dimension in {embedding_path}")
                            yield {
                                "raw_shard_index": locator["raw_shard_index"],
                                "raw_row_index": locator["raw_row_index"],
                                "id": document_id,
                                "embedding": embedding,
                            }
                            selected_index += 1
                        current_row += 1
                if selected_index != len(selected):
                    raise ValueError(f"Harrier rows do not cover normalized locators in {normalized_shard}")
            row_start = row_end
        if pending_locator is not None:
            raise ValueError(
                f"Harrier shard {embedding_path} does not cover normalized row {pending_locator['normalized_row']}"
            )


def _write_feature_range(output_path: str):
    def write(range_key: int, rows: Iterator[dict]) -> QualityFeatureShard:
        path = prefix_join(output_path, f"{FEATURE_FILE_PREFIX}{range_key:05d}.parquet")
        ensure_parent_dir(path)
        documents = 0
        writer = None
        with ExitStack() as stack:
            target = stack.enter_context(StoragePath(path).open("wb"))
            while batch := list(islice(rows, FEATURE_BATCH_ROWS)):
                table = pa.Table.from_pylist(batch, schema=FEATURE_SCHEMA)
                if writer is None:
                    writer = stack.enter_context(pq.ParquetWriter(target, FEATURE_SCHEMA))
                writer.write_table(table)
                documents += len(batch)
        if documents == 0:
            raise ValueError("quality feature ranges must contain at least one document")
        return QualityFeatureShard(range_key, path, documents)

    return write


def prepare_quality_features(
    pool: RawCorpusPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    token_budget: int,
    sources: Sequence[QualityFeatureSource],
) -> PreparedQualityPool:
    """Prepare one bounded raw prefix of row-aligned cached Harrier features."""
    if token_budget <= 0 or token_budget > pool.requested_tokens:
        raise ValueError("quality feature token budget must fit the raw pool's requested capacity")
    source_pins = {source.source: source for source in sources}
    if not source_pins or len(source_pins) != len(sources):
        raise ValueError("quality feature source pins must be non-empty and unique")
    corpus_sources = {source.name for source in pool.sources}
    if source_pins.keys() != corpus_sources:
        raise ValueError("quality feature source pins must match the raw pool's normalized sources")
    normalized_paths = {source.name: source.normalized_path for source in pool.sources}
    if any(source.normalized_path != normalized_paths[source.source] for source in sources):
        raise ValueError("quality feature normalized paths must match the raw pool source pins")
    raw_prefix = take_token_prefix(
        RankedPool(pool.shards, pool.range_totals, pool.documents, pool.actual_tokens),
        ctx=ctx,
        output_path=prefix_join(output_path, "raw_prefix"),
        token_budget=token_budget,
    )
    locators = _normalized_row_locators(raw_prefix.shards)
    matched_features = locators.group_by(
        key=lambda row: row["normalized_shard"],
        reducer=lambda path, rows: _join_source_shard(path, rows, source_pins=source_pins),
        sort_by=lambda row: row["normalized_row"],
        num_output_shards=min(FEATURE_JOIN_SHARDS, math.ceil(raw_prefix.total_documents / FEATURE_BATCH_ROWS)),
    )
    feature_rows = matched_features.group_by(
        key=lambda row: row["raw_shard_index"],
        reducer=_write_feature_range(output_path),
        sort_by=lambda row: row["raw_row_index"],
        num_output_shards=len(raw_prefix.shards),
    )
    feature_shards = tuple(sorted(ctx.execute(feature_rows).results, key=lambda item: item.range_key))
    if tuple(item.range_key for item in feature_shards) != tuple(range(len(raw_prefix.shards))):
        raise ValueError("Harrier feature joins did not produce every raw prefix shard")
    if sum(item.documents for item in feature_shards) != raw_prefix.total_documents:
        raise ValueError("Harrier feature joins did not preserve the raw prefix document count")
    source_pins_tuple = tuple(sorted(sources, key=lambda item: item.source))
    source_identity = [
        {"source": item.source, "normalized_path": item.normalized_path, "harrier_path": item.harrier_path}
        for item in source_pins_tuple
    ]
    identity = {
        **HARRIER_FEATURE_IDENTITY,
        "sources_sha256": hashlib.sha256(canonical_json(source_identity).encode()).hexdigest(),
    }
    return PreparedQualityPool(
        raw_manifest_path=pool.manifest_path,
        raw_seed=pool.seed,
        tokenizer=pool.tokenizer,
        tokenizer_hash=pool.tokenizer_hash,
        requested_tokens=token_budget,
        actual_tokens=raw_prefix.total_tokens,
        documents=raw_prefix.total_documents,
        raw_prefix_shards=raw_prefix.shards,
        feature_shards=feature_shards,
        sources=source_pins_tuple,
        feature_identity=identity,
    )
