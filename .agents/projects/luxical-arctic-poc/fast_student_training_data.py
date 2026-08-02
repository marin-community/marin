# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provide bounded training batches for the fast embedding student."""

import logging
import mmap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from luxical.training import dequantize_8bit_uniform_scalar_quantized

logger = logging.getLogger(__name__)


class StagedMemoryReport(TypedDict):
    """Describe the disk-backed loader memory bounds."""

    layout: str
    rows: int
    staging_chunk_rows: int
    maximum_source_rows: int
    source_row_limit: int
    epoch_block_rows: int
    estimated_maximum_source_array_bytes: int
    estimated_epoch_block_array_bytes: int
    full_array_bytes_on_disk: int
    staged_file_bytes: int


class MaterializedMemoryReport(TypedDict):
    """Describe the original materialized loader allocation."""

    layout: str
    rows: int
    array_bytes: int


@dataclass(frozen=True)
class SourceTrainingRows:
    """Identify one aligned source slice."""

    source: str
    source_id: int
    prepared_url: str
    teacher_url: str | None
    rows: int


@dataclass(frozen=True)
class ChunkPlacement:
    """Place one contiguous source chunk in the staged row order."""

    source_id: int
    source_start: int
    row_count: int
    staged_start: int


@dataclass(frozen=True)
class TrainingBatch:
    """Contain one fixed-size student training batch."""

    ids: np.ndarray
    teacher: np.ndarray
    source_ids: np.ndarray


@dataclass(frozen=True)
class StagedTrainingRows:
    """Describe disk-backed training arrays and their memory limits."""

    ids_path: Path
    teacher_path: Path
    source_ids_path: Path
    rows: int
    id_width: int
    teacher_dimension: int
    staging_chunk_rows: int
    maximum_source_rows: int
    source_row_limit: int
    disk_bytes: int

    def audit_ids(self, rows: int) -> np.ndarray:
        """Return a fixed staged prefix for the embedding audit."""
        ids = np.load(self.ids_path, mmap_mode="r")
        if not isinstance(ids, np.memmap):
            raise TypeError(f"The staged ID array has type {type(ids).__name__}")
        audit = np.asarray(ids[: min(rows, self.rows)]).copy()
        release_memmap_pages(ids)
        return audit

    def epoch_batches(
        self,
        epoch: int,
        batch_size: int,
        block_rows: int,
        seed: int,
    ) -> Iterator[TrainingBatch]:
        """Yield block-shuffled batches without loading the full dataset."""
        if batch_size < 1:
            raise ValueError("The batch size must be positive")
        if block_rows < batch_size or block_rows % batch_size:
            raise ValueError("The block row count must be a positive multiple of the batch size")
        ids = np.load(self.ids_path, mmap_mode="r")
        teacher = np.load(self.teacher_path, mmap_mode="r")
        source_ids = np.load(self.source_ids_path, mmap_mode="r")
        if not all(isinstance(array, np.memmap) for array in (ids, teacher, source_ids)):
            raise TypeError("The staged training arrays are not memory maps")
        if ids.shape != (self.rows, self.id_width):
            raise ValueError(f"The staged ID array has shape {ids.shape}")
        if teacher.shape != (self.rows, self.teacher_dimension):
            raise ValueError(f"The staged teacher array has shape {teacher.shape}")
        if source_ids.shape != (self.rows,):
            raise ValueError(f"The staged source array has shape {source_ids.shape}")

        rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
        block_starts = np.arange(0, self.rows, block_rows, dtype=np.int64)
        rng.shuffle(block_starts)
        for block_start_value in block_starts:
            block_start = int(block_start_value)
            block_end = min(self.rows, block_start + block_rows)
            block_ids = np.asarray(ids[block_start:block_end]).copy()
            block_teacher = np.asarray(teacher[block_start:block_end]).copy()
            block_source_ids = np.asarray(source_ids[block_start:block_end]).copy()
            order = rng.permutation(block_end - block_start)
            padded_rows = (-len(order)) % batch_size
            if padded_rows:
                order = np.concatenate((order, np.resize(order, len(order) + padded_rows)[len(order) :]))
            for batch_start in range(0, len(order), batch_size):
                selected = order[batch_start : batch_start + batch_size]
                yield TrainingBatch(
                    ids=block_ids[selected],
                    teacher=block_teacher[selected],
                    source_ids=block_source_ids[selected],
                )
            release_memmap_pages(ids, teacher, source_ids)

    def memory_report(self, block_rows: int) -> StagedMemoryReport:
        """Return conservative array-memory limits for one staged run."""
        bytes_per_row = np.dtype(np.int32).itemsize * (self.id_width + self.teacher_dimension + 1)
        return {
            "layout": "local-numpy-memmap",
            "rows": self.rows,
            "staging_chunk_rows": self.staging_chunk_rows,
            "maximum_source_rows": self.maximum_source_rows,
            "source_row_limit": self.source_row_limit,
            "epoch_block_rows": block_rows,
            "estimated_maximum_source_array_bytes": self.maximum_source_rows * bytes_per_row,
            "estimated_epoch_block_array_bytes": min(self.rows, block_rows) * bytes_per_row,
            "full_array_bytes_on_disk": self.rows * bytes_per_row,
            "staged_file_bytes": self.disk_bytes,
        }


@dataclass(frozen=True)
class MaterializedTrainingRows:
    """Adapt the original in-memory arrays to the batch interface."""

    ids: np.ndarray
    teacher: np.ndarray
    source_ids: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.ids)

    def audit_ids(self, rows: int) -> np.ndarray:
        return self.ids[: min(rows, self.rows)]

    def epoch_batches(
        self,
        epoch: int,
        batch_size: int,
        block_rows: int,
        seed: int,
    ) -> Iterator[TrainingBatch]:
        """Yield the original globally shuffled in-memory batches."""
        del block_rows
        if len(self.teacher) != self.rows or len(self.source_ids) != self.rows:
            raise ValueError("The materialized training arrays have different row counts")
        rng = np.random.default_rng(seed)
        permutation = None
        for _ in range(epoch + 1):
            permutation = rng.permutation(self.rows)
        assert permutation is not None
        padded_rows = (-len(permutation)) % batch_size
        if padded_rows:
            permutation = np.concatenate((permutation, permutation[:padded_rows]))
        for batch_start in range(0, len(permutation), batch_size):
            selected = permutation[batch_start : batch_start + batch_size]
            yield TrainingBatch(
                ids=self.ids[selected],
                teacher=self.teacher[selected],
                source_ids=self.source_ids[selected],
            )

    def memory_report(self, block_rows: int) -> MaterializedMemoryReport:
        del block_rows
        return {
            "layout": "materialized",
            "rows": self.rows,
            "array_bytes": self.ids.nbytes + self.teacher.nbytes + self.source_ids.nbytes,
        }


def interleaved_chunk_placements(
    source_rows: list[int],
    chunk_rows: int,
    seed: int,
) -> list[ChunkPlacement]:
    """Return a deterministic source-interleaved staging order."""
    if chunk_rows < 1:
        raise ValueError("The staging chunk row count must be positive")
    pending = {
        source_id: [(start, min(chunk_rows, rows - start)) for start in range(0, rows, chunk_rows)]
        for source_id, rows in enumerate(source_rows)
    }
    next_chunk = [0] * len(source_rows)
    placements = []
    staged_start = 0
    rng = np.random.default_rng(seed)
    while True:
        active = [source_id for source_id, chunks in pending.items() if next_chunk[source_id] < len(chunks)]
        if not active:
            break
        rng.shuffle(active)
        for source_id in active:
            source_start, row_count = pending[source_id][next_chunk[source_id]]
            next_chunk[source_id] += 1
            placements.append(
                ChunkPlacement(
                    source_id=source_id,
                    source_start=source_start,
                    row_count=row_count,
                    staged_start=staged_start,
                )
            )
            staged_start += row_count
    if staged_start != sum(source_rows):
        raise ValueError(f"Chunk placements cover {staged_start} rows; expected {sum(source_rows)}")
    return placements


def release_memmap_pages(*arrays: np.memmap) -> None:
    """Flush mapped files and release their resident pages."""
    for array in arrays:
        array.flush()
        mapping = array.base
        if not isinstance(mapping, mmap.mmap):
            raise TypeError(f"The memory map has base type {type(mapping).__name__}")
        mapping.madvise(mmap.MADV_DONTNEED)


def _read_table(url: str, columns: list[str], rows: int) -> pa.Table:
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(
        path,
        filesystem=filesystem,
        columns=columns,
        filters=[("train_rank", "<", rows)],
    ).sort_by("train_rank")
    if len(table) != rows:
        raise ValueError(f"Training file {url} returned {len(table)} rows; expected {rows}")
    return table


def _fixed_list_values(column: pa.ChunkedArray, rows: int, width: int, name: str) -> np.ndarray:
    combined = column.combine_chunks()
    if not pa.types.is_fixed_size_list(combined.type) or combined.type.list_size != width:
        raise ValueError(f"The {name} column has type {combined.type}; expected fixed list width {width}")
    return combined.values.to_numpy(zero_copy_only=False).reshape(rows, width)


def source_arrays(
    source: SourceTrainingRows,
    id_width: int,
    teacher_dimension: int,
    teacher_quantization_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read, align, and normalize one bounded source slice."""
    prepared = _read_table(
        source.prepared_url,
        ["raw_sha256", "train_rank", "ids", "embedding"],
        source.rows,
    )
    if source.teacher_url is None:
        teacher = prepared
    else:
        teacher = _read_table(
            source.teacher_url,
            ["raw_sha256", "train_rank", "embedding"],
            source.rows,
        )
        if not prepared["raw_sha256"].equals(teacher["raw_sha256"]):
            raise ValueError(f"Teacher hashes are not aligned for {source.source}")
        if not prepared["train_rank"].equals(teacher["train_rank"]):
            raise ValueError(f"Teacher ranks are not aligned for {source.source}")
    ids = _fixed_list_values(prepared["ids"], source.rows, id_width, "ids").astype(np.int32, copy=False)
    quantized = _fixed_list_values(
        teacher["embedding"],
        source.rows,
        teacher_dimension,
        "embedding",
    )
    vectors = dequantize_8bit_uniform_scalar_quantized(quantized, teacher_quantization_limit).astype(
        np.float32,
        copy=False,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    if not np.isfinite(vectors).all():
        raise ValueError(f"Teacher arrays contain non-finite values for {source.source}")
    return ids, vectors


def stage_training_rows(
    sources: list[SourceTrainingRows],
    output_directory: Path,
    id_width: int,
    teacher_dimension: int,
    teacher_quantization_limit: float,
    chunk_rows: int,
    maximum_source_rows: int,
    seed: int,
) -> StagedTrainingRows:
    """Write source-interleaved disk arrays with bounded source memory."""
    if not sources:
        raise ValueError("At least one training source is required")
    if any(source.source_id != index for index, source in enumerate(sources)):
        raise ValueError("Training source IDs must be contiguous and ordered")
    largest_source = max(source.rows for source in sources)
    if largest_source > maximum_source_rows:
        raise ValueError(f"The largest source quota is {largest_source}; the memory limit is {maximum_source_rows} rows")
    total_rows = sum(source.rows for source in sources)
    placements = interleaved_chunk_placements([source.rows for source in sources], chunk_rows, seed)
    placements_by_source = {source.source_id: [] for source in sources}
    for placement in placements:
        placements_by_source[placement.source_id].append(placement)

    output_directory.mkdir(parents=True, exist_ok=True)
    ids_path = output_directory / "ids.npy"
    teacher_path = output_directory / "teacher.npy"
    source_ids_path = output_directory / "source-ids.npy"
    staged_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int32, shape=(total_rows, id_width))
    staged_teacher = np.lib.format.open_memmap(
        teacher_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, teacher_dimension),
    )
    staged_source_ids = np.lib.format.open_memmap(
        source_ids_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_rows,),
    )
    for source in sources:
        logger.info("Staging source %d/%d: %s (%d rows)", source.source_id + 1, len(sources), source.source, source.rows)
        ids, teacher = source_arrays(
            source,
            id_width,
            teacher_dimension,
            teacher_quantization_limit,
        )
        for placement in placements_by_source[source.source_id]:
            source_slice = slice(placement.source_start, placement.source_start + placement.row_count)
            staged_slice = slice(placement.staged_start, placement.staged_start + placement.row_count)
            staged_ids[staged_slice] = ids[source_slice]
            staged_teacher[staged_slice] = teacher[source_slice]
            staged_source_ids[staged_slice] = source.source_id
        release_memmap_pages(staged_ids, staged_teacher, staged_source_ids)
    release_memmap_pages(staged_ids, staged_teacher, staged_source_ids)
    del staged_ids, staged_teacher, staged_source_ids
    disk_bytes = sum(path.stat().st_size for path in (ids_path, teacher_path, source_ids_path))
    return StagedTrainingRows(
        ids_path=ids_path,
        teacher_path=teacher_path,
        source_ids_path=source_ids_path,
        rows=total_rows,
        id_width=id_width,
        teacher_dimension=teacher_dimension,
        staging_chunk_rows=chunk_rows,
        maximum_source_rows=largest_source,
        source_row_limit=maximum_source_rows,
        disk_bytes=disk_bytes,
    )
