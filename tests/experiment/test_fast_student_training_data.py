# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT = Path(__file__).parents[2] / ".agents/projects/luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from fast_student_training_data import (  # noqa: E402
    MaterializedTrainingRows,
    SourceTrainingRows,
    interleaved_chunk_placements,
    source_arrays,
    stage_training_rows,
)

ID_WIDTH = 4
TEACHER_DIMENSION = 3


def write_rows(path: Path, source_id: int, rows: int, hashes: list[str] | None = None) -> None:
    if hashes is None:
        hashes = [f"hash-{source_id}-{rank}" for rank in range(rows)]
    ids = np.asarray(
        [[source_id * 100 + rank, rank + 1, rank + 2, rank + 3] for rank in range(rows)],
        dtype=np.int32,
    )
    embedding = np.asarray(
        [[100 + source_id, 110 + rank, 120 + source_id + rank] for rank in range(rows)],
        dtype=np.uint8,
    )
    pq.write_table(
        pa.table(
            {
                "raw_sha256": hashes,
                "train_rank": np.arange(rows, dtype=np.int64),
                "ids": pa.FixedSizeListArray.from_arrays(pa.array(ids.reshape(-1)), ID_WIDTH),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(embedding.reshape(-1)),
                    TEACHER_DIMENSION,
                ),
            }
        ),
        path,
        row_group_size=2,
    )


def test_interleaved_chunks_cover_each_source_in_order() -> None:
    placements = interleaved_chunk_placements([5, 4, 2], chunk_rows=2, seed=17)

    assert sum(placement.row_count for placement in placements) == 11
    assert [placement.staged_start for placement in placements] == [0, 2, 4, 6, 8, 10]
    for source_id, rows in enumerate((5, 4, 2)):
        source_chunks = [placement for placement in placements if placement.source_id == source_id]
        assert [placement.source_start for placement in source_chunks] == list(range(0, rows, 2))
        assert sum(placement.row_count for placement in source_chunks) == rows


def test_materialized_batches_keep_the_original_global_shuffle() -> None:
    ids = np.arange(40, dtype=np.int32).reshape(10, ID_WIDTH)
    teacher = np.arange(30, dtype=np.float32).reshape(10, TEACHER_DIMENSION)
    source_ids = np.arange(10, dtype=np.int32)
    rows = MaterializedTrainingRows(ids, teacher, source_ids)

    first_epoch = list(rows.epoch_batches(epoch=0, batch_size=4, block_rows=8, seed=42))
    second_epoch = list(rows.epoch_batches(epoch=1, batch_size=4, block_rows=8, seed=42))
    rng = np.random.default_rng(42)
    expected_first = rng.permutation(10)
    expected_second = rng.permutation(10)
    expected_first = np.concatenate((expected_first, expected_first[:2]))
    expected_second = np.concatenate((expected_second, expected_second[:2]))

    assert np.array_equal(np.concatenate([batch.source_ids for batch in first_epoch]), expected_first)
    assert np.array_equal(np.concatenate([batch.source_ids for batch in second_epoch]), expected_second)


def test_staged_rows_preserve_all_rows_and_return_fixed_batches(tmp_path: Path) -> None:
    row_counts = [5, 7, 5]
    sources = []
    for source_id, rows in enumerate(row_counts):
        path = tmp_path / f"source-{source_id}.parquet"
        write_rows(path, source_id, rows)
        sources.append(SourceTrainingRows(f"source-{source_id}", source_id, str(path), None, rows))

    staged = stage_training_rows(
        sources,
        tmp_path / "staged",
        id_width=ID_WIDTH,
        teacher_dimension=TEACHER_DIMENSION,
        teacher_quantization_limit=0.3,
        chunk_rows=2,
        maximum_source_rows=8,
        seed=42,
    )

    ids = np.load(staged.ids_path)
    source_ids = np.load(staged.source_ids_path)
    observed = {(int(source_id), int(row[0])) for source_id, row in zip(source_ids, ids, strict=True)}
    expected = {(source_id, source_id * 100 + rank) for source_id, rows in enumerate(row_counts) for rank in range(rows)}
    assert observed == expected
    batches = list(staged.epoch_batches(epoch=0, batch_size=4, block_rows=8, seed=42))
    assert len(batches) == 5
    assert all(batch.ids.shape == (4, ID_WIDTH) for batch in batches)
    assert all(batch.teacher.shape == (4, TEACHER_DIMENSION) for batch in batches)
    assert all(np.isfinite(batch.teacher).all() for batch in batches)
    assert all(np.linalg.norm(batch.teacher, axis=1) == pytest.approx(1.0) for batch in batches)
    first_order = np.concatenate([batch.ids[:, 0] for batch in batches])
    repeated_order = np.concatenate(
        [batch.ids[:, 0] for batch in staged.epoch_batches(epoch=0, batch_size=4, block_rows=8, seed=42)]
    )
    assert np.array_equal(first_order, repeated_order)
    assert expected <= {
        (int(source_id), int(row[0]))
        for batch in batches
        for source_id, row in zip(batch.source_ids, batch.ids, strict=True)
    }
    report = staged.memory_report(block_rows=8)
    assert report["maximum_source_rows"] == 7
    assert report["epoch_block_rows"] == 8
    assert report["full_array_bytes_on_disk"] > report["estimated_epoch_block_array_bytes"]


def test_staging_rejects_a_source_above_the_memory_limit(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    write_rows(path, source_id=0, rows=3)

    with pytest.raises(ValueError, match="largest source quota is 3"):
        stage_training_rows(
            [SourceTrainingRows("source", 0, str(path), None, 3)],
            tmp_path / "staged",
            id_width=ID_WIDTH,
            teacher_dimension=TEACHER_DIMENSION,
            teacher_quantization_limit=0.3,
            chunk_rows=2,
            maximum_source_rows=2,
            seed=42,
        )


def test_separate_teacher_must_align_with_prepared_rows(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.parquet"
    teacher_path = tmp_path / "teacher.parquet"
    write_rows(prepared_path, source_id=0, rows=2)
    write_rows(teacher_path, source_id=0, rows=2, hashes=["different-0", "different-1"])
    source = SourceTrainingRows("source", 0, str(prepared_path), str(teacher_path), 2)

    with pytest.raises(ValueError, match="Teacher hashes are not aligned"):
        source_arrays(source, ID_WIDTH, TEACHER_DIMENSION, teacher_quantization_limit=0.3)
