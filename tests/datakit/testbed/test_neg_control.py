# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""First-K replay recipe for the Datakit Testbed duplication arm (#5310)."""

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData

from experiments.datakit.testbed.neg_control import (
    duplicate_normalized_shards,
    first_k_replay,
    unique_row_count,
)


def _ids(n: int) -> list[str]:
    return [f"r{i}" for i in range(n)]


def _write_shard(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": ids, "text": [f"t{i}" for i in ids]}), path)


def _normalized(tmp_path: Path, shards: dict[str, list[str]]) -> NormalizedData:
    main_dir = tmp_path / "norm" / "outputs" / "main"
    dup_dir = tmp_path / "norm" / "outputs" / "dups"
    dup_dir.mkdir(parents=True, exist_ok=True)
    for rel, ids in shards.items():
        _write_shard(main_dir / rel, ids)
    return NormalizedData(main_output_dir=str(main_dir), dup_output_dir=str(dup_dir), counters={})


def _read_ids(path: Path) -> list[str]:
    return pq.read_table(path).column("id").to_pylist()


@pytest.mark.parametrize(
    ("n", "unique_fraction"),
    [
        (10, 0.50),
        (7, 0.50),
        (8, 0.25),
        (10, 0.10),
        (20, 0.05),
        (1, 0.05),
        (2, 0.05),
        (5, 1.0),
    ],
)
def test_unique_count_is_ceil_fraction_of_n(n: int, unique_fraction: float):
    assert unique_row_count(n, unique_fraction) == math.ceil(unique_fraction * n)


def test_replay_preserves_length_and_first_k_pool():
    rows = _ids(10)
    out = first_k_replay(rows, 0.25)
    unique_n = math.ceil(0.25 * 10)
    assert len(out) == 10
    assert unique_n == 3
    assert out[:unique_n] == rows[:unique_n]
    assert set(out) <= set(rows[:unique_n])


def test_unique_fraction_0_5_is_5310_even_and_odd_shards():
    """#5310: unique_n = ceil(0.5 * N); replay the prefix until length N."""
    even = _ids(10)
    assert first_k_replay(even, 0.50) == even[:5] + even[:5]

    odd = _ids(7)
    # ceil(0.5 * 7) = 4 → prefix of 4, then 3 more from the start.
    assert first_k_replay(odd, 0.50) == odd[:4] + odd[:3]


def test_replay_is_deterministic():
    rows = _ids(16)
    assert first_k_replay(rows, 0.10) == first_k_replay(rows, 0.10)


def test_unique_fraction_1_is_identity():
    rows = _ids(6)
    assert first_k_replay(rows, 1.0) == rows


def test_small_shard_unique_fraction_rounds_up_to_one_row():
    rows = _ids(1)
    assert first_k_replay(rows, 0.05) == rows
    two = _ids(2)
    assert unique_row_count(2, 0.05) == 1
    assert first_k_replay(two, 0.05) == [two[0], two[0]]


def test_empty_sequence_stays_empty():
    assert first_k_replay([], 0.50) == []
    assert unique_row_count(0, 0.50) == 0


def test_invalid_unique_fraction_rejected():
    rows = _ids(4)
    for bad in (0.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="unique_fraction"):
            first_k_replay(rows, bad)
        with pytest.raises(ValueError, match="unique_fraction"):
            unique_row_count(4, bad)


def test_duplicate_normalized_shards_matches_first_k_replay(tmp_path: Path):
    source = _normalized(
        tmp_path,
        {
            "part-00000-of-00002.parquet": _ids(10),
            "part-00001-of-00002.parquet": _ids(7),
        },
    )
    unique_fraction = 0.50
    out = duplicate_normalized_shards(
        source=source,
        output_path=str(tmp_path / "duped"),
        unique_fraction=unique_fraction,
    )

    src_shards = sorted(Path(source.main_output_dir).glob("*.parquet"))
    dst_shards = sorted(Path(out.main_output_dir).glob("*.parquet"))
    assert [p.name for p in dst_shards] == [p.name for p in src_shards]
    for src, dst in zip(src_shards, dst_shards, strict=True):
        src_ids = _read_ids(src)
        dst_ids = _read_ids(dst)
        assert len(dst_ids) == len(src_ids)
        assert dst_ids == first_k_replay(src_ids, unique_fraction)
    assert out.counters["neg_control/rows_out"] == out.counters["neg_control/rows_in"] == 17
    assert out.counters["neg_control/unique_rows"] == math.ceil(0.5 * 10) + math.ceil(0.5 * 7)


def test_duplicate_identity_and_determinism_on_parquet(tmp_path: Path):
    source = _normalized(tmp_path, {"part-00000-of-00001.parquet": _ids(8)})
    identity = duplicate_normalized_shards(source=source, output_path=str(tmp_path / "id"), unique_fraction=1.0)
    assert _read_ids(next(Path(identity.main_output_dir).glob("*.parquet"))) == _ids(8)

    a = duplicate_normalized_shards(source=source, output_path=str(tmp_path / "a"), unique_fraction=0.25)
    b = duplicate_normalized_shards(source=source, output_path=str(tmp_path / "b"), unique_fraction=0.25)
    assert _read_ids(next(Path(a.main_output_dir).glob("*.parquet"))) == _read_ids(
        next(Path(b.main_output_dir).glob("*.parquet"))
    )


def test_duplicate_rejects_invalid_fraction_and_empty_dir(tmp_path: Path):
    source = _normalized(tmp_path, {"part-00000-of-00001.parquet": _ids(3)})
    with pytest.raises(ValueError, match="unique_fraction"):
        duplicate_normalized_shards(source=source, output_path=str(tmp_path / "bad"), unique_fraction=0.0)

    empty_main = tmp_path / "empty" / "outputs" / "main"
    empty_main.mkdir(parents=True)
    empty = NormalizedData(main_output_dir=str(empty_main), dup_output_dir=str(tmp_path / "d"), counters={})
    with pytest.raises(ValueError, match="No parquet shards"):
        duplicate_normalized_shards(source=empty, output_path=str(tmp_path / "out"), unique_fraction=0.50)
