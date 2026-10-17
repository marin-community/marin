# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.rl.pass_rates import PASS_RATES_FILENAME, pool_pass_rates


def _run(tmp_path, name: str, passed: list[int], exclusion: list[str | None], rows: list[str]) -> str:
    total = [0 if reason else 8 for reason in exclusion]
    table = pa.table(
        {
            "row_id": rows,
            "passed": pa.array(passed, pa.int32()),
            "total": pa.array(total, pa.int32()),
            "pass_rate": pa.array([p / t if t else None for p, t in zip(passed, total, strict=True)], pa.float64()),
            "exclusion": pa.array(exclusion, pa.string()),
            "passed_exact": pa.array([p // 2 for p in passed], pa.int32()),
        }
    )
    path = tmp_path / name
    path.mkdir()
    pq.write_table(table, path / PASS_RATES_FILENAME)
    return str(path)


def test_pooled_counts_sum_seeds_and_drop_rows_any_seed_excluded(tmp_path):
    rows = ["a", "b", "c"]
    first = _run(tmp_path, "seed-none", [3, 8, 0], [None, None, None], rows)
    second = _run(tmp_path, "seed-1", [5, 6, 0], [None, None, "context_window"], rows)

    pool_pass_rates(pass_rates_paths=[first, second], output_path=str(tmp_path))

    pooled = pq.read_table(tmp_path / PASS_RATES_FILENAME).to_pydict()
    assert pooled["row_id"] == rows
    assert pooled["passed"] == [8, 14, 0]
    assert pooled["total"] == [16, 16, 0]
    assert pooled["pass_rate"] == [0.5, 14 / 16, None]
    assert pooled["exclusion"] == [None, None, "context_window"]
    assert pooled["passed_exact"] == [3, 7, 0]


def test_pooling_runs_over_different_rows_fails(tmp_path):
    first = _run(tmp_path, "seed-none", [3], [None], ["a"])
    second = _run(tmp_path, "seed-1", [3], [None], ["b"])
    with pytest.raises(ValueError, match="same rows"):
        pool_pass_rates(pass_rates_paths=[first, second], output_path=str(tmp_path))
