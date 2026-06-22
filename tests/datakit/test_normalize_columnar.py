# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the columnar two-stage normalize.

Runs both the dict pipeline (``columnar=False``) and the columnar pipeline
(``columnar=True``) over the same synthetic Parquet input and asserts the main
output rows are equal as sets and the duplicate / filter counts match. This is
the whole point of the columnar variant: it must be a byte-equivalent drop-in.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray import LocalClient, set_current_client
from marin.datakit.normalize import DedupMode, normalize_to_parquet


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _make_input(input_dir: Path, num_files: int = 3) -> None:
    """Write synthetic Parquet input across *num_files* files.

    Includes: plain rows, exact-duplicate rows (identical in every column so
    dedup's representative choice is unambiguous), rows with pathological
    whitespace runs (>128 chars, spanning ``\\t`` / ``\\n`` / ``\\xa0`` and
    mixed), and blank/empty/None text rows that must be filtered out.
    """
    rows: list[dict] = []

    # Plain unique rows.
    for i in range(2000):
        rows.append({"id": f"src-{i}", "text": f"document number {i} body", "lang": "en", "score": float(i % 7)})

    # Exact duplicates: identical rows (same source id + cols) so "keep first"
    # picks the same representative regardless of pipeline / shard order.
    for i in range(0, 300):
        rows.append({"id": f"src-{i}", "text": f"document number {i} body", "lang": "en", "score": float(i % 7)})

    # Pathological whitespace runs (>128). Recompaction + id recompute must match.
    rows.append({"id": "ws-space", "text": "alpha" + " " * 400 + "beta", "lang": "en", "score": 1.0})
    rows.append({"id": "ws-tab", "text": "a" + "\t" * 150 + "b", "lang": "en", "score": 1.0})
    rows.append({"id": "ws-nl", "text": "c" + "\n" * 200 + "d", "lang": "en", "score": 1.0})
    rows.append({"id": "ws-nbsp", "text": "e" + "\xa0" * 180 + "f", "lang": "en", "score": 1.0})
    rows.append({"id": "ws-mixed", "text": "g" + (" \t\n\xa0" * 50) + "h", "lang": "en", "score": 1.0})
    # A whitespace run exactly at the limit (128) — must be left untouched.
    rows.append({"id": "ws-edge", "text": "i" + " " * 128 + "j", "lang": "en", "score": 1.0})

    # Blank / empty / None text — must be filtered out by both pipelines.
    rows.append({"id": "blank-spaces", "text": "   ", "lang": "en", "score": 0.0})
    rows.append({"id": "blank-empty", "text": "", "lang": "en", "score": 0.0})
    rows.append({"id": "blank-nbsp", "text": "\xa0\xa0\n\n\xa0", "lang": "en", "score": 0.0})
    rows.append({"id": "blank-none", "text": None, "lang": "en", "score": 0.0})

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("lang", pa.string()),
            ("score", pa.float64()),
        ]
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    for f in range(num_files):
        shard_rows = rows[f::num_files]
        table = pa.Table.from_pylist(shard_rows, schema=schema)
        pq.write_table(table, str(input_dir / f"part-{f}.parquet"))


def _read_main(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for pf in sorted((output_dir / "outputs" / "main").glob("*.parquet")):
        records.extend(pq.read_table(str(pf)).to_pylist())
    return records


def _read_dups(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for pf in sorted((output_dir / "outputs" / "dups").glob("*.parquet")):
        records.extend(pq.read_table(str(pf)).to_pylist())
    return records


def _row_key(row: dict) -> tuple:
    return tuple(sorted(row.items()))


def test_columnar_matches_dict_pipeline(tmp_path: Path):
    """Main output set + dup/filter counts are identical between both pipelines."""
    input_dir = tmp_path / "input"
    _make_input(input_dir)

    # Small partition target so num_shards > 1 (exercises routing + the reduce
    # glob across mappers); 3 input files exercise multiple scatter mappers.
    common = dict(input_path=str(input_dir), target_partition_bytes=4096, max_whitespace_run_chars=128)

    dict_out = tmp_path / "out_dict"
    col_out = tmp_path / "out_columnar"
    dict_result = normalize_to_parquet(output_path=str(dict_out), columnar=False, **common)
    col_result = normalize_to_parquet(output_path=str(col_out), columnar=True, **common)

    dict_main = _read_main(dict_out)
    col_main = _read_main(col_out)

    # Main rows equal as sets (id + text + source_id + every other column).
    dict_keys = {_row_key(r) for r in dict_main}
    col_keys = {_row_key(r) for r in col_main}
    assert (
        col_keys == dict_keys
    ), f"main mismatch: only-dict={len(dict_keys - col_keys)} only-col={len(col_keys - dict_keys)}"
    # No id collisions: set size == row count (each pipeline emits unique ids).
    assert len(col_main) == len(col_keys)
    assert len(dict_main) == len(col_main)

    # 2310 raw - 4 blank filtered = 2306 surviving; 300 exact dups → 2006 unique.
    assert len(col_main) == 2006

    # Duplicate counts match (and equal the 300 injected exact duplicates).
    assert len(_read_dups(col_out)) == len(_read_dups(dict_out)) == 300

    # Counter parity on the meaningful normalize counters.
    for key in ("normalize/empty_text_filtered", "normalize/unique_records_out", "normalize/duplicate_records_out"):
        assert col_result.counters.get(key, 0) == dict_result.counters.get(key, 0), key
    assert col_result.counters["normalize/empty_text_filtered"] == 4
    assert col_result.counters["normalize/unique_records_out"] == 2006
    assert col_result.counters["normalize/duplicate_records_out"] == 300


def test_columnar_dedup_none_keeps_all(tmp_path: Path):
    """With dedup disabled, columnar main keeps every surviving row, no dups."""
    input_dir = tmp_path / "input"
    _make_input(input_dir, num_files=2)

    out = tmp_path / "out"
    result = normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(out),
        target_partition_bytes=4096,
        dedup_mode=DedupMode.NONE,
        columnar=True,
    )

    # 2310 raw - 4 blank = 2306 surviving, all kept, none routed to dups.
    assert len(_read_main(out)) == 2306
    assert len(_read_dups(out)) == 0
    assert result.counters["normalize/duplicate_records_out"] == 0
    assert result.counters["normalize/unique_records_out"] == 2306
