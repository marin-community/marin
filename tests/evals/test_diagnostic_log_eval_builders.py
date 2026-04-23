# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

from experiments.evals.diagnostic_log_eval_builders import (
    diagnostic_log_eval_output_path,
    materialize_ghalogs_eval_sample,
    materialize_loghub_eval_sample,
)
from experiments.evals.long_tail_ppl import LongTailPplFamily, long_tail_raw_validation_sets


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _read_jsonl_gz(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_materialize_ghalogs_eval_sample_converts_pre_staged_logs(tmp_path):
    source_root = tmp_path / "ghalogs_source"
    output_path = tmp_path / "raw" / "diagnostic_logs" / "ghalogs" / "runs.jsonl.gz"

    _write_lines(
        source_root / "a.jsonl",
        [
            json.dumps({"message": "Run started"}),
            json.dumps({"log": "Step failed: exit status 1"}),
            "stderr: traceback (most recent call last)",
        ],
    )
    _write_lines(source_root / "b.log", ["plain line from second file"])

    stats = materialize_ghalogs_eval_sample(
        source_path=str(source_root),
        output_path=str(output_path),
        max_files=2,
        max_rows=3,
        max_bytes=10_000,
    )
    records = _read_jsonl_gz(output_path)

    assert [record["text"] for record in records] == [
        "Run started",
        "Step failed: exit status 1",
        "stderr: traceback (most recent call last)",
    ]
    assert stats.files_used == 1
    assert stats.rows_written == 3


def test_materialize_loghub_eval_sample_respects_file_and_row_caps(tmp_path):
    source_root = tmp_path / "loghub_source"
    output_path = tmp_path / "raw" / "diagnostic_logs" / "loghub" / "apache.jsonl.gz"

    _write_lines(source_root / "apache" / "a.log", ["line one", "line two", "line three"])
    _write_lines(source_root / "apache" / "b.log", ["line four"])

    stats = materialize_loghub_eval_sample(
        source_path=str(source_root),
        output_path=str(output_path),
        max_files=1,
        max_rows=2,
        max_bytes=10_000,
    )
    records = _read_jsonl_gz(output_path)

    assert [record["text"] for record in records] == ["line one", "line two"]
    assert stats.files_used == 1
    assert stats.rows_written == 2


def test_materialize_loghub_eval_sample_enforces_max_bytes(tmp_path):
    source_root = tmp_path / "loghub_source"
    output_path = tmp_path / "raw" / "diagnostic_logs" / "loghub" / "apache.jsonl.gz"

    _write_lines(
        source_root / "apache" / "a.log",
        [
            "ERROR this is a long diagnostic line with details 0001",
            "ERROR this is a long diagnostic line with details 0002",
            "ERROR this is a long diagnostic line with details 0003",
        ],
    )

    stats = materialize_loghub_eval_sample(
        source_path=str(source_root),
        output_path=str(output_path),
        max_files=1,
        max_rows=100,
        max_bytes=140,
    )
    records = _read_jsonl_gz(output_path)

    assert 0 < len(records) < 3
    assert stats.rows_written == len(records)
    assert stats.bytes_written <= 140


def test_diagnostic_log_output_paths_match_long_tail_registry_paths():
    raw_root = "gs://example-bucket/raw/long_tail"
    datasets = long_tail_raw_validation_sets(raw_root=raw_root, family=LongTailPplFamily.DIAGNOSTIC_LOGS)

    assert (
        diagnostic_log_eval_output_path(raw_root, slice_name="ghalogs")
        == datasets["long_tail_ppl/diagnostic_logs/ghalogs"].input_path
    )
    assert (
        diagnostic_log_eval_output_path(raw_root, slice_name="loghub_apache")
        == datasets["long_tail_ppl/diagnostic_logs/loghub_apache"].input_path
    )
