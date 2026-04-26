# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from marin.datakit.download.diagnostic_logs import (
    DiagnosticSourceStatus,
    extract_starcoder_fixture_logs,
    looks_like_diagnostic_log_row,
    sanitize_diagnostic_log_text,
    source_inventory,
    starcoder_fixture_row_to_record,
    training_ready_sources,
)


def _write_parquet(path: str, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _read_jsonl(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_source_inventory_contains_required_candidates():
    inventory = {entry.name: entry for entry in source_inventory()}

    assert "ghalogs" in inventory
    assert "logchunks" in inventory
    assert "loghub" in inventory
    assert "github_fixture_logs_from_source_corpora" in inventory
    assert inventory["ghalogs"].status == DiagnosticSourceStatus.TRAINING_READY
    assert inventory["logchunks"].status == DiagnosticSourceStatus.EVAL_ONLY
    assert inventory["loghub"].status == DiagnosticSourceStatus.EVAL_ONLY


def test_training_ready_sources_are_opt_in_and_narrow():
    ready = training_ready_sources()
    assert [source.name for source in ready] == ["ghalogs", "github_fixture_logs_from_source_corpora"]


def test_sanitize_diagnostic_log_text_redacts_secrets_and_identifiers():
    text = (
        "token=supersecretvalue123 ghp_abcdefghijklmnopqrstuvwxyz123456 "
        "email alice@example.com path=/Users/alice/project user Alice failed"
    )
    redacted = sanitize_diagnostic_log_text(text)
    assert "supersecretvalue123" not in redacted
    assert "alice@example.com" not in redacted
    assert "/Users/alice" not in redacted
    assert "user Alice failed" not in redacted
    assert "<REDACTED_SECRET>" in redacted
    assert "<REDACTED_GITHUB_TOKEN>" in redacted
    assert "<USER_0_EMAIL>" in redacted
    assert "/Users/<USER_0>/project" in redacted
    assert "user <USER_0> failed" in redacted


def test_root_level_log_directory_is_detected():
    assert looks_like_diagnostic_log_row("logs/build.txt", "ERROR traceback (most recent call last)")


def test_starcoder_fixture_row_to_record_detects_and_sanitizes():
    row = {
        "max_stars_repo_path": "tests/fixtures/build_logs/stderr.log",
        "max_stars_repo_name": "example/repo",
        "content": "ERROR token=abc123456789 traceback (most recent call last)",
    }

    record = starcoder_fixture_row_to_record(row)
    assert record is not None
    assert record["source"] == "github_fixture_logs"
    assert record["repo_name"] == "example/repo"
    assert record["repo_path"] == "tests/fixtures/build_logs/stderr.log"
    assert "abc123456789" not in record["text"]
    assert "<REDACTED_SECRET>" in record["text"]


def test_extract_starcoder_fixture_logs_is_sample_capped(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    shard_dir = input_dir / "nested"
    shard_dir.mkdir(parents=True)

    _write_parquet(
        str(shard_dir / "a.parquet"),
        [
            {
                "max_stars_repo_path": "ci/logs/stderr.log",
                "max_stars_repo_name": "a/repo",
                "content": "ERROR token=abc123456789 traceback (most recent call last)",
            },
            {
                "max_stars_repo_path": "src/main.py",
                "max_stars_repo_name": "a/repo",
                "content": "print('hello')",
            },
        ],
    )
    _write_parquet(
        str(shard_dir / "b.parquet"),
        [
            {
                "max_stars_repo_path": "tests/golden/failure.log",
                "max_stars_repo_name": "b/repo",
                "content": "FAILED: build panic: something bad happened",
            }
        ],
    )

    extract_starcoder_fixture_logs(str(input_dir), str(output_dir), max_parquet_files=1, max_rows=1)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["sample_limits"]["max_parquet_files"] == 1
    assert metadata["sample_limits"]["max_rows"] == 1
    assert metadata["sampling"]["sampled_parquet_files"] == 1
    assert metadata["counters"]["seen_rows"] == 1
    assert metadata["counters"]["kept_rows"] == 1

    kept_records = []
    for partition in ("train", "dev", "test", "issue_5093_holdout"):
        kept_records.extend(_read_jsonl(str(output_dir / partition / "data-00000-of-00001.jsonl")))
    assert len(kept_records) == 1
