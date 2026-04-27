# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import zipfile

from marin.datakit.download.diagnostic_logs import (
    extract_ghalogs,
    ghalogs_member_to_record,
    sanitize_diagnostic_log_text,
)


def _read_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def test_ghalogs_member_to_record_sanitizes_and_partitions():
    record = ghalogs_member_to_record(
        "owner/repo/run-1/job.log",
        b"ERROR token=abc123456789 contact alice@example.com path=/home/alice/project",
    )

    assert record is not None
    assert record["source"] == "ghalogs"
    assert record["archive_path"] == "owner/repo/run-1/job.log"
    assert "abc123456789" not in record["text"]
    assert "alice@example.com" not in record["text"]
    assert "<REDACTED_SECRET>" in record["text"]
    assert "<USER_0_EMAIL>" in record["text"]
    assert "/home/<USER_0>/project" in record["text"]
    assert record["partition"] in {"train", "dev", "test", "issue_5093_holdout"}


def test_extract_ghalogs_is_sample_capped(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with zipfile.ZipFile(input_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr("repo-a/run-1/job.log", "ERROR token=abc123456789 traceback")
        archive.writestr("repo-b/run-2/job.log", "FAILED alice@example.com /Users/alice/project")

    extract_ghalogs(str(input_dir), str(output_dir), max_members=1)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["sample_limits"]["max_members"] == 1
    assert metadata["counters"]["seen_members"] == 1
    assert metadata["counters"]["kept_records"] == 1

    kept_records = []
    for partition in ("train", "dev", "test", "issue_5093_holdout"):
        kept_records.extend(_read_jsonl(str(output_dir / partition / "data-00000-of-00001.jsonl")))
    assert len(kept_records) == 1
