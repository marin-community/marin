# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import xml.etree.ElementTree as ET
import zipfile

from marin.datakit.download.diagnostic_logs import (
    extract_diagnostic_logs,
    ghalogs_member_to_record,
    logchunks_example_to_record,
    loghub_file_to_record,
    sanitize_diagnostic_log_text,
    source_inventory,
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


def test_logchunks_example_to_record_sanitizes():
    example = ET.fromstring(
        """
        <Example>
          <Log>JavaScript/example@repo/failed/1.log</Log>
          <Keywords>Error</Keywords>
          <Category>0</Category>
          <Chunk>Error token=abc123456789 path=/home/alice/project</Chunk>
        </Example>
        """
    )

    record = logchunks_example_to_record("annotations.xml", 0, example)

    assert record is not None
    assert record["source"] == "logchunks"
    assert record["log_path"] == "JavaScript/example@repo/failed/1.log"
    assert "abc123456789" not in record["text"]
    assert "<REDACTED_SECRET>" in record["text"]
    assert "/home/<USER_0>/project" in record["text"]


def test_loghub_file_to_record_sanitizes():
    record = loghub_file_to_record("Linux/Linux_2k.log", b"FAILED contact alice@example.com")

    assert record is not None
    assert record["source"] == "loghub"
    assert record["source_path"] == "Linux/Linux_2k.log"
    assert "alice@example.com" not in record["text"]
    assert "<USER_0_EMAIL>" in record["text"]


def test_source_inventory_uses_shared_manifest_policy_metadata():
    inventory = {source.source_label: source for source in source_inventory()}

    assert inventory["ghalogs"].policy.training_allowed is True
    assert inventory["ghalogs"].policy.requires_sanitization is True
    assert inventory["logchunks"].policy.eval_only is True
    assert inventory["loghub"].compressed_size_bytes == 7_513_088


def test_extract_diagnostic_logs_is_sample_capped(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with zipfile.ZipFile(input_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr("repo-a/run-1/job.log", "ERROR token=abc123456789 traceback")
        archive.writestr("repo-b/run-2/job.log", "FAILED alice@example.com /Users/alice/project")

    with zipfile.ZipFile(input_dir / "LogChunks.zip", "w") as archive:
        archive.writestr(
            "LogChunks/build-failure-reason/Python/example@repo.xml",
            """
            <Examples>
              <Example>
                <Log>Python/example@repo/failed/1.log</Log>
                <Keywords>Error</Keywords>
                <Category>0</Category>
                <Chunk>Traceback token=abc123456789</Chunk>
              </Example>
              <Example>
                <Log>Python/example@repo/failed/2.log</Log>
                <Keywords>Failed</Keywords>
                <Category>1</Category>
                <Chunk>FAILED alice@example.com</Chunk>
              </Example>
            </Examples>
            """,
        )

    loghub_dir = input_dir / "loghub" / "Linux"
    loghub_dir.mkdir(parents=True)
    (loghub_dir / "Linux_2k.log").write_text("FAILED path=/home/alice/project", encoding="utf-8")
    (loghub_dir / "Linux_2k.log_structured.csv").write_text("not ingested", encoding="utf-8")

    extract_diagnostic_logs(
        str(input_dir),
        str(output_dir),
        max_ghalogs_members=1,
        max_logchunks_examples=1,
        max_loghub_files=1,
    )

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert "content_fingerprint" in metadata
    assert metadata["source_manifest"]["source_label"] == "ghalogs"
    assert metadata["source_manifest"]["policy"]["training_allowed"] is True
    assert metadata["source_manifest"]["policy"]["requires_sanitization"] is True
    assert metadata["materialized_output"]["metadata"]["sample_limits"]["max_members"] == 1
    assert metadata["materialized_output"]["metadata"]["counters"]["seen_members"] == 1
    assert metadata["materialized_output"]["record_count"] == 1

    kept_records = []
    for partition in ("train", "dev", "test", "issue_5093_holdout"):
        kept_records.extend(_read_jsonl(str(output_dir / partition / "data-00000-of-00001.jsonl")))
    assert len(kept_records) == 1

    logchunks_records = _read_jsonl(str(output_dir / "eval_only" / "logchunks" / "data-00000-of-00001.jsonl"))
    assert len(logchunks_records) == 1
    assert logchunks_records[0]["source"] == "logchunks"

    loghub_records = _read_jsonl(str(output_dir / "eval_only" / "loghub" / "data-00000-of-00001.jsonl"))
    assert len(loghub_records) == 1
    assert loghub_records[0]["source"] == "loghub"
    loghub_metadata = json.loads((output_dir / "eval_only" / "loghub" / "metadata.json").read_text())
    assert loghub_metadata["source_manifest"]["policy"]["eval_only"] is True
