# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from marin.datakit.download import diagnostic_logs
from marin.datakit.download.diagnostic_logs import (
    GHALOGS_ROUGH_TOKENS_B,
    GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH,
    GHALOGS_STAGED_PREFIX,
    SOURCE_INVENTORY,
    DiagnosticPartition,
    ExtractedDiagnosticLogs,
    ExtractedPartitionedDiagnosticLogs,
    MaterializedDiagnosticLogParquet,
    assign_partition,
    download_ghalogs_step,
    extract_diagnostic_logs,
    extract_ghalogs_step,
    ghalogs_member_to_record,
    ghalogs_public_normalize_steps,
    logchunks_example_to_record,
    loghub_file_to_record,
    materialize_ghalogs_partition_to_parquet,
    materialize_ghalogs_to_parquet,
    sanitize_diagnostic_log_text,
    stage_ghalogs_archive,
)
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from marin.execution.lazy import materialized_config
from marin.execution.step_runner import StepRunner

from experiments.datasets.diagnostic_logs import _ghalogs_normalized, ghalogs_dataset


def _read_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_parquet_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _member_path_for_partition(partition: DiagnosticPartition) -> str:
    for index in range(10_000):
        member_path = f"repo-{partition.value}/run-{index}/job.log"
        if assign_partition(f"ghalogs:{member_path}") == partition:
            return member_path
    raise AssertionError(f"Could not find member path for {partition}")


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
    inventory = {source.source_label: source for source in SOURCE_INVENTORY}

    assert inventory["ghalogs"].policy.training_allowed is True
    assert inventory["ghalogs"].policy.requires_sanitization is True
    assert inventory["ghalogs"].rough_tokens_b == GHALOGS_ROUGH_TOKENS_B
    assert inventory["logchunks"].policy.eval_only is True
    assert inventory["loghub"].compressed_size_bytes == 7_513_088


def test_all_sources_includes_normalized_ghalogs_public():
    source = all_sources()["ghalogs/public"]

    assert source.rough_token_count_b == GHALOGS_ROUGH_TOKENS_B
    assert [step.name for step in source.normalize_steps] == [
        "raw/diagnostic_logs/ghalogs_public_archive",
        "processed/diagnostic_logs/ghalogs_public_parquet",
        "processed/diagnostic_logs/ghalogs_public_train_parquet",
        "normalized/ghalogs/public",
    ]
    # normalize depends on the train partition; the parquet materialize depends
    # on the archive download so the ~142 GB archive is auto-staged first.
    assert source.normalized.deps == [source.normalize_steps[-2]]
    assert source.normalize_steps[1].deps == [source.normalize_steps[0]]


def test_ghalogs_dataset_reads_datakit_normalized_output():
    step = ghalogs_dataset(tokenizer="test-tokenizer")
    normalized = _ghalogs_normalized()

    assert normalized.name == "normalized/ghalogs/public"
    cfg = materialized_config(step, "gs://prefix")
    assert cfg.train_paths == [f"{normalized.path('gs://prefix')}/outputs/main/*.parquet"]
    assert cfg.validation_paths == []


def test_extract_diagnostic_logs_is_sample_capped(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    ghalogs_dir = input_dir / "ghalogs" / "zenodo-14796970" / "zenodo.org" / "records" / "14796970" / "files"
    ghalogs_dir.mkdir(parents=True)
    with zipfile.ZipFile(ghalogs_dir / "github_run_logs.zip", "w") as archive:
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

    extracted = extract_diagnostic_logs(
        str(input_dir / "ghalogs" / "zenodo-14796970"),
        str(output_dir),
        logchunks_input_path=str(input_dir),
        loghub_input_path=str(input_dir),
        max_ghalogs_members=1,
        max_logchunks_examples=1,
        max_loghub_files=1,
    )

    assert isinstance(extracted, ExtractedDiagnosticLogs)
    assert extracted.ghalogs.record_count == 1
    assert extracted.logchunks.record_count == 1
    assert extracted.loghub.record_count == 1
    assert extracted.ghalogs.metadata_path == str(output_dir / "metadata.json")
    assert extracted.logchunks.metadata_path == str(output_dir / "eval_only" / "logchunks" / "metadata.json")
    assert extracted.loghub.metadata_path == str(output_dir / "eval_only" / "loghub" / "metadata.json")

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


def test_extract_diagnostic_logs_uses_staged_ghalogs_and_fetches_missing_eval_sources(tmp_path, monkeypatch):
    ghalogs_input_dir = tmp_path / "ghalogs" / "zenodo-14796970"
    ghalogs_archive_dir = ghalogs_input_dir / "zenodo.org" / "records" / "14796970" / "files"
    output_dir = tmp_path / "output"
    ghalogs_archive_dir.mkdir(parents=True)

    with zipfile.ZipFile(ghalogs_archive_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr("repo-a/run-1/job.log", "ERROR token=abc123456789 traceback")

    def _fake_fetch_logchunks(destination_dir: str) -> str:
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination / "LogChunks.zip", "w") as archive:
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
                </Examples>
                """,
            )
        return str(destination)

    def _fake_fetch_loghub(destination_dir: str) -> str:
        destination_root = Path(destination_dir)
        destination = destination_root / "loghub"
        linux_dir = destination / "Linux"
        linux_dir.mkdir(parents=True, exist_ok=True)
        (linux_dir / "Linux_2k.log").write_text("FAILED path=/home/alice/project", encoding="utf-8")
        return str(destination_root)

    monkeypatch.setattr(
        "marin.datakit.download.diagnostic_logs._stage_logchunks_if_missing",
        _fake_fetch_logchunks,
    )
    monkeypatch.setattr(
        "marin.datakit.download.diagnostic_logs._stage_loghub_if_missing",
        _fake_fetch_loghub,
    )

    extracted = extract_diagnostic_logs(
        str(ghalogs_input_dir),
        str(output_dir),
        logchunks_input_path=str(tmp_path / "auto" / "logchunks"),
        loghub_input_path=str(tmp_path / "auto" / "loghub"),
        max_ghalogs_members=1,
        max_logchunks_examples=1,
        max_loghub_files=1,
    )

    assert isinstance(extracted, ExtractedDiagnosticLogs)
    assert extracted.ghalogs.record_count == 1
    assert extracted.logchunks.record_count == 1
    assert extracted.loghub.record_count == 1


def test_extract_ghalogs_step_persists_typed_artifact(tmp_path):
    input_dir = tmp_path / "input"
    archive_dir = input_dir / "zenodo.org" / "records" / "14796970" / "files"
    archive_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr("repo-a/run-1/job.log", "ERROR token=abc123456789 traceback")

    step = extract_ghalogs_step(
        source_path=str(input_dir),
        max_members=1,
        output_path_prefix=str(tmp_path / "steps"),
    )
    StepRunner().run([step])

    loaded = read_artifact(step.output_path, ExtractedPartitionedDiagnosticLogs)
    assert loaded.source_label == "ghalogs"
    assert loaded.record_count == 1
    assert loaded.metadata_path.endswith("/metadata.json")


def test_materialize_ghalogs_to_parquet_writes_reusable_shards(tmp_path):
    input_dir = tmp_path / "input" / "ghalogs" / "zenodo-14796970"
    archive_dir = input_dir / "zenodo.org" / "records" / "14796970" / "files"
    output_dir = tmp_path / "materialized"
    archive_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr("repo-a/run-1/job.log", "ERROR token=abc123456789 traceback")
        archive.writestr("repo-b/run-2/job.log", "FAILED alice@example.com /Users/alice/project")
        archive.writestr("repo-c/run-3/job.log", "WARNING path=/home/bob/src")

    materialized = materialize_ghalogs_to_parquet(
        str(input_dir),
        str(output_dir),
        max_members=3,
        num_shards=2,
        max_workers=1,
    )

    assert isinstance(materialized, MaterializedDiagnosticLogParquet)
    assert materialized.source_label == "ghalogs"
    assert materialized.record_count == 3
    rows = _read_parquet_rows(output_dir)
    assert len(rows) == 3
    assert {row["source"] for row in rows} == {"ghalogs"}
    assert {row["partition"] for row in rows} <= {partition.value for partition in DiagnosticPartition}
    assert all("abc123456789" not in row["text"] for row in rows)


def test_materialize_ghalogs_partition_to_parquet_filters_one_partition(tmp_path):
    input_dir = tmp_path / "input" / "ghalogs" / "zenodo-14796970"
    archive_dir = input_dir / "zenodo.org" / "records" / "14796970" / "files"
    materialized_dir = tmp_path / "materialized"
    partition_dir = tmp_path / "train_only"
    archive_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr("repo-a/run-1/job.log", "ERROR token=abc123456789 traceback")
        archive.writestr("repo-b/run-2/job.log", "FAILED alice@example.com /Users/alice/project")
        archive.writestr("repo-c/run-3/job.log", "WARNING path=/home/bob/src")

    materialize_ghalogs_to_parquet(
        str(input_dir),
        str(materialized_dir),
        max_members=3,
        num_shards=2,
        max_workers=1,
    )
    train_partition = materialize_ghalogs_partition_to_parquet(
        str(materialized_dir),
        str(partition_dir),
        partition=DiagnosticPartition.TRAIN,
        max_workers=1,
    )

    train_rows = _read_parquet_rows(partition_dir)
    assert isinstance(train_partition, MaterializedDiagnosticLogParquet)
    assert train_partition.source_label == "ghalogs"
    assert train_partition.record_count == len(train_rows)
    assert all(row["partition"] == DiagnosticPartition.TRAIN.value for row in train_rows)


def test_ghalogs_public_normalize_steps_write_datakit_normalized_train_partition(tmp_path, monkeypatch):
    input_dir = tmp_path / "input" / "ghalogs" / "zenodo-14796970"
    archive_dir = input_dir / "zenodo.org" / "records" / "14796970" / "files"
    archive_dir.mkdir(parents=True)

    train_member = _member_path_for_partition(DiagnosticPartition.TRAIN)
    dev_member = _member_path_for_partition(DiagnosticPartition.DEV)
    with zipfile.ZipFile(archive_dir / "github_run_logs.zip", "w") as archive:
        archive.writestr(train_member, "ERROR token=abc123456789 traceback")
        archive.writestr(dev_member, "FAILED validation-only log")

    # The archive is already staged at ``source_path``; no-op the Zenodo stream
    # so the download step doesn't try to fetch the real ~142 GB archive.
    monkeypatch.setattr(diagnostic_logs, "stage_ghalogs_archive", lambda output_path: None)

    steps = ghalogs_public_normalize_steps(
        source_path=str(input_dir),
        max_members=2,
        num_materialize_shards=1,
        output_path_prefix=str(tmp_path / "steps"),
    )
    StepRunner().run(list(steps))

    normalized = read_artifact(steps[-1].output_path, NormalizedData)
    rows = _read_parquet_rows(Path(normalized.main_output_dir))

    assert len(rows) == 1
    assert rows[0]["source"] == "ghalogs"
    assert rows[0]["archive_path"] == train_member
    assert rows[0]["partition"] == DiagnosticPartition.TRAIN.value
    assert "abc123456789" not in rows[0]["text"]
    assert rows[0]["source_id"] != rows[0]["id"]


def test_ghalogs_public_normalize_steps_read_where_download_wrote(tmp_path, monkeypatch):
    # Regression: with a custom output_path_prefix and the default relative
    # source_path, the download step resolves its output under output_path_prefix
    # while materialize must read from that same location (not marin_prefix()).
    steps_prefix = tmp_path / "steps"
    download, materialized, _train, _normalized = ghalogs_public_normalize_steps(
        max_members=1,
        num_materialize_shards=1,
        output_path_prefix=str(steps_prefix),
    )

    # Stage the fixture archive exactly where the download step resolves its
    # output — a location distinct from marin_prefix(), which the misaligned
    # version read from and would have missed.
    staged_archive = Path(download.output_path) / GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH
    staged_archive.parent.mkdir(parents=True)
    train_member = _member_path_for_partition(DiagnosticPartition.TRAIN)
    with zipfile.ZipFile(staged_archive, "w") as archive:
        archive.writestr(train_member, "ERROR token=abc123456789 traceback")

    # Archive is pre-staged; no-op the Zenodo stream.
    monkeypatch.setattr(diagnostic_logs, "stage_ghalogs_archive", lambda output_path: None)
    StepRunner().run([download, materialized])

    rows = _read_parquet_rows(Path(materialized.output_path))
    assert [row["archive_path"] for row in rows] == [train_member]


class _FakeStreamResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield from self._chunks


class _FakeSession:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.get_calls = 0

    def get(self, url: str, *, stream: bool, timeout) -> _FakeStreamResponse:
        self.get_calls += 1
        return _FakeStreamResponse(self._chunks)


def _patch_zenodo_stream(monkeypatch, payload: bytes, chunks: list[bytes]) -> _FakeSession:
    session = _FakeSession(chunks)
    monkeypatch.setattr(diagnostic_logs, "build_retrying_session", lambda: session)
    monkeypatch.setattr(diagnostic_logs, "GHALOGS_ARCHIVE_BYTES", len(payload))
    return session


def test_stage_ghalogs_archive_streams_to_expected_path(tmp_path, monkeypatch):
    payload = b"PK\x03\x04" + b"github-run-log-bytes" * 32
    _patch_zenodo_stream(monkeypatch, payload, [payload[:40], payload[40:]])

    stage_ghalogs_archive(str(tmp_path))

    staged = tmp_path / GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH
    assert staged.read_bytes() == payload


def test_stage_ghalogs_archive_skips_when_correctly_sized_copy_exists(tmp_path, monkeypatch):
    payload = b"already-staged-archive-bytes"
    session = _patch_zenodo_stream(monkeypatch, payload, [payload])

    staged = tmp_path / GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH
    staged.parent.mkdir(parents=True)
    staged.write_bytes(payload)

    stage_ghalogs_archive(str(tmp_path))

    # A correctly-sized copy short-circuits before any HTTP request.
    assert session.get_calls == 0
    assert staged.read_bytes() == payload


def test_stage_ghalogs_archive_raises_on_size_mismatch(tmp_path, monkeypatch):
    payload = b"the-full-expected-archive"
    # Server truncates the stream: streamed bytes fall short of the expected size.
    _patch_zenodo_stream(monkeypatch, payload, [payload[:10]])

    with pytest.raises(RuntimeError, match="size mismatch"):
        stage_ghalogs_archive(str(tmp_path))

    # Failed staging must not publish a partial archive at the final path.
    assert not (tmp_path / GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH).exists()


def test_download_ghalogs_step_targets_staged_prefix_and_streams_archive(tmp_path, monkeypatch):
    payload = b"downloaded-archive-payload"
    _patch_zenodo_stream(monkeypatch, payload, [payload])

    step = download_ghalogs_step(output_path_prefix=str(tmp_path))
    assert step.output_path == f"{tmp_path}/{GHALOGS_STAGED_PREFIX}"

    StepRunner().run([step])

    staged = Path(step.output_path) / GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH
    assert staged.read_bytes() == payload
