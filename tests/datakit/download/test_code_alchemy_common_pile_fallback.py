# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

import marin.datakit.download.code_alchemy_common_pile_fallback as fallback


def _record(text: str, encoding: str = "UTF-8", **metadata_overrides: object) -> dict[str, object]:
    raw = text.encode(encoding)
    blob_id = hashlib.sha1(raw).hexdigest()
    content_id = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    metadata = {
        "blob_id": blob_id,
        "content_id": content_id,
        "src_encoding": encoding,
        "length_bytes": len(raw),
        "repo_name": "owner/repo",
        "path": "src/file.py",
        **metadata_overrides,
    }
    return {"id": blob_id, "text": text, "metadata": metadata, "source": "stackv2"}


def _task(tmp_path: Path, targets: tuple[str, ...]) -> fallback.MirrorScanTask:
    return fallback.MirrorScanTask(
        file_index=0,
        repo_id="mirror/repo",
        revision="1" * 40,
        manifest_fingerprint="manifest",
        mirror_file=fallback.MirrorFile(
            path="00000_python0.json.gz",
            size=1,
            git_blob_id="2" * 40,
            lfs_sha256="3" * 64,
        ),
        signed_url="https://signed.invalid/object",
        target_blob_ids=targets,
        candidate_path=str(tmp_path / "candidate.parquet"),
        metrics_path=str(tmp_path / "metrics.json"),
        request_connect_timeout_seconds=1,
        request_read_timeout_seconds=1,
    )


def test_verify_record_identity_uses_declared_source_encoding():
    # U+00E9 is one byte in Latin-1 and two in UTF-8.
    record = _record("café\n", "ISO-8859-1")
    expected = record["id"]

    verified = fallback.verify_record_identity(record, expected)

    assert verified.blob_id == hashlib.sha1(record["text"].encode("ISO-8859-1")).hexdigest()
    assert verified.git_blob_sha1 == record["metadata"]["content_id"]
    assert verified.source == record["text"]


def test_verify_record_identity_rejects_raw_or_git_hash_disagreement():
    record = _record("print('exact bytes')\n")
    expected = record["id"]
    record["text"] = "print('changed bytes')\n"

    with pytest.raises(ValueError, match="raw Stack-Edu SHA-1 mismatch"):
        fallback.verify_record_identity(record, expected)

    record = _record("print('exact bytes')\n")
    record["metadata"]["content_id"] = "f" * 40
    with pytest.raises(ValueError, match="metadata.content_id mismatch"):
        fallback.verify_record_identity(record, record["id"])


def test_scan_stream_retains_only_targets_and_records_line_number(tmp_path: Path):
    ignored = _record("ignored\n")
    target = _record("target\n")
    payload = gzip.compress((json.dumps(ignored) + "\n" + json.dumps(target) + "\n").encode())
    task = _task(tmp_path, (target["id"],))

    rows, matches, line_numbers = fallback._scan_stream(io.BytesIO(payload), task)

    assert rows == 2
    assert [match.blob_id for match in matches] == [target["id"]]
    assert line_numbers == [2]


def test_list_pinned_mirror_files_requires_complete_target_language_shards(monkeypatch):
    monkeypatch.setattr(fallback, "SHARDS_PER_LANGUAGE", 2)

    def sibling(path: str):
        return SimpleNamespace(
            rfilename=path,
            size=10,
            blob_id="a" * 40,
            lfs=SimpleNamespace(sha256="b" * 64),
        )

    info = SimpleNamespace(
        sha="1" * 40,
        siblings=[
            sibling("00000_python0.json.gz"),
            sibling("00000_python1.json.gz"),
            sibling("00000_shell0.json.gz"),
            sibling("00000_shell1.json.gz"),
            sibling("00000_cpp0.json.gz"),
            sibling("README.md"),
        ],
    )
    monkeypatch.setattr(fallback, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *args, **kwargs: info))
    cfg = fallback.CodeAlchemyCommonPileFallbackConfig(revision="1" * 40)

    files = fallback.list_pinned_mirror_files(cfg)

    assert [item.path for item in files] == [
        "00000_python0.json.gz",
        "00000_python1.json.gz",
        "00000_shell0.json.gz",
        "00000_shell1.json.gz",
    ]

    info.siblings.pop(1)
    with pytest.raises(RuntimeError, match="incomplete python shard set"):
        fallback.list_pinned_mirror_files(cfg)


def test_compaction_writes_exact_canonical_schema_and_provenance(tmp_path: Path):
    records = [_record("python source\n"), _record("shell source\n")]
    target_ids = tuple(record["id"] for record in records)
    output_path = tmp_path / "fallback-sources"
    cfg = fallback.CodeAlchemyCommonPileFallbackConfig(
        revision="1" * 40,
        target_blob_ids=target_ids,
        output_path=str(output_path),
    )
    tasks = []
    results = []
    for index, record in enumerate(records):
        task = _task(tmp_path, target_ids)
        task = dataclasses.replace(
            task,
            file_index=index,
            mirror_file=dataclasses.replace(task.mirror_file, path=f"00000_python{index}.json.gz"),
            candidate_path=str(tmp_path / f"candidate-{index}.parquet"),
            metrics_path=str(tmp_path / f"metrics-{index}.json"),
        )
        verified = fallback.verify_record_identity(record, record["id"])
        fallback._candidate_frame([verified], task, [index + 1]).write_parquet(task.candidate_path)
        tasks.append(task)
        results.append(
            fallback.MirrorFileResult(
                file_index=index,
                mirror_file=task.mirror_file.path,
                expected_compressed_bytes=1,
                input_rows=1,
                matched_rows=1,
                matched_blob_ids=(record["id"],),
                candidate_path=task.candidate_path,
                elapsed_seconds=0.1,
            )
        )

    totals = fallback.compact_exact_results(cfg, tasks, results)

    assert totals["canonical_rows"] == 2
    for record in records:
        canonical = pl.read_parquet(output_path / f"data/blob_prefix={record['id'][:2]}/part-00000.parquet")
        assert canonical.schema == {"blob_id": pl.String, "source": pl.String}
        assert canonical.to_dicts() == [{"blob_id": record["id"], "source": record["text"]}]
        provenance = pl.read_parquet(
            output_path / f"provenance/data/blob_prefix={record['id'][:2]}/part-00000.parquet"
        )
        assert "source" not in provenance.columns
        assert provenance.get_column("raw_sha1").to_list() == [record["id"]]
