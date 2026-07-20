# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

from tests.cluster.vllm import snowball_export
from tests.cluster.vllm.snowball_export import ExportReport, load_pinned_tokenizer, tree_manifest


def test_tree_manifest_matches_nested_file_contract(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a").write_bytes(b"alpha")
    (tmp_path / "nested" / "b").write_bytes(b"beta")

    actual_digest, files = tree_manifest(tmp_path)

    expected = hashlib.sha256()
    for relative_path, payload in (("a", b"alpha"), ("nested/b", b"beta")):
        expected.update(relative_path.encode())
        expected.update(b"\0")
        expected.update(hashlib.sha256(payload).digest())
    assert actual_digest == expected.hexdigest()
    assert [(file.path, file.size) for file in files] == [("a", 5), ("nested/b", 4)]


def test_export_report_json_round_trip(tmp_path) -> None:
    (tmp_path / "file").write_bytes(b"payload")
    tree_sha256, files = tree_manifest(tmp_path)
    report = ExportReport(
        platform="tpu",
        logical_bf16_parameters_sha256="parameters",
        executor_config_sha256="config",
        tokenizer="tokenizer",
        tokenizer_revision="revision",
        tree_sha256=tree_sha256,
        total_bytes=7,
        files=files,
        canonical_tree_match=True,
        uploaded_uri="gs://bucket/export",
    )

    assert ExportReport.from_json_bytes(report.to_json_bytes()) == report


def test_load_pinned_tokenizer_serializes_canonical_hub_provenance(monkeypatch, tmp_path) -> None:
    source = Path(__file__).parents[3] / "lib" / "levanter" / "tests"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    shutil.copy(source / "gpt2_tokenizer.json", snapshot / "tokenizer.json")
    shutil.copy(source / "gpt2_tokenizer_config.json", snapshot / "tokenizer_config.json")
    monkeypatch.setattr(snowball_export, "snapshot_download", lambda *args, **kwargs: str(snapshot))

    tokenizer = load_pinned_tokenizer("org/tokenizer", "deadbeef")
    output = tmp_path / "output"
    tokenizer.save_pretrained(output)
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())

    assert tokenizer.name_or_path == "org/tokenizer"
    assert tokenizer_config["is_local"] is False
    assert tokenizer_config["local_files_only"] is False
