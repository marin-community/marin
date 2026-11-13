# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from experiments.post_training.task_curation import census_binding, registry
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards


def test_registry_loads_the_pinned_file_and_rejects_corrupted_cache_bytes(tmp_path, monkeypatch):
    payload = b'{"source": {"count": 3}}\n'
    cached = tmp_path / "cached.json"
    cached.write_bytes(payload)
    reference = {
        "repo": "open-athena/registry-fixture",
        "revision": "a" * 40,
        "path": "receipts/part-000.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))

    def download(**kwargs):
        assert kwargs == {
            "repo_id": reference["repo"],
            "repo_type": "dataset",
            "revision": reference["revision"],
            "filename": reference["path"],
        }
        return str(cached)

    monkeypatch.setattr(registry, "hf_hub_download", download)
    assert registry.read_registry_json(path) == {"source": {"count": 3}}
    cached.write_bytes(b'{"source": {"count": 4}}\n')
    with pytest.raises(ValueError, match="Registry bytes changed"):
        registry.read_registry_json(path)


@pytest.mark.parametrize("corrupt_readback", [False, True])
def test_archive_updates_references_only_after_every_document_verifies(tmp_path, monkeypatch, corrupt_readback):
    staged = tmp_path / "staged"
    staged.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    cached = tmp_path / "cache"
    cached.mkdir()
    originals = {"receipts/part-000.json": b'{"kept": {"source_id": "kept"}}\n', "lexical.json": b'{"matched": 1}\n'}
    tail = "receipts/part-001.json"
    for name, data in originals.items():
        (staged / name).parent.mkdir(parents=True, exist_ok=True)
        (staged / name).write_bytes(data)
        (checkout / name).parent.mkdir(parents=True, exist_ok=True)
        (checkout / name).write_text("existing reference")
    (checkout / tail).write_text("existing tail reference")
    remote = {"older.json": b'{"preserved": true}', tail: b'{"kept": {"source_id": "kept"}}'}

    class Archive:
        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)

        def create_commit(self, *, operations, **kwargs):
            assert kwargs["parent_commit"] == "a" * 40
            for operation in operations:
                remote[operation.path_in_repo] = operation.path_or_fileobj
            return SimpleNamespace(oid="b" * 40)

    def download(*, repo_id, repo_type, revision, filename):
        assert (repo_id, repo_type, revision) == ("fixture/registry", "dataset", "b" * 40)
        path = cached / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = remote[filename]
        path.write_bytes(b"corrupt" if corrupt_readback and filename == "lexical.json" else data)
        return str(path)

    monkeypatch.setattr(registry, "HfApi", Archive)
    monkeypatch.setattr(registry, "hf_hub_download", download)
    arguments = dict(repo="fixture/registry", parent_revision="a" * 40, commit_message="Record measurements")
    if corrupt_readback:
        with pytest.raises(ValueError, match="Registry bytes changed"):
            registry.archive_registry_documents(staged, checkout, **arguments)
        assert {name: (checkout / name).read_text() for name in originals} == dict.fromkeys(
            originals, "existing reference"
        )
        assert (checkout / tail).read_text() == "existing tail reference"
    else:
        registry.archive_registry_documents(staged, checkout, **arguments)
        assert {name: registry.read_registry_json(checkout / name) for name in originals} == {
            "receipts/part-000.json": {"kept": {"source_id": "kept"}},
            "lexical.json": {"matched": 1},
        }
        assert registry.read_registry_json(checkout / tail) == {}

        class Receipt(BaseModel):
            source_id: str

        receipts = read_receipt_shards(checkout / "receipts", Receipt, "fixture", load_json=registry.read_registry_json)
        assert {name: receipt.source_id for name, receipt in receipts.items()} == {"kept": "kept"}
    assert remote == {"older.json": b'{"preserved": true}', **originals, tail: b"{}\n"}
    assert {name: (staged / name).read_bytes() for name in originals} == originals


def test_census_staging_preserves_an_existing_reference_directory(tmp_path, monkeypatch):
    censuses = tmp_path / "censuses.json"
    censuses.write_text("[]")
    output = tmp_path / "receipts"
    output.mkdir()
    reference = output / "part-000.json"
    reference.write_text('{"revision": "existing"}')
    monkeypatch.setattr(sys, "argv", ["census_binding", "--censuses", str(censuses), "--output-dir", str(output)])
    with pytest.raises(FileExistsError):
        census_binding.main()
    assert reference.read_text() == '{"revision": "existing"}'
