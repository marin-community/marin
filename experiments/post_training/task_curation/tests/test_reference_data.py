# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The HF reference layer and the license-registry reader and writer behind it."""

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from finestore.rl.licenses import LicenseReceipt

from experiments.post_training.task_curation import license_binding, reference_data
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards


def _receipt(source_id: str, dataset: str, revision: str) -> LicenseReceipt:
    return LicenseReceipt(
        resolver_version="fixture-resolver",
        source_id=source_id,
        dataset_id=dataset,
        revision=revision,
        resolved_at="2026-11-16T00:00:00+00:00",
        evidence=[],
        bundled_third_party=False,
        model_generated_outputs=False,
    )


@pytest.fixture(autouse=True)
def real_reference_layer(monkeypatch: pytest.MonkeyPatch):
    """These tests exercise the reference layer itself; undo the registry isolation."""
    monkeypatch.setattr(license_binding, "load_referenced_json", reference_data.load_referenced_json)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """An in-memory private dataset: heads chain through create_commit."""
    state: dict = {"files": {}, "head": None, "private": True, "fail_next_commit": False, "revisions": set()}
    counter = itertools.count(1)

    class FakeApi:
        def repo_info(self, repo_id, repo_type="dataset"):
            assert state["head"] is not None, "dataset does not exist"
            return SimpleNamespace(sha=state["head"], private=state["private"])

        def create_repo(self, repo_id, repo_type="dataset", private=True, exist_ok=False):
            state["private"] = private
            if state["head"] is None:
                state["head"] = f"{next(counter):x}".zfill(40)  # the empty-repository commit
                state["revisions"].add(state["head"])

        def create_commit(self, repo_id, repo_type="dataset", parent_commit=None, commit_message="", operations=()):
            if state["fail_next_commit"]:
                state["fail_next_commit"] = False
                raise RuntimeError("simulated stale head")
            if parent_commit is not None:
                assert parent_commit == state["head"], "commit parented at a stale head"
            for operation in operations:
                state["files"][operation.path_in_repo] = operation.path_or_fileobj
            state["head"] = f"{next(counter):x}".zfill(40)
            state["revisions"].add(state["head"])
            return SimpleNamespace(oid=state["head"])

    def download(repo_id, repo_type="dataset", revision=None, filename=None):
        assert revision in state["revisions"], "download at an unknown revision"
        payload = state["files"][filename]
        path = tmp_path / filename.replace("/", "-")
        path.write_bytes(payload)
        return str(path)

    monkeypatch.setattr(reference_data, "HfApi", FakeApi)
    monkeypatch.setattr(reference_data, "hf_hub_download", download)
    return state


def test_read_verifies_referenced_bytes(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(
        index_path, "open-athena/example", {"part-000.json": b'{"a": 1}\n'}, message="Seed"
    )
    assert reference_data.load_referenced_json(index_path) == {"part-000.json": {"a": 1}}
    hub["files"]["part-000.json"] = b'{"a": 2}\n'
    with pytest.raises(ValueError, match="bytes changed"):
        reference_data.load_referenced_json(index_path)


def test_commit_reference_documents_upserts_one_shard_and_bumps_the_index(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(
        index_path, "open-athena/example", {"part-000.json": b'{"a": 1}\n'}, message="Seed"
    )
    revision = reference_data.commit_reference_documents(
        index_path,
        {"part-000.json": b'{"a": 2}\n', "part-001.json": b'{"b": 3}\n'},
        message="Upsert",
    )
    assert revision == hub["head"]
    index = reference_data.read_reference_index(index_path)
    assert index.revision == revision
    assert sorted((reference.path, reference.sha256) for reference in index.references) == sorted(
        [
            ("part-000.json", hashlib.sha256(b'{"a": 2}\n').hexdigest()),
            ("part-001.json", hashlib.sha256(b'{"b": 3}\n').hexdigest()),
        ]
    )
    assert reference_data.load_referenced_json(index_path) == {"part-000.json": {"a": 2}, "part-001.json": {"b": 3}}


def test_commit_reference_documents_retries_a_stale_head(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(index_path, "open-athena/example", {"part-000.json": b"{}\n"}, message="Seed")
    stale = hub["head"]
    hub["files"]["part-000.json"] = b'{"moved": true}\n'
    hub["head"] = f"{99000000000:x}".zfill(40)  # a concurrent writer moved the head
    hub["fail_next_commit"] = True
    reference_data.commit_reference_documents(index_path, {"part-000.json": b'{"a": 1}\n'}, message="Upsert")
    assert hub["files"]["part-000.json"] == b'{"a": 1}\n'
    assert reference_data.read_reference_index(index_path).revision == hub["head"] != stale


def test_append_license_receipt_appends_a_new_part(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {"part-000.json": json.dumps({"fixture:source": first.model_dump(mode="json")}, indent=2).encode()},
        message="Seed",
    )
    second = _receipt("fixture:other", "fixture/dataset", "b" * 40)
    revision = license_binding.append_license_receipt(index_path, second, message="Append")
    receipts = license_binding.read_license_receipts(index_path)
    assert set(receipts) == {"fixture:source", "fixture:other"}
    assert receipts["fixture:other"].revision == "b" * 40
    assert revision == hub["head"]


def test_append_license_receipt_upserts_the_row_in_place(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {"part-000.json": json.dumps({"fixture:source": first.model_dump(mode="json")}, indent=2).encode()},
        message="Seed",
    )
    updated = _receipt("fixture:source", "fixture/dataset", "c" * 40)
    license_binding.append_license_receipt(index_path, updated, message="Update pin")
    receipts = license_binding.read_license_receipts(index_path)
    assert len(receipts) == 1
    assert receipts["fixture:source"].revision == "c" * 40


def test_append_license_receipt_rejects_a_mismatched_key(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(index_path, license_binding.LICENSE_RECEIPT_REPO, {}, message="Seed")
    with pytest.raises(ValueError, match="differs from the receipt source identity"):
        license_binding.append_license_receipt(
            index_path,
            _receipt("MarinSkyRL:math500", "fixture/dataset", "a" * 40),
            message="Append",
        )


def test_read_license_receipts_still_reads_staging_directories(tmp_path: Path):
    directory = tmp_path / "receipts"
    directory.mkdir()
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    (directory / "part-000.json").write_text(json.dumps({"fixture:source": first.model_dump(mode="json")}))
    receipts = license_binding.read_license_receipts(directory)
    assert set(receipts) == {"fixture:source"}
    with pytest.raises(ValueError, match="Duplicate or mismatched"):
        (directory / "part-001.json").write_text(
            json.dumps(
                {"fixture:source": _receipt("fixture:source", "fixture/dataset", "b" * 40).model_dump(mode="json")}
            )
        )
        license_binding.read_license_receipts(directory)


def test_read_receipt_shards_rejects_duplicate_identities(tmp_path: Path):
    directory = tmp_path / "receipts"
    directory.mkdir()
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    second = _receipt("fixture:other", "fixture/dataset", "b" * 40)
    (directory / "part-000.json").write_text(json.dumps({"fixture:source": first.model_dump(mode="json")}))
    (directory / "part-001.json").write_text(json.dumps({"fixture:source": second.model_dump(mode="json")}))
    with pytest.raises(ValueError, match="Duplicate or mismatched"):
        read_receipt_shards(
            directory, LicenseReceipt, "license receipt", load_json=lambda shard: json.loads(shard.read_text())
        )
