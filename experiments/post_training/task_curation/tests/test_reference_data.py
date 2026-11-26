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

from experiments.post_training.task_curation import license_binding, reference_data, source_duplicates
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


# Captured at import, before any fixture patches the reference layer.
_REAL_LOAD_JSON = reference_data.load_referenced_json
_REAL_LOAD_DOCUMENT = reference_data.load_referenced_document


@pytest.fixture(autouse=True)
def real_reference_layer(monkeypatch: pytest.MonkeyPatch, isolated_license_registry: None):
    """These tests exercise the reference layer itself; undo the registry isolation."""
    monkeypatch.setattr(license_binding, "load_referenced_json", _REAL_LOAD_JSON)
    monkeypatch.setattr(reference_data, "load_referenced_document", _REAL_LOAD_DOCUMENT)


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

        def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
            assert revision == state["head"]
            return list(state["files"])

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


def test_acknowledgements_resolve_from_a_committed_pointer(tmp_path: Path, hub: dict):
    pointer = tmp_path / "acknowledgements.json"
    document = {"a" * 64: "Advisory distinct; overlap_fraction=0.0", "b" * 64: "Duplicate; replace left"}
    reference_data.create_reference_index(
        pointer,
        "open-athena/marin-source-duplicate-acknowledgements",
        {"acknowledgements.json": json.dumps({}, indent=2).encode()},
        message="Seed the pointer",
    )
    source_duplicates.publish_acknowledgements(pointer, document, message="Seed acknowledgements")
    assert source_duplicates.load_acknowledgements(pointer) == document


def test_acknowledgements_accept_a_plain_document(tmp_path: Path):
    pointer = tmp_path / "acknowledgements.json"
    document = {"c" * 64: "Advisory partial_overlap"}
    pointer.write_text(json.dumps(document))
    assert source_duplicates.load_acknowledgements(pointer) == document


def test_acknowledgements_reject_duplicate_pairs_across_documents(tmp_path: Path, hub: dict):
    pointer = tmp_path / "acknowledgements.json"
    reference_data.create_reference_index(
        pointer,
        "open-athena/marin-source-duplicate-acknowledgements",
        {"acknowledgements.json": json.dumps({}, indent=2).encode()},
        message="Seed the pointer",
    )
    source_duplicates.publish_acknowledgements(pointer, {"a" * 64: "one"}, message="Seed")
    reference_data.commit_reference_documents(
        pointer,
        {"extra.json": json.dumps({"a" * 64: "two"}, indent=2).encode()},
        message="Add a colliding document",
    )
    with pytest.raises(ValueError, match="Duplicate acknowledgement pair"):
        source_duplicates.load_acknowledgements(pointer)


def test_load_referenced_document_reads_one_pinned_file(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(
        index_path,
        "open-athena/example",
        {"part-000.json": b'{"a": 1}\n', "other.json": b'{"b": 2}\n'},
        message="Seed",
    )
    assert reference_data.load_referenced_document(index_path, "other.json") == {"b": 2}
    with pytest.raises(ValueError, match="does not list"):
        reference_data.load_referenced_document(index_path, "missing.json")


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
    hub["files"]["unrelated.json"] = b'{"moved": true}\n'
    hub["head"] = f"{99000000000:x}".zfill(40)  # a concurrent writer moved the head
    hub["revisions"].add(hub["head"])
    hub["fail_next_commit"] = True
    reference_data.commit_reference_documents(index_path, {"part-000.json": b'{"a": 1}\n'}, message="Upsert")
    assert hub["files"]["part-000.json"] == b'{"a": 1}\n'
    assert reference_data.read_reference_index(index_path).revision == hub["head"] != stale


@pytest.mark.parametrize("prefix", ["", "campaign-v1/"])
def test_append_license_receipt_appends_a_new_part(tmp_path: Path, hub: dict, prefix: str):
    index_path = tmp_path / "index.json"
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {prefix + "part-000.json": json.dumps({"fixture:source": first.model_dump(mode="json")}, indent=2).encode()},
        message="Seed",
    )
    if prefix:
        hub["files"]["part-000.json"] = b"preserved historical root shard"
    original_root = hub["files"]["part-000.json"]
    second = _receipt("fixture:other", "fixture/dataset", "b" * 40)
    revision = license_binding.append_license_receipt(index_path, second, message="Append")
    receipts = license_binding.read_license_receipts(index_path)
    assert set(receipts) == {"fixture:source", "fixture:other"}
    assert receipts["fixture:other"].revision == "b" * 40
    assert revision == hub["head"]
    assert prefix + "part-001.json" in hub["files"]
    assert hub["files"]["part-000.json"] == original_root


def test_append_license_receipt_refuses_a_source_that_already_holds_a_row(tmp_path: Path, hub: dict):
    """A receipt binding is append-only: a source whose receipt sits in an existing
    shard is refused with a typed error instead of rewriting that shared shard and
    invalidating every concurrent writer's verify-current-head gate (coordinator
    ruling 2026-11-26, the InverseIFEval receipts break)."""
    index_path = tmp_path / "index.json"
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    seeded = json.dumps({"fixture:source": first.model_dump(mode="json")}, indent=2).encode()
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {"part-000.json": seeded},
        message="Seed",
    )
    updated = _receipt("fixture:source", "fixture/dataset", "c" * 40)
    with pytest.raises(license_binding.AppendWouldRewriteSharedShard):
        license_binding.append_license_receipt(index_path, updated, message="Update pin")
    assert hub["files"]["part-000.json"] == seeded
    assert license_binding.read_license_receipts(index_path)["fixture:source"].revision == "a" * 40


def test_append_license_receipt_never_modifies_existing_shards(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    seeded = json.dumps(
        {"fixture:source": _receipt("fixture:source", "fixture/dataset", "a" * 40).model_dump(mode="json")}, indent=2
    ).encode()
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {"part-000.json": seeded},
        message="Seed",
    )
    second = _receipt("fixture:other", "fixture/dataset", "b" * 40)
    license_binding.append_license_receipt(index_path, second, message="Append")
    assert hub["files"]["part-000.json"] == seeded
    assert set(license_binding.read_license_receipts(index_path)) == {"fixture:source", "fixture:other"}


def test_append_license_receipt_cannot_replace_an_unindexed_part(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {"part-000.json": b"{}\n", "part-001.json": b"{}\n"},
        message="Seed",
    )
    existing = {
        f"fixture:source-{number}": (
            _receipt(f"fixture:source-{number}", "fixture/dataset", "a" * 40).model_dump(mode="json")
        )
        for number in range(64)
    }
    hub["files"]["part-002.json"] = json.dumps(existing).encode()
    hub["head"] = "f" * 40
    hub["revisions"].add(hub["head"])
    prior_files, prior_index = dict(hub["files"]), index_path.read_bytes()
    with pytest.raises(ValueError, match="New reference path already exists"):
        license_binding.append_license_receipt(
            index_path, _receipt("fixture:new", "fixture/dataset", "b" * 40), message="Append"
        )
    assert hub["head"] == "f" * 40
    assert hub["files"] == prior_files
    assert index_path.read_bytes() == prior_index


def test_commit_reference_documents_rejects_changed_shards_before_upload(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(
        index_path, "open-athena/example", {"part-002.json": b'{"old": 1}\n'}, message="Seed"
    )
    hub["files"]["part-002.json"] = b'{"old": 1, "concurrent": 2}\n'
    hub["head"] = "f" * 40
    hub["revisions"].add(hub["head"])
    prior_index = index_path.read_bytes()
    with pytest.raises(ValueError, match="bytes changed"):
        reference_data.commit_reference_documents(index_path, {"part-002.json": b'{"old": 3}\n'}, message="Update")
    assert hub["head"] == "f" * 40
    assert json.loads(hub["files"]["part-002.json"]) == {"old": 1, "concurrent": 2}
    assert index_path.read_bytes() == prior_index


def test_append_license_receipt_rejects_a_mismatched_key(tmp_path: Path, hub: dict):
    index_path = tmp_path / "index.json"
    reference_data.create_reference_index(index_path, license_binding.LICENSE_RECEIPT_REPO, {}, message="Seed")
    with pytest.raises(ValueError, match="differs from the receipt source identity"):
        license_binding.append_license_receipt(
            index_path,
            _receipt("MarinSkyRL:math500", "fixture/dataset", "a" * 40),
            message="Append",
        )


def test_read_license_receipts_reads_a_staged_receipts_file(tmp_path: Path):
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    second = _receipt("fixture:other", "fixture/dataset", "b" * 40)
    staged = tmp_path / "license-receipts.json"
    staged.write_text(
        json.dumps(
            {
                "fixture:source": first.model_dump(mode="json"),
                "fixture:other": second.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    receipts = license_binding.read_license_receipts(staged)
    assert set(receipts) == {"fixture:source", "fixture:other"}
    assert receipts["fixture:other"].revision == "b" * 40


def test_license_history_preserves_distinct_pins_and_rejects_duplicate_pins(tmp_path: Path, hub: dict):
    first = _receipt("fixture:source", "fixture/dataset", "a" * 40)
    second = _receipt("fixture:source", "fixture/next-release", "b" * 40)
    index = tmp_path / "history.json"
    reference_data.create_reference_index(
        index,
        license_binding.LICENSE_RECEIPT_REPO,
        {
            "history/part-000.json": json.dumps({first.source_id: first.model_dump(mode="json")}).encode(),
            "history/part-001.json": json.dumps({second.source_id: second.model_dump(mode="json")}).encode(),
        },
        message="Archive two reviewed pins",
    )
    archived = license_binding.read_license_receipt_history(index)
    assert archived[(first.source_id, first.dataset_id, first.revision)] == first
    assert archived[(second.source_id, second.dataset_id, second.revision)] == second
    reference_data.commit_reference_documents(
        index,
        {"history/part-002.json": json.dumps({first.source_id: first.model_dump(mode="json")}).encode()},
        message="Duplicate pin",
    )
    with pytest.raises(ValueError, match="Duplicate or mismatched archived license pin"):
        license_binding.read_license_receipt_history(index)


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


def test_adopt_license_receipt_selects_a_complete_namespace(tmp_path: Path, hub: dict):
    """Ingest B's design (coordinator ruling 2026-11-26): a re-pinned receipt lands as
    a NEW complete immutable selection namespace — the selected union with only the
    owned source's row replaced — and the index atomically selects that family.
    Shared shards keep their exact bytes and the strict duplicate rejection holds."""
    index_path = tmp_path / "index.json"
    seeded = json.dumps(
        {
            "fixture:source": _receipt("fixture:source", "fixture/dataset", "a" * 40).model_dump(mode="json"),
            "fixture:other": _receipt("fixture:other", "fixture/dataset", "b" * 40).model_dump(mode="json"),
        },
        indent=2,
    ).encode()
    reference_data.create_reference_index(
        index_path,
        license_binding.LICENSE_RECEIPT_REPO,
        {"part-000.json": seeded},
        message="Seed",
    )
    adopted = _receipt("fixture:source", "fixture/dataset", "c" * 40)
    revision = license_binding.adopt_license_receipt(
        index_path, adopted, message="Adopt the bind receipt", namespace="binds-20261126-a-v1"
    )
    assert revision == hub["head"]
    assert hub["files"]["part-000.json"] == seeded  # the shared shard is untouched
    receipts = license_binding.read_license_receipts(index_path)
    assert set(receipts) == {"fixture:source", "fixture:other"}
    assert receipts["fixture:source"].revision == "c" * 40
    assert receipts["fixture:other"].revision == "b" * 40
    index = reference_data.read_reference_index(index_path)
    assert [reference.path for reference in index.references] == ["binds-20261126-a-v1/part-000.json"]
