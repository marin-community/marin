# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concurrent source publication preserves the selected index and every locator."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import CommitOperationAdd

from experiments.post_training.task_curation import registry
from experiments.post_training.task_curation.ingest_stages import registry_update
from experiments.post_training.task_curation.ingest_stages.registry_update import (
    FINGERPRINT_POINTER,
    LOCATOR_POINTER,
    OBJECT_DIRECTORY,
    RegistryPointers,
    RegistryUpdate,
    check_landing,
    publish_update,
    stage_update,
)
from experiments.post_training.task_curation.registry import RegistryReference, read_registry_document
from experiments.post_training.task_curation.source_duplicate_ci import FingerprintReference
from experiments.post_training.task_curation.source_population_cache import PopulationLocations, PreparedPopulation

REPO = "fixture/registry"


@dataclass
class RegistryArchive:
    """Versioned HF API boundary; overwrites are allowed, as on the real API."""

    cache: Path
    head: str = "0" * 40
    versions: dict[str, dict[str, bytes]] = field(default_factory=lambda: {"0" * 40: {}})
    reject_next_commit: bool = False

    def repo_info(self, *, repo_id, repo_type, revision=None):
        assert repo_id == REPO and repo_type == "dataset"
        assert (revision or self.head) in self.versions
        return SimpleNamespace(sha=self.head, private=True)

    def file_exists(self, *, repo_id, repo_type, revision, filename):
        assert repo_id == REPO and repo_type == "dataset"
        return filename in self.versions[revision]

    def create_commit(self, *, repo_id, repo_type, parent_commit, commit_message, operations):
        assert repo_id == REPO and repo_type == "dataset"
        if self.reject_next_commit:
            self.reject_next_commit = False
            raise RuntimeError("archive parent moved")
        if parent_commit != self.head:
            raise RuntimeError("archive parent moved")
        files = dict(self.versions[self.head])
        files.update({operation.path_in_repo: operation.path_or_fileobj for operation in operations})
        self.head = f"{len(self.versions):040x}"
        self.versions[self.head] = files
        return SimpleNamespace(oid=self.head)

    def download(self, *, repo_id, repo_type, revision, filename):
        assert repo_id == REPO and repo_type == "dataset"
        path = self.cache / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.versions[revision][filename])
        return str(path)

    def store(self, document: dict, name: str | None = None) -> RegistryReference:
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        sha = hashlib.sha256(payload).hexdigest()
        name = name or (OBJECT_DIRECTORY / f"{sha}.json").as_posix()
        self.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            parent_commit=self.head,
            commit_message="Seed fixture evidence",
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=payload)],
        )
        return RegistryReference(repo=REPO, revision=self.head, path=name, sha256=sha)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    archive = RegistryArchive(tmp_path / "cache")
    monkeypatch.setattr(registry, "HfApi", lambda: archive)
    monkeypatch.setattr(registry, "hf_hub_download", archive.download)
    monkeypatch.setattr(registry_update, "HfApi", lambda: archive)
    return archive


@pytest.fixture
def selected(archive, tmp_path):
    fingerprint = FingerprintReference(uri="s3://fixture/survey.json", sha256="a" * 64, input_sha256="b" * 64)
    prepared = PreparedPopulation(
        uri="s3://fixture/prepared.parquet", sha256="c" * 64, inputs_sha256="d" * 64, task_count=3
    )
    fingerprints = archive.store({"Source:first": fingerprint.model_dump(mode="json")}, "old-index.json")
    locators = archive.store(
        PopulationLocations(
            files={"e" * 64: "s3://fixture/first.parquet"}, prepared={"Source:first": prepared}
        ).model_dump(mode="json"),
        "old-locators.json",
    )
    pointers = RegistryPointers(fingerprints=fingerprints, locators=locators)
    checkout = tmp_path / "checkout"
    for path, reference in ((FINGERPRINT_POINTER, fingerprints), (LOCATOR_POINTER, locators)):
        target = checkout / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(reference.model_dump_json(indent=2) + "\n")
    return checkout, pointers, fingerprint


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def test_concurrent_updates_rebase_onto_the_selected_union_before_landing(archive, selected):
    checkout, expected, fingerprint = selected
    git(checkout, "init", "-q", "-b", "worker")
    git(checkout, "config", "user.name", "Worker")
    git(checkout, "config", "user.email", "worker@example.com")
    git(checkout, "add", ".")
    git(checkout, "commit", "-q", "-m", "Seed the registry pointers")
    base = git(checkout, "rev-parse", "HEAD")
    old_files = dict(archive.versions[archive.head])
    first = publish_update(
        expected,
        {"Source:second": fingerprint},
        PopulationLocations(files={"f" * 64: "s3://fixture/second.parquet"}),
        registry_head=archive.head,
    )
    concurrent = publish_update(
        expected, {"Source:third": fingerprint}, PopulationLocations(), registry_head=archive.head
    )
    proposal = Path("update.json")
    stage_update(checkout, first, proposal)
    git(checkout, "add", ".")
    git(checkout, "commit", "-q", "-m", "Select the first source update")
    first_commit = git(checkout, "rev-parse", "HEAD")
    check_landing(checkout, base, proposal)

    with pytest.raises(ValueError, match="registry pointers moved"):
        stage_update(checkout, concurrent, proposal)
    assert (
        RegistryReference.model_validate_json((checkout / FINGERPRINT_POINTER).read_bytes())
        == first.proposed.fingerprints
    )

    rebased = publish_update(
        first.proposed, {"Source:third": fingerprint}, PopulationLocations(), registry_head=archive.head
    )
    stage_update(checkout, rebased, proposal)
    git(checkout, "add", ".")
    git(checkout, "commit", "-q", "-m", "Preserve both concurrent source updates")
    check_landing(checkout, first_commit, proposal)

    final_index = read_registry_document(rebased.proposed.fingerprints)
    assert set(final_index) == {"Source:first", "Source:second", "Source:third"}
    final_locators = PopulationLocations.model_validate(read_registry_document(rebased.proposed.locators))
    assert set(final_locators.files.values()) == {"s3://fixture/first.parquet", "s3://fixture/second.parquet"}
    assert "Source:first" in final_locators.prepared
    assert all(archive.versions[archive.head][name] == payload for name, payload in old_files.items())


@pytest.mark.parametrize("dropped_map", ["fingerprints", "files", "prepared"])
def test_dropped_selected_entries_refuse_pointer_changes(archive, selected, dropped_map):
    checkout, expected, _ = selected
    fingerprints = read_registry_document(expected.fingerprints)
    locators = read_registry_document(expected.locators)
    if dropped_map == "fingerprints":
        fingerprints.clear()
    else:
        locators[dropped_map].clear()
    update = RegistryUpdate(
        expected=expected,
        proposed=RegistryPointers(fingerprints=archive.store(fingerprints), locators=archive.store(locators)),
    )
    original = {path: (checkout / path).read_bytes() for path in (FINGERPRINT_POINTER, LOCATOR_POINTER)}

    with pytest.raises(ValueError, match="registry update drops"):
        stage_update(checkout, update, Path("update.json"))

    assert {path: (checkout / path).read_bytes() for path in original} == original
    assert not (checkout / "update.json").exists()


def test_existing_content_address_cannot_be_overwritten_with_different_bytes(archive, selected):
    checkout, expected, _ = selected
    first = publish_update(expected, {}, PopulationLocations(), registry_head=archive.head)
    # An external archive corruption keeps the object name while replacing bytes.
    name = first.proposed.fingerprints.path
    archive.versions[archive.head][name] = b'{"corrupted": true}\n'
    before = archive.head

    with pytest.raises(ValueError, match="Registry bytes changed"):
        publish_update(expected, {}, PopulationLocations(), registry_head=archive.head)

    assert archive.head == before
    assert archive.versions[archive.head][name] == b'{"corrupted": true}\n'
    assert RegistryReference.model_validate_json((checkout / FINGERPRINT_POINTER).read_bytes()) == expected.fingerprints


def test_archive_parent_conflict_leaves_repository_pointers_unchanged(archive, selected):
    checkout, expected, fingerprint = selected
    original = {path: (checkout / path).read_bytes() for path in (FINGERPRINT_POINTER, LOCATOR_POINTER)}
    archive.reject_next_commit = True

    with pytest.raises(RuntimeError, match="archive parent moved"):
        publish_update(expected, {"Source:second": fingerprint}, PopulationLocations(), registry_head=archive.head)

    assert {path: (checkout / path).read_bytes() for path in original} == original
