# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import validate_repo_id

from experiments.post_training.task_curation import license_publication, raw_source_publish
from experiments.post_training.task_curation.license_resolution import license_evidence


@pytest.mark.parametrize("mutation", [None, "incomplete_inventory", "changed_source_bytes"])
@pytest.mark.parametrize("source_id", ["fixture:raw-source", "PrimeIntellect:PrimeIntellect/Multi-SWE-RL-Verified"])
def test_raw_archive_preserves_complete_source_and_rejects_mutated_staging_before_upload(
    tmp_path, monkeypatch, mutation, source_id
):
    dataset, revision = "fixture/raw-source", "a" * 40
    upstream = tmp_path / "upstream"
    staged = tmp_path / "staged"
    published = tmp_path / "published"
    upstream.mkdir()
    staged.mkdir()
    original = {"README.md": b"---\nlicense: mit\n---\nOriginal attribution.\n", "data.parquet": b"original raw payload"}
    for name, content in original.items():
        (upstream / name).write_bytes(content)
        (staged / name).write_bytes(content)
    snapshot = {
        "source_id": source_id,
        "dataset_id": dataset,
        "revision": revision,
        "data_restriction": DataRestriction.OPEN.value,
        "artifact": str(staged),
        "source_files": list(original),
        "files": [
            {"path": name, "uri": str(staged / name), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in original.items()
        ],
    }
    if mutation == "incomplete_inventory":
        snapshot["source_files"].remove("data.parquet")
    if mutation == "changed_source_bytes":
        (staged / "data.parquet").write_bytes(b"changed raw payload")
    receipt = LicenseReceipt(
        source_id=source_id,
        dataset_id=dataset,
        revision=revision,
        resolved_at="2026-11-15T00:00:00+00:00",
        resolver_version=LICENSE_RESOLVER_VERSION,
        evidence=[
            license_evidence(
                f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/README.md",
                original["README.md"],
                "dataset_card",
            )
        ],
        bundled_third_party=False,
        model_generated_outputs=False,
    )
    created = []

    class SnapshotHub:
        def dataset_info(self, repo, *, revision):
            assert (repo, revision) == (dataset, "a" * 40)
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=name) for name in original])

        def list_repo_tree(self, repo, *, revision, **kwargs):
            assert (repo, revision) == (dataset, "a" * 40)
            return [RepoFile(path=name, size=len(content), oid="c" * 40) for name, content in original.items()]

        def create_repo(self, repo, *, repo_type, private, exist_ok):
            validate_repo_id(repo)
            assert (repo_type, private, exist_ok) == ("dataset", True, False)
            created.append(repo)

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True)

        def upload_folder(self, *, folder_path, allow_patterns, **kwargs):
            for name in allow_patterns:
                destination = published / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(folder_path) / name, destination)
            return SimpleNamespace(oid="b" * 40)

    def download(repo, name, *, revision, **kwargs):
        if repo == dataset:
            assert revision == "a" * 40
            return str(upstream / name)
        assert created == [repo] and revision == "b" * 40
        return str(published / name)

    monkeypatch.setattr(raw_source_publish, "HfApi", SnapshotHub)
    monkeypatch.setattr(license_publication, "HfApi", SnapshotHub)
    monkeypatch.setattr(raw_source_publish, "hf_hub_download", download)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    monkeypatch.setattr(
        license_publication,
        "resolve_license",
        lambda target: receipt.model_copy(update={"dataset_id": target.dataset_id, "revision": target.revision}),
    )
    if mutation is not None:
        with pytest.raises(ValueError, match=r"inventory differs|source bytes changed"):
            raw_source_publish.publish_raw_source(
                snapshot, receipt, version="fixture", evidence_root=str(tmp_path / "evidence"), use="train"
            )
        assert created == []
        return
    archive = raw_source_publish.publish_raw_source(
        snapshot, receipt, version="fixture", evidence_root=str(tmp_path / "evidence"), use="train"
    )
    assert archive.verified_raw_files == 2 and archive.license_check["verdict"] == "PASS"
    for name, content in original.items():
        assert (published / "raw" / name).read_bytes() == content
    manifest = json.loads((published / "raw-snapshot.json").read_text())
    assert (manifest["dataset_id"], manifest["revision"]) == (dataset, revision)
