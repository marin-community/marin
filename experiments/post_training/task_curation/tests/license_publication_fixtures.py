# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared staged-Hub fixtures for the licensed-publication tests."""

import json

import pytest
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt
from huggingface_hub.hf_api import RepoFile

from experiments.post_training.task_curation import license_publication
from experiments.post_training.task_curation.license_targets import license_evidence
from infra.marina.apps.rl_data_catalog.server import licensing

DATASET = "fixture/upstream"
REVISION = "a" * 40
PREFIX = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"


def source_receipt(card):
    url = PREFIX + "README.md"
    return LicenseReceipt(
        source_id="fixture:source",
        dataset_id=DATASET,
        revision=REVISION,
        resolved_at="2026-11-15T00:00:00+00:00",
        resolver_version=LICENSE_RESOLVER_VERSION,
        evidence=[
            license_evidence(url, card, "dataset_card"),
            license_evidence(url, card, "readme_license"),
            license_evidence(url, card, "readme_notice"),
        ],
        bundled_third_party=False,
        model_generated_outputs=False,
    )


def snapshot_hub(tmp_path, monkeypatch, files):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, content in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    class SnapshotHub:
        def list_repo_tree(self, repo, *, repo_type, revision, recursive):
            # Folder-lineage upstreams keep license documents in subfolders, so the
            # publication listing is recursive and the fixture names are flat paths.
            assert (repo, repo_type, revision, recursive) == (DATASET, "dataset", REVISION, True)
            return [RepoFile(path=name, size=len(data), oid="b" * 40) for name, data in files.items()]

    def download(repo, name, *, repo_type, revision):
        assert (repo, repo_type, revision) == (DATASET, "dataset", REVISION)
        return str(snapshot / name)

    monkeypatch.setattr(license_publication, "HfApi", SnapshotHub)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    return snapshot


def package_receipt(wheel: str, wheel_sha: str, metadata: dict) -> LicenseReceipt:
    url = "https://pypi.org/pypi/reasoning_gym/0.1.25/json"
    content = json.dumps(metadata).encode()
    return LicenseReceipt(
        source_id="Reasoning Gym:reasoning_gym",
        dataset_id=wheel,
        revision=wheel_sha,
        resolved_at="2026-11-15T00:00:00+00:00",
        resolver_version=LICENSE_RESOLVER_VERSION,
        evidence=[license_evidence(url, content, "package_metadata")],
        bundled_third_party=False,
        model_generated_outputs=False,
    )


@pytest.fixture(autouse=True)
def isolated_rulings(monkeypatch):
    """Unit tests rule against staged empty rulings; the pinned registry loads only
    where data-integration runs resolve the committed reference."""
    monkeypatch.setattr(
        licensing,
        "read_registry_json",
        lambda _reference: {"version": "test", "bans": []},
    )
    licensing.license_bans.cache_clear()
    yield
    getattr(licensing.license_bans, "cache_clear", lambda: None)()
