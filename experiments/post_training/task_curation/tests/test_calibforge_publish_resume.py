# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The publication helper resumes partial runs and refuses foreign content pins."""

import pathlib
import tempfile

import pytest

from experiments.post_training.task_curation.calibforge_publish import _source_pin_line, publish

PLAN_PIN = (
    "Source pin: `open-athena/campaign-release-tasktrove-competitive-coding-rebuilt-v"
    "-6ce7836f-retained-2026.11.18.1-r1@467bb8d4`."
)


class FakeApi:
    """Records create_repo calls, serves a configurable existing card, fakes the upload."""

    def __init__(self, existing_card: str | None):
        self.existing_card = existing_card
        self.calls = []
        self.uploaded = {}

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool):
        self.calls.append(exist_ok)
        return None

    def repo_info(self, repo_id: str, repo_type: str = "dataset"):
        return type("Info", (), {"private": True})()

    def hf_hub_download(self, repo_id: str, filename: str, repo_type: str, revision=None):
        if self.existing_card is not None and filename == "README.md" and filename not in self.uploaded:
            return str(self.existing_card)
        return str(self.uploaded[filename])

    def download(self, filename: str):
        if self.existing_card is not None and filename == "README.md" and filename not in self.uploaded:
            return str(self.existing_card)
        return str(self.uploaded[filename])

    def upload_folder(self, repo_id: str, repo_type: str, folder_path, commit_message: str):
        for item in pathlib.Path(folder_path).iterdir():
            target = pathlib.Path(tempfile.mkdtemp(prefix="atlas-fake-upload-")) / item.name
            target.write_bytes(item.read_bytes())
            self.uploaded[item.name] = target
        return type("Commit", (), {"oid": "fake" * 10})()


@pytest.fixture
def cards(tmp_path):
    tasks = tmp_path / "tasks.parquet"
    tasks.write_bytes(b"parquet")
    new_card = tmp_path / "new-card.md"
    new_card.write_text(f"# Atlas retention release\n\n{PLAN_PIN}\n\nBody.\n")
    return tasks, new_card, tmp_path


def test_source_pin_line_extracts_the_declared_pin():
    assert _source_pin_line(f"header\n\n{PLAN_PIN}\n\nbody") == PLAN_PIN.strip()
    assert _source_pin_line("no pin here") is None


def test_publish_resumes_into_a_repo_left_by_a_partial_run(cards, monkeypatch):
    tasks, new_card, tmp_path = cards
    existing = tmp_path / "existing-card.md"
    existing.write_text(f"# Atlas retention release\n\n{PLAN_PIN}\n")
    fake = FakeApi(existing)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.calibforge_publish.hf_hub_download",
        lambda repo_id, filename, repo_type, revision=None: fake.download(filename),
    )
    publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert fake.calls == [True], "the partial run's repo must be resumed with exist_ok=True"


def test_publish_refuses_a_repo_holding_another_content_pin(cards, monkeypatch):
    tasks, new_card, tmp_path = cards
    existing = tmp_path / "existing-card.md"
    existing.write_text("Source pin: `open-athena/some-other-dataset@1234567890123456789012345678901234567890`.\n")
    fake = FakeApi(existing)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.calibforge_publish.hf_hub_download",
        lambda repo_id, filename, repo_type, revision=None: fake.download(filename),
    )
    with pytest.raises(ValueError, match="holds another content pin"):
        publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert fake.calls == [], "a foreign pin must refuse before any create_repo"


def test_publish_creates_a_fresh_repo_without_exist_ok(cards, monkeypatch):
    tasks, new_card, _ = cards
    fake = FakeApi(None)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.calibforge_publish.hf_hub_download",
        lambda repo_id, filename, repo_type, revision=None: fake.download(filename),
    )
    publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert fake.calls == [False], "a fresh repo must create with exist_ok=False"
