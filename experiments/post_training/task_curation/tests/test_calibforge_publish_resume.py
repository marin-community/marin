# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The publication helper resumes partial runs and refuses foreign content pins."""

import hashlib
import json
import pathlib
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

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
    tasks = _stamped_tasks(tmp_path / "tasks.parquet")
    new_card = tmp_path / "new-card.md"
    new_card.write_text(f"# Atlas retention release\n\n{PLAN_PIN}\n\nBody.\n")
    return tasks, new_card, tmp_path


def _stamped_tasks(path: pathlib.Path, *, rows: int = 2) -> pathlib.Path:
    """A minimal tasks parquet carrying the loader contract's row-level hashes."""
    specs = [f'{{"id": "task-{index}", "prompt": "solve"}}' for index in range(rows)]
    table = pa.table(
        {
            "task_spec": pa.array(specs, pa.large_string()),
            "task_sha256": pa.array([hashlib.sha256(spec.encode()).hexdigest() for spec in specs], pa.large_string()),
        }
    )
    pq.write_table(table, path)
    return path


def test_source_pin_line_extracts_the_declared_pin():
    assert _source_pin_line(f"header\n\n{PLAN_PIN}\n\nbody") == PLAN_PIN.strip()
    assert _source_pin_line("no pin here") is None


def test_publish_refuses_tasks_without_the_loader_hash_column(cards, monkeypatch):
    tasks, new_card, _ = cards
    pq.write_table(pa.table({"task_spec": pa.array(['{"id": "task-0"}'], pa.large_string())}), tasks)
    fake = FakeApi(None)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    with pytest.raises(ValueError, match="lacks the row-level task_sha256 column"):
        publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert fake.calls == [], "the publication must refuse before any upload"


def test_publish_refuses_tasks_whose_hash_differs(cards, monkeypatch):
    tasks, new_card, _ = cards
    table = pq.read_table(tasks)
    table = table.set_column(
        table.column_names.index("task_sha256"),
        "task_sha256",
        pa.array(["0" * 64] * table.num_rows, pa.large_string()),
    )
    pq.write_table(table, tasks)
    fake = FakeApi(None)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    with pytest.raises(ValueError, match="differs from its task_spec content"):
        publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert fake.calls == [], "the publication must refuse before any upload"


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


def _lowered_tasks(path: pathlib.Path) -> pathlib.Path:
    """A lowered-shape tasks parquet: rollout envelopes with runtime machines."""
    spec = TaskSpec(
        id="d01.a/01",
        context=ConversationInput(events=(TextMessage(role="user", content="Solve it."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.SHELL_SIMULATOR, working_directory="/app"
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        source=Source(dataset="capability", revision="r", row="d01.a/01", importer_revision="1"),
    )
    envelope = json.dumps({"task": spec.model_dump(mode="json"), "runtime": {"task_machine": {"memory_mb": 8192}}})
    pq.write_table(pa.table({"path": ["d01.a/01"], "lowered_task_spec": [envelope]}), path)
    return path


def test_publish_checks_each_lowered_row_the_loader_would_refuse(cards, monkeypatch):
    tasks, new_card, _ = cards
    _lowered_tasks(tasks)
    fake = FakeApi(None)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.calibforge_publish.hf_hub_download",
        lambda repo_id, filename, repo_type, revision=None: fake.download(filename),
    )
    publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert set(fake.uploaded) == {"tasks.parquet", "README.md"}


def test_publish_refuses_a_lowered_row_without_runtime_machines(cards, monkeypatch):
    tasks, new_card, _ = cards
    _lowered_tasks(tasks)
    table = pq.read_table(tasks).to_pylist()
    table[0]["lowered_task_spec"] = json.dumps({"task": {"id": "d01.a/01"}})
    pq.write_table(pa.table({"path": ["d01.a/01"], "lowered_task_spec": [table[0]["lowered_task_spec"]]}), tasks)
    fake = FakeApi(None)
    monkeypatch.setattr("experiments.post_training.task_curation.calibforge_publish.HfApi", lambda: fake)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.calibforge_publish.hf_hub_download",
        lambda repo_id, filename, repo_type, revision=None: fake.download(filename),
    )
    with pytest.raises(ValueError, match="envelope carries no runtime machines"):
        publish(tasks, new_card, "open-athena/retained-repo", data_restriction="open", evidence_files=())
    assert fake.calls == [], "the publication must refuse before any upload"
