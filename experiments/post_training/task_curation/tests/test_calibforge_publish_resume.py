# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cohort publish helper refuses rows the cohort loader would reject, before any upload."""

import hashlib
import json
import pathlib
from types import SimpleNamespace

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

from experiments.post_training.task_curation.calibforge_publish import publish

PUBLISH_KWARGS = {
    "data_restriction": "open",
    "source_id": "fixture:source",
    "parent_dataset_id": "fixture/upstream",
    "parent_revision": "a" * 40,
    "use": "train",
    "pretty_name": "Fixture retention release",
}


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


@pytest.fixture
def cards(tmp_path):
    tasks = _stamped_tasks(tmp_path / "tasks.parquet")
    new_card = tmp_path / "new-card.md"
    new_card.write_text("# Atlas retention release\n\nBody.\n")
    return tasks, new_card


def test_publish_refuses_tasks_without_the_loader_hash_column(cards):
    tasks, new_card = cards
    pq.write_table(pa.table({"task_spec": pa.array(['{"id": "task-0"}'], pa.large_string())}), tasks)
    with pytest.raises(ValueError, match="lacks the row-level task_sha256 column"):
        publish(tasks, new_card, "open-athena/retained-repo", evidence_files=(), **PUBLISH_KWARGS)


def test_publish_refuses_tasks_whose_hash_differs(cards):
    tasks, new_card = cards
    table = pq.read_table(tasks)
    table = table.set_column(
        table.column_names.index("task_sha256"),
        "task_sha256",
        pa.array(["0" * 64] * table.num_rows, pa.large_string()),
    )
    pq.write_table(table, tasks)
    with pytest.raises(ValueError, match="differs from its task_spec content"):
        publish(tasks, new_card, "open-athena/retained-repo", evidence_files=(), **PUBLISH_KWARGS)


def test_publish_checks_each_lowered_row_the_loader_would_refuse(cards, monkeypatch):
    tasks, new_card = cards
    _lowered_tasks(tasks)
    calls = []

    calls = []

    def fake_publication(source_id, parent_dataset, parent_revision, repo, folder, files, **kwargs):
        calls.append((repo, tuple(sorted(files)), kwargs["body"]))
        return SimpleNamespace(revision="f" * 40)

    monkeypatch.setattr(
        "experiments.post_training.task_curation.calibforge_publish.publish_licensed_population",
        fake_publication,
    )
    publish(tasks, new_card, "open-athena/retained-repo", evidence_files=(), **PUBLISH_KWARGS)
    assert calls == [
        (
            "open-athena/retained-repo",
            ("tasks.parquet",),
            new_card.read_text().strip(),
        )
    ]


def test_publish_refuses_a_lowered_row_without_runtime_machines(cards):
    tasks, new_card = cards
    _lowered_tasks(tasks)
    table = pq.read_table(tasks).to_pylist()
    table[0]["lowered_task_spec"] = json.dumps({"task": {"id": "d01.a/01"}})
    pq.write_table(pa.table({"path": ["d01.a/01"], "lowered_task_spec": [table[0]["lowered_task_spec"]]}), tasks)
    with pytest.raises(ValueError, match="envelope carries no runtime machines"):
        publish(tasks, new_card, "open-athena/retained-repo", evidence_files=(), **PUBLISH_KWARGS)
