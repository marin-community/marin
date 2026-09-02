# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from marin.transform.stack_edu import hydrate
from marin.transform.stack_edu.hydrate import StackEduHydrationConfig, stack_edu_record_id


class _FakeDataset:
    @classmethod
    def from_list(cls, tasks):
        assert tasks
        return cls()

    def map(self, _fn):
        return self

    def write_jsonl(self, _path, *, skip_existing):
        assert skip_existing
        return self


class _FakeZephyrContext:
    def __init__(self, **_kwargs):
        pass

    def execute(self, _pipeline):
        return SimpleNamespace(results=["metrics.jsonl"])


def test_stack_edu_record_id_matches_historical_hydration_contract():
    row = {
        "blob_id": "abc123",
        "repo_name": "owner/repo",
        "path": "/src/main.py",
    }

    assert stack_edu_record_id("Python", row) == "7bcbb583fdbd86d3765d4d039e518292603ca78d"


def test_hydrate_stack_edu_returns_structured_artifact_result(monkeypatch):
    monkeypatch.setattr(hydrate, "_build_hydration_tasks", lambda _cfg: [object()])
    monkeypatch.setattr(hydrate, "Dataset", _FakeDataset)
    monkeypatch.setattr(hydrate, "ZephyrContext", _FakeZephyrContext)
    monkeypatch.setattr(
        hydrate,
        "load_jsonl",
        lambda _path: [
            {
                "count": 17,
                "decoded_fallback": 2,
                "missing_blob": 3,
                "corrupt_blob": 5,
                "empty_blob": 7,
                "fetch_error": 11,
                "skipped": False,
                "missing_blob_examples": [],
                "corrupt_blob_examples": [],
                "empty_blob_examples": [],
                "fetch_error_examples": [],
            }
        ],
    )

    result = hydrate.hydrate_stack_edu(
        StackEduHydrationConfig(
            input_path="memory://input",
            output_path="memory://output",
            language="Python",
        )
    )

    assert result == {
        "output_path": "memory://output",
        "language": "Python",
        "tasks": 1,
        "skipped_tasks": 0,
        "rows_written": 17,
        "decoded_fallback": 2,
        "missing_blob": 3,
        "corrupt_blob": 5,
        "empty_blob": 7,
        "fetch_error": 11,
    }
