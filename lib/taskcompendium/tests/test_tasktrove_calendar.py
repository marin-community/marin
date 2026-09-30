# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import and direct-answer grading for the two TaskTrove Clean calendar sources."""

import json
import subprocess
import sys

import pytest

from taskcompendium.importers.tasktrove.calendar import CHECKER, DATA, SOURCES, import_task
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, TextMessage, VerifierKind
from taskcompendium.verifiers.script import ScriptVerifier, materialize_private_resources

RUNTIME_IMAGE = "python:3.11-slim@sha256:" + "a" * 64
TAGS = ["tool-use", "calendar", "scheduling", "state-tracking", "nemotron"]
EXPECTED = {
    "expected_events": {
        "0": {
            "event_name": "Project sync",
            "duration": 30,
            "min_time": "09:00",
            "max_time": "12:00",
        }
    }
}
SOURCE_CHECKER = b"""\
import json
import os
from pathlib import Path

expected = json.loads((Path(os.environ["TASKTROVE_TESTS_DIR"]) / "expected_events.json").read_text())
answer = json.loads((Path(os.environ["TASKTROVE_WORKSPACE"]) / "answer.txt").read_text())
logs = Path(os.environ["TASKTROVE_LOGS_DIR"])
(logs / "reward.json").write_text(json.dumps({"reward": float(answer == expected["expected_events"])}))
"""


def _archive(source: str = "laion__nemotron-gym-agent-calendar-v2") -> TaskArchive:
    family, converter = SOURCES[source]
    manifest = f"""[metadata]
tasktrove_source = "{source}"
tasktrove_path = "calendar-fixture.tar.gz"
family = "{family}"
converter = "{converter}"
mode = "script"
tags = {json.dumps(TAGS)}
"""
    return TaskArchive(
        upstream_subset=source,
        archive_path="calendar-fixture.tar.gz",
        release_uri="https://huggingface.co/datasets/open-athena/task-trove",
        release_revision="9065fa568394f286dab0081e43dc76fc87c48984",
        files={
            "task.toml": manifest.encode(),
            "instruction.md": b"Schedule the named event inside its allowed time window.",
            "tests/verifier.toml": b'mode = "script"\npath = "agent_calendar_checker.py"\ntimeout = 5\n',
            f"tests/{CHECKER}": SOURCE_CHECKER,
            f"tests/{DATA}": json.dumps(EXPECTED).encode(),
        },
    )


@pytest.mark.parametrize("source", tuple(SOURCES))
def test_calendar_import_keeps_tags_and_expected_schedule_private(source, tmp_path):
    specification = import_task(_archive(source), runtime_image=RUNTIME_IMAGE)
    verifier = ScriptVerifier.model_validate_json(specification.verifier.parameters_json)
    resources = materialize_private_resources(verifier.resources, tmp_path / source.replace("/", "_"))

    assert specification.answer_type is AnswerType.TEXT
    assert specification.verifier.kind is VerifierKind.SCRIPT
    assert specification.tags == tuple(TAGS)
    assert specification.source.row == f"{source}:calendar-fixture.tar.gz"
    assert isinstance(specification.context.events[0], TextMessage)
    assert "Project sync" not in specification.context.events[0].content
    assert json.loads(resources[DATA].read_text()) == EXPECTED


def test_calendar_imported_script_scores_a_correct_schedule_and_rejects_a_wrong_one(tmp_path):
    specification = import_task(_archive(), runtime_image=RUNTIME_IMAGE)
    verifier = ScriptVerifier.model_validate_json(specification.verifier.parameters_json)
    private_tests = tmp_path / "tests"
    tests = materialize_private_resources(verifier.resources, private_tests)

    grades = []
    for answer in (EXPECTED["expected_events"], {"0": {"event_name": "Project sync", "duration": 45}}):
        workspace = tmp_path / f"app-{len(grades)}"
        workspace.mkdir()
        private_verifier = tmp_path / f"verifier-{len(grades)}"
        private_verifier.mkdir()
        (private_verifier / "submission.json").write_text(
            json.dumps(
                {
                    "protocol_version": 1,
                    "answer_type": "text",
                    "convention_id": "plain",
                    "answer": json.dumps(answer),
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(tests[verifier.entrypoint]),
                str(private_tests),
                str(workspace),
                str(private_verifier),
                "5",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        grades.append(json.loads((private_verifier / "result.json").read_text()))

    assert grades == [{"status": "scored", "reward": 1.0}, {"status": "scored", "reward": 0.0}]


def test_calendar_import_rejects_a_different_source_checker():
    archive = _archive()
    archive.files["tests/verifier.toml"] = b'mode = "script"\npath = "other.py"\n'

    with pytest.raises(ValueError, match="Unsupported TaskTrove calendar script contract"):
        import_task(archive, runtime_image=RUNTIME_IMAGE)
