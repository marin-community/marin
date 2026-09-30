# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import and direct-answer grading for the two TaskTrove Clean calendar sources."""

import json
import subprocess
import sys

import pytest

from taskcompendium.importers.tasktrove import calendar_adapter
from taskcompendium.importers.tasktrove.calendar import (
    _FINAL_ANSWER_INSTRUCTION,
    _OUTPUT_INSTRUCTION,
    CHECKER,
    DATA,
    SOURCES,
    import_task,
)
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
CHECKER_EXITS_AFTER_WRITING_REWARD = b"""\
import json
import os
from pathlib import Path

Path(os.environ["TASKTROVE_LOGS_DIR"], "reward.json").write_text(json.dumps({"reward": 1}))
raise SystemExit(7)
"""
CHECKER_WRITES_OUT_OF_RANGE_REWARD = b"""\
import json
import os
from pathlib import Path

Path(os.environ["TASKTROVE_LOGS_DIR"], "reward.json").write_text(json.dumps({"reward": 1.5}))
"""
CHECKER_WRITES_NONFINITE_REWARD = b"""\
import json
import os
from pathlib import Path

Path(os.environ["TASKTROVE_LOGS_DIR"], "reward.json").write_text(json.dumps({"reward": float("nan")}))
"""
CHECKER_TIMES_OUT = b"""\
import time

time.sleep(10)
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
            "instruction.md": (
                (_OUTPUT_INSTRUCTION + "Schedule the named event inside its allowed time window.").encode()
            ),
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
    assert specification.context.events[0].content.startswith(_FINAL_ANSWER_INSTRUCTION)
    assert "/app/answer.txt" not in specification.context.events[0].content
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


@pytest.mark.parametrize(
    ("checker", "timeout", "expected_error"),
    [
        (
            CHECKER_EXITS_AFTER_WRITING_REWARD,
            5,
            "exited with status 7",
        ),
        (
            CHECKER_WRITES_OUT_OF_RANGE_REWARD,
            5,
            "reward is outside [0, 1]",
        ),
        (
            CHECKER_WRITES_NONFINITE_REWARD,
            5,
            "reward is outside [0, 1]",
        ),
        (CHECKER_TIMES_OUT, 0.1, "timed out"),
    ],
)
def test_calendar_checker_failures_are_infra_errors(tmp_path, checker, timeout, expected_error):
    archive = _archive()
    archive.files[f"tests/{CHECKER}"] = checker
    specification = import_task(archive, runtime_image=RUNTIME_IMAGE)
    verifier = ScriptVerifier.model_validate_json(specification.verifier.parameters_json)
    tests = materialize_private_resources(verifier.resources, tmp_path / "tests")
    workspace = tmp_path / "app"
    workspace.mkdir()
    private_verifier = tmp_path / "verifier"
    private_verifier.mkdir()
    (private_verifier / "submission.json").write_text(
        json.dumps({"protocol_version": 1, "answer_type": "text", "convention_id": "plain", "answer": "{}"})
    )

    calendar_adapter.main(tests[verifier.entrypoint].parent, workspace, private_verifier, timeout)

    result = json.loads((private_verifier / "result.json").read_text())
    assert result == {"status": "infra_error", "error": f"Source checker {expected_error}"}


def test_calendar_checker_logs_are_private_and_bounded(tmp_path):
    archive = _archive()
    archive.files[
        f"tests/{CHECKER}"
    ] = b"""\
print("x" * 100000)
print("y" * 100000, file=__import__("sys").stderr)
"""
    specification = import_task(archive, runtime_image=RUNTIME_IMAGE)
    verifier = ScriptVerifier.model_validate_json(specification.verifier.parameters_json)
    tests = materialize_private_resources(verifier.resources, tmp_path / "tests")
    workspace = tmp_path / "app"
    workspace.mkdir()
    private_verifier = tmp_path / "verifier"
    private_verifier.mkdir()
    (private_verifier / "submission.json").write_text(
        json.dumps({"protocol_version": 1, "answer_type": "text", "convention_id": "plain", "answer": "{}"})
    )

    calendar_adapter.main(tests[verifier.entrypoint].parent, workspace, private_verifier, 5)

    logs = private_verifier / "source_logs"
    assert (logs / "stdout.log").stat().st_size == calendar_adapter.MAX_LOG_BYTES
    assert (logs / "stderr.log").stat().st_size == calendar_adapter.MAX_LOG_BYTES
    result = json.loads((private_verifier / "result.json").read_text())
    assert result["status"] == "infra_error"
    assert result["error"].startswith("Source checker did not write a valid reward:")


def test_calendar_import_rejects_a_different_source_checker():
    archive = _archive()
    archive.files["tests/verifier.toml"] = b'mode = "script"\npath = "other.py"\n'

    with pytest.raises(ValueError, match="Unsupported TaskTrove calendar script contract"):
        import_task(archive, runtime_image=RUNTIME_IMAGE)


def test_calendar_import_rejects_an_unrecognized_output_instruction():
    archive = _archive()
    archive.files["instruction.md"] = b"Please save your response to /app/answer.txt."

    with pytest.raises(ValueError, match="Unsupported TaskTrove calendar instruction shape"):
        import_task(archive, runtime_image=RUNTIME_IMAGE)
