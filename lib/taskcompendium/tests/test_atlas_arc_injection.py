# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC and injection conversion retains original private grader files."""

import base64
import hashlib
import json

from taskcompendium.datasets import atlas_arc_injection
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.models import CheckStatus, RawRow
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import PlainText


def source_row(name: str, verifier_data: dict, files: dict[str, bytes]) -> RawRow:
    return RawRow(
        name,
        Source(dataset="open-thoughts/TaskTrove", revision="fixture", row=name, importer_revision="test"),
        {
            "instruction": "Return the requested answer in /app/answer.txt.",
            "verifier_data": verifier_data,
            "files": {path: base64.b64encode(content).decode() for path, content in files.items()},
            "file_metadata": {
                path: {"mode": "0755" if path.endswith(".sh") else "0644", "mtime_ns": 1} for path in files
            },
            "archive_sha256": "fixture-archive",
            "path": f"fixture/{name}",
            "archive_links": {},
        },
    )


def grade(task: TaskSpec, answer: str):
    return grade_task(
        task,
        PlainText(id="plain"),
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
    )


def test_arc_transductive_preserves_source_files_without_inventing_a_comparator():
    files = {
        "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py\n",
        "tests/verifier.py": b"# original source verifier bytes\n",
        "tests/verifier_data.json": b'{"expected_output": [[1, 2], [3, 4]]}',
    }
    task = atlas_arc_injection.normalize(
        source_row("arc_transductive", {"expected_output": [[1, 2], [3, 4]]}, files), "arc_transductive"
    )
    assert isinstance(task, TaskSpec)
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert all(private[path.removeprefix("tests/")] == content for path, content in files.items())
    assert (
        grader_config(task)["contract"]["source_grader_sha256"] == hashlib.sha256(files["tests/verifier.py"]).hexdigest()
    )
    assert grade(task, "1 2\n3 4").status == Outcome.UNAVAILABLE


def test_indirect_injection_preserves_private_target_and_source_grader():
    data = {
        "target_tool": "send_message",
        "target_args": {"recipient": "external@example.org"},
        "available_tools": ["send_message"],
    }
    files = {
        "tests/verifier.py": b"# original injection verifier bytes\n",
        "tests/verifier_data.json": json.dumps(data).encode(),
    }
    task = atlas_arc_injection.normalize(source_row("indirect_injection", data, files), "indirect_injection")
    assert isinstance(task, TaskSpec)
    assert "external@example.org" not in task.context.events[0].content
    assert not task.resources.worker and not task.resources.all
    assert {resource.path: resource_bytes(resource) for resource in task.resources.verifier}["verifier.py"] == files[
        "tests/verifier.py"
    ]
    assert [(check.check, check.status) for check in atlas_arc_injection.verification_report(task).checks] == [
        ("source_grader_runtime", CheckStatus.UNSUPPORTED)
    ]


def test_arc_inductive_without_runtime_does_not_invent_a_grade():
    row = source_row(
        "arc_inductive",
        {"test_cases": [{"input": [[1, 2]], "output": [[2, 1]]}]},
        {"tests/verifier.py": b"# original inductive verifier bytes\n"},
    )
    task = atlas_arc_injection.normalize(row, "arc_inductive")
    assert isinstance(task, TaskSpec)
    assert task.output_paths == ("/app/solution.py", "/app/answer.txt")
    result = grade(task, "def transform(grid): return [row[::-1] for row in grid]")
    assert (result.status, result.reward) == (Outcome.UNAVAILABLE, None)
