# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from typing import Any

import pytest
from taskcompendium.models import CommandSemantics, TextMessage
from taskcompendium.pipeline.models import Reply, WorkspaceFiles

from experiments.post_training.task_curation.datasets.taskforge import tasks
from experiments.post_training.task_curation.datasets.taskforge.release import Band, accepted_item, ledger_rows
from experiments.post_training.task_curation.tests.conversion import converted_task

PYTHON_IMAGE = "docker.io/library/python@sha256:" + "a" * 64
SOURCE = tasks.sources()[0].config
assert SOURCE is not None


def _environment(image: str | None, capabilities: tuple[str, ...]) -> dict[str, Any]:
    return {
        "capabilities": list(capabilities),
        "compatible_backends": [],
        "docker_image": image,
        "working_directory": "/workspace" if capabilities else None,
        "setup_commands": [],
        "environment_variables": {},
        "tool_providers": {},
        "packages_lock": None,
    }


def _inline(path: str, content: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(content).decode()
    return {"path": path, "source": {"kind": "inline_file", "content_base64": encoded}, "mode": None, "mtime_ns": None}


def _row(*, image: str | None, reference: dict[str, Any], output_directories: list | None = None) -> dict[str, Any]:
    task = {
        "id": "d01.algebra.exact-linear-systems--4",
        "schema_version": "0.25",
        "context": {"events": [{"type": "message", "role": "user", "content": "Solve the system."}]},
        "environment_requirements": _environment(image, ("shell", "filesystem")),
        "final_tools": [],
        "interaction_tools": [],
        "output_paths": [],
        "output_directories": output_directories or [],
        "answer_type": "text",
        "answer_format": {"kind": "plain_text"},
        "grader": {
            "kind": "script",
            "argv": ["python3", "/tests/grade.py"],
            "cwd": "/app",
            "env": {},
            "environment": _environment(PYTHON_IMAGE, ()),
            "collect": [],
            "artifacts": [],
            "answer_path": "/app/answer.txt",
            "conversation_path": "/tests/conversation.json",
            "reward": {
                "kind": "file",
                "files": [{"path": "/logs/verifier/reward.json", "format": "json", "key": "reward"}],
                "pass_above": None,
            },
            "timeout": 300.0,
        },
        "source": {"dataset": "taskforge", "revision": "0" * 64, "row": "d01/4", "importer_revision": "builder/8"},
        "resources": {"all": [], "worker": [], "oracle": [], "verifier": [_inline("grade.py", b"print(1)\n")]},
        "tags": ["reasoning"],
    }
    controls = [
        {"id": "c-plus-reference", "kind": "positive", "category": "known_correct", "concern": "reference"}
        | {"payload": reference},
        {"id": "c-empty", "kind": "negative", "category": "empty", "concern": "acceptance", "payload": {"turns": []}},
    ]
    return {
        "id": "taskforge:d01.algebra.exact-linear-systems:testgen20261013/4",
        "capability": "d01.algebra.exact-linear-systems",
        "subject": "d01",
        "band": "calibrated",
        "task": task,
        "controls": controls,
    }


def test_image_task_upgrades_to_linux_process_and_keeps_its_reference_reply():
    reply = {"turns": [{"type": "message", "role": "assistant", "content": "D: 207/100"}]}
    task = converted_task(SOURCE, _row(image=PYTHON_IMAGE, reference=reply))

    assert task.environment_requirements.command_semantics == CommandSemantics.LINUX_PROCESS
    assert task.grader.environment.command_semantics == CommandSemantics.LINUX_PROCESS
    assert "taskforge:d01.algebra.exact-linear-systems:testgen20261013/4" in task.tags
    assert "capability:d01.algebra.exact-linear-systems" in task.tags
    assert tasks.taskforge_golden(task) == Reply(TextMessage(role="assistant", content="D: 207/100"))


def test_imageless_shell_task_runs_in_the_simulator_and_submits_reference_files():
    files = {"files": [_inline("workspace/report.txt", b"verdict: valid\n")]}
    task = converted_task(SOURCE, _row(image=None, reference=files))

    assert task.environment_requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR
    assert tasks.taskforge_golden(task) == WorkspaceFiles({"/workspace/report.txt": b"verdict: valid\n"})


def test_shell_call_reference_falls_back_to_the_empty_control():
    calls = {
        "turns": [
            {
                "type": "assistant_tool_calls",
                "calls": [{"call_id": "1", "name": "shell", "arguments": {"command": "ls"}}],
            },
            {"type": "message", "role": "assistant", "content": "done"},
        ]
    }
    task = converted_task(SOURCE, _row(image=PYTHON_IMAGE, reference=calls))

    assert tasks.taskforge_golden(task) is None


def test_output_directories_are_rejected_rather_than_dropped():
    row = _row(image=PYTHON_IMAGE, reference={"turns": []}, output_directories=[{"root": "/workspace/out"}])
    with pytest.raises(ValueError, match="Output directories"):
        converted_task(SOURCE, row)


def _event(step: str, round_: int, **attrs: str) -> dict[str, Any]:
    return {"kind": "event", "item_id": "d03.optics.lenses--2", "step": step, "round": round_, "attrs": attrs}


def test_accepted_item_uses_the_last_build_of_the_accepted_round():
    rows = [
        _event("built", 0) | {"input_hash": "aaaaaaaaaaaa1111"},
        _event("terminal", 0, terminal="rejected", reason="too_hard"),
        _event("built", 1) | {"input_hash": "bbbbbbbbbbbb2222"},
        _event("built", 1) | {"input_hash": "cccccccccccc3333"},
        _event("terminal", 1, terminal="accepted", reason="accepted outside the band: too_easy"),
    ]
    text = "\n".join(json.dumps(row) for row in rows) + '\n{"kind": "ev'

    item = accepted_item(ledger_rows(text))

    assert item is not None
    assert (item.round, item.evidence_hash, item.band, item.group) == (1, "cccccccccccc", Band.TOO_EASY, "d03-too-easy")


def test_malformed_complete_ledger_record_is_rejected():
    rows = [_event("built", 0) | {"input_hash": "aaaaaaaaaaaa1111"}]
    with pytest.raises(json.JSONDecodeError):
        ledger_rows(json.dumps(rows[0]) + '\n{"kind": "ev\n')


def test_item_whose_last_terminal_is_not_accepted_is_excluded():
    rows = [
        _event("built", 0) | {"input_hash": "aaaaaaaaaaaa1111"},
        _event("terminal", 0, terminal="accepted", reason="calibrated"),
        _event("terminal", 0, terminal="abandoned", reason="rewound"),
    ]
    assert accepted_item(rows) is None
