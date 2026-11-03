# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The unstaged program driver: running an authored program, and saving each workpiece's task and completions.

Invariants: none (pins the driver's program runner and output files).
"""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError
from rigging.timing import Duration, ExponentialBackoff
from taskcompendium.models import TaskSpec
from taskforge.llm.client import (
    Completion,
    ContextExhausted,
    FinishReason,
    ModelUnavailable,
    StructuredResult,
    Usage,
)
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError

from experiments.post_training.capability_driven_envs.catalog import example_catalog_path
from experiments.post_training.capability_driven_envs.generate_tasks import (
    SUMMARY_FILE,
    TASK_STATUS,
    generate_tasks,
    run_program,
    selected_ideas,
    task_workpieces,
    write_outputs,
)
from experiments.post_training.capability_driven_envs.program_prompt import (
    ProgramSubmission,
    example_program_source,
)

CAPABILITY = "d43.culinary.scaling"
COMPLETION = Completion(
    content="",
    reasoning="a recipe-scaling task",
    tool_calls=(),
    finish_reason=FinishReason.STOP,
    usage=Usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=10, cached_tokens=0),
    wall_time=1.0,
    ttft=0.1,
    decode_time=0.9,
    continuations=0,
    attempts=(),
)


class CannedModel:
    """Submits the example program for every request, except that the first calls raise ``errors`` in order."""

    model = "canned"

    def __init__(self, errors: Sequence[Exception] = ()):
        self.errors = list(errors)
        self.output_types: list[type[BaseModel]] = []

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("the driver asks only for structured output")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        self.output_types.append(output_type)
        if self.errors:
            raise self.errors.pop(0)
        value = output_type.model_validate({"source": example_program_source()})
        return StructuredResult(value, (COMPLETION,))


@pytest.mark.parametrize(
    ("source", "failure", "stderr"),
    [
        ("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n", "exited with status 3", "boom"),
        ("pass\n", "wrote no file", ""),
        ("import sys, pathlib\npathlib.Path(sys.argv[2]).write_text('{\"id\": 1}')\n", "not a TaskSpec", ""),
        ("import time\ntime.sleep(30)\n", "did not finish", ""),
    ],
    ids=["exit-status", "no-output", "not-a-task-spec", "timeout"],
)
def test_program_failures_are_recorded_with_the_programs_stderr(source, failure, stderr):
    run = asyncio.run(run_program(source, "workpiece--00", timeout=1))

    assert run.task is None
    assert failure in (run.failure or "")
    assert stderr in run.stderr


def test_timeout_kills_what_the_program_spawned(tmp_path: Path):
    pid_file = tmp_path / "grandchild.pid"
    source = (
        "import pathlib, subprocess, time\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )

    run = asyncio.run(run_program(source, "workpiece--00", timeout=0.5))

    assert "did not finish" in (run.failure or "")
    grandchild = int(pid_file.read_text())

    def gone() -> bool:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return True
        return False

    # SIGKILL is delivered asynchronously and init reaps the orphan, so the pid lingers briefly.
    assert ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(gone, Duration.from_seconds(10))


@pytest.mark.parametrize(
    "source", ["def broken(:\n", "x = 1\x00\n", "-" * 1_000_000 + "1\n"], ids=["syntax", "null-byte", "deep-nesting"]
)
def test_program_that_does_not_compile_is_rejected_at_submission(source: str):
    with pytest.raises(ValidationError, match="does not compile"):
        ProgramSubmission(source=source)


def test_driver_saves_each_task_and_its_completions_once_at_the_end(tmp_path: Path):
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY])
    model = CannedModel()

    generated = asyncio.run(generate_tasks(model, task_workpieces(ideas, 2), LLMPolicy(), 60, concurrency=2))
    output = tmp_path / "run"
    assert not output.exists()
    write_outputs(output, generated)

    workpiece_ids = [f"{CAPABILITY}--00", f"{CAPABILITY}--01"]
    assert json.loads((output / SUMMARY_FILE).read_text()) == dict.fromkeys(workpiece_ids, TASK_STATUS)
    assert model.output_types == [ProgramSubmission, ProgramSubmission]
    for workpiece_id in workpiece_ids:
        workpiece_dir = output / workpiece_id
        assert TaskSpec.model_validate_json((workpiece_dir / "task.json").read_text()).id == workpiece_id
        assert (workpiece_dir / "program.py").read_text() == example_program_source()
        [completion] = json.loads((workpiece_dir / "completions.json").read_text())
        assert completion["usage"]["prompt_tokens"] == 100
        request = json.loads((workpiece_dir / "request.json").read_text())
        assert CAPABILITY in request[-2]["content"]
        assert request[-1]["content"].startswith("Reply with one JSON object")


def test_workpiece_without_a_valid_submission_keeps_its_completions_and_no_program(tmp_path: Path):
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY])

    generated = asyncio.run(
        generate_tasks(
            CannedModel(errors=[StructuredOutputError(ValueError("not a JSON object"), (COMPLETION, COMPLETION))]),
            task_workpieces(ideas, 1),
            LLMPolicy(),
            60,
            1,
        )
    )
    write_outputs(tmp_path / "run", generated)

    workpiece_dir = tmp_path / "run" / f"{CAPABILITY}--00"
    assert "structured output invalid" in json.loads((tmp_path / "run" / SUMMARY_FILE).read_text())[workpiece_dir.name]
    assert len(json.loads((workpiece_dir / "completions.json").read_text())) == 2
    assert not (workpiece_dir / "program.py").exists() and not (workpiece_dir / "task.json").exists()


@pytest.mark.parametrize(
    "error",
    [ModelUnavailable("GLM attempts exhausted after 8: boom", ()), ContextExhausted("prompt too long", ())],
    ids=["unavailable", "context-exhausted"],
)
def test_model_error_fails_only_its_workpiece(tmp_path: Path, error: Exception):
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY])

    generated = asyncio.run(generate_tasks(CannedModel(errors=[error]), task_workpieces(ideas, 2), LLMPolicy(), 60, 1))
    write_outputs(tmp_path / "run", generated)

    failed, written = f"{CAPABILITY}--00", f"{CAPABILITY}--01"
    assert json.loads((tmp_path / "run" / SUMMARY_FILE).read_text()) == {failed: str(error), written: TASK_STATUS}
    assert json.loads((tmp_path / "run" / failed / "completions.json").read_text()) == []
    assert not (tmp_path / "run" / failed / "task.json").exists()
    assert TaskSpec.model_validate_json((tmp_path / "run" / written / "task.json").read_text()).id == written
