# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The graded step's checks on converted exemplars: a sound task passes, each defect is named."""

import re
from pathlib import Path

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import INSTALL_MARKER, VERIFIER_TOML
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    TaskFiles,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import Check, verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _converted(name: str, family: str) -> TaskFiles:
    info = SourceInfo(name, SourceVerdict.KEEP, family, "")
    record = convert_one(info, "t.tar.gz", (FIXTURES / f"{name}.tar.gz").read_bytes(), converter_index(), "ref")
    assert record.task_binary is not None
    return read_task_binary(record.task_binary)


def _with(task: TaskFiles, **files: bytes) -> bytes:
    return write_task_binary(TaskFiles({**task.files, **files}))


def test_converted_exemplars_pass_every_check():
    assert verify_task(write_task_binary(_converted("nemotron_mcqa", "qa-short-answer"))) is None
    assert verify_task(write_task_binary(_converted("nemotron_math", "math-answer"))) is None


def test_missing_install_block_is_a_dockerfile_rejection():
    task = _converted("nemotron_mcqa", "qa-short-answer")
    stripped = task.text(DOCKERFILE).split(INSTALL_MARKER)[0]
    rejection = verify_task(_with(task, **{DOCKERFILE: stripped.encode()}))
    assert rejection is not None and rejection.check == Check.DOCKERFILE


def test_old_grader_dependency_in_dockerfile_is_rejected():
    task = _converted("nemotron_mcqa", "qa-short-answer")
    dockerfile = task.text(DOCKERFILE).replace("WORKDIR /app", "RUN pip install rewardkit\nWORKDIR /app")
    rejection = verify_task(_with(task, **{DOCKERFILE: dockerfile.encode()}))
    assert rejection is not None and rejection.check == Check.DOCKERFILE and "rewardkit" in rejection.detail


def test_expected_answer_in_instruction_is_a_gold_leak():
    task = _converted("nemotron_math", "math-answer")
    spec = re.sub(r'expected = ".*"', 'expected = "x^2 + 2x + 1 = 0"', task.text(VERIFIER_TOML))
    instruction = task.text(INSTRUCTION) + "\n\nHint: the answer is x^2 + 2x + 1 = 0."
    rejection = verify_task(_with(task, **{VERIFIER_TOML: spec.encode(), INSTRUCTION: instruction.encode()}))
    assert rejection is not None and rejection.check == Check.GOLD_LEAK


def test_solution_inside_the_binary_is_a_gold_leak():
    task = _converted("nemotron_math", "math-answer")
    rejection = verify_task(_with(task, **{"solution/solve.sh": b"echo 42\n"}))
    assert rejection is not None and rejection.check == Check.GOLD_LEAK


def test_unparseable_spec_is_rejected_by_the_spec_check():
    task = _converted("nemotron_mcqa", "qa-short-answer")
    rejection = verify_task(_with(task, **{VERIFIER_TOML: b'mode = "mcq"\n'}))
    assert rejection is not None and rejection.check == Check.SPEC


def test_spec_the_grader_cannot_use_is_rejected_with_the_grader_reason():
    task = _converted("nemotron_mcqa", "qa-short-answer")
    spec = task.text(VERIFIER_TOML).replace('expected = "', 'expected = "9')
    rejection = verify_task(_with(task, **{VERIFIER_TOML: spec.encode()}))
    assert rejection is not None and rejection.check == Check.EMPTY and "invalid_task" in rejection.detail
