# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rubric-only exemplars become checklist judges tagged no-reference."""

import json
from pathlib import Path

from tasktrove_verify.spec import JudgeSpec, parse_spec

from experiments.post_training.tasktrove.contract import VERIFIER_TOML
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, read_task_binary, write_task_binary
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
VERIFIER_DATA = "tests/verifier_data.json"


def _convert(name: str, source: str, blob: bytes | None = None):
    info = SourceInfo(source, SourceVerdict.KEEP, "llm-judge-freeform", "")
    return convert_one(info, "t.tar.gz", blob or (FIXTURES / f"{name}.tar.gz").read_bytes(), converter_index(), "ref")


def test_stackexchange_rubric_becomes_a_checklist_over_the_response_file():
    record = _convert("judge_rubric", "laion__stackexchange-unix-sandboxes-verified-v2")
    assert record.status == ConvertStatus.CONVERTED and record.converter == "judge_rubric" and record.mode == "judge"
    assert record.tags == ["judge", "rubric", "no-reference", "stackexchange", "unix", "shell"]
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec) and spec.rubric == "checklist" and not spec.exact_gate
    assert len(spec.criteria) == 4 and spec.criteria[0].startswith("The commands/config are technically correct")
    assert spec.question.startswith("I tried to make && make install") and spec.output == "/app/response.txt"
    assert "tests/judge.toml" not in task.files and VERIFIER_DATA not in task.files
    assert "rewardkit" not in task.text(DOCKERFILE)
    assert verify_task(record.task_binary) is None


def test_safety_principle_lines_become_the_criteria():
    record = _convert("judge_rubric_safety", "laion__nemotron-gym-safety-v3")
    assert record.status == ConvertStatus.CONVERTED
    assert record.tags == ["judge", "rubric", "no-reference", "safety"]
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec)
    assert len(spec.criteria) == 2 and spec.criteria[0].startswith("The response should engage helpfully")


def test_empty_rubric_is_a_null_grader():
    task = read_task_binary((FIXTURES / "judge_rubric.tar.gz").read_bytes())
    data = json.loads(task.text(VERIFIER_DATA))
    data["rubric"] = []
    task.files[VERIFIER_DATA] = json.dumps(data).encode()
    record = _convert("judge_rubric", "laion__stackexchange-unix-sandboxes-verified-v2", write_task_binary(task))
    assert record.status == ConvertStatus.NULL_GRADER
