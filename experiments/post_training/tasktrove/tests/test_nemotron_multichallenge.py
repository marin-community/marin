# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The multichallenge exemplar becomes a checklist judge over the shipped transcript."""

from pathlib import Path

from tasktrove_verify.spec import JudgeSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.nemotron_multichallenge import JUDGE_TOML
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import VERIFIER_TOML
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, read_task_binary, write_task_binary
from experiments.post_training.tasktrove.verify import verify_task

FIXTURE = Path(__file__).parents[1] / "fixtures" / "nemotron_multichallenge.tar.gz"
INFO = SourceInfo("laion__nemotron-gym-multichallenge-advanced-v4", SourceVerdict.KEEP, "llm-judge-freeform", "")


def _convert(blob: bytes):
    return convert_one(INFO, "t.tar.gz", blob, converter_index(), "ref")


def test_exemplar_converts_to_a_checklist_judge_with_the_transcript_as_context():
    record = _convert(FIXTURE.read_bytes())
    assert record.status == ConvertStatus.CONVERTED and record.mode == "judge"
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec) and spec.rubric == "checklist" and not spec.exact_gate
    assert len(spec.criteria) == 4 and all(question.endswith("?") for question in spec.criteria)
    assert spec.context == "conversation.txt" and "tests/conversation.txt" in task.files
    assert JUDGE_TOML not in task.files and "tests/sitecustomize.py" not in task.files
    assert "rewardkit" not in task.text(DOCKERFILE)
    assert verify_task(record.task_binary) is None


def test_missing_requirement_block_is_a_null_grader():
    task = read_task_binary(FIXTURE.read_bytes())
    task.files[JUDGE_TOML] = task.text(JUDGE_TOML).replace("Requirement:", "Note:").encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.NULL_GRADER


def test_negated_requirement_preserves_criterion_polarity():
    task = read_task_binary(FIXTURE.read_bytes())
    task.files[JUDGE_TOML] = (
        task.text(JUDGE_TOML)
        .replace(
            "Pass when the candidate clearly satisfies this requirement.",
            "Pass when the candidate clearly does not satisfy the condition queried by this requirement.",
            1,
        )
        .encode()
    )
    record = _convert(write_task_binary(task))
    converted = read_task_binary(record.task_binary)
    spec = parse_spec(converted.text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec)
    assert spec.criteria[0] == (
        "The candidate must answer no to this question: Does the response present a single consolidated smoothie or "
        "bowl (not multiple separate recipes), expressed as one cohesive preparation?"
    )
