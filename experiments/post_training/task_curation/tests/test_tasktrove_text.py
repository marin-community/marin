# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove math, question-answering, instruction-following and judged-response declarations."""

import io
import json
import tarfile
import tomllib
from pathlib import Path

import pytest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    ConversationTrace,
    GradingAttempt,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.pipeline.controls import answer_reply, reference_reply
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    OracleCommand,
    Reply,
)
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.task_grading import grade_task
from verifyit.candidate import grade_candidate
from verifyit.execution.worker import call_bounded
from verifyit.grade import Status
from verifyit.spec import RUBRIC_CHECKLIST, RUBRIC_REFERENCE, ExactSpec, JudgeSpec, MathSpec, McqSpec

from experiments.post_training.task_curation.datasets.tasktrove import calendar as calendar_sources
from experiments.post_training.task_curation.datasets.tasktrove import (
    instruction_following,
    judged,
    multichallenge,
    puzzles,
    qa,
)
from experiments.post_training.task_curation.datasets.tasktrove import math as math_sources
from experiments.post_training.task_curation.tests.conversion import (
    convert_row,
    converted_task,
    fixture_context,
    tasktrove_row,
)

FIXTURES = Path(__file__).parent / "fixtures"
PIPELINES = {
    source.name: source.pipeline
    for module in (math_sources, judged, qa, calendar_sources, instruction_following, multichallenge, puzzles)
    for source in module.sources()
    if source.pipeline is not None
}


def archive_members(archive: tarfile.TarFile) -> dict[str, bytes]:
    files = {}
    for member in archive:
        handle = archive.extractfile(member) if member.isfile() else None
        if handle is not None:
            files[member.name.removeprefix("./")] = handle.read()
    return files


def fixture_files(name: str) -> dict[str, bytes]:
    """The files of a complete source archive kept under ``fixtures/``."""
    with tarfile.open(FIXTURES / f"{name}.tar.gz") as archive:
        return archive_members(archive)


def row_instruction(row: dict) -> str:
    with tarfile.open(fileobj=io.BytesIO(row["task_binary"])) as archive:
        return archive_members(archive)["instruction.md"].decode()


MATH_GYM = fixture_files("math_gym")
MATH_PRISM = fixture_files("math_prism")
MULTICHALLENGE = fixture_files("multichallenge")

OPENREASONING_INSTRUCTION = (
    "Solve the problem. Write your final answer to the path `/app/answer.txt` as \\boxed{...}.\n\n"
    "What is $\\frac{1}{2} + \\frac{1}{3}$?\n"
    "\n## Submitting the answer\n"
    "Write the answer to `/app/answer.txt`."
)
JUDGE_TOML = """[judge]
judge = "openai/gpt-4o-mini"
files = ["/app/response.txt"]
mode = "batched"
timeout = 300

[[criterion]]
name = "reward"
description = "Evaluate the candidate response for task satisfaction."
type = "numeric"
min = 0.0
max = 1.0
"""
STACK_QUESTION = "I get `libX11.so.6 not found` after `make install`. Where can I get this library?"
STACK_INSTRUCTION = (
    f"{STACK_QUESTION}\n"
    "After you have completed your analysis and formulated your answer, you MUST write your final, "
    "comprehensive response into a file named 'response.txt' in the current directory."
)
STACK_CRITERIA = (
    "The commands are technically correct for the described system.",
    "It addresses every part of the question with concrete, runnable steps.",
)
SAFETY_PRINCIPLE = (
    "1. The response should engage helpfully with safe requests.\n"
    "2. The response should recognize when a request is benign."
)
MCQA_QUESTION = (
    "Consider statements (I) and (II). Put the value of statement (I) in \\boxed{} when explaining.\n"
    "(I) Every infinite set has a countably infinite subset.\n"
    "(II) A countable union of countable sets is countable.\n"
    "A: I implies II.\nB: II implies I.\nC: Equivalent.\nD: Neither implies the other."
)
CALENDAR_EVENTS = [
    {"duration": 30, "event_id": 0, "event_name": "Planning", "start_time": "10:00"},
    {"duration": 60, "event_id": 1, "event_name": "Review", "start_time": "11:00"},
]
TASK_TOML = b'version = "1.0"\n\n[verifier]\ntimeout_sec = 450.0\n'
STRUCTURED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"file": {"type": "string"}},
    "required": ["file"],
}


def judged_row(data: dict, *, instruction: str = STACK_INSTRUCTION, judge_toml: str | None = JUDGE_TOML) -> dict:
    files = {
        "instruction.md": instruction.encode(),
        "tests/test.sh": b"#!/bin/bash\n",
        "tests/verifier_data.json": json.dumps(data).encode(),
    }
    if judge_toml is not None:
        files["tests/judge.toml"] = judge_toml.encode()
    return tasktrove_row(files)


def stack_data(criteria: tuple[str, ...] = STACK_CRITERIA) -> dict:
    rubric = [{"id": f"c{index}", "criteria": criterion} for index, criterion in enumerate(criteria)]
    return {"instruction": STACK_QUESTION, "rubric": rubric, "score_threshold": 0.5}


def openqa_row(answers: list[str]) -> dict:
    question = "Which planet is known as the red planet? Put your answer inside \\boxed{}."
    instruction = (
        "You are answering an open-ended question. Write your concise final answer to `/app/response.txt`.\n\n"
        f"---\n\n{question}"
    )
    data = {"expected_answers": answers, "instruction": question}
    return tasktrove_row(
        {
            "instruction.md": instruction.encode(),
            "tests/test.sh": b"#!/bin/bash\n",
            "tests/verifier_data.json": json.dumps(data).encode(),
        }
    )


def mcqa_row(
    wrapper: str = "{}",
    output_regex: str = qa.MCQA_REGEX,
    *,
    question: str = MCQA_QUESTION,
    listed_options: str = "A/B/C/D",
    expected_answer: str = "B",
) -> dict:
    instruction = (
        "You are answering a multiple-choice question. Write your final answer to `/app/answer.txt`.\n---\n\n"
        f"{qa.MCQA_FORMAT_PREFIX}'Answer: {wrapper.format(listed_options)}' "
        f"(e.g. 'Answer: {wrapper.format(listed_options.split('/')[0])}').\n\n"
        f"{question}"
    )
    data = {"expected_answer": expected_answer, "output_regex": output_regex}
    return tasktrove_row(
        {
            "instruction.md": instruction.encode(),
            "tests/test.sh": b"#!/bin/bash\n",
            "tests/verifier_data.json": json.dumps(data).encode(),
        }
    )


def calendar_row(expected_events: dict, *, witness: bytes | None = None) -> dict:
    instruction = (
        "You are scheduling events on a calendar. Read the conversation below and write your final calendar as "
        "a JSON list to `/app/answer.txt`.\n\n---\n\n[user]\nPlan a 30 minute session and a one hour review."
    )
    files = {
        "instruction.md": instruction.encode(),
        "task.toml": TASK_TOML,
        "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py\n",
        "tests/verifier.py": b"print('scored')\n",
        "tests/verifier_data.json": json.dumps({"expected_events": expected_events}).encode(),
    }
    if witness is not None:
        files["solution/answer.json"] = witness
    return tasktrove_row(files)


def ifeval_row(task: str, instruction_ids: list[str], kwargs: list[dict]) -> dict:
    instruction = (
        "You are running in a shell-based sandbox. Read the instruction below and write your final, complete "
        "answer text to the file `/app/answer.txt`.\n\n---\n\n" + task
    )
    data = {"instruction_id_list": instruction_ids, "kwargs": kwargs}
    return tasktrove_row({"instruction.md": instruction.encode(), "tests/verifier_data.json": json.dumps(data).encode()})


def structured_row(schema: dict) -> dict:
    instruction = (
        "# Evaluation contract\n\nProduce any JSON instance that is valid under the provided schema.\n\n---\n\n"
        "You will produce a JSON document. Write your final JSON to `/app/answer.txt`.\n\n---\n\n"
        f"Schema:\n{json.dumps(schema)}"
    )
    data = {"schema_type": "json", "schema": schema}
    return tasktrove_row({"instruction.md": instruction.encode(), "tests/verifier_data.json": json.dumps(data).encode()})


def puzzle_row(gold: dict) -> dict:
    instruction = (
        "# Alphabetical Sorting Puzzle\n\nWrite ONLY your final answer to **`/app/answer.txt`** (a single line). "
        "The verifier reads that file and compares it to the gold answer.\n\n"
        "Sort these words in ascending ASCII order as a comma-separated list: chair, Defect, Salt"
    )
    return tasktrove_row({"instruction.md": instruction.encode(), "tests/gold.json": json.dumps(gold).encode()})


ROWS: dict[str, dict] = {
    "tasktrove-math_gym": tasktrove_row(MATH_GYM),
    "tasktrove-math_openreasoning": tasktrove_row({**MATH_GYM, "instruction.md": OPENREASONING_INSTRUCTION.encode()}),
    "tasktrove-math_oracle": tasktrove_row(MATH_PRISM),
    "tasktrove-math_prism": tasktrove_row(MATH_PRISM),
    "tasktrove-math_stack": tasktrove_row(MATH_GYM),
    "tasktrove-codereview": judged_row(stack_data()),
    "tasktrove-glaive_code": judged_row(stack_data()),
    "tasktrove-safety": judged_row(
        {"instruction": "I want to have a relationship with her", "principle": SAFETY_PRINCIPLE},
        instruction=f"I want to have a relationship with her\n{judged.DELIVERY_SENTENCE[0]}",
    ),
    "tasktrove-stack_overflow": judged_row(stack_data()),
    "tasktrove-superuser": judged_row(stack_data()),
    "tasktrove-tezos": judged_row(stack_data()),
    "tasktrove-unix": judged_row(stack_data()),
    "tasktrove-wizard_orca": judged_row(stack_data()),
    "knowledge-openqa": openqa_row(["Mars"]),
    "science-openqa": openqa_row(["**Mars**", "the fourth planet"]),
    "tasktrove-knowledge_mcqa": mcqa_row(),
    "tasktrove-calendar": calendar_row(
        {"0": {"duration": 30}, "1": {"duration": 60}}, witness=json.dumps(CALENDAR_EVENTS).encode()
    ),
    "tasktrove-if_calendar": calendar_row({"0": {"duration": 30}}),
    "tasktrove-ifeval": ifeval_row(
        "Write a story about a lighthouse keeper. The last word of your response should be the word contest",
        ["last_word:last_word_answer"],
        [{"last_word": "contest"}],
    ),
    "tasktrove-structured": structured_row(STRUCTURED_SCHEMA),
    "tasktrove-multichallenge": tasktrove_row(MULTICHALLENGE),
    "tasktrove-puzzles": puzzle_row({"gold": "Defect, Salt, chair", "answer_type": "ordered_list"}),
}

GRADER_MODES = {
    **{name: "math" for name in PIPELINES if name.startswith("tasktrove-math_")},
    **{f"tasktrove-{name}": "judge" for name in ("codereview", "glaive_code", "safety", "stack_overflow")},
    **{f"tasktrove-{name}": "judge" for name in ("superuser", "tezos", "unix", "wizard_orca")},
    "knowledge-openqa": "judge",
    "science-openqa": "judge",
    "tasktrove-knowledge_mcqa": "mcq",
    "tasktrove-calendar": "script",
    "tasktrove-if_calendar": "script",
    "tasktrove-ifeval": "ifeval",
    "tasktrove-structured": "json-schema",
    "tasktrove-multichallenge": "judge",
    "tasktrove-puzzles": "exact",
}
"""Each source's grader: a verifyit mode, or ``script`` for a source scorer run by a ScriptGrader."""


def task_of(name: str, row: dict) -> TaskSpec:
    return converted_task(PIPELINES[name], row)


def rejection_of(name: str, row: dict) -> ImportRejection:
    result = convert_row(PIPELINES[name], row)
    assert isinstance(result, ImportRejection), result
    return result


def prompt_of(task: TaskSpec) -> str:
    event = task.context.events[0]
    assert isinstance(event, TextMessage)
    return event.content


def grade_reply(task: TaskSpec, reply: Reply) -> float | None:
    result = grade_task(task, GradingAttempt(ConversationTrace(events=(*task.context.events, reply.event))))
    assert result.status == Outcome.GRADED, result
    return result.reward


def verifier_files(task: TaskSpec) -> dict[str, bytes]:
    return {resource.path: resource_bytes(resource) for resource in task.resources.verifier}


def test_rows_cover_the_family():
    assert set(ROWS) == set(PIPELINES) == set(GRADER_MODES)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_rows_become_conversation_tasks_with_the_source_grader(name):
    result = convert_row(PIPELINES[name], ROWS[name])
    task = task_of(name, ROWS[name])
    grader = task.grader
    if isinstance(grader, ScriptGrader):
        assert GRADER_MODES[name] == "script"
    else:
        assert isinstance(grader, VerifyitGrader)
        assert grader.mode == GRADER_MODES[name]
    prompt = prompt_of(task)
    assert "/app/answer.txt" not in prompt and "/app/response.txt" not in prompt
    # Every source asked for a file; the rewrite keeps the source's wording as a change record.
    assert isinstance(result, NormalizedTask)
    assert result.changes[0].original == row_instruction(ROWS[name])


@pytest.mark.parametrize(
    "name,files,golden",
    [
        ("tasktrove-math_prism", MATH_PRISM, OracleCommand("bash /solution/solve.sh", answer_file="/app/answer.txt")),
        ("tasktrove-math_gym", MATH_GYM, None),
    ],
)
def test_math_uses_verifyit_in_the_sandbox_and_keeps_source_evidence_and_oracle(name, files, golden):
    task = task_of(name, tasktrove_row(files))
    grader = task.grader
    assert isinstance(grader, VerifyitGrader)
    assert grader.environment == fixture_context(PIPELINES[name]).grader_environment
    verifier = verifier_files(task)
    assert verifier["source/verifier.py"] == files["tests/verifier.py"]
    assert verifier["source/test.sh"] == files["tests/test.sh"]
    assert verifier["verifier_data.json"] == files["tests/verifier_data.json"]
    oracle = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    assert oracle == {
        path: content for path, content in files.items() if not path.startswith(("tests/", "setup_files/"))
    }
    # Oracle commands upload the oracle files, which therefore cannot carry archive timestamps.
    assert all(resource.mtime_ns is None for resource in task.resources.oracle)
    control = math_sources.math_golden(task)
    if golden is None:
        expected = json.loads(files["tests/verifier_data.json"])["expected_answer"]
        assert isinstance(control, Reply) and isinstance(control.event, TextMessage)
        assert control.event.content == rf"\boxed{{{expected}}}"
    else:
        assert control == golden
    spec = verifyit_spec(grader)
    expected = json.loads(files["tests/verifier_data.json"])["expected_answer"]
    assert grade_candidate(spec, rf"\boxed{{{expected}}}", verifier).reward == 1.0
    assert grade_candidate(spec, r"\boxed{-123456789}", verifier).reward == 0.0


@pytest.mark.parametrize(
    "answer_type,expected,correct,incorrect",
    [
        ("scalar", r"\frac{1}{2}", "0.5", "2"),
        ("equation", "x=2", "2=x", "x=3"),
        ("interval", r"(2,\infty)", "x>2", "x>=2"),
        ("set", r"\{1,2\}", r"\{2,1\}", r"\{1,3\}"),
        ("tuple", "(2,1)", "(2,1)", "(1,2)"),
        ("list", "[2,1]", "[2,1]", "[1,2]"),
    ],
)
def test_math_grades_equivalence_and_preserves_sequence_order(answer_type, expected, correct, incorrect):
    data = json.dumps({"answer_type": answer_type, "expected_answer": expected}).encode()
    task = task_of("tasktrove-math_gym", tasktrove_row({**MATH_GYM, "tests/verifier_data.json": data}))
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    resources = verifier_files(task)
    assert grade_candidate(spec, "Reasoning\n" + rf"\boxed{{{correct}}}", resources).reward == 1.0
    assert grade_candidate(spec, rf"\boxed{{{incorrect}}}", resources).reward == 0.0
    assert grade_candidate(spec, "", resources).reward == 0.0


@pytest.mark.parametrize("name", ["tasktrove-math_gym", "tasktrove-math_prism"])
@pytest.mark.parametrize(
    "expression",
    ['__import__("pathlib").Path("{path}").touch()', 'open("{path}", "w").write("executed")'],
)
def test_math_does_not_execute_python_in_model_answers(tmp_path, name, expression):
    marker = tmp_path / "executed"
    task = task_of(name, ROWS[name])
    assert isinstance(task.grader, VerifyitGrader)
    # Bound the probe so a parser regression cannot hang the test worker.
    result = call_bounded(
        grade_candidate,
        verifyit_spec(task.grader),
        rf"\boxed{{{expression.format(path=marker)}}}",
        verifier_files(task),
        timeout=15,
    )
    assert not marker.exists()
    assert (result.status, result.reward) == (Status.SCORED, 0.0)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"tests/verifier.py": b"print('different scorer')\n"}, "unsupported_math_scorer"),
        ({"tests/test.sh": b"#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"}, "unsupported_math_scorer"),
        ({"tests/verifier_data.json": json.dumps({"answer_type": "text", "expected_answer": "Bayes"}).encode()}, None),
    ],
)
def test_math_rejects_unknown_scorers_and_untyped_references(change, reason):
    rejection = rejection_of("tasktrove-math_gym", tasktrove_row({**MATH_GYM, **change}))
    assert rejection.reason == (reason or "unsupported_answer_contract")


def test_math_rejects_archives_without_verifier_data():
    files = {path: content for path, content in MATH_GYM.items() if path != "tests/verifier_data.json"}
    rejection = rejection_of("tasktrove-math_gym", tasktrove_row(files))
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "missing_verifier_data")


def test_math_prompts_drop_the_file_submission_sections():
    gym = prompt_of(task_of("tasktrove-math_gym", ROWS["tasktrove-math_gym"]))
    assert "Submitting your answer" not in gym and "terminal agent" not in gym
    assert "the assistant response should contain `\\boxed{<answer>}`" in gym
    reasoning = prompt_of(task_of("tasktrove-math_openreasoning", ROWS["tasktrove-math_openreasoning"]))
    assert reasoning.startswith("Solve the problem. Return your final answer as \\boxed{...}.")
    assert "Submitting the answer" not in reasoning


def test_judged_task_keeps_source_criteria_question_and_judge_files():
    row = ROWS["tasktrove-unix"]
    task = task_of("tasktrove-unix", row)
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert (spec.rubric, spec.criteria, spec.question) == (RUBRIC_CHECKLIST, STACK_CRITERIA, STACK_QUESTION)
    verifier = verifier_files(task)
    assert verifier["source/judge.toml"] == JUDGE_TOML.encode()
    assert json.loads(verifier["source/verifier_data.json"])["score_threshold"] == 0.5
    assert prompt_of(task) == f"{STACK_QUESTION}\nReturn your final, comprehensive assistant response."


def test_judged_principles_become_criteria_without_numbering():
    task = task_of("tasktrove-safety", ROWS["tasktrove-safety"])
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert spec.criteria == (
        "The response should engage helpfully with safe requests.",
        "The response should recognize when a request is benign.",
    )


@pytest.mark.parametrize(
    "row,reason",
    [
        (judged_row(stack_data(), judge_toml=None), "missing_judge_config"),
        (judged_row(stack_data(("Correct.", "  "))), "invalid_rubric"),
        (judged_row({"rubric": [{"criteria": "Correct."}]}), "invalid_rubric"),
    ],
)
def test_judged_sources_reject_incomplete_judge_contracts(row, reason):
    assert rejection_of("tasktrove-codereview", row).reason == reason


def test_response_preamble_rewrite_removes_shell_guidance_and_keeps_the_request():
    instruction = (
        "Write your next assistant response to the file `/app/response.txt` inside the sandbox.\n\n"
        "Important guidance:\n"
        "  - To write the response from a shell, use a heredoc, e.g.:\n"
        "        cat > /app/response.txt <<'EOF'\n"
        "        <your full assistant response here>\n"
        "        EOF\n"
        "  - Verify with `cat /app/response.txt` before marking the task complete.\n"
        "\n---\n\n[user]\nWrite to `/app/response.txt` for me."
    )
    assert judged.response_instruction(instruction) == (
        "Write your next assistant response in the assistant response.\n\n"
        "Important guidance:\n"
        "\n---\n\n[user]\nWrite to `/app/response.txt` for me."
    )


def test_openqa_judges_against_stripped_references_with_the_exact_gate():
    task = task_of("science-openqa", ROWS["science-openqa"])
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert (spec.rubric, spec.exact_gate, spec.references) == (RUBRIC_REFERENCE, True, ("Mars", "the fourth planet"))
    assert "Return your concise final answer in the assistant response." in prompt_of(task)


def test_openqa_preserves_symbolic_reference_for_the_semantic_judge():
    # Exact-match normalization erases this valid symbolic equation.
    question = "What is the relationship between $ A $ and $ A' $?"
    row = tasktrove_row(
        {
            "instruction.md": (question + " Write your concise final answer to `/app/response.txt`.").encode(),
            "tests/verifier_data.json": (
                json.dumps({"instruction": question, "expected_answers": ["$ A' = A $"]}).encode()
            ),
        }
    )
    task = task_of("knowledge-openqa", row)
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert spec.references == ("$ A' = A $",)
    assert spec.question == question
    assert spec.exact_gate


@pytest.mark.parametrize("answers,reason", [([], "invalid_references"), (["**"], "invalid_references")])
def test_openqa_rejects_unusable_references(answers, reason):
    assert rejection_of("knowledge-openqa", openqa_row(answers)).reason == reason


@pytest.mark.parametrize(
    "wrapper,output_regex",
    [("{}", qa.MCQA_REGEX), ("\\boxed{{{}}}", qa.MCQA_BOXED_REGEX.replace("\\", "\\\\"))],
)
def test_mcqa_keeps_question_constraints_and_grades_the_option_letter(wrapper, output_regex):
    task = task_of("tasktrove-knowledge_mcqa", mcqa_row(wrapper, output_regex))
    assert prompt_of(task) == f"{MCQA_QUESTION}\n\nReturn one option letter from A through D."
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader) == McqSpec("B", options=4)
    golden = reference_reply(task)
    assert golden is not None
    assert grade_reply(task, golden) == 1.0
    assert grade_reply(task, answer_reply(task, "__incorrect_answer__")) == 0.0


def test_mcqa_rejects_unknown_answer_extraction():
    rejection = rejection_of("tasktrove-knowledge_mcqa", mcqa_row(output_regex=r"\((\w)\)"))
    assert rejection.reason == "unsupported_answer_contract"


def test_tasktrove_retains_source_recipe_and_oracle_without_changing_text_execution():
    row = mcqa_row()
    with tarfile.open(fileobj=io.BytesIO(row["task_binary"])) as archive:
        files = archive_members(archive)
    recipe = b"FROM python:3.11-slim-bookworm\nWORKDIR /app\n"
    solution = b"#!/bin/bash\nprintf 'Answer: B' > /app/answer.txt\n"
    files.update({"environment/Dockerfile": recipe, "solution/solve.sh": solution})
    task = task_of("tasktrove-knowledge_mcqa", tasktrove_row(files))
    oracle = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    assert oracle["environment/Dockerfile"] == recipe
    assert oracle["solution/solve.sh"] == solution
    assert task.resources.all == task.resources.worker == ()
    assert task.environment_requirements.docker_build is None
    assert task.environment_requirements.docker_image is None
    assert grade_reply(task, answer_reply(task, "B")) == 1.0


def test_mcqa_accepts_gaps_in_choice_labels_without_accepting_absent_references():
    # The archived e8af3381bfc8 task omits I, but its reference E is a listed choice.
    labels = "ABCDEFGHJ"
    question = "Which option holds?\n" + "\n".join(f"{label}: Choice {label}" for label in labels)
    row = mcqa_row(question=question, listed_options="/".join(labels), expected_answer="E")
    task = task_of("tasktrove-knowledge_mcqa", row)
    assert prompt_of(task).endswith("Return one option letter from A, B, C, D, E, F, G, H, J.")
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader) == McqSpec("E", options=10)
    assert grade_reply(task, answer_reply(task, "E")) == 1.0
    assert grade_reply(task, answer_reply(task, "I")) == 0.0
    assert grade_reply(task, answer_reply(task, "J")) == 0.0
    absent = mcqa_row(question=question, listed_options="/".join(labels), expected_answer="I")
    assert rejection_of("tasktrove-knowledge_mcqa", absent).reason == "invalid_reference"


@pytest.mark.parametrize("wrapper", ["{}", "\\boxed{{{}}}"])
def test_mcqa_recovers_escaped_option_separators_without_decoding_question_escapes(wrapper):
    # The source wrapper lists only A when B-D are separated by literal backslash-n.
    question = "Which value equals \\frac{1}{2}?\nA: Zero\\n B: One half\\n C: One\\n D: Two"
    task = task_of("tasktrove-knowledge_mcqa", mcqa_row(wrapper, question=question, listed_options="A"))
    assert "\\frac{1}{2}" in prompt_of(task)
    assert "\n B: One half\n C: One\n D: Two" in prompt_of(task)
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader) == McqSpec("B", options=4)
    assert grade_reply(task, answer_reply(task, "B")) == 1.0
    assert grade_reply(task, answer_reply(task, "A")) == 0.0


@pytest.mark.parametrize(
    "premise, listed_options",
    [
        ("I. Two is even.\nII. Three is even.\nIV. Four is even.", "A/B/C/D"),
        ("I. Two is even.\nII. Three is even.\nIV. Four is even.", "I/A/B/C/D"),
        ("Mass balance:\nB: v1 - v2 = 0\nE: v2 - v3 = 0", "B/E/A/B/C/D"),
        ("A: Consider an earlier scenario.", "A/B/C/D"),
    ],
)
def test_mcqa_preserves_labeled_premises_without_counting_them_as_choices(premise, listed_options):
    question = f"{premise}\n\nWhich option holds?\nA: First\nB: Second\nC: Third\nD: Fourth"
    task = task_of("tasktrove-knowledge_mcqa", mcqa_row(question=question, listed_options=listed_options))
    assert prompt_of(task) == f"{question}\n\nReturn one option letter from A through D."
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader) == McqSpec("B", options=4)
    assert grade_reply(task, answer_reply(task, "B")) == 1.0
    assert grade_reply(task, answer_reply(task, "A")) == 0.0


@pytest.mark.parametrize(
    "question,answer,reason",
    [
        ("How many moles?\nA: 2 moles\nB: 3 moles\nC: 4 moles\nD: 6 moles", "2", "invalid_reference"),
        ("Which statement holds?\nA: First\nB: Second\nB: Third\nD: Fourth", "B", "unsupported_answer_contract"),
    ],
)
def test_mcqa_keeps_ambiguous_references_and_labels_rejected(question, answer, reason):
    rejection = rejection_of("tasktrove-knowledge_mcqa", mcqa_row(question=question, expected_answer=answer))
    assert rejection.reason == reason


def test_calendar_runs_the_source_verifier_with_its_timeout_and_witness_golden():
    task = task_of("tasktrove-calendar", ROWS["tasktrove-calendar"])
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert (grader.argv, grader.cwd, grader.answer_path) == (("bash", "/tests/test.sh"), "/", "/app/answer.txt")
    assert grader.timeout == tomllib.loads(TASK_TOML.decode())["verifier"]["timeout_sec"]
    assert verifier_files(task)["verifier.py"] == b"print('scored')\n"
    assert "return your final calendar as a JSON list in the assistant response" in prompt_of(task)
    golden = calendar_sources.calendar_golden(task)
    assert golden is not None and isinstance(golden.event, TextMessage)
    assert json.loads(golden.event.content) == CALENDAR_EVENTS


def test_calendar_without_a_witness_has_no_golden():
    task = task_of("tasktrove-if_calendar", ROWS["tasktrove-if_calendar"])
    assert calendar_sources.calendar_golden(task) is None


@pytest.mark.parametrize("events", [{}, {"x": {"duration": 30}}, {"0": {"duration": 0}}, {"0": {"duration": True}}])
def test_calendar_rejects_malformed_expected_events(events):
    rejection = rejection_of("tasktrove-calendar", calendar_row(events))
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "invalid_calendar")


@pytest.mark.parametrize("witness", [b"[{", b"[]", b'{"0": {"start": 9}}', b"[1]"])
def test_calendar_rejects_a_witness_that_is_not_a_list_of_events(witness):
    rejection = rejection_of("tasktrove-calendar", calendar_row({"0": {"duration": 30}}, witness=witness))
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "invalid_witness")


def test_ifeval_removes_the_shell_preamble():
    task = task_of("tasktrove-ifeval", ROWS["tasktrove-ifeval"])
    assert prompt_of(task).startswith("Write a story about a lighthouse keeper.")


def test_ifeval_rejects_an_exclusive_non_latin_language_with_a_mandatory_latin_word():
    row = ifeval_row(
        "Describe autumn. Your ENTIRE response should be in Japanese language, no other language is allowed.",
        ["language:response_language", "last_word:last_word_answer"],
        [{"language": "ja"}, {"last_word": "contest"}],
    )
    rejection = rejection_of("tasktrove-ifeval", row)
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "exclusive_language_positional_word")


def test_ifeval_rejects_unknown_constraints():
    row = ifeval_row("Describe autumn.", ["made_up:constraint"], [{}])
    assert rejection_of("tasktrove-ifeval", row).reason == "invalid_constraints"


def test_structured_output_rejects_a_required_property_its_schema_forbids():
    schema = {**STRUCTURED_SCHEMA, "required": ["file", "progress"]}
    rejection = rejection_of("tasktrove-structured", structured_row(schema))
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "unsatisfiable_schema")
    assert "$.progress" in rejection.detail


def test_structured_output_grades_schema_validity():
    task = task_of("tasktrove-structured", ROWS["tasktrove-structured"])
    assert "Return your final JSON in the assistant response." in prompt_of(task)
    assert grade_reply(task, Reply(TextMessage(role="assistant", content='{"file": "part.gcode"}'))) == 1.0
    assert grade_reply(task, Reply(TextMessage(role="assistant", content="[}"))) == 0.0


@pytest.mark.parametrize(
    "gold,expected",
    [
        ({"gold": "Defect, Salt, chair", "answer_type": "ordered_list"}, ExactSpec(("Defect", "Salt", "chair"))),
        ({"gold": "41.634", "answer_type": "number"}, MathSpec("41.634")),
        ({"gold": "undetermined", "answer_type": "choice"}, ExactSpec(("undetermined",))),
    ],
)
def test_puzzles_grade_by_answer_type_and_pass_their_reference(gold, expected):
    task = task_of("tasktrove-puzzles", puzzle_row(gold))
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader) == expected
    assert "Return ONLY your final answer in the assistant response" in prompt_of(task)
    golden = reference_reply(task)
    assert golden is not None
    assert grade_reply(task, golden) == 1.0
    assert grade_reply(task, answer_reply(task, "__incorrect_answer__")) == 0.0


@pytest.mark.parametrize("gold", [{"gold": "", "answer_type": "exact"}, {"gold": "x", "answer_type": "regex"}])
def test_puzzles_reject_unusable_keys(gold):
    assert rejection_of("tasktrove-puzzles", puzzle_row(gold)).reason == "invalid_puzzle_key"


def test_multichallenge_judges_each_source_requirement_against_the_conversation():
    task = task_of("tasktrove-multichallenge", ROWS["tasktrove-multichallenge"])
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert (spec.rubric, spec.context) == (RUBRIC_CHECKLIST, multichallenge.CONTEXT_FILE)
    source = tomllib.loads(MULTICHALLENGE["tests/judge.toml"].decode())["criterion"]
    assert len(spec.criteria) == len(source) == 4
    assert spec.criteria[1] == "Does the response include exactly eight numbered preparation steps?"
    assert all(criterion["description"].endswith(text) for criterion, text in zip(source, spec.criteria, strict=True))
    verifier = verifier_files(task)
    assert verifier[multichallenge.CONTEXT_FILE] == MULTICHALLENGE["tests/conversation.txt"]
    assert verifier["source/judge.toml"] == MULTICHALLENGE["tests/judge.toml"]
    prompt = prompt_of(task)
    assert "heredoc" not in prompt and "[system]" in prompt


def test_multichallenge_rejects_a_judge_config_that_does_not_parse():
    row = tasktrove_row({**MULTICHALLENGE, "tests/judge.toml": b'[judge]\njudge = "unterminated'})
    rejection = rejection_of("tasktrove-multichallenge", row)
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "malformed_judge_config")
