# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove math, question-answering, instruction-following and judged-response declarations."""

import io
import json
import shlex
import subprocess
import sys
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
from taskcompendium.pipeline.controls import reference_reply, wrong_reply
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    OracleCommand,
    Reply,
)
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.task_grading import grade_task
from verifyit.spec import RUBRIC_CHECKLIST, RUBRIC_REFERENCE, ExactSpec, JudgeSpec, MathSpec, McqSpec, ScriptSpec

from experiments.post_training.task_curation.datasets.tasktrove import calendar as calendar_sources
from experiments.post_training.task_curation.datasets.tasktrove import (
    instruction_following,
    judged,
    multichallenge,
    multichallenge_grade,
    puzzles,
    qa,
)
from experiments.post_training.task_curation.datasets.tasktrove import math as math_sources
from experiments.post_training.task_curation.tests.conversion import (
    FIXTURE_GRADER_ENVIRONMENT,
    convert_row,
    converted_task,
    tasktrove_row,
)

FIXTURES = Path(__file__).parent / "fixtures"
PIPELINES = {
    pipeline.name: pipeline
    for module in (math_sources, judged, qa, calendar_sources, instruction_following, multichallenge, puzzles)
    for pipeline in module.pipelines()
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


def mcqa_row(wrapper: str = "{}", output_regex: str = qa.MCQA_REGEX) -> dict:
    instruction = (
        "You are answering a multiple-choice question. Write your final answer to `/app/answer.txt`.\n---\n\n"
        f"{qa.MCQA_FORMAT_PREFIX}'Answer: {wrapper.format('A/B/C/D')}' (e.g. 'Answer: {wrapper.format('B')}').\n\n"
        f"{MCQA_QUESTION}"
    )
    data = {"expected_answer": "B", "output_regex": output_regex}
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
    **{name: "script" for name in PIPELINES if name.startswith("tasktrove-math_")},
    **{f"tasktrove-{name}": "judge" for name in ("codereview", "glaive_code", "safety", "stack_overflow")},
    **{f"tasktrove-{name}": "judge" for name in ("superuser", "tezos", "unix", "wizard_orca")},
    "knowledge-openqa": "judge",
    "science-openqa": "judge",
    "tasktrove-knowledge_mcqa": "mcq",
    "tasktrove-calendar": "script",
    "tasktrove-if_calendar": "script",
    "tasktrove-ifeval": "ifeval",
    "tasktrove-structured": "json-schema",
    "tasktrove-multichallenge": "script",
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
def test_math_runs_the_source_scorer_and_its_oracle(name, files, golden):
    task = task_of(name, tasktrove_row(files))
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert grader.environment == FIXTURE_GRADER_ENVIRONMENT
    verifier = verifier_files(task)
    assert verifier["verifier.py"] == files["tests/verifier.py"]
    assert verifier["test.sh"] == files["tests/test.sh"]
    assert verifier["verifier_data.json"] == files["tests/verifier_data.json"]
    oracle = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    assert oracle == {path: content for path, content in files.items() if path.startswith("solution/")}
    # Oracle commands upload the oracle files, which therefore cannot carry archive timestamps.
    assert all(resource.mtime_ns is None for resource in task.resources.oracle)
    control = math_sources.math_golden(task)
    if golden is None:
        expected = json.loads(files["tests/verifier_data.json"])["expected_answer"]
        assert isinstance(control, Reply) and isinstance(control.event, TextMessage)
        assert control.event.content == rf"\boxed{{{expected}}}"
    else:
        assert control == golden


def test_math_pins_numpy_only_for_the_gym_scorer():
    gym = task_of("tasktrove-math_gym", ROWS["tasktrove-math_gym"])
    prism = task_of("tasktrove-math_prism", ROWS["tasktrove-math_prism"])
    assert isinstance(gym.grader, ScriptGrader) and isinstance(prism.grader, ScriptGrader)
    assert "numpy==2.1.3" in gym.grader.argv
    assert "numpy==2.1.3" not in prism.grader.argv


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


def run_math_grade(tmp_path: Path, *pins: str) -> tuple[int, bool]:
    """Run the grade script beside a stand-in ``test.sh``; return its exit code and whether test.sh ran."""
    script = tmp_path / math_sources.GRADE_SCRIPT
    script.write_bytes(Path(math_sources.__file__).with_name(math_sources.GRADE_SCRIPT).read_bytes())
    marker = tmp_path / "ran"
    (tmp_path / "test.sh").write_text(f"touch {shlex.quote(str(marker))}\nexit 3\n")
    completed = subprocess.run([sys.executable, str(script), *pins], capture_output=True, check=False)
    return completed.returncode, marker.exists()


def test_math_grade_runs_the_source_runner_under_matching_pins(tmp_path):
    python = f"python=={sys.version_info.major}.{sys.version_info.minor}"
    assert run_math_grade(tmp_path, python, f"pytest=={pytest.__version__}") == (3, True)


@pytest.mark.parametrize("pin", ["python==2.7", "pytest==0.0.1", "not-an-installed-distribution==1.0"])
def test_math_grade_refuses_mismatched_pins_before_the_runner(tmp_path, pin):
    assert run_math_grade(tmp_path, pin) == (2, False)


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
    assert grade_reply(task, wrong_reply(task)) == 0.0


def test_mcqa_rejects_unknown_answer_extraction():
    rejection = rejection_of("tasktrove-knowledge_mcqa", mcqa_row(output_regex=r"\((\w)\)"))
    assert rejection.reason == "unsupported_answer_contract"


def test_calendar_runs_the_source_verifier_with_its_timeout_and_witness_controls():
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
    negative = calendar_sources.calendar_negative(task).event
    assert isinstance(negative, TextMessage)
    assert json.loads(negative.content) == CALENDAR_EVENTS[1:]


def test_calendar_without_a_witness_has_no_golden_and_a_wrong_negative():
    task = task_of("tasktrove-if_calendar", ROWS["tasktrove-if_calendar"])
    assert calendar_sources.calendar_golden(task) is None
    assert calendar_sources.calendar_negative(task) == wrong_reply(task)


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
    assert grade_reply(task, instruction_following.malformed_json(task)) == 0.0


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
    assert grade_reply(task, wrong_reply(task)) == 0.0


@pytest.mark.parametrize("gold", [{"gold": "", "answer_type": "exact"}, {"gold": "x", "answer_type": "regex"}])
def test_puzzles_reject_unusable_keys(gold):
    assert rejection_of("tasktrove-puzzles", puzzle_row(gold)).reason == "invalid_puzzle_key"


def test_multichallenge_runs_the_source_suite_with_its_files_beside_the_grade_script():
    task = task_of("tasktrove-multichallenge", ROWS["tasktrove-multichallenge"])
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, ScriptSpec)
    assert (spec.path, spec.verdict_file) == (multichallenge.GRADE_PATH, multichallenge_grade.VERDICT_FILENAME)
    assert task.grader.environment == FIXTURE_GRADER_ENVIRONMENT
    verifier = verifier_files(task)
    for path, content in MULTICHALLENGE.items():
        target = path.removeprefix("tests/") if path.startswith("tests/") else f"__source/{path}"
        assert verifier[target] == content
    # RewardKit reads a visible subdirectory instead of the flat suite, so added files hide under "__".
    assert all("/" not in path or path.startswith("__") for path in verifier)
    prompt = prompt_of(task)
    assert "heredoc" not in prompt and "[system]" in prompt


@pytest.mark.parametrize(
    "path,original,replacement",
    [
        (
            "tests/judge.toml",
            'files = ["/tests/conversation.txt", "/app/response.txt"]',
            'files = ["/proc/self/environ"]',
        ),
        ("tests/judge.toml", "[judge]", '[judge]\napi_base = "https://unexpected.invalid"'),
        ("tests/judge.toml", 'type = "numeric"', 'files = ["/proc/self/environ"]\ntype = "numeric"'),
        ("tests/test.sh", "set -euo pipefail", 'set -euo pipefail\necho "$TOGETHER_API_KEY"'),
        ("task.toml", "together_ai/Qwen/Qwen3.5-9B", "openai/gpt-4o"),
    ],
)
def test_multichallenge_rejects_redirected_providers_files_and_scripts(path, original, replacement):
    content = MULTICHALLENGE[path].decode()
    assert original in content
    row = tasktrove_row({**MULTICHALLENGE, path: content.replace(original, replacement).encode()})
    assert rejection_of("tasktrove-multichallenge", row).reason in {
        "unsupported_rewardkit_judge",
        "unsupported_rewardkit_runtime",
        "unsupported_rewardkit_provider",
    }


def test_multichallenge_rejects_an_extra_test_file():
    row = tasktrove_row({**MULTICHALLENGE, "tests/extra.py": b"print('extra')\n"})
    assert rejection_of("tasktrove-multichallenge", row).reason == "unsupported_rewardkit_layout"


@pytest.mark.parametrize("value", [0.0, 0.125, 1.0])
def test_multichallenge_grade_reports_the_source_reward_unchanged(tmp_path, value):
    (tmp_path / "test.sh").write_text(
        f"printf '%s' '{json.dumps({'reward': value})}' > {shlex.quote(str(tmp_path / 'reward.json'))}\n"
    )
    assert multichallenge_grade.run_source(tmp_path, tmp_path, 5) == {
        "status": "scored",
        "reward": value,
        "detail": {"source": "harbor-rewardkit"},
    }


def test_multichallenge_grade_reports_a_failed_run_as_infrastructure_without_its_logs(tmp_path):
    (tmp_path / "reward.json").write_text('{"reward": 1.0}')
    (tmp_path / "test.sh").write_text("echo 'provider-header-secret-value' >&2\nexit 7\n")
    verdict = multichallenge_grade.run_source(tmp_path, tmp_path, 5)
    assert (verdict["status"], verdict["reward"], verdict["detail"]["exit_code"]) == ("infra_error", 0.0, 7)
    assert "provider-header-secret-value" not in json.dumps(verdict)


@pytest.mark.parametrize(
    "name,content",
    [
        ("deterministic_gate", b"def broken(hidden_source_text\n"),
        ("verifier.py", b"def broken(hidden_source_text\n"),
        ("verifier_data.json", b'{"hidden_reference_text":'),
        ("judge.toml", b'[judge]\nreference = "hidden_reference_text'),
    ],
)
def test_multichallenge_grade_rejects_a_malformed_suite_before_running_it(tmp_path, name, content):
    marker = tmp_path / "runner-started"
    (tmp_path / "test.sh").write_text(f"touch {shlex.quote(str(marker))}\nexit 7\n")
    (tmp_path / name).write_bytes(content)
    verdict = multichallenge_grade.run_source(tmp_path, tmp_path, 5)
    assert (verdict["status"], verdict["detail"]["file"]) == ("invalid_task", name)
    assert not marker.exists()
    assert "hidden_" not in json.dumps(verdict)


@pytest.mark.parametrize("payload", [{"other": 1}, {"reward": 1, "extra": 0}, {"reward": True}, {"reward": 2}])
def test_multichallenge_grade_refuses_a_malformed_source_reward(tmp_path, payload):
    (tmp_path / "test.sh").write_text("exit 0\n")
    (tmp_path / "reward.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        multichallenge_grade.run_source(tmp_path, tmp_path, 5)
