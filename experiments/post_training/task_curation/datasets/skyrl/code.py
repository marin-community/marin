# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL code and text-to-SQL sources, graded by their source scorers in the code/SQL image.

Each reply is scored by a ``*_grade.py`` script next to this module that calls the scorer the
image installs: the APPS evaluator for APPS, the SkyRL LiveCodeBench evaluator for Eurus-2 and
verifiable-coding problems, and the SkyRL text-to-SQL comparator for Gretel. The script receives
the hidden cases in ``config.json``; ``config.json`` also keeps the source's known solution, which
the golden control replays.
"""

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.code import has_code_block, validate_code_cases
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import SourceFormat, StagedInputs
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow, Reply
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.images import APPS_IMAGE, SKYRL_CODE_SQL_IMAGE
from experiments.post_training.task_curation.pipeline import HfSource, Image, RlDataPipeline, ShellSim

APPS_GRADE = "apps_grade.py"
LCB_GRADE = "lcb_grade.py"
SQL_GRADE = "sql_grade.py"
GRADE_SCRIPTS = {name: Path(__file__).with_name(name).read_bytes() for name in (APPS_GRADE, LCB_GRADE, SQL_GRADE)}
APPS_TESTING_UTIL_PATH = "/opt/apps/eval/testing_util.py"
APPS_TESTING_UTIL_SHA256 = "9a4e58ff2634ef606c42597457c0733910862e0588ea750d901265bbfe65d36f"
SKYRL_GYM_ROOT = "/opt/skyrl_gym"
ANSWER_PATH = "/app/answer.txt"
SCORE_PATH = "/logs/verifier/score.json"
GRADER_TIMEOUT = 330.0
# The LiveCodeBench child process may use 4 GiB; the grading machine also needs room for its runtime.
GRADER_MEMORY_MB = 5120
THREAD_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
CODE_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"
SQL_INSTRUCTION = (
    "\nTarget dialect is SQLite. Write exactly one SELECT statement (a leading WITH is allowed) that "
    "answers the question. Return only the query, inside <solution></solution>."
)
FAILING_CODE = '```python\nraise RuntimeError("negative control")\n```'
FAILING_SQL = "SELECT FROM"
NONDETERMINISTIC_SQL = re.compile(
    r"\b(?:random|randomblob|current_date|current_time|current_timestamp)\b", re.IGNORECASE
)

APPS_RUBRIC = """
Check the public problem and starter code against every hidden test. Multiple valid outputs and permissive source
comparisons must not be replaced by exact text matching.

Identify missing public context separately from implementation difficulty.
"""

EURUS2_CODE_RUBRIC = """
Require ability=code. Preserve the source reward_model ground truth and every prompt message; hidden function tests and
source scorer requirements stay hidden from the solver.

Identify missing public context separately from implementation difficulty.
"""

VERIFIABLE_CODE_RUBRIC = """
Check the public problem_statement against every hidden verification_info test. The gold_standard_solution is reference
evidence. Multiple valid outputs and permissive source comparisons must not be replaced by exact text matching.

Identify missing public context separately from implementation difficulty.
"""

GRETEL_TEXT_TO_SQL_RUBRIC = """
Evaluate the requested ordering, join keys and grouping against the public database fixtures. Reference SQL can be
defective; do not infer a dialect or demand textual equality.

Identify missing public context separately from implementation difficulty.
"""


def python_reply(solution: Any) -> str | None:
    """A source solution as a reply the scorers extract code from, or ``None`` when there is none."""
    if not isinstance(solution, str) or not solution.strip():
        return None
    return solution if has_code_block(solution) else f"```python\n{solution}\n```"


def _with_instruction(events: Sequence[TextMessage], instruction: str) -> tuple[TextMessage, ...]:
    """Append the answer instruction to the last user message."""
    last_user = max(index for index, event in enumerate(events) if event.role == "user")
    return tuple(
        event.model_copy(update={"content": event.content + instruction}) if index == last_user else event
        for index, event in enumerate(events)
    )


def scored_task(
    row: RawRow,
    *,
    events: Sequence[TextMessage],
    instruction: str,
    script: str,
    scorer: str,
    image: Image,
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> TaskSpec:
    """A conversation task whose reply ``script`` grades by calling the image-installed ``scorer``."""
    package = GraderPackage(
        ScriptGrader(
            argv=("python3", f"/tests/{script}", scorer, "/tests/config.json", ANSWER_PATH, SCORE_PATH),
            cwd="/",
            env=THREAD_ENVIRONMENT,
            environment=image.requirements(),
            answer_path=ANSWER_PATH,
            reward=FileReward(files=(RewardFile(path=SCORE_PATH, format=RewardFileFormat.JSON),)),
            timeout=GRADER_TIMEOUT,
        ),
        (
            inline_resource(script, GRADE_SCRIPTS[script]),
            inline_resource("config.json", json.dumps(dict(config), allow_nan=False).encode()),
        ),
    )
    return conversation_task(row, events=_with_instruction(events, instruction), package=package, evidence=evidence)


def _apps_reference(solutions: Any) -> str | None:
    """The first nonempty solution of APPS's JSON-encoded solution list."""
    try:
        decoded = json.loads(solutions) if isinstance(solutions, str) else solutions
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list):
        return None
    return next((reply for reply in map(python_reply, decoded) if reply is not None), None)


def convert_apps(row: RawRow) -> TaskSpec | ImportRejection:
    question, encoded = row.data.get("question"), row.data.get("input_output")
    if not isinstance(question, str) or not isinstance(encoded, str):
        return source_defect("missing_prompt_or_tests", "question and input_output JSON are required")
    try:
        tests = json.loads(encoded)
    except json.JSONDecodeError:
        return source_defect("invalid_test_contract", "input_output is not valid JSON")
    inputs, outputs = (tests.get("inputs"), tests.get("outputs")) if isinstance(tests, dict) else (None, None)
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return source_defect("invalid_test_contract", "Paired nonempty inputs and outputs are required")
    starter = row.data.get("starter_code")
    prompt = question + (f"\n\nStarter code:\n{starter}" if starter else "")
    return scored_task(
        row,
        events=(TextMessage(role="user", content=prompt),),
        instruction=CODE_INSTRUCTION,
        script=APPS_GRADE,
        scorer=APPS_TESTING_UTIL_PATH,
        image=APPS_IMAGE,
        config={
            "apps_source_sha256": APPS_TESTING_UTIL_SHA256,
            "input_output": encoded,
            "reference_reply": _apps_reference(row.data.get("solutions")),
        },
        evidence={
            key: row.data[key] for key in ("solutions", "difficulty", "url", "id", "problem_id") if key in row.data
        },
    )


def _lcb_task(
    row: RawRow, events: Sequence[TextMessage], test_cases: Any, reference: str | None, evidence: Mapping[str, Any]
) -> TaskSpec | ImportRejection:
    try:
        validate_code_cases(test_cases)
    except (ValueError, TypeError) as error:
        return unsupported("unsupported_test_cases", str(error))
    return scored_task(
        row,
        events=events,
        instruction=CODE_INSTRUCTION,
        script=LCB_GRADE,
        scorer=SKYRL_GYM_ROOT,
        image=SKYRL_CODE_SQL_IMAGE,
        config={"test_cases": test_cases, "reference_reply": reference},
        evidence=evidence,
    )


def is_code_row(row: dict[str, Any], _inputs: StagedInputs) -> bool:
    return row["ability"] == "code"


def convert_eurus2_code(row: RawRow) -> TaskSpec | ImportRejection:
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or not reward.get("ground_truth"):
        return source_defect("missing_prompt_or_tests", "prompt and reward_model ground_truth are required")
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("extra_info", "data_source", "ability") if key in row.data}
    # Eurus-2 publishes no solutions, so these tasks have no golden control.
    return _lcb_task(row, events, reward["ground_truth"], None, evidence)


def convert_verifiable_code(row: RawRow) -> TaskSpec | ImportRejection:
    problem, verification = row.data.get("problem_statement"), row.data.get("verification_info")
    if not isinstance(problem, str) or not isinstance(verification, dict) or not verification.get("test_cases"):
        return source_defect(
            "missing_prompt_or_tests", "problem_statement and verification_info test_cases are required"
        )
    solution = row.data.get("gold_standard_solution")
    evidence = {
        key: row.data[key]
        for key in ("gold_standard_solution", "metadata", "source", "task_type", "problem_id", "in_source_id")
        if key in row.data
    }
    return _lcb_task(row, (TextMessage(role="user", content=problem),), verification, python_reply(solution), evidence)


def validate_seeded_reference(context: str, reference: str) -> None:
    """Reject a reference that is not one deterministic SELECT running on a seeded SQLite database."""
    if re.match(r"\s*(?:select|with)\b", reference, re.IGNORECASE) is None or ";" in reference.strip().rstrip(";"):
        raise ValueError("Reference must be one SELECT")
    if NONDETERMINISTIC_SQL.search(reference):
        raise ValueError("Reference is nondeterministic")
    if not re.search(r"\bcreate\s+(?:temp(?:orary)?\s+)?table\b", context, re.IGNORECASE):
        raise ValueError("Context has no CREATE TABLE")
    if not re.search(r"\binsert\s+(?:into|or)\b", context, re.IGNORECASE):
        raise ValueError("Context has no INSERT")
    with sqlite3.connect(":memory:") as database:
        try:
            database.executescript(context)
            database.execute(reference).fetchone()
        except sqlite3.Error as error:
            raise ValueError("Seeded reference cannot execute") from error


def convert_gretel_text_to_sql(row: RawRow) -> TaskSpec | ImportRejection:
    question, context, reference = (row.data.get(key) for key in ("sql_prompt", "sql_context", "sql"))
    if not all(isinstance(value, str) and value.strip() for value in (question, context, reference)):
        return source_defect("missing_prompt_or_reference", "SQL prompt, context and reference are required")
    assert isinstance(context, str) and isinstance(reference, str)
    try:
        validate_seeded_reference(context, reference)
    except ValueError as error:
        return unsupported("unsupported_sql_context", str(error))
    return scored_task(
        row,
        events=(TextMessage(role="user", content=f"{question}\n\nDatabase context:\n{context}"),),
        instruction=SQL_INSTRUCTION,
        script=SQL_GRADE,
        scorer=SKYRL_GYM_ROOT,
        image=SKYRL_CODE_SQL_IMAGE,
        config={
            "reference_sql": reference,
            "context_sql": context,
            "reference_reply": f"<solution>{reference}</solution>",
        },
        evidence={
            key: row.data[key] for key in ("sql_explanation", "sql_complexity", "sql_task_type", "id") if key in row.data
        },
    )


def reference_solution(task: TaskSpec) -> Reply | None:
    """The source's known solution, when it has one."""
    reply = grader_config(task)["reference_reply"]
    return answer_reply(task, reply) if reply is not None else None


def failing_code(task: TaskSpec) -> Reply:
    return answer_reply(task, FAILING_CODE)


def failing_sql(task: TaskSpec) -> Reply:
    return answer_reply(task, FAILING_SQL)


CODE_CONTROLS = Controls(golden=reference_solution, negative=failing_code, memory_mb=GRADER_MEMORY_MB)
SQL_CONTROLS = Controls(golden=reference_solution, negative=failing_sql, memory_mb=GRADER_MEMORY_MB)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="apps",
            source=HfSource(
                "codeparrot/apps", "21e74ddf8de1a21436da12e3e653065c5213e9d1", ("train.jsonl",), SourceFormat.JSONL
            ),
            convert=convert_apps,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=APPS_RUBRIC,
            controls=CODE_CONTROLS,
            atlas_id="MarinSkyRL:apps",
        ),
        RlDataPipeline(
            name="eurus2_code",
            source=HfSource(
                "PRIME-RL/Eurus-2-RL-Data",
                "9776b13264b5aaa0b16495fcf086a0a8d86fd655",
                ("train.parquet",),
                SourceFormat.PARQUET,
                select=is_code_row,
            ),
            convert=convert_eurus2_code,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=EURUS2_CODE_RUBRIC,
            controls=CODE_CONTROLS,
            atlas_id="MarinSkyRL:eurus2_code",
        ),
        RlDataPipeline(
            name="verifiable_code",
            source=HfSource(
                "open-r1/verifiable-coding-problems-python",
                "b761a24a95fa03289a231d2d31c183636ffb9833",
                ("data/train-*.parquet",),
                SourceFormat.PARQUET,
            ),
            convert=convert_verifiable_code,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=VERIFIABLE_CODE_RUBRIC,
            controls=CODE_CONTROLS,
            atlas_id="MarinSkyRL:verifiable_code",
        ),
        RlDataPipeline(
            name="gretel_text_to_sql",
            source=HfSource(
                "gretelai/synthetic_text_to_sql",
                "740ab236e64503fba51be1101df7a1be83bf455d",
                ("synthetic_text_to_sql_train.snappy.parquet",),
                SourceFormat.PARQUET,
            ),
            convert=convert_gretel_text_to_sql,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=GRETEL_TEXT_TO_SQL_RUBRIC,
            controls=SQL_CONTROLS,
            atlas_id="MarinSkyRL:gretel_text_to_sql",
        ),
    ]
