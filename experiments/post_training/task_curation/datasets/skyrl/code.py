# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL code and text-to-SQL sources, graded by their vendored source scorers.

Each task ships a grade script from beside this module with the scorer it imports from
``scorers/``: the APPS evaluator for APPS, the SkyRL LiveCodeBench evaluator for Eurus-2 and
verifiable-coding problems, and the SkyRL text-to-SQL comparator for Gretel. Conversion prepares
LiveCodeBench cases and the text-to-SQL ground truth with the scorer's own preparation functions,
as SkyRL's dataset builders do, so a row the scorer cannot grade is rejected here rather than
failing at grading. ``config.json`` also keeps the source's known solution, which the golden
control replays.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.code import CODE_GRADER_MEMORY_MB, THREAD_ENVIRONMENT, python_reply
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import grader_config
from taskcompendium.models import TaskResource, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow, Reply
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.skyrl.scorers import livecodebench, text_to_sql_scoring
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference

LCB_VERIFIER = SourceReference(
    "lcb",
    "a0fa569b0d62eed8c1439904a57b619978f9f0c53f8af732dcae0a971135d33d",
    (
        "https://github.com/marin-community/MarinSkyRL/tree/"
        "e44c4bfcb62c489286a1264094e6d9c883aaf0d2/skyrl-gym/skyrl_gym/envs/lcb"
    ),
)

HERE = Path(__file__).parent
SCORERS = HERE / "scorers"
APPS_GRADE = grade_script(HERE / "apps_grade.py", *shipped_files(SCORERS, "apps_testing_util.py"))
LCB_GRADE = grade_script(HERE / "lcb_grade.py", *shipped_files(SCORERS, "livecodebench.py"))
SQL_GRADE = grade_script(HERE / "sql_grade.py", *shipped_files(SCORERS, "text_to_sql_scoring.py"))
GRADER_TIMEOUT = 330.0
CODE_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"
SQL_INSTRUCTION = (
    "\nTarget dialect is SQLite. Write exactly one SELECT statement (a leading WITH is allowed) that "
    "answers the question. Return only the query, inside <solution></solution>."
)
GEEKSFORGEEKS_HOST = "geeksforgeeks.org"

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


def _with_instruction(events: Sequence[TextMessage], instruction: str) -> tuple[TextMessage, ...]:
    """Append the answer instruction to the last user message."""
    last_user = max(index for index, event in enumerate(events) if event.role == "user")
    return tuple(
        event.model_copy(update={"content": event.content + instruction}) if index == last_user else event
        for index, event in enumerate(events)
    )


def scored_task(
    row: RawRow,
    context: ConversionContext,
    *,
    events: Sequence[TextMessage],
    instruction: str,
    grade: tuple[TaskResource, ...],
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> TaskSpec:
    """A conversation task whose reply the ``grade`` script scores with its vendored scorer."""
    package = script_package(
        grade,
        config,
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=DEFAULT_OUTPUT,
        env=THREAD_ENVIRONMENT,
    )
    return conversation_task(row, events=_with_instruction(events, instruction), package=package, evidence=evidence)


def convert_apps(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
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
    # APPS leaves ``solutions`` empty when it has none; anything else must be a JSON list.
    try:
        solutions = json.loads(row.data.get("solutions") or "[]")
    except (json.JSONDecodeError, TypeError) as error:
        return source_defect("invalid_solutions", f"solutions is not JSON: {error}")
    if not isinstance(solutions, list):
        return source_defect("invalid_solutions", "solutions must be a JSON list")
    starter = row.data.get("starter_code")
    prompt = question + (f"\n\nStarter code:\n{starter}" if starter else "")
    return scored_task(
        row,
        context,
        events=(TextMessage(role="user", content=prompt),),
        instruction=CODE_INSTRUCTION,
        grade=APPS_GRADE,
        config={
            "input_output": tests,
            "reference_reply": next((reply for reply in map(python_reply, solutions) if reply is not None), None),
        },
        evidence={
            key: row.data[key] for key in ("solutions", "difficulty", "url", "id", "problem_id") if key in row.data
        },
    )


def lcb_test_cases(source_cases: Any) -> list[dict[str, Any]]:
    """The LiveCodeBench evaluator's canonical cases for a source's test layout.

    Raises ``ValueError`` or ``TypeError`` for a layout the evaluator cannot normalize, and
    ``json.JSONDecodeError`` for a functional case whose arguments or result are not JSON: the
    evaluator decodes those only when it runs a reply, and then fails every reply.
    """
    cases = json.loads(livecodebench.normalize_lcb_ground_truth(source_cases))
    if cases[0]["testtype"] == livecodebench.FUNCTIONAL_TEST_TYPE:
        for case in cases:
            for argument in case["input"].split("\n"):
                json.loads(argument)
            json.loads(case["output"])
    return cases


def _lcb_task(
    row: RawRow,
    context: ConversionContext,
    events: Sequence[TextMessage],
    source_cases: Any,
    reference: str | None,
    evidence: Mapping[str, Any],
) -> TaskSpec | ImportRejection:
    try:
        cases = lcb_test_cases(source_cases)
    except json.JSONDecodeError as error:
        return source_defect("invalid_test_cases", f"Functional test case is not JSON: {error}")
    except (ValueError, TypeError) as error:
        return unsupported("unsupported_test_cases", str(error))
    return scored_task(
        row,
        context,
        events=events,
        instruction=CODE_INSTRUCTION,
        grade=LCB_GRADE,
        config={"test_cases": cases, "reference_reply": reference},
        evidence=evidence,
    )


def is_code_row(row: dict[str, Any], _context: ConversionContext) -> bool:
    return row["ability"] == "code"


def convert_eurus2_code(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    ground_truth = reward.get("ground_truth") if isinstance(reward, dict) else None
    if not isinstance(messages, list) or not messages or not isinstance(ground_truth, str) or not ground_truth:
        return source_defect("missing_prompt_or_tests", "prompt and reward_model ground_truth are required")
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("extra_info", "data_source", "ability") if key in row.data}
    # Eurus-2 publishes no solutions, so these tasks have no golden control.
    return _lcb_task(row, context, events, ground_truth, None, evidence)


def convert_verifiable_code(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    problem, verification = row.data.get("problem_statement"), row.data.get("verification_info")
    if not isinstance(problem, str) or not isinstance(verification, dict) or not verification.get("test_cases"):
        return source_defect(
            "missing_prompt_or_tests", "problem_statement and verification_info test_cases are required"
        )
    # GeeksforGeeks prompts ask the solver to complete a function and not to read input, but the
    # hidden stdin tests are the statement's display examples, such as ``N = 9, K = 4``. No reply
    # that follows the prompt can pass them.
    metadata = row.data.get("metadata") or {}
    if GEEKSFORGEEKS_HOST in (metadata.get("problem_url") or ""):
        return source_defect(
            "function_template_with_example_tests",
            "GeeksforGeeks rows ask for a function body; their stdin tests are the statement's display examples",
        )
    solution = row.data.get("gold_standard_solution")
    evidence = {
        key: row.data[key]
        for key in ("gold_standard_solution", "metadata", "source", "task_type", "problem_id", "in_source_id")
        if key in row.data
    }
    return _lcb_task(
        row, context, (TextMessage(role="user", content=problem),), verification, python_reply(solution), evidence
    )


def sql_ground_truth(sql_context: str, reference: str) -> dict[str, Any]:
    """The text-to-SQL comparator's ground truth for a seeded context and its reference query.

    Raises ``ValueError`` unless the context holds only CREATE TABLE and INSERT statements that
    load in SQLite and the reference is one deterministic, read-only query that matches itself on
    the seeded and perturbed databases.
    """
    statements: dict[str, list[str]] = {"create_table": [], "insert": []}
    for statement in text_to_sql_scoring.split_statements(sql_context):
        kind = text_to_sql_scoring.classify_statement(statement)
        if kind not in statements:
            raise ValueError(f"Context has a {kind} statement")
        statements[kind].append(statement)
    ground_truth = json.loads(
        text_to_sql_scoring.normalize_ground_truth(
            {
                "schema_sql": ";\n".join(statements["create_table"]),
                "insert_sql": ";\n".join(statements["insert"]),
                "reference_sql": reference,
                "order_significant": text_to_sql_scoring.has_top_level_order_by(reference),
            }
        )
    )
    outcome, detail = text_to_sql_scoring.grade(ground_truth, reference)
    if outcome != text_to_sql_scoring.GradeOutcome.MATCH:
        raise ValueError(f"Reference does not grade as correct: {detail}")
    return ground_truth


def convert_gretel_text_to_sql(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    question, sql_context, reference = (row.data.get(key) for key in ("sql_prompt", "sql_context", "sql"))
    if not all(isinstance(value, str) and value.strip() for value in (question, sql_context, reference)):
        return source_defect("missing_prompt_or_reference", "SQL prompt, context and reference are required")
    assert isinstance(sql_context, str) and isinstance(reference, str)
    try:
        ground_truth = sql_ground_truth(sql_context, reference)
    except ValueError as error:
        return unsupported("unsupported_sql_context", str(error))
    return scored_task(
        row,
        context,
        events=(TextMessage(role="user", content=f"{question}\n\nDatabase context:\n{sql_context}"),),
        instruction=SQL_INSTRUCTION,
        grade=SQL_GRADE,
        config={"ground_truth": ground_truth, "reference_reply": f"<solution>{reference}</solution>"},
        evidence={
            key: row.data[key] for key in ("sql_explanation", "sql_complexity", "sql_task_type", "id") if key in row.data
        },
    )


def reference_solution(task: TaskSpec) -> Reply | None:
    """The source's known solution, when it has one."""
    reply = grader_config(task)["reference_reply"]
    return answer_reply(task, reply) if reply is not None else None


CONTROLS = Controls(golden=reference_solution, memory_mb=CODE_GRADER_MEMORY_MB)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="MarinSkyRL:apps",
                title="codeparrot/apps",
                origin="MarinSkyRL",
                family="competitive-programming",
                tags=("rlvr", "single-turn", "benchmark", "license:mit", "gym/lcb"),
                verifier=LCB_VERIFIER,
            ),
            config=CurationRecipe(
                name="apps",
                source=HfSource(
                    "codeparrot/apps", "21e74ddf8de1a21436da12e3e653065c5213e9d1", ("train.jsonl",), SourceFormat.JSONL
                ),
                convert=convert_apps,
                version="2",
                intended_use=IntendedUse.TRAIN,
                rubric=APPS_RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="MarinSkyRL:eurus2_code",
                title="PRIME-RL/Eurus-2-RL-Data · code",
                origin="MarinSkyRL",
                family="competitive-programming",
                tags=("rlvr", "single-turn", "license:mit", "gym/lcb"),
                verifier=LCB_VERIFIER,
            ),
            config=CurationRecipe(
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
                intended_use=IntendedUse.TRAIN,
                rubric=EURUS2_CODE_RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="MarinSkyRL:verifiable_code",
                title="open-r1/verifiable-coding-problems-python",
                origin="MarinSkyRL",
                family="competitive-programming",
                tags=("rlvr", "single-turn", "gym/lcb"),
                verifier=LCB_VERIFIER,
            ),
            config=CurationRecipe(
                name="verifiable_code",
                source=HfSource(
                    "open-r1/verifiable-coding-problems-python",
                    "b761a24a95fa03289a231d2d31c183636ffb9833",
                    ("data/train-*.parquet",),
                    SourceFormat.PARQUET,
                ),
                convert=convert_verifiable_code,
                version="2",
                intended_use=IntendedUse.TRAIN,
                rubric=VERIFIABLE_CODE_RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="MarinSkyRL:gretel_text_to_sql",
                title="gretelai/synthetic_text_to_sql",
                origin="MarinSkyRL",
                family="text-to-sql",
                tags=("rlvr", "single-turn", "license:apache-2.0", "gym/text_to_sql"),
                verifier=SourceReference(
                    "text_to_sql",
                    "1d8cec1db5f4d68cb643344f8c777f73d197656040652ff2c7a588dd88ce7ed5",
                    (
                        "https://github.com/marin-community/MarinSkyRL/tree/"
                        "e44c4bfcb62c489286a1264094e6d9c883aaf0d2/skyrl-gym/skyrl_gym/envs/text_to_sql"
                    ),
                ),
            ),
            config=CurationRecipe(
                name="gretel_text_to_sql",
                source=HfSource(
                    "gretelai/synthetic_text_to_sql",
                    "740ab236e64503fba51be1101df7a1be83bf455d",
                    ("synthetic_text_to_sql_train.snappy.parquet",),
                    SourceFormat.PARQUET,
                ),
                convert=convert_gretel_text_to_sql,
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=GRETEL_TEXT_TO_SQL_RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
    ]
