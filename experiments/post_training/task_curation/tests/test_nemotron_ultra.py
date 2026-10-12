# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Ultra blend components become tasks with their NeMo Gym agent's grader, or none."""

import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    GradingAttempt,
    NoGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grades_in_process,
)
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import CheckStatus, ImportFailureKind, ImportRejection, NormalizedTask, Reply
from taskcompendium.pipeline.sources import SourceShard, staged_raw_file_rows
from taskcompendium.runtime.task_grading import grade_task

from experiments.post_training.task_curation.datasets.nemotron_ultra.components import (
    PLACEHOLDER_INPUTS,
    SKYWORK,
    SWE_GYM,
    pipeline_name,
    sources,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import (
    CODE_CONTROLS,
    DAPO,
    MCQA_CONTROLS,
    RDKIT_CONTROLS,
    REASONING_GYM_CONTROLS,
    TOOL_ACTION_CONTROLS,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, source_files
from experiments.post_training.task_curation.tests.conversion import (
    convert_row,
    converted_task,
)

FIXTURES = Path(__file__).parent / "fixtures/nemotron_ultra"
RECIPES = {source.name: cast(CurationRecipe, source.config) for source in sources()}
SWE_GYM_INSTANCE = "gym-1"
SWE_REBENCH_INSTANCE = "rebench-1"
DAPO_QUESTION = "What is 2 + 3?"
DAPO_ROW = {"prompt": [{"content": DAPO_QUESTION}], "reward_model": {"ground_truth": "5"}}
SHELL_TOOL = {
    "type": "function",
    "name": "shell_command",
    "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}
PYTHON_TOOL = {
    "type": "function",
    "name": "stateful_python_code_exec",
    "description": "Call this function to execute Python code in a stateful Jupyter notebook environment.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Code to execute"}},
        "required": ["code"],
    },
    "strict": True,
}
REPORT_TOOL = {
    "type": "function",
    "name": "report",
    "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]},
}
MATH_QUESTION = (
    "Find all real solutions to the equation $x^5 - 15x^4 + 10x^3 - 30x^2 + 5x - 3 = 0$. "
    "Express the answer using \\boxed{}."
)
MATH_ANSWER = r"\( \frac{2^{\frac{1}{5}} + 1}{2^{\frac{1}{5}} - 1} \)"
"""A sampled math_cot row's reference, delimiters included, as the blend stores it."""
UNPARSEABLE_MATH_ANSWER = (
    "\\text{No \u2013 the limit need not exist; the matrices can stay bounded\nbut fail to converge.}"
)
"""A sampled reference that only the source's LLM judge could compare."""
COUNT_SCHEMA = json.dumps({"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]})
GRID = [[1, 2], [3, 4]]


def ultra_row(component: str, agent: str, prompt: str = "Answer the question.", tools=(), **fields) -> dict:
    """A blend row; agent-keyed components carry no ``dataset`` field."""
    request: dict = {"input": [{"role": "user", "content": prompt}]}
    if tools:
        request["tools"] = list(tools)
    row = {"agent_ref": {"type": "responses_api_agents", "name": agent}, "responses_create_params": request, **fields}
    if not component.startswith("agent:"):
        row["dataset"] = component
    return row


def fixture_row(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def swe_row(component: str, instance: str) -> dict:
    return ultra_row(
        component,
        "swe_pivot_agent",
        "Fix the failing test in the repository.",
        tools=[SHELL_TOOL],
        metadata={"instance_id": instance},
        environment={"image": "swe-env", "workdir": "/workspace/repo"},
    )


def format_row(component: str, agent: str) -> dict:
    verifier = {"type": "regex", "verify_regex": ["^- "], "verify_min_matches": 3}
    return ultra_row(component, agent, "List three fruits as bullets.", verifier=verifier)


def structured_row(component: str, agent: str) -> dict:
    return ultra_row(component, agent, "Return the count as JSON.", schema_str=COUNT_SCHEMA, schema_type="json")


def math_row(component: str, agent: str, prompt: str, answer: str, tools=()) -> dict:
    return ultra_row(
        component,
        agent,
        prompt,
        tools=tools,
        question=prompt,
        expected_answer=answer,
        verifier_type="math_with_judge",
        pass_rate=0.375,
    )


def mcqa_row(component: str, grading_mode: str) -> dict:
    return ultra_row(
        component,
        "mcqa_simple_agent",
        "What is 2 + 2? A) 3 B) 4",
        options=[{"A": "3"}, {"B": "4"}],
        expected_answer="B",
        grading_mode=grading_mode,
    )


COMPONENT_ROWS: dict[str, dict] = {
    "ultra_sft_step3200_calendar_v2": ultra_row(
        "ultra_sft_step3200_calendar_v2",
        "calendar_simple_agent",
        "Schedule event 0 after 10am.",
        exp_cal_state={
            "0": {"event_id": 0, "duration": 30, "constraint": "after 10am", "min_time": "09:00", "max_time": "12:00"}
        },
    ),
    "ultra_sft_step3200_comp_coding": ultra_row(
        "ultra_sft_step3200_comp_coding",
        "code_gen_simple_agent",
        "Read two integers and print their sum.",
        verifier_metadata={"unit_tests": {"inputs": ["1 2\n"], "outputs": ["3\n"]}},
        gold_standard_solution="a, b = map(int, input().split())\nprint(a + b)",
    ),
    "ultra_sft_step3200_ds2_freeform": format_row("ultra_sft_step3200_ds2_freeform", "freeform_formatting_simple_agent"),
    "ultra_sft_step3200_instruction_following": ultra_row(
        "ultra_sft_step3200_instruction_following",
        "instruction_following_simple_agent",
        "Describe tea without commas.",
        instruction_id_list=["punctuation:no_comma"],
        kwargs=[{}],
    ),
    "ultra_sft_step3200_jailbreak": ultra_row(
        "ultra_sft_step3200_jailbreak", "safety_agent", "Explain safe handling.", response_policy_mapped="helpful"
    ),
    "ultra_sft_step3200_math_cot": math_row(
        "ultra_sft_step3200_math_cot", "math_with_judge_simple_agent", MATH_QUESTION, MATH_ANSWER
    ),
    "ultra_sft_step3200_math_tir": math_row(
        "ultra_sft_step3200_math_tir",
        "ns_tools_simple_agent",
        "Compute 2 + 2 with Python. Your answer should be placed inside \\boxed{}.",
        "4",
        tools=[PYTHON_TOOL],
    ),
    "ultra_sft_step3200_nvarc_transductive": ultra_row(
        "ultra_sft_step3200_nvarc_transductive",
        "nvarc_transductive_simple_agent",
        "Give the output grid.",
        expected_output=GRID,
    ),
    "ultra_sft_step3200_rdkit": fixture_row("rdkit_mopd.json"),
    "ultra_sft_step3200_reasoning_gym": ultra_row(
        "ultra_sft_step3200_reasoning_gym",
        "reasoning_gym_simple_agent",
        "Solve x + 8 = 50.",
        question="Solve x + 8 = 50.",
        answer="42",
        metadata={"source_dataset": "simple_equations"},
    ),
    "ultra_sft_step3200_stem_mcqa": mcqa_row("ultra_sft_step3200_stem_mcqa", "strict_single_letter_boxed"),
    "ultra_sft_step3200_structured_outputs_v2": structured_row(
        "ultra_sft_step3200_structured_outputs_v2", "structured_outputs_simple_agent"
    ),
    "ultra_sft_step3200_tau_pivot": ultra_row(
        "ultra_sft_step3200_tau_pivot",
        "tau_agent",
        "Change my flight to Friday.",
        tools=[SHELL_TOOL],
        scenario={"user": "traveler"},
    ),
    "ultra_sft_step3200_toolcall_schema": fixture_row("toolcall_schema.json"),
    "ultra_v3_agentic_rl_step73_structured_outputs_v2": ultra_row(
        "ultra_v3_agentic_rl_step73_structured_outputs_v2",
        "structured_outputs_simple_agent",
        "Report the count.",
        tools=[REPORT_TOOL],
        schema_str=COUNT_SCHEMA,
        response_mode="tool_call",
    ),
}


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> dict[str, StoragePath]:
    """SWE-Gym membership and placeholder question files at their repo-relative paths."""
    root = tmp_path_factory.mktemp("inputs")
    tables = {
        SWE_GYM.repo: (SWE_GYM.files[0], [{"instance_id": SWE_GYM_INSTANCE}]),
        DAPO: (PLACEHOLDER_INPUTS[DAPO].files[0], [DAPO_ROW]),
        SKYWORK: (PLACEHOLDER_INPUTS[SKYWORK].files[0], [DAPO_ROW]),
    }
    inputs = {}
    for repo, (file, rows) in tables.items():
        path = root / repo / file
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        inputs[repo] = StoragePath(str(root / repo))
    return inputs


def test_swe_components_split_by_swe_gym_membership(tmp_path, staged):
    rows = [swe_row("swe_pivot_len40k", SWE_GYM_INSTANCE), swe_row("swe_pivot_len40k", SWE_REBENCH_INSTANCE)]
    rows.append(ultra_row("ultra_sft_step3200_jailbreak", "safety_agent"))
    (tmp_path / "mopd.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = {
        split: [
            record["locator"]
            for record in staged_raw_file_rows(
                str(tmp_path),
                SourceShard("mopd.jsonl", 0, 1, None),
                source_files(RECIPES[f"nemotron_ultra_mopd_swe_pivot_len40k_{split}"].source),
                ConversionContext(staged, None),
            )
        ]
        for split in ("swe_gym_swe_gym", "nebius_swe_rebench_v2")
    }
    assert selected == {"swe_gym_swe_gym": ["mopd.jsonl:0"], "nebius_swe_rebench_v2": ["mopd.jsonl:1"]}


@pytest.mark.parametrize(("ground_truth", "expected"), [("5", "5"), ('["5"]', "5"), ("{1, 2}", "{1, 2}"), (["7"], "7")])
def test_math_placeholder_restores_question_and_answer(tmp_path, ground_truth, expected):
    placeholder = tmp_path / DAPO / PLACEHOLDER_INPUTS[DAPO].files[0]
    placeholder.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"prompt": [{"content": DAPO_QUESTION}], "reward_model": {"ground_truth": ground_truth}}]),
        placeholder,
    )
    inputs = {DAPO: StoragePath(str(tmp_path / DAPO))}
    row = {
        **COMPONENT_ROWS["ultra_sft_step3200_math_cot"],
        "_hf_question_placeholder": {"dataset": DAPO, "split": "train", "row": 0, "mode": "canonical"},
    }
    result = convert_row(RECIPES["nemotron_ultra_rlvr1_ultra_sft_step3200_math_cot"], row, inputs=inputs)
    assert isinstance(result, NormalizedTask)
    assert result.task.context.events == (TextMessage(role="user", content=DAPO_QUESTION),)
    contract = grader_config(result.task)["contract"]
    assert contract["expected_answer"] == expected
    assert contract["placeholder_provenance"]["dataset"] == DAPO
    assert "_hf_question_placeholder" not in contract
    assert [change.field for change in result.changes] == ["question", "expected_answer", "grader"]


@pytest.mark.parametrize(
    ("path", "changes", "kind", "reason"),
    [
        (
            "ultra_sft_step3200_calendar_v2",
            {"agent_ref": {"name": "calendar_judge_agent"}},
            "unsupported",
            "unsupported_agent",
        ),
        (
            "ultra_sft_step3200_instruction_following",
            {"responses_create_params": {"input": [{"role": "user", "content": "Hi"}], "tools": [SHELL_TOOL]}},
            "unsupported",
            "unsupported_tool_request",
        ),
        (
            "ultra_sft_step3200_comp_coding",
            {"verifier_metadata": {"unit_tests": {"inputs": ["1\n"], "outputs": []}}},
            "source_defect",
            "invalid_code_tests",
        ),
        (
            "ultra_sft_step3200_structured_outputs_v2",
            {"schema_str": json.dumps({"type": "object", "required": "count"})},
            "source_defect",
            "invalid_structured_schema",
        ),
        (
            "ultra_sft_step3200_structured_outputs_v2",
            {"schema_type": "protobuf"},
            "unsupported",
            "unsupported_structured_schema_type",
        ),
        (
            "ultra_sft_step3200_structured_outputs_v2",
            {"response_mode": "tool_call"},
            "unsupported",
            "missing_structured_tool_schema",
        ),
        ("ultra_sft_step3200_rdkit", {"property_type": "similarity"}, "unsupported", "unsupported_chemistry_property"),
        ("ultra_sft_step3200_rdkit", {"expected_answer": "nan"}, "source_defect", "nonfinite_chemistry_target"),
        (
            "ultra_sft_step3200_toolcall_schema",
            {"expected_action": {"type": "computer_call"}},
            "unsupported",
            "unsupported_action_type",
        ),
        (
            "ultra_sft_step3200_nvarc_transductive",
            {"agent_ref": {"name": "other_agent"}},
            "unsupported",
            "unsupported_agent",
        ),
        (
            "ultra_sft_step3200_ds2_freeform",
            {"verifier": {"type": "llm_judge"}},
            "unsupported",
            "unsupported_format_verifier",
        ),
        (
            "ultra_sft_step3200_jailbreak",
            {"_hf_question_placeholder": {"dataset": DAPO, "split": "train", "row": 0}},
            "unsupported",
            "unresolved_external_placeholder",
        ),
        (
            "ultra_sft_step3200_math_cot",
            {"expected_answer": UNPARSEABLE_MATH_ANSWER},
            "unsupported",
            "unparseable_math_reference",
        ),
    ],
)
def test_converter_rejects_rows_its_grader_cannot_score(path, changes, kind, reason):
    name = pipeline_name("rlvr2", path)
    result = convert_row(RECIPES[name], {**COMPONENT_ROWS[path], **changes})
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind(kind), reason)


def task_for(path: str, staged, **changes) -> TaskSpec:
    return converted_task(RECIPES[pipeline_name("rlvr2", path)], {**COMPONENT_ROWS[path], **changes}, inputs=staged)


def reply_text(reply: object) -> str:
    assert isinstance(reply, Reply) and isinstance(reply.event, TextMessage)
    return reply.event.content


@pytest.mark.parametrize(
    ("changes", "golden"),
    [
        ({}, "\\boxed{B}"),
        ({"grading_mode": "lenient_answer_colon"}, "Answer: B"),
        ({"grading_mode": "lenient_answer_colon_md"}, "**Answer**: B"),
        ({"template_metadata": {"output_regex": r"Final: ([A-D])"}}, None),
        ({"expected_answer": "E"}, None),
    ],
)
def test_multiple_choice_golden_follows_the_grading_mode(staged, changes, golden):
    assert MCQA_CONTROLS.golden is not None
    reply = MCQA_CONTROLS.golden(task_for("ultra_sft_step3200_stem_mcqa", staged, **changes))
    assert (reply_text(reply) if reply is not None else None) == golden


@pytest.mark.parametrize(("box", "golden"), [(False, "((1))"), (True, "\\boxed{1}")])
def test_chemistry_golden_uses_the_rows_answer_wrapper(staged, box, golden):
    task = task_for("ultra_sft_step3200_rdkit", staged, use_box_format=box)
    assert RDKIT_CONTROLS.golden is not None
    assert reply_text(RDKIT_CONTROLS.golden(task)) == golden


def test_code_golden_submits_the_source_solution(staged):
    task = task_for("ultra_sft_step3200_comp_coding", staged)
    assert CODE_CONTROLS.golden is not None
    golden = reply_text(CODE_CONTROLS.golden(task))
    assert golden == "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    assert CODE_CONTROLS.golden(task_for("ultra_sft_step3200_comp_coding", staged, gold_standard_solution="")) is None


def test_tool_action_golden_calls_the_expected_tool(staged):
    task = task_for("ultra_sft_step3200_toolcall_schema", staged)
    assert TOOL_ACTION_CONTROLS.golden is not None
    expected = COMPONENT_ROWS["ultra_sft_step3200_toolcall_schema"]["expected_action"]
    call = ConversationToolCall(call_id="control", name=expected["name"], arguments=json.loads(expected["arguments"]))
    assert TOOL_ACTION_CONTROLS.golden(task) == Reply(AssistantToolCalls(calls=(call,)))


def test_reasoning_gym_golden_is_the_rows_answer(staged):
    assert REASONING_GYM_CONTROLS.golden is not None
    assert reply_text(REASONING_GYM_CONTROLS.golden(task_for("ultra_sft_step3200_reasoning_gym", staged))) == "42"
    unanswered = task_for("ultra_sft_step3200_reasoning_gym", staged, answer=None)
    assert REASONING_GYM_CONTROLS.golden(unanswered) is None


def math_grade(task: TaskSpec, reply: str) -> tuple[Outcome, float | None]:
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=reply)))
    result = grade_task(task, GradingAttempt(trace))
    return result.status, result.reward


@pytest.mark.parametrize("path", ["ultra_sft_step3200_math_cot", "ultra_sft_step3200_math_tir"])
def test_math_golden_scores_the_boxed_reference_one(staged, path):
    controls = RECIPES[pipeline_name("rlvr2", path)].controls
    assert controls is not None
    report = run_controls(task_for(path, staged), controls=controls, machines=None)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"golden": CheckStatus.PASS}, report


def test_math_grader_compares_the_answer_symbolically_with_the_expected_answer(staged):
    task = task_for("ultra_sft_step3200_math_cot", staged)
    assert isinstance(task.grader, VerifyitGrader) and task.grader.mode == "math"
    equivalent = r"So $x = \boxed{\frac{\sqrt[5]{2}+1}{\sqrt[5]{2}-1}}$."
    assert math_grade(task, equivalent) == (Outcome.GRADED, 1.0)
    assert math_grade(task, r"\boxed{\frac{\sqrt[5]{2}-1}{\sqrt[5]{2}+1}}") == (Outcome.GRADED, 0.0)


def test_tool_math_rows_keep_the_python_tool_with_its_agent(staged):
    task = task_for("ultra_sft_step3200_math_tir", staged)
    assert grades_in_process(task.grader)
    assert [tool.name for tool in task.interaction_tools] == ["stateful_python_code_exec"]
    provider = task.environment_requirements.tool_providers["nemotron_agent"]
    assert provider.action_interface == "nemotron-agent:ns_tools_simple_agent"


def test_ungraded_agent_components_keep_tools_and_initial_state_with_the_agent(staged):
    task = task_for("ultra_sft_step3200_tau_pivot", staged)
    assert isinstance(task.grader, NoGrader)
    assert [tool.name for tool in task.interaction_tools] == ["shell_command"]
    provider = task.environment_requirements.tool_providers["nemotron_agent"]
    assert provider.action_interface == "nemotron-agent:tau_agent"
    assert provider.initial_state == {"scenario": {"user": "traveler"}}
