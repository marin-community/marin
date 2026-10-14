# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Ultra blend components become tasks with their NeMo Gym agent's grader, or none."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationTrace,
    FinalAction,
    GradingAttempt,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grades_in_process,
    verifyit_spec,
)
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import CheckStatus, ImportFailureKind, ImportRejection, NormalizedTask, Reply
from taskcompendium.pipeline.sources import SourceShard, prepared_component_path, staged_raw_file_rows
from taskcompendium.runtime.task_grading import grade_task
from verifyit.modes.extract import extract_markdown_answer
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import JudgeSpec
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.datasets.nemotron_ultra.components import (
    PLACEHOLDER_INPUTS,
    SKYWORK,
    SWE_GYM,
    pipeline_name,
    prepare_blend,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import (
    CODE_CONTROLS,
    DAPO,
    MCQA_CONTROLS,
    RDKIT_CONTROLS,
    REASONING_GYM_CONTROLS,
)
from experiments.post_training.task_curation.pipeline import source_files
from experiments.post_training.task_curation.tests.conversion import (
    convert_row,
    converted_task,
)
from experiments.post_training.task_curation.tests.review_server import JudgeReply
from experiments.post_training.task_curation.tests.ultra_rows import (
    COMPONENT_ROWS,
    DAPO_QUESTION,
    DAPO_ROW,
    RECIPES,
    SHELL_TOOL,
    SWE_GYM_INSTANCE,
    SWE_REBENCH_INSTANCE,
    UNPARSEABLE_MATH_ANSWER,
    swe_row,
    ultra_row,
)


@pytest.mark.parametrize("max_workers", [1, 3])
def test_blend_preparation_preserves_nested_records_and_native_indices(tmp_path, max_workers):
    rows = [
        {"dataset": "fixture/one", "payload": {"values": [1, "é", None]}},
        {"agent_ref": {"name": "fixture_agent"}, "tools": [{"name": "read", "arguments": {"path": "/app"}}]},
        {"dataset": "fixture/one", "payload": {"text": "x" * 34_000_000}},
    ]
    source = tmp_path / "blend.jsonl"
    source.write_text("\n\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    with ZephyrContext(
        client=LocalClient(), max_workers=max_workers, chunk_storage_prefix=str(tmp_path / "chunks")
    ) as context:
        manifest = prepare_blend(str(source), context)
    assert manifest == {"source_rows": 3, "components": {"fixture/one": 2, "agent:fixture_agent": 1}}
    records = [
        record
        for component in manifest["components"]
        for record in load_parquet(str(prepared_component_path(StoragePath(str(source)), component)))
    ]
    assert {record["source_index"]: json.loads(record["raw_json"]) for record in records} == dict(enumerate(rows))
    for component in manifest["components"]:
        parquet = pq.ParquetFile(str(prepared_component_path(StoragePath(str(source)), component)))
        for index in range(parquet.num_row_groups):
            assert set(parquet.read_row_group(index, columns=["component"]).column("component").to_pylist()) == {
                component
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


def test_reasoning_gym_rejects_confirmed_invalid_tsumego_ground_truth():
    path = "ultra_sft_step3200_reasoning_gym"
    result = convert_row(
        RECIPES[pipeline_name("rlvr2", path)],
        {**COMPONENT_ROWS[path], "uuid": "3fb53469-5bbd-4b76-b449-c0414766f760"},
    )
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (
        ImportFailureKind.SOURCE_DEFECT,
        "invalid_reasoning_gym_ground_truth",
    )


def test_competitive_code_rejects_constructive_problem_without_its_special_judge():
    path = "ultra_sft_step3200_comp_coding"
    result = convert_row(
        RECIPES[pipeline_name("rlvr2", path)],
        {**COMPONENT_ROWS[path], "uuid": "2a48b444-3b6f-498c-ba55-b9d6be73a0ed"},
    )
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (
        ImportFailureKind.SOURCE_DEFECT,
        "missing_constructive_problem_judge",
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            "Reasoning.\n\n## Answer\n\n**The maximum area is 9.**\n\n---\n\nExtra algorithm notes.",
            "The maximum area is 9.",
        ),
        ("Reasoning.\n\n## Answer\n\nThe answer is not 9.", "The answer is not 9."),
        (
            "## Answer\nThe answer is 41; the provided target was 42.",
            "The answer is 41; the provided target was 42.",
        ),
    ],
)
def test_reasoning_gym_keeps_the_explicit_markdown_answer_section(response, expected):
    assert extract_markdown_answer(response) == expected


def test_structured_output_repairs_repeated_required_fields():
    path = "ultra_sft_step3200_structured_outputs_v2"
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count", "count"],
    }
    result = convert_row(
        RECIPES[pipeline_name("rlvr2", path)],
        {**COMPONENT_ROWS[path], "schema_str": json.dumps(schema)},
    )
    assert isinstance(result, NormalizedTask)
    assert json.loads(grader_config(result.task)["contract"]["schema_str"])["required"] == ["count"]
    assert [(change.field, change.reason) for change in result.changes] == [
        ("schema_str", "Remove repeated names from JSON Schema required lists")
    ]


def test_structured_output_does_not_rewrite_schema_data():
    path = "ultra_sft_step3200_structured_outputs_v2"
    schema = {"const": {"required": ["x", "x"]}}
    result = convert_row(
        RECIPES[pipeline_name("rlvr2", path)],
        {**COMPONENT_ROWS[path], "schema_str": json.dumps(schema)},
    )
    assert isinstance(result, NormalizedTask)
    assert json.loads(grader_config(result.task)["contract"]["schema_str"]) == schema
    assert result.changes == ()


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


def test_tool_action_conversion_removes_identical_duplicate_public_tools(staged):
    path = "ultra_sft_step3200_toolcall_schema"
    row = COMPONENT_ROWS[path]
    request = row["responses_create_params"]
    tools = request["tools"]
    duplicate = {**row, "responses_create_params": {**request, "tools": [*tools, tools[0]]}}
    result = convert_row(RECIPES[pipeline_name("rlvr2", path)], duplicate, inputs=staged)
    assert isinstance(result, NormalizedTask)
    assert [tool.name for tool in result.task.final_tools] == [tool["name"] for tool in tools]
    assert [(change.field, change.replacement) for change in result.changes if change.field.startswith("tools")] == [
        (f"tools[{len(tools)}]", "Removed")
    ]

    task = task_for(path, staged)
    assert task.answer_type == AnswerType.NATIVE_ACTION
    assert task.answer_format == FinalAction(max_calls=1)
    assert [tool.name for tool in task.final_tools] == [tool["name"] for tool in tools]
    assert any(isinstance(event, AssistantToolCalls) for event in task.context.events)
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert not spec.exact_gate_answers
    assert task.resources.verifier == ()
    assert row["expected_action"]["arguments"] not in json.dumps(spec.references)


def test_tool_message_action_uses_semantic_grading(staged):
    content = "Please tell me which company you want me to look up."
    task = task_for(
        "ultra_sft_step3200_toolcall_schema",
        staged,
        expected_action={"type": "message", "content": content},
    )
    assert task.answer_type == AnswerType.NATIVE_ACTION
    assert task.answer_format == FinalAction(max_calls=1)
    assert isinstance(task.grader, VerifyitGrader)
    assert isinstance(verifyit_spec(task.grader), JudgeSpec)

    row = COMPONENT_ROWS["ultra_sft_step3200_toolcall_schema"]
    request = row["responses_create_params"]
    no_tools = {
        **row,
        "expected_action": {"type": "message", "content": content},
        "responses_create_params": {**request, "tools": []},
    }
    text_task = converted_task(
        RECIPES[pipeline_name("rlvr2", "ultra_sft_step3200_toolcall_schema")], no_tools, inputs=staged
    )
    assert text_task.answer_type == AnswerType.TEXT


def test_toolcall_schema_ignores_broken_legacy_target(staged):
    task = task_for("ultra_sft_step3200_toolcall_schema", staged, expected_action={"type": "broken"})
    assert task.answer_type == AnswerType.NATIVE_ACTION
    assert isinstance(task.grader, VerifyitGrader)
    assert isinstance(verifyit_spec(task.grader), JudgeSpec)
    assert task.resources.verifier == ()
    assert '"expected_action"' not in task.model_dump_json()


def test_toolcall_schema_legacy_targets_do_not_create_conflicting_references(staged):
    original = task_for("ultra_sft_step3200_toolcall_schema", staged)
    alternative = task_for(
        "ultra_sft_step3200_toolcall_schema",
        staged,
        expected_action={"type": "message", "content": "Please clarify the location."},
    )

    assert deduplication_key(original) == deduplication_key(alternative)
    assert semantic_digest(original, include_reference=True) == semantic_digest(alternative, include_reference=True)


def test_reasoning_gym_golden_is_the_rows_answer(staged):
    assert REASONING_GYM_CONTROLS.golden is not None
    assert reply_text(REASONING_GYM_CONTROLS.golden(task_for("ultra_sft_step3200_reasoning_gym", staged))) == "42"
    unanswered = task_for("ultra_sft_step3200_reasoning_gym", staged, answer=None)
    assert REASONING_GYM_CONTROLS.golden(unanswered) is None


def text_grade(task: TaskSpec, reply: str) -> tuple[Outcome, float | None]:
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
    assert text_grade(task, equivalent) == (Outcome.GRADED, 1.0)
    assert text_grade(task, r"\boxed{\frac{\sqrt[5]{2}-1}{\sqrt[5]{2}+1}}") == (Outcome.GRADED, 0.0)


def test_tool_math_rows_keep_the_python_tool_with_its_agent(staged):
    task = task_for("ultra_sft_step3200_math_tir", staged)
    assert grades_in_process(task.grader)
    assert [tool.name for tool in task.interaction_tools] == ["stateful_python_code_exec"]
    provider = task.environment_requirements.tool_providers["nemotron_agent"]
    assert provider.action_interface == "nemotron-agent:ns_tools_simple_agent"


def test_abstention_idk_gets_source_partial_credit_after_reasoning_is_removed(staged):
    task = task_for("ultra_sft_step3200_abstention", staged)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    result = grade_judge_candidate(spec, r"<think>I cannot verify this.</think>\boxed{[IDK]}", runtime=None)
    assert (result.status, result.reward) == ("scored", 0.5)


def test_instruction_following_accepts_source_null_constraint_parameters(staged):
    task = task_for("ultra_sft_step3200_instruction_following", staged, kwargs=[None])
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert spec.constraints[0].params == {}


def test_instruction_following_semantically_grades_approximate_language_check(staged):
    task = task_for(
        "ultra_sft_step3200_instruction_following",
        staged,
        instruction_id_list=["language:response_language"],
        kwargs=[{"language": "en"}],
    )
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, JudgeSpec)
    assert spec.constraints == ()


def test_language_mixing_removes_empty_interior_user_turns(staged):
    path = "language_mixing_hs3_ultra_genrm_fmt"
    row = COMPONENT_ROWS[path]
    request = row["responses_create_params"]
    input_messages = [
        {"role": "user", "content": "First request."},
        {"role": "assistant", "content": "First response."},
        {"role": "user", "content": ""},
        *request["input"],
    ]
    result = convert_row(
        RECIPES[pipeline_name("rlvr2", path)],
        {**row, "responses_create_params": {**request, "input": input_messages}},
        inputs=staged,
    )
    assert isinstance(result, NormalizedTask)
    assert all(event.content for event in result.task.context.events if isinstance(event, TextMessage))
    assert [(change.field, change.replacement) for change in result.changes] == [("input[2]", "Removed")]


@pytest.mark.parametrize(
    ("path", "overrides", "replies", "expected"),
    [
        ("ultra_sft_step3200_instruction_following", {}, ["SCORE: 0"], CheckStatus.PASS),
        (
            "ultra_sft_step3200_instruction_following",
            {"instruction_id_list": ["startend:end_checker"], "kwargs": [{"end_phrase": "Sincerely."}]},
            [],
            CheckStatus.FAIL,
        ),
        ("ultra_sft_step3200_multichallenge_len40k", {}, ["SCORE: 0", "SCORE: 1"], CheckStatus.PASS),
    ],
    ids=["instruction-remote-verdict", "instruction-local-gate", "multichallenge-mixed-polarity"],
)
def test_judge_controls_require_remote_verdicts_and_score_the_declared_probe(
    staged, review_server, path, overrides, replies, expected
):
    review_server.judge_replies = [JudgeReply(reply) for reply in replies]
    task = task_for(path, staged, **overrides)
    controls = RECIPES[pipeline_name("rlvr2", path)].controls
    assert controls is not None
    report = run_controls(
        task,
        controls=controls,
        machines=None,
        judge_connection=JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture-key"),
    )
    assert [check.status for check in report.checks] == [expected]
    assert len(review_server.requests) == len(replies)


def test_tool_action_control_retries_a_truncated_judgment_without_accepting_its_partial_label(staged, review_server):
    task = task_for("ultra_sft_step3200_toolcall_schema", staged)
    controls = RECIPES[pipeline_name("rlvr2", "ultra_sft_step3200_toolcall_schema")].controls
    review_server.judge_replies = [JudgeReply("PASS", "length"), JudgeReply("FAIL")]
    report = run_controls(
        task,
        controls=controls,
        machines=None,
        judge_connection=JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture-key"),
    )
    assert [check.status for check in report.checks] == [CheckStatus.PASS]
    assert [request["max_completion_tokens"] for request in review_server.requests] == [8192, 16384]
