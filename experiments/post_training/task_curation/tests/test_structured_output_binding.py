# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from functools import partial

import pytest
from shellbox.machine import DockerImage, MachineSpec
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    Source,
)
from taskcompendium.pipeline.models import CheckStatus, ImportFailureKind, ImportRejection, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import structured_output_binding

from .test_calendar_binding import SourceScoreMachine, SourceScoreMachines, grade, grader_image

SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
    "additionalProperties": False,
}


@dataclass
class DiagnosticMachine(SourceScoreMachine):
    diagnostic: dict = field(default_factory=dict)

    async def run(self, command):
        result = await super().run(command)
        if command.argv[0] == "python3":
            verdict = json.loads(self.files["/logs/verifier/score.json"])
            verdict["detail"] = self.diagnostic
            self.files["/logs/verifier/score.json"] = json.dumps(verdict).encode()
        return result


@dataclass
class DiagnosticMachines(SourceScoreMachines):
    diagnostic: dict = field(default_factory=dict)

    async def create(self, spec):
        machine = DiagnosticMachine(diagnostic=self.diagnostic)
        self.machines.append(machine)
        return machine


@pytest.fixture
def native_dependencies():
    pytest.importorskip("xmltodict")
    pytest.importorskip("openapi_schema_validator")


@pytest.fixture
def row():
    return RawRow(
        "structured",
        Source(dataset="fixture", revision="pinned", row="1", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "structured_outputs_simple_agent"},
            "responses_create_params": {"input": [{"role": "user", "content": "Return count as an integer."}]},
            "schema_str": json.dumps(SCHEMA),
        },
    )


def task_for(row):
    result = structured_output_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "1" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
    )
    assert isinstance(result, NormalizedTask)
    return result.task


@pytest.mark.parametrize(
    "schema_type,candidate,reward",
    [
        ("json", '{"count": 2}', 1.0),
        ("yaml", "count: 2", 1.0),
        ("toml", "count = 2", 1.0),
        ("json", '<think>reasoning</think>{"count": 2}', 1.0),
        ("json", '{"count": "2"}', 0.0),
        ("json", '```json\n{"count": 2}\n```', 0.0),
        ("json", '{"other": 2}', 0.0),
    ],
)
def test_original_text_parsers_schema_validation_and_dispatcher(
    row, native_dependencies, schema_type, candidate, reward
):
    row.data["schema_type"] = schema_type
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    "schema_type,schema,candidate",
    [
        (
            "xml",
            {"type": "object", "properties": {"root": SCHEMA}, "required": ["root"]},
            "<root><count>2</count></root>",
        ),
        ("csv", {"type": "array", "items": SCHEMA}, "count\n2\n"),
    ],
)
def test_original_xml_and_csv_coercion(row, native_dependencies, schema_type, schema, candidate):
    row.data.update(schema_type=schema_type, schema_str=json.dumps(schema))
    assert grade(task_for(row), candidate).reward == 1.0


@pytest.mark.parametrize(
    "name,arguments,reward",
    [("emit", {"payload": {"count": 2}}, 1.0), ("other", {"payload": {"count": 2}}, 0.0), ("emit", {"count": 2}, 0.0)],
)
def test_original_typed_tool_name_payload_and_schema_without_tool_execution(
    row, native_dependencies, name, arguments, reward
):
    row.data.update(response_mode="tool_call", tool_name="emit", tool_payload_key="payload")
    row.data["responses_create_params"]["tools"] = [
        {"type": "function", "name": "emit", "parameters": {"type": "object", "properties": {"payload": SCHEMA}}}
    ]
    task = task_for(row)
    candidate = AssistantToolCalls(calls=(ConversationToolCall(call_id="1", name=name, arguments=arguments),))
    result = grade(task, candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert task.interaction_tools == ()
    assert [tool.name for tool in task.final_tools] == ["emit"]
    assert task.environment_requirements.tool_providers == {}


def test_private_schema_and_public_source_request_are_preserved(row):
    task = task_for(row)
    original = normalization.normalize(row, "fixture", "instruction-following")
    assert isinstance(original, NormalizedTask)
    assert task.context == original.task.context
    assert grader_config(task)["contract"]["schema_str"] == row.data["schema_str"]
    assert task.resources.worker == original.task.resources.worker
    assert task.resources.oracle == original.task.resources.oracle


def test_original_malformed_source_schema_keeps_zero_reward_diagnostic(row, native_dependencies):
    task = task_for(row)
    config = grader_config(task)
    config["contract"]["schema_str"] = "malformed JSON schema"
    package = structured_output_binding.grader_package(config, grader_image(task))
    historical_task = task.model_copy(
        update={
            "grader": package.grader,
            "resources": task.resources.model_copy(update={"verifier": package.resources}),
        }
    )
    result = grade(historical_task, '{"count": 2}')
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0)
    assert result.detail is not None and result.detail["error_type"] == "schema_error"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "additionalProperties": "false"},
        {"type": "object", "required": {"item": ["count"]}},
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "number",
            "exclusiveMinimum": True,
        },
    ],
)
def test_invalid_original_schema_is_a_source_defect_before_runtime(row, schema):
    row.data["schema_str"] = json.dumps(schema)
    result = structured_output_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "1" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
    )
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.SOURCE_DEFECT, "invalid_original_structured_schema")


def test_original_fixed_dialect_accepts_numeric_exclusive_bound_despite_declared_draft4(row, native_dependencies):
    row.data["schema_str"] = json.dumps(
        {"$schema": "http://json-schema.org/draft-04/schema#", "type": "number", "exclusiveMinimum": 0}
    )
    task = task_for(row)
    assert grade(task, "1").reward == 1.0
    assert grade(task, "0").reward == 0.0


@pytest.mark.parametrize(
    "verdict_status,expected",
    [("scored", CheckStatus.PASS), ("invalid_task", CheckStatus.INFRA_ERROR), ("infra_error", CheckStatus.INFRA_ERROR)],
)
@pytest.mark.asyncio
async def test_runtime_diagnostic_never_certifies_schema_correctness(row, verdict_status, expected):
    task = task_for(row)
    machines = SourceScoreMachines(verdict_status=verdict_status)
    report = await structured_output_binding.isolated_checks(
        task, factory=machines, machine_spec=MachineSpec(DockerImage(grader_image(task))), timeout=10
    )
    assert [(check.check, check.status) for check in report.checks] == [
        ("native_runtime", expected),
        ("positive_witness", CheckStatus.SKIPPED),
    ]
    assert len(machines.machines) == 1 and machines.machines[0].closed


@pytest.mark.parametrize(
    "error_type,error_message,expected",
    [
        ("schema_error", "Invalid source schema JSON", CheckStatus.FAIL),
        ("validation_error", "SchemaError: invalid source schema", CheckStatus.FAIL),
        ("validation_error", "ValidationError: candidate does not satisfy schema", CheckStatus.PASS),
        ("parse_error", "JSONDecodeError: candidate is not JSON", CheckStatus.PASS),
    ],
)
@pytest.mark.asyncio
async def test_source_schema_errors_reject_runtime_but_wrong_candidates_do_not(row, error_type, error_message, expected):
    task = task_for(row)
    machines = DiagnosticMachines(diagnostic={"error_type": error_type, "error_message": error_message})
    report = await structured_output_binding.isolated_checks(
        task, factory=machines, machine_spec=MachineSpec(DockerImage(grader_image(task))), timeout=10
    )
    assert report.checks[0].status == expected
    assert report.checks[1].status == CheckStatus.SKIPPED
