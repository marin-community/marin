# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise native Ultra chemistry contracts through packaged script grading."""

import json
from functools import partial
from pathlib import Path

import pytest
from pydantic import JsonValue
from taskcompendium.datasets.nemotron_ultra.normalization import normalize
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, EnvironmentRequirements, Source, TextMessage
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import (
    normalize_rdkit,
    normalize_tool_action,
)

from .test_calendar_binding import grade

FIXTURES = Path(__file__).parent / "fixtures/nemotron_ultra"


def configured_task(task, config):
    return task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={
                    "verifier": tuple(
                        (
                            inline_resource("config.json", json.dumps(config).encode())
                            if resource.path == "config.json"
                            else resource
                        )
                        for resource in task.resources.verifier
                    )
                }
            )
        }
    )


@pytest.fixture(params=["mopd", "rlvr2"])
def chemistry_task(request):
    data = json.loads((FIXTURES / f"rdkit_{request.param}.json").read_text())
    row = RawRow(
        "native-fixture",
        Source(dataset="fixture", revision="pinned", row="0", importer_revision="fixture"),
        data,
    )
    original = normalize(row, "ultra_sft_step3200_rdkit", "chemistry")
    result = normalize_rdkit(
        row,
        image="example.org/grader@sha256:" + "a" * 64,
        normalize_task=partial(normalize, selector="ultra_sft_step3200_rdkit", family="chemistry"),
    )
    assert isinstance(original, NormalizedTask) and isinstance(result, NormalizedTask)
    assert result.task.context == original.task.context
    assert result.task.source == original.task.source and result.task.id == original.task.id
    assert grader_config(result.task) == grader_config(original.task)
    assert result.task.environment_requirements == EnvironmentRequirements()
    assert not result.task.resources.worker and not result.task.resources.all
    return result.task


@pytest.mark.parametrize(
    "wrapper,answer,expected",
    [
        ("boxed", r"\boxed{4.1}", 1),
        ("boxed", r"\boxed{4.6}", 0),
        ("boxed", "4", 0),
        ("boxed", "((4))", 0),
        ("boxed", r"\boxed{4} then \boxed{5}", 0),
        ("boxed", r"<think>\boxed{4}</think> no answer", 0),
        ("boxed", r"<think>wrong</think>\boxed{4}", 1),
        ("parentheses", "((4))", 1),
        ("parentheses", r"\boxed{4}", 0),
        ("parentheses", "((nan))", 0),
        ("parentheses", "((value 4.1))", 1),
    ],
)
def test_native_chemistry_preserves_wrapper_rounding_and_reasoning_extraction(chemistry_task, wrapper, answer, expected):
    config = grader_config(chemistry_task)
    config["contract"].update(expected_answer="4", use_box_format=wrapper == "boxed")
    chemistry_task = configured_task(chemistry_task, config)
    result = grade(chemistry_task, answer)
    assert (result.status, result.reward) == (Outcome.GRADED, expected)


@pytest.mark.parametrize(
    "kind,expected_reward",
    [
        ("matching", 1),
        ("tolerant_float", 1),
        ("wrong_float", 0),
        ("wrong_name", 0),
        ("extra_key", 0),
        ("wrong_type", 0),
        ("extra_call", 0),
        ("different_words", 0),
        ("empty_message", 0),
        ("nonliteral_message", 1),
        ("message_with_call", 0),
    ],
)
def test_native_tool_action_preserves_typed_recursive_comparison_and_message_contract(kind, expected_reward):
    data = json.loads((FIXTURES / "toolcall_schema.json").read_text())
    expected = {
        "type": "function_call",
        "name": "search",
        "arguments": json.dumps({"query": "red green blue", "scores": [1.0], "year": 2026}),
    }
    data["expected_action"] = expected
    source = Source(
        dataset="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        row="0",
        importer_revision="fixture",
    )
    row = RawRow("native-tool", source, data)
    result = normalize_tool_action(
        row,
        image="example.org/grader@sha256:" + "a" * 64,
        normalize_task=partial(normalize, selector="ultra_sft_step3200_toolcall_schema", family="tool-use"),
    )
    assert isinstance(result, NormalizedTask)
    task = result.task
    original = normalize(row, "ultra_sft_step3200_toolcall_schema", "tool-use")
    assert isinstance(original, NormalizedTask)
    assert task.context == original.task.context and task.final_tools == original.task.final_tools
    assert task.source == original.task.source and task.id == original.task.id
    assert grader_config(task) == grader_config(original.task)
    assert not task.interaction_tools and not task.environment_requirements.tool_providers
    config = grader_config(task)
    arguments: dict[str, JsonValue] = {"query": "red green blue", "scores": [1.0], "year": 2026}
    if kind == "tolerant_float":
        arguments["scores"] = [1.0000001]
    if kind == "wrong_float":
        arguments["scores"] = [1.00001]
    if kind == "extra_key":
        arguments["unused"] = 0
    if kind == "wrong_type":
        arguments["year"] = "2026"
    if kind == "different_words":
        arguments["query"] = "red blue green"
    call = ConversationToolCall(
        call_id="actual", name="browse" if kind == "wrong_name" else "search", arguments=arguments
    )
    event = AssistantToolCalls(
        calls=(call, call.model_copy(update={"call_id": "extra"})) if kind == "extra_call" else (call,)
    )
    if kind in {"empty_message", "nonliteral_message", "message_with_call"}:
        config = grader_config(task)
        config["contract"]["expected_action"] = {"type": "message", "content": "private literal reference"}
        if kind != "message_with_call":
            event = TextMessage(role="assistant", content="" if kind == "empty_message" else "A different useful answer")
    task = configured_task(task, config)
    verdict = grade(task, event)
    assert (verdict.status, verdict.reward) == (Outcome.GRADED, expected_reward)


@pytest.mark.parametrize("selection", ["nebius", "swe_gym"])
@pytest.mark.parametrize("control", ["reference", "wrong_tool", "changed_argument", "empty"])
def test_retained_swe_single_step_actions_preserve_original_terminal_scoring(selection, control):
    fixture = json.loads((FIXTURES / "swe_single_step_actions.json").read_text())
    record = next(record for record in fixture["records"] if record["selection"] == selection)
    selector = "agent:" + record["agent_ref"]["name"]
    data = {
        "agent_ref": record["agent_ref"],
        "expected_action": record["expected_action"],
        "responses_create_params": {
            "input": [{"role": "user", "content": "Predict the next action for the supplied tool interface."}],
            "tools": [{"type": "function", "function": tool} for tool in record["final_tools"]],
        },
    }
    row = RawRow("swe-next-action", Source.model_validate(record["source"]), data)
    original = normalize(row, selector, "swe-repo")
    result = normalize_tool_action(
        row,
        image="example.org/grader@sha256:" + "a" * 64,
        normalize_task=partial(normalize, selector=selector, family="swe-repo"),
    )
    assert isinstance(original, NormalizedTask) and isinstance(result, NormalizedTask)
    task = result.task
    assert task.context == original.task.context and task.final_tools == original.task.final_tools
    assert grader_config(task) == grader_config(original.task)
    assert not task.interaction_tools and not task.environment_requirements.tool_providers
    expected = record["expected_action"]
    arguments = json.loads(expected["arguments"])
    if control == "changed_argument":
        key = next(iter(arguments))
        arguments[key] = str(arguments[key]) + "__wrong_required_argument__"
    event = AssistantToolCalls(
        calls=(
            ConversationToolCall(
                call_id="candidate",
                name="__wrong_tool__" if control == "wrong_tool" else expected["name"],
                arguments=arguments,
            ),
        )
    )
    if control == "empty":
        event = TextMessage(role="assistant", content="")
    verdict = grade(task, event)
    assert (verdict.status, verdict.reward) == (Outcome.GRADED, 1.0 if control == "reference" else 0.0)
