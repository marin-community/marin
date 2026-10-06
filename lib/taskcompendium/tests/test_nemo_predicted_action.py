# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned NeMo final-action import, chat evidence, and pure grading behavior."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from taskcompendium.chat import assistant_message, chat_conversation
from taskcompendium.grading import grade_answer, validate_verifier
from taskcompendium.grading_contract import GradingAttempt
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    TaskSpec,
)
from taskcompendium.submission import FinalAction, chat_request, render_instruction

FIXTURES = Path(__file__).parent / "fixtures/nemo"


def _action(name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-final", "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def test_pinned_nemo_row_keeps_expected_action_private():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    assert canonical_sha256(row) == provenance["canonical_json_sha256"]
    specification, convention = import_row(row, provenance["canonical_json_sha256"])
    assert specification.answer_type == AnswerType.NATIVE_ACTION
    assert convention.supports(AnswerType.NATIVE_ACTION)
    assert not convention.supports(AnswerType.FILE)
    request = specification.context
    assert specification.source.dataset == provenance["dataset"]
    assert specification.source.revision == provenance["dataset_revision"]
    assert [message.role for message in request.events] == ["system", "user", "assistant", "user"]
    assert request.events[0].content == row["responses_create_params"]["input"][0]["content"]
    assert request.events[-1].content == row["responses_create_params"]["input"][-1]["content"]
    saved_specification = json.loads(specification.model_dump_json())
    assert set(saved_specification["context"]) == {"events"}
    assert saved_specification["answer_type"] == "native_action"
    assert isinstance(saved_specification["final_tools"], list)
    assert saved_specification["final_tools"]
    public = render_instruction(specification, convention) + convention.model_dump_json()
    assert row["expected_action"]["arguments"] not in public
    assert "Okay, let me figure out how to handle this user's query" not in public
    assert "authenticate_user" in {function.name for function in specification.final_tools}
    with pytest.raises(ValueError, match="pinned canonical hash"):
        import_row(row, "0" * 64)


def test_serialized_nemo_verifier_grades_in_fresh_process(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    (tmp_path / "specification.json").write_text(specification.model_dump_json())
    (tmp_path / "convention.json").write_text(convention.model_dump_json())
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.grading import grade_answer; "
        "from taskcompendium.grading_contract import GradingAttempt; "
        "from taskcompendium.submission import FinalAction; "
        "from taskcompendium.models import TaskSpec, ConversationTrace; "
        "root = Path(sys.argv[1]); "
        "specification = TaskSpec.model_validate_json((root/'specification.json').read_text()); "
        "convention = FinalAction.model_validate_json((root/'convention.json').read_text()); "
        "conversation = ConversationTrace.model_validate_json(sys.argv[2]); "
        "result = grade_answer(specification, convention, GradingAttempt(conversation)); "
        "print(json.dumps({'status':result.status, 'reward':result.reward}))"
    )
    response = assistant_message(_action(row["expected_action"]["name"], row["expected_action"]["arguments"]))
    conversation = ConversationTrace(events=(*specification.context.events, response))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), conversation.model_dump_json()],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}


def test_predicted_action_rejects_source_request_settings_it_cannot_preserve():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["instructions"] = "Additional system instruction"

    with pytest.raises(ValueError, match="unsupported source request settings"):
        import_row(row, canonical_sha256(row))


@pytest.mark.parametrize("before_final_user", [False, True])
def test_predicted_action_rejects_reasoning_without_visible_result(before_final_user):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    reasoning = row["responses_create_params"]["input"][2]
    position = -1 if before_final_user else len(row["responses_create_params"]["input"])
    row["responses_create_params"]["input"].insert(position, reasoning)

    with pytest.raises(ValueError, match="reasoning has no visible assistant result"):
        import_row(row, canonical_sha256(row))


def test_predicted_action_rejects_message_target_with_weak_source_scoring():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["expected_action"] = {"type": "message", "content": "A specific answer"}

    with pytest.raises(ValueError, match="message targets have no correctness comparison"):
        import_row(row, canonical_sha256(row))


def test_predicted_action_rejects_source_settings_that_prevent_expected_calls():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["tool_choice"] = "none"
    with pytest.raises(ValueError, match="tool_choice=none"):
        import_row(row, canonical_sha256(row))

    row["responses_create_params"]["tool_choice"] = "auto"
    row["responses_create_params"]["parallel_tool_calls"] = False
    row["expected_action"] = {"type": "function_call_batch", "calls": [row["expected_action"]] * 2}
    with pytest.raises(ValueError, match="parallel_tool_calls=false"):
        import_row(row, canonical_sha256(row))


@pytest.mark.parametrize("arguments", ["[1]", '{"id":NaN}', '{"id":1e309}'])
def test_predicted_action_rejects_invalid_expected_arguments(arguments):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["expected_action"]["arguments"] = arguments

    with pytest.raises(ValueError, match=r"dictionary|finite"):
        import_row(row, canonical_sha256(row))


@pytest.mark.parametrize(
    "parameters",
    [
        {"expected_message": "Any response"},
        {"expected_calls": [{"name": "lookup", "arguments": {"id": 1}}], "numeric_tolerance": 10**400},
    ],
)
def test_predicted_action_rejects_invalid_contract_on_private_read(parameters):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, _ = import_row(row, canonical_sha256(row))
    data = json.loads(specification.model_dump_json())
    data["verifier"]["parameters_json"] = json.dumps(parameters)
    restored = TaskSpec.model_validate_json(json.dumps(data))
    with pytest.raises(ValueError):
        validate_verifier(restored.verifier)


def test_predicted_action_task_roundtrip_preserves_source_context():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    restored = TaskSpec.model_validate_json(specification.model_dump_json())
    assert restored.context == specification.context
    assert chat_request(restored, convention) == chat_request(specification, convention)


@pytest.mark.parametrize(
    "response,reward,status",
    [
        (
            _action("authenticate_user", '{"user_id":"GROOM2024","event_confirmation_code":"NIGHTCLUB2024"}'),
            1.0,
            "graded",
        ),
        (_action("get_event_details", "{}"), 0.0, "graded"),
        ({"role": "assistant", "content": "I cannot do that"}, 0.0, "graded"),
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-auth",
                        "type": "function",
                        "function": {
                            "name": "authenticate_user",
                            "arguments": '{"user_id":"GROOM2024","event_confirmation_code":"NIGHTCLUB2024"}',
                        },
                    },
                    {
                        "id": "call-details",
                        "type": "function",
                        "function": {"name": "get_event_details", "arguments": "{}"},
                    },
                ],
            },
            0.0,
            "submission_failure",
        ),
    ],
)
def test_predicted_action_chat_evidence_distinguishes_wrong_and_invalid_submission(response, reward, status):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    trace = chat_conversation([*chat_request(specification, convention)["messages"], response])
    restored = ConversationTrace.model_validate_json(trace.model_dump_json())
    result = grade_answer(specification, convention, GradingAttempt(restored))
    assert (result.status, result.reward) == (status, reward)


def test_predicted_action_chat_request_preserves_source_history_and_tools():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["input"][-1:-1] = [
        {
            "type": "function_call",
            "call_id": "call-profile",
            "name": "get_user_profile",
            "arguments": '{"user_id":"GROOM2024"}',
        },
        {"type": "function_call_output", "call_id": "call-profile", "output": '{"verified":false}'},
    ]
    specification, convention = import_row(row, canonical_sha256(row))
    request = chat_request(specification, convention)
    native_request = specification.final_tools
    assert [tool["function"]["name"] for tool in request["tools"]] == [function.name for function in native_request]
    assert [tool["function"]["parameters"] for tool in request["tools"]] == [
        function.parameters for function in native_request
    ]
    assert [message["role"] for message in request["messages"]] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "user",
    ]
    assert request["messages"][0]["content"] == row["responses_create_params"]["input"][0]["content"]
    assert request["messages"][3] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-profile",
                "type": "function",
                "function": {"name": "get_user_profile", "arguments": '{"user_id":"GROOM2024"}'},
            }
        ],
    }
    assert request["messages"][4] == {
        "role": "tool",
        "tool_call_id": "call-profile",
        "content": '{"verified":false}',
    }
    assert request["messages"][-1]["content"] == row["responses_create_params"]["input"][-1]["content"]
    assert "tool_choice" not in request
    assert request["parallel_tool_calls"] is False
    trace = chat_conversation(
        [*request["messages"], _action(row["expected_action"]["name"], row["expected_action"]["arguments"])]
    )
    trace = ConversationTrace.model_validate_json(trace.model_dump_json())
    assert trace.events[3].calls[0].arguments == {"user_id": "GROOM2024"}
    assert trace.events[4].content == '{"verified":false}'
    result = grade_answer(specification, convention, GradingAttempt(trace))
    assert (result.status, result.reward) == ("graded", 1.0)


def test_predicted_action_grades_typed_evidence_from_any_harness():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    final = AssistantToolCalls(
        calls=(
            ConversationToolCall(
                call_id="another-harness-call",
                name=row["expected_action"]["name"],
                arguments=json.loads(row["expected_action"]["arguments"]),
            ),
        )
    )
    conversation = ConversationTrace(events=(*specification.context.events, final))

    result = grade_answer(specification, convention, GradingAttempt(conversation))

    assert (result.status, result.reward) == ("graded", 1.0)


@pytest.mark.parametrize("require_call", [False, True])
def test_imported_final_call_constraints_distinguish_invalid_submission(require_call):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["tool_choice"] = "required" if require_call else "auto"
    specification, convention = import_row(row, canonical_sha256(row))
    final = assistant_message({"role": "assistant", "content": "No action"})
    attempt = GradingAttempt(ConversationTrace(events=(*specification.context.events, final)))
    result = grade_answer(specification, convention, attempt)
    assert (result.status, result.reward) == ("submission_failure" if require_call else "graded", 0.0)
    request = chat_request(specification, convention)
    assert request.get("tool_choice") == ("required" if require_call else None)
    assert request["parallel_tool_calls"] is False


def test_imported_parallel_actions_accept_multiple_final_calls():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["parallel_tool_calls"] = True
    original_call = row["expected_action"]
    row["expected_action"] = {"type": "function_call_batch", "calls": [original_call, original_call]}
    specification, convention = import_row(row, canonical_sha256(row))
    single = _action(original_call["name"], original_call["arguments"])["tool_calls"][0]
    final = assistant_message({"role": "assistant", "tool_calls": [single, {**single, "id": "second"}]})
    attempt = GradingAttempt(ConversationTrace(events=(*specification.context.events, final)))
    result = grade_answer(specification, convention, attempt)
    assert (result.status, result.reward) == ("graded", 1.0)
    assert "parallel_tool_calls" not in chat_request(specification, convention)


@pytest.mark.parametrize("call_count,status,reward", [(2, "graded", 1.0), (3, "submission_failure", 0.0)])
def test_final_action_max_two_preserves_the_submission_limit_before_scoring(call_count, status, reward):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["parallel_tool_calls"] = True
    expected = row["expected_action"]
    row["expected_action"] = {"type": "function_call_batch", "calls": [expected] * call_count}
    specification, _ = import_row(row, canonical_sha256(row))
    final = AssistantToolCalls(
        calls=tuple(
            ConversationToolCall(
                call_id=f"final-{index}", name=expected["name"], arguments=json.loads(expected["arguments"])
            )
            for index in range(call_count)
        )
    )
    attempt = GradingAttempt(ConversationTrace(events=(*specification.context.events, final)))
    result = grade_answer(specification, FinalAction(id="max-two", require_call=True, max_calls=2), attempt)
    assert (result.status, result.reward) == (status, reward)
