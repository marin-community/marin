# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned NeMo final-action import and Harbor replay behavior."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

from taskcompendium.grading import grade_answer, validate_verifier
from taskcompendium.harbor.protocol import assistant_message
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row
from taskcompendium.lowering import (
    HarborEnvironmentConfig,
    compatible_lowerings,
    lower_to_harbor,
    read_specification,
    read_submission_convention,
)
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
)
from taskcompendium.submission import AnswerFormat, FinalAction, SubmissionConvention, chat_request

from .harbor_replay import run_replay_trial

FIXTURES = Path(__file__).parent / "fixtures/nemo"


def _action(name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-final", "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def test_pinned_nemo_row_keeps_expected_action_private(tmp_path):
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
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    saved_specification = json.loads((task / "specification.json").read_text())
    assert set(saved_specification["context"]) == {"events"}
    assert saved_specification["answer_type"] == "native_action"
    assert isinstance(saved_specification["final_tools"], list)
    assert saved_specification["final_tools"]
    assert saved_specification["environment_requirements"]["capabilities"] == []
    assert saved_specification["environment_requirements"]["tool_providers"] == {}
    public = (task / "instruction.md").read_text() + (task / "submission_convention.json").read_text()
    assert row["expected_action"]["arguments"] not in public
    assert "Okay, let me figure out how to handle this user's query" not in public
    assert "authenticate_user" in {function.name for function in specification.final_tools}
    assert not (task / "tests").exists()
    with pytest.raises(ValueError, match="pinned canonical hash"):
        import_row(row, "0" * 64)


def test_exported_nemo_verifier_grades_in_fresh_process(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.grading import grade_answer; "
        "from taskcompendium.harbor.protocol import chat_conversation; "
        "from taskcompendium.submission import chat_request; "
        "from taskcompendium.lowering import read_submission_convention, read_specification; "
        "root = Path(sys.argv[1]); "
        "specification = read_specification(root / 'specification.json'); "
        "convention = read_submission_convention(root / 'submission_convention.json'); "
        "conversation = chat_conversation([*chat_request(specification, convention)['messages'], "
        "json.loads(sys.argv[2])]); "
        "result = grade_answer(specification, convention, conversation); "
        "print(json.dumps({'status': result.status, 'reward': result.reward}))"
    )
    response = json.dumps(_action(row["expected_action"]["name"], row["expected_action"]["arguments"]))

    completed = subprocess.run(
        [sys.executable, "-c", script, str(task), response], capture_output=True, text=True, check=True
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


def test_predicted_action_rejects_crafted_message_target_on_private_read(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    data = json.loads((task / "specification.json").read_text())
    data["verifier"]["parameters_json"] = json.dumps({"expected_message": "Any response"})
    (task / "specification.json").write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Invalid 'predicted_action' verifier parameters"):
        validate_verifier(read_specification(task / "specification.json").verifier)


def test_predicted_action_reuses_final_action_convention_without_changing_source_request(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    candidates = compatible_lowerings(specification, (convention,), (HarborEnvironmentConfig(),))

    assert len(candidates) == 1
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    exported = read_specification(task / "specification.json")
    assert exported.context == specification.context


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
            None,
            "extraction_error",
        ),
    ],
)
async def test_predicted_action_harbor_replay_outcomes(tmp_path, response, reward, status):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")

    result = await run_replay_trial(task, response, tmp_path / "trials", "run")

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert (outcome["status"], outcome["reward"]) == (status, reward)
    if reward is None:
        assert result.verifier_result is None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
    assert (tmp_path / "trials/run/agent/submission.json").exists()


async def test_predicted_action_chat_requests_native_output_without_dispatch(tmp_path, monkeypatch):
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
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")
    requests = []
    monkeypatch.setenv("NEMO_TEST_API_KEY", "test-token")

    def respond(request, **_kwargs):
        requests.append((json.loads(request.data), request.get_header("Authorization")))
        response = {"choices": [{"message": _action("authenticate_user", row["expected_action"]["arguments"])}]}
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task,
        environment_config,
        ChatLaunch(model="model", api_base="https://example.invalid", api_key_env="NEMO_TEST_API_KEY"),
        tmp_path / "trials",
        "run",
    )

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    request, authorization = requests[0]
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
    assert authorization == "Bearer test-token"
    assert len(requests) == 1


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

    result = grade_answer(specification, convention, conversation)

    assert (result.status, result.reward) == ("graded", 1.0)


@pytest.mark.parametrize(
    "response",
    [
        {"role": "assistant", "tool_calls": "not-a-list"},
        _action("authenticate_user", "not-json"),
    ],
)
async def test_chat_protocol_failure_is_ungraded_and_retains_raw_response(tmp_path, monkeypatch, response):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")

    def respond(*_args, **_kwargs):
        return BytesIO(json.dumps({"choices": [{"message": response}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task,
        environment_config,
        ChatLaunch(model="model", api_base="https://example.invalid"),
        tmp_path / "trials",
        "run",
    )

    assert result.exception_info is not None
    assert result.verifier_result is None
    assert json.loads((tmp_path / "trials/run/agent/chat-response.json").read_text()) == response
    assert not (tmp_path / "trials/run/agent/submission.json").exists()


def test_imported_call_policy_survives_export_and_convention_reload(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["tool_choice"] = "required"
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    restored = read_submission_convention(task / "submission_convention.json")
    request = chat_request(read_specification(task / "specification.json"), restored)
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False


@pytest.mark.parametrize("call_count,status,reward", [(2, "graded", 1.0), (3, "extraction_error", None)])
def test_final_action_call_limit_applies_before_matching_expected_calls(call_count, status, reward):
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
    conversation = ConversationTrace(events=(*specification.context.events, final))
    result = grade_answer(specification, FinalAction(id="max-two", require_call=True, max_calls=2), conversation)
    assert (result.status, result.reward) == (status, reward)


@pytest.mark.parametrize("require_call,status,reward", [(False, "graded", 0.0), (True, "extraction_error", None)])
def test_final_action_required_call_rejects_text_before_scoring(require_call, status, reward):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, _ = import_row(row, canonical_sha256(row))
    final = assistant_message({"role": "assistant", "content": "No action"})
    conversation = ConversationTrace(events=(*specification.context.events, final))
    result = grade_answer(specification, FinalAction(id="call-policy", require_call=require_call), conversation)
    assert (result.status, result.reward) == (status, reward)


def test_final_action_cannot_bypass_call_policy_through_a_text_convention(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, _ = import_row(row, canonical_sha256(row))
    with pytest.raises(ValueError):
        SubmissionConvention(id="bypass", answer_format=AnswerFormat.FINAL_ACTION)
    convention = FinalAction(id="one-call", require_call=True, max_calls=1)
    path = tmp_path / "convention.json"
    path.write_text(convention.model_dump_json())
    restored = read_submission_convention(path)
    for current in (convention, restored):
        request = chat_request(specification, current)
        assert request["tool_choice"] == "required"
        assert request["parallel_tool_calls"] is False
        response = AssistantToolCalls(
            calls=(
                ConversationToolCall(call_id="first", name="lookup", arguments={}),
                ConversationToolCall(call_id="second", name="lookup", arguments={}),
            )
        )
        result = grade_answer(
            specification, current, ConversationTrace(events=(*specification.context.events, response))
        )
        assert (result.status, result.reward) == ("extraction_error", None)
