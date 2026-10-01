# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned NeMo final-action import and Harbor replay behavior."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

from taskcompendium.harbor.protocol import assistant_message
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row
from taskcompendium.lowering import HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor, read_specification
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionCall,
    FunctionDefinition,
    Source,
    TaskSpec,
    TextMessage,
    ToolCallComparatorConfig,
)
from taskcompendium.submission import FinalAction, GradingAttempt, chat_request
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.predicted_action import compare, predicted_action_verifier

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
    assert saved_specification["environment_requirements"] == {"capabilities": []}
    assert saved_specification["tool_providers"] == {}
    public = (task / "instruction.md").read_text() + (task / "submission_convention.json").read_text()
    assert row["expected_action"]["arguments"] not in public
    assert "Okay, let me figure out how to handle this user's query" not in public
    assert "authenticate_user" in {function.name for function in specification.final_tools}
    assert row["expected_action"]["arguments"] not in (task / "tests/test.sh").read_text()
    with pytest.raises(ValueError, match="pinned canonical hash"):
        import_row(row, "0" * 64)


def test_exported_nemo_verifier_grades_in_fresh_process(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    script = (
        "import asyncio, json, sys; from pathlib import Path; "
        "from taskcompendium.verifier_registry import grade_answer; "
        "from taskcompendium.harbor.protocol import chat_conversation; "
        "from taskcompendium.submission import GradingAttempt, chat_request; "
        "from taskcompendium.lowering import read_submission_convention, read_specification; "
        "root = Path(sys.argv[1]); "
        "specification = read_specification(root / 'specification.json'); "
        "convention = read_submission_convention(root / 'submission_convention.json'); "
        "conversation = chat_conversation([*chat_request(specification, convention)['messages'], "
        "json.loads(sys.argv[2])]); "
        "result = asyncio.run(grade_answer(specification, convention, GradingAttempt(conversation, {}, object()))); "
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
        read_specification(task / "specification.json")


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
            0.0,
            "submission_failure",
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
    with pytest.raises(ValueError, match="conflicts with the submission convention"):
        await run_trial(
            task,
            environment_config,
            ChatLaunch(model="model", api_base="https://example.invalid", parallel_tool_calls=True),
            tmp_path / "trials",
            "conflicting-launch",
        )
    assert len(requests) == 1


def test_predicted_action_requires_exact_call_count_and_argument_types():
    config = ToolCallComparatorConfig()
    expected = (FunctionCall(name="lookup", arguments={"id": 1}),)
    extra = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call-lookup", "type": "function", "function": {"name": "lookup", "arguments": '{"id":1}'}},
            {"id": "call-other", "type": "function", "function": {"name": "other", "arguments": "{}"}},
        ],
    }
    assert compare(expected, assistant_message(extra), config) == 0.0
    assert compare(expected, assistant_message(_action("lookup", '{"id":true}')), config) == 0.0
    assert compare(expected, assistant_message({"role": "assistant", "content": "different"}), config) == 0.0


@pytest.mark.parametrize(
    "call_count,parallel_tool_calls,rejected",
    [(2, False, True), (1, False, False), (2, True, False)],
    ids=["batch-disabled", "single-call-with-unbounded-convention", "batch-enabled"],
)
async def test_launch_parallel_policy_preserves_expected_action(
    tmp_path, monkeypatch, call_count, parallel_tool_calls, rejected
):
    expected = tuple(FunctionCall(name="lookup", arguments={"id": index}) for index in range(call_count))
    specification = TaskSpec(
        id="synthetic-final-action",
        context=ConversationInput(events=(TextMessage(role="user", content="Look up the requested records."),)),
        environment_requirements=EnvironmentRequirements(),
        final_tools=(FunctionDefinition(name="lookup", parameters={"type": "object"}),),
        answer_type=AnswerType.NATIVE_ACTION,
        verifier=predicted_action_verifier(expected),
        source=Source(dataset="synthetic", revision="1", row="parallel-policy", importer_revision="1"),
    )
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(specification, FinalAction(id="unbounded"), environment_config, tmp_path / "task")
    requests = []

    def respond(request, **_kwargs):
        requests.append(json.loads(request.data))
        calls = [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for index, call in enumerate(expected)
        ]
        response = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    launch = ChatLaunch(model="model", api_base="https://example.invalid", parallel_tool_calls=parallel_tool_calls)
    if rejected:
        with pytest.raises(ValueError, match="disables parallel calls required by the task"):
            await run_trial(task, environment_config, launch, tmp_path / "trials", "run")
        assert requests == []
        assert not (tmp_path / "trials").exists()
        return

    result = await run_trial(task, environment_config, launch, tmp_path / "trials", "run")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert requests[0]["parallel_tool_calls"] is parallel_tool_calls


def test_predicted_action_requires_exact_strings_and_explicit_numeric_tolerance():
    expected_text = (FunctionCall(name="respond", arguments={"note": "refund approved"}),)
    wrong_text = assistant_message(_action("respond", '{"note":"refund denied"}'))
    assert compare(expected_text, wrong_text, ToolCallComparatorConfig()) == 0.0

    expected_number = (FunctionCall(name="set_value", arguments={"value": 1.0}),)
    nearby_number = assistant_message(_action("set_value", '{"value":1.005}'))
    assert compare(expected_number, nearby_number, ToolCallComparatorConfig()) == 0.0
    assert compare(expected_number, nearby_number, ToolCallComparatorConfig(numeric_tolerance=0.01)) == 1.0


async def test_predicted_action_grades_typed_evidence_from_any_harness():
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

    result = await grade_answer(specification, convention, GradingAttempt(conversation, {}, object()))

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


@pytest.mark.parametrize("require_call", [False, True])
async def test_imported_final_call_constraints_distinguish_invalid_submission(require_call):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["tool_choice"] = "required" if require_call else "auto"
    specification, convention = import_row(row, canonical_sha256(row))
    final = assistant_message({"role": "assistant", "content": "No action"})
    attempt = GradingAttempt(ConversationTrace(events=(*specification.context.events, final)), {}, object())
    result = await grade_answer(specification, convention, attempt)
    assert (result.status, result.reward) == ("submission_failure" if require_call else "graded", 0.0)
    request = chat_request(specification, convention)
    assert request.get("tool_choice") == ("required" if require_call else None)
    assert request["parallel_tool_calls"] is False


async def test_imported_parallel_actions_accept_multiple_final_calls():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["responses_create_params"]["parallel_tool_calls"] = True
    original_call = row["expected_action"]
    row["expected_action"] = {"type": "function_call_batch", "calls": [original_call, original_call]}
    specification, convention = import_row(row, canonical_sha256(row))
    single = _action(original_call["name"], original_call["arguments"])["tool_calls"][0]
    final = assistant_message({"role": "assistant", "tool_calls": [single, {**single, "id": "second"}]})
    attempt = GradingAttempt(ConversationTrace(events=(*specification.context.events, final)), {}, object())
    result = await grade_answer(specification, convention, attempt)
    assert (result.status, result.reward) == ("graded", 1.0)
    assert "parallel_tool_calls" not in chat_request(specification, convention)
