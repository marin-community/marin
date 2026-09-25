# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned NeMo final-action import and Harbor replay behavior."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

from taskcompendium.harbor.runner import HarborLaunch, run_trial
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row
from taskcompendium.lowering import HarborTaskBinding, compatible_lowerings, lower_to_harbor, read_specification
from taskcompendium.models import AnswerType, FunctionCall, ToolCallComparatorConfig
from taskcompendium.predicted_action import compare, decode_action

FIXTURES = Path(__file__).parent / "fixtures/nemo"


def _action(name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def test_pinned_nemo_row_keeps_expected_action_private(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    assert canonical_sha256(row) == provenance["canonical_json_sha256"]
    specification, convention = import_row(row, provenance["canonical_json_sha256"])
    assert specification.answer_type == AnswerType.NATIVE_ACTION
    assert convention.supports(AnswerType.NATIVE_ACTION)
    assert not convention.supports(AnswerType.FILE)
    assert specification.source.dataset == provenance["dataset"]
    assert specification.source.revision == provenance["dataset_revision"]
    assert [message.role for message in convention.messages] == ["system", "user", "assistant", "user"]
    assert convention.messages[0].content == row["responses_create_params"]["input"][0]["content"]
    assert convention.messages[-1].content == row["responses_create_params"]["input"][-1]["content"]
    task = lower_to_harbor(specification, convention, HarborTaskBinding(), tmp_path / "task")
    public = (task / "instruction.md").read_text() + (task / "submission_convention.json").read_text()
    assert row["expected_action"]["arguments"] not in public
    convention_data = json.loads((task / "submission_convention.json").read_text())
    assert "authenticate_user" in {function["name"] for function in convention_data["functions"]}
    assert not (task / "tests").exists()
    with pytest.raises(ValueError, match="pinned canonical hash"):
        import_row(row, "0" * 64)


def test_exported_nemo_verifier_grades_in_fresh_process(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborTaskBinding(), tmp_path / "task")
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.grading import GradingAttempt, grade_attempt; "
        "from taskcompendium.lowering import read_submission_convention, read_specification; "
        "root = Path(sys.argv[1]); "
        "result = grade_attempt(read_specification(root / 'specification.json'), "
        "GradingAttempt(read_submission_convention(root / 'submission_convention.json'), sys.argv[2], object())); "
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


def test_predicted_action_rejects_message_target_with_weak_source_scoring():
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["expected_action"] = {"type": "message", "content": "A specific answer"}

    with pytest.raises(ValueError, match="message targets have no correctness comparison"):
        import_row(row, canonical_sha256(row))


@pytest.mark.parametrize("arguments", ["[1]", '{"id":1,"id":2}', '{"id":NaN}', '{"id":1e309}'])
def test_predicted_action_rejects_invalid_expected_arguments(arguments):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    row["expected_action"]["arguments"] = arguments

    with pytest.raises(ValueError, match=r"JSON object|Duplicate JSON field|Non-finite JSON argument"):
        import_row(row, canonical_sha256(row))


def test_predicted_action_rejects_crafted_message_target_on_private_read(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborTaskBinding(), tmp_path / "task")
    data = json.loads((task / "specification.json").read_text())
    data["verifier"]["parameters"] = {"expected_message": "Any response"}
    (task / "specification.json").write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Invalid 'nemo_predicted_action' verifier parameters"):
        read_specification(task / "specification.json")


def test_predicted_action_rejects_boolean_numeric_tolerance_on_private_read(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    task = lower_to_harbor(specification, convention, HarborTaskBinding(), tmp_path / "task")
    data = json.loads((task / "specification.json").read_text())
    data["verifier"]["parameters"]["numeric_tolerance"] = True
    (task / "specification.json").write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Invalid 'nemo_predicted_action' verifier parameters"):
        read_specification(task / "specification.json")


def test_predicted_action_rejects_instruction_message_drift(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    changed = specification.model_copy(update={"instructions": "Changed instruction"})

    assert compatible_lowerings(changed, (convention,), (HarborTaskBinding(),)) == ()

    with pytest.raises(ValueError, match="differ from source messages"):
        lower_to_harbor(
            changed,
            convention,
            HarborTaskBinding(),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


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
                        "type": "function",
                        "function": {
                            "name": "authenticate_user",
                            "arguments": '{"user_id":"GROOM2024","event_confirmation_code":"NIGHTCLUB2024"}',
                        },
                    },
                    {"type": "function", "function": {"name": "get_event_details", "arguments": "{}"}},
                ],
            },
            0.0,
            "graded",
        ),
        ({"role": "assistant", "tool_calls": "not-a-list"}, None, "extraction_error"),
    ],
)
async def test_predicted_action_harbor_replay_outcomes(tmp_path, response, reward, status):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    binding = HarborTaskBinding()
    task = lower_to_harbor(specification, convention, binding, tmp_path / "task")

    result = await run_trial(
        task, binding, HarborLaunch("action_replay", agent_kwargs={"response": response}), tmp_path / "trials", "run"
    )

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert (outcome["status"], outcome["reward"]) == (status, reward)
    if reward is None:
        assert result.verifier_result is None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
    assert not (tmp_path / "trials/run/agent/response.txt").exists()


@pytest.mark.parametrize(
    "launch,message",
    [
        (HarborLaunch("replay", agent_kwargs={"response": "text"}), "Text replay requires"),
        (HarborLaunch("action_replay", agent_kwargs={"response": "text"}), "Action replay requires"),
        (HarborLaunch("action_replay", model="model", agent_kwargs={"response": {}}), "cannot select a model"),
        (HarborLaunch("action_replay"), "requires only a response"),
        (HarborLaunch("chat", model="model"), "requires api_base"),
        (HarborLaunch("chat", model="model", agent_kwargs={"api_base": "url", "response": {}}), "Unsupported chat"),
    ],
)
async def test_predicted_action_rejects_incompatible_launch_before_trial(tmp_path, launch, message):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    binding = HarborTaskBinding()
    task = lower_to_harbor(specification, convention, binding, tmp_path / "task")

    with pytest.raises(ValueError, match=message):
        await run_trial(task, binding, launch, tmp_path / "trials", "run")
    assert not (tmp_path / "trials").exists()


async def test_predicted_action_chat_requests_native_output_without_dispatch(tmp_path, monkeypatch):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    specification, convention = import_row(row, canonical_sha256(row))
    binding = HarborTaskBinding()
    task = lower_to_harbor(specification, convention, binding, tmp_path / "task")
    requests = []
    monkeypatch.setenv("NEMO_TEST_API_KEY", "test-token")

    def respond(request, timeout):
        requests.append((json.loads(request.data), request.get_header("Authorization")))
        response = {"choices": [{"message": _action("authenticate_user", row["expected_action"]["arguments"])}]}
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task,
        binding,
        HarborLaunch("chat", "model", {"api_base": "https://example.invalid", "api_key_env": "NEMO_TEST_API_KEY"}),
        tmp_path / "trials",
        "run",
    )

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    request, authorization = requests[0]
    assert request["tools"][0]["function"]["name"] == "authenticate_user"
    assert request["messages"] == [{"role": message.role, "content": message.content} for message in convention.messages]
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False
    assert authorization == "Bearer test-token"
    assert len(requests) == 1


def test_predicted_action_requires_exact_call_count_and_argument_types():
    config = ToolCallComparatorConfig()
    expected = (FunctionCall("lookup", '{"id":1}'),)
    extra = {
        "role": "assistant",
        "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": '{"id":1}'}},
            {"type": "function", "function": {"name": "other", "arguments": "{}"}},
        ],
    }
    assert compare(expected, decode_action(extra), config) == 0.0
    assert compare(expected, decode_action(_action("lookup", '{"id":true}')), config) == 0.0
    assert compare(expected, decode_action({"role": "assistant", "content": "different"}), config) == 0.0
    assert compare(expected, decode_action(_action("lookup", "not-json")), config) == 0.0


def test_predicted_action_requires_exact_strings_and_explicit_numeric_tolerance():
    expected_text = (FunctionCall("respond", '{"note":"refund approved"}'),)
    wrong_text = decode_action(_action("respond", '{"note":"refund denied"}'))
    assert compare(expected_text, wrong_text, ToolCallComparatorConfig()) == 0.0

    expected_number = (FunctionCall("set_value", '{"value":1.0}'),)
    nearby_number = decode_action(_action("set_value", '{"value":1.005}'))
    assert compare(expected_number, nearby_number, ToolCallComparatorConfig()) == 0.0
    assert compare(expected_number, nearby_number, ToolCallComparatorConfig(numeric_tolerance=0.01)) == 1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ToolCallComparatorConfig(numeric_tolerance=float("inf"))
