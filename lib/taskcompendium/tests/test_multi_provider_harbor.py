# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two independent tool services through one Harbor chat trial."""

import hashlib
import json
from io import BytesIO

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from upath import UPath

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.harbor.adapter import CompositeToolEnvironment
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding, lower_to_harbor
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionCall,
    FunctionDefinition,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import AnswerCall, FinalAction, PlainText, ProviderState
from taskcompendium.verifiers.predicted_action import predicted_action_verifier

from .conftest import ServiceFault


def _definition(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Increment {name}",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class AlphaProvider:
    ACTION_INTERFACE = "alpha:v1"
    SEED_SHA256 = "a" * 64
    PROVIDER_REVISION = "alpha-1"
    TOOL_DEFINITIONS = (_definition("increment_alpha"),)


class BetaProvider(AlphaProvider):
    ACTION_INTERFACE = "beta:v1"
    SEED_SHA256 = "b" * 64
    PROVIDER_REVISION = "beta-1"
    TOOL_DEFINITIONS = (_definition("increment_beta"),)


class ExpandedAlphaProvider(AlphaProvider):
    PROVIDER_REVISION = "alpha-2"
    TOOL_DEFINITIONS = (*AlphaProvider.TOOL_DEFINITIONS, _definition("unrelated"))


class BrokenStateProvider(BetaProvider):
    PROVIDER_REVISION = "beta-broken"


def _binding(provider: type[AlphaProvider], runtime_factory) -> ToolBinding:
    return ToolBinding(
        action_interface=provider.ACTION_INTERFACE,
        seed_sha256=provider.SEED_SHA256,
        runtime=runtime_factory(
            provider.ACTION_INTERFACE,
            provider.SEED_SHA256,
            provider.PROVIDER_REVISION,
            provider.TOOL_DEFINITIONS,
            fault=ServiceFault.STATE_UNAVAILABLE if provider is BrokenStateProvider else ServiceFault.NORMAL,
        ),
        tool_definitions=tuple(provider.TOOL_DEFINITIONS),
        state_available=True,
        provider_revision=provider.PROVIDER_REVISION,
        tools=(provider.TOOL_DEFINITIONS[0]["function"]["name"],),
        tools_sha256=_digest(provider.TOOL_DEFINITIONS),
    )


def _specification(answer_type: AnswerType) -> TaskSpec:
    if answer_type == AnswerType.STATE:
        verifier = structured_exact(1)
    elif answer_type == AnswerType.NATIVE_ACTION:
        verifier = predicted_action_verifier((FunctionCall(name="finish", arguments={"answer": "done"}),))
    else:
        verifier = exact_answer("done")
    return TaskSpec(
        id="two-providers",
        context=ConversationInput(events=(TextMessage(role="user", content="Use the tools."),)),
        environment_requirements=EnvironmentRequirements(),
        tool_providers={
            "alpha": ProviderRequirement(action_interface=AlphaProvider.ACTION_INTERFACE, seed_sha256="a" * 64),
            "beta": ProviderRequirement(action_interface=BetaProvider.ACTION_INTERFACE, seed_sha256="b" * 64),
        },
        answer_type=answer_type,
        verifier=verifier,
        final_tools=(
            (
                FunctionDefinition(
                    name="finish",
                    parameters={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                ),
            )
            if answer_type == AnswerType.NATIVE_ACTION
            else ()
        ),
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def _call(name: str, call_id: str, arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def test_two_providers_dispatch_and_grade_named_state(tmp_path, monkeypatch, service_processes, runtime_factory):
    binding = HarborEnvironmentConfig(
        tool_providers={
            "alpha": _binding(AlphaProvider, runtime_factory),
            "beta": _binding(BetaProvider, runtime_factory),
        }
    )
    task = lower_to_harbor(
        _specification(AnswerType.STATE),
        ProviderState(id="state", provider="beta"),
        binding,
        tmp_path / "task",
    )
    responses = iter(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("increment_alpha", "a1"), _call("increment_beta", "b1")],
            },
            {"role": "assistant", "content": "Done."},
        ]
    )
    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"choices": [{"message": next(responses)}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, binding, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "run"
    )

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["increment_alpha", "increment_beta"]
    assert [message["tool_call_id"] for message in requests[1]["messages"] if message["role"] == "tool"] == [
        "a1",
        "b1",
    ]
    assert [
        json.loads(message["content"])["value"] for message in requests[1]["messages"] if message["role"] == "tool"
    ] == [
        1,
        1,
    ]


@pytest.mark.parametrize("beta_calls,expected_reward", [(0, 0.0), (2, 0.0)])
async def test_provider_state_noop_and_wrong_result_score_zero(
    tmp_path, monkeypatch, beta_calls, expected_reward, service_processes, runtime_factory
):
    binding = HarborEnvironmentConfig(
        tool_providers={
            "alpha": _binding(AlphaProvider, runtime_factory),
            "beta": _binding(BetaProvider, runtime_factory),
        }
    )
    task = lower_to_harbor(
        _specification(AnswerType.STATE), ProviderState(id="state", provider="beta"), binding, tmp_path / "task"
    )
    calls = [_call("increment_beta", f"b{index}") for index in range(beta_calls)]
    responses = iter(
        ([{"role": "assistant", "content": None, "tool_calls": calls}] if calls else [])
        + [{"role": "assistant", "content": "Done."}]
    )

    def respond(request, timeout):
        return BytesIO(json.dumps({"choices": [{"message": next(responses)}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, binding, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "run"
    )

    assert result.verifier_result.rewards == {"reward": expected_reward}
    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "graded"


async def test_unavailable_provider_state_remains_ungraded(tmp_path, monkeypatch, service_processes, runtime_factory):
    binding = HarborEnvironmentConfig(
        tool_providers={
            "alpha": _binding(AlphaProvider, runtime_factory),
            "beta": _binding(BrokenStateProvider, runtime_factory),
        }
    )
    task = lower_to_harbor(
        _specification(AnswerType.STATE), ProviderState(id="state", provider="beta"), binding, tmp_path / "task"
    )

    def respond(request, timeout):
        return BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": "Done."}}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, binding, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "run"
    )

    assert result.verifier_result is None
    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "infra_error"
    assert outcome["reward"] is None


async def test_added_provider_tool_does_not_change_bound_surface(
    tmp_path, monkeypatch, service_processes, runtime_factory
):
    alpha = ToolBinding(
        action_interface=ExpandedAlphaProvider.ACTION_INTERFACE,
        seed_sha256=ExpandedAlphaProvider.SEED_SHA256,
        runtime=runtime_factory(
            ExpandedAlphaProvider.ACTION_INTERFACE,
            ExpandedAlphaProvider.SEED_SHA256,
            ExpandedAlphaProvider.PROVIDER_REVISION,
            ExpandedAlphaProvider.TOOL_DEFINITIONS,
        ),
        tool_definitions=tuple(ExpandedAlphaProvider.TOOL_DEFINITIONS),
        state_available=True,
        provider_revision=ExpandedAlphaProvider.PROVIDER_REVISION,
        tools=("increment_alpha",),
        tools_sha256=_digest(AlphaProvider.TOOL_DEFINITIONS),
    )
    binding = HarborEnvironmentConfig(tool_providers={"alpha": alpha, "beta": _binding(BetaProvider, runtime_factory)})
    task = lower_to_harbor(
        _specification(AnswerType.TEXT),
        PlainText(id="plain"),
        binding,
        tmp_path / "task",
    )
    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": "done"}}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, binding, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "run"
    )

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["increment_alpha", "increment_beta"]


async def test_provider_call_can_precede_terminal_answer_call(tmp_path, monkeypatch, service_processes, runtime_factory):
    binding = HarborEnvironmentConfig(
        tool_providers={
            "alpha": _binding(AlphaProvider, runtime_factory),
            "beta": _binding(BetaProvider, runtime_factory),
        }
    )
    task = lower_to_harbor(
        _specification(AnswerType.TEXT),
        AnswerCall(id="answer-call"),
        binding,
        tmp_path / "task",
    )
    responses = iter(
        [
            {"role": "assistant", "content": None, "tool_calls": [_call("increment_alpha", "a1")]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("submit_answer", "final", '{"answer":"done"}')],
            },
        ]
    )
    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"choices": [{"message": next(responses)}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, binding, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "run"
    )

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == [
        "increment_alpha",
        "increment_beta",
        "submit_answer",
    ]
    submission = ConversationTrace.model_validate_json((tmp_path / "trials/run/agent/submission.json").read_text())
    assert isinstance(submission.events[-1], AssistantToolCalls)
    assert submission.events[-1].calls[0].call_id == "final"


async def test_provider_call_can_precede_source_final_action(tmp_path, monkeypatch, service_processes, runtime_factory):
    binding = HarborEnvironmentConfig(
        tool_providers={
            "alpha": _binding(AlphaProvider, runtime_factory),
            "beta": _binding(BetaProvider, runtime_factory),
        }
    )
    task = lower_to_harbor(
        _specification(AnswerType.NATIVE_ACTION),
        FinalAction(id="final-action"),
        binding,
        tmp_path / "task",
    )
    responses = iter(
        [
            {"role": "assistant", "content": None, "tool_calls": [_call("increment_alpha", "a1")]},
            {"role": "assistant", "content": None, "tool_calls": [_call("finish", "final", '{"answer":"done"}')]},
        ]
    )

    def respond(request, timeout):
        return BytesIO(json.dumps({"choices": [{"message": next(responses)}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, binding, ChatLaunch(model="model", api_base="https://example.invalid"), tmp_path / "trials", "run"
    )

    assert result.verifier_result.rewards == {"reward": 1.0}
    submission = ConversationTrace.model_validate_json((tmp_path / "trials/run/agent/submission.json").read_text())
    assert isinstance(submission.events[-1], AssistantToolCalls)
    assert submission.events[-1].calls[0].name == "finish"


async def test_composite_keeps_workspace_operations_outside_tool_providers(tmp_path, runtime_factory):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    composite = CompositeToolEnvironment(
        environment_dir=environment_dir,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(UPath(tmp_path / "trial")),
        task_env_config=EnvironmentConfig(),
        tool_providers={
            "alpha": _binding(AlphaProvider, runtime_factory).model_dump(mode="json"),
        },
    )

    result = await composite.exec("pwd")
    with pytest.raises(ValueError, match="No filesystem provider"):
        await composite.upload_dir(tmp_path / "source", "/workspace")

    assert result.stdout == "/app\n"


async def test_malformed_provider_response_remains_ungraded_with_raw_trace(
    tmp_path, monkeypatch, service_processes, runtime_factory
):
    beta = _binding(BetaProvider, runtime_factory).model_copy(
        update={
            "runtime": runtime_factory(
                BetaProvider.ACTION_INTERFACE,
                BetaProvider.SEED_SHA256,
                BetaProvider.PROVIDER_REVISION,
                BetaProvider.TOOL_DEFINITIONS,
                fault=ServiceFault.WRONG_RESPONSE_ID,
            )
        }
    )
    config = HarborEnvironmentConfig(tool_providers={"alpha": _binding(AlphaProvider, runtime_factory), "beta": beta})
    task = lower_to_harbor(
        _specification(AnswerType.STATE), ProviderState(id="state", provider="beta"), config, tmp_path / "task"
    )

    def respond(request, timeout):
        return BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": "Done."}}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, config, ChatLaunch(model="test", api_base="https://example.invalid"), tmp_path / "trials", "bad"
    )
    assert result.verifier_result is None
    outcome = json.loads((tmp_path / "trials/bad/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "infra_error" and outcome["reward"] is None
    trace = [json.loads(line) for line in (tmp_path / "trials/bad/agent/provider-beta.jsonl").read_text().splitlines()]
    assert any(json.loads(event["raw_response"]).get("id") == "invalid" for event in trace if "raw_response" in event)


async def test_service_action_failure_retains_call_once_without_final_message(
    tmp_path, monkeypatch, service_processes, runtime_factory
):
    beta = _binding(BetaProvider, runtime_factory).model_copy(
        update={
            "runtime": runtime_factory(
                BetaProvider.ACTION_INTERFACE,
                BetaProvider.SEED_SHA256,
                BetaProvider.PROVIDER_REVISION,
                BetaProvider.TOOL_DEFINITIONS,
                fault=ServiceFault.CALL_FAILURE,
            )
        }
    )
    config = HarborEnvironmentConfig(tool_providers={"alpha": _binding(AlphaProvider, runtime_factory), "beta": beta})
    task = lower_to_harbor(
        _specification(AnswerType.STATE), ProviderState(id="state", provider="beta"), config, tmp_path / "task"
    )

    def respond(request, timeout):
        return BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [_call("increment_beta", "failed-1")],
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task, config, ChatLaunch(model="test", api_base="https://example.invalid"), tmp_path / "trials", "failed"
    )
    assert result.exception_info is not None and result.verifier_result is None
    assert not (tmp_path / "trials/failed/agent/submission.json").exists()
    events = [
        json.loads(line) for line in (tmp_path / "trials/failed/agent/provider-beta.jsonl").read_text().splitlines()
    ]
    calls = [event["request"] for event in events if event.get("request", {}).get("method") == "call"]
    assert [request["params"] for request in calls] == [
        {"name": "increment_beta", "arguments": "{}", "call_id": "failed-1"}
    ]
    assert any("error" in json.loads(event["raw_response"]) for event in events if "raw_response" in event)
