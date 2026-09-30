# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two independent tool services through one Harbor chat trial."""

import hashlib
import json
from io import BytesIO
from typing import ClassVar

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
    FinalTools,
    FunctionCall,
    FunctionDefinition,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import AnswerCall, FinalAction, PlainText, ProviderState
from taskcompendium.verifiers.predicted_action import predicted_action_verifier


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

    def __init__(self, *, seed_sha256: str, action_interface: str):
        if (seed_sha256, action_interface) != (self.SEED_SHA256, self.ACTION_INTERFACE):
            raise ValueError("Alpha binding differs")
        self.value = 0

    async def native_tool_definitions(self) -> list[dict]:
        return list(self.TOOL_DEFINITIONS)

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        assert name == "increment_alpha"
        self.value += 1
        return json.dumps({"value": self.value})

    def canonical_state(self) -> int:
        return self.value


class BetaProvider(AlphaProvider):
    ACTION_INTERFACE = "beta:v1"
    SEED_SHA256 = "b" * 64
    PROVIDER_REVISION = "beta-1"
    TOOL_DEFINITIONS = (_definition("increment_beta"),)

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        assert name == "increment_beta"
        self.value += 1
        return json.dumps({"value": self.value})


class ExpandedAlphaProvider(AlphaProvider):
    PROVIDER_REVISION = "alpha-2"
    TOOL_DEFINITIONS = (*AlphaProvider.TOOL_DEFINITIONS, _definition("unrelated"))


class BrokenStateProvider(BetaProvider):
    PROVIDER_REVISION = "beta-broken"

    def canonical_state(self) -> int:
        raise RuntimeError("state unavailable")


class ManagedProvider(AlphaProvider):
    ACTION_INTERFACE = "managed:v1"
    SEED_SHA256 = "f" * 64
    PROVIDER_REVISION = "managed-1"
    TOOL_DEFINITIONS = (_definition("managed_tool"),)
    events: ClassVar[list[str]] = []

    async def start(self) -> None:
        self.events.append("managed-start")

    async def stop(self) -> None:
        self.events.append("managed-stop")


class FailingProvider(ManagedProvider):
    ACTION_INTERFACE = "failing:v1"
    SEED_SHA256 = "e" * 64
    PROVIDER_REVISION = "failing-1"
    TOOL_DEFINITIONS = (_definition("fail_tool"),)

    async def start(self) -> None:
        self.events.append("failing-start")
        raise RuntimeError("provider failed to start")

    async def stop(self) -> None:
        self.events.append("failing-stop")


def _binding(provider: type[AlphaProvider]) -> ToolBinding:
    return ToolBinding(
        action_interface=provider.ACTION_INTERFACE,
        seed_sha256=provider.SEED_SHA256,
        provider=f"python:{__name__}:{provider.__name__}",
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
            FinalTools(
                functions=(
                    FunctionDefinition(
                        name="finish",
                        parameters={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    ),
                )
            )
            if answer_type == AnswerType.NATIVE_ACTION
            else FinalTools()
        ),
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def _call(name: str, call_id: str, arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def test_two_providers_dispatch_and_grade_named_state(tmp_path, monkeypatch):
    binding = HarborEnvironmentConfig(tool_providers={"alpha": _binding(AlphaProvider), "beta": _binding(BetaProvider)})
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
async def test_provider_state_noop_and_wrong_result_score_zero(tmp_path, monkeypatch, beta_calls, expected_reward):
    binding = HarborEnvironmentConfig(tool_providers={"alpha": _binding(AlphaProvider), "beta": _binding(BetaProvider)})
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


async def test_unavailable_provider_state_remains_ungraded(tmp_path, monkeypatch):
    binding = HarborEnvironmentConfig(
        tool_providers={"alpha": _binding(AlphaProvider), "beta": _binding(BrokenStateProvider)}
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


async def test_added_provider_tool_does_not_change_bound_surface(tmp_path, monkeypatch):
    alpha = ToolBinding(
        action_interface=ExpandedAlphaProvider.ACTION_INTERFACE,
        seed_sha256=ExpandedAlphaProvider.SEED_SHA256,
        provider=f"python:{__name__}:ExpandedAlphaProvider",
        provider_revision=ExpandedAlphaProvider.PROVIDER_REVISION,
        tools=("increment_alpha",),
        tools_sha256=_digest(AlphaProvider.TOOL_DEFINITIONS),
    )
    binding = HarborEnvironmentConfig(tool_providers={"alpha": alpha, "beta": _binding(BetaProvider)})
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


async def test_provider_call_can_precede_terminal_answer_call(tmp_path, monkeypatch):
    binding = HarborEnvironmentConfig(tool_providers={"alpha": _binding(AlphaProvider), "beta": _binding(BetaProvider)})
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


async def test_provider_call_can_precede_source_final_action(tmp_path, monkeypatch):
    binding = HarborEnvironmentConfig(tool_providers={"alpha": _binding(AlphaProvider), "beta": _binding(BetaProvider)})
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


async def test_composite_keeps_workspace_operations_outside_tool_providers(tmp_path):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    composite = CompositeToolEnvironment(
        environment_dir=environment_dir,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(UPath(tmp_path / "trial")),
        task_env_config=EnvironmentConfig(),
        tool_providers={
            "managed": _binding(ManagedProvider).model_dump(mode="json"),
            "alpha": _binding(AlphaProvider).model_dump(mode="json"),
        },
    )

    result = await composite.exec("pwd")
    with pytest.raises(ValueError, match="No filesystem provider"):
        await composite.upload_dir(tmp_path / "source", "/workspace")

    assert result.stdout == "/app\n"


async def test_composite_cleans_started_providers_after_start_failure(tmp_path):
    ManagedProvider.events = []
    composite = CompositeToolEnvironment(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(UPath(tmp_path / "trial")),
        task_env_config=EnvironmentConfig(),
        tool_providers={
            "managed": _binding(ManagedProvider).model_dump(mode="json"),
            "failing": _binding(FailingProvider).model_dump(mode="json"),
        },
    )

    with pytest.raises(RuntimeError, match="provider failed to start"):
        await composite.start(False)

    assert ManagedProvider.events == ["managed-start", "failing-start", "failing-stop", "managed-stop"]
