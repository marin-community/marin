# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import httpx
import openai
import pytest
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import PackageSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    FunctionDefinition,
    NoGrader,
    ProviderRequirement,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TextMessage,
)

from rolloutengine.chat import ChatModel
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.assistant_prefix import AssistantPrefixHarness
from rolloutengine.harness.exact_chat import EXACT_CHAT_HARNESS, ExactChatHarness
from rolloutengine.harness.shell import ShellHarness, ToolCallResult, ToolContext
from rolloutengine.models import (
    Continuation,
    GradedRolloutInterrupted,
    ModelRequest,
    RolloutFailure,
    RolloutOperation,
    TokenUsage,
    TraceMode,
)
from rolloutengine.test_support import ReplayModel, arithmetic_task, machine_runtime, rollout_spec

LOOKUP = FunctionDefinition(
    name="lookup",
    description="Look up the value of a key.",
    parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
)
LOOKUP_INTERFACE = "lookup/v1"


def completion(message: dict[str, Any], *, finish_reason: str = "stop", usage: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "completion-1",
        "object": "chat.completion",
        "created": 0,
        "model": "chat-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    return body


@dataclass
class ChatServer:
    """An OpenAI-compatible chat endpoint that answers with ``responses`` in order and keeps each request body."""

    responses: list[dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self.responses[len(self.requests) - 1])

    def model(self, **extra_body: Any) -> ChatModel:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://chat.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )
        return ChatModel(client, "chat-model", max_tokens=1024, extra_body=extra_body)


class StatusGraderFactory(DockerMachineFactory):
    """Run the fixture grader against ShellSim's filesystem instead of pulling an image."""

    async def create(self, spec):
        return await ShellSimMachineFactory().create(replace(spec, source=PackageSpec()))


def status_graded_task():
    command = (
        'jq -s \'{reward: (if .[1][-1].content == "12" and .[0].truncated == false and '
        ".[0].provider_refusal == false and (.[0].tool_events | length) == 0 "
        "then 1 else 0 end), detail: .[0]}' "
        "/app/answer_status.json /tests/conversation.json > /logs/verifier/reward.json"
    )
    return arithmetic_task(
        grader=ScriptGrader(
            argv=("sh", "-c", command),
            environment=EnvironmentRequirements(
                docker_image="fixture@sha256:" + "0" * 64,
                command_semantics=CommandSemantics.LINUX_PROCESS,
            ),
            answer_path=None,
            reply_status_path="/app/answer_status.json",
            reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),)),
        )
    )


@pytest.mark.parametrize(
    "finish_reason,message_fields,response_fields,reward",
    [
        ("stop", {}, {}, 1.0),
        ("length", {}, {}, 0.0),
        ("stop", {"refusal": "Cannot answer"}, {}, 0.0),
        (
            "tool_calls",
            {"tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {},
            0.0,
        ),
        ("stop", {"annotations": [{"type": "url_citation", "url": "https://example.test"}]}, {}, 0.0),
        ("stop", {}, {"provider_metadata": {"events": [{"type": "web_search_call"}]}}, 0.0),
    ],
    ids=["complete", "truncated", "refusal", "tool-call", "message-annotations", "provider-tool-event"],
)
async def test_script_grader_receives_reply_status_from_chat_transport(
    finish_reason, message_fields, response_fields, reward
):
    response = {**completion({"content": "12", **message_fields}, finish_reason=finish_reason), **response_fields}
    model = ChatServer([response]).model()
    runtime = rollout_spec(
        status_graded_task(),
        harness=EXACT_CHAT_HARNESS,
        verifier_machine=machine_runtime(backend="docker"),
        trace=TraceMode.MESSAGES,
    )
    try:
        result = await RolloutEngine(
            {"docker": StatusGraderFactory()}, {EXACT_CHAT_HARNESS: ExactChatHarness()}
        ).run_task(runtime, model)
    finally:
        await model.client.close()
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward), result.grade
    assert result.grade.detail["origin"] == "provider"
    assert result.grade.detail["finish_reason"] == finish_reason
    saved = json.loads(json.dumps(asdict(result.rollout)))
    assert saved["conversations"][-1]["steps"][-1]["turn"]["reply_status"] == result.grade.detail


async def test_required_reply_status_is_not_inferred_from_answer_text():
    runtime = rollout_spec(
        status_graded_task(), harness=EXACT_CHAT_HARNESS, verifier_machine=machine_runtime(backend="docker")
    )
    model = ReplayModel([{"role": "assistant", "content": "12"}])
    with pytest.raises(GradedRolloutInterrupted) as caught:
        await RolloutEngine({"docker": StatusGraderFactory()}, {EXACT_CHAT_HARNESS: ExactChatHarness()}).run_task(
            runtime, model.complete
        )
    assert caught.value.operation == RolloutOperation.GRADE
    assert caught.value.grade.reward is None


@dataclass(frozen=True)
class Lookup:
    values: Mapping[str, str]

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return (LOOKUP,)

    async def open(self, context: ToolContext) -> "Lookup":
        return self

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        assert name == LOOKUP.name
        return ToolCallResult(self.values[str(arguments["key"])])

    async def close(self) -> None:
        pass


async def test_a_chat_turn_is_one_completion_whose_message_the_next_request_can_send_back():
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "a"}'}}
    server = ChatServer(
        [
            completion(
                {"content": None, "tool_calls": [tool_call], "reasoning_content": "Look it up.", "refusal": "no"},
                finish_reason="tool_calls",
            )
        ]
    )
    model = server.model(chat_template_kwargs={"reasoning_effort": "low"}, top_k=20)
    request = ModelRequest(
        ({"role": "user", "content": "What is a?"},),
        {
            "tools": [{"type": "function", "function": LOOKUP.model_dump(exclude_none=True)}],
            "max_tokens": 64,
            "extra_body": {"top_k": 5, "prompt_cache_key": "attempt-1"},
        },
    )

    turn = await model(request)

    [sent] = server.requests
    assert (sent["max_tokens"], sent["top_k"], sent["prompt_cache_key"]) == (64, 5, "attempt-1")
    assert sent["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert turn.message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call],
        "reasoning_content": "Look it up.",
    }
    assert turn.metadata == {"response_id": "completion-1", "refusal": "no"}
    assert (turn.stop_reason, turn.usage) == ("tool_calls", TokenUsage(12, 5))


@pytest.mark.parametrize(
    "request_",
    [
        ModelRequest((), {}, continuation=Continuation((1, 2), 0)),
        ModelRequest((), {"stream": True}),
        ModelRequest((), {"extra_body": {"messages": []}}),
    ],
    ids=["token-continuation", "reserved-option", "reserved-extra-body-key"],
)
async def test_a_chat_model_rejects_a_request_that_it_cannot_serve_before_it_sends_it(request_):
    server = ChatServer([])
    with pytest.raises(ValueError):
        await server.model()(request_)
    assert server.requests == []


async def test_a_managed_agent_runs_host_tools_against_a_chat_endpoint_with_no_machine():
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "a"}'}}
    server = ChatServer(
        [
            completion(
                {"content": None, "tool_calls": [tool_call], "reasoning": "Look up a before answering."},
                finish_reason="tool_calls",
            ),
            completion({"content": "The value of a is 12."}),
        ]
    )
    task = arithmetic_task(
        environment_requirements=EnvironmentRequirements(
            tool_providers={"values": ProviderRequirement(action_interface=LOOKUP_INTERFACE, initial_state=None)}
        ),
        grader=NoGrader(reason="The agent's work has no grader"),
    )
    engine = RolloutEngine(
        {}, {"agent": ShellHarness(command_timeout=None, tools={LOOKUP_INTERFACE: Lookup({"a": "12"})})}
    )

    result = await engine.run_task(
        rollout_spec(task, harness="agent", trace=TraceMode.MESSAGES, max_total_tokens=100), server.model()
    )

    assert (result.grade.status, result.grade.error) == (Outcome.UNAVAILABLE, "The agent's work has no grader")
    assert result.rollout.conversations[-1].messages[-1] == {"role": "assistant", "content": "The value of a is 12."}
    assert server.requests[1]["messages"][-2:] == [
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "12"},
    ]


async def test_a_response_without_usage_is_a_model_failure():
    server = ChatServer([completion({"content": "12"}, usage=False)])
    engine = RolloutEngine({}, {"agent": ShellHarness(command_timeout=None)})
    spec = rollout_spec(arithmetic_task(), harness="agent", trace=TraceMode.MESSAGES, max_total_tokens=100)

    with pytest.raises(GradedRolloutInterrupted) as caught:
        await engine.run_task(spec, server.model())

    assert caught.value.rollout.failure == RolloutFailure(
        "ModelResponseError", {"operation": "model", "error": "The chat endpoint reported no token usage"}
    )


async def test_assistant_prefix_harness_reaches_chat_endpoint_as_text_without_pinned_tokens():
    server = ChatServer([completion({"content": "2"})])
    task = arithmetic_task(
        context=ConversationInput(
            events=(
                TextMessage(role="user", content="What is six plus six?"),
                TextMessage(role="assistant", content="1"),
            )
        ),
        grader=NoGrader(reason="Inspect continuation transport"),
    )
    model = server.model()
    try:
        result = await RolloutEngine({}, {"assistant_prefix": AssistantPrefixHarness()}).run_task(
            rollout_spec(task, harness="assistant_prefix", trace=TraceMode.MESSAGES), model
        )
    finally:
        await model.client.close()
    assert server.requests[0]["messages"][-1] == {"role": "assistant", "content": "1"}
    assert server.requests[0]["continue_final_message"] is True
    assert server.requests[0]["add_generation_prompt"] is False
    assert "prompt_token_ids" not in server.requests[0]
    assert result.rollout.conversations[0].messages[-1] == {"role": "assistant", "content": "2"}


@pytest.mark.parametrize("channel", ["reasoning_content", "reasoning"])
async def test_assistant_prefix_rejects_reasoning_parsed_by_a_chat_endpoint(channel):
    server = ChatServer([completion({"content": "", channel: "2"})])
    task = arithmetic_task(
        context=ConversationInput(
            events=(
                TextMessage(role="user", content="Complete the answer."),
                TextMessage(role="assistant", content="1"),
            )
        ),
        grader=NoGrader(reason="Inspect continuation transport"),
    )
    model = server.model()
    try:
        with pytest.raises(GradedRolloutInterrupted) as caught:
            await RolloutEngine({}, {"assistant_prefix": AssistantPrefixHarness()}).run_task(
                rollout_spec(task, harness="assistant_prefix", trace=TraceMode.MESSAGES), model
            )
    finally:
        await model.client.close()
    assert caught.value.rollout.failure.exception_type == "ValueError"


async def test_an_uncapped_chat_request_omits_both_output_limit_fields():
    server = ChatServer([completion({"content": "12"})])
    model = replace(server.model(), max_tokens=None)
    async with model.client:
        result = await model(ModelRequest(messages=({"role": "user", "content": "Return 12"},), options={}))
    assert result.message["content"] == "12"
    assert "max_tokens" not in server.requests[0] and "max_completion_tokens" not in server.requests[0]


@pytest.mark.parametrize(("context_window", "stop_reason"), [(17, "length"), (18, "tool_calls"), (None, "tool_calls")])
async def test_a_tool_call_that_fills_the_context_window_is_a_length_stop(context_window, stop_reason):
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    server = ChatServer([completion({"content": None, "tool_calls": [tool_call]}, finish_reason="tool_calls")])
    model = replace(server.model(), context_window=context_window)
    async with model.client:
        turn = await model(ModelRequest(messages=({"role": "user", "content": "What is a?"},), options={}))
    assert turn.stop_reason == stop_reason


async def test_exact_chat_preserves_source_conversation_without_appending_submission_instructions():
    server = ChatServer([completion({"content": "12"})])
    task = arithmetic_task(
        context=ConversationInput(
            events=(
                TextMessage(role="system", content="Preserve this system instruction."),
                TextMessage(role="user", content="  What is six plus six?\nAnswer:"),
            )
        )
    )
    model = server.model()
    try:
        result = await RolloutEngine({}, {EXACT_CHAT_HARNESS: ExactChatHarness()}).run_task(
            rollout_spec(task, harness=EXACT_CHAT_HARNESS, trace=TraceMode.MESSAGES), model
        )
    finally:
        await model.client.close()
    assert server.requests[0]["messages"] == [
        {"role": "system", "content": "Preserve this system instruction."},
        {"role": "user", "content": "  What is six plus six?\nAnswer:"},
    ]
    assert result.grade.status is Outcome.GRADED
    assert result.grade.reward == 1
