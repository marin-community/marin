# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import socket
from contextvars import Context
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from rigging.inference_clock import InferenceClock, current_inference_clock
from shellbox.http_service import HttpRequest, HttpResponse
from shellbox.machine import Command, ExitReason, Result
from shellbox.process_session import SubprocessSession
from taskcompendium.importers.harbor.package import HARBOR_SHELL_PREFIX
from taskcompendium.models import ConversationInput, ProviderRequirement, TextMessage

from rolloutengine.agents.environment import MachineEnvironment
from rolloutengine.agents.model_relay import ModelRelay, VllmTokenizer, chat_prompt
from rolloutengine.agents.pi.agent import (
    MCP_INTERFACE,
    AgentKilledBySignalError,
    PiAgent,
    PiOptions,
    PiThinking,
    PiThinkingFormat,
    mcp_servers,
    pi_harness,
)
from rolloutengine.harness.external import AgentHarness, LengthStopPolicy, ReplyMatch
from rolloutengine.models import (
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    ContextWindowExceeded,
    ModelRequest,
    ModelTurn,
    Prompt,
    RolloutEnded,
    RolloutInterrupted,
    TokenUsage,
    TraceMode,
)
from rolloutengine.recorder import harness_rollout
from rolloutengine.test_support import (
    REPLAY_MODEL,
    ReplayEndpoint,
    ReplayMachine,
    arithmetic_task,
    recorded_commands,
    recorded_requests,
    replay_rollout,
    replay_scenarios,
    rollout_limits,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pi"
AGENT_REPLAY = replay_scenarios(FIXTURES / "harbor_replay.json")
RELAY_REPLAY = replay_scenarios(FIXTURES / "harbor_relay_replay.json")
MODEL = REPLAY_MODEL
CONTEXT_REJECTION = (
    "This model's maximum context length is 32768 tokens. However, your request has 40000 input tokens. "
    "Please reduce the length of the input messages."
)
# The lifted harness also creates Harbor's agent log directory, which Harbor's trial creates before the agent runs.
LOG_DIRECTORY_EXCHANGE = {
    "command": "mkdir -p /logs/agent && chmod 777 /logs/agent",
    "user": "root",
    "env": {},
    "cwd": None,
    "timeout": None,
    "exit_code": 0,
    "stdout": "",
    "stderr": "",
}


def task(instruction: str = "Say done.", **update):
    return arithmetic_task(context=ConversationInput(events=(TextMessage(role="user", content=instruction),)), **update)


def pi_options(recorded: dict[str, Any], **overrides) -> PiOptions:
    return PiOptions(
        model=MODEL,
        thinking_format=PiThinkingFormat(recorded["thinking_format"]),
        max_input_tokens=recorded["max_input_tokens"],
        max_output_tokens=recorded["max_output_tokens"],
        compaction_reserve_tokens=recorded.get("compaction_reserve_tokens"),
        version=recorded.get("version"),
        thinking=None if recorded.get("thinking") is None else PiThinking(recorded["thinking"]),
        mcp_excluded_tools=tuple(recorded.get("mcp_excluded_tools", ())),
        **overrides,
    )


def mcp_provider(transport: str, url: str) -> ProviderRequirement:
    return ProviderRequirement(action_interface=MCP_INTERFACE, initial_state={"transport": transport, "url": url})


@pytest.mark.parametrize("scenario", sorted(AGENT_REPLAY))
async def test_pi_runs_harbors_commands_and_classifies_its_failures(scenario):
    recorded = AGENT_REPLAY[scenario]
    exchanges = [LOG_DIRECTORY_EXCHANGE, *recorded["machine"]]
    machine = ReplayMachine(exchanges, HARBOR_SHELL_PREFIX)
    providers = {server["name"]: mcp_provider(server["transport"], server["url"]) for server in recorded["mcp_servers"]}
    agent = PiAgent(pi_options(recorded["options"]), MachineEnvironment(machine, "fixture"), mcp_servers(providers))

    outcome = None
    try:
        await agent.setup()
        await agent.run(recorded["instruction"])
    except Exception as error:
        outcome = type(error).__name__

    assert outcome == recorded["outcome"]
    assert machine.commands == recorded_commands(exchanges)


def answer_view(status: int, content_type: str, body: str) -> dict[str, Any]:
    """An answer with its JSON or its events parsed, without creation times.

    ``ChatModel`` keeps a ``null`` assistant content as an empty string, which Pi reads the same way.
    """
    if content_type != "text/event-stream":
        return {"status": status, "content_type": content_type, "body": json.loads(body)}
    events = []
    for line in body.split("\n\n"):
        if not line or line == "data: [DONE]":
            continue
        event = json.loads(line.removeprefix("data: "))
        del event["created"]
        for choice in event["choices"]:
            if choice["delta"].get("content") is None and "content" in choice["delta"]:
                choice["delta"]["content"] = ""
        events.append(event)
    return {"status": status, "content_type": content_type, "events": events}


@pytest.mark.parametrize("scenario", sorted(RELAY_REPLAY))
async def test_relay_serves_pis_requests_as_harbors_relay_does(scenario):
    recorded = RELAY_REPLAY[scenario]
    vllm = ReplayEndpoint(recorded["model"])
    answers: list[HttpResponse] = []

    def relay_harness(client: openai.AsyncOpenAI) -> AgentHarness:
        async def relay_requests(context, complete):
            relay = ModelRelay(complete, VllmTokenizer(client, MODEL))
            for request in recorded["relay_requests"]:
                body = json.dumps(request["body"]).encode()
                answers.append(await relay.handle(HttpRequest(request["method"], request["path"], body)))

        return AgentHarness(relay_requests, length_stop=LengthStopPolicy.NEW_CONVERSATION, reply_match=ReplyMatch.EXACT)

    try:
        await replay_rollout(relay_harness, recorded["instruction"], vllm)
    except RolloutInterrupted as interrupted:
        # A rejected first request leaves no turn, so the rollout stops before its first turn.
        assert not interrupted.rollout.conversations

    assert vllm.requests == recorded_requests(recorded["model"])
    assert [answer_view(answer.status, answer.content_type, answer.body.decode()) for answer in answers] == [
        answer_view(answer["status"], answer["content_type"], answer["body"]) for answer in recorded["answers"]
    ]


def test_chat_prompt_keeps_openai_parameters_and_moves_server_arguments_to_extra_body():
    payload = {
        "model": "pi-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
        "max_completion_tokens": 512,
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 20,
    }

    assert chat_prompt(payload) == Prompt(
        ({"role": "user", "content": "hi"},),
        {
            "max_completion_tokens": 512,
            "tools": payload["tools"],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
        },
    )


async def unavailable_tokenizer(payload: dict[str, Any]) -> Any:
    raise AssertionError("This request does not tokenize")


def turn(message: dict[str, Any], stop_reason: str = "stop", **metadata) -> ModelTurn:
    return ModelTurn(message, (), (), None, stop_reason, metadata=metadata, usage=TokenUsage(7, 3))


@pytest.mark.parametrize(
    ("method", "path", "body", "outcome", "status"),
    [
        ("GET", "/v1/chat/completions", b"{}", None, 404),
        ("POST", "/v1/responses", b"{}", None, 404),
        ("POST", "/v1/chat/completions", b"not json", None, 400),
        ("POST", "/v1/chat/completions", b"[]", None, 400),
        ("POST", "/v1/chat/completions", b'{"messages": [], "n": 2}', None, 400),
        ("POST", "/v1/chat/completions", b'{"messages": []}', ContextWindowExceeded("too long"), 400),
        ("POST", "/v1/chat/completions", b'{"messages": []}', RolloutEnded("ended"), 503),
    ],
)
async def test_relay_rejects_unsupported_requests_and_reports_context_rejections(method, path, body, outcome, status):
    async def complete(prompt: Prompt) -> ModelTurn:
        assert outcome is not None, "The relay must not send this request"
        raise outcome

    response = await ModelRelay(complete, unavailable_tokenizer).handle(HttpRequest(method, path, body))

    assert response.status == status
    if isinstance(outcome, ContextWindowExceeded):
        assert json.loads(response.body)["error"]["code"] == "context_length_exceeded"


async def test_relay_callback_keeps_trial_clock_across_http_task_context():
    clock = InferenceClock(2280)

    async def complete(prompt: Prompt) -> ModelTurn:
        assert current_inference_clock() is clock
        return turn({"role": "assistant", "content": "12"})

    with clock.scope():
        relay = ModelRelay(complete, unavailable_tokenizer)
        callback = asyncio.create_task(
            relay.handle(HttpRequest("POST", "/v1/chat/completions", b'{"messages": []}')),
            context=Context(),
        )
        response = await callback
    assert json.loads(response.body)["choices"][0]["message"]["content"] == "12"


async def test_relay_streams_a_turn_with_its_tool_calls_reasoning_and_usage():
    call = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}

    async def complete(prompt: Prompt) -> ModelTurn:
        return turn({"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls", reasoning="Run it.")

    body = json.dumps({"model": "pi-model", "messages": [], "stream": True}).encode()
    response = await ModelRelay(complete, unavailable_tokenizer).handle(
        HttpRequest("POST", "/v1/chat/completions", body)
    )

    view = answer_view(response.status, response.content_type, response.body.decode())
    assert [event["choices"] for event in view["events"]] == [
        [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{**call, "index": 0}],
                    "reasoning": "Run it.",
                },
                "finish_reason": None,
            }
        ],
        [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        [],
    ]
    assert view["events"][-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


@pytest.mark.parametrize(
    ("status", "error", "expected_status", "expected_code"),
    [
        (400, {"message": CONTEXT_REJECTION, "code": 400}, 400, "context_length_exceeded"),
        (401, {"message": "bad key"}, 401, "authentication_error"),
        (500, {"message": "down"}, 502, None),
    ],
)
async def test_relay_hides_tokenizer_errors_but_keeps_their_classification(
    status, error, expected_status, expected_code
):
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": error})

    async with openai.AsyncOpenAI(
        api_key="host-secret",
        base_url="http://vllm.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        relay = ModelRelay(lambda prompt: pytest.fail("no chat request"), VllmTokenizer(client, MODEL))
        response = await relay.handle(HttpRequest("POST", "/tokenize", b'{"messages": []}'))

    assert response.status == expected_status
    assert b"host-secret" not in response.body and error["message"].encode() not in response.body
    assert (json.loads(response.body)["error"].get("code") if expected_code else None) == expected_code


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@dataclass
class SimulatedPiMachine:
    """A task machine whose image has Pi 0.87.0, and whose Pi sends one streamed chat request through the relay.

    The relay's guest listener runs as a local process, so the simulated Pi reaches it at the relay port. With
    ``second_tokenize``, Pi counts its next prompt after the first turn, then exits.
    """

    port: int
    exit_code: int
    second_tokenize: bool = False
    answers: list[httpx.Response] = field(default_factory=list)
    processes: list[asyncio.subprocess.Process] = field(default_factory=list)

    async def run(self, command: Command) -> Result:
        script = command.argv[4]
        stdout = b""
        if script.endswith("pi --version"):
            stdout = b"0.87.0\n"
        elif "mktemp -d" in script:
            stdout = b"/tmp/harbor-pi.Fixture1\n"
        elif "pi --print" in script:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=30) as client:
                messages = [{"role": "user", "content": "Say done."}]
                self.answers.append(await client.post("/tokenize", json={"model": MODEL, "messages": messages}))
                self.answers.append(
                    await client.post(
                        "/v1/chat/completions",
                        json={"model": MODEL, "messages": messages, "stream": True, "max_completion_tokens": 64},
                    )
                )
                if self.second_tokenize:
                    messages = [*messages, {"role": "assistant", "content": "Done."}, {"role": "user", "content": "?"}]
                    self.answers.append(await client.post("/tokenize", json={"model": MODEL, "messages": messages}))
        return Result(self.exit_code if "pi --print" in script else 0, stdout, b"", False, False, ExitReason.EXITED)

    async def open_process(self, command: Command) -> SubprocessSession:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ["PATH"]},
            start_new_session=True,
        )
        self.processes.append(process)

        async def stop() -> None:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

        return SubprocessSession(process, stop=stop)

    async def upload(self, source: Path, target: str) -> None:
        pass

    async def download(self, source: str, target: Path) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_pi_harness_serves_pi_through_the_relay_and_grades_its_failures(exit_code):
    machine = SimulatedPiMachine(available_port(), exit_code)
    requests: list[ModelRequest] = []
    tokenized: list[dict[str, Any]] = []

    async def model(request: ModelRequest) -> ModelTurn:
        requests.append(request)
        return turn({"role": "assistant", "content": "Done."})

    async def tokenize(payload: dict[str, Any]) -> Any:
        tokenized.append(payload)
        return {"count": 2, "tokens": [1, 2]}

    options = PiOptions(
        model=MODEL,
        thinking_format=PiThinkingFormat.CHAT_TEMPLATE,
        max_input_tokens=32768,
        max_output_tokens=8192,
        version="0.87.0",
        relay_port=machine.port,
    )
    rollout = await harness_rollout(
        pi_harness(options, tokenize),
        task(),
        model,
        limits=rollout_limits(max_turns=None),
        machine=machine,
        trace=TraceMode.MESSAGES,
    )

    assert [request.options for request in requests] == [{"max_completion_tokens": 64}]
    assert [len(conversation.steps) for conversation in rollout.conversations] == [1]
    assert rollout.failure is None
    assert tokenized == [{"model": MODEL, "messages": [{"role": "user", "content": "Say done."}]}]
    tokens, chat = machine.answers
    assert tokens.json() == {"count": 2, "tokens": [1, 2]}
    assert chat.headers["content-type"] == "text/event-stream"
    assert '"content": "Done."' in chat.text
    assert all(process.returncode is not None for process in machine.processes)


async def test_pi_agent_deadline_includes_time_before_its_first_model_request():
    class WaitingPi(SimulatedPiMachine):
        async def run(self, command: Command) -> Result:
            if "pi --print" in command.argv[4]:
                await asyncio.Future()
            return await super().run(command)

    async def unexpected_model(_request):
        raise AssertionError("Pi never reached the model")

    machine = WaitingPi(available_port(), 0)
    options = PiOptions(
        model=MODEL,
        thinking_format=PiThinkingFormat.CHAT_TEMPLATE,
        max_input_tokens=32768,
        max_output_tokens=8192,
        version="0.87.0",
        relay_port=machine.port,
        setup_timeout=5,
    )
    async with asyncio.timeout(5):
        rollout = await harness_rollout(
            pi_harness(options, unavailable_tokenizer),
            task(),
            unexpected_model,
            limits=rollout_limits(total_turn_timeout=0.1),
            machine=machine,
        )

    assert rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON
    assert not rollout.conversations
    assert machine.processes and all(process.returncode is not None for process in machine.processes)


async def test_pi_harness_fails_the_attempt_when_a_signal_kills_pi():
    machine = SimulatedPiMachine(available_port(), 137)

    async def model(request: ModelRequest) -> ModelTurn:
        return turn({"role": "assistant", "content": "Done."})

    async def tokenize(payload: dict[str, Any]) -> Any:
        return {"count": 2, "tokens": [1, 2]}

    options = PiOptions(
        model=MODEL,
        thinking_format=PiThinkingFormat.CHAT_TEMPLATE,
        max_input_tokens=32768,
        max_output_tokens=8192,
        relay_port=machine.port,
    )
    with pytest.raises(RolloutInterrupted) as interrupted:
        await harness_rollout(
            pi_harness(options, tokenize),
            task(),
            model,
            limits=rollout_limits(max_turns=None),
            machine=machine,
            trace=TraceMode.MESSAGES,
        )

    assert isinstance(interrupted.value.__cause__, AgentKilledBySignalError)


async def test_pi_harness_fails_the_attempt_with_the_error_of_a_failed_tokenizer():
    machine = SimulatedPiMachine(available_port(), 1, second_tokenize=True)
    tokenized: list[dict[str, Any]] = []

    async def model(request: ModelRequest) -> ModelTurn:
        return turn({"role": "assistant", "content": "Done."})

    async def tokenize(payload: dict[str, Any]) -> Any:
        tokenized.append(payload)
        if len(tokenized) > 1:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://vllm.test/tokenize"))
        return {"count": 2, "tokens": [1, 2]}

    options = PiOptions(
        model=MODEL,
        thinking_format=PiThinkingFormat.CHAT_TEMPLATE,
        max_input_tokens=32768,
        max_output_tokens=8192,
        relay_port=machine.port,
    )
    with pytest.raises(RolloutInterrupted) as interrupted:
        await harness_rollout(
            pi_harness(options, tokenize),
            task(),
            model,
            limits=rollout_limits(max_turns=None),
            machine=machine,
            trace=TraceMode.MESSAGES,
        )

    assert isinstance(interrupted.value.__cause__, openai.APIConnectionError)
    assert [len(conversation.steps) for conversation in interrupted.value.rollout.conversations] == [1]
    assert machine.answers[-1].status_code == 502


@pytest.mark.parametrize(
    ("providers", "error"),
    [
        ({"search": mcp_provider("sse", "http://search:8000/sse")}, None),
        ({"search": mcp_provider("stdio", "http://search:8000/sse")}, ValueError),
        ({"lookup": ProviderRequirement(action_interface="lookup/v1", initial_state={})}, NotImplementedError),
    ],
)
def test_pi_harness_accepts_only_the_tasks_http_mcp_servers(providers, error):
    harness = pi_harness(
        PiOptions(
            model=MODEL, thinking_format=PiThinkingFormat.CHAT_TEMPLATE, max_input_tokens=32768, max_output_tokens=8192
        ),
        unavailable_tokenizer,
    )
    spec = task()
    spec = spec.model_copy(
        update={
            "environment_requirements": spec.environment_requirements.model_copy(update={"tool_providers": providers})
        }
    )

    if error is None:
        harness.check(spec, rollout_limits())
    else:
        with pytest.raises(error):
            harness.check(spec, rollout_limits())
