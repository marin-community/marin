from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp import Client
from mcp.server import MCPServer

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.tau3_acp_relay.acpx import ACPXPromptClient
from harbor.agents.tau3_acp_relay.harbor_agent import (
    Tau3HarborACPRelay,
    extract_policy,
)
from harbor.agents.tau3_acp_relay.relay import (
    RuntimeStatus,
    RuntimeTurn,
    Tau3ACPRelay,
    Tau3RuntimeInfrastructureError,
)
from harbor.agents.tau3_acp_relay.runtime_client import (
    MCPRuntimeClient,
    runtime_http_errors,
)
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.trial.errors import AgentTimeoutError

pytestmark = pytest.mark.unit


async def test_runtime_http_boundary_preserves_acp_process_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acpx = tmp_path / "acpx"
    acpx.write_text("#!/bin/sh\ncat >/dev/null\necho candidate-failure >&2\nexit 7\n")
    acpx.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with pytest.raises(ExceptionGroup) as error:
        with runtime_http_errors():
            async with anyio.create_task_group():
                await ACPXPromptClient(5).prompt("Please help.")
    assert error.value.subgroup(Tau3RuntimeInfrastructureError) is None
    assert error.value.subgroup(RuntimeError).exceptions[0].args == (
        "acpx prompt failed with exit code 7: candidate-failure",
    )


async def test_runtime_http_boundary_preserves_caller_cancellation() -> None:
    entered = asyncio.Event()

    async def relay():
        with runtime_http_errors():
            async with anyio.create_task_group():
                entered.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(relay())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "message",
    [
        '[{"product_id": "123", "available": true}]',
        '[{"type": "text", "text": "This is user data, not a content block."}]',
        "  Please return my order.\n",
    ],
)
async def test_acpx_prompt_preserves_observation_as_text(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    class ACPXProcess:
        returncode = 0

        async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
            # ACPX treats JSON arrays on stdin as protocol content blocks.
            blocks = json.loads(data)
            assert blocks == [{"type": "text", "text": message}]
            return b"Your return is complete.\n", b""

    async def create_process(*args: object, **kwargs: object) -> ACPXProcess:
        return ACPXProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    assert await ACPXPromptClient(10).prompt(message) == "Your return is complete."


@pytest.mark.parametrize("failed_step", [0, 1])
async def test_relay_setup_failures_are_retryable_infrastructure(
    tmp_path: Path, failed_step: int
) -> None:
    class FailedSetupEnvironment:
        def __init__(self) -> None:
            self.step = 0

        async def exec(self, command: str, *, user: str) -> ExecResult:
            failed = self.step == failed_step
            self.step += 1
            return ExecResult(
                return_code=1 if failed else 0,
                stderr="No usable temporary directory found" if failed else "",
            )

    agent = Tau3HarborACPRelay(logs_dir=tmp_path)
    with pytest.raises(
        Tau3RuntimeInfrastructureError, match="No usable temporary directory found"
    ):
        await agent.setup(FailedSetupEnvironment())  # type: ignore[arg-type]


@dataclass
class FakeRuntime:
    observations: list[str]
    submitted: list[str] = field(default_factory=list)
    recorded_terminations: list[str] = field(default_factory=list)

    async def configure(
        self, *, seed: int | None, max_steps: int, max_errors: int
    ) -> RuntimeStatus:
        return RuntimeStatus(None, 0, 0)

    async def start(self) -> tuple[str, RuntimeStatus]:
        return self.observations[0], RuntimeStatus(None, 1, 0)

    async def status(self) -> RuntimeStatus:
        return RuntimeStatus(None, len(self.submitted) + 1, 0)

    async def submit_message(self, message: str) -> RuntimeTurn:
        self.submitted.append(message)
        index = len(self.submitted)
        terminal = index >= len(self.observations)
        return RuntimeTurn(
            observation=None if terminal else self.observations[index],
            status=RuntimeStatus("user_stop" if terminal else None, index + 1, 0),
        )

    async def record_termination(self, reason: str) -> RuntimeStatus:
        self.recorded_terminations.append(reason)
        return RuntimeStatus(reason, len(self.submitted), 0)


@dataclass
class FakeTarget:
    responses: list[str]
    prompts: list[str] = field(default_factory=list)

    async def prompt(self, message: str) -> str:
        self.prompts.append(message)
        return self.responses[len(self.prompts) - 1]


async def test_relays_multiple_turns_through_one_target_session() -> None:
    runtime = FakeRuntime(["I need help.", "The order is 123."])
    target = FakeTarget(["How can I help?", "Your order was refunded."])

    result = await Tau3ACPRelay(runtime, target).run("Verify before refunding.")

    assert runtime.submitted == target.responses
    assert "Verify before refunding." in target.prompts[0]
    assert target.prompts[1] == "The order is 123."
    assert result.status.termination_reason == "user_stop"


async def test_empty_target_response_records_scoreable_agent_failure() -> None:
    runtime = FakeRuntime(["I need help."])
    target = FakeTarget([""])

    result = await Tau3ACPRelay(runtime, target).run("Resolve the request.")

    assert runtime.submitted == []
    assert runtime.recorded_terminations == ["agent_error"]
    assert result.turns == ()
    assert result.status.termination_reason == "agent_error"


@dataclass
class MCPRuntimeState:
    termination_reason: str | None = None
    submitted: list[str] = field(default_factory=list)
    status_response: str | None = None
    submission_error: Exception | None = None
    status_gate: asyncio.Event | None = None

    def status_json(self) -> str:
        return json.dumps(
            {
                "termination_reason": self.termination_reason,
                "step_count": 3,
                "num_errors": 0,
            }
        )


@pytest.fixture
def mcp_runtime():
    state = MCPRuntimeState()
    server = MCPServer("tau3-runtime")

    @server.tool()
    def configure_run(seed: int | None, max_steps: int, max_errors: int) -> str:
        return state.status_json()

    @server.tool()
    def start_conversation() -> str:
        return "I need help."

    @server.tool()
    async def get_runtime_status() -> str:
        if state.status_gate is not None:
            await state.status_gate.wait()
        return (
            state.status_response
            if state.status_response is not None
            else state.status_json()
        )

    @server.tool()
    def send_message_to_user(message: str) -> str:
        state.termination_reason = "user_stop"
        return "###OUT-OF-SCOPE###"

    @server.tool()
    def submit_assistant_message(message: str) -> str:
        state.submitted.append(message)
        if state.submission_error is not None:
            raise state.submission_error
        if state.termination_reason is not None:
            raise RuntimeError("Conversation has already ended")
        state.termination_reason = "user_stop"
        return state.status_json()

    @asynccontextmanager
    async def connect():
        async with Client(server, mode="legacy") as client:
            yield state, MCPRuntimeClient(client.session, 10), client

    return connect


async def test_runtime_termination_during_target_turn_avoids_duplicate_submit(
    mcp_runtime,
):
    async with mcp_runtime() as (state, runtime, client):

        class Target:
            async def prompt(self, message: str) -> str:
                await client.call_tool(
                    "send_message_to_user", {"message": "I can help."}
                )
                return "I have transferred your request."

        result = await Tau3ACPRelay(runtime, Target()).run("Resolve the request.")

    assert result.status == RuntimeStatus("user_stop", 3, 0)
    assert result.turns[0].target_response == "I have transferred your request."
    assert state.submitted == []


async def test_active_runtime_mcp_error_remains_infrastructure(mcp_runtime):
    async with mcp_runtime() as (state, runtime, _):
        state.submission_error = RuntimeError("Runtime storage unavailable")

        with pytest.raises(Tau3RuntimeInfrastructureError):
            await Tau3ACPRelay(runtime, FakeTarget(["Please refund the order."])).run(
                "Resolve the request."
            )
    assert state.termination_reason is None
    assert state.submitted == ["Please refund the order."]


@pytest.mark.parametrize(
    "response", ["not-json", "[]", '{"termination_reason": null, "num_errors": 0}']
)
async def test_malformed_mcp_status_is_infrastructure(mcp_runtime, response):
    async with mcp_runtime() as (state, runtime, _):

        class Target:
            async def prompt(self, message: str) -> str:
                state.status_response = response
                return "Please refund the order."

        with pytest.raises(Tau3RuntimeInfrastructureError):
            await Tau3ACPRelay(runtime, Target()).run("Resolve the request.")
    assert state.submitted == []


async def test_mcp_status_deadline_is_infrastructure(mcp_runtime):
    async with mcp_runtime() as (state, _, client):
        runtime = MCPRuntimeClient(client.session, 0.05)

        class Target:
            async def prompt(self, message: str) -> str:
                state.status_gate = asyncio.Event()
                return "Please refund the order."

        with pytest.raises(Tau3RuntimeInfrastructureError) as failure:
            await Tau3ACPRelay(runtime, Target()).run("Resolve the request.")
        assert isinstance(failure.value.__cause__, TimeoutError)
    assert state.submitted == []


@pytest.mark.parametrize(
    (
        "agent_timeout_sec",
        "configured_tool_timeout",
        "configured_turn_timeout",
        "expected_tool_timeout",
        "expected_turn_flag",
        "expected_exec_timeout",
    ),
    [
        (1800.0, None, None, 1800.0, "1800.0", 1800.0),
        (1800.0, 60.0, None, 60.0, "1800.0", 1800.0),
        (None, None, None, 120.0, None, None),
        (None, None, 900.0, 120.0, "900.0", 90000.0),
    ],
)
async def test_harbor_adapter_inherits_turn_timeout_from_agent_phase(
    tmp_path: Path,
    agent_timeout_sec: float | None,
    configured_tool_timeout: float | None,
    configured_turn_timeout: float | None,
    expected_tool_timeout: float,
    expected_turn_flag: str | None,
    expected_exec_timeout: float | None,
) -> None:
    runtime = MCPServerConfig(
        name="tau3-runtime",
        transport="streamable-http",
        url="http://tau3-runtime:8000/mcp",
    )
    agent = Tau3HarborACPRelay(
        logs_dir=tmp_path,
        environment_logs_dir=Path("/logs/user-agent"),
        mcp_servers=[runtime],
        tool_timeout_sec=configured_tool_timeout,
        turn_timeout_sec=configured_turn_timeout,
    )

    class FakeEnvironment:
        def __init__(self) -> None:
            self.command = ""
            self.timeout_sec: float | None = None
            self.agent_timeout_sec: float | None = agent_timeout_sec

        async def upload_file(self, source: Path, target: Path) -> None:
            pass

        async def exec(
            self,
            command: str,
            *,
            env: dict[str, str],
            timeout_sec: float | None,
        ) -> SimpleNamespace:
            self.command = command
            self.timeout_sec = timeout_sec
            return SimpleNamespace(return_code=0, stderr="", stdout="")

    environment = FakeEnvironment()
    await agent.run(
        "<policy>Resolve the request.</policy>",
        environment,  # type: ignore[arg-type]
        AgentContext(),
    )

    relay_command = environment.command.removeprefix("set -o pipefail; ").split(
        " 2>&1 | tee ", maxsplit=1
    )[0]
    argv = shlex.split(relay_command)
    assert environment.timeout_sec == expected_exec_timeout
    assert argv[argv.index("--tool-timeout-sec") + 1] == str(expected_tool_timeout)
    if expected_turn_flag is None:
        assert "--turn-timeout-sec" not in argv
    else:
        assert argv[argv.index("--turn-timeout-sec") + 1] == expected_turn_flag


def test_extract_policy_uses_final_policy_block() -> None:
    instruction = "<policy>decoy</policy>\n<policy>real policy</policy>"
    assert extract_policy(instruction) == "real policy"


def test_extract_policy_excludes_instructions_with_inline_policy_reference() -> None:
    instruction = """<instructions>
Help a simulated user according to the <policy> provided below.
Call `send_message_to_user` to speak to the user, then call `end_conversation`.
</instructions>

<policy>
Verify before refunding.
</policy>
"""
    assert extract_policy(instruction) == "Verify before refunding."


@pytest.mark.parametrize(
    ("remote_error", "expected_error", "return_code"),
    [
        (
            "Tau3RuntimeInfrastructureError: provider unavailable",
            Tau3RuntimeInfrastructureError,
            1,
        ),
        (
            "TimeoutError: ACP target turn timed out after 1800.0 seconds",
            AgentTimeoutError,
            1,
        ),
        (
            "RuntimeError: acpx prompt failed with exit code 1: boom",
            NonZeroAgentExitCodeError,
            1,
        ),
        ("Killed", NonZeroAgentExitCodeError, 137),
        ("", NonZeroAgentExitCodeError, 1),
    ],
)
async def test_harbor_adapter_preserves_relay_failure_category(
    tmp_path: Path,
    remote_error: str,
    expected_error: type[Exception],
    return_code: int,
) -> None:
    runtime = MCPServerConfig(
        name="tau3-runtime",
        transport="streamable-http",
        url="http://tau3-runtime:8000/mcp",
    )
    agent = Tau3HarborACPRelay(
        logs_dir=tmp_path,
        environment_logs_dir=Path("/logs/user-agent"),
        mcp_servers=[runtime],
    )

    class FailedEnvironment:
        agent_timeout_sec = 1800.0

        async def upload_file(self, source: Path, target: Path) -> None:
            pass

        async def exec(
            self,
            command: str,
            *,
            env: dict[str, str],
            timeout_sec: float,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                return_code=return_code, stderr=remote_error, stdout=""
            )

    with pytest.raises(expected_error):
        await agent.run(
            "<policy>Resolve the request.</policy>",
            FailedEnvironment(),  # type: ignore[arg-type]
            AgentContext(),
        )
