from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.tau3_acp_relay.harbor_agent import (
    Tau3HarborACPRelay,
    Tau3RuntimeInfrastructureError,
    extract_policy,
)
from harbor.agents.tau3_acp_relay.relay import RuntimeStatus, RuntimeTurn, Tau3ACPRelay
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.trial.errors import AgentTimeoutError

pytestmark = pytest.mark.unit


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
