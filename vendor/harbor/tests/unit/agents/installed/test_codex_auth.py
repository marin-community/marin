"""Installed agents keep host authentication out of guest commands and files."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.aider import Aider
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.copilot_cli import CopilotCli
from harbor.models.agent.context import AgentContext


@pytest.mark.asyncio
async def test_codex_keeps_host_authentication_out_of_guest_state(
    tmp_path, monkeypatch
):
    auth_file = tmp_path / ".codex" / "auth.json"
    auth_file.parent.mkdir()
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": "host-oauth-private-canary"}})
    )
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth_file))
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "host-provider-private-canary")
    monkeypatch.setenv("GITHUB_TOKEN", "host-github-private-canary")
    agent = Codex(logs_dir=tmp_path, model_name="openai/model")
    environment = AsyncMock()
    environment.agent_timeout_sec = None
    environment.default_user = "agent"
    environment.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="")

    await agent.run("Solve the task", environment, AgentContext())

    environment.upload_file.assert_not_awaited()
    calls = [call.kwargs for call in environment.exec.call_args_list]
    serialized = json.dumps(calls)
    assert "host-oauth-private-canary" not in serialized
    assert "host-provider-private-canary" not in serialized
    assert "host-github-private-canary" not in serialized
    assert any(call["env"].get("OPENAI_API_KEY") == "EMPTY" for call in calls)
    assert (
        json.loads(auth_file.read_text())["tokens"]["access_token"]
        == "host-oauth-private-canary"
    )


@pytest.mark.parametrize("control", ["CODEX_AUTH_JSON_PATH", "CODEX_FORCE_AUTH_JSON"])
def test_codex_auth_file_controls_cannot_be_granted_to_agents(tmp_path, control):
    with pytest.raises(ValueError, match="forbidden"):
        Codex(logs_dir=tmp_path, extra_env={control: "/host/auth.json"})


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [Aider, CopilotCli])
async def test_unsupported_agent_auth_cannot_reach_guest_even_when_preinstalled(
    tmp_path, monkeypatch, agent_type
):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        monkeypatch.setenv(key, "host-private-canary")
    agent = agent_type(logs_dir=tmp_path, model_name="openai/model")
    environment = AsyncMock()

    with pytest.raises(RuntimeError):
        await agent.install(environment)
    # Prepared images can skip installation, so execution needs the same boundary.
    with pytest.raises(RuntimeError):
        await agent.run("Solve the task", environment, AgentContext())

    environment.exec.assert_not_awaited()
    environment.upload_file.assert_not_awaited()
