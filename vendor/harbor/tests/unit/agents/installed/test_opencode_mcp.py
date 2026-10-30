"""Unit tests for OpenCode MCP server integration."""

import json
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.opencode import OpenCode
from harbor.models.task.config import MCPServerConfig


class TestRegisterMcpServers:
    """Test _build_register_config_command() output."""

    def _parse_config(self, command: str) -> dict:
        """Extract the JSON config from the echo command."""
        start = command.index("'") + 1
        end = command.rindex("'")
        return json.loads(command[start:end])

    def test_no_mcp_no_model_returns_none(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name=None)
        assert agent._build_register_config_command() is None

    def test_sse_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="mcp-server", transport="sse", url="http://mcp-server:8000/sse"
            )
        ]
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="anthropic/claude-sonnet-4-5",
            mcp_servers=servers,
        )
        result = self._parse_config(agent._build_register_config_command())

        assert result["mcp"]["mcp-server"]["type"] == "remote"
        assert result["mcp"]["mcp-server"]["url"] == "http://mcp-server:8000/sse"
        assert result["mcp"]["mcp-server"]["oauth"] is False

    def test_streamable_http_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="http-server",
                transport="streamable-http",
                url="http://mcp-server:8000/mcp",
            )
        ]
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="anthropic/claude-sonnet-4-5",
            mcp_servers=servers,
        )
        result = self._parse_config(agent._build_register_config_command())

        assert result["mcp"]["http-server"]["type"] == "remote"
        assert result["mcp"]["http-server"]["url"] == "http://mcp-server:8000/mcp"
        assert result["mcp"]["http-server"]["oauth"] is False

    def test_stdio_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="stdio-server",
                transport="stdio",
                command="npx",
                args=["-y", "my-mcp"],
            )
        ]
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="anthropic/claude-sonnet-4-5",
            mcp_servers=servers,
        )
        result = self._parse_config(agent._build_register_config_command())

        assert result["mcp"]["stdio-server"]["type"] == "local"
        assert result["mcp"]["stdio-server"]["command"] == ["npx", "-y", "my-mcp"]
        assert "oauth" not in result["mcp"]["stdio-server"]

    def test_multiple_servers(self, temp_dir):
        servers = [
            MCPServerConfig(name="server-a", transport="sse", url="http://a:8000/sse"),
            MCPServerConfig(name="server-b", transport="stdio", command="server-b"),
        ]
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="anthropic/claude-sonnet-4-5",
            mcp_servers=servers,
        )
        result = self._parse_config(agent._build_register_config_command())

        assert "server-a" in result["mcp"]
        assert "server-b" in result["mcp"]


class TestProviderBaseUrl:
    """Test each provider receives only its configured base URL."""

    def _parse_config(self, command: str) -> dict:
        start = command.index("'") + 1
        end = command.rindex("'")
        return json.loads(command[start:end])

    def test_base_url_included_for_openai_provider(self, temp_dir, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-4o")
        result = self._parse_config(agent._build_register_config_command())
        assert (
            result["provider"]["openai"]["options"]["baseURL"]
            == "http://localhost:8080/v1"
        )

    def test_base_url_excluded_for_non_openai_provider(self, temp_dir, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
        agent = OpenCode(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5")
        result = self._parse_config(agent._build_register_config_command())
        assert "options" not in result["provider"]["anthropic"]

    @pytest.mark.parametrize(
        ("provider", "model", "variable", "endpoint"),
        [
            (
                "anthropic",
                "claude-sonnet-4-5",
                "ANTHROPIC_BASE_URL",
                "https://anthropic.example.test/v1",
            ),
            (
                "google",
                "gemini-2.5-pro",
                "GOOGLE_BASE_URL",
                "https://google.example.test/genai",
            ),
        ],
    )
    def test_provider_base_url_included(
        self, temp_dir, monkeypatch, provider, model, variable, endpoint
    ):
        monkeypatch.setenv(variable, endpoint)
        agent = OpenCode(logs_dir=temp_dir, model_name=f"{provider}/{model}")

        result = self._parse_config(agent._build_register_config_command())

        assert result["provider"][provider]["options"]["baseURL"] == endpoint

    @pytest.mark.parametrize(
        "provider", ["anthropic", "google", "openai", "hosted_vllm"]
    )
    def test_explicit_api_base_takes_precedence(self, temp_dir, monkeypatch, provider):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example.test")
        monkeypatch.setenv("GOOGLE_BASE_URL", "https://google.example.test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test")
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name=f"{provider}/model",
            api_base="https://explicit.example.test/v1",
        )

        result = self._parse_config(agent._build_register_config_command())

        assert result["provider"][provider]["options"]["baseURL"] == (
            "https://explicit.example.test/v1"
        )

    def test_no_base_url_when_env_unset(self, temp_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-4o")
        result = self._parse_config(agent._build_register_config_command())
        assert "options" not in result["provider"]["openai"]

    def test_vllm_provider_uses_openai_compatible_npm_and_options(
        self, temp_dir, monkeypatch
    ):
        # The `vllm` provider id must route through the openai-compatible npm
        # package (POST /v1/chat/completions, not the Responses API) with
        # baseURL + apiKey sourced from the OPENAI_* env.
        monkeypatch.setenv("OPENAI_BASE_URL", "https://ingress/proxy/otagent-j/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/1783107198924068")
        result = self._parse_config(agent._build_register_config_command())
        vllm = result["provider"]["vllm"]
        assert vllm["npm"] == "@ai-sdk/openai-compatible"
        assert vllm["options"]["baseURL"] == "https://ingress/proxy/otagent-j/v1"
        assert vllm["options"]["apiKey"] == "sk-test-123"
        assert "1783107198924068" in vllm["models"]


class TestCreateRunAgentCommandsMCP:
    """Test that run() handles MCP servers correctly."""

    @pytest.mark.asyncio
    async def test_no_mcp_servers_still_writes_config(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5")
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert len(exec_calls) == 2
        assert "opencode.json" in exec_calls[0].kwargs["command"]

    @pytest.mark.asyncio
    async def test_mcp_servers_adds_setup_command(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="mcp-server", transport="sse", url="http://mcp-server:8000/sse"
            )
        ]
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="anthropic/claude-sonnet-4-5",
            mcp_servers=servers,
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert len(exec_calls) == 2
        assert "opencode.json" in exec_calls[0].kwargs["command"]
        assert '"mcp"' in exec_calls[0].kwargs["command"]
