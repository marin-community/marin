"""Unit tests for Gemini CLI MCP server integration."""

import pytest

from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.models.task.config import MCPServerConfig


class TestRegisterMcpServers:
    """Test MCP server settings config output."""

    def test_no_mcp_servers_returns_none(self, temp_dir):
        agent = GeminiCli(logs_dir=temp_dir, model_name="google/gemini-2.5-pro")
        assert agent._build_settings_config() == (None, None)

    def test_sse_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="mcp-server", transport="sse", url="http://mcp-server:8000/sse"
            )
        ]
        agent = GeminiCli(
            logs_dir=temp_dir,
            model_name="google/gemini-2.5-pro",
            mcp_servers=servers,
        )
        result, model_alias = agent._build_settings_config()

        assert result is not None
        assert model_alias is None
        assert "mcpServers" in result
        assert "mcp-server" in result["mcpServers"]
        assert result["mcpServers"]["mcp-server"]["url"] == "http://mcp-server:8000/sse"

    def test_streamable_http_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="http-server",
                transport="streamable-http",
                url="http://mcp-server:8000/mcp",
            )
        ]
        agent = GeminiCli(
            logs_dir=temp_dir,
            model_name="google/gemini-2.5-pro",
            mcp_servers=servers,
        )
        result, model_alias = agent._build_settings_config()

        assert result is not None
        assert model_alias is None
        assert (
            result["mcpServers"]["http-server"]["httpUrl"]
            == "http://mcp-server:8000/mcp"
        )

    def test_stdio_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="stdio-server",
                transport="stdio",
                command="npx",
                args=["-y", "my-mcp"],
            )
        ]
        agent = GeminiCli(
            logs_dir=temp_dir,
            model_name="google/gemini-2.5-pro",
            mcp_servers=servers,
        )
        result, model_alias = agent._build_settings_config()

        assert result is not None
        assert model_alias is None
        assert result["mcpServers"]["stdio-server"]["command"] == "npx"
        assert result["mcpServers"]["stdio-server"]["args"] == ["-y", "my-mcp"]

    def test_multiple_servers(self, temp_dir):
        servers = [
            MCPServerConfig(name="server-a", transport="sse", url="http://a:8000/sse"),
            MCPServerConfig(name="server-b", transport="stdio", command="server-b"),
        ]
        agent = GeminiCli(
            logs_dir=temp_dir,
            model_name="google/gemini-2.5-pro",
            mcp_servers=servers,
        )
        result, model_alias = agent._build_settings_config()

        assert result is not None
        assert model_alias is None
        assert "server-a" in result["mcpServers"]
        assert "server-b" in result["mcpServers"]


class TestReasoningEffort:
    """Test Gemini CLI reasoning effort settings."""

    def test_gemini_3_reasoning_effort_uses_thinking_level(self, temp_dir):
        agent = GeminiCli(
            logs_dir=temp_dir,
            model_name="google/gemini-3-pro-preview",
            reasoning_effort="low",
        )

        result, model_alias = agent._build_settings_config("gemini-3-pro-preview")
        assert result is not None
        assert model_alias is not None

        assert model_alias == "harbor-gemini-3-pro-preview-low"
        alias = result["modelConfigs"]["customAliases"][model_alias]
        thinking_config = alias["modelConfig"]["generateContentConfig"][
            "thinkingConfig"
        ]
        assert alias["modelConfig"]["model"] == "gemini-3-pro-preview"
        assert thinking_config == {"includeThoughts": True, "thinkingLevel": "LOW"}

    def test_gemini_3_flash_accepts_flash_only_reasoning_effort(self, temp_dir):
        agent = GeminiCli(
            logs_dir=temp_dir,
            model_name="google/gemini-3-flash-preview",
            reasoning_effort="medium",
        )

        result, model_alias = agent._build_settings_config("gemini-3-flash-preview")
        assert result is not None
        assert model_alias is not None
        alias = result["modelConfigs"]["customAliases"][model_alias]
        thinking_config = alias["modelConfig"]["generateContentConfig"][
            "thinkingConfig"
        ]
        assert thinking_config == {"includeThoughts": True, "thinkingLevel": "MEDIUM"}

    def test_gemini_3_pro_rejects_flash_only_reasoning_effort(self, temp_dir):
        with pytest.raises(ValueError, match="choose a Gemini 3 Flash model"):
            GeminiCli(
                logs_dir=temp_dir,
                model_name="google/gemini-3-pro-preview",
                reasoning_effort="medium",
            )

    def test_gemini_25_reasoning_effort_raises(self, temp_dir):
        with pytest.raises(ValueError, match="do not support reasoning_effort"):
            GeminiCli(
                logs_dir=temp_dir,
                model_name="google/gemini-2.5-flash",
                reasoning_effort="medium",
            )

    def test_invalid_reasoning_effort_raises(self, temp_dir):
        with pytest.raises(ValueError, match="Valid values"):
            GeminiCli(logs_dir=temp_dir, reasoning_effort="extreme")
