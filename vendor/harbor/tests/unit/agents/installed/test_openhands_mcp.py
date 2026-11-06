"""Unit tests for OpenHands MCP server integration."""

from harbor.agents.installed.openhands import OpenHands
from harbor.models.task.config import MCPServerConfig


class TestBuildMCPConfigToml:
    """Test _build_mcp_config_toml() output."""

    def test_no_mcp_servers_returns_none(self, temp_dir):
        agent = OpenHands(logs_dir=temp_dir)
        assert agent._build_mcp_config_toml() is None

    def test_sse_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="mcp-server", transport="sse", url="http://mcp-server:8000/sse"
            )
        ]
        agent = OpenHands(logs_dir=temp_dir, mcp_servers=servers)
        result = agent._build_mcp_config_toml()

        assert "[mcp]" in result
        assert 'sse_servers = [{url = "http://mcp-server:8000/sse"}]' in result
        assert "shttp_servers" not in result
        assert "stdio_servers" not in result

    def test_streamable_http_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="mcp-server",
                transport="streamable-http",
                url="http://mcp-server:8000/mcp",
            )
        ]
        agent = OpenHands(logs_dir=temp_dir, mcp_servers=servers)
        result = agent._build_mcp_config_toml()

        assert "[mcp]" in result
        assert 'shttp_servers = [{url = "http://mcp-server:8000/mcp"}]' in result
        assert "sse_servers" not in result
        assert "stdio_servers" not in result

    def test_stdio_server(self, temp_dir):
        servers = [
            MCPServerConfig(
                name="stdio-server",
                transport="stdio",
                command="npx",
                args=["-y", "my-mcp"],
            )
        ]
        agent = OpenHands(logs_dir=temp_dir, mcp_servers=servers)
        result = agent._build_mcp_config_toml()

        assert "[mcp]" in result
        assert (
            'stdio_servers = [{name = "stdio-server", command = "npx", args = ["-y", "my-mcp"]}]'
            in result
        )
        assert "sse_servers" not in result
        assert "shttp_servers" not in result

    def test_multiple_servers(self, temp_dir):
        servers = [
            MCPServerConfig(name="server-a", transport="sse", url="http://a:8000/sse"),
            MCPServerConfig(
                name="server-b",
                transport="streamable-http",
                url="http://b:8000/mcp",
            ),
            MCPServerConfig(name="server-c", transport="stdio", command="server-c"),
        ]
        agent = OpenHands(logs_dir=temp_dir, mcp_servers=servers)
        result = agent._build_mcp_config_toml()

        assert "sse_servers" in result
        assert "shttp_servers" in result
        assert "stdio_servers" in result
