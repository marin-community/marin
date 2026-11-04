"""OpenHands SDK artifact extraction and guest MCP configuration."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig, MCPTransport


def test_populate_context_trajectory_extracts_usage_and_cost(tmp_path: Path):
    agent = OpenHandsSDK(logs_dir=tmp_path, model_name="test/model")
    trajectory = {
        "schema_version": "ATIF-v1.5",
        "session_id": "test-session",
        "agent": {"name": "openhands-sdk", "version": "1.0.0"},
        "steps": [],
        "final_metrics": {
            "total_prompt_tokens": 1000,
            "total_completion_tokens": 500,
            "total_cached_tokens": 200,
            "total_cost_usd": 0.05,
        },
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.cost_usd == 0.05
    assert context.n_input_tokens == 1000
    assert context.n_output_tokens == 500
    assert context.n_cache_tokens == 200


def test_populate_context_missing_trajectory_preserves_context(tmp_path: Path):
    agent = OpenHandsSDK(logs_dir=tmp_path, model_name="test/model")
    context = AgentContext(cost_usd=0.25, n_input_tokens=7)

    agent.populate_context_post_run(context)

    assert context.cost_usd == 0.25
    assert context.n_input_tokens == 7


@pytest.mark.parametrize(
    ("transport", "expected"),
    [
        (
            "stdio",
            {
                "name": "test-server",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js", "--port=3000"],
            },
        ),
        (
            "sse",
            {
                "name": "test-server",
                "transport": "sse",
                "url": "http://localhost:8080/sse",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_mcp_configuration_uses_transport_wire_fields(
    tmp_path: Path, transport: MCPTransport, expected: dict[str, str | list[str]]
):
    agent = OpenHandsSDK(
        logs_dir=tmp_path,
        model_name="test/model",
        extra_env={"LLM_API_KEY": "EMPTY", "LLM_BASE_URL": "http://127.0.0.1:8788/v1"},
        mcp_servers=[
            MCPServerConfig(
                name="test-server",
                transport=transport,
                command="node",
                args=["server.js", "--port=3000"],
                url="http://localhost:8080/sse",
            )
        ],
    )
    environment = AsyncMock()
    environment.exec.return_value = ExecResult(return_code=0, stdout="", stderr="")

    await agent.run("Test instruction", environment, AgentContext())

    assert json.loads(environment.exec.call_args.kwargs["env"]["MCP_SERVERS_JSON"]) == [
        expected
    ]
