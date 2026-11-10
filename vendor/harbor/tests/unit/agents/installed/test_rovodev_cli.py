"""RovoDev trajectory output, token accounting, and tool conversion."""

import json

from harbor.agents.installed.rovodev_cli import RovodevCli
from harbor.models.agent.context import AgentContext


def test_trajectory_json_written(temp_dir):
    agent = RovodevCli(logs_dir=temp_dir)

    session_file = temp_dir / "rovodev_session_context.json"
    session_data = {
        "id": "test-session-123",
        "message_history": [
            {
                "kind": "request",
                "timestamp": "2024-01-01T00:00:00Z",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": "Hello",
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_tokens": 2,
            "cache_write_tokens": 1,
        },
    }
    session_file.write_text(json.dumps(session_data))

    context = AgentContext()
    agent.populate_context_post_run(context)

    trajectory = json.loads((temp_dir / "trajectory.json").read_text())
    assert trajectory["session_id"] == "test-session-123"
    assert [(step["source"], step["message"]) for step in trajectory["steps"]] == [("user", "Hello")]
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 12
    assert trajectory["final_metrics"]["total_completion_tokens"] == 5
    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 2
    assert context.n_output_tokens == 5


def test_build_metrics_from_usage(temp_dir):
    agent = RovodevCli(logs_dir=temp_dir)

    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 25,
        "cache_write_tokens": 10,
    }

    metrics = agent._build_rovodev_metrics(usage)

    assert metrics.prompt_tokens == 125
    assert metrics.completion_tokens == 50
    assert metrics.cached_tokens == 25


def test_build_final_metrics(temp_dir):
    agent = RovodevCli(logs_dir=temp_dir)

    session_context = {
        "usage": {
            "input_tokens": 500,
            "output_tokens": 200,
            "cache_read_tokens": 100,
            "cache_write_tokens": 50,
        }
    }

    final_metrics = agent._build_final_metrics(session_context)

    assert final_metrics.total_prompt_tokens == 600
    assert final_metrics.total_completion_tokens == 200
    assert final_metrics.total_cached_tokens == 100
    assert final_metrics.extra.get("cache_write_tokens") == 50


def test_create_tool_call_with_valid_json(temp_dir):
    agent = RovodevCli(logs_dir=temp_dir)

    part = {
        "tool_call_id": "tc-1",
        "tool_name": "Shell",
        "args": '{"command": "ls -la"}',
    }

    tool_call = agent._create_tool_call(part)

    assert tool_call.tool_call_id == "tc-1"
    assert tool_call.function_name == "Shell"
    assert tool_call.arguments == {"command": "ls -la"}


def test_create_tool_call_with_invalid_json(temp_dir):
    agent = RovodevCli(logs_dir=temp_dir)

    part = {
        "tool_call_id": "tc-2",
        "tool_name": "ReadFile",
        "args": "not valid json {",
    }

    tool_call = agent._create_tool_call(part)

    assert tool_call.tool_call_id == "tc-2"
    assert tool_call.function_name == "ReadFile"
    assert "raw_args" in tool_call.arguments
    assert "parse_error" in tool_call.arguments


def test_create_observation_result_truncates_large_output(temp_dir):
    agent = RovodevCli(logs_dir=temp_dir)

    large_output = "x" * 15000

    tool_return = {
        "content": large_output,
        "timestamp": "2024-01-01T00:00:00Z",
    }

    obs = agent._create_observation_result("tc-1", tool_return)

    assert len(obs.content) < len(large_output)
    assert "truncated" in obs.content
