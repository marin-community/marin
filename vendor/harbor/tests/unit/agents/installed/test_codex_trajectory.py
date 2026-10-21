import json
import uuid

import pytest
from upath import UPath

from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext


@pytest.mark.parametrize("protocol", ["file", "memory"])
def test_codex_session_persists_trajectory_and_usage(tmp_path, protocol):
    logs = (
        tmp_path
        if protocol == "file"
        else UPath(f"memory://codex-{uuid.uuid4().hex}/agent")
    )
    session = logs / "sessions/2026/10/04"
    session.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "session_meta", "payload": {"id": "session-1", "cli_version": "test"}},
        {
            "type": "response_item",
            "timestamp": "2026-10-04T08:00:00Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"text": "Wrote result.json"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_cost": 0,
                    "total_token_usage": {
                        "input_tokens": 123,
                        "cached_input_tokens": 23,
                        "output_tokens": 7,
                    },
                },
            },
        },
    ]
    (session / "rollout.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    agent = Codex(logs_dir=logs, model_name="openai/test")
    context = AgentContext()
    agent.populate_context_post_run(context)
    trajectory = json.loads((logs / "trajectory.json").read_text())
    assert trajectory["session_id"] == "session-1"
    assert trajectory["steps"][0]["message"] == "Wrote result.json"
    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
        context.cost_usd,
    ) == (123, 23, 7, 0)
