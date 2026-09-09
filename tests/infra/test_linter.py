# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from infra.linter import AgentSpec, _agent_output, _agent_spec, _with_usage_output


def test_agent_spec_requires_explicit_model_and_effort():
    with pytest.raises(ValueError, match="explicit model and effort"):
        _agent_spec(["codex", "exec"])

    assert _agent_spec(
        ["codex", "exec", "--model", "gpt-5.6-terra", "--config", "model_reasoning_effort=low"]
    ) == AgentSpec("codex", "gpt-5.6-terra", "low")
    assert _agent_spec(
        ["claude", "-p", "--model", "claude-haiku-4-5-20251001", "--effort", "low"]
    ) == AgentSpec("claude", "claude-haiku-4-5-20251001", "low")


def test_claude_output_preserves_findings_and_usage():
    stdout = json.dumps(
        {
            "type": "result",
            "result": "src/example.py:4: ml-example (0.90) finding",
            "total_cost_usd": 0.012,
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "output_tokens": 4,
            },
        }
    )

    findings, usage = _agent_output(AgentSpec("claude", "claude-haiku-4-5-20251001", "low"), stdout)

    assert findings == "src/example.py:4: ml-example (0.90) finding"
    assert usage.total_tokens == 19
    assert usage.cost_usd == 0.012


def test_codex_output_preserves_findings_and_usage():
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "src/example.py:4: ml-example (0.90) finding",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 6},
                }
            ),
        ]
    )

    findings, usage = _agent_output(AgentSpec("codex", "gpt-5.6-terra", "low"), stdout)

    assert findings == "src/example.py:4: ml-example (0.90) finding"
    assert usage.total_tokens == 26
    assert usage.cost_usd is None


def test_supported_agent_commands_enable_usage_output():
    claude = _with_usage_output(
        ["claude", "-p", "--model", "claude-haiku-4-5-20251001", "--effort", "low"],
        AgentSpec("claude", "claude-haiku-4-5-20251001", "low"),
    )
    codex = _with_usage_output(
        ["codex", "exec", "--model", "gpt-5.6-terra", "--config", "model_reasoning_effort=low"],
        AgentSpec("codex", "gpt-5.6-terra", "low"),
    )

    assert claude[-2:] == ["--output-format", "json"]
    assert codex[-1] == "--json"
