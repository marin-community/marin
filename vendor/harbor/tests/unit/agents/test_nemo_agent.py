"""NeMo version parsing and post-run artifact extraction."""

import json
from pathlib import Path

import pytest

from harbor.agents.installed.nemo_agent import NemoAgent
from harbor.models.agent.context import AgentContext


@pytest.fixture
def agent(tmp_path: Path) -> NemoAgent:
    return NemoAgent(logs_dir=tmp_path, model_name="nvidia/meta/llama-3.3-70b-instruct")


@pytest.mark.parametrize(
    ("stdout", "version"),
    [
        ("nat, version 1.6.0", "1.6.0"),
        ("nat, version 1.6.0.dev45+ga8462a16", "1.6.0.dev45+ga8462a16"),
        ("1.6.0", "1.6.0"),
    ],
)
def test_parse_version_cli_output(agent: NemoAgent, stdout: str, version: str):
    assert agent.parse_version(stdout) == version


@pytest.mark.parametrize(
    ("output", "answer"),
    [
        (None, None),
        ("", None),
        ("The answer is 42\n", "The answer is 42"),
        ("Loading...\nProcessing...\nThe final answer is Paris\n", "The final answer is Paris"),
    ],
)
def test_populate_context_output_uses_final_line(agent: NemoAgent, output: str | None, answer: str | None):
    if output is not None:
        (agent.logs_dir / "nemo-agent-output.txt").write_text(output)

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata == ({"nemo_agent_final_output": answer} if answer is not None else None)


def test_populate_context_trajectory_extracts_usage_and_cost(agent: NemoAgent):
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": "test-session",
        "agent": {"name": "nemo-agent", "version": "1.6.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Hello"},
            {"step_id": 2, "source": "agent", "message": "World"},
        ],
        "final_metrics": {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 10,
            "total_cached_tokens": 5,
            "total_cost_usd": 0.0035,
            "total_steps": 2,
        },
    }
    (agent.logs_dir / "trajectory.json").write_text(json.dumps(trajectory))
    (agent.logs_dir / "nemo-agent-output.txt").write_text("World\n")

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 100
    assert context.n_output_tokens == 10
    assert context.n_cache_tokens == 5
    assert context.cost_usd == 0.0035
    assert context.metadata == {"nemo_agent_final_output": "World"}


@pytest.mark.parametrize("trajectory", ["not valid json {{{", '{"steps": []}'])
def test_populate_context_unusable_trajectory_preserves_text(agent: NemoAgent, trajectory: str):
    (agent.logs_dir / "trajectory.json").write_text(trajectory)
    (agent.logs_dir / "nemo-agent-output.txt").write_text("42\n")

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata == {"nemo_agent_final_output": "42"}
    assert context.n_input_tokens is None
    assert context.n_output_tokens is None
    assert context.cost_usd is None
