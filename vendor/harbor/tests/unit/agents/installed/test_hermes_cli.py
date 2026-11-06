"""Hermes trajectory conversion and context population."""

import json

from harbor.agents.installed.hermes import Hermes
from harbor.models.agent.context import AgentContext
from harbor.utils.artifact_writer import get_artifact_writer


class TestHermesAtifConversion:
    """Test ATIF trajectory conversion from hermes session data."""

    SAMPLE_SESSION = json.dumps(
        {
            "id": "session-1",
            "source": "cli",
            "messages": [
                {"role": "user", "content": "Complete the task."},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps({"command": "ls"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc-1",
                    "content": "file1.txt",
                },
                {
                    "role": "assistant",
                    "content": "Done.",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            ],
        }
    )

    def test_produces_valid_trajectory(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        trajectory = agent._convert_hermes_session_to_atif(
            self.SAMPLE_SESSION, "test-session"
        )
        assert trajectory is not None
        assert trajectory.schema_version == "ATIF-v1.2"
        assert trajectory.agent.name == "hermes"

    def test_step_sources(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        trajectory = agent._convert_hermes_session_to_atif(
            self.SAMPLE_SESSION, "test-session"
        )
        sources = [s.source for s in trajectory.steps]
        assert sources == ["user", "agent", "agent"]

    def test_tool_call_and_observation(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        trajectory = agent._convert_hermes_session_to_atif(
            self.SAMPLE_SESSION, "test-session"
        )
        tool_step = [s for s in trajectory.steps if s.tool_calls][0]
        assert tool_step.tool_calls[0].function_name == "terminal"
        assert tool_step.observation is not None
        assert tool_step.observation.results[0].source_call_id == "tc-1"

    def test_token_counts(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        trajectory = agent._convert_hermes_session_to_atif(
            self.SAMPLE_SESSION, "test-session"
        )
        assert trajectory.final_metrics.total_prompt_tokens == 100
        assert trajectory.final_metrics.total_completion_tokens == 50

    def test_empty_input_returns_none(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        assert agent._convert_hermes_session_to_atif("", "s") is None

    def test_sequential_step_ids(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        trajectory = agent._convert_hermes_session_to_atif(
            self.SAMPLE_SESSION, "test-session"
        )
        for i, step in enumerate(trajectory.steps):
            assert step.step_id == i + 1


class TestHermesPopulateContext:
    """Test populate_context_post_run."""

    def test_writes_trajectory_and_sets_tokens(self, temp_dir):
        session = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": "Hi!",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                ]
            }
        )
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        (temp_dir / "hermes-session.jsonl").write_text(session)

        context = AgentContext()
        agent.populate_context_post_run(context)
        get_artifact_writer().flush(timeout=30)

        assert (temp_dir / "trajectory.json").exists()
        data = json.loads((temp_dir / "trajectory.json").read_text())
        assert data["schema_version"] == "ATIF-v1.2"
        assert context.n_input_tokens == 10
        assert context.n_output_tokens == 5

    def test_no_session_file_no_trajectory(self, temp_dir):
        agent = Hermes(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-6")
        context = AgentContext()
        agent.populate_context_post_run(context)
        get_artifact_writer().flush(timeout=30)
        assert not (temp_dir / "trajectory.json").exists()
