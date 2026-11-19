import json

import pytest
from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext


@pytest.mark.parametrize("agent_type", [Codex])
def test_native_agent_captured_tokens_survive_missing_cli_session(tmp_path, agent_type):
    agent = agent_type(
        logs_dir=tmp_path,
        model_name="hosted_vllm/bfcl-test",
        api_base="http://literal/v1",
        model_info={"max_input_tokens": 32768, "max_output_tokens": 8192},
        collect_rollout_details=True,
    )
    (tmp_path / "literal.jsonl").write_text(
        json.dumps(
            {
                "literal": {
                    "prompt_token_ids": [7, 8],
                    "completion_token_ids": [9],
                    "logprobs": [-0.25],
                }
            }
        )
        + "\n"
    )
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.rollout_details == [
        {
            "prompt_token_ids": [[7, 8]],
            "completion_token_ids": [[9]],
            "logprobs": [[-0.25]],
        }
    ]
    assert context.metadata is not None
    assert context.metadata["rollout_correlation_id"]
