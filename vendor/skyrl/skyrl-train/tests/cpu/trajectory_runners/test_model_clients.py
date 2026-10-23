from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl_train.trajectory_runners.model_clients import DirectModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["", "<|start_think|>Check facts.", "<|start_think|>Done.<|end_think|>"])
async def test_midturn_repair_serves_exact_prefix_without_wrong_claim_or_closing_marker(partial):
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.eos_token_id = -1
    engine.tokenizer.decode.return_value = "I am Marin Snowball"
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "I am Marin Snowball"},
                "finish_reason": "stop",
                "token_ids": [100, 101],
                "logprobs": {"content": [{"logprob": -0.2}, {"logprob": -0.3}]},
            }
        ]
    }
    served_prefix = list(map(ord, "USER: Who are you? ASSISTANT: " + partial))
    output = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": partial}]],
            "chat_completion_params": [{"chat_template_kwargs": {"enable_thinking": bool(partial)}}],
            "chat_continuations": [
                {"served_prefix_token_ids": served_prefix, "assistant_message_index": 1, "continue_final_message": True}
            ],
            "sampling_params": {"logprobs": 0},
        }
    )
    payload = engine.chat_completion.await_args.args[0]["json"]
    assert payload["_skyrl_exact_prompt_token_ids"] == served_prefix
    assert payload["messages"][-1]["content"] == partial
    assert output["prompt_ids"] == [served_prefix]
    assert output["response_logprobs"] == [[-0.2, -0.3]]
