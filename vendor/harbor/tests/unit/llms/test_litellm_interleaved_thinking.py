import pytest

from harbor.llms.lite_llm import LiteLLM


@pytest.mark.asyncio
async def test_litellm_sends_reasoning_content_in_messages():
    captured_kwargs = {}

    async def fake_chat(completion_kwargs):
        captured_kwargs.update(completion_kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "Second response",
                        "reasoning_content": "Second reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    llm = LiteLLM(model_name="test-model")
    # Transport-agnostic (message building happens before dispatch); patch both
    # the AsyncOpenAI and litellm-fallback seams so the route doesn't matter.
    llm._acreate_chat_raw = fake_chat  # type: ignore[assignment]
    llm._acreate_chat_litellm = fake_chat  # type: ignore[assignment]

    message_history = [
        {"role": "user", "content": "First message"},
        {
            "role": "assistant",
            "content": "First response",
            "reasoning_content": "First reasoning",
        },
    ]

    await llm.call(prompt="Second message", message_history=message_history)

    sent_messages = captured_kwargs["messages"]
    assert len(sent_messages) == 3
    assert sent_messages[0] == {"role": "user", "content": "First message"}
    assert sent_messages[1] == {
        "role": "assistant",
        "content": "First response",
        "reasoning_content": "First reasoning",
    }
    assert sent_messages[2] == {"role": "user", "content": "Second message"}


@pytest.mark.asyncio
async def test_hosted_vllm_replays_reasoning_in_the_field_vllm_reads():
    captured_kwargs = {}

    async def fake_chat(completion_kwargs):
        captured_kwargs.update(completion_kwargs)
        return {
            "choices": [
                {"message": {"content": "Second response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    llm = LiteLLM(
        model_name="hosted_vllm/test-model",
        api_base="http://localhost/v1",
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    llm._acreate_chat_raw = fake_chat  # type: ignore[assignment]

    message_history = [
        {"role": "user", "content": "First message"},
        {
            "role": "assistant",
            "content": "First response",
            "reasoning_content": "First reasoning",
        },
    ]

    await llm.call(prompt="Second message", message_history=message_history)

    assert captured_kwargs["messages"][1] == {
        "role": "assistant",
        "content": "First response",
        "reasoning_content": "First reasoning",
        "reasoning": "First reasoning",
    }
    assert "reasoning" not in message_history[1]
