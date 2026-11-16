import pytest

from harbor.llms.base import BaseLLM, LLMResponse
from harbor.llms.chat import Chat
from harbor.models.metric import UsageInfo


class FakeLLM(BaseLLM):
    """Minimal BaseLLM stub for testing Chat."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        continuation_prompt_token_ids: list[int] | None = None,
    ):
        self._responses = responses or []
        self._call_index = 0
        self.call_kwargs_history: list[dict] = []
        self.continuation_prompt_token_ids = continuation_prompt_token_ids

    async def call(self, prompt, **kwargs) -> LLMResponse:
        captured_kwargs = dict(kwargs)
        if isinstance(captured_kwargs.get("message_history"), list):
            captured_kwargs["message_history"] = list(
                captured_kwargs["message_history"]
            )
        self.call_kwargs_history.append({"prompt": prompt, **captured_kwargs})
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return LLMResponse(content="default response")

    async def build_continuation_prompt_token_ids(
        self,
        prompt,
        previous_prompt_token_ids,
        previous_completion_token_ids,
        **kwargs,
    ) -> list[int] | None:
        return self.continuation_prompt_token_ids

    def get_model_context_limit(self) -> int:
        return 100000

    def get_model_output_limit(self) -> int | None:
        return 4096


@pytest.mark.asyncio
async def test_chat_tracks_response_id():
    """Verify _last_response_id is set from LLMResponse.response_id."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="first",
                usage=UsageInfo(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_tokens=0,
                    cost_usd=0.0,
                ),
                response_id="resp_001",
            ),
        ]
    )
    chat = Chat(model=fake_llm)

    assert chat._last_response_id is None
    await chat.chat("hello")
    assert chat._last_response_id == "resp_001"


@pytest.mark.asyncio
async def test_chat_passes_previous_response_id():
    """Verify previous_response_id is passed as kwarg to model.call()."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="first",
                usage=UsageInfo(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_tokens=0,
                    cost_usd=0.0,
                ),
                response_id="resp_001",
            ),
            LLMResponse(
                content="second",
                usage=UsageInfo(
                    prompt_tokens=20,
                    completion_tokens=10,
                    cache_tokens=0,
                    cost_usd=0.0,
                ),
                response_id="resp_002",
            ),
        ]
    )
    chat = Chat(model=fake_llm)

    await chat.chat("first message")
    # First call should have previous_response_id=None
    assert fake_llm.call_kwargs_history[0]["previous_response_id"] is None

    await chat.chat("second message")
    # Second call should have previous_response_id="resp_001"
    assert fake_llm.call_kwargs_history[1]["previous_response_id"] == "resp_001"
    assert chat._last_response_id == "resp_002"


@pytest.mark.asyncio
async def test_chat_interleaved_thinking_preserves_empty_reasoning_field():
    fake_llm = FakeLLM(responses=[LLMResponse(content="answer")])
    chat = Chat(model=fake_llm, interleaved_thinking=True)

    await chat.chat("question")

    assert chat.messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning_content": ""},
    ]


@pytest.mark.asyncio
async def test_chat_reset_response_chain():
    """Verify reset_response_chain() clears _last_response_id."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="first",
                usage=UsageInfo(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_tokens=0,
                    cost_usd=0.0,
                ),
                response_id="resp_001",
            ),
            LLMResponse(
                content="after reset",
                usage=UsageInfo(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_tokens=0,
                    cost_usd=0.0,
                ),
                response_id="resp_002",
            ),
        ]
    )
    chat = Chat(model=fake_llm)

    await chat.chat("hello")
    assert chat._last_response_id == "resp_001"

    chat.reset_response_chain()
    assert chat._last_response_id is None

    await chat.chat("after reset")
    # After reset, previous_response_id should be None
    assert fake_llm.call_kwargs_history[1]["previous_response_id"] is None


@pytest.mark.asyncio
async def test_chat_no_response_id_when_none():
    """Verify _last_response_id stays None when response has no response_id."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="no id",
                usage=UsageInfo(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_tokens=0,
                    cost_usd=0.0,
                ),
                # response_id defaults to None
            ),
        ]
    )
    chat = Chat(model=fake_llm)

    await chat.chat("hello")
    assert chat._last_response_id is None


@pytest.mark.asyncio
async def test_chat_preserves_generated_ids_in_continuation_prompt():
    exact_continuation = [10, 11, 10122, 90, 20, 21]
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="(/{",
                prompt_token_ids=[10, 11],
                completion_token_ids=[10122, 90],
            ),
            LLMResponse(
                content="done",
                prompt_token_ids=exact_continuation,
                completion_token_ids=[30],
            ),
        ],
        continuation_prompt_token_ids=exact_continuation,
    )
    chat = Chat(model=fake_llm)

    await chat.chat("first")
    await chat.chat("second")

    assert fake_llm.call_kwargs_history[1]["prompt_token_ids"] == exact_continuation
    assert chat.rollout_details[0]["prompt_token_ids"] == [
        [10, 11],
        exact_continuation,
    ]


@pytest.mark.asyncio
async def test_chat_native_tool_result_forms_valid_next_turn_history():
    tool_call = {
        "id": "call-7",
        "type": "function",
        "function": {"name": "execute_commands", "arguments": '{"commands": []}'},
    }
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(content="", response_id="response-1", tool_calls=[tool_call]),
            LLMResponse(content="done"),
        ]
    )
    chat = Chat(model=fake_llm)

    await chat.chat("inspect")
    chat.append_tool_results([tool_call], "terminal output")
    await chat.chat("continue")

    assert fake_llm.call_kwargs_history[1]["message_history"] == [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call-7", "content": "terminal output"},
    ]
    assert fake_llm.call_kwargs_history[1]["previous_response_id"] is None


@pytest.mark.asyncio
async def test_chat_rejects_provider_retokenization_of_exact_prompt():
    exact_continuation = [10, 11, 10122, 90, 20, 21]
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="(/{",
                prompt_token_ids=[10, 11],
                completion_token_ids=[10122, 90],
            ),
            LLMResponse(
                content="wrong context",
                prompt_token_ids=[10, 11, 7, 9573, 20, 21],
                completion_token_ids=[30],
            ),
        ],
        continuation_prompt_token_ids=exact_continuation,
    )
    chat = Chat(model=fake_llm)
    await chat.chat("first")

    with pytest.raises(ValueError, match="did not serve the exact continuation"):
        await chat.chat("second")


def _usage() -> UsageInfo:
    return UsageInfo(
        prompt_tokens=10, completion_tokens=5, cache_tokens=0, cost_usd=0.0
    )


@pytest.mark.asyncio
async def test_chat_accumulates_extra_in_rollout_details():
    """Verify extra provider-specific fields are accumulated and pivoted by field name."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="turn1",
                usage=_usage(),
                completion_token_ids=[1, 2, 3],
                extra={"routed_experts": [[0, 1], [2, 3]], "router_logits": [0.5, 0.8]},
            ),
            LLMResponse(
                content="turn2",
                usage=_usage(),
                completion_token_ids=[4, 5, 6],
                extra={"routed_experts": [[4, 5], [6, 7]], "router_logits": [0.1, 0.9]},
            ),
        ]
    )
    chat = Chat(model=fake_llm)
    await chat.chat("msg1")
    await chat.chat("msg2")

    details = chat.rollout_details
    assert len(details) == 1
    assert "extra" in details[0]
    extra = details[0]["extra"]
    assert extra["routed_experts"] == [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    assert extra["router_logits"] == [[0.5, 0.8], [0.1, 0.9]]


@pytest.mark.asyncio
async def test_chat_rollout_details_no_extra_when_absent():
    """Verify 'extra' key is absent from RolloutDetail when no response has extra."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="turn1",
                usage=_usage(),
                completion_token_ids=[1, 2, 3],
            ),
        ]
    )
    chat = Chat(model=fake_llm)
    await chat.chat("msg1")

    details = chat.rollout_details
    assert len(details) == 1
    assert "extra" not in details[0]
    assert "completion_token_ids" in details[0]


@pytest.mark.asyncio
async def test_chat_logprobs_parallel_to_token_ids():
    """Matching-length logprobs are stored 1:1 with completion_token_ids (TIS exact path)."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="turn1",
                usage=_usage(),
                completion_token_ids=[1, 2, 3],
                logprobs=[-0.1, -0.2, -0.3],
            ),
        ]
    )
    chat = Chat(model=fake_llm)
    await chat.chat("msg1")

    details = chat.rollout_details
    assert details[0]["completion_token_ids"] == [[1, 2, 3]]
    assert details[0]["logprobs"] == [[-0.1, -0.2, -0.3]]


@pytest.mark.asyncio
async def test_chat_logprobs_length_mismatch_recorded_as_empty():
    """A logprob/token-id length mismatch must NOT mis-pair: record empty list,
    keeping per-turn index alignment so downstream TIS flags it (fallback)."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="turn1",
                usage=_usage(),
                completion_token_ids=[1, 2, 3],
                logprobs=[-0.1, -0.2],  # too few -> mismatch
            ),
        ]
    )
    chat = Chat(model=fake_llm)
    await chat.chat("msg1")

    details = chat.rollout_details
    # token ids preserved; logprobs for this turn replaced with [] (not mis-zipped),
    # and the lists stay index-aligned (one entry each).
    assert details[0]["completion_token_ids"] == [[1, 2, 3]]
    assert details[0]["logprobs"] == [[]]


@pytest.mark.asyncio
async def test_chat_rollout_details_mixed_extra():
    """Verify None fills for turns missing a given extra key."""
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="turn1",
                usage=_usage(),
                completion_token_ids=[1, 2],
                extra={"routed_experts": [[0, 1]], "field_a": "val1"},
            ),
            LLMResponse(
                content="turn2",
                usage=_usage(),
                completion_token_ids=[3, 4],
                extra={"routed_experts": [[2, 3]]},
            ),
        ]
    )
    chat = Chat(model=fake_llm)
    await chat.chat("msg1")
    await chat.chat("msg2")

    details = chat.rollout_details
    extra = details[0]["extra"]
    # field_a only present in turn 1
    assert extra["field_a"] == ["val1", None]
    # routed_experts present in both turns
    assert extra["routed_experts"] == [[[0, 1]], [[2, 3]]]


def _truncated_response() -> LLMResponse:
    return LLMResponse(
        content="unfinished response",
        prompt_token_ids=[10, 11],
        completion_token_ids=[20, 21],
        response_id="resp_001",
    )


@pytest.mark.asyncio
async def test_chat_provider_chain_reset_keeps_exact_token_continuation():
    exact_continuation = [10, 11, 20, 21, 90, 91]
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                content="recovered",
                prompt_token_ids=exact_continuation,
                completion_token_ids=[30],
            )
        ],
        continuation_prompt_token_ids=exact_continuation,
    )
    chat = Chat(model=fake_llm)

    chat.append_response("first", _truncated_response())
    chat.reset_provider_response_chain()
    await chat.chat("ERROR!! ... re-issue")

    assert fake_llm.call_kwargs_history[0]["previous_response_id"] is None
    assert fake_llm.call_kwargs_history[0]["prompt_token_ids"] == exact_continuation


@pytest.mark.asyncio
async def test_chat_full_reset_still_drops_exact_token_continuation():
    fake_llm = FakeLLM(continuation_prompt_token_ids=[10, 11, 20, 21, 90, 91])
    chat = Chat(model=fake_llm)

    chat.append_response("first", _truncated_response())
    chat.reset_response_chain()

    assert await chat.prepare_prompt_token_ids("next") is None
