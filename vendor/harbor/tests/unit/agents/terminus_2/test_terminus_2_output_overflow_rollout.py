from types import SimpleNamespace

import pytest

from harbor.agents.base import MalformedModelResponseError
from harbor.agents.terminus_2.terminus_2 import (
    EpisodeLoggingPaths,
    LLMRequestContext,
    Terminus2,
)
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.metric import UsageInfo


class _ContextFailingLLM(BaseLLM):
    async def call(self, prompt, **kwargs) -> LLMResponse:
        raise ContextLengthExceededError("context full")

    def get_model_context_limit(self) -> int:
        return 32_768

    def get_model_output_limit(self) -> int | None:
        return 4_096


class _UnsalvageableParser:
    def salvage_truncated_response(self, truncated_response: str) -> None:
        return None

    def parse_response(self, response: str):
        return SimpleNamespace(warning="")


def _recovery_case(
    llm: BaseLLM,
    *,
    include_rollout_details: bool = True,
) -> tuple[Terminus2, Chat, OutputLengthExceededError, LLMRequestContext]:
    chat = Chat(llm)
    agent = object.__new__(Terminus2)
    agent.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    agent._parser = _UnsalvageableParser()
    agent._llm = llm
    agent._llm_call_kwargs = {}
    agent._enable_summarize = False
    agent._api_request_times = []
    response_kwargs = dict(
        content="unfinished response",
        usage=UsageInfo(
            prompt_tokens=2,
            completion_tokens=2,
            cache_tokens=0,
            cost_usd=0.0,
        ),
    )
    if include_rollout_details:
        response_kwargs.update(
            prompt_token_ids=[10, 11],
            completion_token_ids=[20, 21],
            logprobs=[-0.1, -0.2],
        )
    error = OutputLengthExceededError("output full", LLMResponse(**response_kwargs))
    request_context = LLMRequestContext(
        logging_paths=EpisodeLoggingPaths(debug=None, prompt=None, response=None),
        original_instruction="",
        session=None,
    )
    return agent, chat, error, request_context


@pytest.mark.asyncio
async def test_output_overflow_recovery_retains_truncated_rollout_evidence():
    agent, chat, error, request_context = _recovery_case(_ContextFailingLLM())

    with pytest.raises(ContextLengthExceededError):
        await agent._recover_output_overflow(
            chat=chat,
            prompt="initial prompt",
            error=error,
            request_context=request_context,
        )

    assert chat.messages[:2] == [
        {"role": "user", "content": "initial prompt"},
        {"role": "assistant", "content": "unfinished response"},
    ]
    assert chat.rollout_details == [
        {
            "prompt_token_ids": [[10, 11]],
            "completion_token_ids": [[20, 21]],
            "logprobs": [[-0.1, -0.2]],
        }
    ]


class _ContinuationRecordingLLM(BaseLLM):
    """Serves the exact token prompt it is given and records every request."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, prompt, **kwargs) -> LLMResponse:
        # Snapshot the history: Chat passes its live list and appends to it
        # after the call returns.
        history = kwargs.get("message_history")
        self.calls.append(
            {**kwargs, "prompt": prompt, "message_history": list(history or [])}
        )
        prompt_token_ids = kwargs.get("prompt_token_ids")
        return LLMResponse(
            content="recovered response",
            usage=UsageInfo(
                prompt_tokens=len(prompt_token_ids or []),
                completion_tokens=1,
                cache_tokens=0,
                cost_usd=0.0,
            ),
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=[30],
        )

    async def build_continuation_prompt_token_ids(
        self,
        prompt,
        previous_prompt_token_ids,
        previous_completion_token_ids,
        **kwargs,
    ) -> list[int] | None:
        return [*previous_prompt_token_ids, *previous_completion_token_ids, 40, 41]

    def get_model_context_limit(self) -> int:
        return 32_768

    def get_model_output_limit(self) -> int | None:
        return 4_096


@pytest.mark.asyncio
async def test_output_overflow_recovery_continues_the_exact_token_chain():
    """The correction turn continues from the served ids, not a text re-render.

    Resetting the token chain here made the follow-up request travel as text,
    so the server re-rendered the history through the chat template and re-cut
    the model's own sampled tokens (the TITO prefix mismatch).
    """
    llm = _ContinuationRecordingLLM()
    agent, chat, error, request_context = _recovery_case(llm)

    response = await agent._recover_output_overflow(
        chat=chat,
        prompt="initial prompt",
        error=error,
        request_context=request_context,
    )

    assert response.content == "recovered response"
    # The truncated turn's served prompt+completion ids stay the prefix.
    assert llm.calls[0]["prompt_token_ids"] == [10, 11, 20, 21, 40, 41]
    assert chat.rollout_details == [
        {
            "prompt_token_ids": [[10, 11], [10, 11, 20, 21, 40, 41]],
            "completion_token_ids": [[20, 21], [30]],
            "logprobs": [[-0.1, -0.2]],
        }
    ]


@pytest.mark.asyncio
async def test_output_overflow_recovery_without_tito_sends_message_history():
    """No served ids => no chain to keep; the correction travels as text."""
    llm = _ContinuationRecordingLLM()
    agent, chat, error, request_context = _recovery_case(
        llm,
        include_rollout_details=False,
    )

    await agent._recover_output_overflow(
        chat=chat,
        prompt="initial prompt",
        error=error,
        request_context=request_context,
    )

    assert llm.calls[0]["prompt_token_ids"] is None
    assert llm.calls[0]["message_history"] == [
        {"role": "user", "content": "initial prompt"},
        {"role": "assistant", "content": "unfinished response"},
    ]


def test_output_length_limiting_classifies_surrogate_output():
    """Terminal output with an unpaired surrogate ends the attempt as a
    model-output failure instead of crashing the agent loop unclassified."""
    agent = object.__new__(Terminus2)

    with pytest.raises(MalformedModelResponseError):
        agent._limit_output_length("analysis \ud842 tail")
