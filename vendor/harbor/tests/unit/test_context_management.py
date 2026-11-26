import pytest

from harbor.context_management import ContextWindowManager
from harbor.llms.base import (
    ContextBudgetExceededError,
    ContextLengthExceededError,
    ContextManagementInfrastructureError,
    LLMResponse,
)


class RejectUntilHistoryFitsLLM:
    def __init__(self, maximum_messages: int, rejection=ContextLengthExceededError):
        self.maximum_messages = maximum_messages
        self.rejection = rejection
        self.histories: list[list[dict]] = []

    async def call(self, prompt: str, message_history: list[dict], **kwargs):
        self.histories.append(list(message_history))
        if len(message_history) > self.maximum_messages:
            raise self.rejection
        return LLMResponse(content="recovered")


@pytest.mark.asyncio
async def test_context_manager_uses_provider_admission_without_mutating_input():
    llm = RejectUntilHistoryFitsLLM(maximum_messages=3)
    manager = ContextWindowManager(llm)
    history = [{"role": "user", "content": str(index)} for index in range(9)]

    result = await manager.call(
        prompt="summarize",
        message_history=history,
        max_tokens=4096,
        call_kwargs={},
    )

    assert result.response.content == "recovered"
    assert len(result.accepted_history) == 3
    assert history == [{"role": "user", "content": str(index)} for index in range(9)]


@pytest.mark.asyncio
async def test_context_manager_does_not_trim_for_invalid_output_budget():
    llm = RejectUntilHistoryFitsLLM(
        maximum_messages=3, rejection=ContextBudgetExceededError
    )
    manager = ContextWindowManager(llm)
    history = [{"role": "user", "content": str(index)} for index in range(9)]

    with pytest.raises(ContextBudgetExceededError):
        await manager.call(
            prompt="summarize",
            message_history=history,
            max_tokens=4096,
            call_kwargs={},
        )

    assert llm.histories == [history]


@pytest.mark.asyncio
async def test_context_manager_rejects_zero_output_before_provider_call():
    llm = RejectUntilHistoryFitsLLM(maximum_messages=3)
    manager = ContextWindowManager(llm)

    with pytest.raises(ContextManagementInfrastructureError):
        await manager.call(
            prompt="summarize",
            message_history=[],
            max_tokens=0,
            call_kwargs={},
        )

    assert llm.histories == []


@pytest.mark.asyncio
async def test_context_manager_types_exhausted_provider_rejection():
    llm = RejectUntilHistoryFitsLLM(maximum_messages=-1)
    manager = ContextWindowManager(llm)

    with pytest.raises(ContextManagementInfrastructureError) as exc_info:
        await manager.call(
            prompt="summarize",
            message_history=[{"role": "system", "content": "task"}],
            max_tokens=4096,
            call_kwargs={},
        )

    assert isinstance(exc_info.value.__cause__, ContextLengthExceededError)
