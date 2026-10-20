from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from harbor.models.metric import UsageInfo


class LLMBackend(str, Enum):
    """Enum for available LLM backends."""

    LITELLM = "litellm"
    TINKER = "tinker"


@dataclass
class LLMResponse:
    """Response from an LLM call containing the generated content and metadata.

    Attributes:
        content: The generated text response
        reasoning_content: The LLM's explicit internal reasoning
        usage: Token usage and cost information
        tool_calls: Native function calls returned by a chat-completion provider
        prompt_token_ids: Full prompt token IDs including conversation history (if collect_rollout_details=True)
        completion_token_ids: Token IDs for the generated completion (if collect_rollout_details=True)
        logprobs: Log probabilities for each completion token (if collect_rollout_details=True)
    """

    content: str
    reasoning_content: str | None = None
    model_name: str | None = None
    usage: UsageInfo | None = None
    response_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class ContextLengthExceededError(Exception):
    """Raised when the LLM response indicates the context length was exceeded."""

    pass


class OutputLengthExceededError(Exception):
    """Raised when a response is truncated by the output-token limit.

    ``partial_response`` preserves the returned usage and rollout evidence for
    callers that retain the truncated generation in a conversation.
    """

    def __init__(
        self,
        message: str,
        partial_response: LLMResponse,
    ) -> None:
        super().__init__(message)
        self.partial_response = partial_response
        self.truncated_response = partial_response.content


class ContextBudgetExceededError(Exception):
    """Raised when a request has no positive output-token budget."""


class LLMRequestTimeoutError(Exception):
    """A transport deadline expired before an LLM response completed."""

    def __init__(
        self,
        *,
        request_id: str,
        session_id: str | None,
        request_sequence: int,
        model_name: str,
        timeout_phase: str,
        connect_timeout_sec: float | None,
        read_timeout_sec: float | None,
        write_timeout_sec: float | None,
        pool_timeout_sec: float | None,
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.request_sequence = request_sequence
        self.model_name = model_name
        self.timeout_phase = timeout_phase
        self.connect_timeout_sec = connect_timeout_sec
        self.read_timeout_sec = read_timeout_sec
        self.write_timeout_sec = write_timeout_sec
        self.pool_timeout_sec = pool_timeout_sec
        super().__init__(
            "LLM request timed out "
            f"(request_id={request_id}, session_id={session_id}, "
            f"request_sequence={request_sequence}, model={model_name}, "
            f"phase={timeout_phase}, connect_timeout={connect_timeout_sec}, "
            f"read_timeout={read_timeout_sec}, write_timeout={write_timeout_sec}, "
            f"pool_timeout={pool_timeout_sec})"
        )


class OpenAITransportConnectTimeoutError(LLMRequestTimeoutError):
    """The OpenAI-compatible transport could not establish a connection."""


class ContextManagementInfrastructureError(Exception):
    """Raised when Harbor's own context-management (summarization / budgeting)
    failed to produce a valid request.

    This is an *infrastructure* failure, not a model answer: the evaluated model
    never returned a final response and the verifier never scored one. Harbor
    owns budgeting, summarization, and request construction, so a failure here
    must be retryable (see ``_RESUME_RETRYABLE_ERROR_TYPES``) rather than
    persisted as a generic provider error or counted toward model quality.

    Provider context-length signals remain recoverable. This exception is raised
    only after Harbor cannot construct an admissible request or exhausts the
    summary output-reserve ladder.
    """


class BaseLLM(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    async def call(self, prompt: str, **kwargs) -> LLMResponse:
        pass

    async def build_continuation_prompt_token_ids(
        self,
        prompt: str,
        previous_prompt_token_ids: list[int],
        previous_completion_token_ids: list[int],
        **kwargs,
    ) -> list[int] | None:
        """Build an exact token prompt for a continuation request when supported."""
        return None

    @abstractmethod
    def get_model_context_limit(self) -> int:
        """Get the context limit (max input tokens) for the current model.

        Returns:
            int: The maximum input tokens the model can accept, or a fallback value if unavailable.
        """
        pass

    @abstractmethod
    def get_model_output_limit(self) -> int | None:
        """Get the output limit (max output tokens) for the current model.

        Returns:
            int | None: The maximum output tokens the model can generate, or None if unavailable.
        """
        pass
