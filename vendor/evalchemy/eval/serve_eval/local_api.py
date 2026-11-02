"""Evalchemy's OpenAI-compatible endpoint adapters."""

from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


class _TransportRetryBudget:
    def __init__(self, *args, transport_retry_budget=None, transport_attempt_timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transport_attempt_timeout = float(
            self.timeout if transport_attempt_timeout is None else transport_attempt_timeout
        )
        if self.transport_attempt_timeout <= 0:
            raise ValueError("transport_attempt_timeout must be positive")
        # Allow a full attempt and recovery time without shortening a caller's
        # request timeout. Explicit budgets retain their wall-clock semantics.
        self.transport_retry_budget = float(
            max(900, 3 * self.transport_attempt_timeout)
            if transport_retry_budget is None else transport_retry_budget
        )
        if self.transport_retry_budget <= 0:
            raise ValueError("transport_retry_budget must be positive")
        self.endpoint_concurrency = self._concurrent


class RetryingLocalCompletions(_TransportRetryBudget, LocalCompletionsAPI):
    """Local completions with a wall-clock transport retry budget."""


class RetryingLocalChatCompletions(_TransportRetryBudget, LocalChatCompletion):
    """Local chat completions with a wall-clock transport retry budget."""
