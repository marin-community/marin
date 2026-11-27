"""Request-scoped usage of the single-row generation calls made for one policy turn."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class EngineGenerationUsage:
    turn_limit: int
    calls: int = 0
    tokens: int = 0
    turns: int = 0
    length_stops: int = 0
    unobserved_calls: int = 0
    unobserved_token_limit: int = 0

    def begin(self, limit: int | None) -> int:
        upper = self.turn_limit if limit is None else min(self.turn_limit, limit)
        self.calls += 1
        self.unobserved_calls += 1
        self.unobserved_token_limit += upper
        return upper

    def observed(self, upper: int, tokens: int, finish: str | None = None) -> None:
        self.tokens += tokens
        self.turns += int(finish is not None and finish != "abort")
        self.length_stops += int(finish == "length")
        self.unobserved_calls -= 1
        self.unobserved_token_limit -= upper


current_generation_usage: ContextVar[EngineGenerationUsage | None] = ContextVar("current_generation_usage", default=None)


@contextmanager
def measure_engine_generation(limit: int) -> Iterator[EngineGenerationUsage]:
    """Install the lease's meter across asynchronous chat dispatch and reset it on return."""
    usage = EngineGenerationUsage(limit)
    token = current_generation_usage.set(usage)
    try:
        yield usage
    finally:
        current_generation_usage.reset(token)
