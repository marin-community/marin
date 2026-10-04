"""The authored context declaration and its launch-side token limits."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import Field, PositiveInt, model_validator

from .documents import MISSING, get_path
from .model import Section
from .ownership import DERIVED_PATHS, REMOVED


class ContextBudget(Section):
    """Reserve a complete response within one rollout request window."""

    request_window_tokens: PositiveInt
    max_new_tokens_per_turn: PositiveInt
    max_turns: PositiveInt
    generated_budget_fraction: Annotated[int | float, Field(gt=0, le=1)] = 0.5
    overlong_cache_fraction: Annotated[int | float, Field(ge=0, le=1)] = 0.25

    @model_validator(mode="after")
    def _response_fits(self) -> Self:
        if self.request_window_tokens <= self.max_new_tokens_per_turn:
            raise ValueError("request_window_tokens must exceed max_new_tokens_per_turn")
        return self

    @property
    def max_input_tokens(self) -> int:
        return self.request_window_tokens - self.max_new_tokens_per_turn

    @property
    def opencode_limit_output(self) -> int:
        return min(self.max_new_tokens_per_turn, max(1, self.max_input_tokens - 1))

    @property
    def opencode_limit_context(self) -> int:
        output = self.opencode_limit_output
        margin = min(1024, max(0, self.max_input_tokens - output - 1))
        return max(1, self.max_input_tokens - output - margin)

    @property
    def generated_tokens_per_trajectory(self) -> int:
        if self.max_turns == 1:
            return self.max_new_tokens_per_turn
        return max(1, int(self.request_window_tokens * self.generated_budget_fraction))

    @property
    def overlong_cache_tokens(self) -> int:
        return int(self.generated_tokens_per_trajectory * self.overlong_cache_fraction)

    def as_dict(self) -> dict[str, int | float]:
        return {
            **self.model_dump(),
            "max_input_tokens": self.max_input_tokens,
            "generated_tokens_per_trajectory": self.generated_tokens_per_trajectory,
            "overlong_cache_tokens": self.overlong_cache_tokens,
            "opencode_limit_context": self.opencode_limit_context,
            "opencode_limit_output": self.opencode_limit_output,
        }


def resolve_context_budget(raw: Mapping[str, Any], config_path: Path) -> ContextBudget:
    """Validate authored context fields and return their launch token budget."""
    for path, message in REMOVED.items():
        if get_path(raw, path) is not MISSING:
            raise ValueError(f"{config_path}: {path}: {message}")
    declared = sorted(path for path in DERIVED_PATHS if get_path(raw, path) is not MISSING)
    if declared:
        raise ValueError(
            f"{config_path} declares derived context fields: {', '.join(declared)}; set context_budget instead"
        )
    config = raw.get("context_budget")
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_path}: context_budget must be a mapping")
    budget = ContextBudget.model_validate(config)
    return ContextBudget(
        request_window_tokens=budget.request_window_tokens,
        max_new_tokens_per_turn=budget.max_new_tokens_per_turn,
        max_turns=budget.max_turns,
        generated_budget_fraction=float(budget.generated_budget_fraction),
        overlong_cache_fraction=float(budget.overlong_cache_fraction),
    )
