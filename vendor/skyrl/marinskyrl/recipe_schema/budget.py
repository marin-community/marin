"""The authored request budget and its launch-side token limits."""

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import Field, PositiveInt, model_validator

from .model import Section, unset_field


class ContextBudget(Section):
    """Reserve a complete response within one rollout request window."""

    request_window_tokens: PositiveInt
    serving_window_tokens: PositiveInt = unset_field()
    max_new_tokens_per_turn: PositiveInt
    max_turns: PositiveInt
    generated_budget_fraction: Annotated[int | float, Field(gt=0, le=1)] = 0.5
    overlong_cache_fraction: Annotated[int | float, Field(ge=0, le=1)] = 0.25

    @model_validator(mode="after")
    def _response_fits(self) -> Self:
        if self.request_window_tokens <= self.max_new_tokens_per_turn:
            raise ValueError("request_window_tokens must exceed max_new_tokens_per_turn")
        if "serving_window_tokens" in self.model_fields_set and self.serving_window_tokens < self.request_window_tokens:
            raise ValueError("serving_window_tokens must cover the complete request_window_tokens")
        return self

    @property
    def max_input_tokens(self) -> int:
        return self.request_window_tokens - self.max_new_tokens_per_turn

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
            "request_window_tokens": self.request_window_tokens,
            "serving_window_tokens": self.serving_window_tokens
            if "serving_window_tokens" in self.model_fields_set
            else self.request_window_tokens,
            "max_new_tokens_per_turn": self.max_new_tokens_per_turn,
            "max_turns": self.max_turns,
            "generated_budget_fraction": float(self.generated_budget_fraction),
            "overlong_cache_fraction": float(self.overlong_cache_fraction),
            "max_input_tokens": self.max_input_tokens,
            "generated_tokens_per_trajectory": self.generated_tokens_per_trajectory,
            "overlong_cache_tokens": self.overlong_cache_tokens,
        }

    def derived_values(self, terminal_bench: Mapping[str, Any] | None) -> dict[str, int]:
        values = {
            "trainer.max_prompt_length": self.max_input_tokens,
            "generator.max_input_length": self.max_input_tokens,
            "generator.max_turns": self.max_turns,
            "generator.sampling_params.max_generate_length": self.max_new_tokens_per_turn,
            "generator.engine_init_kwargs.max_model_len": self.serving_window_tokens
            if "serving_window_tokens" in self.model_fields_set
            else self.request_window_tokens,
            "generator.trajectory_reward_shaping.overlong.l_max": self.generated_tokens_per_trajectory,
            "generator.trajectory_reward_shaping.overlong.l_cache": self.overlong_cache_tokens,
            "environment.task_sessions.session.max_turns": self.max_turns,
        }
        if terminal_bench is not None:
            values["terminal_bench.harbor.max_turns"] = self.max_turns
        return values
