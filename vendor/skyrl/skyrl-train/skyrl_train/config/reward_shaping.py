"""Configuration for trajectory reward shaping."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
from typing import Any

from skyrl_train.config.reasoning_span_penalty import ReasoningSpanPenaltyConfig


SHAPING_SCHEMA_VERSION = 2
DEFAULT_ACCEPTED_STOP_REASONS = ("complete", "end_turn", "eos", "stop")


@dataclass(frozen=True)
class LoopCreditConfig:
    max_period_tokens: int = 64
    tail_tokens: int = 256
    minimum_occurrences: int = 4
    advantage_penalty_per_token: float = 0.0
    max_advantage_penalty: float = 0.2


@dataclass(frozen=True)
class NonTerminationPenaltyConfig:
    penalty: float = 0.0
    accepted_stop_reasons: tuple[str, ...] = DEFAULT_ACCEPTED_STOP_REASONS


@dataclass(frozen=True)
class PassthroughPenaltyConfig:
    penalty: float = 0.0


@dataclass(frozen=True)
class SuccessfulLengthPenaltyConfig:
    free_tokens: int = 0
    penalty_per_token: float = 0.0
    max_penalty: float = 0.2


@dataclass(frozen=True)
class OverlongPenaltyConfig:
    """DAPO soft-overlong parameters, named to match the paper's formula."""

    l_max: int = 0
    """Maximum generated trajectory length."""

    l_cache: int = 0
    """Penalty-window width below ``l_max``; zero disables the component."""

    penalty_scale: float = 1.0
    """Maximum penalty magnitude, scaled to the optimization reward range."""


@dataclass(frozen=True)
class TrajectoryRewardShapingConfig:
    schema_version: int = SHAPING_SCHEMA_VERSION
    enabled: bool = False
    loop: LoopCreditConfig = LoopCreditConfig()
    passthrough: PassthroughPenaltyConfig = PassthroughPenaltyConfig()
    non_termination: NonTerminationPenaltyConfig = NonTerminationPenaltyConfig()
    overlong: OverlongPenaltyConfig = OverlongPenaltyConfig()
    successful_length: SuccessfulLengthPenaltyConfig = SuccessfulLengthPenaltyConfig()
    reasoning_span: ReasoningSpanPenaltyConfig = ReasoningSpanPenaltyConfig()


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"generator.trajectory_reward_shaping.{name} must be a mapping")
    return value


def parse_trajectory_reward_shaping_config(config: Mapping[str, Any] | None) -> TrajectoryRewardShapingConfig:
    """Parse and validate the runner-independent reward-shaping configuration."""
    if config is None:
        return TrajectoryRewardShapingConfig()
    if not isinstance(config, Mapping):
        raise ValueError("generator.trajectory_reward_shaping must be a mapping")

    loop = _section(config, "loop")
    allowed_loop_keys = {field.name for field in fields(LoopCreditConfig)}
    unknown_loop_keys = set(loop).difference(allowed_loop_keys)
    if unknown_loop_keys:
        raise ValueError(f"unknown loop settings: {', '.join(sorted(unknown_loop_keys))}")
    non_termination = _section(config, "non_termination")
    passthrough = _section(config, "passthrough")
    overlong = _section(config, "overlong")
    successful_length = _section(config, "successful_length")
    defaults = TrajectoryRewardShapingConfig()
    raw_stop_reasons = non_termination.get("accepted_stop_reasons", defaults.non_termination.accepted_stop_reasons)
    if not isinstance(raw_stop_reasons, Sequence) or isinstance(raw_stop_reasons, (str, bytes)):
        raise ValueError("non_termination.accepted_stop_reasons must be a sequence of strings")
    accepted_stop_reasons = tuple(str(reason).strip().lower() for reason in raw_stop_reasons)
    parsed = TrajectoryRewardShapingConfig(
        schema_version=int(config.get("schema_version", defaults.schema_version)),
        enabled=bool(config.get("enabled", defaults.enabled)),
        loop=LoopCreditConfig(
            max_period_tokens=int(loop.get("max_period_tokens", defaults.loop.max_period_tokens)),
            tail_tokens=int(loop.get("tail_tokens", defaults.loop.tail_tokens)),
            minimum_occurrences=int(loop.get("minimum_occurrences", defaults.loop.minimum_occurrences)),
            advantage_penalty_per_token=float(
                loop.get("advantage_penalty_per_token", defaults.loop.advantage_penalty_per_token)
            ),
            max_advantage_penalty=float(loop.get("max_advantage_penalty", defaults.loop.max_advantage_penalty)),
        ),
        passthrough=PassthroughPenaltyConfig(
            penalty=float(passthrough.get("penalty", defaults.passthrough.penalty)),
        ),
        non_termination=NonTerminationPenaltyConfig(
            penalty=float(non_termination.get("penalty", defaults.non_termination.penalty)),
            accepted_stop_reasons=accepted_stop_reasons,
        ),
        overlong=OverlongPenaltyConfig(
            l_max=int(overlong.get("l_max", defaults.overlong.l_max)),
            l_cache=int(overlong.get("l_cache", defaults.overlong.l_cache)),
            penalty_scale=float(overlong.get("penalty_scale", defaults.overlong.penalty_scale)),
        ),
        successful_length=SuccessfulLengthPenaltyConfig(
            free_tokens=int(successful_length.get("free_tokens", defaults.successful_length.free_tokens)),
            penalty_per_token=float(
                successful_length.get("penalty_per_token", defaults.successful_length.penalty_per_token)
            ),
            max_penalty=float(successful_length.get("max_penalty", defaults.successful_length.max_penalty)),
        ),
        reasoning_span=ReasoningSpanPenaltyConfig.from_section(_section(config, "reasoning_span")),
    )
    _validate_config(parsed)
    return parsed


def _validate_config(config: TrajectoryRewardShapingConfig) -> None:
    if config.schema_version != SHAPING_SCHEMA_VERSION:
        raise ValueError(
            f"trajectory reward shaping schema_version must be {SHAPING_SCHEMA_VERSION}, got {config.schema_version}"
        )
    if config.loop.max_period_tokens < 1:
        raise ValueError("loop.max_period_tokens must be at least 1")
    if config.loop.tail_tokens < config.loop.max_period_tokens * config.loop.minimum_occurrences:
        raise ValueError("loop.tail_tokens must cover max_period_tokens * minimum_occurrences")
    if config.loop.minimum_occurrences < 2:
        raise ValueError("loop.minimum_occurrences must be at least 2")
    if config.successful_length.free_tokens < 0:
        raise ValueError("successful_length.free_tokens must be non-negative")
    if config.overlong.l_max < 0:
        raise ValueError("overlong.l_max must be non-negative")
    if config.overlong.l_cache < 0:
        raise ValueError("overlong.l_cache must be non-negative")
    if config.overlong.l_cache > config.overlong.l_max:
        raise ValueError("overlong.l_cache must not exceed overlong.l_max")
    if not math.isfinite(config.overlong.penalty_scale) or config.overlong.penalty_scale < 0:
        raise ValueError("overlong.penalty_scale must be non-negative and finite")

    magnitudes = {
        "loop.advantage_penalty_per_token": config.loop.advantage_penalty_per_token,
        "loop.max_advantage_penalty": config.loop.max_advantage_penalty,
        "passthrough.penalty": config.passthrough.penalty,
        "non_termination.penalty": config.non_termination.penalty,
        "successful_length.penalty_per_token": config.successful_length.penalty_per_token,
        "successful_length.max_penalty": config.successful_length.max_penalty,
        "reasoning_span.penalty": config.reasoning_span.penalty,
    }
    for name, value in magnitudes.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.loop.advantage_penalty_per_token > 0 and config.loop.max_advantage_penalty == 0:
        raise ValueError(
            "loop.max_advantage_penalty must be positive when loop.advantage_penalty_per_token is positive"
        )
    if config.non_termination.penalty > 0 and not config.non_termination.accepted_stop_reasons:
        raise ValueError("non_termination.accepted_stop_reasons cannot be empty when its penalty is positive")
