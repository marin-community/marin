from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
import re
from typing import Any

import numpy as np

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.trajectory_runners.types import (
    REWARD_SHAPING_COMPONENT_NAMES,
    TrajectoryBatch,
    RewardShapingComponents,
    RewardShapingLoopSpan,
)


SHAPING_METRIC_PREFIX = "generate/reward_shaping"
SHAPING_SCHEMA_VERSION = 2
REWARD_SHAPING_ROW_KEYS = (
    "reward_shaping_components",
    "reward_shaping_loop_spans",
    "loop_advantages",
    "reward_shaping_versions",
    "verifier_tests",
    "verification_results",
    "evidence_messages",
    "exception_types",
    "error_treatments",
    "server_errors",
)
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


@dataclass(frozen=True)
class NormalizedReward:
    values: tuple[float, ...]
    scalar: bool

    @classmethod
    def from_output(cls, reward: float | Sequence[float]) -> "NormalizedReward":
        if isinstance(reward, Sequence) and not isinstance(reward, (str, bytes)):
            return cls(tuple(float(value) for value in reward), scalar=False)
        return cls((float(reward),), scalar=True)

    @property
    def outcome(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def total(self) -> float:
        return sum(self.values)

    def to_output(self) -> float | list[float]:
        return self.values[0] if self.scalar else list(self.values)

    def with_penalty(self, loss_mask: Sequence[int], penalty: float) -> float | list[float]:
        if self.scalar:
            return self.values[0] + penalty
        shaped = list(self.values)
        if penalty == 0:
            return shaped
        active_positions = [position for position, active in enumerate(loss_mask) if active]
        if not active_positions:
            raise ValueError("cannot apply a trajectory penalty without a trainable response token")
        shaped[active_positions[-1]] += penalty
        return shaped


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


def _active_segments(token_ids: Sequence[int], loss_mask: Sequence[int]) -> list[tuple[list[int], list[int]]]:
    if len(token_ids) != len(loss_mask):
        raise ValueError("response IDs and loss masks must have the same length")

    segments: list[tuple[list[int], list[int]]] = []
    segment_tokens: list[int] = []
    segment_positions: list[int] = []
    for position, (token_id, active) in enumerate(zip(token_ids, loss_mask)):
        if active:
            segment_tokens.append(int(token_id))
            segment_positions.append(position)
        elif segment_tokens:
            segments.append((segment_tokens, segment_positions))
            segment_tokens = []
            segment_positions = []
    if segment_tokens:
        segments.append((segment_tokens, segment_positions))
    return segments


@dataclass(frozen=True)
class TailLoop:
    """Token coordinates for a periodic suffix within the last trainable segment."""

    start: int
    period: int
    end: int

    @property
    def repeat_start(self) -> int:
        return self.start + self.period


def tail_loop(
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    config: LoopCreditConfig,
) -> TailLoop | None:
    """Find a repeating trainable suffix and return its original token coordinates."""
    segments = _active_segments(response_ids, loss_mask)
    if not segments:
        return None

    # Only a loop that survives to the final trainable suffix is charged. An
    # earlier loop followed by more assistant work is a recovery, not a failure.
    segment_tokens, segment_positions = segments[-1]
    tail = segment_tokens[-config.tail_tokens :]
    max_period = min(config.max_period_tokens, len(tail) // config.minimum_occurrences)
    period = next(
        (
            candidate
            for candidate in range(1, max_period + 1)
            if all(
                tail[-offset] == tail[-offset - candidate]
                for offset in range(1, candidate * (config.minimum_occurrences - 1) + 1)
            )
        ),
        None,
    )
    if period is None:
        return None

    periodic_start = len(segment_tokens) - period * config.minimum_occurrences
    while periodic_start > 0 and segment_tokens[periodic_start - 1] == segment_tokens[periodic_start - 1 + period]:
        periodic_start -= 1
    return TailLoop(segment_positions[periodic_start], period, segment_positions[-1] + 1)


def _tail_loop_span(
    response_ids: Sequence[int], loss_mask: Sequence[int], config: LoopCreditConfig
) -> RewardShapingLoopSpan | None:
    hit = tail_loop(response_ids, loss_mask, config)
    if hit is None:
        return None
    return {"start": hit.start + hit.period * (config.minimum_occurrences - 1), "end": hit.end}


def _build_loop_credit(
    response_ids: Sequence[Sequence[int]],
    loss_masks: Sequence[Sequence[int]],
    groups: Sequence[Sequence[int]],
    excluded: Sequence[bool],
    config: LoopCreditConfig,
) -> tuple[list[list[RewardShapingLoopSpan]], list[list[float]]]:
    final_indices = {group[-1] for group in groups}
    detected_spans = [
        _tail_loop_span(response, loss_mask, config) if index in final_indices else None
        for index, (response, loss_mask) in enumerate(zip(response_ids, loss_masks))
    ]
    spans = [[span] if span is not None else [] for span in detected_spans]
    advantages = [[0.0] * len(response) for response in response_ids]
    for group in groups:
        final_index = group[-1]
        span = detected_spans[final_index]
        charged_positions = set() if span is None else set(range(span["start"], span["end"]))
        if excluded[final_index] or not charged_positions or config.advantage_penalty_per_token == 0:
            continue
        realized_penalty = min(
            config.advantage_penalty_per_token,
            config.max_advantage_penalty / len(charged_positions),
        )
        for position in charged_positions:
            advantages[final_index][position] = -realized_penalty
    return spans, advantages


def _metric_key(value: str | None) -> str:
    normalized = "missing" if value is None else str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "missing"


def _empty_components() -> RewardShapingComponents:
    return {"passthrough": 0.0, "non_termination": 0.0, "overlong": 0.0, "successful_length": 0.0}


def aggregate_reward_shaping_components(
    components: Sequence[RewardShapingComponents], indices: Sequence[int]
) -> RewardShapingComponents:
    return {name: sum(components[index].get(name, 0.0) for index in indices) for name in REWARD_SHAPING_COMPONENT_NAMES}


def _soft_overlong_penalty(response_length: int, config: OverlongPenaltyConfig) -> float:
    if config.l_cache == 0:
        return 0.0
    penalty_start = config.l_max - config.l_cache
    if response_length <= penalty_start:
        return 0.0
    penalty_fraction = min((response_length - penalty_start) / config.l_cache, 1.0)
    return -penalty_fraction * config.penalty_scale


def _trajectory_groups(output: TrajectoryBatch, batch_size: int) -> list[list[int]]:
    is_last_step = output.get("is_last_step")
    if is_last_step is None:
        return [[index] for index in range(batch_size)]
    if len(is_last_step) != batch_size:
        raise ValueError("is_last_step must have one entry per generated row")

    groups: list[list[int]] = []
    current: list[int] = []
    for index, last in enumerate(is_last_step):
        current.append(index)
        if last:
            groups.append(current)
            current = []
    if current:
        raise ValueError("step-wise trajectory batch must mark the final row of every trajectory")
    return groups


def refresh_trajectory_reward_shaping_metrics(output: TrajectoryBatch) -> None:
    """Recompute shaping metrics after concatenation, filtering, or replacement."""
    components = output.get("reward_shaping_components")
    if components is None:
        return

    batch_size = len(output["response_ids"])
    if len(components) != batch_size:
        raise ValueError("reward shaping components must have one entry per generated row")
    outcomes = output.get("unshaped_rewards")
    spans = output.get("reward_shaping_loop_spans")
    loop_advantages = output.get("loop_advantages")
    if (
        outcomes is None
        or spans is None
        or loop_advantages is None
        or len(outcomes) != batch_size
        or len(spans) != batch_size
        or len(loop_advantages) != batch_size
    ):
        raise ValueError("shaped outputs must retain outcomes, loop spans, and loop advantages for every generated row")

    stop_reasons = output.get("stop_reasons")
    if stop_reasons is None:
        stop_reasons = [None] * batch_size
    groups = _trajectory_groups(output, batch_size)
    response_lengths = [sum(bool(value) for value in loss_mask) for loss_mask in output["loss_masks"]]
    trajectory_components = [aggregate_reward_shaping_components(components, group) for group in groups]
    shaped_totals = [NormalizedReward.from_output(output["rewards"][group[-1]]).total for group in groups]
    penalties = [sum(values.values()) for values in trajectory_components]
    trajectory_lengths = [sum(response_lengths[index] for index in group) for group in groups]
    trajectory_stops = [stop_reasons[group[-1]] for group in groups]
    trajectory_outcomes = [float(outcomes[group[-1]]) for group in groups]
    trajectory_loop_advantages = [sum(sum(loop_advantages[index]) for index in group) for group in groups]
    trajectory_loop_token_counts = [
        sum(sum(value < 0 for value in loop_advantages[index]) for index in group) for group in groups
    ]
    charged_loop_advantages = [
        value for sample_advantages in loop_advantages for value in sample_advantages if value < 0
    ]
    loop_incidence = [any(spans[index] for index in group) for group in groups]
    correct_loop_incidence = [
        incidence for incidence, outcome in zip(loop_incidence, trajectory_outcomes) if outcome > 0
    ]

    metrics = output.get("rollout_metrics") or {}
    refreshed_metrics = {
        f"{SHAPING_METRIC_PREFIX}/outcome_reward_mean": float(np.mean(trajectory_outcomes)),
        f"{SHAPING_METRIC_PREFIX}/optimization_reward_before_mean": float(
            np.mean([shaped - penalty for shaped, penalty in zip(shaped_totals, penalties)])
        ),
        f"{SHAPING_METRIC_PREFIX}/shaped_reward_mean": float(np.mean(shaped_totals)),
        f"{SHAPING_METRIC_PREFIX}/penalty_mean": float(np.mean(penalties)),
        f"{SHAPING_METRIC_PREFIX}/passthrough_penalty_mean": float(
            np.mean([values.get("passthrough", 0.0) for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/non_termination_penalty_mean": float(
            np.mean([values["non_termination"] for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/successful_length_penalty_mean": float(
            np.mean([values["successful_length"] for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/overlong_penalty_mean": float(
            np.mean([values["overlong"] for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/loop_incidence": float(np.mean(loop_incidence)),
        f"{SHAPING_METRIC_PREFIX}/loop_incidence_correct": float(
            np.mean(correct_loop_incidence) if correct_loop_incidence else 0.0
        ),
        f"{SHAPING_METRIC_PREFIX}/loop_advantage_mean": float(np.mean(trajectory_loop_advantages)),
        f"{SHAPING_METRIC_PREFIX}/loop_advantage_per_token_mean": float(
            np.mean(charged_loop_advantages) if charged_loop_advantages else 0.0
        ),
        f"{SHAPING_METRIC_PREFIX}/loop_charged_tokens_mean": float(np.mean(trajectory_loop_token_counts)),
        f"{SHAPING_METRIC_PREFIX}/non_termination_incidence": float(
            np.mean([values["non_termination"] < 0 for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/passthrough_incidence": float(
            np.mean([values.get("passthrough", 0.0) < 0 for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/successful_length_incidence": float(
            np.mean([values["successful_length"] < 0 for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/overlong_incidence": float(
            np.mean([values["overlong"] < 0 for values in trajectory_components])
        ),
        f"{SHAPING_METRIC_PREFIX}/response_tokens_mean": float(np.mean(trajectory_lengths)),
        f"{SHAPING_METRIC_PREFIX}/response_tokens_max": float(max(trajectory_lengths, default=0)),
    }
    stop_reason_prefix = f"{SHAPING_METRIC_PREFIX}/stop_reason/"
    for key in [key for key in metrics if key in refreshed_metrics or key.startswith(stop_reason_prefix)]:
        del metrics[key]
    metrics.update(refreshed_metrics)
    for stop_reason in trajectory_stops:
        key = f"{SHAPING_METRIC_PREFIX}/stop_reason/{_metric_key(stop_reason)}"
        metrics[key] = metrics.get(key, 0.0) + 1.0
    output["rollout_metrics"] = metrics


def shape_trajectory_rewards(output: TrajectoryBatch, raw_config: Mapping[str, Any] | None) -> None:
    """Build runner-independent reward penalties and token-local loop credit.

    Raw task outcomes are copied to ``unshaped_rewards`` before any shared
    shaping. Pass-through, non-termination, DAPO soft-overlong, and successful-length
    penalties remain additive to the optimization reward. Loop detection
    instead emits ``loop_advantages``; the trainer adds that channel after
    advantage normalization. Neither path alters pass-rate or verifier-accuracy
    metrics that consume the raw channel.
    """
    config = parse_trajectory_reward_shaping_config(raw_config)
    if not config.enabled:
        return

    response_ids = output["response_ids"]
    loss_masks = output["loss_masks"]
    rewards = output["rewards"]
    batch_size = len(response_ids)
    if not (len(loss_masks) == len(rewards) == batch_size):
        raise ValueError("response IDs, loss masks, and rewards must have the same batch size")

    stop_reasons = output.get("stop_reasons")
    if stop_reasons is None:
        stop_reasons = [None] * batch_size
    if len(stop_reasons) != batch_size:
        raise ValueError("stop reasons must have one entry per trajectory")

    error_treatments = output.get("error_treatments")
    if error_treatments is None:
        error_treatments = [None] * batch_size
    if len(error_treatments) != batch_size:
        raise ValueError("error treatments must have one entry per trajectory")

    normalized_rewards = [NormalizedReward.from_output(reward) for reward in rewards]
    existing_outcomes = output.get("unshaped_rewards")
    if existing_outcomes is not None and len(existing_outcomes) != batch_size:
        raise ValueError("unshaped rewards must have one entry per trajectory")
    outcomes = (
        [float(value) for value in existing_outcomes]
        if existing_outcomes is not None
        else [reward.outcome for reward in normalized_rewards]
    )
    output["unshaped_rewards"] = outcomes

    excluded = output.get("exclude_from_baseline")
    if excluded is None:
        excluded = [False] * batch_size
    if len(excluded) != batch_size:
        raise ValueError("exclude_from_baseline must have one entry per trajectory")

    components = [_empty_components() for _ in range(batch_size)]
    shaped_rewards = [reward.to_output() for reward in normalized_rewards]
    response_lengths = [sum(bool(value) for value in loss_mask) for loss_mask in loss_masks]
    accepted_stops = set(config.non_termination.accepted_stop_reasons)
    groups = _trajectory_groups(output, batch_size)
    all_loop_spans, loop_advantages = _build_loop_credit(
        response_ids,
        loss_masks,
        groups,
        excluded,
        config.loop,
    )

    for group in groups:
        final_index = group[-1]
        trajectory_length = sum(response_lengths[index] for index in group)
        final_components = components[final_index]

        if not excluded[final_index]:
            normalized_treatment = (
                None if error_treatments[final_index] is None else str(error_treatments[final_index]).strip().lower()
            )
            if normalized_treatment == ErrorTreatment.PASSTHROUGH.value:
                final_components["passthrough"] = -config.passthrough.penalty
            final_components["overlong"] = _soft_overlong_penalty(trajectory_length, config.overlong)
            normalized_stop = (
                None if stop_reasons[final_index] is None else str(stop_reasons[final_index]).strip().lower()
            )
            if normalized_stop not in accepted_stops:
                final_components["non_termination"] = -config.non_termination.penalty
            if outcomes[final_index] > 0:
                charged_tokens = max(0, trajectory_length - config.successful_length.free_tokens)
                final_components["successful_length"] = -min(
                    config.successful_length.max_penalty,
                    charged_tokens * config.successful_length.penalty_per_token,
                )

        penalty = sum(sum(components[index].values()) for index in group)
        shaped_rewards[final_index] = normalized_rewards[final_index].with_penalty(loss_masks[final_index], penalty)

    output["rewards"] = shaped_rewards
    output["reward_shaping_components"] = components
    output["reward_shaping_loop_spans"] = all_loop_spans
    output["loop_advantages"] = loop_advantages
    output["reward_shaping_versions"] = [config.schema_version] * batch_size
    refresh_trajectory_reward_shaping_metrics(output)
