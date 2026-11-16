"""Where reward shaping charges a malformed reasoning span."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from omegaconf import DictConfig, OmegaConf


class ReasoningSpanPenaltyTarget(StrEnum):
    """Where a malformed reasoning span is charged.

    ROLLOUT subtracts the penalty once from the trajectory's optimization reward. MARKER subtracts it from the
    loop credit at the token where each span goes wrong, falling back to ROLLOUT when a violation has no such
    token.
    """

    ROLLOUT = "rollout"
    MARKER = "marker"


@dataclass(frozen=True)
class ReasoningSpanPenaltyConfig:
    penalty: float = 0.0
    target: ReasoningSpanPenaltyTarget = ReasoningSpanPenaltyTarget.ROLLOUT

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> "ReasoningSpanPenaltyConfig":
        return cls(
            penalty=float(section.get("penalty", cls.penalty)),
            target=ReasoningSpanPenaltyTarget(section.get("target", cls.target)),
        )

    @property
    def charges_tokens(self) -> bool:
        return self.target == ReasoningSpanPenaltyTarget.MARKER and self.penalty > 0


def charges_reasoning_span_tokens(cfg: DictConfig) -> bool:
    """Whether reward shaping charges malformed reasoning spans to tokens in the loop credit."""
    section = OmegaConf.select(cfg, "generator.trajectory_reward_shaping.reasoning_span", default=None) or {}
    return ReasoningSpanPenaltyConfig.from_section(section).charges_tokens
