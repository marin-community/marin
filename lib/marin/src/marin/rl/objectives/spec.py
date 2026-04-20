# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable public objective specification for RL/post-training."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class BatchView(StrEnum):
    """Batch view consumed by an objective runtime."""

    SEQUENCE = "sequence"
    GROUP = "group"


class ReductionKind(StrEnum):
    """Supported token-objective reduction semantics."""

    PPO = "ppo"
    DAPO = "dapo"
    GRPO = "grpo"


class TruncationPolicy(StrEnum):
    """Policy for handling truncated responses inside objective terms."""

    KEEP = "keep"
    FILTER_ENTIRE_RESPONSE = "filter_entire_response"


@dataclass(frozen=True)
class RLOOSignalConfig:
    """Reward leave-one-out sequence-advantage signal."""

    name: str = "rloo"


@dataclass(frozen=True)
class NoRewardSignalConfig:
    """Signal with zero advantages, used by reward-free objectives."""

    name: str = "none"


SignalConfig: TypeAlias = RLOOSignalConfig | NoRewardSignalConfig


@dataclass(frozen=True)
class PolicyGradientTermConfig:
    """Policy-gradient term with clipping and optional mismatch correction."""

    clip_epsilon_low: float
    clip_epsilon_high: float
    tis_importance_sampling_ratio_max: float
    do_trainer_inference_mismatch_importance_sampling: bool = False
    synchronous: bool = False


@dataclass(frozen=True)
class ReferenceKLTermConfig:
    """Reference-model KL penalty."""

    kl_coef: float


LossTermConfig: TypeAlias = PolicyGradientTermConfig | ReferenceKLTermConfig


@dataclass(frozen=True)
class ReductionConfig:
    """How token-level objectives are normalized into a scalar loss."""

    kind: ReductionKind


@dataclass(frozen=True)
class ObjectiveSpec:
    """Stable public surface for configuring RL/post-training objectives."""

    batch_view: BatchView
    signal_builder: SignalConfig
    terms: tuple[LossTermConfig, ...]
    reduction: ReductionConfig
    truncation_policy: TruncationPolicy = TruncationPolicy.KEEP
