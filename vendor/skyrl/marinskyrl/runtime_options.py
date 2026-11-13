"""Typed choices shared by the launcher and training runtime."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class R3Transport(StrEnum):
    BY_VALUE = "by_value"
    RESIDENT = "resident"
    DECENTRAL = "decentral"


class WeightSyncTransport(StrEnum):
    BROADCAST = "broadcast"
    EXPERT_BLOCK = "expert_block"


class GDNBackend(StrEnum):
    TORCH = "torch"
    FLASHQLA = "flashqla"


TRAJECTORY_SELECTOR_TYPE_PATH = "trainer.trajectory_selector.type"


class RolloutGrading(StrEnum):
    """Select terminal task grading without importing the rollout runtime."""

    VERIFY = "verify"
    SKIP = "skip"


class AdvantageEstimator(StrEnum):
    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    RLOO_N = "rloo_n"  # RLOO-Neutral: excludes masked samples from baseline
    REINFORCE_PP = "reinforce++"
    UNIFORM = "uniform"
    REWARD = "reward"


class PolicyLossType(StrEnum):
    REGULAR = "regular"
    DUAL_CLIP = "dual_clip"
    BEHAVIOR_CLIP = "behavior_clip"
    GSPO = "gspo"
    CISPO = "cispo"
    CLIP_COV = "clip_cov"
    KL_COV = "kl_cov"
    SAPO = "sapo"
    SFT = "sft"
    FTPO = "ftpo"
    DPO = "dpo"
    IMPORTANCE_SAMPLING = "importance_sampling"


PREFERENCE_PAIR_ENV_CLASS = "preference_pair"


class EvaluationRunner(StrEnum):
    TRAINING = "training"
    TASK = "task"
    HARBOR = "harbor"


def ftpo_requested(algorithm: Mapping[str, Any]) -> bool:
    """Return whether standalone or auxiliary FTPO is configured."""
    return (
        algorithm.get("policy_loss_type") == PolicyLossType.FTPO
        or algorithm.get("ftpo", {}).get("auxiliary_weight") is not None
    )


def reference_model_required(algorithm: Mapping[str, Any]) -> bool:
    """Return whether the objective needs a frozen reference actor."""
    return bool(
        algorithm.get("use_kl_loss")
        or algorithm.get("use_kl_in_reward")
        or ftpo_requested(algorithm)
        or algorithm.get("policy_loss_type") == PolicyLossType.DPO
    )


def static_preference_pairs_requested(config: Mapping[str, Any]) -> bool:
    """Return whether training reads static chosen/rejected completions instead of generating rollouts."""
    environment = config.get("environment") if isinstance(config, Mapping) else None
    if environment is None:
        return False
    return environment.get("env_class") == PREFERENCE_PAIR_ENV_CLASS


def inference_engines_required(config: Mapping[str, Any]) -> bool:
    """Static preference training needs generation engines for task validation."""
    if not static_preference_pairs_requested(config):
        return True
    return EvaluationRunner(config["trainer"]["evaluation_runner"]) is not EvaluationRunner.TRAINING
