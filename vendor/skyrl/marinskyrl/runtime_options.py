"""Typed choices shared by the launcher and training runtime."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
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
HARBOR_TOKEN_REWARD_DEFAULTS = MappingProxyType(
    {"enable_token_reward_channel": False, "enable_pbs_shaping": False, "enable_span_tagging": True}
)


def pbs_token_credit_enabled(cfg: Mapping[str, Any]) -> bool:
    """Return whether the trainer and Harbor enable PBS token credit."""
    terminal_bench = cfg.get("terminal_bench_config")
    if terminal_bench is None or not cfg["trainer"]["algorithm"]["enable_token_reward_channel"]:
        return False
    harbor = terminal_bench.get("harbor") or {}
    return all(bool(harbor.get(key, default)) for key, default in HARBOR_TOKEN_REWARD_DEFAULTS.items())


class RolloutGrading(StrEnum):
    """Select terminal task grading without importing the rollout runtime."""

    VERIFY = "verify"
    SKIP = "skip"


class AdvantageEstimator(StrEnum):
    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    RLOO_N = "rloo_n"  # RLOO with a configured minimum of baseline-eligible samples per group
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
