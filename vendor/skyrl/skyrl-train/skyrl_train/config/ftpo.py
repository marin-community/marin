"""Configuration for final-token preference optimization."""

import math
from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.runtime_options import PolicyLossType, ftpo_requested


class FTPOBoundary(StrEnum):
    TAIL_LOOP = "tail_loop"
    REPEATED_TOOL_OPENER = "repeated_tool_opener"
    VERIFIED_CLOSURE = "verified_closure"
    VERIFIED_CLOSURE_AND_CONTINUATION = "verified_closure_and_continuation"
    MIXED = "mixed"

    @property
    def uses_verified_boundaries(self) -> bool:
        return self in (
            FTPOBoundary.VERIFIED_CLOSURE,
            FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION,
            FTPOBoundary.MIXED,
        )


@dataclass(frozen=True)
class FTPOConfig:
    boundary: FTPOBoundary = FTPOBoundary.TAIL_LOOP
    margin: float = 2.0
    lambda_mse: float = 0.4
    lambda_mse_target: float = 0.05
    tau_mse_target: float = 0.5
    min_p: float = 0.01
    max_chosen_tokens: int = 20
    min_decoded_chars: int = 1
    require_alnum: bool = False
    rejected_balance_strength: float = 0.3
    chosen_balance_strength: float = 0.5
    early_stopping_chosen_win: float | None = None
    repair_sft_weight: float | None = None
    auxiliary_weight: float | None = None


def ftpo_config(algorithm: DictConfig) -> FTPOConfig | None:
    """Resolve standalone or explicitly weighted auxiliary FTPO parameters."""
    values = dict(algorithm.get("ftpo", {}))
    if not ftpo_requested(algorithm):
        return None
    if "boundary" in values:
        values["boundary"] = FTPOBoundary(values["boundary"])
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(FTPOConfig), values))
    if config.auxiliary_weight is not None:
        if not math.isfinite(config.auxiliary_weight) or config.auxiliary_weight <= 0:
            raise ValueError("FTPO auxiliary_weight must be finite and positive")
        if algorithm.get("policy_loss_type") == PolicyLossType.FTPO:
            raise ValueError("Choose standalone FTPO or auxiliary FTPO, not both")
    for value in (
        config.margin,
        config.lambda_mse,
        config.lambda_mse_target,
        config.tau_mse_target,
        config.min_p,
        config.rejected_balance_strength,
        config.chosen_balance_strength,
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError("FTPO coefficients must be finite and nonnegative")
    if config.margin == 0 or config.min_p > 1 or config.max_chosen_tokens < 1 or config.min_decoded_chars < 0:
        raise ValueError(
            "FTPO requires a positive margin and candidate count, min_p <= 1, and nonnegative character count"
        )
    if config.early_stopping_chosen_win is not None and not 0 < config.early_stopping_chosen_win <= 1:
        raise ValueError("FTPO early_stopping_chosen_win must be in (0, 1]")
    if config.repair_sft_weight is not None and (
        not math.isfinite(config.repair_sft_weight) or config.repair_sft_weight <= 0
    ):
        raise ValueError("FTPO repair_sft_weight must be finite and positive")
    return config


def validate_ftpo(cfg: DictConfig) -> None:
    """Validate FTPO execution and configure raw rollout log probabilities."""
    config = ftpo_config(cfg.trainer.algorithm)
    if config is None:
        return
    if config.boundary.uses_verified_boundaries and config.repair_sft_weight is None:
        raise ValueError("Verified closure training requires an explicit repair_sft_weight")
    algorithm = cfg.trainer.algorithm
    if cfg.trainer.strategy != "megatron":
        raise ValueError("FTPO requires the Megatron trainer")
    if cfg.trainer.use_sample_packing:
        raise ValueError("FTPO does not yet support sample packing")
    for role in (cfg.trainer.policy, cfg.trainer.ref):
        geometry = role.megatron_config
        if (
            role.sequence_parallel_size != 1
            or geometry.tensor_model_parallel_size != 1
            or geometry.context_parallel_size != 1
        ):
            raise ValueError("FTPO requires TP=1, CP=1, and sequence_parallel_size=1; sharded scoring is future work")
    if algorithm.loss_reduction != "token_mean" or algorithm.think_token_weight != 1:
        raise ValueError("FTPO requires token_mean reduction and think_token_weight=1")
    if config.auxiliary_weight is None and (
        algorithm.advantage_estimator != "uniform" or algorithm.advantage_batch_normalize
    ):
        raise ValueError("FTPO requires uniform advantages without advantage normalization")
    if config.auxiliary_weight is not None:
        if algorithm.policy_loss_type not in (
            PolicyLossType.REGULAR,
            PolicyLossType.IMPORTANCE_SAMPLING,
            PolicyLossType.BEHAVIOR_CLIP,
        ):
            raise ValueError("Auxiliary FTPO requires a task-reward policy objective")
        if cfg.generator.sampling_params.temperature != 1.0:
            raise ValueError("Auxiliary FTPO requires temperature 1 for shared policy and raw-logit scoring")
        if config.early_stopping_chosen_win is not None:
            raise ValueError("Auxiliary FTPO cannot stop task learning by structural chosen-win rate")
    if (
        algorithm.use_kl_loss
        or algorithm.use_kl_in_reward
        or algorithm.use_entropy_loss
        or algorithm.get("distillation")
    ):
        raise ValueError("FTPO supplies its own reference regularization; other objectives must be disabled")
    if (
        config.auxiliary_weight is None and algorithm.off_policy_correction not in (None, "none")
    ) or algorithm.dynamic_sampling.type is not None:
        raise ValueError("FTPO requires off_policy_correction=none and no reward-based dynamic sampling")
    if cfg.trainer.critic.model.path is not None or cfg.trainer.get("update_ref_every_epoch", False):
        raise ValueError("FTPO requires a frozen reference and no critic")
    if any(callback.get("type") == "ref_model_update" for callback in (cfg.trainer.get("callbacks") or [])):
        raise ValueError("FTPO cannot use a reference-update callback")
    if cfg.generator.backend != "vllm" or not cfg.generator.run_engines_locally:
        raise ValueError("FTPO requires local vLLM engines with exact token-ID top-K capture")
    if cfg.generator.sampling_params.logprobs is None or cfg.generator.sampling_params.logprobs < 2:
        raise ValueError("FTPO requires sampling_params.logprobs >= 2 for alternative tokens")
    # Candidate filtering needs model probabilities even when decoding is greedy.
    options = cfg.generator.engine_init_kwargs
    if options.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        raise ValueError("FTPO candidate filtering requires raw_logprobs")
    OmegaConf.update(options, "logprobs_mode", "raw_logprobs", force_add=True)
