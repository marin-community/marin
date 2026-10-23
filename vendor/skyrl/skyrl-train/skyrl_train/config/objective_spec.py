"""Torch-free contracts for policy objectives and their configuration."""

import math
from dataclasses import dataclass
from functools import cache
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    compile_distillation_plan_from_config,
)
from skyrl_train.config.ftpo import validate_ftpo
from skyrl_train.dynamic_sampling import DynamicSamplingType

from marinskyrl.runtime_options import (
    PREFERENCE_PAIR_ENV_CLASS,
    AdvantageEstimator,
    PolicyLossType,
)


class KLEstimator(StrEnum):
    K1 = "k1"
    ABS = "abs"
    K2 = "k2"
    K3 = "k3"
    K3_UNBIASED_GRADIENT = "k3_unbiased_gradient"


class RatioAnchor(StrEnum):
    OLD = "old"
    ROLLOUT = "rollout"
    NONE = "none"


@dataclass(frozen=True)
class LossSpec:
    anchor: RatioAnchor
    sequence_level: bool = False
    row_local: bool = True
    advantage_linear: bool = True


BUILTIN_LOSS_SPECS: Mapping[str, LossSpec] = MappingProxyType(
    {
        PolicyLossType.REGULAR: LossSpec(RatioAnchor.OLD),
        PolicyLossType.DUAL_CLIP: LossSpec(RatioAnchor.OLD),
        PolicyLossType.GSPO: LossSpec(RatioAnchor.OLD, sequence_level=True),
        PolicyLossType.CISPO: LossSpec(RatioAnchor.OLD),
        PolicyLossType.SAPO: LossSpec(RatioAnchor.OLD),
        PolicyLossType.CLIP_COV: LossSpec(RatioAnchor.OLD, row_local=False),
        PolicyLossType.KL_COV: LossSpec(RatioAnchor.OLD, row_local=False, advantage_linear=False),
        PolicyLossType.IMPORTANCE_SAMPLING: LossSpec(RatioAnchor.OLD),
        PolicyLossType.BEHAVIOR_CLIP: LossSpec(RatioAnchor.ROLLOUT),
        PolicyLossType.SFT: LossSpec(RatioAnchor.NONE, advantage_linear=False),
        PolicyLossType.FTPO: LossSpec(RatioAnchor.NONE, advantage_linear=False),
        PolicyLossType.DPO: LossSpec(RatioAnchor.NONE, advantage_linear=False),
    }
)


class LossReduction(StrEnum):
    TOKEN_MEAN = "token_mean"
    SEQUENCE_MEAN = "sequence_mean"
    PAIR_MEAN = "pair_mean"
    SEQ_MEAN_TOKEN_SUM_NORM = "seq_mean_token_sum_norm"
    SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL = "seq_mean_token_sum_norm_global"


class CorrectionAction(StrEnum):
    TRUNCATE = "truncate"
    MASK = "mask"


class SequenceAggregate(StrEnum):
    GEOMETRIC = "geometric"
    PRODUCT = "product"
    EXTREME_TOKEN = "extreme_token"


@dataclass(frozen=True)
class TokenRule:
    action: CorrectionAction
    low: float | int | None = None
    high: float | int | None = None


@dataclass(frozen=True)
class SequenceRule:
    aggregate: SequenceAggregate
    action: CorrectionAction
    low: float | int | None = None
    high: float | int | None = None


@dataclass(frozen=True)
class OffPolicyCorrection:
    name: str
    rules: tuple[TokenRule | SequenceRule, ...]

    @classmethod
    def from_config(cls, config: DictConfig) -> "OffPolicyCorrection":
        rules = []
        for rule in config.rules:
            kind = rule.get("kind")
            if kind not in {"token", "sequence"}:
                raise ValueError("off_policy_correction rule kind must be token or sequence; set kind on every rule")
            fields = OmegaConf.to_container(rule, resolve=True)
            del fields["kind"]
            for name, enum in (("action", CorrectionAction), ("aggregate", SequenceAggregate)):
                value = fields.get(name)
                if isinstance(value, str):
                    fields[name] = {member.value: member for member in enum}.get(value, value)
            schema = TokenRule if kind == "token" else SequenceRule
            try:
                parsed = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(schema), fields))
            except OmegaConfBaseException as error:
                raise ValueError(f"invalid off_policy_correction {kind} rule: {error}") from error
            action, low, high = parsed.action, parsed.low, parsed.high
            for bound in (low, high):
                if bound is not None and (not math.isfinite(bound) or bound <= 0):
                    raise ValueError("off_policy_correction bounds must be positive finite numbers")
            if low is not None and high is not None and low > high:
                raise ValueError("off_policy_correction requires low <= high")
            if action is CorrectionAction.TRUNCATE and (low is not None or high is None):
                raise ValueError("off_policy_correction truncate requires high and low=null")
            if action is CorrectionAction.MASK and low is None and high is None:
                raise ValueError("off_policy_correction mask requires low or high")
            if (
                isinstance(parsed, SequenceRule)
                and parsed.aggregate is SequenceAggregate.EXTREME_TOKEN
                and action is not CorrectionAction.MASK
            ):
                raise ValueError("off_policy_correction extreme_token requires action=mask")
            rules.append(parsed)
        if sum(rule.action is CorrectionAction.TRUNCATE for rule in rules) > 1:
            raise ValueError("off_policy_correction permits at most one truncate rule")
        return cls(str(config.name), tuple(rules))


@cache
def load_correction(name: str) -> OffPolicyCorrection:
    """Read a named correction's rules without importing the training runtime."""
    if name == "none":
        return OffPolicyCorrection(name, ())
    if name not in {"tis", "icepop", "seq_mask_tis", "outlier_mask"}:
        raise ValueError(f"unknown off_policy_correction {name!r}; use a preset, none, null, or custom with rules")
    config = OmegaConf.load(Path(__file__).parent / "off_policy_correction" / f"{name}.yaml")
    return OffPolicyCorrection.from_config(OmegaConf.create({"name": name, "rules": config.rules}))


def off_policy_correction(algorithm: DictConfig) -> OffPolicyCorrection:
    """Compute correction rules from the current algorithm config and immutable presets."""
    name = algorithm.off_policy_correction
    rules = algorithm.off_policy_correction_rules
    if name == "custom":
        if not rules:
            raise ValueError("off_policy_correction=custom requires off_policy_correction_rules")
        return OffPolicyCorrection.from_config(OmegaConf.create({"name": name, "rules": rules}))
    if rules:
        raise ValueError("off_policy_correction_rules requires off_policy_correction=custom")
    return load_correction("none" if name is None else str(name))


@dataclass(frozen=True)
class TopKLossParams:
    objective: DistillationObjectiveKind
    eps_clip_low: float
    eps_clip_high: float
    clip_ratio_c: float
    jsd_beta: float | None = None
    entry_clip: float | None = None


def topk_loss_params(algorithm: DictConfig) -> TopKLossParams:
    """Compute the teacher-loss parameters from an active top-K objective config."""
    return TopKLossParams(
        DistillationObjectiveKind(algorithm.distillation.objective),
        float(algorithm.eps_clip_low),
        float(algorithm.eps_clip_high),
        float(algorithm.clip_ratio_c),
        algorithm.distillation.get("jsd_beta"),
        algorithm.distillation.get("entry_clip"),
    )


def rollout_logprobs_required(algorithm: DictConfig, *, loss_spec: LossSpec | None = None) -> bool:
    """Compute whether active policy rows require behavior-policy log probabilities."""
    spec = loss_spec or BUILTIN_LOSS_SPECS.get(algorithm.policy_loss_type)
    if spec is None:
        raise ValueError(f"policy loss {algorithm.policy_loss_type!r} requires a runtime LossSpec")
    distillation = algorithm.get("distillation")
    if (
        distillation is not None
        and distillation.objective != DistillationObjectiveKind.SAMPLED_REVERSE_KL
        and distillation.reward_mode == DistillationRewardMode.REPLACE
    ):
        return False
    return spec.anchor is RatioAnchor.ROLLOUT or bool(off_policy_correction(algorithm).rules)


def validate_dpo(cfg: DictConfig) -> None:
    """Reject configurations that would break chosen/rejected pair adjacency or the frozen reference."""
    algorithm = cfg.trainer.algorithm
    beta = float(algorithm.dpo.beta)
    label_smoothing = float(algorithm.dpo.label_smoothing)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("trainer.algorithm.dpo.beta must be a positive finite number")
    if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 0.5:
        raise ValueError("trainer.algorithm.dpo.label_smoothing must be in [0, 0.5)")
    if LossReduction(algorithm.loss_reduction) is not LossReduction.PAIR_MEAN:
        raise ValueError("dpo requires trainer.algorithm.loss_reduction=pair_mean")
    if str(cfg.environment.env_class) != PREFERENCE_PAIR_ENV_CLASS:
        raise ValueError(f"dpo requires environment.env_class={PREFERENCE_PAIR_ENV_CLASS} (the static pair runner)")
    if algorithm.advantage_estimator != AdvantageEstimator.UNIFORM or algorithm.advantage_batch_normalize:
        raise ValueError("dpo ignores advantages; use advantage_estimator=uniform and advantage_batch_normalize=false")
    if (
        algorithm.use_kl_loss
        or algorithm.use_kl_in_reward
        or algorithm.think_token_weight != 1
        or algorithm.enable_token_reward_channel
        or algorithm.dynamic_sampling.type is not None
    ):
        raise ValueError(
            "dpo supplies its own beta-weighted reference term; disable use_kl_loss, use_kl_in_reward, "
            "think_token_weight, the token-reward channel, and reward-based dynamic sampling"
        )
    if cfg.generator.n_samples_per_prompt != 2:
        raise ValueError("dpo requires generator.n_samples_per_prompt=2 (chosen and rejected rows per prompt)")
    if cfg.trainer.trajectory_selector.type is not None:
        raise ValueError("dpo cannot drop half a pair; unset trainer.trajectory_selector")
    if cfg.trainer.step_wise_training:
        raise ValueError("dpo trains single-turn preference completions; step-wise training is not supported")
    if cfg.trainer.use_sample_packing:
        raise ValueError("dpo requires use_sample_packing=false; packing reorders rows and breaks pair adjacency")
    if cfg.trainer.critic.model.path:
        raise ValueError("dpo trains the policy against a frozen reference; a critic model is not supported")
    if cfg.trainer.get("update_ref_every_epoch", False) or any(
        callback.get("type") == "ref_model_update" for callback in (cfg.trainer.get("callbacks") or [])
    ):
        raise ValueError("dpo requires a frozen reference; reference-update callbacks must be disabled")
    if cfg.trainer.placement.colocate_all:
        raise ValueError("dpo launches no inference engines; use trainer.placement.colocate_all=false")
    if compile_distillation_plan_from_config(cfg) is not None:
        raise ValueError("dpo does not combine with distillation objectives")
    for role in (cfg.trainer.policy, cfg.trainer.ref):
        geometry = role.megatron_config
        if role.sequence_parallel_size != 1 or geometry.context_parallel_size != 1:
            raise ValueError("dpo requires sequence_parallel_size=1 and context_parallel_size=1 on policy and ref")
    if cfg.trainer.micro_train_batch_size_per_gpu % 2:
        raise ValueError("dpo requires an even trainer.micro_train_batch_size_per_gpu so pairs share a microbatch")

    def data_parallel_size(role: DictConfig, default_nodes: int, default_gpus: int) -> int:
        placement = cfg.trainer.placement
        nodes = placement.get("ref_num_nodes") or default_nodes
        gpus = placement.get("ref_num_gpus_per_node") or default_gpus
        if role is cfg.trainer.policy:
            nodes, gpus = default_nodes, default_gpus
        geometry = role.megatron_config
        return int(nodes * gpus) // (
            geometry.pipeline_model_parallel_size * geometry.context_parallel_size * geometry.tensor_model_parallel_size
        )

    policy_nodes = cfg.trainer.placement.policy_num_nodes
    policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node
    policy_dp = data_parallel_size(cfg.trainer.policy, policy_nodes, policy_gpus)
    ref_dp = data_parallel_size(cfg.trainer.ref, policy_nodes, policy_gpus)
    rows = cfg.trainer.train_batch_size * cfg.generator.n_samples_per_prompt
    if rows % (2 * math.lcm(policy_dp, ref_dp)):
        raise ValueError(
            "dpo requires each data-parallel rank to receive whole pairs: "
            f"{rows} training rows do not divide into pair-aligned shards of {math.lcm(policy_dp, ref_dp)} ranks"
        )


def validate_objective(cfg: DictConfig, *, loss_spec: LossSpec | None = None) -> None:
    """Reject objective settings that cannot affect the selected training rows correctly."""
    validate_ftpo(cfg)
    if cfg.trainer.algorithm.policy_loss_type == PolicyLossType.DPO:
        validate_dpo(cfg)
    limit = cfg.trainer.policy.max_consecutive_nonfinite_steps
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("trainer.policy.max_consecutive_nonfinite_steps must be null or an integer >= 1")
    algorithm = cfg.trainer.algorithm
    for key in ("use_abs_kl", "use_kl_estimator_k3"):
        if key in algorithm:
            raise ValueError(f"trainer.algorithm.{key} is unsupported; configure trainer.algorithm.kl_estimator_type")
    try:
        KLEstimator(algorithm.kl_estimator_type)
    except ValueError as error:
        raise ValueError(
            f"invalid kl_estimator_type: {algorithm.kl_estimator_type}; choose one of {list(KLEstimator)}"
        ) from error
    for key in ("use_tis", "tis_imp_ratio_cap"):
        if key in algorithm:
            raise ValueError(
                f"trainer.algorithm.{key} is unsupported; configure trainer.algorithm.off_policy_correction"
            )
    if (
        algorithm.dynamic_sampling.max_mean_reward is not None
        and algorithm.dynamic_sampling.type != DynamicSamplingType.FILTER
    ):
        raise ValueError("dynamic_sampling.max_mean_reward requires dynamic_sampling.type=filter")
    try:
        reduction = LossReduction(algorithm.loss_reduction)
    except ValueError as error:
        raise ValueError(
            f"invalid loss_reduction: {algorithm.loss_reduction}; choose one of {list(LossReduction)}"
        ) from error
    spec = loss_spec or BUILTIN_LOSS_SPECS.get(algorithm.policy_loss_type)
    correction = off_policy_correction(algorithm)
    plan = compile_distillation_plan_from_config(cfg)
    topk = plan is not None and plan.objective is not DistillationObjectiveKind.SAMPLED_REVERSE_KL
    policy_trains = not (topk and plan.reward_mode is DistillationRewardMode.REPLACE)
    if correction.rules:
        if not policy_trains:
            raise ValueError("off_policy_correction requires an active policy row; use none with top-K REPLACE")
        if spec is not None and spec.anchor is not RatioAnchor.OLD:
            raise ValueError("off_policy_correction requires an OLD-anchored policy loss; use none for this loss")
    if (
        cfg.trainer.rollout_buffer.max_staleness_steps > 0
        and policy_trains
        and spec is not None
        and spec.anchor is RatioAnchor.OLD
        and algorithm.off_policy_correction is None
    ):
        raise ValueError(
            "off-policy OLD-anchored training requires off_policy_correction; use 'none' to run without one"
        )
    loop_credit = float(
        OmegaConf.select(cfg, "generator.trajectory_reward_shaping.loop.advantage_penalty_per_token", default=0)
    )
    if spec is not None and spec.sequence_level:
        if reduction is not LossReduction.SEQUENCE_MEAN:
            raise ValueError(f"{algorithm.policy_loss_type} requires trainer.algorithm.loss_reduction=sequence_mean")
        if loop_credit > 0 or (plan is not None and not topk):
            raise ValueError(f"{algorithm.policy_loss_type} requires sequence-level advantages; use a token-level loss")
    if algorithm.think_token_weight != 1 and not algorithm.enable_token_reward_channel:
        raise ValueError("think_token_weight != 1 requires trainer.algorithm.enable_token_reward_channel=true")
    if plan is None:
        return
    if not topk and algorithm.policy_loss_type == PolicyLossType.SFT:
        raise ValueError(
            "sampled_reverse_kl requires a policy loss that consumes advantages; sft ignores teacher credit"
        )
    if topk:
        policy = cfg.trainer.policy
        if (
            cfg.trainer.use_sample_packing
            or policy.sequence_parallel_size != 1
            or policy.megatron_config.context_parallel_size != 1
            or policy.megatron_config.tensor_model_parallel_size != 1
        ):
            raise ValueError(
                "top-K teacher objectives require use_sample_packing=false, sequence_parallel_size=1, "
                "context_parallel_size=1, and tensor_model_parallel_size=1"
            )
        if reduction is LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL:
            raise ValueError("top-K teacher rows require token_mean, sequence_mean, or seq_mean_token_sum_norm")
        if plan.advantage_clip is not None:
            raise ValueError("advantage_clip requires sampled_reverse_kl chosen-token evidence")
    if plan.objective is DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE:
        widths = {teacher.top_k for teacher in plan.teachers}
        if widths != {cfg.generator.sampling_params.logprobs}:
            raise ValueError("student_topk_policy_surrogate requires sampling_params.logprobs matching teacher top_k")
        if reduction is not LossReduction.TOKEN_MEAN:
            raise ValueError("student_topk_policy_surrogate requires token_mean loss reduction")
        low, high, dual = float(algorithm.eps_clip_low), float(algorithm.eps_clip_high), float(algorithm.clip_ratio_c)
        if not all(math.isfinite(value) for value in (low, high, dual)) or not (
            0 <= low < 1 and high >= 0 and dual > 1
        ):
            raise ValueError(
                "student_topk_policy_surrogate requires 0 <= eps_clip_low < 1, eps_clip_high >= 0, clip_ratio_c > 1"
            )
    if plan.reward_mode is DistillationRewardMode.REPLACE:
        if spec is not None and not spec.advantage_linear:
            raise ValueError("distillation reward_mode=replace requires an advantage-linear policy loss")
        if algorithm.advantage_estimator != AdvantageEstimator.UNIFORM:
            raise ValueError("distillation reward_mode=replace requires advantage_estimator=uniform")
        if (
            algorithm.use_kl_in_reward
            or algorithm.advantage_batch_normalize
            or loop_credit > 0
            or algorithm.dynamic_sampling.type is not None
        ):
            raise ValueError(
                "distillation reward_mode=replace requires use_kl_in_reward=false, advantage_batch_normalize=false, "
                "loop.advantage_penalty_per_token=0 and dynamic_sampling.type=null"
            )
