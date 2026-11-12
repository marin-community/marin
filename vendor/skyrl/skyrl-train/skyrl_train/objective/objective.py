from dataclasses import dataclass

import torch

from marinskyrl.distillation import DistillationRewardMode
from omegaconf import DictConfig, OmegaConf

from skyrl_train.config.objective_spec import LossReduction, TopKLossParams
from skyrl_train.config.ftpo import FTPOConfig
from skyrl_train.ftpo import FTPOInputs, ftpo_loss
from skyrl_train.objective.dpo import DPOInputs
from skyrl_train.objective.losses import PolicyLoss, PolicyLossInputs, TokenLoss, complete_clip_metrics
from skyrl_train.objective.reduction import StepCounts, policy_data_weights, reduce_to_step
from skyrl_train.objective.teacher import TopKEvidence, mask_teacher_evidence, topk_teacher_loss
from skyrl_train.utils.policy_math import differentiable_approx_kl


@dataclass(frozen=True)
class TopKTeacherBatch:
    evidence: TopKEvidence
    student_log_probs_on_support: torch.Tensor
    params: TopKLossParams
    vocabulary_size: int


@dataclass(frozen=True)
class FTPOAuxiliaryBatch:
    """Structural supervision with counts and masks independent of task rewards."""

    inputs: FTPOInputs
    weights: torch.Tensor
    repair_log_probs: torch.Tensor
    params: FTPOConfig
    coefficient: float


@dataclass(frozen=True)
class ObjectiveMicroBatch:
    policy: PolicyLossInputs
    policy_data_weights: torch.Tensor
    ref_log_probs: torch.Tensor | None
    token_entropy: torch.Tensor
    teacher: TopKTeacherBatch | None
    correction_weights: torch.Tensor | None = None
    structural: FTPOAuxiliaryBatch | None = None


@dataclass(frozen=True)
class ObjectiveRows:
    policy: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor
    teacher: torch.Tensor | None
    structural: torch.Tensor | None = None


@dataclass(frozen=True)
class PolicyObjective:
    optimization_loss: torch.Tensor
    rows: ObjectiveRows
    metrics: dict[str, float]


def build_objective_micro_batch(
    *,
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    base_action_log_probs: torch.Tensor | None,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    rollout_logprobs: torch.Tensor | None,
    response_span_tags: torch.Tensor | None,
    token_entropy: torch.Tensor,
    think_token_weight: float,
    teacher: TopKTeacherBatch | None,
    ftpo: FTPOInputs | None = None,
    pair_roles: torch.Tensor | None = None,
    correction_weights: torch.Tensor | None = None,
    structural: FTPOAuxiliaryBatch | None = None,
) -> ObjectiveMicroBatch:
    """Prepare finite values at masked positions before objective formulas run."""
    valid = loss_mask > 0

    def sanitize(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == loss_mask.shape
        return torch.where(valid, value, 0)

    if teacher is not None:
        teacher = TopKTeacherBatch(
            mask_teacher_evidence(teacher.evidence, loss_mask),
            teacher.student_log_probs_on_support,
            teacher.params,
            teacher.vocabulary_size,
        )
    return ObjectiveMicroBatch(
        policy=PolicyLossInputs(
            log_probs=sanitize(action_log_probs),
            old_log_probs=sanitize(old_action_log_probs),
            rollout_log_probs=None if rollout_logprobs is None else sanitize(rollout_logprobs),
            advantages=sanitize(advantages),
            loss_mask=loss_mask,
            ftpo=ftpo,
            ref_log_probs=None if base_action_log_probs is None else sanitize(base_action_log_probs),
            dpo=None if pair_roles is None else DPOInputs(pair_roles),
        ),
        policy_data_weights=policy_data_weights(loss_mask, response_span_tags, think_token_weight),
        ref_log_probs=None if base_action_log_probs is None else sanitize(base_action_log_probs),
        token_entropy=sanitize(token_entropy),
        teacher=teacher,
        correction_weights=None if correction_weights is None else sanitize(correction_weights).detach(),
        structural=structural,
    )


def compute_policy_objective(
    batch: ObjectiveMicroBatch,
    *,
    loss: PolicyLoss,
    counts: StepCounts,
    config: DictConfig,
    loss_scale: float,
    report_scale: float,
) -> PolicyObjective:
    """Compose globally normalized rows with separate backward and reporting scales."""
    teacher_only = (
        batch.teacher is not None
        and OmegaConf.select(config, "distillation.reward_mode") == DistillationRewardMode.REPLACE
    )
    policy = TokenLoss(batch.policy.log_probs * 0, {}) if teacher_only else loss(batch.policy, config)
    assert policy.values.shape == batch.policy.log_probs.shape
    common = {"max_seq_len": counts.max_seq_len, "nonzero_advantage_rows": counts.nonzero_advantage_rows}
    mode = LossReduction(config.loss_reduction)
    policy_row = reduce_to_step(
        policy.values,
        batch.policy_data_weights,
        counts.policy,
        mode,
        numerator_weights=batch.correction_weights,
        **common,
    )
    mask = batch.policy.loss_mask
    entropy = reduce_to_step(batch.token_entropy, mask, counts.mask, LossReduction.TOKEN_MEAN, **common)
    if config.use_kl_loss:
        if batch.ref_log_probs is None:
            raise ValueError("base_action_log_probs are required when use_kl_loss is enabled")
        kl_values = differentiable_approx_kl(
            batch.policy.log_probs, batch.ref_log_probs, kl_estimator_type=config.kl_estimator_type
        )
        kl = reduce_to_step(kl_values, mask, counts.mask, LossReduction.SEQUENCE_MEAN, **common)
    else:
        kl = policy_row.new_zeros(())
    combined = policy_row + config.kl_loss_coef * kl
    if config.use_entropy_loss:
        combined = combined - config.entropy_loss_coef * entropy
    metrics = complete_clip_metrics(policy.metrics)
    structural_row = None
    if batch.structural is not None:
        structural = batch.structural
        if batch.policy.ftpo is not None:
            raise ValueError("FTPO cannot be both the policy objective and an auxiliary objective")
        values, structural_metrics = ftpo_loss(structural.inputs, structural.params)
        weights = structural.weights
        assert values.shape == weights.shape == structural.repair_log_probs.shape
        active = structural.inputs.targets.chosen_mask.any(-1)
        structural_row = (torch.where(active, values, 0) * weights).sum() / (structural.inputs.counts.examples or 1.0)
        repair_mask = structural.inputs.targets.repair_mask
        if repair_mask is not None:
            if structural.params.repair_sft_weight is None:
                raise ValueError("Structural repair supervision requires an explicit repair_sft_weight")
            repair = torch.where(repair_mask, -structural.repair_log_probs, 0) * weights
            structural_row = structural_row + structural.params.repair_sft_weight * repair.sum() / (
                structural.inputs.counts.repair_tokens or 1.0
            )
        combined = combined + structural.coefficient * structural_row
        metrics.update(structural_metrics)
        metrics["structural_loss"] = (structural_row.detach() * report_scale).item()
    teacher_row = None
    if batch.teacher is not None:
        teacher = batch.teacher
        result = topk_teacher_loss(
            teacher.evidence,
            teacher.student_log_probs_on_support,
            teacher.params,
            vocabulary_size=teacher.vocabulary_size,
        )
        assert result.values.shape == mask.shape
        teacher_row = reduce_to_step(
            result.values,
            mask * teacher.evidence.valid_mask,
            counts.teacher,
            mode,
            numerator_weights=teacher.evidence.loss_weights,
            **common,
        )
        combined = combined + teacher_row
        metrics.update(result.metrics)
        metrics["distillation_loss"] = (teacher_row.detach() * report_scale).item()
    return PolicyObjective(
        optimization_loss=combined * loss_scale,
        rows=ObjectiveRows(
            policy=policy_row * report_scale,
            kl=kl * report_scale,
            entropy=entropy * report_scale,
            teacher=None if teacher_row is None else teacher_row * report_scale,
            structural=None if structural_row is None else structural_row * report_scale,
        ),
        metrics=metrics,
    )


def megatron_loss_scale(num_microbatches: int, data_parallel_size: int) -> float:
    """Cancel the scheduler's microbatch division and the DP gradient average."""
    return float(num_microbatches * data_parallel_size)
