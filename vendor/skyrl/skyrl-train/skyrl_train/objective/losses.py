"""Per-token policy losses and clipping diagnostics.

Adapted from VERL's trainer/ppo/core_algos.py (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from dataclasses import asdict, dataclass, fields
from typing import Protocol

import torch
from omegaconf import DictConfig

from marinskyrl.runtime_options import PolicyLossType
from skyrl_train.ftpo import FTPOInputs, ftpo_loss
from skyrl_train.config.ftpo import ftpo_config
from skyrl_train.objective.dpo import DPOInputs, dpo_pair_values
from skyrl_train.tensor_math import masked_mean, safe_exp_delta
from skyrl_train.utils.algorithm_registry import register_policy_loss


@dataclass(frozen=True)
class PolicyLossInputs:
    log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    rollout_log_probs: torch.Tensor | None
    advantages: torch.Tensor
    loss_mask: torch.Tensor
    ftpo: FTPOInputs | None = None
    ref_log_probs: torch.Tensor | None = None
    dpo: DPOInputs | None = None


@dataclass(frozen=True)
class TokenLoss:
    values: torch.Tensor
    metrics: dict[str, float]


class PolicyLoss(Protocol):
    def __call__(self, inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss: ...


@dataclass(frozen=True)
class PolicyClipMetrics:
    ppo_clip_ratio: float = 0.0
    ppo_clip_ratio_low: float = 0.0
    ppo_clip_ratio_high: float = 0.0
    ppo_clip_pressure_low: float = 0.0
    ppo_clip_pressure_high: float = 0.0
    ppo_ratio_exact_unit_fraction: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


POLICY_CLIP_METRIC_KEYS = tuple(field.name for field in fields(PolicyClipMetrics))


def _masked_fraction(condition: torch.Tensor, loss_mask: torch.Tensor) -> float:
    return masked_mean(condition.float(), loss_mask).detach().item()


def clipping_metrics(
    ratio: torch.Tensor,
    selected: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps_clip_low: float,
    eps_clip_high: float,
    pooled_clip_ratio: float | None = None,
) -> dict[str, float]:
    """Report clipping decisions and pressure over trainable tokens."""
    low_pressure = ratio < 1 - eps_clip_low
    high_pressure = ratio > 1 + eps_clip_high
    return PolicyClipMetrics(
        ppo_clip_ratio=_masked_fraction(selected, loss_mask) if pooled_clip_ratio is None else pooled_clip_ratio,
        ppo_clip_ratio_low=_masked_fraction(selected & low_pressure, loss_mask),
        ppo_clip_ratio_high=_masked_fraction(selected & high_pressure, loss_mask),
        ppo_clip_pressure_low=_masked_fraction(low_pressure, loss_mask),
        ppo_clip_pressure_high=_masked_fraction(high_pressure, loss_mask),
        ppo_ratio_exact_unit_fraction=_masked_fraction(ratio == 1, loss_mask),
    ).as_dict()


def complete_clip_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return PolicyClipMetrics().as_dict() | metrics


def _token_loss(values: torch.Tensor, inputs: PolicyLossInputs, metrics: dict[str, float]) -> TokenLoss:
    return TokenLoss(torch.where(inputs.loss_mask > 0, values, 0), metrics)


def _ppo_terms(inputs: PolicyLossInputs, config: DictConfig) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = safe_exp_delta(inputs.log_probs - inputs.old_log_probs)
    unclipped = -ratio * inputs.advantages
    clipped = -ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * inputs.advantages
    return torch.maximum(unclipped, clipped), clipping_metrics(
        ratio,
        clipped > unclipped,
        inputs.loss_mask,
        eps_clip_low=config.eps_clip_low,
        eps_clip_high=config.eps_clip_high,
    )


@register_policy_loss(PolicyLossType.REGULAR)
def ppo_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Return the pessimistic clipped policy surrogate against the old policy."""
    values, metrics = _ppo_terms(inputs, config)
    return _token_loss(values, inputs, metrics)


@register_policy_loss(PolicyLossType.DUAL_CLIP)
def dual_clip_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Bound the clipped surrogate for negative advantages by the dual-clip ratio."""
    values, metrics = _ppo_terms(inputs, config)
    bound = torch.minimum(-inputs.advantages * config.clip_ratio_c, values)
    values = torch.where(inputs.advantages < 0, bound, values)
    return _token_loss(values, inputs, metrics)


@register_policy_loss(PolicyLossType.IMPORTANCE_SAMPLING)
def importance_sampling_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Return the unclipped importance-weighted advantage against the old policy."""
    ratio = safe_exp_delta(inputs.log_probs - inputs.old_log_probs)
    return _token_loss(-ratio * inputs.advantages, inputs, {})


@register_policy_loss(PolicyLossType.SFT)
def sft_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    return _token_loss(-inputs.log_probs, inputs, {})


@register_policy_loss(PolicyLossType.BEHAVIOR_CLIP)
def behavior_clipped_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Return pessimistic PPO clipping against the policy that generated each token."""
    if inputs.rollout_log_probs is None:
        raise ValueError("rollout_logprobs are required for behavior_clip policy loss")
    ratio = safe_exp_delta(inputs.log_probs - inputs.rollout_log_probs)
    unclipped = -inputs.advantages * ratio
    clipped = -inputs.advantages * ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high)
    values = torch.maximum(unclipped, clipped)
    bound = torch.sign(inputs.advantages) * config.clip_ratio_c * inputs.advantages
    values = torch.where(inputs.advantages < 0, torch.minimum(values, bound), values)
    return _token_loss(
        values,
        inputs,
        clipping_metrics(
            ratio,
            unclipped.detach() < clipped.detach(),
            inputs.loss_mask,
            eps_clip_low=config.eps_clip_low,
            eps_clip_high=config.eps_clip_high,
        ),
    )


@register_policy_loss(PolicyLossType.SAPO)
def sapo_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Return the sigmoid-gated advantage with sign-dependent SAPO temperature."""
    ratio = safe_exp_delta(inputs.log_probs - inputs.old_log_probs)
    tau = torch.where(inputs.advantages > 0, config.sapo.tau_pos, config.sapo.tau_neg)
    gate = torch.sigmoid(tau * (ratio - 1)) * (4 / tau)
    return _token_loss(-gate * inputs.advantages, inputs, {})


@register_policy_loss(PolicyLossType.GSPO)
def gspo_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Return GSPO-token clipping with a detached sequence ratio and token gradients."""
    sequence_delta = masked_mean(inputs.log_probs - inputs.old_log_probs, inputs.loss_mask, dim=-1).unsqueeze(-1)
    token_delta = inputs.log_probs - inputs.log_probs.detach() + sequence_delta.detach()
    ratio = token_delta.clamp(max=10).exp()
    unclipped = -ratio * inputs.advantages
    clipped = -ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * inputs.advantages
    return _token_loss(
        torch.maximum(unclipped, clipped),
        inputs,
        clipping_metrics(
            ratio,
            clipped > unclipped,
            inputs.loss_mask,
            eps_clip_low=config.eps_clip_low,
            eps_clip_high=config.eps_clip_high,
        ),
    )


@register_policy_loss(PolicyLossType.CISPO)
def compute_policy_loss_cispo(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Weight log-likelihood gradients by a detached, clipped importance ratio."""
    ratio = safe_exp_delta(inputs.log_probs - inputs.old_log_probs)
    low, high = config.cispo.cispo_eps_clip_low, config.cispo.cispo_eps_clip_high
    weights = ratio.clamp(1 - low, 1 + high).detach()
    return _token_loss(
        -inputs.advantages * weights * inputs.log_probs,
        inputs,
        clipping_metrics(
            ratio, (ratio < 1 - low) | (ratio > 1 + high), inputs.loss_mask, eps_clip_low=low, eps_clip_high=high
        ),
    )


@register_policy_loss(PolicyLossType.CLIP_COV)
def compute_policy_loss_clip_cov(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Suppress sampled covariance-band tokens in the PPO surrogate."""
    advantages, log_probs, loss_mask = inputs.advantages, inputs.log_probs, inputs.loss_mask
    ratio = safe_exp_delta(log_probs - inputs.old_log_probs)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high)
    selected = (clipped > unclipped) & (loss_mask > 0)
    covariance = (advantages - masked_mean(advantages, loss_mask)) * (
        log_probs - masked_mean(log_probs.detach(), loss_mask)
    )
    eligible = (
        (covariance < config.clip_cov.clip_cov_ub)
        & (covariance > config.clip_cov.clip_cov_lb)
        & (loss_mask > 0)
        & ~selected
    )
    candidates = torch.nonzero(eligible)
    clip_num = max(int(config.clip_cov.clip_ratio * loss_mask.sum().item()), 1)
    chosen = candidates[torch.randperm(len(candidates), device=candidates.device)[:clip_num]]
    retained = torch.ones_like(advantages)
    retained[chosen[:, 0], chosen[:, 1]] = 0
    return _token_loss(
        torch.maximum(unclipped, clipped) * retained,
        inputs,
        clipping_metrics(
            ratio,
            selected,
            loss_mask,
            eps_clip_low=config.eps_clip_low,
            eps_clip_high=config.eps_clip_high,
            pooled_clip_ratio=_masked_fraction(retained == 0, loss_mask),
        ),
    )


@register_policy_loss(PolicyLossType.KL_COV)
def compute_policy_loss_kl_cov(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Add the absolute log-ratio penalty on the largest covariance tokens."""
    advantages, log_probs = inputs.advantages, inputs.log_probs
    delta = log_probs - inputs.old_log_probs
    values = -advantages * safe_exp_delta(delta)
    valid = inputs.loss_mask > 0
    valid_indices = torch.nonzero(valid.flatten(), as_tuple=True)[0]
    if len(valid_indices):
        valid_advantages = advantages[valid].detach().cpu()
        valid_log_probs = log_probs[valid].detach().cpu()
        covariance = (valid_advantages - valid_advantages.mean()) * (valid_log_probs - valid_log_probs.mean())
        num_selected = min(max(1, int(len(valid_indices) * config.kl_cov.kl_cov_frac)), len(valid_indices))
        selected = valid_indices[torch.topk(covariance, num_selected).indices.to(valid_indices.device)]
        penalty_mask = torch.zeros_like(values).flatten()
        penalty_mask[selected] = 1
        values = values + penalty_mask.reshape_as(values) * config.kl_cov.ppo_kl_coef * delta.abs()
    return _token_loss(values, inputs, {})


@register_policy_loss(PolicyLossType.FTPO)
def ftpo_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Optimize selected token alternatives and optional verified repair suffixes."""
    params = ftpo_config(config)
    if inputs.ftpo is None or params is None:
        raise ValueError("FTPO requires candidate masks and raw reference logits")
    values, metrics = ftpo_loss(inputs.ftpo, params)
    repair_mask = inputs.ftpo.targets.repair_mask
    counts = inputs.ftpo.counts
    if counts.repair_tokens:
        if params.repair_sft_weight is None:
            raise ValueError("Joint closure training requires an explicit repair_sft_weight")
        # The outer reducer divides by the combined active weights. Undo that
        # denominator separately for each objective so repair length cannot
        # dilute the FTPO preference or its reference regularization.
        total_weight = counts.examples + counts.repair_tokens
        values = values * (total_weight / (counts.examples or 1.0))
        if repair_mask is not None:
            repair_values = torch.where(repair_mask, -inputs.log_probs, 0)
            values = values + params.repair_sft_weight * repair_values * total_weight / counts.repair_tokens
    return TokenLoss(values, metrics)


@register_policy_loss(PolicyLossType.DPO)
def dpo_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
    """Optimize adjacent preference pairs against the frozen reference model."""
    if inputs.dpo is None:
        raise ValueError("dpo requires pair_roles on every training row")
    if inputs.ref_log_probs is None:
        raise ValueError("dpo requires base_action_log_probs from a frozen reference model")
    values, metrics = dpo_pair_values(
        inputs.log_probs,
        inputs.ref_log_probs,
        inputs.loss_mask,
        inputs.dpo,
        config.dpo.beta,
        config.dpo.label_smoothing,
    )
    return _token_loss(values.to(inputs.log_probs.dtype), inputs, metrics)
