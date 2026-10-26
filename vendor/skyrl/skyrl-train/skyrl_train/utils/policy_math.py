"""Shared tensor math for policy optimization.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from jaxtyping import Float
from omegaconf import DictConfig

from skyrl_train.config.objective_spec import KLEstimator
from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP, masked_mean
from skyrl_train.training_batch import TrainingInputBatch


def right_pad_to_match(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Right-truncate or zero-pad the last dimension to match a reference tensor."""
    if tensor.shape == reference.shape and (dtype is None or tensor.dtype == dtype):
        return tensor
    width = min(tensor.shape[-1], reference.shape[-1])
    aligned = torch.zeros_like(reference, dtype=dtype or tensor.dtype)
    aligned[..., :width] = tensor[..., :width]
    return aligned


def differentiable_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    *,
    kl_estimator_type: str,
) -> torch.Tensor:
    """Return per-token KL estimates with the selected value and gradient convention."""
    estimator = KLEstimator(kl_estimator_type)
    if estimator is KLEstimator.K1:
        kld = log_probs - log_probs_base
    elif estimator is KLEstimator.ABS:
        kld = (log_probs - log_probs_base).abs()
    elif estimator is KLEstimator.K2:
        kld = 0.5 * (log_probs - log_probs_base).square()
    else:
        log_ratio = (log_probs_base - log_probs).clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
        k3 = (log_ratio.exp() - log_ratio - 1).contiguous().clamp(-10, 10)
        if estimator is KLEstimator.K3_UNBIASED_GRADIENT:
            k2 = 0.5 * log_ratio.square()
            kld = k3.detach() + (k2 - k2.detach())
        else:
            kld = k3

    if loss_mask is not None:
        kld = kld * loss_mask
    return kld


@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    *,
    kl_estimator_type: str,
) -> torch.Tensor:
    """Compute approximate KL without gradients for metrics and reward shaping.

    Use ``differentiable_approx_kl`` for a differentiable KL regularization loss.
    """
    return differentiable_approx_kl(log_probs, log_probs_base, loss_mask=loss_mask, kl_estimator_type=kl_estimator_type)


@torch.no_grad()
def normalize_advantages_dict(data: TrainingInputBatch) -> TrainingInputBatch:
    """Normalizes the advantages in the data batch.

    Expects:
        - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
        - `["loss_mask"]`: Float[torch.Tensor, "batch_size seqlen"]
    """
    advantages: Float[torch.Tensor, "batch_size seqlen"] = data["advantages"]
    loss_mask = data["loss_mask"]
    valid = loss_mask > 0
    masked = torch.where(valid, advantages, 0)
    num_actions = loss_mask.sum().clamp(min=1)
    mean = (masked * loss_mask).sum() / num_actions
    centered = torch.where(valid, masked - mean, 0)
    variance = (centered.square() * loss_mask).sum() / num_actions
    data["advantages"] = centered * variance.clamp(min=1e-8).rsqrt()
    return data


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def ppo_critic_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[float]]:
    if config.value_clip is not None:
        values_clipped = old_values + (values - old_values).clamp(-config.value_clip, config.value_clip)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
        clipfrac = masked_mean((surr1 > surr2).float(), loss_mask).mean().detach().item()
    else:
        clipfrac = None
        loss = (values - returns) ** 2

    loss = masked_mean(loss, loss_mask, dim=-1).mean()
    return 0.5 * loss, clipfrac
