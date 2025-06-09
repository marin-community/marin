"""
Core functions to implement PPO algorithms for LLM token-level training.
"""

from typing import Dict, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from flax import struct
from dataclasses import dataclass
import numpy as np
from enum import Enum

@struct.dataclass
class PPOParams:
    """PPO hyperparameters."""
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    kl_coef: float = 0.1
    target_kl: float = 0.1
    kl_horizon: int = 10000

class AdvantageEstimator(str, Enum):
    """Enumeration of advantage estimation methods."""
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"

@jit
def compute_gae_advantage_return(
    token_level_rewards: jnp.ndarray,
    values: jnp.ndarray,
    response_mask: jnp.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns for token-level rewards.
    
    Args:
        token_level_rewards: Array of shape (batch_size, response_length)
        values: Array of shape (batch_size, response_length)
        response_mask: Array of shape (batch_size, response_length)
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: Array of shape (batch_size, response_length)
        returns: Array of shape (batch_size, response_length)
    """
    def scan_fn(carry, x):
        next_value, last_gae_lam = carry
        reward, value, mask = x
        
        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * lam * last_gae_lam * mask
        next_value = value
        
        return (next_value, gae), gae
    
    # Reverse the sequence for scanning
    rewards_rev = jnp.flip(token_level_rewards, axis=1)
    values_rev = jnp.flip(values, axis=1)
    mask_rev = jnp.flip(response_mask, axis=1)
    
    # Initialize carry
    init_carry = (jnp.zeros_like(values[:, 0]), jnp.zeros_like(values[:, 0]))
    
    # Scan through the sequence
    _, advantages_rev = jax.lax.scan(
        scan_fn,
        init_carry,
        (rewards_rev, values_rev, mask_rev)
    )
    
    # Reverse back and compute returns
    advantages = jnp.flip(advantages_rev, axis=1)
    returns = advantages + values
    
    # Normalize advantages
    advantages = masked_whiten(advantages, response_mask)
    
    return advantages, returns

@jit
def masked_whiten(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute whitened values with masking.
    
    Args:
        values: Array of shape (batch_size, seq_length)
        mask: Array of shape (batch_size, seq_length)
        
    Returns:
        whitened_values: Array of shape (batch_size, seq_length)
    """
    masked_values = values * mask
    mean = jnp.sum(masked_values, axis=1, keepdims=True) / (jnp.sum(mask, axis=1, keepdims=True) + 1e-8)
    var = jnp.sum(mask * (values - mean) ** 2, axis=1, keepdims=True) / (jnp.sum(mask, axis=1, keepdims=True) + 1e-8)
    std = jnp.sqrt(var + 1e-8)
    return (values - mean) / std

@jit
def compute_policy_loss(
    log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    response_mask: jnp.ndarray,
    clip_epsilon: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute PPO policy loss for token-level training.
    
    Args:
        log_probs: Array of shape (batch_size, response_length)
        old_log_probs: Array of shape (batch_size, response_length)
        advantages: Array of shape (batch_size, response_length)
        response_mask: Array of shape (batch_size, response_length)
        clip_epsilon: PPO clip parameter
        loss_agg_mode: How to aggregate the loss ("token-mean" or "token-sum")
        
    Returns:
        loss: Scalar policy loss
        metrics: Dictionary containing metrics
    """
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    
    policy_loss = -jnp.minimum(
        ratio * advantages,
        clipped_ratio * advantages
    )
    
    if loss_agg_mode == "token-mean":
        policy_loss = jnp.sum(policy_loss * response_mask) / (jnp.sum(response_mask) + 1e-8)
    else:  # token-sum
        policy_loss = jnp.sum(policy_loss * response_mask)
    
    metrics = {
        'policy_loss': policy_loss,
        'ratio_mean': jnp.mean(ratio),
        'ratio_std': jnp.std(ratio),
        'clip_fraction': jnp.mean(jnp.abs(ratio - 1.0) > clip_epsilon)
    }
    
    return policy_loss, metrics

@jit
def compute_value_loss(
    values: jnp.ndarray,
    returns: jnp.ndarray,
    response_mask: jnp.ndarray,
    clip_epsilon: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute PPO value loss for token-level training.
    
    Args:
        values: Array of shape (batch_size, response_length)
        returns: Array of shape (batch_size, response_length)
        response_mask: Array of shape (batch_size, response_length)
        clip_epsilon: PPO clip parameter
        loss_agg_mode: How to aggregate the loss ("token-mean" or "token-sum")
        
    Returns:
        loss: Scalar value loss
        metrics: Dictionary containing metrics
    """
    value_pred_clipped = jnp.clip(
        values,
        returns - clip_epsilon,
        returns + clip_epsilon
    )
    
    value_loss = jnp.maximum(
        (values - returns) ** 2,
        (value_pred_clipped - returns) ** 2
    )
    
    if loss_agg_mode == "token-mean":
        value_loss = jnp.sum(value_loss * response_mask) / (jnp.sum(response_mask) + 1e-8)
    else:  # token-sum
        value_loss = jnp.sum(value_loss * response_mask)
    
    metrics = {
        'value_loss': value_loss,
        'value_mean': jnp.mean(values),
        'value_std': jnp.std(values)
    }
    
    return value_loss, metrics

@jit
def compute_entropy_loss(
    logits: jnp.ndarray,
    response_mask: jnp.ndarray,
    loss_agg_mode: str = "token-mean",
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute entropy loss for token-level training.
    
    Args:
        logits: Array of shape (batch_size, response_length, vocab_size)
        response_mask: Array of shape (batch_size, response_length)
        loss_agg_mode: How to aggregate the loss ("token-mean" or "token-sum")
        
    Returns:
        loss: Scalar entropy loss
        metrics: Dictionary containing metrics
    """
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    
    if loss_agg_mode == "token-mean":
        entropy_loss = -jnp.sum(entropy * response_mask) / (jnp.sum(response_mask) + 1e-8)
    else:  # token-sum
        entropy_loss = -jnp.sum(entropy * response_mask)
    
    metrics = {
        'entropy': jnp.mean(entropy),
        'entropy_loss': entropy_loss
    }
    
    return entropy_loss, metrics

@jit
def kl_penalty(
    log_probs: jnp.ndarray,
    ref_log_probs: jnp.ndarray,
    response_mask: jnp.ndarray,
    kl_coef: float,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute KL penalty between current and reference policy.
    
    Args:
        log_probs: Array of shape (batch_size, response_length)
        ref_log_probs: Array of shape (batch_size, response_length)
        response_mask: Array of shape (batch_size, response_length)
        kl_coef: KL penalty coefficient
        
    Returns:
        penalty: Scalar KL penalty
        metrics: Dictionary containing metrics
    """
    kl = log_probs - ref_log_probs
    kl = jnp.sum(kl * response_mask) / (jnp.sum(response_mask) + 1e-8)
    penalty = kl_coef * kl
    
    metrics = {
        'kl': kl,
        'kl_penalty': penalty
    }
    
    return penalty, metrics

def compute_ppo_loss(
    params: PPOParams,
    log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    values: jnp.ndarray,
    returns: jnp.ndarray,
    advantages: jnp.ndarray,
    response_mask: jnp.ndarray,
    ref_log_probs: Optional[jnp.ndarray] = None,
    loss_agg_mode: str = "token-mean",
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute total PPO loss for token-level training.
    
    Args:
        params: PPO hyperparameters
        log_probs: Array of shape (batch_size, response_length)
        old_log_probs: Array of shape (batch_size, response_length)
        values: Array of shape (batch_size, response_length)
        returns: Array of shape (batch_size, response_length)
        advantages: Array of shape (batch_size, response_length)
        response_mask: Array of shape (batch_size, response_length)
        ref_log_probs: Optional array of shape (batch_size, response_length)
        loss_agg_mode: How to aggregate the loss ("token-mean" or "token-sum")
        
    Returns:
        loss: Scalar total loss
        metrics: Dictionary containing metrics
    """
    policy_loss, policy_metrics = compute_policy_loss(
        log_probs, old_log_probs, advantages, response_mask,
        params.clip_epsilon, loss_agg_mode
    )
    
    value_loss, value_metrics = compute_value_loss(
        values, returns, response_mask,
        params.value_clip_epsilon, loss_agg_mode
    )
    
    entropy_loss, entropy_metrics = compute_entropy_loss(
        log_probs, response_mask, loss_agg_mode
    )
    
    total_loss = (
        policy_loss +
        params.value_coef * value_loss +
        params.entropy_coef * entropy_loss
    )
    
    metrics = {
        **policy_metrics,
        **value_metrics,
        **entropy_metrics,
        'total_loss': total_loss
    }
    
    if ref_log_probs is not None:
        kl_penalty_value, kl_metrics = kl_penalty(
            log_probs, ref_log_probs, response_mask, params.kl_coef
        )
        total_loss += kl_penalty_value
        metrics.update(kl_metrics)
    
    return total_loss, metrics 