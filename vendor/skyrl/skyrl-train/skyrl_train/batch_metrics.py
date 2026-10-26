from collections.abc import Sequence

import torch
from omegaconf import DictConfig

from skyrl_train.rollouts.buffer import RowFacts

from skyrl_train.config.objective_spec import off_policy_correction
from skyrl_train.objective.correction import compute_correction
from skyrl_train.training_batch import TrainingInputBatch


def install_correction(batch: TrainingInputBatch, algorithm: DictConfig) -> dict[str, float]:
    """Install configured policy correction weights and return their diagnostics."""
    correction = off_policy_correction(algorithm)
    if not correction.rules:
        return {}
    rollout_logprobs = batch.get("rollout_logprobs")
    if rollout_logprobs is None:
        raise ValueError("off_policy_correction requires rollout_logprobs")
    result = compute_correction(batch["action_log_probs"], rollout_logprobs, batch["loss_mask"], correction)
    batch["correction_weights"] = result.weights
    return result.metrics


def token_staleness_metrics(batch: TrainingInputBatch, facts: Sequence[RowFacts]) -> dict[str, float]:
    """Report loss-bearing token staleness and compact policy-stamp coverage."""
    trained = batch["loss_mask"] > 0
    count = trained.sum().item()
    if not count:
        return {}
    charged = batch["rollout_staleness"].masked_fill(~trained, 0)
    spanning = [value for group in facts for value in group.spanning_policy_steps]
    return {
        "async/staleness_tokens_mean": charged.sum().item() / count,
        "async/staleness_tokens_max": charged.max().item(),
        "async/staleness_tokens_ratio": (charged != 0).sum().item() / count,
        "async/staleness_tokens_stamped": sum(sum(group.stamped_loss_tokens) for group in facts) / count,
        "async/staleness_tokens_spanning": torch.tensor(spanning).float().mean().item() if spanning else 0.0,
    }
