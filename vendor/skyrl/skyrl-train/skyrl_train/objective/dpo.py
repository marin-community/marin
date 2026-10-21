"""Direct Preference Optimization over adjacent chosen/rejected row pairs.

The loss formula follows HuggingFace TRL's ``DPOTrainer`` "sigmoid" variant with the
label-smoothed "robust" generalization (trl/trainer/dpo_trainer.py, Apache-2.0), and
matches NVIDIA NeMo-Aligner's ``MegatronGPTDPOModel.loss_func`` as the Megatron-native
cross-check. Like ``gspo_policy_loss``, a per-pair nonlinearity cannot be a token sum,
so the loss returns per-token surrogate values whose gradient is exactly the DPO
gradient; the true loss value is reported through metrics (``dpo/loss``).

For pair ``i`` with chosen completion ``c`` and rejected completion ``r``:

    delta_i = sum_{t in c}(log pi - log pi_ref)_t - sum_{t in r}(log pi - log pi_ref)_t
    L_i = k * (-(1-eps) * log sigmoid(beta * delta_i) - eps * log sigmoid(-beta * delta_i))

with ``k = 1 / (1 - 2 * eps)`` (k = 1 for the default eps = 0). The exact per-token
gradient is ``dL_i/d(log pi_t) = w_i * s_t`` with the detached pair weight

    w_i = k * beta * ((1 - eps) - sigmoid(beta * delta_i))

and the row role ``s_t = +1`` on chosen tokens and ``-1`` on rejected tokens, so the
returned values are ``w_i.detach() * s_t * (-(log pi_t - log pi_ref_t.detach()))``.
``sum_t values / N_pairs`` therefore has the mean-DPO-loss gradient while the reduced
``policy_loss`` row reads ``-mean_i(w_i * delta_i)``, a GSPO-style surrogate.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOInputs:
    """Pairing evidence for the DPO loss: +1 chosen / -1 rejected row roles."""

    pair_roles: torch.Tensor


def _pair_logratios(log_probs, ref_log_probs, loss_mask, roles) -> torch.Tensor:
    """Return validated per-pair policy-minus-reference log-prob sums as (pairs, 2) rows."""
    batch_size = log_probs.shape[0]
    if batch_size % 2:
        raise ValueError(f"dpo pairs require an even microbatch, got {batch_size} rows")
    roles = roles.to(log_probs.dtype)
    expected = torch.tensor([1.0, -1.0], device=roles.device, dtype=roles.dtype).repeat(batch_size // 2)
    if not torch.equal(roles, expected):
        raise ValueError("dpo requires adjacent chosen/rejected pairs: pair_roles must alternate +1, -1 per row")
    mask = loss_mask.to(log_probs.dtype)
    pair_eligible = (mask.sum(-1) > 0).reshape(batch_size // 2, 2)
    if not bool(pair_eligible.all()):
        raise ValueError("every chosen and rejected row needs at least one eligible loss-mask token")
    logratios = ((log_probs.float() - ref_log_probs.float()) * mask).sum(dim=-1)
    return logratios.reshape(batch_size // 2, 2)


def dpo_pair_values(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    dpo: DPOInputs,
    beta: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict]:
    """Return per-token surrogate values with the exact DPO gradient plus detached metrics."""
    pair_logratios = _pair_logratios(log_probs, ref_log_probs, loss_mask, dpo.pair_roles)
    chosen, rejected = pair_logratios[:, 0], pair_logratios[:, 1]
    delta = chosen - rejected
    scale = 1.0 / (1.0 - 2.0 * label_smoothing)
    sigmoid = torch.sigmoid(beta * delta)
    weight = (scale * beta * ((1.0 - label_smoothing) - sigmoid)).detach()
    loss = scale * (
        -(1.0 - label_smoothing) * F.logsigmoid(beta * delta) - label_smoothing * F.logsigmoid(-beta * delta)
    )
    values = (
        weight.repeat_interleave(2, dim=0)[:, None]
        * dpo.pair_roles[:, None].to(weight.dtype)
        * (-(log_probs.float() - ref_log_probs.detach().float()))
    )
    chosen_reward = beta * chosen.detach()
    rejected_reward = beta * rejected.detach()
    metrics = {
        "dpo/loss": float(loss.mean()),
        "dpo/accuracy": float((delta > 0).float().mean()),
        "dpo/margin": float((chosen_reward - rejected_reward).mean()),
        "dpo/chosen_reward": float(chosen_reward.mean()),
        "dpo/rejected_reward": float(rejected_reward.mean()),
    }
    return values, metrics
