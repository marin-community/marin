"""Score centering for the PPO objective with truncated behavior importance weights.

The current trainer policy is ``p``, the stored policy before training the
batch is ``o``, and the policy that sampled each token is ``q``.
For a sampled token, PPO contributes ``A * min(o/q, cap) * (p/o) * score(p)``
while its directional clip is inactive, and zero while it is active. The
correction subtracts the expectation of this same score coefficient under q.
All probabilities and clipping decisions in the correction are detached.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP
from skyrl_train.distillation_adapters import collate_student_selected_rollout
from skyrl_train.trajectory_runners.types import TrajectoryBatch

TAIL_MASS_FLOOR = 1e-6
PROBABILITY_MASS_TOLERANCE = 1e-4


def ppo_tis_score_centering_correction(
    current_topk_logprobs: torch.Tensor,
    old_topk_logprobs: torch.Tensor,
    behavior_topk_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tis_cap: float,
    eps_clip_low: float,
    eps_clip_high: float,
) -> torch.Tensor:
    """Return the per-token term to add before the PPO loss reduction.

    The head contains exact full-vocabulary logprobs for the same top-k token
    IDs under all three policies. Outside the head, q and o are modeled as
    rescaled copies of p, each preserving its measured tail mass. This keeps
    their ratios and PPO clip decision constant over the modeled tail.
    """
    shape = current_topk_logprobs.shape
    if (
        len(shape) != 3
        or shape[-1] == 0
        or old_topk_logprobs.shape != shape
        or behavior_topk_logprobs.shape != shape
        or advantages.shape != shape[:2]
        or loss_mask.shape != shape[:2]
    ):
        raise ValueError("score centering requires aligned [batch, response, top-k] logprobs and advantages")
    if tis_cap <= 0 or not 0 <= eps_clip_low < 1 or eps_clip_high < 0:
        raise ValueError("invalid score-centering TIS or PPO clipping parameter")

    # Invalid sentinel rows are allowed only where the policy loss is masked.
    # Replace them before any exp or multiply so NaN * 0 cannot poison a batch.
    valid = loss_mask.to(torch.bool).unsqueeze(-1)
    compute_dtype = torch.float64 if current_topk_logprobs.dtype == torch.float64 else torch.float32
    masked_logprob = -math.log(shape[-1] + 1)
    current = current_topk_logprobs.to(compute_dtype).masked_fill(~valid, masked_logprob)
    old = old_topk_logprobs.to(compute_dtype).masked_fill(~valid, masked_logprob)
    behavior = behavior_topk_logprobs.to(compute_dtype).masked_fill(~valid, masked_logprob)
    if not all(torch.isfinite(tensor).all() for tensor in (current, old, behavior)):
        raise ValueError("score centering requires finite head logprobs on every supplied position")

    current_mass = current.exp()
    old_mass = old.exp()
    behavior_mass = behavior.exp()
    if torch.any(current_mass.sum(dim=-1) > 1 + PROBABILITY_MASS_TOLERANCE) or torch.any(
        old_mass.sum(dim=-1) > 1 + PROBABILITY_MASS_TOLERANCE
    ):
        raise ValueError("trainer top-k probabilities exceed full-vocabulary mass")
    if torch.any(behavior_mass.sum(dim=-1) > 1 + PROBABILITY_MASS_TOLERANCE):
        raise ValueError("behavior top-k probabilities exceed full-vocabulary mass")

    p_tail = (1 - current_mass.sum(dim=-1)).clamp_min(TAIL_MASS_FLOOR)
    o_tail = (1 - old_mass.sum(dim=-1)).clamp_min(TAIL_MASS_FLOOR)
    q_tail = (1 - behavior_mass.sum(dim=-1)).clamp_min(TAIL_MASS_FLOOR)

    ppo_delta = current - old
    ppo_ratio = ppo_delta.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP).exp()
    # Match compute_correction's upper truncation without introducing a lower
    # ratio floor. PPO's bounded exponential has zero gradient outside its bounds.
    tis_weight = (old - behavior).clamp(max=math.log(tis_cap)).exp()
    positive = advantages.unsqueeze(-1) >= 0
    active = torch.where(positive, ppo_ratio <= 1 + eps_clip_high, ppo_ratio >= 1 - eps_clip_low)
    active &= ppo_delta.abs() <= LOG_PROB_DELTA_CLIP
    head_coefficient = behavior_mass * tis_weight * ppo_ratio * active

    # On the modeled tail, old_v / p_v = o_tail / p_tail and
    # q_v / p_v = q_tail / p_tail. The resulting weighted score coefficient
    # can therefore be collapsed with sum_v p_v * score_v = 0.
    behavior_over_p = q_tail / p_tail
    tail_ppo_ratio = p_tail / o_tail
    tail_tis_weight = (o_tail / q_tail).clamp(max=tis_cap)
    tail_active = torch.where(
        advantages >= 0,
        tail_ppo_ratio <= 1 + eps_clip_high,
        tail_ppo_ratio >= 1 - eps_clip_low,
    )
    tail_coefficient = behavior_over_p * tail_tis_weight * tail_ppo_ratio * tail_active
    residual = head_coefficient - tail_coefficient.unsqueeze(-1) * current_mass
    correction = advantages * (residual.detach() * current).sum(dim=-1)
    return correction.masked_fill(~loss_mask.to(torch.bool), 0)


def collate_score_centering(
    trajectory_batch: TrajectoryBatch,
    response_token_ids: list[list[int]],
    response_mask: torch.Tensor,
    top_k: int,
    *,
    sampled_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad aligned behavior evidence and check its full-vocabulary probabilities.

    Trajectory loss masks select evidence; response_mask sets the padded shape.
    Sampled scores must agree where the chosen token is in the head. The caller
    remains responsible for evidence provenance.
    """
    if sampled_logprobs.shape != response_mask.shape:
        raise ValueError("sampled behavior logprobs must align with response rows")
    for field in ("student_topk_indices", "behavior_topk_logprobs"):
        rows = trajectory_batch.get(field)
        if rows is None or len(rows) != len(response_token_ids):
            raise ValueError("behavior top-k evidence must align with response rows")
        if any(row.shape != (len(response), top_k) for row, response in zip(rows, response_token_ids, strict=True)):
            raise ValueError("behavior top-k width must match the configured capture width")
    index_rows = trajectory_batch["student_topk_indices"]
    index_dtype = (
        torch.int32 if all(np.can_cast(row.dtype, np.int32, casting="safe") for row in index_rows) else torch.long
    )
    indices, behavior, selected = collate_student_selected_rollout(
        trajectory_batch, response_token_ids, response_mask, top_k, index_dtype=index_dtype
    )
    ids, scores = indices[selected], behavior[selected]
    # This CPU check needs no per-entry int64 sort permutation.
    ordered = torch.from_numpy(np.sort(ids.numpy(), axis=-1))
    if torch.any(ids < 0) or torch.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("trainable behavior top-k token IDs must be nonnegative and unique")
    if not torch.isfinite(scores).all() or torch.any(scores > 0):
        raise ValueError("trainable behavior top-k logprobs must be finite and nonpositive")
    if torch.any(scores.exp().sum(dim=-1) > 1 + PROBABILITY_MASS_TOLERANCE):
        raise ValueError("behavior top-k probabilities must be normalized over the full vocabulary")
    padded_response = torch.zeros(response_mask.shape, dtype=torch.long)
    for row, response in enumerate(response_token_ids):
        padded_response[row, : len(response)] = torch.tensor(response, dtype=torch.long)
    matched = ids == padded_response[selected].unsqueeze(-1)
    expected = sampled_logprobs[selected].unsqueeze(-1).expand_as(matched)
    if not torch.allclose(scores[matched], expected[matched], rtol=0, atol=1e-4):
        raise ValueError("behavior top-k and sampled-token logprobs disagree for the same token IDs")
    return indices, behavior
