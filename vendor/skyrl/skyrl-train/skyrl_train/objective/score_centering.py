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
from typing import NamedTuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP
from skyrl_train.distillation import INVALID_TOPK_INDEX

TAIL_MASS_FLOOR = 1e-6
PROBABILITY_MASS_TOLERANCE = 1e-4


class ScoreCenteringTerms(NamedTuple):
    """Advantage-scaled additive loss and detached, unscaled head coefficients."""

    loss: torch.Tensor
    head_coefficients: torch.Tensor


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
) -> ScoreCenteringTerms:
    """Return the advantage-scaled loss and detached, unscaled head coefficients.

    On loss-eligible positions, the loss derivative with respect to head logprobs
    is ``advantages[..., None] * head_coefficients``. Coefficients already
    integrate TIS and PPO clipping. The caller owns policy weights and reduction.

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
    coefficients = residual.detach()
    correction = advantages * (coefficients * current).sum(dim=-1)
    return ScoreCenteringTerms(correction.masked_fill(~loss_mask.to(torch.bool), 0), coefficients)


def collate_score_centering(
    index_rows: list[np.ndarray],
    behavior_rows: list[np.ndarray],
    response_token_ids: list[list[int]],
    loss_masks: list[list[int]],
    top_k: int,
    *,
    sampled_logprobs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pad full response heads, mask loss holes and validate behavior probabilities.

    Rows share response order and shape [response, top_k]. Returned learner
    fields ``score_topk_indices`` and ``score_behavior_logprobs`` have shape
    [batch, response, top_k] and invalid-ID/NaN padding. Sampled scores must agree where the
    chosen token is in the head. Callers own evidence provenance.
    """
    size = len(response_token_ids)
    if top_k <= 0 or any(len(rows) != size for rows in (index_rows, behavior_rows, loss_masks)):
        raise ValueError("behavior top-k channels must align with response rows")
    for ids, scores, response, mask in zip(index_rows, behavior_rows, response_token_ids, loss_masks, strict=True):
        if ids.shape != (len(response), top_k) or scores.shape != ids.shape:
            raise ValueError("behavior top-k width must match the configured capture width")
        values = np.asarray(mask)
        if values.shape != (len(response),) or not np.isin(values, (0, 1)).all():
            raise ValueError("behavior loss masks must align with response tokens")
    index_dtype = (
        torch.int32 if all(np.can_cast(row.dtype, np.int32, casting="safe") for row in index_rows) else torch.long
    )
    indices = pad_sequence(
        [torch.as_tensor(row, dtype=index_dtype) for row in index_rows],
        batch_first=True,
        padding_value=INVALID_TOPK_INDEX,
    )
    behavior = pad_sequence(
        [torch.as_tensor(row, dtype=torch.float32) for row in behavior_rows], batch_first=True, padding_value=torch.nan
    )
    selected = pad_sequence([torch.tensor(row, dtype=torch.bool) for row in loss_masks], batch_first=True)
    if selected.shape != sampled_logprobs.shape:
        raise ValueError("sampled behavior logprobs must align with response rows")
    indices.masked_fill_(~selected.unsqueeze(-1), INVALID_TOPK_INDEX)
    behavior.masked_fill_(~selected.unsqueeze(-1), torch.nan)
    ids, scores = indices[selected], behavior[selected]
    ordered = torch.from_numpy(np.sort(ids.numpy(), axis=-1))
    if torch.any(ids < 0) or torch.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("trainable behavior top-k token IDs must be nonnegative and unique")
    if not torch.isfinite(scores).all() or torch.any(scores > 0):
        raise ValueError("trainable behavior top-k logprobs must be finite and nonpositive")
    if torch.any(scores.exp().sum(dim=-1) > 1 + PROBABILITY_MASS_TOLERANCE):
        raise ValueError("behavior top-k probabilities must be normalized over the full vocabulary")
    tokens = pad_sequence([torch.tensor(row, dtype=torch.long) for row in response_token_ids], batch_first=True)
    matched = ids == tokens[selected].unsqueeze(-1)
    expected = sampled_logprobs[selected].unsqueeze(-1).expand_as(matched)
    if not torch.allclose(scores[matched], expected[matched], rtol=0, atol=1e-4):
        raise ValueError("behavior top-k and sampled-token logprobs disagree for the same token IDs")
    return {"score_topk_indices": indices, "score_behavior_logprobs": behavior}
