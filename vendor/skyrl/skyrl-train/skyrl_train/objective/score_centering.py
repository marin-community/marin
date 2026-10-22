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
from skyrl_train.utils.importance_ratio_diagnostics import LONG_RESPONSE_TOKENS, POSITION_WINDOW
from skyrl_train.distillation_adapters import collate_student_selected_rollout
from skyrl_train.trajectory_runners.types import TrajectoryBatch

TAIL_MASS_FLOOR = 1e-6
PROBABILITY_MASS_TOLERANCE = 1e-4


def _bounded_ratio(numerator_logprob: torch.Tensor, denominator_logprob: torch.Tensor) -> torch.Tensor:
    return (numerator_logprob - denominator_logprob).clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP).exp()


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
    ppo_ratio = _bounded_ratio(current, old)
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


@torch.no_grad()
def score_centering_diagnostics(
    current_head: torch.Tensor,
    old_head: torch.Tensor,
    behavior_head: torch.Tensor,
    current_sampled: torch.Tensor,
    behavior_sampled: torch.Tensor,
    correction: torch.Tensor | None,
    loss_mask: torch.Tensor,
    token_limit: int,
) -> dict[str, float]:
    """Summarize a bounded prefix of valid tokens in one microbatch.

    The correction is a scalar loss diagnostic, not a parameter-gradient change.
    Existing metric aggregation averages these microbatch samples across ranks.
    """
    selected = loss_mask.reshape(-1).nonzero().flatten()[:token_limit]
    metrics = {"score_centering/sampled_tokens": float(selected.numel())}

    def mean(values: torch.Tensor) -> float:
        return values.mean().item() if selected.numel() else 0.0

    for name, head in (("current", current_head), ("old", old_head), ("behavior", behavior_head)):
        mass = head.detach().reshape(-1, head.shape[-1])[selected].float().exp().sum(-1)
        metrics[f"score_centering/sampled_{name}_retained_mass"] = mean(mass)
        metrics[f"score_centering/sampled_{name}_omitted_mass"] = mean((1 - mass).clamp_min(0))
    delta = current_sampled.detach().reshape(-1)[selected] - behavior_sampled.detach().reshape(-1)[selected]
    metrics["score_centering/sampled_current_behavior_log_ratio_mean"] = mean(delta)
    metrics["score_centering/sampled_current_behavior_log_ratio_abs_mean"] = mean(delta.abs())
    values = torch.zeros_like(delta) if correction is None else correction.detach().reshape(-1)[selected]
    metrics["score_centering/sampled_loss_mean"] = mean(values)
    metrics["score_centering/sampled_loss_abs_mean"] = mean(values.abs())
    return metrics


def collate_score_centering(
    trajectory_batch: TrajectoryBatch,
    response_token_ids: list[list[int]],
    response_mask: torch.Tensor,
    top_k: int,
    *,
    sampled_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad aligned behavior evidence and check its full-vocabulary probabilities.

    The existing collator uses the trajectory loss masks and response_mask's
    shape. Check sampled-score agreement where the chosen token is in the head.
    The caller remains responsible for evidence provenance.
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


def collate_score_probe(
    trajectory_batch: TrajectoryBatch,
    response_token_ids: list[list[int]],
    top_k: int,
    *,
    sampled_logprobs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Collate compact probe positions without a response-length head canvas."""
    positions = trajectory_batch["score_probe_positions"]
    index_rows = trajectory_batch["score_probe_indices"]
    score_rows = trajectory_batch["score_probe_logprobs"]
    size = len(response_token_ids)
    if any(len(rows) != size for rows in (positions, index_rows, score_rows)):
        raise ValueError("score probe channels must align with response rows")
    width = max((len(row) for row in positions if row is not None), default=0)
    if width == 0:
        return {}
    compact_positions = torch.zeros((size, width), dtype=torch.long)
    sampled = torch.zeros((size, width), dtype=torch.float32)
    mask = torch.zeros((size, width), dtype=torch.bool)
    compact_responses, compact_indices, compact_scores, compact_masks = [], [], [], []
    for row, response in enumerate(response_token_ids):
        selected = (
            np.asarray(positions[row], dtype=np.int64) if positions[row] is not None else np.empty(0, dtype=np.int64)
        )
        count = len(selected)
        if count and (np.any(selected < 0) or np.any(selected >= len(response)) or len(np.unique(selected)) != count):
            raise ValueError("score probe positions must be unique response positions")
        if count and (index_rows[row] is None or score_rows[row] is None):
            raise ValueError("score probe positions require aligned behavior head evidence")
        # Trainer-side filters can mask tokens after the rollout worker sampled
        # positions. Drop their evidence without changing the optimization mask.
        keep = np.asarray(trajectory_batch["loss_masks"][row], dtype=np.bool_)[selected]
        selected = selected[keep]
        count = len(selected)
        compact_positions[row, :count] = torch.from_numpy(selected)
        sampled[row, :count] = sampled_logprobs[row, selected]
        mask[row, :count] = True
        compact_responses.append([response[index] for index in selected])
        compact_indices.append(index_rows[row][keep] if count else np.empty((0, top_k), dtype=np.int32))
        compact_scores.append(score_rows[row][keep] if count else np.empty((0, top_k), dtype=np.float32))
        compact_masks.append([1] * count)
    width = max(map(len, compact_responses))
    if width == 0:
        return {}
    compact_positions, sampled, mask = compact_positions[:, :width], sampled[:, :width], mask[:, :width]
    compact = {
        "student_topk_indices": compact_indices,
        "behavior_topk_logprobs": compact_scores,
        "loss_masks": compact_masks,
    }
    indices, behavior = collate_score_centering(compact, compact_responses, mask, top_k, sampled_logprobs=sampled)
    return {
        "score_topk_indices": indices,
        "score_behavior_logprobs": behavior,
        "score_probe_positions": compact_positions,
    }


@torch.no_grad()
def hypothetical_score_probe(
    head: torch.Tensor,
    behavior_head: torch.Tensor,
    sampled: torch.Tensor,
    behavior_sampled: torch.Tensor,
    advantages: torch.Tensor,
    valid: torch.Tensor,
    source_steps: torch.Tensor,
    *,
    response_positions: torch.Tensor,
    response_lengths: torch.Tensor,
    training_step: int,
    loss_tokens: int,
    tis_cap: float,
    eps_clip_low: float,
    eps_clip_high: float,
) -> dict[str, float]:
    """Pool a pre-update hypothetical correction; current and old weights match.

    Fresh old/behavior disagreement measures engine mismatch. A stale source
    mixes engine mismatch and weight drift; it cannot isolate either one.
    This scalar diagnostic does not measure a parameter-gradient change.
    """
    correction = ppo_tis_score_centering_correction(
        head,
        head,
        behavior_head,
        advantages,
        valid,
        tis_cap=tis_cap,
        eps_clip_low=eps_clip_low,
        eps_clip_high=eps_clip_high,
    )
    count = int(valid.sum())
    metrics = {
        "observed": float(count > 0),
        "sampled_tokens": float(count),
        "sampled_responses": float(valid.any(-1).sum()),
        "loss_tokens": float(loss_tokens),
        "coverage": count / loss_tokens if loss_tokens else 0.0,
        "training_step": float(training_step),
        "current_equals_old": 1.0,
    }
    if not count:
        return {f"score_centering_probe/{name}": value for name, value in metrics.items()}
    unit_advantage = ppo_tis_score_centering_correction(
        head,
        head,
        behavior_head,
        torch.ones_like(advantages),
        valid,
        tis_cap=tis_cap,
        eps_clip_low=eps_clip_low,
        eps_clip_high=eps_clip_high,
    )[valid]
    current_mass = head[valid].float().exp().sum(-1)
    behavior_mass = behavior_head[valid].float().exp().sum(-1)
    delta = (sampled - behavior_sampled)[valid]
    values = correction[valid]
    age = training_step - source_steps[valid]
    known = source_steps[valid] >= 0
    lengths = response_lengths.unsqueeze(-1).expand_as(valid)[valid]
    positions = response_positions[valid]
    long_response = lengths >= LONG_RESPONSE_TOKENS
    metrics.update(
        sampled_response_length_mean=float(lengths.float().mean()),
        sampled_response_length_max=float(lengths.max()),
        first256_tokens=float((positions < POSITION_WINDOW).sum()),
        last256_tokens=float((positions >= lengths - POSITION_WINDOW).sum()),
        long_response_tokens=float(long_response.sum()),
        long_responses=float((valid.any(-1) & (response_lengths >= LONG_RESPONSE_TOKENS)).sum()),
    )
    if torch.any(age[known] < 0):
        raise ValueError("score probe source version is ahead of the learner snapshot")
    ordinary = -advantages[valid] * delta.clamp(max=math.log(tis_cap)).exp()
    metrics.update(
        {
            "hypothetical_loss_mean": float(values.mean()),
            "hypothetical_loss_abs_mean": float(values.abs().mean()),
            "unit_advantage_loss_mean": float(unit_advantage.mean()),
            "unit_advantage_loss_abs_mean": float(unit_advantage.abs().mean()),
            "ordinary_loss_abs_mean": float(ordinary.abs().mean()),
            "current_retained_mass_mean": float(current_mass.mean()),
            "behavior_retained_mass_mean": float(behavior_mass.mean()),
            "current_omitted_mass_mean": float((1 - current_mass).clamp_min(0).mean()),
            "behavior_omitted_mass_mean": float((1 - behavior_mass).clamp_min(0).mean()),
            "old_behavior_log_ratio_mean": float(delta.mean()),
            "old_behavior_log_ratio_abs_mean": float(delta.abs().mean()),
            "sampled_tis_clipped_fraction": float((delta > math.log(tis_cap)).float().mean()),
            "zero_advantage_fraction": float((advantages[valid] == 0).float().mean()),
            "unknown_source_tokens": float((~known).sum()),
            "tail_floor_fraction": float(
                ((1 - current_mass <= TAIL_MASS_FLOOR) | (1 - behavior_mass <= TAIL_MASS_FLOOR)).float().mean()
            ),
        }
    )
    for name, selected in (("fresh", known & (age == 0)), ("stale", known & (age > 0)), ("unknown", ~known)):
        metrics[f"{name}_source_tokens"] = float(selected.sum())
        if selected.any():
            metrics[f"{name}_old_behavior_log_ratio_abs_mean"] = float(delta[selected].abs().mean())
            metrics[f"{name}_hypothetical_loss_abs_mean"] = float(values[selected].abs().mean())
        selected_long = selected & long_response
        metrics[f"long_response/{name}_tokens"] = float(selected_long.sum())
        if selected_long.any():
            metrics[f"long_response/{name}_old_behavior_log_ratio_abs_mean"] = float(delta[selected_long].abs().mean())
    if known.any():
        metrics["source_age_mean"] = float(age[known].float().mean())
        metrics["source_age_max"] = float(age[known].max())
        metrics["source_policy_step_min"] = float(source_steps[valid][known].min())
        metrics["source_policy_step_max"] = float(source_steps[valid][known].max())
    return {f"score_centering_probe/{name}": value for name, value in metrics.items()}
