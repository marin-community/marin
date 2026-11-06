"""Compact evidence and pre-update summaries for hypothetical score centering."""

import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from skyrl_train.objective.score_centering import (
    collate_score_centering,
    ppo_tis_score_centering_correction,
    TAIL_MASS_FLOOR,
)
from skyrl_train.trajectory_runners.types import TrajectoryBatch
from skyrl_train.utils.importance_ratio_diagnostics import LONG_RESPONSE_TOKENS, POSITION_WINDOW


def collate_score_probe(
    trajectory_batch: TrajectoryBatch,
    response_token_ids: list[list[int]],
    top_k: int,
    *,
    sampled_logprobs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select surviving probe positions, then collate their compact behavior heads.

    Positions index full responses. The returned head fields share the dense
    collation contract and add ``score_probe_positions`` [batch, sampled].
    Output pads only the selected positions, with invalid head IDs distinguishing
    padding from response position zero. The caller supplies old-policy scores
    and source steps aligned to the returned full-response positions.
    """
    positions = trajectory_batch["score_probe_positions"]
    index_rows = trajectory_batch["score_probe_indices"]
    score_rows = trajectory_batch["score_probe_logprobs"]
    if any(len(rows) != len(response_token_ids) for rows in (positions, index_rows, score_rows)):
        raise ValueError("score probe channels must align with response rows")
    position_rows, token_rows, compact_indices, compact_scores = [], [], [], []
    for row, response in enumerate(response_token_ids):
        selected = np.asarray(positions[row] if positions[row] is not None else [], dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= len(response)) or len(np.unique(selected)) != len(selected):
            raise ValueError("score probe positions must be unique response positions")
        if len(selected) and (index_rows[row] is None or score_rows[row] is None):
            raise ValueError("score probe positions require aligned behavior head evidence")
        keep = np.asarray(trajectory_batch["loss_masks"][row], dtype=np.bool_)[selected]
        selected = selected[keep]
        position_rows.append(torch.from_numpy(selected))
        token_rows.append(np.asarray(response, dtype=np.int64)[selected].tolist())
        compact_indices.append(index_rows[row][keep] if len(selected) else np.empty((0, top_k), dtype=np.int32))
        compact_scores.append(score_rows[row][keep] if len(selected) else np.empty((0, top_k), dtype=np.float32))
    if not any(len(row) for row in position_rows):
        return {}
    padded_positions = pad_sequence(position_rows, batch_first=True)
    tensors = collate_score_centering(
        compact_indices,
        compact_scores,
        token_rows,
        [[1] * len(row) for row in token_rows],
        top_k,
        sampled_logprobs=sampled_logprobs.gather(1, padded_positions).float(),
    )
    tensors["score_probe_positions"] = padded_positions
    return tensors


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

    PPO clipping is inactive for either advantage sign at this snapshot, so
    scale one unit-advantage correction by the finalized advantages.
    Fresh old/behavior disagreement measures engine mismatch; stale sources
    also contain weight drift. The logit-gradient L1 metrics describe the
    hypothetical correction, including the modeled tail. They do not measure
    parameter gradients or learning benefit.
    """
    terms = ppo_tis_score_centering_correction(
        head,
        head,
        behavior_head,
        torch.ones_like(advantages),
        valid,
        tis_cap=tis_cap,
        eps_clip_low=eps_clip_low,
        eps_clip_high=eps_clip_high,
    )
    unit_correction = terms.loss
    residual = terms.head_coefficients[valid]
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
    unit_advantage = unit_correction[valid]
    current_mass = head[valid].float().exp().sum(-1)
    behavior_mass = behavior_head[valid].float().exp().sum(-1)
    delta = (sampled - behavior_sampled)[valid]
    values = advantages[valid] * unit_advantage
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
    p = head[valid].to(unit_advantage.dtype).exp()
    total = residual.sum(-1)
    unit_logit_l1 = (residual - p * total[:, None]).abs().sum(-1)
    unit_logit_l1 += (1 - p.sum(-1)).clamp_min(0) * total.abs()
    logit_l1 = advantages[valid].abs() * unit_logit_l1
    ordinary_logit_l1 = 2 * ordinary.abs() * (1 - sampled[valid].exp())
    metrics.update(
        {
            "hypothetical_loss_mean": float(values.mean()),
            "hypothetical_loss_abs_mean": float(values.abs().mean()),
            "unit_advantage_loss_mean": float(unit_advantage.mean()),
            "unit_advantage_loss_abs_mean": float(unit_advantage.abs().mean()),
            "ordinary_loss_abs_mean": float(ordinary.abs().mean()),
            "unit_advantage_logit_correction_l1_mean": float(unit_logit_l1.mean()),
            "logit_correction_l1_mean": float(logit_l1.mean()),
            "ordinary_logit_gradient_l1_mean": float(ordinary_logit_l1.mean()),
            "current_omitted_mass_mean": float((1 - current_mass).clamp_min(0).mean()),
            "behavior_omitted_mass_mean": float((1 - behavior_mass).clamp_min(0).mean()),
            "old_behavior_log_ratio_mean": float(delta.mean()),
            "old_behavior_log_ratio_abs_mean": float(delta.abs().mean()),
            "sampled_tis_clipped_fraction": float((delta > math.log(tis_cap)).float().mean()),
            "zero_advantage_fraction": float((advantages[valid] == 0).float().mean()),
            "tail_floor_fraction": float(
                ((1 - current_mass <= TAIL_MASS_FLOOR) | (1 - behavior_mass <= TAIL_MASS_FLOOR)).float().mean()
            ),
        }
    )
    if ordinary_logit_l1.sum() > 0:
        metrics["logit_correction_l1_ratio"] = float(logit_l1.sum() / ordinary_logit_l1.sum())
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
