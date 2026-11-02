from dataclasses import replace
import math

import torch

from marinskyrl.distillation import DistillationObjectiveKind
from skyrl_train.config.objective_spec import TopKLossParams
from skyrl_train.distillation import DISTILLATION_TOPK_METRIC, TeacherTopKInput, TopKEvidence
from skyrl_train.objective.losses import TokenLoss
from skyrl_train.tensor_math import safe_exp_delta


# A partial-vocabulary teacher tail has probability at least 1e-12, bounding its negative log at 27.6311.
TEACHER_TAIL_MASS_FLOOR = 1e-12


@torch.no_grad()
def teacher_advantages(
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    route_weights: torch.Tensor,
    clip: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return detached teacher credit; route weights include the teacher coefficient."""
    assert teacher_log_probs.shape == old_log_probs.shape == valid_mask.shape == route_weights.shape
    valid = valid_mask.bool()
    teacher = torch.where(valid, teacher_log_probs, 0)
    old = torch.where(valid, old_log_probs, 0)
    weights = torch.where(valid, route_weights, 0)
    delta = teacher - old
    clipped = delta if clip is None else delta.clamp(-clip, clip)
    advantage = clipped * weights
    count = valid.sum()
    denominator = count.clamp(min=1)
    return advantage, {
        "distillation/teacher_advantage_mean": (advantage.sum() / denominator).item(),
        "distillation/teacher_advantage_abs_mean": (advantage.abs().sum() / denominator).item(),
        "distillation/teacher_advantage_clipped_fraction": ((clipped != delta).sum() / denominator).item(),
        "distillation/valid_tokens": count.item(),
    }


def mask_teacher_evidence(evidence: TopKEvidence, loss_mask: torch.Tensor) -> TopKEvidence:
    """Use response eligibility as well as teacher validity for the objective row."""
    assert evidence.valid_mask.shape == loss_mask.shape
    return replace(evidence, valid_mask=evidence.valid_mask & (loss_mask > 0))


def _relative_entropy(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    positive = probability > 0
    log_probability = torch.where(positive, probability, 1).log()
    log_reference = torch.where(positive, reference, 1).log()
    return torch.where(positive, probability * (log_probability - log_reference), 0)


def topk_teacher_loss(
    evidence: TopKEvidence, student_log_probs_on_support: torch.Tensor, params: TopKLossParams, *, vocabulary_size: int
) -> TokenLoss:
    """Return teacher values per response token, before route weighting and reduction."""
    valid = evidence.valid_mask
    assert student_log_probs_on_support.shape[:-1] == valid.shape
    # Invalid transport positions contain NaN and require sanitization before nonlinear operations.
    selected = valid.unsqueeze(-1)
    current = torch.where(selected, student_log_probs_on_support, 0)
    denominator = valid.sum().clamp(min=1)
    metrics = {DISTILLATION_TOPK_METRIC: float(current.shape[-1])}
    if isinstance(evidence, TeacherTopKInput):
        assert evidence.teacher_topk_logprobs.shape == current.shape
        teacher = torch.where(selected, evidence.teacher_topk_logprobs, 0).float()
        mass = torch.where(valid, evidence.retained_mass, 1).float()
        if params.objective is DistillationObjectiveKind.SPARSE_FORWARD_KL:
            conditional = teacher - mass.log().unsqueeze(-1)
            entries = conditional.exp() * (conditional - current.float())
            if params.entry_clip is not None:
                entries = entries.clamp(max=params.entry_clip)
            values = entries.sum(-1)
        elif params.objective in {DistillationObjectiveKind.SPARSE_REVERSE_KL, DistillationObjectiveKind.SPARSE_JSD}:
            student_support = current.float().exp()
            if current.shape[-1] == vocabulary_size:
                student_tail = torch.zeros_like(student_support[..., :1])
                teacher_tail = torch.zeros_like(mass.unsqueeze(-1))
            else:
                student_tail = (1 - student_support.sum(-1, keepdim=True)).clamp(min=0)
                log_mass = torch.where(selected, evidence.teacher_topk_logprobs, 0).double().logsumexp(-1, keepdim=True)
                log_mass = log_mass.clamp(max=math.log1p(-TEACHER_TAIL_MASS_FLOOR))
                log_tail = torch.where(
                    log_mass < -math.log(2),
                    torch.log1p(-log_mass.exp()),
                    torch.log(-torch.expm1(log_mass)),
                )
                teacher_tail = log_tail.exp().to(student_support.dtype)
            student_bins = torch.cat((student_support, student_tail), dim=-1)
            teacher_bins = torch.cat((teacher.exp(), teacher_tail), dim=-1)
            if params.objective is DistillationObjectiveKind.SPARSE_REVERSE_KL:
                positive = student_support > 0
                student_logprobs = torch.where(positive, current.float(), 0)
                teacher_logprobs = torch.where(positive, teacher, 0)
                support = student_support * (student_logprobs - teacher_logprobs)
                values = support.sum(-1) + _relative_entropy(student_tail, teacher_tail).sum(-1)
            else:
                assert params.jsd_beta is not None
                beta = params.jsd_beta
                mixture = beta * teacher_bins + (1 - beta) * student_bins
                values = (
                    beta * _relative_entropy(teacher_bins, mixture)
                    + (1 - beta) * _relative_entropy(student_bins, mixture)
                ).sum(-1)
        else:
            raise ValueError(f"teacher-support evidence cannot use {params.objective}")
        metrics.update(
            distillation_retained_mass_mean=(torch.where(valid, mass, 0).sum() / denominator).item(),
            distillation_retained_mass_min=mass[valid].min().item() if valid.any() else 0.0,
        )
    else:
        if params.objective is not DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE:
            raise ValueError(f"student-support evidence cannot use {params.objective}")
        assert evidence.behavior_topk_logprobs.shape == evidence.teacher_on_student_logprobs.shape == current.shape
        behavior = torch.where(selected, evidence.behavior_topk_logprobs, 0)
        teacher = torch.where(selected, evidence.teacher_on_student_logprobs, 0)
        student_weights = current.detach().float().softmax(dim=-1).to(current.dtype)
        advantages = -(current.detach() - teacher) * student_weights
        ratio = safe_exp_delta(current - behavior, out_dtype=current.dtype)
        raw = -advantages * ratio
        clipped = -advantages * ratio.clamp(1 - params.eps_clip_low, 1 + params.eps_clip_high)
        pessimistic = torch.maximum(raw, clipped)
        bound = -advantages * params.clip_ratio_c
        values = torch.where(advantages < 0, torch.minimum(pessimistic, bound), pessimistic).sum(-1)
        entry_count = denominator * current.shape[-1]
        metrics.update(
            distillation_student_retained_mass_mean=(
                torch.where(valid, behavior.exp().sum(-1), 0).sum() / denominator
            ).item(),
            distillation_clip_fraction=((clipped > raw) & selected).sum().div(entry_count).item(),
            distillation_dual_clip_fraction=(
                ((advantages < 0) & (pessimistic > bound) & selected).sum().div(entry_count).item()
            ),
        )
    return TokenLoss(torch.where(valid, values, 0), metrics)
