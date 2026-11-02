"""Teacher evidence contracts and backend-neutral distillation objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Self

import torch

from marinskyrl.distillation import TeacherEvidenceKind

INVALID_TOPK_INDEX = -1
RETAINED_MASS_ATOL = 1e-6
DISTILLATION_SCORED_TOKENS_METRIC = "distillation/scored_tokens"
DISTILLATION_TOPK_METRIC = "distillation_topk"


@dataclass(frozen=True)
class TeacherScoreRequest:
    """Exact rollout tokens and requested evidence coordinates for one teacher."""

    trajectory_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    teacher_id: str
    tokenizer_fingerprint: str
    plan_version: str
    prompt_token_ids: torch.Tensor
    prompt_mask: torch.Tensor
    response_token_ids: torch.Tensor
    response_mask: torch.Tensor
    evidence: TeacherEvidenceKind
    top_k: Optional[int] = None
    student_topk_indices: Optional[torch.Tensor] = None
    behavior_topk_logprobs: Optional[torch.Tensor] = None
    student_selected_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TeacherEvidence:
    """Provenance common to every teacher evidence representation."""

    trajectory_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    teacher_id: str
    teacher_revision: str
    plan_version: str
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ChosenTokenTeacherEvidence(TeacherEvidence):
    """Teacher probability of each emitted student token."""

    chosen_logprobs: torch.Tensor
    kind: ClassVar[TeacherEvidenceKind] = TeacherEvidenceKind.CHOSEN_TOKEN


@dataclass(frozen=True)
class TopKTeacherEvidence(TeacherEvidence):
    """Sparse teacher distributions with explicit retained probability mass."""

    topk_indices: torch.Tensor
    topk_logprobs: torch.Tensor
    retained_mass: torch.Tensor
    kind: ClassVar[TeacherEvidenceKind] = TeacherEvidenceKind.TOPK_DISTRIBUTION


@dataclass(frozen=True)
class StudentSelectedTeacherEvidence(TeacherEvidence):
    """Teacher scores on the exact student-selected token IDs in the request."""

    student_topk_indices: torch.Tensor
    teacher_on_student_logprobs: torch.Tensor
    kind: ClassVar[TeacherEvidenceKind] = TeacherEvidenceKind.STUDENT_SELECTED_TOPK


TeacherEvidenceBatch = ChosenTokenTeacherEvidence | TopKTeacherEvidence | StudentSelectedTeacherEvidence


@dataclass(frozen=True)
class ChosenTokenTeacherInput:
    """Driver payload for chosen-token teacher credit."""

    teacher_action_log_probs: torch.Tensor
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def training_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "teacher_action_log_probs": self.teacher_action_log_probs,
            "teacher_valid_mask": self.valid_mask,
            "distillation_loss_weights": self.loss_weights,
        }


@dataclass(frozen=True)
class TeacherTopKInput:
    """Teacher-support tensors for a top-K objective row."""

    teacher_topk_indices: torch.Tensor
    teacher_topk_logprobs: torch.Tensor
    retained_mass: torch.Tensor
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def to(self, device: torch.device) -> Self:
        return type(self)(
            teacher_topk_indices=self.teacher_topk_indices.to(device),
            teacher_topk_logprobs=self.teacher_topk_logprobs.to(device),
            retained_mass=self.retained_mass.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_weights=self.loss_weights.to(device),
        )

    def pin_memory(self) -> Self:
        return type(self)(
            teacher_topk_indices=self.teacher_topk_indices.pin_memory(),
            teacher_topk_logprobs=self.teacher_topk_logprobs.pin_memory(),
            retained_mass=self.retained_mass.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
            loss_weights=self.loss_weights.pin_memory(),
        )

    def student_token_ids(self) -> torch.Tensor:
        return self.teacher_topk_indices

    def training_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "teacher_topk_indices": self.teacher_topk_indices,
            "teacher_topk_logprobs": self.teacher_topk_logprobs,
            "teacher_retained_mass": self.retained_mass,
            "teacher_valid_mask": self.valid_mask,
            "distillation_loss_weights": self.loss_weights,
        }


@dataclass(frozen=True)
class StudentTopKInput:
    """Student-support tensors for a top-K objective row."""

    student_topk_indices: torch.Tensor
    behavior_topk_logprobs: torch.Tensor
    teacher_on_student_logprobs: torch.Tensor
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def to(self, device: torch.device) -> Self:
        return type(self)(
            student_topk_indices=self.student_topk_indices.to(device),
            behavior_topk_logprobs=self.behavior_topk_logprobs.to(device),
            teacher_on_student_logprobs=self.teacher_on_student_logprobs.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_weights=self.loss_weights.to(device),
        )

    def pin_memory(self) -> Self:
        return type(self)(
            student_topk_indices=self.student_topk_indices.pin_memory(),
            behavior_topk_logprobs=self.behavior_topk_logprobs.pin_memory(),
            teacher_on_student_logprobs=self.teacher_on_student_logprobs.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
            loss_weights=self.loss_weights.pin_memory(),
        )

    def student_token_ids(self) -> torch.Tensor:
        return self.student_topk_indices

    def training_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "student_topk_indices": self.student_topk_indices,
            "behavior_topk_logprobs": self.behavior_topk_logprobs,
            "teacher_on_student_logprobs": self.teacher_on_student_logprobs,
            "teacher_valid_mask": self.valid_mask,
            "distillation_loss_weights": self.loss_weights,
        }


TopKEvidence = TeacherTopKInput | StudentTopKInput
PreparedTeacherInput = ChosenTokenTeacherInput | TopKEvidence


def distillation_input_from_tensors(
    *,
    teacher_action_log_probs: Optional[torch.Tensor],
    teacher_topk_indices: Optional[torch.Tensor],
    teacher_topk_logprobs: Optional[torch.Tensor],
    teacher_retained_mass: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    loss_weights: Optional[torch.Tensor],
    student_topk_indices: Optional[torch.Tensor] = None,
    behavior_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_on_student_logprobs: Optional[torch.Tensor] = None,
) -> Optional[TopKEvidence]:
    """Build exactly one learner payload and reject partial or mixed variants."""
    if teacher_action_log_probs is not None:
        raise ValueError("chosen-token teacher tensors must be consumed on the driver before worker dispatch")
    topk_values = (teacher_topk_indices, teacher_topk_logprobs, teacher_retained_mass)
    topk_present = any(value is not None for value in topk_values)
    student_topk_values = (student_topk_indices, behavior_topk_logprobs, teacher_on_student_logprobs)
    student_topk_present = any(value is not None for value in student_topk_values)
    if not topk_present and not student_topk_present:
        if valid_mask is not None or loss_weights is not None:
            raise ValueError("distillation masks and weights require teacher evidence")
        return None
    if topk_present and student_topk_present:
        raise ValueError("distillation cannot mix objective evidence variants")
    if valid_mask is None or loss_weights is None:
        raise ValueError("distillation evidence requires valid_mask and loss_weights")
    if student_topk_present:
        if any(value is None for value in student_topk_values):
            raise ValueError("student-top-K OPD requires selected IDs, behavior logprobs, and teacher scores together")
        assert student_topk_indices is not None
        assert behavior_topk_logprobs is not None
        assert teacher_on_student_logprobs is not None
        return StudentTopKInput(
            student_topk_indices, behavior_topk_logprobs, teacher_on_student_logprobs, valid_mask, loss_weights
        )
    if any(value is None for value in topk_values):
        raise ValueError("teacher-support evidence requires top-K indices, logprobs, and retained_mass together")
    assert teacher_topk_indices is not None
    assert teacher_topk_logprobs is not None
    assert teacher_retained_mass is not None
    return TeacherTopKInput(
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_logprobs=teacher_topk_logprobs,
        retained_mass=teacher_retained_mass,
        valid_mask=valid_mask,
        loss_weights=loss_weights,
    )


def _require_nonempty(value: str, path: str) -> None:
    if not value.strip():
        raise ValueError(f"{path} must be a non-empty string")


def _validate_masked_logprobs(logprobs: torch.Tensor, valid_mask: torch.Tensor, label: str) -> None:
    if not torch.is_floating_point(logprobs):
        raise ValueError(f"{label} must have floating-point dtype")
    valid_values = logprobs[valid_mask]
    if not torch.all(torch.isfinite(valid_values)) or torch.any(valid_values > 0):
        raise ValueError(f"valid {label} must be finite and no greater than zero")
    if not torch.all(torch.isnan(logprobs[~valid_mask])):
        raise ValueError(f"invalid {label} must be NaN, not plausible scores")


def _validate_loss_weights(loss_weights: torch.Tensor, label: str) -> None:
    if not torch.is_floating_point(loss_weights):
        raise ValueError(f"{label} must have floating-point dtype")
    if not torch.all(torch.isfinite(loss_weights)) or torch.any(loss_weights < 0):
        raise ValueError(f"{label} must be finite and non-negative")


def _validate_selected_token_ids(indices: torch.Tensor, valid_mask: torch.Tensor, label: str) -> None:
    valid_indices = indices[valid_mask]
    if torch.any(valid_indices < 0):
        raise ValueError(f"valid {label} indices must be non-negative")
    if not torch.all(indices[~valid_mask] == INVALID_TOPK_INDEX):
        raise ValueError(f"invalid {label} indices must use the {INVALID_TOPK_INDEX} sentinel")
    if valid_indices.numel():
        sorted_indices = valid_indices.sort(dim=-1).values
        if torch.any(sorted_indices[..., 1:] == sorted_indices[..., :-1]):
            raise ValueError(f"valid {label} indices must be unique per token")


def _validate_topk_distribution(
    indices: torch.Tensor,
    logprobs: torch.Tensor,
    retained_mass: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if indices.shape != logprobs.shape or indices.ndim != 3:
        raise ValueError("teacher top-K indices and logprobs must have matching [batch, response_len, K] shapes")
    if indices.shape[2] == 0:
        raise ValueError("teacher top-K evidence must retain at least one token per valid position")
    if indices.shape[:2] != valid_mask.shape:
        raise ValueError("teacher top-K evidence must match valid_mask response coordinates")
    if retained_mass.shape != valid_mask.shape:
        raise ValueError("teacher retained_mass must match valid_mask")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher top-K indices must have integer dtype")
    _validate_masked_logprobs(logprobs, valid_mask, "teacher top-K logprobs")
    if not torch.is_floating_point(retained_mass):
        raise ValueError("teacher retained_mass must have floating-point dtype")
    valid_logprobs = logprobs[valid_mask]
    valid_mass = retained_mass[valid_mask]
    _validate_selected_token_ids(indices, valid_mask, "teacher top-K")
    if not torch.all(torch.isfinite(valid_mass)) or torch.any(
        (valid_mass <= 0) | (valid_mass > 1 + RETAINED_MASS_ATOL)
    ):
        raise ValueError("valid teacher retained_mass must lie in (0, 1]")
    if not torch.all(torch.isnan(retained_mass[~valid_mask])):
        raise ValueError("invalid teacher retained_mass must be NaN")
    observed_mass = valid_logprobs.float().exp().sum(dim=-1)
    if not torch.allclose(observed_mass, valid_mass.float(), rtol=1e-4, atol=RETAINED_MASS_ATOL):
        raise ValueError("teacher retained_mass must equal the probability mass of top-K logprobs")


def _validate_evidence_coordinates(
    evidence: TeacherEvidenceBatch,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
    route_ids: Optional[tuple[str, ...]] = None,
) -> None:
    if evidence.trajectory_ids != trajectory_ids:
        raise ValueError("teacher evidence trajectory_ids do not match the requested trajectories")
    if route_ids is not None:
        if evidence.route_ids != route_ids:
            raise ValueError("teacher evidence route_ids do not match the requested routes")
    elif len(evidence.route_ids) != len(trajectory_ids):
        raise ValueError("teacher evidence route_ids do not align with the requested trajectories")
    if response_mask.dtype is not torch.bool:
        raise ValueError("response_mask must have bool dtype")
    if evidence.valid_mask.shape != response_mask.shape:
        raise ValueError("teacher evidence valid_mask must match the requested response coordinates")
    if evidence.valid_mask.dtype is not torch.bool:
        raise ValueError("teacher evidence valid_mask must have bool dtype")
    if torch.any(evidence.valid_mask & ~response_mask):
        raise ValueError("teacher evidence cannot mark padded response positions valid")


def validate_teacher_score_request(request: TeacherScoreRequest) -> None:
    """Reject requests with ambiguous identities or token coordinates."""
    batch_size = len(request.trajectory_ids)
    if batch_size == 0:
        raise ValueError("teacher score request must contain at least one trajectory")
    if len(set(request.trajectory_ids)) != batch_size:
        raise ValueError("teacher score request trajectory_ids must be unique")
    if len(request.route_ids) != batch_size:
        raise ValueError("teacher score request route_ids must align with trajectory_ids")
    for trajectory_id in request.trajectory_ids:
        _require_nonempty(trajectory_id, "teacher score request trajectory_id")
    for route_id in request.route_ids:
        _require_nonempty(route_id, "teacher score request route_id")
    _require_nonempty(request.teacher_id, "teacher score request teacher_id")
    _require_nonempty(request.tokenizer_fingerprint, "teacher score request tokenizer_fingerprint")
    _require_nonempty(request.plan_version, "teacher score request plan_version")
    if request.prompt_token_ids.ndim != 2:
        raise ValueError("teacher score request prompt_token_ids must have shape [batch, prompt_len]")
    if request.prompt_mask.shape != request.prompt_token_ids.shape:
        raise ValueError("teacher score request prompt_mask must match prompt_token_ids")
    if request.prompt_mask.dtype is not torch.bool:
        raise ValueError("teacher score request prompt_mask must have bool dtype")
    if request.response_token_ids.ndim != 2:
        raise ValueError("teacher score request response_token_ids must have shape [batch, response_len]")
    if request.response_mask.shape != request.response_token_ids.shape:
        raise ValueError("teacher score request response_mask must match response_token_ids")
    if request.response_mask.dtype is not torch.bool:
        raise ValueError("teacher score request response_mask must have bool dtype")
    if request.prompt_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher score request prompt_token_ids must have integer dtype")
    if request.response_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher score request response_token_ids must have integer dtype")
    if torch.any(request.prompt_token_ids < 0) or torch.any(request.response_token_ids < 0):
        raise ValueError("teacher score request token IDs must be non-negative")
    if request.prompt_token_ids.shape[0] != batch_size or request.response_token_ids.shape[0] != batch_size:
        raise ValueError("teacher score request tensors must align with trajectory_ids")
    if request.evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK:
        if (
            request.student_topk_indices is None
            or request.behavior_topk_logprobs is None
            or request.student_selected_mask is None
        ):
            raise ValueError("student-selected teacher requests require indices, behavior logprobs, and a valid mask")
        if request.student_selected_mask.shape != request.response_mask.shape:
            raise ValueError("student-selected valid mask must match response coordinates")
        if request.student_selected_mask.dtype is not torch.bool:
            raise ValueError("student-selected valid mask must have bool dtype")
        if torch.any(request.student_selected_mask & ~request.response_mask):
            raise ValueError("student-selected valid mask cannot include padded response positions")
        if not torch.any(request.student_selected_mask):
            raise ValueError("student-selected requests require at least one valid training token")
        if (
            request.student_topk_indices.ndim != 3
            or request.student_topk_indices.shape[:2] != request.response_mask.shape
        ):
            raise ValueError("student-selected indices must have [batch, response_len, K] coordinates")
        if request.student_topk_indices.shape[-1] == 0:
            raise ValueError("student-selected requests require at least one token per valid position")
        if request.behavior_topk_logprobs.shape != request.student_topk_indices.shape:
            raise ValueError("student-selected behavior logprobs must match selected indices")
        if isinstance(request.top_k, bool) or request.top_k != request.student_topk_indices.shape[-1]:
            raise ValueError("student-selected top_k must match selected indices")
        if request.student_topk_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("student-selected indices must have integer dtype")
        _validate_selected_token_ids(request.student_topk_indices, request.student_selected_mask, "student-selected")
        _validate_masked_logprobs(
            request.behavior_topk_logprobs, request.student_selected_mask, "behavior top-K logprobs"
        )
    elif any(
        value is not None
        for value in (request.student_topk_indices, request.behavior_topk_logprobs, request.student_selected_mask)
    ):
        raise ValueError("student-selected fields require student-selected teacher evidence")
    if request.evidence is TeacherEvidenceKind.TOPK_DISTRIBUTION:
        if isinstance(request.top_k, bool) or not isinstance(request.top_k, int) or request.top_k <= 0:
            raise ValueError("top-K teacher score requests require a positive top_k")
    elif request.evidence is not TeacherEvidenceKind.STUDENT_SELECTED_TOPK and request.top_k is not None:
        raise ValueError("top_k is only valid for top-K teacher score requests")


def validate_teacher_evidence(request: TeacherScoreRequest, evidence: TeacherEvidenceBatch) -> None:
    """Reject evidence whose identity, token coordinates, shape, or values are ambiguous."""
    validate_teacher_score_request(request)
    if not isinstance(evidence, (ChosenTokenTeacherEvidence, TopKTeacherEvidence, StudentSelectedTeacherEvidence)):
        raise TypeError(f"unsupported teacher evidence type: {type(evidence).__name__}")
    _validate_evidence_coordinates(
        evidence,
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        response_mask=request.response_mask,
    )
    if evidence.teacher_id != request.teacher_id:
        raise ValueError("teacher evidence teacher_id does not match the score request")
    if evidence.kind is not request.evidence:
        raise ValueError("teacher evidence kind does not match the score request")
    _require_nonempty(evidence.teacher_revision, "teacher evidence teacher_revision")
    _require_nonempty(evidence.plan_version, "teacher evidence plan_version")
    if evidence.plan_version != request.plan_version:
        raise ValueError("teacher evidence plan_version does not match the score request")

    if isinstance(evidence, ChosenTokenTeacherEvidence):
        if evidence.chosen_logprobs.shape != evidence.valid_mask.shape:
            raise ValueError("teacher chosen_logprobs must match valid_mask")
        _validate_masked_logprobs(evidence.chosen_logprobs, evidence.valid_mask, "teacher chosen_logprobs")
        return

    if isinstance(evidence, StudentSelectedTeacherEvidence):
        assert request.student_topk_indices is not None
        assert request.student_selected_mask is not None
        if not torch.equal(evidence.valid_mask, request.student_selected_mask):
            raise ValueError("teacher evidence valid mask must match the selected-score request mask")
        if not torch.equal(evidence.student_topk_indices, request.student_topk_indices):
            raise ValueError("teacher evidence must identify the exact student-selected token IDs")
        if evidence.teacher_on_student_logprobs.shape != request.student_topk_indices.shape:
            raise ValueError("teacher scores must match student-selected token coordinates")
        _validate_masked_logprobs(
            evidence.teacher_on_student_logprobs, evidence.valid_mask, "teacher-on-student logprobs"
        )
        return

    _validate_topk_distribution(
        evidence.topk_indices,
        evidence.topk_logprobs,
        evidence.retained_mass,
        evidence.valid_mask,
    )
    if evidence.topk_indices.shape[-1] != request.top_k:
        raise ValueError(f"teacher evidence top-K width must match requested top_k={request.top_k}")


def _weighted_route_weights(
    *, coefficient: float, route_weights: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("distillation coefficient must be a positive finite number")
    if route_weights.shape != valid_mask.shape:
        raise ValueError("distillation route_weights must match teacher response coordinates")
    _validate_loss_weights(route_weights, "distillation route_weights")
    return route_weights * coefficient


def prepare_sampled_reverse_kl(
    request: TeacherScoreRequest,
    evidence: ChosenTokenTeacherEvidence,
    *,
    coefficient: float,
    route_weights: torch.Tensor,
) -> ChosenTokenTeacherInput:
    """Validate chosen-token evidence and compile its minimal learner payload."""
    validate_teacher_evidence(request, evidence)
    return ChosenTokenTeacherInput(
        teacher_action_log_probs=evidence.chosen_logprobs,
        valid_mask=evidence.valid_mask,
        loss_weights=_weighted_route_weights(
            coefficient=coefficient, route_weights=route_weights, valid_mask=evidence.valid_mask
        ),
    )


def prepare_teacher_topk(
    request: TeacherScoreRequest,
    evidence: TopKTeacherEvidence,
    *,
    coefficient: float,
    route_weights: torch.Tensor,
) -> TeacherTopKInput:
    """Validate top-K evidence and compile its sparse learner payload."""
    validate_teacher_evidence(request, evidence)
    return TeacherTopKInput(
        teacher_topk_indices=evidence.topk_indices,
        teacher_topk_logprobs=evidence.topk_logprobs,
        retained_mass=evidence.retained_mass,
        valid_mask=evidence.valid_mask,
        loss_weights=_weighted_route_weights(
            coefficient=coefficient, route_weights=route_weights, valid_mask=evidence.valid_mask
        ),
    )


def prepare_student_topk_policy_surrogate(
    request: TeacherScoreRequest,
    evidence: StudentSelectedTeacherEvidence,
    *,
    coefficient: float,
    route_weights: torch.Tensor,
) -> StudentTopKInput:
    """Compile exact student-selected evidence into the shared learner payload."""
    validate_teacher_evidence(request, evidence)
    assert request.student_topk_indices is not None
    assert request.behavior_topk_logprobs is not None
    return StudentTopKInput(
        student_topk_indices=request.student_topk_indices,
        behavior_topk_logprobs=request.behavior_topk_logprobs,
        teacher_on_student_logprobs=evidence.teacher_on_student_logprobs,
        valid_mask=evidence.valid_mask,
        loss_weights=_weighted_route_weights(
            coefficient=coefficient, route_weights=route_weights, valid_mask=evidence.valid_mask
        ),
    )


def _validate_attachment_common(
    evidence: TeacherEvidenceBatch,
    *,
    valid_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
) -> None:
    _validate_evidence_coordinates(evidence, trajectory_ids=trajectory_ids, response_mask=response_mask)
    if not torch.equal(valid_mask, evidence.valid_mask):
        raise ValueError("prepared distillation valid_mask does not match teacher evidence")
    if loss_weights.shape != response_mask.shape:
        raise ValueError("prepared distillation loss_weights must match trajectory response coordinates")
    _validate_loss_weights(loss_weights, "prepared distillation loss_weights")


def validate_sampled_reverse_kl_attachment(
    evidence: ChosenTokenTeacherEvidence,
    distillation: ChosenTokenTeacherInput,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
) -> None:
    """Validate prepared evidence at the trajectory-to-learner boundary."""
    _validate_attachment_common(
        evidence,
        valid_mask=distillation.valid_mask,
        loss_weights=distillation.loss_weights,
        trajectory_ids=trajectory_ids,
        response_mask=response_mask,
    )
    if not torch.allclose(
        distillation.teacher_action_log_probs,
        evidence.chosen_logprobs,
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise ValueError("prepared distillation logprobs do not match teacher evidence")


def validate_student_selected_attachment(
    evidence: StudentSelectedTeacherEvidence,
    distillation: StudentTopKInput,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
) -> None:
    """Require exact selected-ID and score provenance at learner admission."""
    _validate_attachment_common(
        evidence,
        valid_mask=distillation.valid_mask,
        loss_weights=distillation.loss_weights,
        trajectory_ids=trajectory_ids,
        response_mask=response_mask,
    )
    if not torch.equal(distillation.student_topk_indices, evidence.student_topk_indices):
        raise ValueError("prepared student-selected token IDs do not match teacher evidence")
    if distillation.student_topk_indices.shape[:2] != response_mask.shape:
        raise ValueError("prepared student-selected token IDs do not match response coordinates")
    _validate_selected_token_ids(distillation.student_topk_indices, distillation.valid_mask, "student-selected")
    if distillation.behavior_topk_logprobs.shape != distillation.student_topk_indices.shape:
        raise ValueError("prepared student-selected behavior scores do not match selected token IDs")
    _validate_masked_logprobs(
        distillation.behavior_topk_logprobs, distillation.valid_mask, "prepared behavior top-K logprobs"
    )
    if not torch.allclose(
        distillation.teacher_on_student_logprobs,
        evidence.teacher_on_student_logprobs,
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise ValueError("prepared student-selected teacher scores do not match teacher evidence")


def validate_distillation_attachment(
    evidence: TeacherEvidenceBatch,
    distillation: PreparedTeacherInput,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
) -> None:
    """Validate either evidence representation at the trajectory-to-learner boundary."""
    if isinstance(evidence, ChosenTokenTeacherEvidence) and isinstance(distillation, ChosenTokenTeacherInput):
        validate_sampled_reverse_kl_attachment(
            evidence,
            distillation,
            trajectory_ids=trajectory_ids,
            response_mask=response_mask,
        )
        return
    if isinstance(evidence, StudentSelectedTeacherEvidence) and isinstance(distillation, StudentTopKInput):
        validate_student_selected_attachment(
            evidence, distillation, trajectory_ids=trajectory_ids, response_mask=response_mask
        )
        return
    if not isinstance(evidence, TopKTeacherEvidence) or not isinstance(distillation, TeacherTopKInput):
        raise ValueError("teacher evidence and prepared distillation objective kinds must match")
    _validate_attachment_common(
        evidence,
        valid_mask=distillation.valid_mask,
        loss_weights=distillation.loss_weights,
        trajectory_ids=trajectory_ids,
        response_mask=response_mask,
    )
    payload_pairs = (
        ("indices", distillation.teacher_topk_indices, evidence.topk_indices),
        ("logprobs", distillation.teacher_topk_logprobs, evidence.topk_logprobs),
        ("retained_mass", distillation.retained_mass, evidence.retained_mass),
    )
    for label, actual, expected in payload_pairs:
        if not torch.allclose(actual, expected, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"prepared distillation {label} does not match teacher evidence")


def student_topk_logprobs(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Return selected student logprobs, with NaN wherever the token ID is the invalid sentinel."""
    if logits.ndim != 3 or topk_indices.ndim != 3 or logits.shape[:2] != topk_indices.shape[:2]:
        raise ValueError("student logits and selected indices must have [batch, response_len, vocab/K] shapes")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected top-K indices must have integer dtype")
    if topk_indices.device != logits.device:
        raise ValueError("selected top-K indices must be on the student logits device")
    valid_indices = topk_indices != INVALID_TOPK_INDEX
    if torch.any(topk_indices < INVALID_TOPK_INDEX):
        raise ValueError(f"selected top-K indices cannot be smaller than the {INVALID_TOPK_INDEX} sentinel")
    if torch.any(topk_indices[valid_indices] >= logits.shape[-1]):
        raise ValueError(f"selected top-K indices must be smaller than student vocabulary size {logits.shape[-1]}")
    safe_indices = topk_indices.masked_fill(~valid_indices, 0).long()
    float_logits = logits.float()
    selected = torch.gather(float_logits, dim=-1, index=safe_indices)
    normalized = selected - torch.logsumexp(float_logits, dim=-1, keepdim=True)
    return normalized.masked_fill(~valid_indices, torch.nan)
