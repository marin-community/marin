"""Batch-level domain gradient allocation for student-selected OPD."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from marinskyrl.distillation import DomainGradientBalanceSpec
from skyrl_train.distillation import PreparedTeacherInput, StudentTopKInput

MIN_GAP_DRIFT = 0.05
MAX_GAP_DRIFT = 20.0
DOMAIN_METRIC_PREFIX = "distillation/domain"


class DomainGradientBalancer:
    """Balance response-token shares and track first-observed teacher-gap scales."""

    def __init__(self, spec: DomainGradientBalanceSpec) -> None:
        self.spec = spec
        self._anchor: dict[str, float] = {}

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"anchor": dict(self._anchor)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"anchor"} or not isinstance(state["anchor"], Mapping):
            raise ValueError("domain gradient balance state requires an anchor mapping")
        anchor = state["anchor"]
        expected = {domain for domain, _ in self.spec.target_shares}
        if set(anchor) - expected:
            raise ValueError("domain gradient balance anchor has unknown domains")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in anchor.values()
        ):
            raise ValueError("domain gradient balance anchors must be finite non-negative numbers")
        self._anchor = {str(domain): float(value) for domain, value in anchor.items()}

    def apply(
        self, evidence: PreparedTeacherInput, route_ids: Sequence[str]
    ) -> tuple[StudentTopKInput, dict[str, float]]:
        if not isinstance(evidence, StudentTopKInput):
            raise ValueError("domain gradient balance requires student-selected top-K evidence")
        mask = evidence.valid_mask
        if mask.ndim != 2 or evidence.loss_weights.shape != mask.shape:
            raise ValueError("domain gradient balance requires aligned response masks and weights")
        if len(route_ids) > mask.shape[0] or mask[len(route_ids) :].any():
            raise ValueError("domain gradient balance routes must cover every non-padding row")
        configured = dict(self.spec.target_shares)
        if set(route_ids) - configured.keys():
            raise ValueError("domain gradient balance encountered an unconfigured route")
        if not mask.any():
            raise ValueError("domain gradient balance requires valid response tokens")

        # The released only_stu reward is the student-weighted, signed teacher
        # gap summed over selected IDs, then absolute-valued for the M2 scale.
        rewards = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        behavior = evidence.behavior_topk_logprobs[mask].float()
        teacher = evidence.teacher_on_student_logprobs[mask].float()
        rewards[mask] = (-(behavior - teacher) * behavior.softmax(dim=-1)).sum(dim=-1)

        counts = mask.sum(dim=-1).float()
        total_tokens = counts.sum().item()
        present = sorted({route for route, count in zip(route_ids, counts, strict=False) if count > 0})
        target_total = sum(configured[route] for route in present)
        row_scale = torch.ones(mask.shape[0], dtype=torch.float32, device=mask.device)
        metrics: dict[str, float] = {}
        for route in present:
            indices = torch.tensor(
                [index for index, label in enumerate(route_ids) if label == route], device=mask.device
            )
            domain_tokens = counts[indices].sum().item()
            magnitude = (rewards[indices].abs() * mask[indices]).sum().item() / domain_tokens
            anchor = self._anchor.setdefault(route, magnitude)
            drift = (max(magnitude, 1e-12) / max(anchor, 1e-12)) ** self.spec.gap_scale_alpha
            drift = min(max(drift, MIN_GAP_DRIFT), MAX_GAP_DRIFT)
            scale = (configured[route] / target_total) / (domain_tokens / total_tokens) * drift
            row_scale[indices] = scale
            metrics[f"{DOMAIN_METRIC_PREFIX}/{route}/token_share"] = domain_tokens / total_tokens
            metrics[f"{DOMAIN_METRIC_PREFIX}/{route}/gap_abs_mean"] = magnitude
        normalizer = total_tokens / (row_scale * counts).sum().item()
        row_scale *= normalizer
        for route in present:
            index = route_ids.index(route)
            metrics[f"{DOMAIN_METRIC_PREFIX}/{route}/loss_weight"] = row_scale[index].item()
        loss_weights = evidence.loss_weights * row_scale.unsqueeze(-1)
        return replace(evidence, loss_weights=loss_weights), metrics
