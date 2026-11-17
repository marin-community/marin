"""Turn closure-session evidence into FTPO alternatives and masked suffix repairs."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from skyrl_train.closure_repair import (
    ClosureTarget,
    ClosureTargetKind,
    verified_closure_candidates,
    verified_closure_target,
    verified_continuation_boundary,
)
from verifyit.closure_scoring import closure_state, score_closure
from skyrl_gym.closure_task import SNOWBALL_TURN_END_MARKERS, ClosurePrompt
from skyrl_gym.verification import VerificationResult, VerificationStatus

from skyrl_train.ftpo import VerifiedFTPOBoundary, repeated_tool_opener_boundary
from skyrl_train.config.ftpo import FTPOBoundary


@dataclass(frozen=True)
class ClosureTrainingTargets:
    boundaries: list[VerifiedFTPOBoundary | None]
    repairs: list[ClosureTarget | None]


def mixed_boundary_modes(
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    masks: Sequence[Sequence[int]],
    verifications: Sequence[VerificationResult | None],
    decode: Callable[[list[int]], str],
) -> list[FTPOBoundary]:
    """Honor closure contracts before detecting repeated openers or tail loops."""
    modes = []
    for prompt, response, mask, verification in zip(prompts, responses, masks, verifications, strict=True):
        if verification is not None and "closure" in verification.diagnostics:
            mode = FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION
        elif repeated_tool_opener_boundary(prompt, response, mask, decode) is not None:
            mode = FTPOBoundary.REPEATED_TOOL_OPENER
        else:
            mode = FTPOBoundary.TAIL_LOOP
        modes.append(mode)
    return modes


def boundary_route_metrics(
    modes: Sequence[FTPOBoundary],
    eligible: Sequence[bool],
    chosen: torch.Tensor,
    weights: torch.Tensor,
    repair_mask: torch.Tensor | None,
) -> dict[str, float]:
    """Count prepared structural supervision, including zero counts for absent routes."""
    metrics = {}
    for mode in (
        FTPOBoundary.TAIL_LOOP,
        FTPOBoundary.REPEATED_TOOL_OPENER,
        FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION,
    ):
        rows = [row for row, routed in enumerate(modes) if routed == mode]
        prefix = f"ftpo/routes/{mode.value}"
        metrics.update(
            {
                f"{prefix}/rows": float(len(rows)),
                f"{prefix}/eligible_rows": float(sum(eligible[row] for row in rows)),
                f"{prefix}/selected_boundaries": float(chosen[rows].any(-1).sum()),
                f"{prefix}/target_weight": float(weights[rows].sum()),
                f"{prefix}/repair_tokens": 0.0 if repair_mask is None else float(repair_mask[rows].sum()),
            }
        )
    return metrics


def closure_target_metrics(
    verifications: Sequence[VerificationResult | None],
    targets: ClosureTrainingTargets,
    chosen: torch.Tensor,
    loss_mask: torch.Tensor,
    repair_mask: torch.Tensor | None,
) -> dict[str, float]:
    """Count prepared supervision by format family and control, before learner consumption."""
    selected = chosen.any(-1) & (loss_mask > 0)
    if repair_mask is not None:
        selected = selected & ~repair_mask
    metrics: dict[str, float] = {}
    for row, (verification, boundary) in enumerate(zip(verifications, targets.boundaries, strict=True)):
        if verification is None or verification.status != VerificationStatus.VERIFIED:
            continue
        if "closure" not in verification.diagnostics:
            continue
        contract = verification.diagnostics["closure"]
        prefix = f"closure/targets/{contract['family']}/{contract['control']}"
        counts = {
            "rows": 1,
            "verified_boundaries": int(boundary is not None),
            "boundaries_with_candidates": int(boundary is not None and bool(boundary.chosen_token_ids)),
            "selected_ftpo_tokens": int(selected[row].sum()),
            "repair_tokens": 0 if repair_mask is None else int((repair_mask[row] & (loss_mask[row] > 0)).sum()),
        }
        for name, value in counts.items():
            key = f"{prefix}/{name}"
            metrics[key] = metrics.get(key, 0.0) + value
    return metrics


def closure_training_targets(
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    loss_masks: Sequence[Sequence[int]],
    candidates: torch.Tensor,
    verifications: Sequence[VerificationResult | None],
    *,
    boundary: FTPOBoundary,
    eligible: Sequence[bool],
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    eos_token_ids: frozenset[int],
) -> ClosureTrainingTargets:
    """Use verified closure contracts; infrastructure failures produce no target.

    Repeated-opener targets select JSON-opening alternatives at the first
    repeated marker. Without a repeated marker, retention examples use the same
    content-preserving closure targets as other families. Missing suffixes are
    reserved for masked SFT.
    The continuation mode also selects exact-reference alternatives at verified
    premature closing tokens, without supplying content-repair SFT.
    """
    boundaries = []
    repairs = []
    for row, (prompt, response, mask, verification, allowed) in enumerate(
        zip(prompt_ids, response_ids, loss_masks, verifications, eligible, strict=True)
    ):
        boundaries.append(None)
        repairs.append(None)
        if not allowed or verification is None or verification.status != VerificationStatus.VERIFIED:
            continue
        if boundary == FTPOBoundary.MIXED and "closure" not in verification.diagnostics:
            continue
        contract = ClosurePrompt.model_validate(
            {**verification.diagnostics["closure"], "prompt_token_ids": list(prompt)}
        )
        if len(mask) != len(response) or not all(mask):
            raise ValueError("Closure training requires a single unstitched model response")
        if contract.family == "repeated_opener_retention":
            position = repeated_tool_opener_boundary(prompt, response, mask, decode)
            if position is not None:
                chosen = tuple(token for token in candidates[row, position].tolist() if decode([token]).strip() == "{")
                boundaries[row] = VerifiedFTPOBoundary(position, chosen)
                continue
        content_response = response
        if (
            response
            and (response[-1] in eos_token_ids or decode([response[-1]]) in SNOWBALL_TURN_END_MARKERS)
            and verification.diagnostics.get("stop_reason") == "stop"
        ):
            content_response = response[:-1]
        if boundary in (FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION, FTPOBoundary.MIXED):
            continuation = verified_continuation_boundary(
                prompt,
                content_response,
                candidates[row].tolist(),
                structure_prefix=contract.prefix,
                expected=contract.expected,
                structure_format=contract.format,
                decode=decode,
            )
            if continuation is not None:
                boundaries[row] = VerifiedFTPOBoundary(continuation.response_position, continuation.candidate_token_ids)
                continue
        target = verified_closure_target(
            prompt,
            content_response,
            structure_prefix=contract.prefix,
            inspect=lambda text: closure_state(text, contract.format),
            verify_complete=lambda text: score_closure(text, contract.expected, contract.format).success,
            encode=encode,
            decode=decode,
        )
        if target is None:
            continue
        if target.kind == ClosureTargetKind.MISSING_SUFFIX:
            repairs[row] = target
            continue
        chosen = verified_closure_candidates(
            target,
            prompt,
            response,
            candidates[row, target.response_position].tolist(),
            structure_prefix=contract.prefix,
            verify_complete=lambda value: score_closure(value, contract.expected, contract.format).success,
            decode=decode,
        )
        boundaries[row] = VerifiedFTPOBoundary(target.response_position, chosen)
    return ClosureTrainingTargets(boundaries, repairs)
